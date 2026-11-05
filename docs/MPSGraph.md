# MPSGraph Backend (`"mps"`) Implementation Guide

The `"mps"` backend (`github.com/gomlx/go-darwinml/compute/mps`) implements the [`github.com/gomlx/compute`](https://github.com/gomlx/compute) backend interface on macOS (Apple Silicon) using Apple's **Metal** and **Metal Performance Shaders Graph (`MPSGraph`)** frameworks.

---

## 1. Architecture Overview

The repository is split into two packages:

- **`compute/mps`**: Implements `compute.Backend`, `compute.Builder`, `compute.Function`, `compute.Executable`, and `compute.Buffer`, registering itself under the backend name `"mps"` on `darwin && cgo`.
- **`internal/bridge`**: An Objective-C / CGo wrapper (`-fobjc-arc`) around `<Metal/Metal.h>`, `<MetalPerformanceShaders/MetalPerformanceShaders.h>`, and `<MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>`.
  - Manages the singleton `id<MTLDevice>` and shared `id<MTLCommandQueue>`.
  - Exposes `MPSGraph` construction, compilation to `MPSGraphExecutable`, and execution via `MPSCommandBuffer`.

### Graph Compilation & Execution Details

- **Feed & Target Permutation Tracking**:
  - `[MPSGraph compileWithDevice:feeds:targetTensors:targetOperations:compilationDescriptor:]` takes feeds as an `NSDictionary<MPSGraphTensor*, MPSGraphTensorDataType*>`. Because dictionary iteration order is arbitrary, `[executable feedTensors]` may not match the parameter order declared by the GoMLX caller.
  - During compilation, `internal/bridge/bridge.m` compares the caller's `feeds` and `targets` arrays against `[executable feedTensors]` and `[executable targetTensors]` by pointer identity, storing `feedPermutation` and `targetPermutation` inside `MPSGraphExecCtx` so `mpsgraph_execute` reorders inputs and outputs transparently.
- **Supported DTypes**:
  - Apple Silicon GPUs and `MPSGraph` natively support `Bool`, `Int8`, `Int16`, `Int32`, `Int64`, `Uint8`, `Uint16`, `Uint32`, `Uint64`, `Float16`, `BFloat16`, and `Float32`.
  - `Float64`, `Complex64`, and `Complex128` are **not** supported by Metal / `MPSGraph` and return `compute.ErrNotImplemented`.
- **Control Flow (`While`, `If`, `Sort`, `Call`)**:
  - Functions containing control flow operations compile the pre-control-flow DAG and closure functions into separate `MPSGraphExecutable` instances (`ExecutableWithCF`), orchestrating loop/branch/sort execution from Go while keeping tensor buffers in unified memory.
- **Zero-Sized Tensors**:
  - Operations producing empty tensors (`shape.Size() == 0`, such as `Slice` with `starts[i] == limits[i]`) bypass `MPSGraph` kernel dispatch and return an empty `gpuBuffer`.

---

## 2. Buffer Management

### Unified Memory & Zero-Copy Shared Buffers

Apple Silicon uses a Unified Memory Architecture (UMA) where CPU and GPU share physical DRAM. The `"mps"` backend takes full advantage of this in `compute/mps/buffer.go` and `internal/bridge/bridge.m`:

1. **`MTLBuffer` Allocation (`MTLResourceStorageModeShared`)**:
   - Every non-empty `gpuBuffer` allocates a Metal `id<MTLBuffer>` via `[device newBufferWithLength:nbytes options:MTLResourceStorageModeShared]`.
   - The `MTLBuffer`'s page-aligned unified memory pointer (`[mtlBuf contents]`) is exposed to Go as a typed slice (`[]float32`, `[]int32`, etc.) via `unsafe.Slice`.
2. **Zero-Copy Graph Execution**:
   - Because `Backend.HasSharedBuffers()` returns `true` and `Backend.NewSharedBuffer()` returns the `MTLBuffer`-backed slice directly, GoMLX tensors (`Tensor.MutableFlatData`, constant uploads, and output reads) read and write GPU memory in-place without intermediate `memcpy` staging.
   - During `Executable.Execute`, inputs and outputs are wrapped via `[[MPSGraphTensorData alloc] initWithMTLBuffer:mtlBuf shape:shape dataType:mpsType]`, allowing `MPSGraphExecutable` to read inputs and write results directly to the underlying `MTLBuffer`s.
3. **Deterministic & Garbage-Collected Cleanup (`runtime.AddCleanup`)**:
   - Following the same dual-cleanup design as `go-xla`, every `gpuBuffer`, `bridge.Context`, `bridge.Exec`, and `Executable` supports both explicit `.Finalize()` / `.Destroy()` and automatic Go garbage collection cleanup via Go 1.24+'s `runtime.AddCleanup`.
   - When `Finalize()` is called explicitly, `cleanup.Stop()` is invoked first and the underlying Objective-C / Metal handle is released immediately via `CFBridgingRelease`.

### Limitations of `MPSGraph` Compared to `compute.Backend`

In the `compute.Backend` execution model (`Executable.Execute(inputs, donate, defaultDevice)`), callers can mark input buffers with `donate[i] = true` (for example, model weights and optimizer state variables during a training step `w_new = w_old - lr * grad`).

In backends like XLA/PJRT:
- The compiler knows at compile time (or via PJRT buffer donation) which inputs are donated and performs live-range / alias analysis across the HLO graph.
- XLA schedules kernels so that `w_new` is written directly into `w_old`'s physical buffer *after* all consumers of `w_old` have finished reading it.

**`MPSGraph` does not support safe same-execution input/output buffer aliasing:**
- `[MPSGraph compileWithDevice:...]` has no API to declare that a `feedTensor` is donated or aliases a `targetTensor`.
- While passing the same `MTLBuffer` in both `inputsArray` and `resultsArray` of `[MPSGraphExecutable encodeToCommandBuffer:inputsArray:resultsArray:...]` happens to work for purely elementwise graphs or graphs with an intermediate barrier (such as a `matmul` before a residual add), **`MPSGraph`'s MLIR compiler stitches non-elementwise operations (such as `Transpose`, `Gather`, `Slice`, `Reverse`) into single fused Metal kernels assuming inputs and outputs do not alias**.
- If the same `MTLBuffer` is passed as both input and output to a stitched `Transpose` kernel on a large matrix (e.g. `512×512`), GPU threadgroups overwrite transposed coordinates `(r, c)` before other threadgroups have finished reading `(c, r)`, silently corrupting the output.
- Consequently, the output `MTLBuffer` passed to a single `MPSGraphExecutable` invocation **must be distinct** from all input `MTLBuffer`s read by that invocation.

### Ping-Pong Buffer Pooling for Donated Inputs

Although an input `MTLBuffer` cannot be aliased as an output within the *same* `MPSGraphExecutable` call, a donated input buffer is completely free as soon as `[cmdBuf waitUntilCompleted]` returns at the end of `Execute`.

To avoid allocating and freeing `MTLBuffer`s on every training step, the `"mps"` backend implements a **Ping-Pong `MTLBuffer` Pool**:
- **Step $t$**: Inputs live in buffer set $A$; outputs are written to buffer set $B$. At the end of Step $t$, donated inputs in set $A$ are placed into the pool instead of being destroyed.
- **Step $t+1$**: Inputs now live in buffer set $B$ (the outputs of Step $t$). When preparing outputs for Step $t+1$, `Execute` pops buffer set $A$ from the pool, runs the graph, and returns the donated set $B$ back to the pool.
- From Step 2 onward, training runs with **zero Metal buffer allocations** and **one extra buffer set** in flight during the kernel execution window.

#### Pooling Rules to Prevent Unbounded Memory Growth

A naive pool that caches every freed or donated buffer by size would grow without bound if a user repeatedly creates new input tensors of varying shapes or donates inputs that are never reused as outputs. To prevent leaks and unbounded growth, the pool strictly enforces three rules during `Executable.Execute`:

1. **Condition 1 — Only Donated Inputs Are Pooled**:
   - An input buffer is only eligible for pooling if `donate[i] == true`. Non-donated buffers finalized via `Buffer.Finalize()` or garbage collection are destroyed immediately.
2. **Condition 2 — Must Match an Output Size of the Execution (Capped per Execution)**:
   - A donated input of size `nbytes` is only pooled if the execution produces at least one output of that exact `nbytes`.
   - **Condition 2.a**: If an execution has $M$ donated inputs of size `nbytes` and $N$ outputs of size `nbytes`, at most $\min(M, N)$ donated inputs of size `nbytes` are pooled; any excess donated inputs are destroyed immediately. This guarantees the pool can never accumulate more buffers of a given size than a single execution of the graph outputs.
3. **Condition 3 — Immediate Eviction of Unused Local Buffers**:
   - Immediately after an `Executable` pops matching buffers from its pool to construct its output buffers for the current step, **any buffers still remaining in that `Executable`'s pool were not used by this execution and are destroyed immediately**.

#### Mixed-Scope Pooling: Avoiding Memory Duplication Across Multi-Task / Multi-Shape Graphs

In GoMLX, a single model often compiles multiple `compute.Executable` instances that share the same model variables:
- **Multi-task training**: Alternating between different loss or task graphs that update the same weights.
- **Dynamic shape bucketing**: Separate compiled graphs for different batch sizes or sequence-length buckets.

If each `Executable` kept a strictly isolated pool, alternating between $K$ executables would leave 1 ping-pong copy of all model weights and optimizer states parked in *each* of the $K$ executables' pools—multiplying memory overhead by $K\times$. Conversely, if all executables shared a single flat pool and applied Condition 3 globally, running a smaller graph would evict pooled buffers needed by a larger graph.

To solve both problems, the `"mps"` backend uses a **Mixed-Scope Pool** (`Backend.acquireOutputBuffers` in `compute/mps/backend.go`):

1. **Per-`Executable` Primary Pool**:
   - Each `Executable` owns a `pingPongPool` registered with its parent `Backend` (and automatically unregistered and destroyed when the `Executable` is finalized or garbage-collected).
   - Donated inputs from an execution of `Executable E` (filtered by Conditions 1, 2, and 2.a) are stored in `E.pool`.
2. **Cross-Executable Stealing (Fallback)**:
   - When `Executable B` prepares its output buffers of size `nbytes`:
     1. It first checks its own `B.pool`.
     2. If `B.pool` does not have a buffer of size `nbytes`, `B` **steals** a buffer of size `nbytes` from another active `Executable`'s pool on the same `Backend` (e.g. `A.pool`).
     3. Only if no pool on the `Backend` has a buffer of size `nbytes` does `B` allocate a new `MTLBuffer`.
3. **Local-Only Unused Eviction + Idle TTL Eviction**:
   - After `B` acquires its outputs, Condition 3 evicts unused buffers **only from `B.pool`**, leaving task-specific buffers in `A.pool` intact for the next time `A` runs.
   - To prevent buffers in a rarely executed shape bucket from sitting idle indefinitely if their sizes are not shared with other executables, the `Backend` tracks an execution step counter (`execStep`). Any `Executable` pool that has not been used for more than `maxIdlePoolSteps` (32 executions) has its remaining un-stolen buffers automatically evicted.
