# MLX Backend (`"mlx"`) Design Plan

This document tracks the architecture and implementation plan for the `"mlx"` backend (`github.com/gomlx/go-darwinml/compute/mlx`), which accelerates GoMLX (`github.com/gomlx/compute`) on Apple Silicon using Apple's [MLX](https://github.com/ml-explore/mlx) framework.

---

## 1. Execution Modes: Compiled Static vs. Lazy Dynamic (`"mlx"` vs. `"mlx:dynamic"`)

### 1.1 MLX's Lazy Evaluation Model (`mlx::core::array`)
Unlike PyTorch (eager) or `MPSGraph` (explicit static `MPSGraph*` object), MLX uses a **lazy array** model:
- Calling an operation like `c = mlx::core::add(a, b)` does **not** execute on the GPU immediately. Instead, it returns an unevaluated `mlx::core::array` whose internal `Primitive` points to parent arrays `a` and `b`.
- Calling `mlx::core::eval(outputs)` traverses the lazy DAG of `mlx::core::array` nodes, schedules Metal kernels on the GPU stream, materializes the output buffers in Apple Silicon Unified Memory, and **detaches the graph history** from the evaluated arrays so intermediate buffers can be freed.
- Because `mlx::core::eval` consumes/detaches the DAG links on the `array` instances, `Builder` / `Function` records operations into an **Op Tape** (a linear list of instructions in topological order).

### 1.2 Dual Execution Modes (Agreed Decision)
Both `compute.Backend` and GoMLX support input-conditioned dynamic shapes (dynamic dimensions with fixed rank, advertised via `Capabilities.DynamicShapes` and `DynamicOps`).

We will support **both** execution modes over the same recorded Op Tape, controlled by the backend configuration string:

1. **Default Mode (`"mlx"`) — Compiled Static Execution**:
   - **Capabilities**: `DynamicShapes: compute.DynamicShapesNone`.
   - **Compilation (`Builder.Compile()`)**: Wraps the recorded Op Tape in `mlx::core::compile(tape_fn, /*shapeless=*/false)` and traces it at `Compile()` time using the static parameter shapes (`Parameter(name, shape)`). MLX optimizes the graph and fuses chains of elementwise/reduction operations into single Metal shaders.
   - **Execution (`Executable.Execute()`)**: Invokes the compiled function on the input `mlx::core::array`s and calls `mlx::core::eval(outputs)`.
   - **Trade-off**: Fastest execution (full kernel fusion); requires static shapes per `Executable` (GoMLX recompiles a new `Executable` if input shapes change).

2. **Dynamic Mode (`"mlx:dynamic"`) — Lazy Tape Replay with Dynamic Shapes**:
   - **Capabilities**: `DynamicShapes: compute.DynamicShapesNative` (and `DynamicAxes: true`), supporting `compute.DynamicOps` (`DynamicDimensionSize`, `DynamicReshape`, `DynamicBroadcastInDim`, `DynamicIota`, `DynamicPad`).
   - **Compilation (`Builder.Compile()`)**: Keeps the recorded Op Tape without static compilation.
   - **Execution (`Executable.Execute()`)**: Replays the Op Tape lazily on the concrete runtime input `mlx::core::array`s and calls `mlx::core::eval(outputs)`.
     - Because every `mlx::core::array` carries its concrete runtime dimensions during tape replay, `DynamicOps` (such as `DynamicDimensionSize`, `DynamicReshape`, `DynamicBroadcastInDim`, `DynamicIota`, `DynamicPad`) resolve concrete dimensions directly at replay time with zero recompilation overhead.
   - **Trade-off**: Zero recompilation across variable input dimensions (e.g. variable batch sizes or sequence lengths), at the cost of skipping `mlx::core::compile` elementwise fusion.

---

## 2. Bridge Architecture (`internal/mlxbridge`) & Code Sharing

### 2.1 Binding Mechanism: CGo (`darwin && cgo`) (Agreed Decision)
- Like `./compute/mps` and `./internal/bridge`, `./compute/mlx` and `./internal/mlxbridge` use **CGo exclusively** (`//go:build darwin && cgo`, with lightweight `nocgo_darwin.go` and `stub_other.go` stubs).
- **Why CGo C++ (`bridge.h` + `bridge.cpp` wrapping `mlx/mlx.h`)**:
  - Allows the C++ bridge to hold the recorded Op Tape (`MLXTape`), compile it directly with C++ `mlx::core::compile(std::function<std::vector<array>(const std::vector<array>&)>)`, and execute either the compiled function (`"mlx"`) or tape replay (`"mlx:dynamic"`) in a **single CGo call** per `Executable.Execute()`.
- **Graceful Handling When `mlx` Is Not Installed**:
  - At compile time, `bridge.cpp` checks `#if __has_include(<mlx/mlx.h>)` and links with `-Wl,-undefined,dynamic_lookup` so `go build` succeeds even on machines where MLX headers/libraries are not yet installed.
  - At runtime, `mlx_bridge_init` checks for `libmlx.dylib` via `dlopen(..., RTLD_NOW | RTLD_GLOBAL)` (checking `MLX_LIBRARY_PATH`, `"libmlx.dylib"`, `"/opt/homebrew/lib/libmlx.dylib"`, and `"/usr/local/lib/libmlx.dylib"`). If MLX is not installed, `mlx.New()` returns a clear error instructing the user to run `brew install mlx` (with a link to https://ml-explore.github.io/mlx/build/html/install.html).
- **Note on `purego` / FFI without CGo**:
  - A `purego` / dynamic-FFI approach is **deferred** until native FFI support is incorporated into the standard Go distribution (see proposal [golang/go#81450](https://github.com/golang/go/issues/81450)).

## 2.2 Code Sharing with `./compute/mps` (Agreed Decision)
- **Separate C/C++ Bridges (`internal/bridge` vs. `internal/mlxbridge`)**:
  - `internal/bridge` links only system frameworks (`Metal`, `MPSGraph`). Keeping `internal/mlxbridge` separate ensures users of `compute/mps` do not need `libmlx` installed.
- **Shared Slice Helpers in `github.com/gomlx/compute/dtypes`**:
  - Zero-copy unified-memory slice & flat data helpers (`UnsafeAnySliceFromBytes`, `MakeAnySlice`, `FromAnySlice`, `CheckAnySlice`, `SafeCopyAnySlice`) are shared directly via `github.com/gomlx/compute/dtypes`.
- **Shared Pure-Go Package (`./internal/common`)**:
  - `DotGeneral` axis classification & rank-3 normalization helpers (`DotGeneralOutputShape`, `IsSimpleMatMul`, `ClassifyAxes`, `IsIdentityPerm`).
  - Layout permutation helpers for `ConvGeneral` and padding helpers for `ReduceWindow`.
  - XLA-compliant Philox 4x32-10 `RNGBitGenerator` graph builder parameterized over a minimal 8-op interface so both `"mps"` and `"mlx"` share the exact same implementation.
- **Native MLX Operations in `./compute/mlx`**:
  - `./compute/mlx` implements its own `Function` layer so it can directly use MLX's native operations (`mlx::core::gather`, `mlx::core::scatter`, `mlx::core::conv_general`, `mlx::fast::scaled_dot_product_attention`, `mlx::fast::layer_norm`, `mlx::fast::rms_norm`, `Complex64`, and `compute.DynamicOps`) without `MPSGraph`-specific workarounds.

---

## 3. Buffer Management (`mlx::core::array`)

Every `*mlx.Buffer` wraps a heap-allocated `mlx::core::array*` handle backed by Apple Silicon Unified Memory (`MTLResourceStorageModeShared`).

### 3.1 Buffer Donation (`donate[i] == true`) — True Single-Copy Variable Updates
Unlike `MPSGraph` (which required a manual ping-pong buffer pool), **MLX natively supports in-place buffer donation and buffer recycling** when an input `mlx::core::array`'s reference count drops to `1` during evaluation:
- Inside MLX (`Primitive::accept_donate` / `Compiled::eval_gpu`), when an operation (or a compiled fused kernel such as an optimizer update `w_new = w_old - lr * grad`) is about to allocate its output buffer, MLX checks if any of its input `array`s has:
  1. `array_desc_.use_count() == 1` (no external references outside the graph, and no subsequent graph nodes depend on it),
  2. `data_shared_ptr().use_count() == 1` (not aliased by another view), and
  3. matching byte size (`data_size() == out.size() && itemsize() == out.itemsize()`).
- If so, MLX calls `out.move_shared_buffer(in)` — **reusing `w_old`'s physical `MTLBuffer` in-place for `w_new` with zero allocation and only 1 copy of the variable in memory**.
- Even if `w_old` is consumed earlier in the graph than `w_new` is produced, as soon as the last consumer of `w_old` finishes, `w_old`'s `use_count() == 1` causes MLX to immediately return its `MTLBuffer` to MLX's internal `MetalAllocator` cache, where `w_new` reclaims it without allocating new Metal memory.

**Critical Implementation Requirement in `Execute(inputs, donate, defaultDevice)`**:
To enable MLX's native in-place donation during `mlx::core::eval(outputs)`, the external reference held by the Go `*mlx.Buffer` **must be released before `mlx::core::eval(outputs)` runs**:
1. Construct the lazy output `std::vector<mlx::core::array> outputs` by invoking the compiled function (or replaying the tape) on `inputs`. (At this point, the lazy graph nodes hold `shared_ptr<ArrayDesc>` references to the inputs, and no GPU kernels have run yet.)
2. **Before calling `mlx::core::eval(outputs)`**: for every input `i` where `donate[i] == true`, detach and destroy the external `mlx::core::array` handle (and clear `inputs[i]` in the C++ input vector, marking the Go `*mlx.Buffer` finalized).
3. Now call `mlx::core::eval(outputs)`. Because the lazy graph holds the *only* remaining reference (`use_count() == 1`) to each donated input, MLX's evaluator and compiled kernels will automatically reuse the donated buffers in-place!

### 3.2 Zero-Copy Shared Buffers (`HasSharedBuffers() == true`)
GoMLX users and higher-level tensor packages can read/write buffer memory directly on the CPU before or after graph execution via `Backend.NewSharedBuffer`, `Buffer.Data()`, and `Backend.BufferFromFlatData`:

1. **Direct Allocation (`NewSharedBuffer` / `BufferFromFlatData`)**:
   - In C++, we allocate uninitialized unified memory directly from MLX's allocator via:
     ```cpp
     mlx::core::allocator::Buffer buf = mlx::core::allocator::malloc(nbytes);
     auto* arr = new mlx::core::array(buf, mlx_shape, mlx_dtype);
     ```
   - This immediately allocates a page-aligned `MTLBuffer` (`MTLResourceStorageModeShared`) in MLX's allocator without launching any GPU kernel.
   - `arr->data<void>()` returns the shared CPU/GPU memory pointer, which Go wraps in a typed slice (`[]float32`, `[]int32`, etc.) via `common.MakeTypedSlice(dtype, ptr, size)`.
2. **Contiguity on Graph Outputs (`mlx::core::contiguous`) — Default `eager_contiguous` vs. `lazy_contiguous`**:
   - In MLX, view operations like `transpose`, `broadcast_to`, or strided `slice` produce an `mlx::core::array` with non-standard strides (`arr.flags().row_contiguous == false`) or `data_size() != arr.size()`.
   - If such a view is read directly on the CPU via `Buffer.Data()` or `Buffer.ToFlatData()`, `arr.data<void>()` would not have a flat row-major layout matching the tensor's logical shape.
   - Furthermore, in compiled mode (`"mlx"`), `mlx::core::compile` includes input contiguity/strides in its compilation cache key (since fused Metal kernels emit faster indexing for contiguous inputs). If an output view with non-standard strides is fed back as an input on the next step, `mlx::core::compile` would have to re-trace and compile a second kernel variant for the strided layout.
   - **Agreed Decision**:
     - **Default (`eager_contiguous`)**: Always used in compiled mode (`"mlx"`), and default in `"mlx:dynamic"`. Appends `out = mlx::core::contiguous(out)` to outputs before `mlx::core::eval(outputs)`. Because `mlx::core::contiguous` is a zero-cost no-op when `out.flags().row_contiguous` is already `true` (which is the case for all math/reduction/matmul outputs), this adds zero overhead in the common case while guaranteeing consistent contiguous layout for compiled graphs.
     - **Optional `lazy_contiguous` (available only in `"mlx:dynamic"` mode, e.g. `"mlx:dynamic,lazy_contiguous"`)**: Skips `mlx::core::contiguous` at the end of `Execute()`. If a non-contiguous output is simply passed as an input to the next dynamic execution, no copy occurs. Only when `Buffer.Data()` or `Buffer.ToFlatData()` is called on the CPU does it check `arr->flags().row_contiguous` and lazily materialize `*arr = mlx::core::contiguous(*arr); mlx::core::eval({*arr});` in-place.
     - *(In both modes, constructing the Go `unsafe.Slice` in `buf.flat` for output buffers is deferred until `Buffer.Data()` or `Buffer.ToFlatData()` is called.)*

### 3.3 Deterministic & GC Cleanup (`Finalize` + `runtime.AddCleanup`)
Because `mlx::core::array` wraps C++-allocated resources (and Metal buffers), `*mlx.Buffer` must support both explicit `buf.Finalize()` (or donation during `Execute`) and automatic Go garbage collection via `runtime.AddCleanup`, without any risk of double-free or use-after-free:

- **Atomic Handle Cell (`arrayHandle`)**:
  - Rather than storing a raw C pointer directly on `Buffer` and passing a copy to `runtime.AddCleanup`, each `Buffer` references a tiny heap-allocated `arrayHandle` holding an `atomic.Uintptr`:
    ```go
    type arrayHandle struct {
        ptr atomic.Uintptr // C.MLXArrayHandle
    }

    func (h *arrayHandle) destroy() {
        if raw := h.ptr.Swap(0); raw != 0 {
            mlxbridge.DestroyArray(mlxbridge.ArrayHandle(raw))
        }
    }
    ```
  - When a `Buffer` is created:
    ```go
    h := &arrayHandle{}
    h.ptr.Store(uintptr(cArr))
    buf.handle = h
    buf.cleanup = runtime.AddCleanup(buf, (*arrayHandle).destroy, h)
    ```
  - When `buf.Finalize()` is called explicitly (or when `buf` is donated during `Execute()`):
    ```go
    buf.cleanup.Stop()
    buf.handle.destroy() // atomic Swap(0) guarantees C++ delete runs at most once
    buf.flat = nil
    buf.valid = false
    ```
- **Why this works in all orders**:
  - **Pre-finalized / Donated**: `buf.Finalize()` stops the cleanup and atomically swaps `ptr` to `0`, calling `delete arr` in C++ immediately. Even if the GC had already queued the cleanup function concurrently, `h.destroy()` sees `0` and is a safe no-op.
  - **Not finalized by user**: When `buf` becomes unreachable, the GC invokes `(*arrayHandle).destroy(h)`, which swaps `ptr` to `0` and deletes the C++ `mlx::core::array`.
  - **Idempotent**: Calling `buf.Finalize()` multiple times (or calling `Finalize()` after donation) is completely safe.

---

## 4. Control Flow & Functions (`Call`, `Sort`, `If`, `While`) (Agreed Decision)

Every compiled `Executable` in `./compute/mlx` is represented as a **single flat Op Tape** (avoiding multi-tape splitting and intermediate buffer-donation hazards):

1. **`If` and `While` (`compute.OpTypeIf`, `compute.OpTypeWhile`) — Not Supported for Now**:
   - MLX does not natively support data-dependent `If` or `While` inside a single compiled graph (`mlx::core::compile`), and splitting a graph into multiple sub-tapes introduces complex liveness and buffer-donation hazards.
   - Therefore, `OpTypeIf` and `OpTypeWhile` are omitted from `Capabilities.Operations` and return `compute.ErrNotImplemented`. (Elementwise `Where` is fully supported.)

2. **Sub-Functions and `Call` (`Builder.NewFunction`, `compute.OpTypeCall`) — Supported via Tape Inlining**:
   - Sub-functions record their operations in the same format as `Main`.
   - When `Call(targetFn, inputs...)` is invoked, `targetFn`'s recorded operations are replayed/inlined directly into the caller's tape with parameter nodes remapped to the caller's argument nodes.
   - The resulting graph remains a **single flat Op Tape** that compiles and fuses cleanly with `mlx::core::compile`.

3. **`Sort` (`compute.OpTypeSort`) — Pattern-Matched Native GPU Sort**:
   - When `Sort(comparator, axis, isStable, inputs...)` is called, we inspect the tiny graph recorded inside the `comparator` closure:
     - **Supported Standard Comparisons**: If the comparator closure is a direct ascending (`<`, `<=`, `LessThan`, `LessOrEqual`, `LessThanTotalOrder`, `LessOrEqualTotalOrder`) or descending (`>`, `>=`, `GreaterThan`, `GreaterOrEqual`, `GreaterThanTotalOrder`, `GreaterOrEqualTotalOrder`) comparison between `lhs` and `rhs` of a key tensor:
       - **Single tensor**: Emit native GPU `mlx::core::sort(x, axis)` (followed by `Reverse` along `axis` if descending).
       - **Multiple tensors (key-value sort)**: Emit native GPU `indices = mlx::core::argsort(key, axis)` (reversed along `axis` if descending) and `mlx::core::take_along_axis(t, indices, axis)` for each sorted tensor.
       - This keeps `Sort` inside the **single compiled GPU tape**.
     - **Unsupported Custom Comparators**: If the comparator closure performs a non-standard comparison, return an error wrapping `compute.ErrNotImplemented` explaining that the `"mlx"` backend only supports standard ascending/descending comparisons in `Sort`, and suggesting refactoring the sort into a standard ascending/descending sort.

---

## 5. Operations And DTypes

*(Living section — populated as operations are implemented and tested)*

### 5.1 Data Types (`dtypes.DType`)
- **Supported**: `Bool`, `Int8`, `Int16`, `Int32`, `Int64`, `Uint8`, `Uint16`, `Uint32`, `Uint64`, `Float16`, `BFloat16`, `Float32`, `Complex64`.
- **Unsupported on Metal GPU (`compute.ErrNotImplemented`)**:
  - `Float64`: Apple Silicon Metal GPUs do not natively support FP64 tensor operations in MLX GPU streams.
  - `Complex128`: Not supported (requires FP64 real/imaginary components).

### 5.2 Unsupported Operations
- **`If` (`OpTypeIf`) & `While` (`OpTypeWhile`)**: Not supported; return `compute.ErrNotImplemented` (see Section 4).
- **`Sort` (`OpTypeSort`) with non-standard comparator**: Only standard ascending (`<`, `<=`) and descending (`>`, `>=`) comparators on a key tensor are supported via native GPU `mlx::core::sort` / `mlx::core::argsort`. Custom comparators return `compute.ErrNotImplemented` with a message advising refactoring into a standard ascending/descending sort.

### 5.3 Operation Quirks & Semantics Notes
- **Graph Constants (`OpTypeConstant`)**:
  - When `Function.Constant(flat, dims...)` is called during graph building, the constant is immediately materialized once as an `mlx::core::array` owned by the `MLXTape` (zero per-execution allocation or copy overhead) and freed when the `Executable` / `Builder` is finalized.
- **`ScatterSum`, `ScatterMax`, `ScatterMin`**:
  - MLX provides native `mlx::core::scatter_add`, `mlx::core::scatter_max`, and `mlx::core::scatter_min` taking `(operand, indices, updates, axes)`. Duplicate indices accumulate properly using the corresponding reduction mode.
- **`RNGBitGenerator`**:
  - GoMLX expects deterministic Philox 4x32-10 state updates (`newState, values`). Rather than using MLX's stateful/keyed `mlx::core::random::bits` (which uses Threefry), `"mlx"` will share the pure-graph XLA-compliant Philox 4x32-10 implementation in `./internal/common` with `"mps"`.

---

## 6. Staged Implementation Plan

1. **Step 1 — Shared Helpers & MLX Bridge Foundation**:
   - Ensure `mlx` is installed (`brew install mlx`).
   - Extract shared helpers into `./internal/common` (`slice.go`, `dotgeneral.go`, `rng.go`) and verify `./compute/mps` tests still pass.
   - Implement `./internal/mlxbridge` (`bridge.h`, `bridge.cpp`, `bridge.go`, `bridge_nocgo.go`): device/stream initialization, `mlx::core::array` buffer lifecycle, `MLXTape` recording, `mlx::core::compile` wrapper, and `Execute` (with pre-`eval` buffer donation and `eager_contiguous` / `lazy_contiguous` modes).
2. **Step 2 — `./compute/mlx` Core & Basic Operations**:
   - Implement `Backend`, `Buffer` (with `arrayHandle` atomic cleanup), `Builder`, `Executable`, and `Function` with `Parameter`, `Constant`, `Identity`, `ConvertDType`, `Iota`, unary math/logical/bitwise ops, binary math/logical/bitwise ops, and comparison ops.
3. **Step 3 — Shape, Matrix, Reduction & Selection Operations**:
   - Implement `Reshape`, `Transpose`, `BroadcastInDim`, `Slice`, `Concatenate`, `Reverse`, `Pad`, `Where`, `Clamp`, `DotGeneral`, `Reduce*`, and `ArgMinMax`.
4. **Step 4 — Indexing, Convolutions, Pooling, RNG, Control Flow & Fused Ops**:
   - Implement `Gather`, `Scatter*`, `DynamicSlice`, `DynamicUpdateSlice`, `ConvGeneral`, `ReduceWindow`, `SelectAndScatter*`, `BatchNorm*`, `RNGBitGenerator`, `Call`, pattern-matched `Sort`, and fused ops (`FusedSoftmax`, `FusedLayerNorm`, `FusedActivation`, `FusedDense`, `FusedAttentionQKVProjection`, `FusedScaledDotProductAttention`).
5. **Step 5 — `"mlx:dynamic"` Mode & Full Compliance Testing**:
   - Implement `compute.DynamicOps` (`DynamicDimensionSize`, `DynamicReshape`, `DynamicBroadcastInDim`, `DynamicIota`, `DynamicPad`) and `lazy_contiguous` option.
   - Run `backendtest.RunAll` compliance tests and benchmarks in `./compute/mlx/mlx_test.go`.


