# MLX Backend (`"mlx"`) Architecture & Implementation

This document describes the architecture and implementation details of the `"mlx"` backend (`github.com/gomlx/go-darwinml/compute/mlx`), which accelerates GoMLX (`github.com/gomlx/compute`) on Apple Silicon using Apple's [MLX](https://github.com/ml-explore/mlx) framework.

---

## 1. Execution Modes: Compiled Static vs. Lazy Dynamic (`"mlx"` vs. `"mlx:dynamic"`)

### 1.1 MLX's Lazy Evaluation Model (`mlx::core::array`)
Unlike PyTorch (eager) or `MPSGraph` (explicit static `MPSGraph*` object), MLX uses a **lazy array** model:
- Calling an operation like `c = mlx::core::add(a, b)` does **not** execute on the GPU immediately. Instead, it returns an unevaluated `mlx::core::array` whose internal `Primitive` points to parent arrays `a` and `b`.
- Calling `mlx::core::eval(outputs)` traverses the lazy DAG of `mlx::core::array` nodes, schedules Metal kernels on the GPU stream, materializes the output buffers in Apple Silicon Unified Memory, and **detaches the graph history** from the evaluated arrays so intermediate buffers can be freed.
- Because `mlx::core::eval` consumes/detaches the DAG links on the `array` instances, `Builder` / `Function` records operations into an **Op Tape** (a linear list of instructions in topological order).

### 1.2 Dual Execution Modes
Both `compute.Backend` and GoMLX support input-conditioned dynamic shapes (dynamic dimensions with fixed rank, advertised via `Capabilities.DynamicShapes` and `DynamicAxes`).

The `"mlx"` backend supports **both** execution modes over the same recorded Op Tape, controlled by the backend configuration string:

1. **Default Mode (`"mlx"`) — Compiled Static Execution**:
   - **Capabilities**: `DynamicShapes: compute.DynamicShapesNone`.
   - **Compilation (`Builder.Compile()`)**: Prunes unreachable tape nodes (`computeReachable`), wraps the recorded Op Tape in `mlx::core::compile(tape_fn, /*shapeless=*/false)`, and traces it on first execution. MLX optimizes the graph and fuses chains of elementwise/reduction operations into single Metal shaders.
   - **Execution (`Executable.Execute()`)**: Invokes the compiled function on the input `mlx::core::array`s and calls `mlx::core::eval(outputs)`.
   - **Trade-off**: Fastest execution (full kernel fusion); requires static shapes per `Executable` (GoMLX recompiles a new `Executable` if input shapes change).

2. **Dynamic Mode (`"mlx:dynamic"`) — Lazy Tape Replay with Dynamic Shapes**:
   - **Capabilities**: `DynamicShapes: compute.DynamicShapesNative` (and `DynamicAxes: true`).
   - **Compilation (`Builder.Compile()`)**: Prunes unreachable tape nodes and retains the recorded Op Tape without static compilation.
   - **Execution (`Executable.Execute()`)**: Replays the Op Tape lazily on the concrete runtime input `mlx::core::array`s and calls `mlx::core::eval(outputs)`.
     - Because every `mlx::core::array` carries its concrete runtime dimensions during tape replay, operations with `shapes.DynamicDim` (such as `BroadcastInDim`, `Reshape`, `AsStrided` in `ReduceWindow`, `Gather`, `ConvGeneral`, `DotGeneral`, `FusedAttentionQKVProjection`, and `FusedScaledDotProductAttention`) encode negative dynamic dimension references (`-(2 + nodeID*16 + axis)`) and resolve concrete runtime dimensions directly at replay time with zero recompilation overhead. Derived anonymous dynamic axes produced by `ConvGeneral` and `DotGeneral` are assigned unique internal axis names (`#conv_*`, `#dot_*`) so downstream operations can reference their runtime dimensions.
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
- **Native MLX Operations in `./compute/mlx`**:
  - `./compute/mlx` implements its own `Function` layer so it can directly use MLX's native operations (`mlx::core::gather`, `mlx::core::scatter_*`, `mlx::core::conv_general`, `mlx::core::as_strided`, `mlx::core::fast::layer_norm`, `Complex64`, and native dynamic shape resolution) without `MPSGraph`-specific workarounds.

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

### 5.1 Data Types (`dtypes.DType`)
- **Supported**: `Bool`, `Int8`, `Int16`, `Int32`, `Int64`, `Uint8`, `Uint16`, `Uint32`, `Uint64`, `Float16`, `BFloat16`, `Float32`, `Complex64`.
- **Unsupported on Metal GPU (`compute.ErrNotImplemented`)**:
  - `Float64`: Apple Silicon Metal GPUs do not natively support FP64 tensor operations in MLX GPU streams.
  - `Complex128`: Not supported (requires FP64 real/imaginary components).

### 5.2 Unsupported Operations
- **`If` (`OpTypeIf`) & `While` (`OpTypeWhile`)**: Not supported; return `compute.ErrNotImplemented` (see Section 4).
- **`Sort` (`OpTypeSort`) with non-standard comparator**: Only standard ascending (`<`, `<=`) and descending (`>`, `>=`) comparators on a key tensor are supported via native GPU `mlx::core::sort` / `mlx::core::argsort`. Custom comparators return `compute.ErrNotImplemented` with a message advising refactoring into a standard ascending/descending sort.
- **`SelectAndScatterSum` (`OpTypeSelectAndScatterSum`)**: Not supported (`SelectAndScatterMax` and `SelectAndScatterMin` are supported).
- **FFT & Quantized/Collective Ops (`OpTypeFFT`, `OpTypeFusedQuantizedDense`, `OpTypeQuantizedEmbeddingLookup`, `OpTypeAllReduce`)**: Not supported (`compute.ErrNotImplemented`).

### 5.3 Operation Quirks & Semantics Notes
- **Shape Inference & Auto-Broadcasting**:
  - Because `MLXTape` records operations lazily without executing MLX during graph construction, all shape validation and output shape computation are performed in Go using `github.com/gomlx/compute/shapeinference` (`shapeinference.UnaryOp`, `shapeinference.BinaryOp`, `shapeinference.ComparisonOp`, `shapeinference.Where`, etc.).
  - MLX's C++ binary, comparison, and ternary operations (`add`, `subtract`, `multiply`, `divide`, `equal`, `less`, `where`, `clip`, etc.) natively implement NumPy-style auto-broadcasting via zero-copy stride-0 views. Since NumPy broadcasting is a strict superset of GoMLX's implicit broadcasting rules (scalars broadcast to the other operand's shape; non-scalar operands have matching rank with size-1 dimensions broadcast to the larger dimension, including `shapes.DynamicDim`), **no intermediate `broadcast_to` operations need to be injected** before binary, comparison, `Where`, or `Clamp` ops.
- **Integer `Div` and `Rem` (`OpTypeDiv`, `OpTypeRem`)**:
  - In MLX, `mlx::core::divide(a, b)` implements Python true division (promoting integer inputs to `float32`), and `mlx::core::floor_divide(a, b)` / `mlx::core::remainder(a, b)` implement floor division toward $-\infty$ (sign matching the divisor `b`).
  - GoMLX/XLA `Div` and `Rem` follow C/Go truncation-toward-zero semantics (`q = trunc(a / b)` and `r = a - q * b`, where the remainder sign matches the dividend `a`).
  - For unsigned integers, `mlx::core::floor_divide` and `mlx::core::remainder` match C/Go semantics directly. For signed integers, the C++ bridge computes exact integer truncation division (`evalIntTruncDiv`) by dividing unsigned magnitudes (`mlx::core::floor_divide` on `astype(abs(a), u_dt)`) and restoring the quotient sign when `(a < 0) != (b < 0)`, preserving full precision across all integer widths (`int8` through `int64`). For floating-point `Rem`, the bridge computes `a - trunc(a / b) * b` via `where(q >= 0, floor(q), ceil(q))`.
- **`ShiftRightLogical` vs. `ShiftRightArithmetic`**:
  - In MLX, `mlx::core::right_shift(a, b)` performs an arithmetic right shift on signed integer types and a logical right shift on unsigned integer types. For `ShiftRightLogical` on signed integers (`Int8`, `Int16`, `Int32`, `Int64`), the bridge casts operands to the corresponding unsigned integer dtype (`uint8`, `uint16`, `uint32`, `uint64`) via `astype`, applies `right_shift`, and casts back to the signed dtype.
- **`BitCount` (`Popcnt`) and `Clz` (Count Leading Zeros)**:
  - Implemented in the C++ tape replay using standard SWAR bitwise operations (`right_shift`, `bitwise_and`, `subtract`, `add`, `where`) on unsigned views of the integer tensor, which `mlx::core::compile` fuses into a single register-only Metal kernel.
- **`Bitcast` (`OpTypeBitcast`)**:
  - Uses `mlx::core::view(a, target_dtype)` followed by `mlx::core::reshape` to match GoMLX's trailing-axis expansion/contraction semantics when bit-widths differ.
- **Reductions (`ReduceSum`, `ReduceProduct`, `ReduceLogicalXor`, `ReduceBitwise*`)**:
  - `mlx::core::sum` and `mlx::core::prod` promote small integer/boolean dtypes (`int8`, `int16`, `uint8`, `uint16`, `bool`) to 32-bit integers (`int32`/`uint32`); the C++ bridge casts back to the operand's original dtype via `astype` to preserve GoMLX reduction semantics.
  - `ReduceLogicalXor` is implemented by summing `astype(a, int32)` over the reduction axes and checking parity (`(sum & 1) != 0`).
  - Because MLX does not expose native axis-wise bitwise reductions in `<mlx/ops.h>`, `ReduceBitwiseAnd`, `ReduceBitwiseOr`, and `ReduceBitwiseXor` are implemented in `evalBitwiseReduce` using an $O(\log N)$ vectorized slice-pair tree reduction along each reduction axis followed by `squeeze`.
- **`Pad` (`OpTypePad`)**:
  - Supports positive edge padding (`mlx::core::pad`), negative edge padding (cropping via `mlx::core::slice`), and interior padding (decomposed into `[N] -> [N, 1]` reshape, edge pad by `Interior`, flatten to `[N*(Interior+1)]`, and slice to `[N + (N-1)*Interior]`).
- **`DotGeneral` (`OpTypeDotGeneral`)**:
  - Normalizes arbitrary batch, cross, and contracting axes into rank-3 `[B, M, K] @ [B, K, N]` (with a fast path for `[..., M, K] @ [K, N]` matmul that preserves leading dynamic dimensions).
  - Supports `Float16`, `BFloat16`, and `Float32` via hardware `mlx::core::matmul`, `Complex64` via 4-real-matmul decomposition (`(ac - bd) + (ad + bc)i`), and integer types via exact broadcast multiply + sum reduction. Also honors `DotGeneralConfig.AccumulatorDType` and `DotGeneralConfig.OutputDType`.
- **Graph Constants (`OpTypeConstant`)**:
  - When `Function.Constant(flat, dims...)` is called during graph building, the constant is immediately materialized once as an `mlx::core::array` owned by the `MLXTape` (zero per-execution allocation or copy overhead) and freed when the `Executable` / `Builder` is finalized.
- **`Gather`, `ScatterSum`, `ScatterMax`, `ScatterMin`, `DynamicSlice`, `DynamicUpdateSlice`**:
  - Maps general XLA `Gather` and `Scatter` index vector axes, offset/window axes, and collapsed/inserted axes directly to `mlx::core::gather`, `mlx::core::scatter`, `mlx::core::scatter_add`, `mlx::core::scatter_max`, and `mlx::core::scatter_min`.
  - `DynamicSlice` and `DynamicUpdateSlice` normalize scalar or 1D vector `startIndices`, clamp them to `[0, dim - sliceSize]` per XLA semantics, and dispatch to `mlx::core::slice` / `mlx::core::slice_update`.
- **`ConvGeneral` (`OpTypeConvGeneral`)**:
  - Supports 1D, 2D, and 3D spatial convolutions, channel grouping (`featureGroupCount`), batch grouping (`batchGroupCount`), kernel dilation, input dilation, and asymmetric spatial padding via `mlx::core::conv_general`.
- **`ReduceWindow` & `SelectAndScatter` via `mlx::core::as_strided`**:
  - Arbitrary N-D sliding window reductions (`ReduceWindow` with `Sum`, `Product`, `Max`, `Min`, `LogicalAnd`, `LogicalOr`, `LogicalXor`, `BitwiseAnd`, `BitwiseOr`, `BitwiseXor`) and `SelectAndScatterMax` / `SelectAndScatterMin` are implemented on the GPU by constructing a zero-copy strided window view `[outDims..., winDims...]` with `mlx::core::as_strided(contiguous(padded), viewShape, viewStrides, 0)` and reducing/arg-reducing over the window axes.
  - **MLX `col_reduce` Quirk**: When `windowDilation > 1` on the innermost axis (`strides.back() != 1`) and `windowStride == 1` (`outDim > windowDilation`), MLX's Metal `col_reduce_small` kernel (`GeneralStridedReduce`) assumes `column < reduction_stride`. Materializing `mlx::core::contiguous` on `AsStrided` views whenever `strides.back() != 1` routes the reduction to `ContiguousReduce` (`row_reduce`) and avoids the stride assumption.
- **`RNGBitGenerator` (`OpTypeRNGBitGenerator`)**:
  - Uses `mlx::core::random::split` and `mlx::core::random::bits` seeded from the `[3]uint64` GoMLX RNG state (`[2]uint32` Threefry key + counter increment). For 64-bit integer types (`uint64`, `int64`), generates `[dims..., 2]` of `uint32` bits (`width = 4`) and views as the 64-bit dtype via `mlx::core::view`.
- **Dead-Node Pruning (`computeReachable`)**:
  - When `Call` inlines sub-functions or fused operations construct auxiliary outputs (such as `statesForVJP` in `FusedScaledDotProductAttention`), any tape nodes not reachable from the function's final `Return` outputs are pruned before compilation/replay so unused subgraphs incur zero runtime cost.



