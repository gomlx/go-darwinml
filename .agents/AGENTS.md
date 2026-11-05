# `go-darwinml` Agent Instructions

## Overview
`github.com/gomlx/go-darwinml` provides Apple Silicon hardware acceleration backends for GoMLX (`github.com/gomlx/compute`).

Currently implemented backend:
- **`mps`** (`./compute/mps`): Apple Metal Performance Shaders Graph (`MPSGraph`) GPU backend.

## Repository Layout
- `./compute/mps/`: Implementation of `compute.Backend`, `compute.Builder`, `compute.Function`, `compute.Executable`, and `compute.Buffer` for the `"mps"` backend.
  - `register_darwin.go` / `nocgo_darwin.go` / `stub_other.go`: Platform build tags (`darwin && cgo`, `darwin && !cgo`, `!darwin`) and registration (`backends.Register("mps", New)`).
  - `backend.go`: `Backend` struct and lifecycle (`New`, `Finalize`, `Capabilities`).
  - `buffer.go`: `gpuBuffer` (zero-copy `MTLBuffer` with `MTLResourceStorageModeShared` exposed as a typed Go slice via `unsafe.Slice`, with `runtime.AddCleanup` for automatic GC release) and supported dtype validation (`checkDTypeSupported`).
  - `builder.go`: `Builder` compilation (`compileSimple` and `compileWithControlFlow`).
  - `executable.go`: `Executable` (with mixed-scope per-`Executable` ping-pong `MTLBuffer` pooling and cross-executable stealing for donated inputs; see [`docs/MPSGraph.md`](../docs/MPSGraph.md)) and `ExecutableWithCF` (supports `While`, `If`, `Sort`, and `Call` by orchestrating compiled `MPSGraphExecutable` subgraphs).
  - `function.go`, `dotgeneral.go`, `gather.go`: Graph construction for standard, control-flow, and fused operations.
  - `capabilities.go`: Declares supported `compute.OpType`s and `dtypes.DType`s.
  - `mps_test.go`: Unit tests and full `backendtest.RunAll` compliance suite (`TestCompliance`).
- `./internal/bridge/`: CGo + Objective-C (`-fobjc-arc`) bridge wrapping Apple's `<Metal/Metal.h>`, `<MetalPerformanceShaders/MetalPerformanceShaders.h>`, and `<MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>`.
  - `bridge.h` / `bridge.m`: C API and Objective-C implementation (`MPSGraph`, `MPSGraphExecutable`, `MPSCommandBuffer`).
  - `bridge.go`: Go wrapper types (`Context`, `Exec`, `Tensor`, `DType`).

## Important Nuances & Conventions
1. **Supported DTypes**:
   - `MPSGraph` does **not** support `Float64`, `Complex64`, or `Complex128` on Apple GPUs. Always validate dtypes via `checkDTypeSupported` and wrap unsupported dtypes or sub-features with `compute.ErrNotImplemented` so GoMLX and `backendtest` can fall back or skip gracefully.
2. **Compilation & Feed/Target Ordering**:
   - `[MPSGraph compileWithDevice:feeds:targetTensors:...]` uses an `NSDictionary` for feeds, whose enumeration order is not guaranteed to match caller parameter order. `bridge.m` records `feedPermutation` and `targetPermutation` from `[executable feedTensors]` / `[executable targetTensors]` and reorders inputs/outputs in `mpsgraph_execute`.
3. **Zero-Sized Tensors**:
   - Empty outputs (e.g., `Slice` with `starts[i] == limits[i]`, where `outShape.Size() == 0`) have `tensor == nil` and bypass `MPSGraph` compilation/execution, returning an allocated empty `gpuBuffer`.
4. **`Scatter` Semantics**:
   - `mpsgraph_scatter_nd` uses `[ctx.graph scatterNDWithDataTensor:updatesTensor:indicesTensor:batchDimensions:0 mode:scatterMode name:nil]` so scatter updates are applied into the initial operand tensor `data` (preserving untouched operand elements).
5. **No Dependency on `github.com/gomlx/gomlx`**:
   - Like `./go-xla` and `./compute-onnx`, `./go-darwinml` depends only on `github.com/gomlx/compute` (not `github.com/gomlx/gomlx`) to avoid circular module dependencies. Use `compute/support/backendtest` and `compute/support/testutil` for testing.
