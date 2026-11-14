# `go-darwinml` Agent Instructions

## Overview
`github.com/gomlx/go-darwinml` provides Apple Silicon hardware acceleration backends for GoMLX (`github.com/gomlx/compute`).

Currently implemented backends:
- **`mps`** (`./compute/mps`): Apple Metal Performance Shaders Graph (`MPSGraph`) GPU backend.
- **`mlx`** (`./compute/mlx`): Apple [MLX](https://github.com/ml-explore/mlx) (`mlx::core`) GPU backend (supports both `"mlx"` compiled static execution and `"mlx:dynamic"` native dynamic-shape execution).

## Related Repositories (including this one)

- **[`github.com/gomlx/gomlx`](https://github.com/gomlx/gomlx)**: High-level ML framework in Go (tensors, layers, automatic differentiation, optimizers, model training loops). GoMLX delegates lower-level graph compilation and execution to a `compute.Backend`.
- **[`github.com/gomlx/compute`](https://github.com/gomlx/compute)**: Foundational execution and backend abstraction interface (`compute.Backend`, `compute.Builder`, `compute.Function`, `compute.Executable`, `compute.Buffer`), shared `dtypes`, `shapes`, `shapeinference`, and backend compliance tests (`compute/support/backendtest`).
  - It also includes the "go" backend: written in pure Go, portable (including WASM), optionally supporting SIMD.
- **`github.com/gomlx/go-xla`**: Depends **only** on `github.com/gomlx/compute` (never on `github.com/gomlx/gomlx`, to avoid circular dependencies).
- **`github.com/gomlx/compute-onnx`**: Implements a ONNX Runtime (ORT) based backend. Supports Dynamic Shapes. Depends **only** on `github.com/gomlx/compute`
- **`github.com/gomlx/go-darwinml`**: Implements the "mps" (Apple's MPSGraph based) and "mlx" (Apple's MLX based) backends. Depends **only** on `github.com/gomlx/compute`.
- **`github.com/gomlx/onnx-gomlx`**: Converts `.onnx` models to GoMLX models. 
- **`github.com/gomlx/go-huggingface`**: Hugging Face models tools: download repo files, iterate over datasets, tokenizer implementations, model implementations (using GoMLX). Currently it supports only a subset of the models ( mainly LLMs for text generation and instruction following) but is expandable.

When navigating the codebase of any of the related projects, read their .agents/AGENTS.md file to get more information about
the structure of the codebase and their specific conventions.

## Documentation & Git Workflow

- Record user-visible or API changes in `docs/CHANGELOG.md` (with `docs/CHANGELOG-pjrt.md` and `docs/CHANGELOG-stablehlo.md` kept as historical archives).
- Keep `.agents/AGENTS.md` (symlinked from `GEMINI.md`) updated whenever architecture, conventions, or non-obvious behavior changes.
- **Never `git commit` or `git push`** without explicit user instruction; leave changes uncommitted in the working tree for review.
- When changing code, if not explicitly asked, always wait for the operator to review the changes before committing them.

## Coding Style & Dependencies

- **Minimal External Dependencies (No Testify)**: Do **not** add external test libraries such as `github.com/stretchr/testify`. Use standard `testing` (`t.Fatalf`, `t.Errorf`, `t.Run`) and `github.com/gomlx/compute/support/testutil`. Note that `cmd/pjrt_installer` has its own `go.mod` specifically to isolate interactive TUI (`charmbracelet/*`) dependencies from the main `go-xla` module.
- **Error Handling**: Wrap errors with stack traces using `github.com/pkg/errors` (`errors.Errorf`, `errors.Wrapf`, `errors.WithMessagef`) and format with `"%+v"` when logging stack traces. In `compute/xla`, wrap unsupported ops or option combinations with `compute.ErrNotImplemented` so GoMLX and `backendtest` can fall back or skip cleanly.
- **Follow Existing Patterns**: Before writing new code, read neighboring files in the same package to match established conventions (arena usage, `runtime.AddCleanup` / `runtime.KeepAlive`, `verifyAndCastValues`, closure caching, etc.).
- **Avoid Large Files (Split Per-Op Implementations > 200 Lines)**: Keep files focused so directory listings stay scannable. Place complex op families (> ~200 lines, e.g., `ops_dotgeneral.go`, `ops_reduce.go`, `ops_scatter.go`, `flash.go`) in dedicated files rather than growing a monolith.
- **Copyright Notes**: Normal source files start with:
  ```go
  // Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0
  ```
  Auto-generated files (`gen_*`) omit the copyright header and instead carry a comment identifying the generator tool.

### Compliance Tests, Unit Tests & Benchmarks

- Generic backend tests and benchmarks belong in `github.com/gomlx/compute/support/backendtest` (`TestCompliance` and `BenchmarkCompliance` in `compute/xla/xla_test.go` invoke `backendtest.RunAll` and `backendtest.RunAllBenchmarks`).

## Repository Layout
- `./compute/mlx/`: Implementation of `compute.Backend`, `compute.Builder`, `compute.Function`, `compute.Executable`, and `compute.Buffer` for the `"mlx"` backend (see [`docs/MLX.md`](../docs/MLX.md)).
  - `register_darwin.go` / `nocgo_darwin.go` / `stub_other.go`: Platform build tags and registration (`compute.Register("mlx", New)`).
  - `backend.go`, `buffer.go`, `builder.go`, `executable.go`: Backend lifecycle, zero-copy unified-memory `mlx::core::array` buffers with `runtime.AddCleanup` and pre-`eval` buffer donation, and compiled (`"mlx"`) vs. dynamic (`"mlx:dynamic"`) execution.
  - `function.go`, `dotgeneral.go`, `gather.go`, `conv_pool_norm.go`, `fused.go`, `capabilities.go`: Tape graph construction for standard, dynamic-shape, sub-function (`Call`), pattern-matched `Sort`, and fused operations.
  - `mlx_test.go`: Unit tests and full `backendtest.RunAll` compliance suite (`TestCompliance`) across `"mlx"`, `"mlx:dynamic"`, and `"mlx:dynamic,lazy_contiguous"`.
- `./internal/mlxbridge/`: CGo + C++17 bridge wrapping Apple's `<mlx/mlx.h>` (`bridge.h`, `bridge.cpp`, `bridge.go`, `bridge_nocgo.go`), with runtime `dlopen` check for `libmlx.dylib`.
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
6. **Generic Backend Tests Belong in `github.com/gomlx/compute/support/backendtest`**:
   - When adding compliance or operation tests that other backends can benefit from, add them directly to `github.com/gomlx/compute/support/backendtest/` rather than keeping backend-specific duplicates. Ensure every test gracefully skips when the capability or dtype is absent (`testutil.SkipIfMissing`, `testutil.SkipIfMissingDType`, `testutil.SkipIfMissingDynamicShapes`) or when an operation returns `compute.ErrNotImplemented` (`compute.IsNotImplemented(err)`).
