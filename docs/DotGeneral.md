# `DotGeneral` Performance & Implementation (`"mlx"` vs `"mps"` vs `"go"`)

This document describes how `DotGeneral` is implemented in the `"mlx"` backend (`compute/mlx/dotgeneral.go`), how MLX handles transposed and batched matrix multiplications on Apple Silicon GPUs, and how its performance compares against the `"mps"` (`MPSGraph`) and `"go"` (`SimpleGo` CPU) backends on `BenchmarkCompliance/DotGeneral`.

---

## 1. How MLX Handles `DotGeneral` and Transposed Matmul

### 1.1 Zero-Copy Metadata Transpose Fused into GEMM
At the C++ API level, `mlx::core::matmul(a, b)` takes two tensors of shape `[..., M, K]` and `[..., K, N]`. However, unlike eager frameworks where `transpose` copies memory into a new buffer before calling GEMM, **in MLX a transpose prior to `matmul` is free and fused directly into the Metal GEMM kernel**:

1. `mlx::core::transpose` is a **zero-copy metadata operation** (`Transpose` primitive) that swaps `shape` and `strides` on the `mlx::core::array` descriptor without copying buffer memory (`mlx/backend/gpu/primitives.cpp`).
2. Inside `matmul`'s GPU dispatch (`mlx/backend/metal/matmul.cpp`), `check_transpose` inspects the strides of each operand:
   - If the last two dimensions have strides `(K, 1)` (row-major), it sets `transpose = false` (`N`).
   - If the last two dimensions have strides `(1, M)` (from `mlx::core::transpose`), it sets `transpose = true` (`T`) and passes the original contiguous buffer directly to the `gemm` (`steel_gemm` / NAX) Metal kernel with `transpose_b = true` (or `transpose_a = true`)—with **zero memory copy and zero extra kernel launch**.

### 1.2 Fast Paths in `compute/mlx/dotgeneral.go`
To ensure `mlx::core::matmul` sees clean `transpose` stride views rather than `reshape`s (which would force `mlx::core::contiguous` copies when applied to non-row-contiguous transposed views), [`DotGeneral`](../compute/mlx/dotgeneral.go) implements four direct fast paths before falling back to the general transpose-and-reshape path:

1. **Standard MatMul (`[B..., M, K] × [B..., K, N]` or `[B..., M, K] × [K, N]`)**:
   - Dispatches directly to a single `mlx::core::matmul(lhs, rhs)` op.
2. **Transposed-RHS MatMul (`[B..., M, K] × [B..., N, K]` or `[B..., M, K] × [N, K]`)**:
   - Common in `Dense` / linear layers (`Y = X · Wᵀ`), attention `Q · Kᵀ`, and backward passes (`dX = dY · W`).
   - Emits a zero-copy `transpose` swapping the last two axes of `rhs`, followed immediately by `mlx::core::matmul(lhs, rhsᵀ)`.
3. **Transposed-LHS MatMul (`[B..., K, M] × [B..., K, N]`)**:
   - Common in weight gradient passes (`dW = Xᵀ · dY`).
   - Emits a zero-copy `transpose` swapping the last two axes of `lhs`, followed immediately by `mlx::core::matmul(lhsᵀ, rhs)`.
4. **General N-D Single-Cross / Single-Contracting Fast Path (e.g., BSHD Attention `[B, S, H, S] × [B, S, H, D]` with `batch=[0, 2]`)**:
   - Whenever both `lhs` and `rhs` have 1 cross axis and 1 contracting axis alongside $B$ arbitrary batch axes (`rank == B + 2`), `DotGeneral` permutes `lhs` directly to `[batch..., M, K]` and `rhs` directly to `[batch..., K, N]` and calls `mlx::core::matmul` **without flattening/reshaping the batch dimensions**.
   - Because MLX's `steel_gemm` natively supports arbitrary leading batch dimensions and batch strides, skipping the `reshape` before and after `matmul` avoids materializing intermediate contiguous copies of non-contiguous transposed inputs.

---

## 2. `BenchmarkCompliance/DotGeneral` Comparison (`"go"` vs `"mps"` vs `"mlx"`)

Benchmarks were run via the standard `github.com/gomlx/compute/support/backendtest` compliance suite on an **Apple M6**:

```bash
go test -bench="BenchmarkCompliance/.*/DotGeneral" -benchtime=100ms ./compute/mlx ./compute/mps
go test -bench="BenchmarkCompliance/.*/DotGeneral" -benchtime=100ms ../compute/gobackend
```

### 2.1 Large 2D & Batched 3D Matrices (`Large`)
On large matrices, **`"mlx"` reaches 8.4 – 12.2 TFlops/s** (leveraging MLX's Metal/NAX GPU Tensor Core GEMM kernels), running **2.7× – 4.3× faster than `"mps"`** and **4.1× – 6.4× faster than `"go"`**:

| Benchmark Case | Shape | `"go"` (GFlops/s \| latency) | `"mps"` (GFlops/s \| latency) | `"mlx"` (GFlops/s \| latency) | `"mlx"` Speedup |
| :--- | :--- | ---: | ---: | ---: | :--- |
| `NoBatch-Large-1` | `1536×1920 * 1920×1024` | 1,872 GF/s (`3.23 ms`) | 2,287 GF/s (`2.64 ms`) | **9,744 GF/s (`620 µs`)** | **4.3× vs MPS**, **5.2× vs Go** |
| `NoBatch-Large-2` | `1024×1920 * 1920×1536` | 2,036 GF/s (`2.97 ms`) | 3,140 GF/s (`1.92 ms`) | **8,439 GF/s (`716 µs`)** | **2.7× vs MPS**, **4.1× vs Go** |
| `NoBatch-Large-3` | `2048×2048 * 2048×2048` | 1,964 GF/s (`8.75 ms`) | 2,987 GF/s (`5.75 ms`) | **10,888 GF/s (`1.58 ms`)** | **3.6× vs MPS**, **5.5× vs Go** |
| `Batched-Large-1` | `16×1536×1920 * 16×1920×1024` | 1,900 GF/s (`50.9 ms`) | 3,734 GF/s (`25.9 ms`) | **12,191 GF/s (`7.93 ms`)** | **3.3× vs MPS**, **6.4× vs Go** |
| `Batched-Large-2` | `16×1024×1920 * 16×1920×1536` | 1,986 GF/s (`48.7 ms`) | 4,151 GF/s (`23.3 ms`) | **11,062 GF/s (`8.74 ms`)** | **2.7× vs MPS**, **5.6× vs Go** |

---

### 2.2 Medium Transformer & Embedding Shapes (`KA-all-MiniLM-L6-v2` & `BAAI-bge-small-en-v1.5`)
Across transformer FFN projections and multi-head attention shapes:
- **`"mlx"` is consistently the fastest backend** (1.5× – 90× faster than `"mps"`, and 2.1× – 6.7× faster than `"go"` on all medium/large shapes).
- **`"mlx"` has zero penalty for transposed RHS (`MxK * NxK`)**: e.g., on `MiniLM/32x13x1536_x_384x1536`, non-transposed runs in `191.6 µs` (`2,561 GF/s`) and transposed runs in `191.1 µs` (`2,567 GF/s`), whereas `"mps"` drops by 40% (`1,648 GF/s` → `990 GF/s`).
- On non-aligned high-rank transposes (`BGE/Incompatible:10x192x12x192_x_10x192x32x12`), `"mps"` hits a pathological layout conversion (`46.4 ms` / `6.1 GF/s`), whereas `"mlx"` completes in **`513 µs` (`552 GF/s`, 90× faster)**.

| Benchmark Case | `"go"` (GF/s \| latency) | `"mps"` (GF/s \| latency) | `"mlx"` (GF/s \| latency) | Notes |
| :--- | ---: | ---: | ---: | :--- |
| `MiniLM/NonTransposed:32x13x1536_x_1536x384` | 1,109 GF/s (`442 µs`) | 1,648 GF/s (`298 µs`) | **2,561 GF/s (`192 µs`)** | **1.55× vs MPS**, **2.3× vs Go** |
| `MiniLM/Transposed:32x13x1536_x_384x1536` | 996 GF/s (`493 µs`) | 990 GF/s (`496 µs`) | **2,567 GF/s (`191 µs`)** | Zero transpose cost in MLX (`191 µs` vs `192 µs`); MPS drops 40% |
| `MiniLM/Transposed:32x13x384_x_1536x384` | 1,179 GF/s (`416 µs`) | 1,354 GF/s (`362 µs`) | **2,447 GF/s (`201 µs`)** | **1.8× vs MPS**, **2.1× vs Go** |
| `MiniLM/32x12x13x13_x_32x12x13x32` (Attention) | **308 GF/s (`13.5 µs`)** | 12.1 GF/s (`342 µs`) | 29.7 GF/s (`140 µs`) | Small seqlen (`13`): CPU wins due to GPU dispatch latency; MLX 2.4× faster than MPS |
| `BGE/Transposed:10x192x1536_x_384x1536` | 1,236 GF/s (`1.83 ms`) | 2,249 GF/s (`1.01 ms`) | **8,262 GF/s (`274 µs`)** | **3.7× vs MPS**, **6.7× vs Go** |
| `BGE/Transposed:10x192x384_x_1536x384` | 1,372 GF/s (`1.65 ms`) | 2,107 GF/s (`1.07 ms`) | **5,020 GF/s (`451 µs`)** | **2.4× vs MPS**, **3.7× vs Go** |
| `BGE/Transposed:10x192x384_x_384x384` | 1,303 GF/s (`435 µs`) | 1,250 GF/s (`453 µs`) | **2,740 GF/s (`207 µs`)** | **2.2× vs MPS**, **2.1× vs Go** |
| `BGE/16x128x384_x_384x384` | 1,386 GF/s (`436 µs`) | 1,109 GF/s (`545 µs`) | **3,062 GF/s (`197 µs`)** | **2.8× vs MPS**, **2.2× vs Go** |
| `BGE/Incompatible:10x192x12x192_x_10x192x12x32` | 290 GF/s (`977 µs`) | 893 GF/s (`317 µs`) | **901 GF/s (`314 µs`)** | BSHD (`batch=[0,2]`): MLX & MPS tied (~3.1× faster than Go) |
| `BGE/Incompatible:10x192x12x192_x_10x192x32x12` | 308 GF/s (`920 µs`) | 6.1 GF/s (`46.4 ms`) | **552 GF/s (`513 µs`)** | Non-aligned transpose: MPS pathological (`46.4 ms`), **MLX 90× faster than MPS** |
| `BGE/8x256x12x256_x_8x256x12x32` | 386 GF/s (`1.04 ms`) | 55.8 GF/s (`7.21 ms`) | **1,045 GF/s (`386 µs`)** | **18.7× vs MPS**, **2.7× vs Go** |

---

### 2.3 Tiny Shapes (`adult-demo` & `test-cases`)
On tiny matrices (e.g., `128×4 * 4×1` or `32×20 * 20×64`, only 1K–80K FLOPs), single-op execution is dominated by Metal GPU command buffer submission and synchronization overhead (`~112 µs` on `"mlx"`, `~130–307 µs` on `"mps"`), so the CPU `"go"` backend is naturally faster when executing isolated micro-operations:

| Benchmark Case | `"go"` (GF/s \| latency) | `"mps"` (GF/s \| latency) | `"mlx"` (GF/s \| latency) |
| :--- | ---: | ---: | ---: |
| `adult-demo/128x4_x_4x1` | **1.32 GF/s (`0.77 µs`)** | 0.008 GF/s (`130 µs`) | 0.009 GF/s (`113 µs`) |
| `adult-demo/128x16_x_16x128` | **266 GF/s (`1.97 µs`)** | 3.74 GF/s (`140 µs`) | 4.65 GF/s (`113 µs`) |
| `test-cases/32x20_x_20x64` | **84.2 GF/s (`0.97 µs`)** | 0.62 GF/s (`132 µs`) | 0.72 GF/s (`114 µs`) |
| `test-cases/TransposedLhs:20x32_x_20x64` | **39.2 GF/s (`2.09 µs`)** | 0.27 GF/s (`307 µs`) | 0.69 GF/s (`119 µs`) |
| `test-cases/TransposedRhs:32x20_x_64x20` | **58.4 GF/s (`1.40 µs`)** | 0.29 GF/s (`280 µs`) | 0.69 GF/s (`119 µs`) |

*(Note: Even on tiny shapes, `"mlx"` has lower per-evaluation dispatch latency than `"mps"`, especially when inputs are transposed: `119 µs` on `"mlx"` vs `280–307 µs` on `"mps"`.)*
