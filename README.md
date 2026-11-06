# go-darwinml

Apple Silicon hardware acceleration backends for [GoMLX](https://github.com/gomlx/gomlx) / [`github.com/gomlx/compute`](https://github.com/gomlx/compute).

## Backends

| Backend Name | Package | Framework | Hardware |
|--------------|---------|-----------|----------|
| **`mps`** | `github.com/gomlx/go-darwinml/compute/mps` | Metal Performance Shaders Graph (`MPSGraph`) | Apple GPU (Metal) |
| **`mlx`** | `github.com/gomlx/go-darwinml/compute/mlx` | Apple [MLX](https://github.com/ml-explore/mlx) (`mlx::core`) | Apple GPU (Metal) |

## Requirements

- macOS 13.3+ (macOS 14.0+ Sonoma recommended for `BFloat16`, native `scaledDotProductAttention`, and MLX)
- Apple Silicon (M1 / M2 / M3 / M4 or later)
- Xcode Command Line Tools (`xcode-select --install`) for CGo + Metal/MPSGraph/MLX
- Go 1.27+
- **For the `"mlx"` backend only**: Apple's MLX library installed on the system:
  ```bash
  brew install mlx
  ```
  *(See the [MLX installation guide](https://ml-explore.github.io/mlx/build/html/install.html) for alternative installation methods. The `"mps"` backend uses macOS system frameworks only and does not require `brew install mlx`.)*

## Installation

```bash
go get github.com/gomlx/go-darwinml
```

## Usage

### `"mps"` Backend (Metal Performance Shaders Graph)

On Darwin (macOS) machines, the `"mps"` backend can be imported with `_ "github.com/gomlx/go-darwinml/compute/mps"` (or via GoMLX's default backends package `_ "github.com/gomlx/gomlx/backends/default"`):

```go
import (
	"github.com/gomlx/compute"
	_ "github.com/gomlx/go-darwinml/compute/mps"
)

func main() {
	backend, err := compute.NewWithConfig("mps")
	if err != nil {
		panic(err)
	}
	defer backend.Finalize()
}
```

### `"mlx"` Backend (Apple MLX)

Import `_ "github.com/gomlx/go-darwinml/compute/mlx"` and select `"mlx"` (or `"mlx:dynamic"` for native dynamic shapes without recompilation):

```go
import (
	"github.com/gomlx/compute"
	_ "github.com/gomlx/go-darwinml/compute/mlx"
)

func main() {
	// Default: static-shape compiled execution via mlx::core::compile ("mlx").
	// Or use "mlx:dynamic" for native dynamic-shape execution without recompilation.
	backend, err := compute.NewWithConfig("mlx")
	if err != nil {
		panic(err)
	}
	defer backend.Finalize()
}
```

#### `"mlx"` Configuration Options
- `"mlx"` *(default)*: Compiles the computation graph with `mlx::core::compile` for kernel fusion (`DynamicShapesNone`).
- `"mlx:dynamic"`: Replays the lazy MLX op tape directly on runtime inputs (`DynamicShapesNative`, `DynamicAxes: true`), supporting dynamic dimensions (`shapes.DynamicDim`) without recompilation.
- `"mlx:dynamic,lazy_contiguous"`: In dynamic mode, defers materializing row-major contiguity on view outputs (`Transpose`, `BroadcastInDim`, `Slice`) until CPU access (`Buffer.Data()` / `Buffer.ToFlatData()`).

You can also select either backend at runtime via the `GOMLX_BACKEND` environment variable (e.g., `GOMLX_BACKEND=mlx`, `GOMLX_BACKEND="mlx:dynamic"`, or `GOMLX_BACKEND=mps`) when calling `compute.New()` / `compute.MustNew()`.

See [`docs/MPSGraph.md`](docs/MPSGraph.md) and [`docs/MLX.md`](docs/MLX.md) for detailed architecture and buffer management documentation.

## Limitations

### `"mlx"` Backend
- **Unsupported DTypes**: Apple Silicon Metal GPUs do not support 64-bit floats (`Float64`), `Complex128`, or sub-byte integer types (`Int2`, `Int4`, `Uint2`, `Uint4`). (`Complex64` is supported.)
- **Unsupported Operations**:
  - **Control flow**: `If` (`OpTypeIf`) and `While` (`OpTypeWhile`) are not supported (`Call` is supported via tape inlining, and `Sort` is supported for standard ascending/descending comparators).
  - **Pooling scatter sum**: `SelectAndScatterSum` (`SelectAndScatterMax` and `SelectAndScatterMin` are supported).
  - **FFT, collective & quantized fused ops**: `FFT`, `AllReduce`, `FusedQuantizedDense`, `QuantizedEmbeddingLookup`.

### `"mps"` Backend
- **Unsupported DTypes**: Apple Metal / `MPSGraph` does not support 64-bit floats (`Float64`), complex numbers (`Complex64`, `Complex128`), or sub-byte integer types (`Int2`, `Int4`, `Uint2`, `Uint4`).
- **Unsupported Operations**:
  - **Complex & FFT ops**: `Complex`, `Real`, `Imag`, `Conj`, `FFT`.
  - **Bit-level & bitwise reduction ops**: `Bitcast`, `BitCount`, `Clz`, `ReduceBitwiseAnd`, `ReduceBitwiseOr`, `ReduceBitwiseXor`, `ReduceLogicalXor`.
  - **Pooling scatter variants**: `SelectAndScatterSum`, `SelectAndScatterMin` (`SelectAndScatterMax` is supported).
  - **Dynamic shape ops**: `DynamicShapesNone`.
  - **Collective & quantized fused ops**: `AllReduce`, `FusedQuantizedDense`, `QuantizedEmbeddingLookup`.
  - **Barriers**: `OptimizationBarrier`, `SchedulingBarrier`.
  - *(Note: Operations with GoMLX graph decompositions, such as `CumSum` and `CumMax`, automatically fall back to their decomposed implementations).*

## Project Structure

```
go-darwinml/
├── compute/
│   ├── mlx/                # GoMLX compute.Backend implementation ("mlx")
│   └── mps/                # GoMLX compute.Backend implementation ("mps")
├── docs/
│   ├── MLX.md              # MLX backend architecture & design notes
│   └── MPSGraph.md         # MPSGraph backend architecture & buffer pooling
└── internal/
    ├── bridge/             # CGo / Objective-C bridge to Metal & MPSGraph
    └── mlxbridge/          # CGo / C++17 bridge to Apple MLX (mlx/mlx.h)
```

## Development

```bash
# Run unit and backend compliance tests
go test ./...
```

## License

Apache 2.0 - see LICENSE file.
