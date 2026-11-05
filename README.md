# go-darwinml

Apple Silicon hardware acceleration backends for [GoMLX](https://github.com/gomlx/gomlx) / [`github.com/gomlx/compute`](https://github.com/gomlx/compute).

## Backends

| Backend Name | Package | Framework | Hardware |
|--------------|---------|-----------|----------|
| **`mps`** | `github.com/gomlx/go-darwinml/compute/mps` | Metal Performance Shaders Graph (`MPSGraph`) | Apple GPU (Metal) |

*(Future planned backend: **`mlx`** using Apple's MLX framework.)*

## Requirements

- macOS 13.0+ (Ventura or later; macOS 14.0+ Sonoma for `BFloat16` and native `scaledDotProductAttention`)
- Xcode Command Line Tools (`xcode-select --install`) for CGo + Metal/MPSGraph frameworks
- Go 1.27+

## Installation

```bash
go get github.com/gomlx/go-darwinml
```

## Usage

On Darwin (macOS) machines, the `"mps"` backend is **included by default** when importing GoMLX's default backends package (`_ "github.com/gomlx/gomlx/backends/default"`), or it can be explicitly imported with `_ "github.com/gomlx/go-darwinml/compute/mps"`:

```go
import (
	"github.com/gomlx/compute"
	_ "github.com/gomlx/gomlx/backends/default" // Includes "mps" by default on Darwin; or use:
	// _ "github.com/gomlx/go-darwinml/compute/mps"
)

func main() {
	backend, err := compute.NewWithConfig("mps")
	if err != nil {
		panic(err)
	}
	defer backend.Finalize()
}
```

Or select it at runtime via the `GOMLX_BACKEND=mps` environment variable when calling `compute.New()` / `compute.MustNew()`.

See [`docs/MPSGraph.md`](docs/MPSGraph.md) for detailed architecture and buffer management documentation.

## Limitations (`"mps"`)

- **Unsupported DTypes**: Apple Metal / `MPSGraph` does not support 64-bit floats (`Float64`), complex numbers (`Complex64`, `Complex128`), or sub-byte integer types (`Int2`, `Int4`, `Uint2`, `Uint4`).
- **Unsupported Operations**:
  - **Complex & FFT ops**: `Complex`, `Real`, `Imag`, `Conj`, `FFT`.
  - **Bit-level & bitwise reduction ops**: `Bitcast`, `BitCount`, `Clz`, `ReduceBitwiseAnd`, `ReduceBitwiseOr`, `ReduceBitwiseXor`, `ReduceLogicalXor`.
  - **Pooling scatter variants**: `SelectAndScatterSum`, `SelectAndScatterMin` (`SelectAndScatterMax` is supported).
  - **Dynamic shape ops**: `DynamicBroadcastInDim`, `DynamicIota`, `DynamicPad`, `DynamicReshape`, `DynamicShape`, `DynamicDimensionSize`.
  - **Collective / distributed ops**: `AllReduce`, `AllGather`, `CollectiveBroadcast`.
  - **Barriers & quantized fused ops**: `OptimizationBarrier`, `SchedulingBarrier`, `FusedQuantizedDense`, `QuantizedEmbeddingLookup`.
  - *(Note: Operations with GoMLX graph decompositions, such as `CumSum` and `CumMax`, automatically fall back to their decomposed implementations).*

## Project Structure

```
go-darwinml/
├── compute/
│   └── mps/                # GoMLX compute.Backend implementation ("mps")
└── internal/
    └── bridge/             # CGo / Objective-C bridge to Metal & MPSGraph
```

## Development

```bash
# Run unit and backend compliance tests
go test ./...
```

## License

Apache 2.0 - see LICENSE file.
