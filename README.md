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

Import the `mps` backend to register `"mps"` with `github.com/gomlx/compute/backends`:

```go
import (
	"github.com/gomlx/compute/backends"
	_ "github.com/gomlx/go-darwinml/compute/mps"
)

func main() {
	backend, err := backends.NewWithConfig("mps")
	if err != nil {
		panic(err)
	}
	defer backend.Finalize()
}
```

Or select it via the `GOMLX_BACKEND=mps` environment variable when using GoMLX.

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
