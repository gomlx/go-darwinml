# go-darwinml

Go backends for machine learning on Apple Silicon, providing [GoMLX](https://github.com/gomlx/gomlx) integration with multiple Apple hardware accelerators.

## Backends

| Backend | Hardware | Import |
|---------|----------|--------|
| **CoreML** | ANE, GPU, CPU | `github.com/gomlx/go-darwinml/coreml/gomlx` |
| **MPSGraph** | GPU (Metal) | `github.com/gomlx/go-darwinml/mpsgraph/gomlx` |

## Status

**Alpha** - Core functionality is implemented but the API may change.

## Requirements

- macOS 12.0+ (Monterey or later)
- Xcode (full installation for coremlcompiler)
- Go 1.25+

## Installation

```bash
# CoreML backend
go get github.com/gomlx/go-darwinml/coreml/gomlx

# MPSGraph backend
go get github.com/gomlx/go-darwinml/mpsgraph/gomlx
```

## Usage

### GoMLX Backend

Import the backend you want to register it with GoMLX:

```go
import _ "github.com/gomlx/go-darwinml/coreml/gomlx"
// or
import _ "github.com/gomlx/go-darwinml/mpsgraph/gomlx"
```

### Building a CoreML MIL Program Directly

```go
package main

import (
    "fmt"
    "github.com/gomlx/go-darwinml/coreml/model"
    "github.com/gomlx/go-darwinml/coreml/runtime"
)

func main() {
    // Build a simple model: y = relu(x)
    b := model.NewBuilder("main")
    x := b.Input("x", model.Float32, 2, 3)
    y := b.Relu(x)
    b.Output("y", y)

    // Compile and load
    rt := runtime.New()
    exec, err := rt.Compile(b)
    if err != nil {
        panic(err)
    }
    defer exec.Close()

    // Run inference
    input := []float32{-1, 2, -3, 4, -5, 6}
    outputs, err := exec.Run(map[string]interface{}{"x": input})
    if err != nil {
        panic(err)
    }

    result := outputs["y"].([]float32)
    fmt.Println("Output:", result)
    // Output: [0 2 0 4 0 6]
}
```

### Compute Unit Selection (CoreML)

```go
import "github.com/gomlx/go-darwinml/coreml/internal/bridge"

// Use all available compute units (ANE + GPU + CPU)
rt := runtime.New(runtime.WithComputeUnits(bridge.ComputeAll))

// CPU only (for debugging)
rt := runtime.New(runtime.WithComputeUnits(bridge.ComputeCPUOnly))

// CPU + GPU (skip Neural Engine)
rt := runtime.New(runtime.WithComputeUnits(bridge.ComputeCPUAndGPU))

// CPU + Neural Engine (skip GPU)
rt := runtime.New(runtime.WithComputeUnits(bridge.ComputeCPUAndANE))
```

## Project Structure

```
go-darwinml/
├── blob/                       # Shared weight blob storage format
├── proto/coreml/               # CoreML protobuf definitions & generated types
├── coreml/                     # CoreML backend
│   ├── gomlx/                  # GoMLX backend implementation
│   ├── model/                  # MIL program builder & serialization
│   ├── runtime/                # Model compilation & execution
│   └── internal/bridge/        # Objective-C++ bridge to CoreML
└── mpsgraph/                   # MPSGraph backend
    └── gomlx/                  # GoMLX backend implementation
        └── internal/bridge/    # Objective-C++ bridge to MPSGraph
```

## Development

```bash
# Test all modules
go test github.com/gomlx/go-darwinml/...

# Regenerate protobuf code
cd proto/coreml && go generate ./...
```

## License

Apache 2.0 - see LICENSE file.

CoreML protobuf definitions are from [Apple's coremltools](https://github.com/apple/coremltools)
and are licensed under BSD-3-Clause.
