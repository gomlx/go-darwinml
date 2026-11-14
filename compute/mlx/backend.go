// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

// Package mlx implements a GoMLX compute backend using Apple's MLX framework
// for GPU-accelerated tensor computation on Apple Silicon.
package mlx

import (
	"fmt"
	"strings"
	"sync"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/go-darwinml/internal/mlxbridge"
	"github.com/pkg/errors"
)

// BackendName is the name used to register this backend.
const BackendName = "mlx"

// Backend implements the compute.Backend interface using Apple's MLX framework.
type Backend struct {
	deviceName      string
	mlxVersion      string
	dynamicShapes   bool
	eagerContiguous bool
	capabilities    compute.Capabilities

	mu          sync.RWMutex
	isFinalized bool
}

// Verify interface compliance.
var _ compute.Backend = (*Backend)(nil)

// parseConfig parses the backend configuration string.
// Supported options (comma- or colon-separated):
//   - "dynamic": enable lazy tape replay with native dynamic shapes support
//   - "eager_contiguous": eagerly make outputs row-contiguous before eval (default)
//   - "lazy_contiguous": defer contiguity materialization until Buffer.Data/ToFlatData (requires "dynamic")
func parseConfig(config string) (dynamicShapes, eagerContiguous bool, err error) {
	eagerContiguous = true
	trimmed := strings.TrimSpace(config)
	trimmed = strings.TrimPrefix(trimmed, BackendName+":")
	if trimmed == "" || trimmed == BackendName {
		return false, true, nil
	}

	fields := strings.FieldsFunc(trimmed, func(r rune) bool {
		return r == ',' || r == ':' || r == ';'
	})
	var wantLazyContiguous bool
	for _, f := range fields {
		opt := strings.TrimSpace(f)
		if opt == "" {
			continue
		}
		switch opt {
		case "dynamic":
			dynamicShapes = true
		case "eager_contiguous":
			eagerContiguous = true
			wantLazyContiguous = false
		case "lazy_contiguous":
			wantLazyContiguous = true
		default:
			return false, false, errors.Errorf(
				"mlx backend: unknown config option %q (supported: \"dynamic\", \"eager_contiguous\", \"lazy_contiguous\")",
				opt,
			)
		}
	}
	if wantLazyContiguous {
		if !dynamicShapes {
			return false, false, errors.New(
				"mlx backend: config option \"lazy_contiguous\" is only supported in \"dynamic\" mode (use \"mlx:dynamic,lazy_contiguous\")",
			)
		}
		eagerContiguous = false
	}
	return dynamicShapes, eagerContiguous, nil
}

// New creates a new MLX backend.
func New(config string) (compute.Backend, error) {
	dynamicShapes, eagerContiguous, err := parseConfig(config)
	if err != nil {
		return nil, err
	}

	if err := mlxbridge.Init(); err != nil {
		return nil, errors.Wrap(err, "creating MLX backend")
	}

	devName, err := mlxbridge.DeviceName()
	if err != nil {
		return nil, errors.Wrap(err, "querying MLX device name")
	}
	ver, _ := mlxbridge.Version()

	caps := baseCapabilities.Clone()
	if dynamicShapes {
		caps.DynamicAxes = true
		caps.DynamicShapes = compute.DynamicShapesNative
		caps.Operations[compute.OpTypeNonZero] = true
	}

	return &Backend{
		deviceName:      devName,
		mlxVersion:      ver,
		dynamicShapes:   dynamicShapes,
		eagerContiguous: eagerContiguous,
		capabilities:    caps,
	}, nil
}

// Name returns the backend name.
func (b *Backend) Name() string { return BackendName }

// String returns the backend name.
func (b *Backend) String() string { return b.Name() }

// Description returns a human-readable description of the MLX backend.
func (b *Backend) Description() string {
	mode := "compiled"
	if b.dynamicShapes {
		mode = "dynamic"
	}
	if b.mlxVersion != "" {
		return fmt.Sprintf("MLX %s GPU backend (%s, %s)", b.mlxVersion, b.deviceName, mode)
	}
	return fmt.Sprintf("MLX GPU backend (%s, %s)", b.deviceName, mode)
}

// NumDevices returns 1 (single Apple Silicon GPU).
func (b *Backend) NumDevices() int { return 1 }

// DeviceDescription returns the Apple Silicon GPU device description.
func (b *Backend) DeviceDescription(deviceNum compute.DeviceNum) string {
	return b.deviceName
}

// Capabilities returns the set of supported ops, dtypes, and dynamic shape capabilities.
func (b *Backend) Capabilities() compute.Capabilities {
	return b.capabilities
}

// Builder creates a new computation builder.
func (b *Backend) Builder(name string) compute.Builder {
	tape, err := mlxbridge.NewTape()
	if err != nil {
		panic(errors.Wrapf(err, "creating MLX builder %q", name))
	}
	return newBuilder(b, name, tape)
}

// Finalize marks the backend as finalized.
func (b *Backend) Finalize() {
	b.mu.Lock()
	b.isFinalized = true
	b.mu.Unlock()
}

// IsFinalized returns whether the backend has been finalized.
func (b *Backend) IsFinalized() bool {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.isFinalized
}

// --- DataInterface implementation ---

// BufferFromFlatData creates a buffer from a Go flat slice.
func (b *Backend) BufferFromFlatData(deviceNum compute.DeviceNum, flat any, shape shapes.Shape) (compute.Buffer, error) {
	if b.IsFinalized() {
		return nil, errors.New("BufferFromFlatData: backend is already finalized")
	}
	if deviceNum != 0 {
		return nil, errors.Errorf("BufferFromFlatData: backend %q only supports deviceNum 0, got %d", BackendName, deviceNum)
	}
	return b.bufferFromFlat(flat, shape)
}

// HasSharedBuffers returns true since Apple Silicon unified memory allows zero-copy access.
func (b *Backend) HasSharedBuffers() bool {
	return true
}

// NewSharedBuffer creates a new buffer in unified memory with direct CPU slice access.
func (b *Backend) NewSharedBuffer(deviceNum compute.DeviceNum, shape shapes.Shape) (buffer compute.Buffer, flat any, err error) {
	if b.IsFinalized() {
		return nil, nil, errors.New("NewSharedBuffer: backend is already finalized")
	}
	if deviceNum != 0 {
		return nil, nil, errors.Errorf("NewSharedBuffer: backend %q only supports deviceNum 0, got %d", BackendName, deviceNum)
	}
	if err := b.checkDTypeSupported(shape.DType); err != nil {
		return nil, nil, err
	}
	buf, err := b.newBuffer(shape)
	if err != nil {
		return nil, nil, err
	}
	flatSlice, err := buf.Data()
	if err != nil {
		_ = buf.Finalize()
		return nil, nil, err
	}
	return buf, flatSlice, nil
}
