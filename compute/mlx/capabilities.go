// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mlx

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
)

// baseCapabilities declares which ops and dtypes the MLX backend supports in default static compiled mode.
var baseCapabilities = compute.Capabilities{
	Functions:      false,
	DynamicAxes:    false,
	DynamicShapes:  compute.DynamicShapesNone,
	DynamicDimDType: dtypes.Int32,

	Operations: map[compute.OpType]bool{
		compute.OpTypeParameter: true,
		compute.OpTypeConstant:  true,
	},

	DTypes: map[dtypes.DType]bool{
		dtypes.Bool:      true,
		dtypes.Int8:      true,
		dtypes.Int16:     true,
		dtypes.Int32:     true,
		dtypes.Int64:     true,
		dtypes.Uint8:     true,
		dtypes.Uint16:    true,
		dtypes.Uint32:    true,
		dtypes.Uint64:    true,
		dtypes.Float16:   true,
		dtypes.BFloat16:  true,
		dtypes.Float32:   true,
		dtypes.Complex64: true,
	},
}
