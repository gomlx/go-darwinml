// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mlx

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
)

// baseCapabilities declares which ops and dtypes the MLX backend supports in default static compiled mode.
var baseCapabilities = compute.Capabilities{
	Functions:       true,
	DynamicAxes:     false,
	DynamicShapes:   compute.DynamicShapesNone,
	DynamicDimDType: dtypes.Int32,

	Operations: map[compute.OpType]bool{
		// Inputs
		compute.OpTypeParameter: true,
		compute.OpTypeConstant:  true,

		// Identity, Barriers, Conversions, Iota
		compute.OpTypeIdentity:            true,
		compute.OpTypeOptimizationBarrier: true,
		compute.OpTypeSchedulingBarrier:   true,
		compute.OpTypeConvertDType:        true,
		compute.OpTypeBitcast:             true,
		compute.OpTypeIota:                true,

		// Unary operations
		compute.OpTypeAbs:        true,
		compute.OpTypeNeg:        true,
		compute.OpTypeSqrt:       true,
		compute.OpTypeRsqrt:      true,
		compute.OpTypeExp:        true,
		compute.OpTypeExpm1:      true,
		compute.OpTypeLog:        true,
		compute.OpTypeLog1p:      true,
		compute.OpTypeSin:        true,
		compute.OpTypeCos:        true,
		compute.OpTypeTanh:       true,
		compute.OpTypeLogistic:   true,
		compute.OpTypeErf:        true,
		compute.OpTypeFloor:      true,
		compute.OpTypeCeil:       true,
		compute.OpTypeRound:      true,
		compute.OpTypeSign:       true,
		compute.OpTypeLogicalNot: true,
		compute.OpTypeBitwiseNot: true,
		compute.OpTypeBitCount:   true,
		compute.OpTypeClz:        true,
		compute.OpTypeIsFinite:   true,
		compute.OpTypeIsNaN:      true,
		compute.OpTypeConj:       true,
		compute.OpTypeReal:       true,
		compute.OpTypeImag:       true,

		// Binary operations
		compute.OpTypeAdd:                  true,
		compute.OpTypeSub:                  true,
		compute.OpTypeMul:                  true,
		compute.OpTypeDiv:                  true,
		compute.OpTypeRem:                  true,
		compute.OpTypePow:                  true,
		compute.OpTypeMax:                  true,
		compute.OpTypeMin:                  true,
		compute.OpTypeAtan2:                true,
		compute.OpTypeLogicalAnd:           true,
		compute.OpTypeLogicalOr:            true,
		compute.OpTypeLogicalXor:           true,
		compute.OpTypeBitwiseAnd:           true,
		compute.OpTypeBitwiseOr:            true,
		compute.OpTypeBitwiseXor:           true,
		compute.OpTypeShiftLeft:            true,
		compute.OpTypeShiftRightArithmetic: true,
		compute.OpTypeShiftRightLogical:    true,
		compute.OpTypeComplex:              true,

		// Comparison operations
		compute.OpTypeEqual:                   true,
		compute.OpTypeNotEqual:                true,
		compute.OpTypeLessThan:                true,
		compute.OpTypeLessOrEqual:             true,
		compute.OpTypeGreaterThan:             true,
		compute.OpTypeGreaterOrEqual:          true,
		compute.OpTypeEqualTotalOrder:         true,
		compute.OpTypeNotEqualTotalOrder:      true,
		compute.OpTypeLessThanTotalOrder:      true,
		compute.OpTypeLessOrEqualTotalOrder:   true,
		compute.OpTypeGreaterThanTotalOrder:   true,
		compute.OpTypeGreaterOrEqualTotalOrder: true,

		// Shape manipulation operations
		compute.OpTypeReshape:        true,
		compute.OpTypeTranspose:      true,
		compute.OpTypeBroadcastInDim: true,
		compute.OpTypeSlice:          true,
		compute.OpTypeConcatenate:    true,
		compute.OpTypeReverse:        true,
		compute.OpTypePad:            true,

		// Selection operations
		compute.OpTypeWhere: true,
		compute.OpTypeClamp: true,

		// Reductions, ArgMinMax, Cumulative, and DotGeneral
		compute.OpTypeReduceSum:        true,
		compute.OpTypeReduceProduct:    true,
		compute.OpTypeReduceMax:        true,
		compute.OpTypeReduceMin:        true,
		compute.OpTypeReduceLogicalAnd: true,
		compute.OpTypeReduceLogicalOr:  true,
		compute.OpTypeReduceLogicalXor: true,
		compute.OpTypeReduceBitwiseAnd: true,
		compute.OpTypeReduceBitwiseOr:  true,
		compute.OpTypeReduceBitwiseXor: true,
		compute.OpTypeArgMinMax:        true,
		compute.OpTypeCumSum:           true,
		compute.OpTypeCumMax:           true,
		compute.OpTypeDotGeneral:       true,

		// Indexing, Gather, Scatter, DynamicSlice
		compute.OpTypeGather:             true,
		compute.OpTypeScatterSum:         true,
		compute.OpTypeScatterMax:         true,
		compute.OpTypeScatterMin:         true,
		compute.OpTypeDynamicSlice:       true,
		compute.OpTypeDynamicUpdateSlice: true,

		// Convolutions, Pooling, BatchNorm, RNG
		compute.OpTypeConvGeneral:          true,
		compute.OpTypeReduceWindow:         true,
		compute.OpTypeSelectAndScatterMax:  true,
		compute.OpTypeSelectAndScatterMin:  true,
		compute.OpTypeBatchNormForInference: true,
		compute.OpTypeBatchNormForTraining:  true,
		compute.OpTypeBatchNormGradient:     true,
		compute.OpTypeRNGBitGenerator:      true,

		// Sub-functions & Sort
		compute.OpTypeCall: true,
		compute.OpTypeSort: true,

		// Fused operations
		compute.OpTypeFusedSoftmax:                   true,
		compute.OpTypeFusedLayerNorm:                 true,
		compute.OpTypeFusedActivation:                true,
		compute.OpTypeFusedDense:                     true,
		compute.OpTypeFusedAttentionQKVProjection:    true,
		compute.OpTypeFusedScaledDotProductAttention: true,
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
