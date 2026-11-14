// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mps

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
)

// backendCapabilities declares which ops and dtypes the MPS backend supports.
var backendCapabilities = compute.Capabilities{
	Functions: true, // Closures and control flow (While, If, Sort, Call) supported.

	Operations: map[compute.OpType]bool{
		// Inputs
		compute.OpTypeParameter: true,
		compute.OpTypeConstant:  true,

		// Unary math
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
		compute.OpTypeIsFinite:   true,
		compute.OpTypeIsNaN:      true,
		compute.OpTypeIdentity:   true,

		// Binary math
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

		// Comparison
		compute.OpTypeEqual:          true,
		compute.OpTypeNotEqual:       true,
		compute.OpTypeGreaterThan:    true,
		compute.OpTypeGreaterOrEqual: true,
		compute.OpTypeLessThan:       true,
		compute.OpTypeLessOrEqual:    true,

		// Shape operations
		compute.OpTypeReshape:        true,
		compute.OpTypeTranspose:      true,
		compute.OpTypeConvertDType:   true,
		compute.OpTypeBroadcastInDim: true,
		compute.OpTypeWhere:          true,
		compute.OpTypeClamp:          true,
		compute.OpTypeSlice:          true,
		compute.OpTypeConcatenate:    true,
		compute.OpTypeReverse:        true,
		compute.OpTypeIota:           true,
		compute.OpTypePad:            true,

		// Matrix operations
		compute.OpTypeDotGeneral: true,

		// Reductions
		compute.OpTypeReduceSum:     true,
		compute.OpTypeReduceMax:     true,
		compute.OpTypeReduceMin:     true,
		compute.OpTypeReduceProduct: true,

		// ArgMin/ArgMax & Cumulative
		compute.OpTypeArgMinMax: true,
		compute.OpTypeCumSum:    true,
		compute.OpTypeCumMax:    true,

		// Batch normalization
		compute.OpTypeBatchNormForInference: true,

		// Gather/Scatter
		compute.OpTypeGather:     true,
		compute.OpTypeScatterSum: true,
		compute.OpTypeScatterMax: true,
		compute.OpTypeScatterMin: true,

		// DynamicSlice / DynamicUpdateSlice
		compute.OpTypeDynamicSlice:       true,
		compute.OpTypeDynamicUpdateSlice: true,

		// RNG
		compute.OpTypeRNGBitGenerator: true,

		// Convolution
		compute.OpTypeConvGeneral: true,

		// Pooling
		compute.OpTypeReduceWindow:        true,
		compute.OpTypeSelectAndScatterMax: true,
		compute.OpTypeSelectAndScatterMin: true,

		// TotalOrder comparisons
		compute.OpTypeEqualTotalOrder:          true,
		compute.OpTypeNotEqualTotalOrder:       true,
		compute.OpTypeGreaterThanTotalOrder:    true,
		compute.OpTypeGreaterOrEqualTotalOrder: true,
		compute.OpTypeLessThanTotalOrder:       true,
		compute.OpTypeLessOrEqualTotalOrder:    true,

		// Logical reductions
		compute.OpTypeReduceLogicalAnd: true,
		compute.OpTypeReduceLogicalOr:  true,

		// Fused operations
		compute.OpTypeFusedSoftmax:                   true,
		compute.OpTypeFusedLayerNorm:                 true,
		compute.OpTypeFusedActivation:                true,
		compute.OpTypeFusedDense:                     true,
		compute.OpTypeFusedAttentionQKVProjection:    true,
		compute.OpTypeFusedScaledDotProductAttention: true,
		compute.OpTypeFusedTopK:                      true,

		// Batch normalization (training)
		compute.OpTypeBatchNormForTraining: true,
		compute.OpTypeBatchNormGradient:    true,

		// Control flow
		compute.OpTypeWhile: true,
		compute.OpTypeIf:    true,
		compute.OpTypeSort:  true,
		compute.OpTypeCall:  true,
	},

	DTypes: map[dtypes.DType]bool{
		dtypes.Float32:  true,
		dtypes.Float16:  true,
		dtypes.Bool:     true,
		dtypes.Int8:     true,
		dtypes.Int16:    true,
		dtypes.Int32:    true,
		dtypes.Int64:    true,
		dtypes.Uint8:    true,
		dtypes.Uint16:   true,
		dtypes.Uint32:   true,
		dtypes.Uint64:   true,
		dtypes.BFloat16: true,
	},
}
