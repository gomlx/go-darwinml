// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mpsgraph

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
)

// backendCapabilities declares which ops and dtypes the MPSGraph backend supports.
// This is progressively expanded as more ops are implemented.
var backendCapabilities = backends.Capabilities{
	Functions: true, // Closures and control flow (While, If, Sort, Call) supported.

	Operations: map[backends.OpType]bool{
		// Inputs
		backends.OpTypeParameter: true,
		backends.OpTypeConstant:  true,

		// Unary math
		backends.OpTypeAbs:        true,
		backends.OpTypeNeg:        true,
		backends.OpTypeSqrt:       true,
		backends.OpTypeRsqrt:      true,
		backends.OpTypeExp:        true,
		backends.OpTypeExpm1:      true,
		backends.OpTypeLog:        true,
		backends.OpTypeLog1p:      true,
		backends.OpTypeSin:        true,
		backends.OpTypeCos:        true,
		backends.OpTypeTanh:       true,
		backends.OpTypeLogistic:   true,
		backends.OpTypeErf:        true,
		backends.OpTypeFloor:      true,
		backends.OpTypeCeil:       true,
		backends.OpTypeRound:      true,
		backends.OpTypeSign:       true,
		backends.OpTypeLogicalNot: true,
		backends.OpTypeBitwiseNot: true,
		backends.OpTypeIsFinite:   true,
		backends.OpTypeIsNaN:      true,
		backends.OpTypeIdentity:   true,

		// Binary math
		backends.OpTypeAdd:        true,
		backends.OpTypeSub:        true,
		backends.OpTypeMul:        true,
		backends.OpTypeDiv:        true,
		backends.OpTypeRem:        true,
		backends.OpTypePow:        true,
		backends.OpTypeMax:        true,
		backends.OpTypeMin:        true,
		backends.OpTypeAtan2:      true,
		backends.OpTypeLogicalAnd: true,
		backends.OpTypeLogicalOr:  true,
		backends.OpTypeLogicalXor: true,
		backends.OpTypeBitwiseAnd: true,
		backends.OpTypeBitwiseOr:  true,
		backends.OpTypeBitwiseXor: true,
		backends.OpTypeShiftLeft:  true,
		backends.OpTypeShiftRightArithmetic: true,
		backends.OpTypeShiftRightLogical:    true,

		// Comparison
		backends.OpTypeEqual:          true,
		backends.OpTypeNotEqual:       true,
		backends.OpTypeGreaterThan:    true,
		backends.OpTypeGreaterOrEqual: true,
		backends.OpTypeLessThan:       true,
		backends.OpTypeLessOrEqual:    true,

		// Shape operations
		backends.OpTypeReshape:        true,
		backends.OpTypeTranspose:      true,
		backends.OpTypeConvertDType:   true,
		backends.OpTypeBroadcastInDim: true,
		backends.OpTypeWhere:          true,
		backends.OpTypeClamp:          true,
		backends.OpTypeSlice:          true,
		backends.OpTypeConcatenate:    true,
		backends.OpTypeReverse:        true,
		backends.OpTypeIota:           true,
		backends.OpTypePad:            true,

		// Matrix operations
		backends.OpTypeDotGeneral: true,

		// Reductions
		backends.OpTypeReduceSum:     true,
		backends.OpTypeReduceMax:     true,
		backends.OpTypeReduceMin:     true,
		backends.OpTypeReduceProduct: true,

		// ArgMin/ArgMax
		backends.OpTypeArgMinMax: true,

		// Batch normalization
		backends.OpTypeBatchNormForInference: true,

		// Phase 2: Gather/Scatter
		backends.OpTypeGather:     true,
		backends.OpTypeScatterSum: true,
		backends.OpTypeScatterMax: true,
		backends.OpTypeScatterMin: true,

		// Phase 2: DynamicSlice / DynamicUpdateSlice
		backends.OpTypeDynamicSlice:       true,
		backends.OpTypeDynamicUpdateSlice: true,

		// Phase 2: RNG
		backends.OpTypeRNGBitGenerator: true,

		// Phase 2: Convolution
		backends.OpTypeConvGeneral: true,

		// Phase 2: Pooling
		backends.OpTypeReduceWindow:        true,
		backends.OpTypeSelectAndScatterMax: true,

		// Phase 2: TotalOrder comparisons
		backends.OpTypeEqualTotalOrder:          true,
		backends.OpTypeNotEqualTotalOrder:       true,
		backends.OpTypeGreaterThanTotalOrder:    true,
		backends.OpTypeGreaterOrEqualTotalOrder: true,
		backends.OpTypeLessThanTotalOrder:       true,
		backends.OpTypeLessOrEqualTotalOrder:    true,

		// Phase 2: Logical reductions
		backends.OpTypeReduceLogicalAnd: true,
		backends.OpTypeReduceLogicalOr:  true,

		// Fused operations
		backends.OpTypeFusedSoftmax:                      true,
		backends.OpTypeFusedLayerNorm:                    true,
		backends.OpTypeFusedGelu:                         true,
		backends.OpTypeFusedDense:                        true,
		backends.OpTypeFusedAttentionQKVProjection:       true,
		backends.OpTypeFusedScaledDotProductAttention:    true,

		// Batch normalization (training)
		backends.OpTypeBatchNormForTraining: true,
		backends.OpTypeBatchNormGradient:    true,

		// Control flow
		backends.OpTypeWhile: true,
		backends.OpTypeIf:    true,
		backends.OpTypeSort:  true,
		backends.OpTypeCall:  true,
	},

	DTypes: map[dtypes.DType]bool{
		dtypes.Float32: true,
		dtypes.Float16: true,
		dtypes.Bool:    true,
		dtypes.Int8:    true,
		dtypes.Int16:   true,
		dtypes.Int32:   true,
		dtypes.Int64:   true,
		dtypes.Uint8:   true,
		dtypes.Uint16:   true,
		dtypes.Uint32:   true,
		dtypes.Uint64:   true,
		dtypes.BFloat16: true,
	},
}
