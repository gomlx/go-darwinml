// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

// Package mlxbridge provides CGo bindings to Apple's MLX C++ framework for GPU-accelerated
// tensor computation on Apple Silicon.
package mlxbridge

/*
#cgo darwin CXXFLAGS: -std=c++20 -I/opt/homebrew/include -I/usr/local/include
#cgo darwin LDFLAGS: -Wl,-undefined,dynamic_lookup -Wl,-rpath,/opt/homebrew/lib -Wl,-rpath,/usr/local/lib
#include "bridge.h"
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"
)

// DType mirrors the MLXDType enum in bridge.h.
type DType = int

const (
	DTypeBool      DType = C.MLX_DTYPE_BOOL
	DTypeInt8      DType = C.MLX_DTYPE_INT8
	DTypeInt16     DType = C.MLX_DTYPE_INT16
	DTypeInt32     DType = C.MLX_DTYPE_INT32
	DTypeInt64     DType = C.MLX_DTYPE_INT64
	DTypeFloat16   DType = C.MLX_DTYPE_FLOAT16
	DTypeBF16      DType = C.MLX_DTYPE_BFLOAT16
	DTypeFloat32   DType = C.MLX_DTYPE_FLOAT32
	DTypeFloat64   DType = C.MLX_DTYPE_FLOAT64
	DTypeUint8     DType = C.MLX_DTYPE_UINT8
	DTypeUint16    DType = C.MLX_DTYPE_UINT16
	DTypeUint32    DType = C.MLX_DTYPE_UINT32
	DTypeUint64    DType = C.MLX_DTYPE_UINT64
	DTypeComplex64 DType = C.MLX_DTYPE_COMPLEX64
)

// UnaryOp mirrors the MLXUnaryOp enum in bridge.h.
type UnaryOp = int

const (
	UnaryAbs        UnaryOp = C.MLX_UNARY_ABS
	UnaryNeg        UnaryOp = C.MLX_UNARY_NEG
	UnarySqrt       UnaryOp = C.MLX_UNARY_SQRT
	UnaryRsqrt      UnaryOp = C.MLX_UNARY_RSQRT
	UnaryExp        UnaryOp = C.MLX_UNARY_EXP
	UnaryExpm1      UnaryOp = C.MLX_UNARY_EXPM1
	UnaryLog        UnaryOp = C.MLX_UNARY_LOG
	UnaryLog1p      UnaryOp = C.MLX_UNARY_LOG1P
	UnarySin        UnaryOp = C.MLX_UNARY_SIN
	UnaryCos        UnaryOp = C.MLX_UNARY_COS
	UnaryTanh       UnaryOp = C.MLX_UNARY_TANH
	UnaryLogistic   UnaryOp = C.MLX_UNARY_LOGISTIC
	UnaryErf        UnaryOp = C.MLX_UNARY_ERF
	UnaryFloor      UnaryOp = C.MLX_UNARY_FLOOR
	UnaryCeil       UnaryOp = C.MLX_UNARY_CEIL
	UnaryRound      UnaryOp = C.MLX_UNARY_ROUND
	UnarySign       UnaryOp = C.MLX_UNARY_SIGN
	UnaryLogicalNot UnaryOp = C.MLX_UNARY_LOGICAL_NOT
	UnaryBitwiseNot UnaryOp = C.MLX_UNARY_BITWISE_NOT
	UnaryIsFinite   UnaryOp = C.MLX_UNARY_IS_FINITE
	UnaryIsNaN      UnaryOp = C.MLX_UNARY_IS_NAN
	UnaryConj       UnaryOp = C.MLX_UNARY_CONJ
	UnaryReal       UnaryOp = C.MLX_UNARY_REAL
	UnaryImag       UnaryOp = C.MLX_UNARY_IMAG
	UnaryBitCount   UnaryOp = C.MLX_UNARY_BIT_COUNT
	UnaryClz        UnaryOp = C.MLX_UNARY_CLZ
	UnaryIdentity   UnaryOp = C.MLX_UNARY_IDENTITY
)

// BinaryOp mirrors the MLXBinaryOp enum in bridge.h.
type BinaryOp = int

const (
	BinaryAdd                  BinaryOp = C.MLX_BINARY_ADD
	BinarySub                  BinaryOp = C.MLX_BINARY_SUB
	BinaryMul                  BinaryOp = C.MLX_BINARY_MUL
	BinaryDiv                  BinaryOp = C.MLX_BINARY_DIV
	BinaryRem                  BinaryOp = C.MLX_BINARY_REM
	BinaryPow                  BinaryOp = C.MLX_BINARY_POW
	BinaryMax                  BinaryOp = C.MLX_BINARY_MAX
	BinaryMin                  BinaryOp = C.MLX_BINARY_MIN
	BinaryAtan2                BinaryOp = C.MLX_BINARY_ATAN2
	BinaryLogicalAnd           BinaryOp = C.MLX_BINARY_LOGICAL_AND
	BinaryLogicalOr            BinaryOp = C.MLX_BINARY_LOGICAL_OR
	BinaryLogicalXor           BinaryOp = C.MLX_BINARY_LOGICAL_XOR
	BinaryBitwiseAnd           BinaryOp = C.MLX_BINARY_BITWISE_AND
	BinaryBitwiseOr            BinaryOp = C.MLX_BINARY_BITWISE_OR
	BinaryBitwiseXor           BinaryOp = C.MLX_BINARY_BITWISE_XOR
	BinaryShiftLeft            BinaryOp = C.MLX_BINARY_SHIFT_LEFT
	BinaryShiftRightArithmetic BinaryOp = C.MLX_BINARY_SHIFT_RIGHT_ARITHMETIC
	BinaryShiftRightLogical    BinaryOp = C.MLX_BINARY_SHIFT_RIGHT_LOGICAL
	BinaryComplex              BinaryOp = C.MLX_BINARY_COMPLEX
	BinaryEqual                BinaryOp = C.MLX_BINARY_EQUAL
	BinaryNotEqual             BinaryOp = C.MLX_BINARY_NOT_EQUAL
	BinaryLessThan             BinaryOp = C.MLX_BINARY_LESS_THAN
	BinaryLessOrEqual          BinaryOp = C.MLX_BINARY_LESS_OR_EQUAL
	BinaryGreaterThan          BinaryOp = C.MLX_BINARY_GREATER_THAN
	BinaryGreaterOrEqual       BinaryOp = C.MLX_BINARY_GREATER_OR_EQUAL
)

// ReduceOp mirrors the MLXReduceOp enum in bridge.h.
type ReduceOp = int

const (
	ReduceSum        ReduceOp = C.MLX_REDUCE_SUM
	ReduceProduct    ReduceOp = C.MLX_REDUCE_PRODUCT
	ReduceMax        ReduceOp = C.MLX_REDUCE_MAX
	ReduceMin        ReduceOp = C.MLX_REDUCE_MIN
	ReduceLogicalAnd ReduceOp = C.MLX_REDUCE_LOGICAL_AND
	ReduceLogicalOr  ReduceOp = C.MLX_REDUCE_LOGICAL_OR
	ReduceLogicalXor ReduceOp = C.MLX_REDUCE_LOGICAL_XOR
	ReduceBitwiseAnd ReduceOp = C.MLX_REDUCE_BITWISE_AND
	ReduceBitwiseOr  ReduceOp = C.MLX_REDUCE_BITWISE_OR
	ReduceBitwiseXor ReduceOp = C.MLX_REDUCE_BITWISE_XOR
)

// CumulativeOp mirrors the MLXCumulativeOp enum in bridge.h.
type CumulativeOp = int

const (
	CumulativeSum CumulativeOp = C.MLX_CUMULATIVE_SUM
	CumulativeMax CumulativeOp = C.MLX_CUMULATIVE_MAX
)

// ScatterOp mirrors the MLXScatterOp enum in bridge.h.
type ScatterOp = int

const (
	ScatterSum ScatterOp = C.MLX_SCATTER_SUM
	ScatterMax ScatterOp = C.MLX_SCATTER_MAX
	ScatterMin ScatterOp = C.MLX_SCATTER_MIN
)

// ArrayHandle is an opaque uintptr handle to a C++ mlx::core::array*.
type ArrayHandle uintptr

// NodeID identifies a recorded node in an MLXTape.
type NodeID = int32

// Tape wraps an MLXTapeHandle recording graph operations.
type Tape struct {
	handle  C.MLXTapeHandle
	cleanup runtime.Cleanup
}

// Exec wraps a compiled or replayable MLXExecHandle.
type Exec struct {
	handle  C.MLXExecHandle
	cleanup runtime.Cleanup
}

// extractError converts an MLXError into a Go error and frees any C message.
func extractError(err C.MLXError) error {
	if err.code == 0 {
		return nil
	}
	msg := "unknown error"
	if err.message != nil {
		msg = C.GoString(err.message)
		C.free(unsafe.Pointer(err.message))
	}
	return fmt.Errorf("mlx: %s", msg)
}

// Init initializes the MLX bridge and verifies that MLX and its Metal GPU backend are available.
func Init() error {
	var cErr C.MLXError
	if !C.mlx_bridge_init(&cErr) {
		return extractError(cErr)
	}
	return nil
}

// DeviceName returns the Apple Silicon GPU device name reported by MLX.
func DeviceName() (string, error) {
	var cErr C.MLXError
	cName := C.mlx_device_name(&cErr)
	if err := extractError(cErr); err != nil {
		return "", err
	}
	if cName == nil {
		return "Apple Silicon GPU", nil
	}
	name := C.GoString(cName)
	C.free(unsafe.Pointer(cName))
	return name, nil
}

// Version returns the installed MLX version string.
func Version() (string, error) {
	var cErr C.MLXError
	cVer := C.mlx_version_string(&cErr)
	if err := extractError(cErr); err != nil {
		return "", err
	}
	if cVer == nil {
		return "", nil
	}
	ver := C.GoString(cVer)
	C.free(unsafe.Pointer(cVer))
	return ver, nil
}

// ===========================================================================
// Array / Buffer Management
// ===========================================================================

func toInt32Shape(shape []int) (*C.int32_t, C.int, []C.int32_t) {
	if len(shape) == 0 {
		return nil, 0, nil
	}
	cShape := make([]C.int32_t, len(shape))
	for i, d := range shape {
		cShape[i] = C.int32_t(d)
	}
	return &cShape[0], C.int(len(shape)), cShape
}

// NewArray allocates an uninitialized unified-memory MLX array of the given shape and dtype.
func NewArray(nbytes int64, dtype DType, shape []int) (ArrayHandle, error) {
	var cErr C.MLXError
	shapePtr, rank, cShape := toInt32Shape(shape)
	h := C.mlx_array_create(C.int64_t(nbytes), C.int(dtype), shapePtr, rank, &cErr)
	runtime.KeepAlive(cShape)
	if err := extractError(cErr); err != nil {
		return 0, err
	}
	return ArrayHandle(uintptr(h)), nil
}

// NewArrayFromData creates an MLX array initialized with a copy of data in unified memory.
func NewArrayFromData(data unsafe.Pointer, nbytes int64, dtype DType, shape []int) (ArrayHandle, error) {
	var cErr C.MLXError
	shapePtr, rank, cShape := toInt32Shape(shape)
	h := C.mlx_array_from_data(data, C.int64_t(nbytes), C.int(dtype), shapePtr, rank, &cErr)
	runtime.KeepAlive(cShape)
	if err := extractError(cErr); err != nil {
		return 0, err
	}
	return ArrayHandle(uintptr(h)), nil
}

// ArrayDataPtr returns the shared CPU/GPU unified memory pointer for an evaluated array.
func ArrayDataPtr(arr ArrayHandle) unsafe.Pointer {
	if arr == 0 {
		return nil
	}
	return C.mlx_array_data_ptr(C.MLXArrayHandle(unsafe.Pointer(arr)))
}

// ArrayNBytes returns the byte size of the array.
func ArrayNBytes(arr ArrayHandle) int64 {
	if arr == 0 {
		return 0
	}
	return int64(C.mlx_array_nbytes(C.MLXArrayHandle(unsafe.Pointer(arr))))
}

// ArrayDType returns the bridge DType of the array.
func ArrayDType(arr ArrayHandle) DType {
	if arr == 0 {
		return DTypeFloat32
	}
	return DType(C.mlx_array_dtype(C.MLXArrayHandle(unsafe.Pointer(arr))))
}

// ArrayShape returns the runtime dimensions of the array.
func ArrayShape(arr ArrayHandle) []int {
	if arr == 0 {
		return nil
	}
	h := C.MLXArrayHandle(unsafe.Pointer(arr))
	ndim := int(C.mlx_array_ndim(h))
	if ndim == 0 {
		return nil
	}
	cShape := make([]C.int32_t, ndim)
	C.mlx_array_shape(h, &cShape[0])
	dims := make([]int, ndim)
	for i, d := range cShape {
		dims[i] = int(d)
	}
	return dims
}

// ArrayIsRowContiguous reports whether the array is row-contiguous and its buffer matches its logical element count.
func ArrayIsRowContiguous(arr ArrayHandle) bool {
	if arr == 0 {
		return false
	}
	return bool(C.mlx_array_is_row_contiguous(C.MLXArrayHandle(unsafe.Pointer(arr))))
}

// ArrayEnsureContiguous evaluates and materializes a row-contiguous buffer in-place if the array is a strided/broadcast view.
func ArrayEnsureContiguous(arr ArrayHandle) error {
	if arr == 0 {
		return fmt.Errorf("mlx: nil array handle")
	}
	var cErr C.MLXError
	if !C.mlx_array_ensure_contiguous(C.MLXArrayHandle(unsafe.Pointer(arr)), &cErr) {
		return extractError(cErr)
	}
	return nil
}

// DestroyArray frees the C++ mlx::core::array* handle.
func DestroyArray(arr ArrayHandle) {
	if arr != 0 {
		C.mlx_array_destroy(C.MLXArrayHandle(unsafe.Pointer(arr)))
	}
}

// ===========================================================================
// Tape & Execution
// ===========================================================================

// NewTape creates a new MLXTape for recording graph operations.
func NewTape() (*Tape, error) {
	var cErr C.MLXError
	h := C.mlx_tape_create(&cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	t := &Tape{handle: h}
	t.cleanup = runtime.AddCleanup(t, func(handle C.MLXTapeHandle) {
		if handle != nil {
			C.mlx_tape_destroy(handle)
		}
	}, h)
	return t, nil
}

// Destroy releases the tape.
func (t *Tape) Destroy() {
	if t != nil && t.handle != nil {
		t.cleanup.Stop()
		C.mlx_tape_destroy(t.handle)
		t.handle = nil
	}
}

// AddParameter records an input parameter node in the tape.
func (t *Tape) AddParameter(dtype DType, shape []int) (NodeID, error) {
	var cErr C.MLXError
	shapePtr, rank, cShape := toInt32Shape(shape)
	id := C.mlx_tape_add_parameter(t.handle, C.int(dtype), shapePtr, rank, &cErr)
	runtime.KeepAlive(cShape)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// AddConstant materializes a constant array in the tape.
func (t *Tape) AddConstant(data unsafe.Pointer, nbytes int64, dtype DType, shape []int) (NodeID, error) {
	var cErr C.MLXError
	shapePtr, rank, cShape := toInt32Shape(shape)
	id := C.mlx_tape_add_constant(t.handle, data, C.int64_t(nbytes), C.int(dtype), shapePtr, rank, &cErr)
	runtime.KeepAlive(cShape)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// AddUnary records a unary operation in the tape.
func (t *Tape) AddUnary(op UnaryOp, input NodeID) (NodeID, error) {
	var cErr C.MLXError
	id := C.mlx_tape_add_unary(t.handle, C.int(op), C.int32_t(input), &cErr)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// AddBinary records a binary or comparison operation in the tape.
func (t *Tape) AddBinary(op BinaryOp, lhs, rhs NodeID) (NodeID, error) {
	var cErr C.MLXError
	id := C.mlx_tape_add_binary(t.handle, C.int(op), C.int32_t(lhs), C.int32_t(rhs), &cErr)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// AddConvertDType records a dtype conversion (astype) in the tape.
func (t *Tape) AddConvertDType(input NodeID, targetDType DType) (NodeID, error) {
	var cErr C.MLXError
	id := C.mlx_tape_add_convert_dtype(t.handle, C.int32_t(input), C.int(targetDType), &cErr)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// AddBitcast records a bitcast (view + reshape) in the tape.
func (t *Tape) AddBitcast(input NodeID, targetDType DType, outShape []int) (NodeID, error) {
	var cErr C.MLXError
	shapePtr, rank, cShape := toInt32Shape(outShape)
	id := C.mlx_tape_add_bitcast(t.handle, C.int32_t(input), C.int(targetDType), shapePtr, rank, &cErr)
	runtime.KeepAlive(cShape)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// AddIota records an Iota operation in the tape.
func (t *Tape) AddIota(dtype DType, shape []int, iotaAxis int) (NodeID, error) {
	var cErr C.MLXError
	shapePtr, rank, cShape := toInt32Shape(shape)
	id := C.mlx_tape_add_iota(t.handle, C.int(dtype), shapePtr, rank, C.int(iotaAxis), &cErr)
	runtime.KeepAlive(cShape)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// AddReshape records a Reshape operation in the tape.
func (t *Tape) AddReshape(input NodeID, outShape []int) (NodeID, error) {
	var cErr C.MLXError
	shapePtr, rank, cShape := toInt32Shape(outShape)
	id := C.mlx_tape_add_reshape(t.handle, C.int32_t(input), shapePtr, rank, &cErr)
	runtime.KeepAlive(cShape)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// AddTranspose records a Transpose operation in the tape.
func (t *Tape) AddTranspose(input NodeID, axes []int) (NodeID, error) {
	var cErr C.MLXError
	axesPtr, rank, cAxes := toInt32Shape(axes)
	id := C.mlx_tape_add_transpose(t.handle, C.int32_t(input), axesPtr, rank, &cErr)
	runtime.KeepAlive(cAxes)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// AddBroadcastInDim records a BroadcastInDim operation in the tape.
func (t *Tape) AddBroadcastInDim(input NodeID, outShape []int, broadcastAxes []int) (NodeID, error) {
	var cErr C.MLXError
	shapePtr, outRank, cShape := toInt32Shape(outShape)
	axesPtr, axesLen, cAxes := toInt32Shape(broadcastAxes)
	id := C.mlx_tape_add_broadcast_in_dim(t.handle, C.int32_t(input), shapePtr, outRank, axesPtr, axesLen, &cErr)
	runtime.KeepAlive(cShape)
	runtime.KeepAlive(cAxes)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// AddSlice records a Slice operation in the tape.
func (t *Tape) AddSlice(input NodeID, starts, limits, strides []int) (NodeID, error) {
	var cErr C.MLXError
	startsPtr, rank, cStarts := toInt32Shape(starts)
	limitsPtr, _, cLimits := toInt32Shape(limits)
	stridesPtr, _, cStrides := toInt32Shape(strides)
	id := C.mlx_tape_add_slice(t.handle, C.int32_t(input), startsPtr, limitsPtr, stridesPtr, rank, &cErr)
	runtime.KeepAlive(cStarts)
	runtime.KeepAlive(cLimits)
	runtime.KeepAlive(cStrides)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// AddConcatenate records a Concatenate operation in the tape.
func (t *Tape) AddConcatenate(inputs []NodeID, axis int) (NodeID, error) {
	var cErr C.MLXError
	var inPtr *C.int32_t
	cIns := make([]C.int32_t, len(inputs))
	for i, id := range inputs {
		cIns[i] = C.int32_t(id)
	}
	if len(cIns) > 0 {
		inPtr = &cIns[0]
	}
	id := C.mlx_tape_add_concatenate(t.handle, inPtr, C.int(len(cIns)), C.int(axis), &cErr)
	runtime.KeepAlive(cIns)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// AddReverse records a Reverse operation in the tape.
func (t *Tape) AddReverse(input NodeID, axes []int) (NodeID, error) {
	var cErr C.MLXError
	axesPtr, axesLen, cAxes := toInt32Shape(axes)
	id := C.mlx_tape_add_reverse(t.handle, C.int32_t(input), axesPtr, axesLen, &cErr)
	runtime.KeepAlive(cAxes)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// AddPad records an edge Pad operation in the tape.
func (t *Tape) AddPad(input, padValue NodeID, axes, lowPads, highPads []int) (NodeID, error) {
	var cErr C.MLXError
	axesPtr, axesLen, cAxes := toInt32Shape(axes)
	lowPtr, _, cLow := toInt32Shape(lowPads)
	highPtr, _, cHigh := toInt32Shape(highPads)
	id := C.mlx_tape_add_pad(t.handle, C.int32_t(input), C.int32_t(padValue), axesPtr, lowPtr, highPtr, axesLen, &cErr)
	runtime.KeepAlive(cAxes)
	runtime.KeepAlive(cLow)
	runtime.KeepAlive(cHigh)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// AddWhere records a Where (select) operation in the tape.
func (t *Tape) AddWhere(cond, onTrue, onFalse NodeID) (NodeID, error) {
	var cErr C.MLXError
	id := C.mlx_tape_add_where(t.handle, C.int32_t(cond), C.int32_t(onTrue), C.int32_t(onFalse), &cErr)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// AddClamp records a Clamp (clip) operation in the tape.
func (t *Tape) AddClamp(minVal, input, maxVal NodeID) (NodeID, error) {
	var cErr C.MLXError
	id := C.mlx_tape_add_clamp(t.handle, C.int32_t(minVal), C.int32_t(input), C.int32_t(maxVal), &cErr)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// AddReduce records a reduction operation in the tape.
func (t *Tape) AddReduce(op ReduceOp, input NodeID, axes []int) (NodeID, error) {
	var cErr C.MLXError
	axesPtr, axesLen, cAxes := toInt32Shape(axes)
	id := C.mlx_tape_add_reduce(t.handle, C.int(op), C.int32_t(input), axesPtr, axesLen, &cErr)
	runtime.KeepAlive(cAxes)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// AddArgMinMax records an ArgMin or ArgMax operation in the tape.
func (t *Tape) AddArgMinMax(input NodeID, axis int, isMin bool, outputDType DType) (NodeID, error) {
	var cErr C.MLXError
	id := C.mlx_tape_add_arg_min_max(t.handle, C.int32_t(input), C.int(axis), C.bool(isMin), C.int(outputDType), &cErr)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// AddCumulative records a cumulative sum or max operation in the tape.
func (t *Tape) AddCumulative(op CumulativeOp, input NodeID, axis int, reverse, exclusive bool) (NodeID, error) {
	var cErr C.MLXError
	id := C.mlx_tape_add_cumulative(t.handle, C.int(op), C.int32_t(input), C.int(axis), C.bool(reverse), C.bool(exclusive), &cErr)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// AddMatmul records a matrix multiplication in the tape.
func (t *Tape) AddMatmul(lhs, rhs NodeID) (NodeID, error) {
	var cErr C.MLXError
	id := C.mlx_tape_add_matmul(t.handle, C.int32_t(lhs), C.int32_t(rhs), &cErr)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// AddSubParameter records a sub-function parameter placeholder node in the tape.
func (t *Tape) AddSubParameter(dtype DType, shape []int) (NodeID, error) {
	var cErr C.MLXError
	shapePtr, rank, cShape := toInt32Shape(shape)
	id := C.mlx_tape_add_sub_parameter(t.handle, C.int(dtype), shapePtr, rank, &cErr)
	runtime.KeepAlive(cShape)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// Inline clones the subgraph reachable from outputIDs, substituting argIDs[i] for paramIDs[i].
func (t *Tape) Inline(paramIDs, argIDs, outputIDs []NodeID) ([]NodeID, error) {
	var cErr C.MLXError
	var pPtr, aPtr, oPtr, cPtr *C.int32_t
	cParams := make([]C.int32_t, len(paramIDs))
	cArgs := make([]C.int32_t, len(argIDs))
	for i := range paramIDs {
		cParams[i] = C.int32_t(paramIDs[i])
		cArgs[i] = C.int32_t(argIDs[i])
	}
	if len(cParams) > 0 {
		pPtr = &cParams[0]
		aPtr = &cArgs[0]
	}
	cOuts := make([]C.int32_t, len(outputIDs))
	cCloned := make([]C.int32_t, len(outputIDs))
	for i, id := range outputIDs {
		cOuts[i] = C.int32_t(id)
	}
	if len(cOuts) > 0 {
		oPtr = &cOuts[0]
		cPtr = &cCloned[0]
	}
	ok := C.mlx_tape_inline(t.handle, pPtr, aPtr, C.int(len(cParams)), oPtr, C.int(len(cOuts)), cPtr, &cErr)
	runtime.KeepAlive(cParams)
	runtime.KeepAlive(cArgs)
	runtime.KeepAlive(cOuts)
	if !ok {
		if err := extractError(cErr); err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("mlx: inline failed")
	}
	res := make([]NodeID, len(cCloned))
	for i, id := range cCloned {
		res[i] = NodeID(id)
	}
	return res, nil
}

// AddGather records a Gather operation in the tape.
func (t *Tape) AddGather(operand, startIndices NodeID, indexVectorAxis int, startIndexMap, sliceSizes []int) (NodeID, error) {
	var cErr C.MLXError
	mapPtr, mapLen, cMap := toInt32Shape(startIndexMap)
	slicePtr, opRank, cSlices := toInt32Shape(sliceSizes)
	id := C.mlx_tape_add_gather(t.handle, C.int32_t(operand), C.int32_t(startIndices),
		C.int(indexVectorAxis), mapPtr, mapLen, slicePtr, opRank, &cErr)
	runtime.KeepAlive(cMap)
	runtime.KeepAlive(cSlices)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// AddScatter records a Scatter operation in the tape.
func (t *Tape) AddScatter(op ScatterOp, operand, indices, updates NodeID, indexVectorAxis int, axes []int) (NodeID, error) {
	var cErr C.MLXError
	axesPtr, axesLen, cAxes := toInt32Shape(axes)
	id := C.mlx_tape_add_scatter(t.handle, C.int(op), C.int32_t(operand), C.int32_t(indices), C.int32_t(updates),
		C.int(indexVectorAxis), axesPtr, axesLen, &cErr)
	runtime.KeepAlive(cAxes)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// AddDynamicSlice records a DynamicSlice operation in the tape.
func (t *Tape) AddDynamicSlice(operand, startIndices NodeID, sliceSizes []int) (NodeID, error) {
	var cErr C.MLXError
	slicePtr, rank, cSlices := toInt32Shape(sliceSizes)
	id := C.mlx_tape_add_dynamic_slice(t.handle, C.int32_t(operand), C.int32_t(startIndices), slicePtr, rank, &cErr)
	runtime.KeepAlive(cSlices)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// AddDynamicUpdateSlice records a DynamicUpdateSlice operation in the tape.
func (t *Tape) AddDynamicUpdateSlice(operand, update, startIndices NodeID, rank int) (NodeID, error) {
	var cErr C.MLXError
	id := C.mlx_tape_add_dynamic_update_slice(t.handle, C.int32_t(operand), C.int32_t(update), C.int32_t(startIndices), C.int(rank), &cErr)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// AddConvGeneral records a ConvGeneral operation in the tape.
func (t *Tape) AddConvGeneral(input, weight NodeID, strides, padLo, padHi, kernelDilations, inputDilations []int, groups int) (NodeID, error) {
	var cErr C.MLXError
	sPtr, numSpatial, cStrides := toInt32Shape(strides)
	loPtr, _, cLo := toInt32Shape(padLo)
	hiPtr, _, cHi := toInt32Shape(padHi)
	kdPtr, _, cKD := toInt32Shape(kernelDilations)
	idPtr, _, cID := toInt32Shape(inputDilations)
	id := C.mlx_tape_add_conv_general(t.handle, C.int32_t(input), C.int32_t(weight),
		sPtr, loPtr, hiPtr, kdPtr, idPtr, numSpatial, C.int(groups), &cErr)
	runtime.KeepAlive(cStrides)
	runtime.KeepAlive(cLo)
	runtime.KeepAlive(cHi)
	runtime.KeepAlive(cKD)
	runtime.KeepAlive(cID)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// AddAsStrided records an as_strided view operation in the tape.
func (t *Tape) AddAsStrided(input NodeID, shape []int, strides []int64, offset int64) (NodeID, error) {
	var cErr C.MLXError
	shapePtr, rank, cShape := toInt32Shape(shape)
	var stridesPtr *C.int64_t
	cStrides := make([]C.int64_t, len(strides))
	for i, s := range strides {
		cStrides[i] = C.int64_t(s)
	}
	if len(cStrides) > 0 {
		stridesPtr = &cStrides[0]
	}
	id := C.mlx_tape_add_as_strided(t.handle, C.int32_t(input), shapePtr, stridesPtr, rank, C.int64_t(offset), &cErr)
	runtime.KeepAlive(cShape)
	runtime.KeepAlive(cStrides)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// AddSelectAndScatter records a SelectAndScatterMax or SelectAndScatterMin operation in the tape.
func (t *Tape) AddSelectAndScatter(operand, source NodeID, isMax bool, windowDims, windowStrides, padLo, padHi []int) (NodeID, error) {
	var cErr C.MLXError
	wPtr, rank, cWin := toInt32Shape(windowDims)
	sPtr, _, cStrides := toInt32Shape(windowStrides)
	loPtr, _, cLo := toInt32Shape(padLo)
	hiPtr, _, cHi := toInt32Shape(padHi)
	id := C.mlx_tape_add_select_and_scatter(t.handle, C.int32_t(operand), C.int32_t(source),
		C.bool(isMax), wPtr, sPtr, loPtr, hiPtr, rank, &cErr)
	runtime.KeepAlive(cWin)
	runtime.KeepAlive(cStrides)
	runtime.KeepAlive(cLo)
	runtime.KeepAlive(cHi)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// AddRNGBitGenerator records an RNGBitGenerator operation in the tape, returning (newStateID, valuesID).
func (t *Tape) AddRNGBitGenerator(state NodeID, dtype DType, shape []int) (NodeID, NodeID, error) {
	var cErr C.MLXError
	shapePtr, rank, cShape := toInt32Shape(shape)
	var newStateID, valuesID C.int32_t
	ok := C.mlx_tape_add_rng_bit_generator(t.handle, C.int32_t(state), C.int(dtype),
		shapePtr, rank, &newStateID, &valuesID, &cErr)
	runtime.KeepAlive(cShape)
	if !ok {
		if err := extractError(cErr); err != nil {
			return -1, -1, err
		}
		return -1, -1, fmt.Errorf("mlx: RNGBitGenerator failed")
	}
	return NodeID(newStateID), NodeID(valuesID), nil
}

// AddSort records a Sort operation in the tape.
func (t *Tape) AddSort(input NodeID, axis int) (NodeID, error) {
	var cErr C.MLXError
	id := C.mlx_tape_add_sort(t.handle, C.int32_t(input), C.int(axis), &cErr)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// AddArgsort records an Argsort operation in the tape.
func (t *Tape) AddArgsort(input NodeID, axis int) (NodeID, error) {
	var cErr C.MLXError
	id := C.mlx_tape_add_argsort(t.handle, C.int32_t(input), C.int(axis), &cErr)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// AddTakeAlongAxis records a take_along_axis operation in the tape.
func (t *Tape) AddTakeAlongAxis(input, indices NodeID, axis int) (NodeID, error) {
	var cErr C.MLXError
	id := C.mlx_tape_add_take_along_axis(t.handle, C.int32_t(input), C.int32_t(indices), C.int(axis), &cErr)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// AddSoftmax records a Softmax operation in the tape.
func (t *Tape) AddSoftmax(input NodeID, axis int) (NodeID, error) {
	var cErr C.MLXError
	id := C.mlx_tape_add_softmax(t.handle, C.int32_t(input), C.int(axis), &cErr)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// AddLayerNorm records a fast LayerNorm operation in the tape (pass -1 for gamma or beta if nil).
func (t *Tape) AddLayerNorm(input, gamma, beta NodeID, eps float32) (NodeID, error) {
	var cErr C.MLXError
	id := C.mlx_tape_add_layer_norm(t.handle, C.int32_t(input), C.int32_t(gamma), C.int32_t(beta), C.float(eps), &cErr)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// AddNonZero records a NonZero operation in the tape.
func (t *Tape) AddNonZero(operand NodeID, transpose bool) (NodeID, error) {
	var cErr C.MLXError
	id := C.mlx_tape_add_nonzero(t.handle, C.int32_t(operand), C.bool(transpose), &cErr)
	if err := extractError(cErr); err != nil {
		return -1, err
	}
	return NodeID(id), nil
}

// Compile compiles the tape into an Exec.
func (t *Tape) Compile(outputIDs []NodeID, compileGraph, eagerContiguous bool) (*Exec, error) {
	var cErr C.MLXError
	var outPtr *C.int32_t
	cOuts := make([]C.int32_t, len(outputIDs))
	for i, id := range outputIDs {
		cOuts[i] = C.int32_t(id)
	}
	if len(cOuts) > 0 {
		outPtr = &cOuts[0]
	}
	h := C.mlx_tape_compile(
		t.handle,
		outPtr,
		C.int(len(cOuts)),
		C.bool(compileGraph),
		C.bool(eagerContiguous),
		&cErr,
	)
	runtime.KeepAlive(cOuts)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	e := &Exec{handle: h}
	e.cleanup = runtime.AddCleanup(e, func(handle C.MLXExecHandle) {
		if handle != nil {
			C.mlx_exec_destroy(handle)
		}
	}, h)
	return e, nil
}

// Destroy releases the compiled executable.
func (e *Exec) Destroy() {
	if e != nil && e.handle != nil {
		e.cleanup.Stop()
		C.mlx_exec_destroy(e.handle)
		e.handle = nil
	}
}

// Execute runs the executable on the given input array handles.
// For any input i where donate[i] is true, the caller must have detached the ArrayHandle
// from its Go Buffer before calling Execute, as Execute will delete the C++ handle
// before mlx::core::eval() runs.
func (e *Exec) Execute(inputs []ArrayHandle, donate []bool, numOutputs int) ([]ArrayHandle, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.MLXError
	numInputs := len(inputs)

	cInputs := make([]C.MLXArrayHandle, numInputs)
	cDonate := make([]C.bool, numInputs)
	for i, in := range inputs {
		cInputs[i] = C.MLXArrayHandle(unsafe.Pointer(in))
		if i < len(donate) && donate[i] {
			cDonate[i] = C.bool(true)
		}
	}

	cOutputs := make([]C.MLXArrayHandle, numOutputs)

	var inPtr *C.MLXArrayHandle
	var donatePtr *C.bool
	if numInputs > 0 {
		inPtr = &cInputs[0]
		donatePtr = &cDonate[0]
	}
	var outPtr *C.MLXArrayHandle
	if numOutputs > 0 {
		outPtr = &cOutputs[0]
	}

	ok := C.mlx_exec_execute(
		e.handle,
		inPtr,
		donatePtr,
		C.int(numInputs),
		outPtr,
		C.int(numOutputs),
		&cErr,
	)
	runtime.KeepAlive(cInputs)
	runtime.KeepAlive(cDonate)
	if !ok {
		if err := extractError(cErr); err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("mlx: execution failed")
	}

	outputs := make([]ArrayHandle, numOutputs)
	for i, h := range cOutputs {
		outputs[i] = ArrayHandle(uintptr(h))
	}
	return outputs, nil
}
