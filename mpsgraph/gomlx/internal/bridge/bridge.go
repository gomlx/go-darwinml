// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

// Package bridge provides CGo bindings to Apple's MPSGraph framework for GPU-accelerated
// tensor computation on Apple Silicon.
package bridge

/*
#cgo darwin CFLAGS: -fobjc-arc
#cgo darwin LDFLAGS: -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph
#include "bridge.h"
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"
)

// DType mirrors the MPSGraphDType enum in bridge.h.
type DType = int

const (
	DTypeBool    DType = C.MPSGRAPH_DTYPE_BOOL
	DTypeInt8    DType = C.MPSGRAPH_DTYPE_INT8
	DTypeInt16   DType = C.MPSGRAPH_DTYPE_INT16
	DTypeInt32   DType = C.MPSGRAPH_DTYPE_INT32
	DTypeInt64   DType = C.MPSGRAPH_DTYPE_INT64
	DTypeFloat16 DType = C.MPSGRAPH_DTYPE_FLOAT16
	DTypeBF16    DType = C.MPSGRAPH_DTYPE_BFLOAT16
	DTypeFloat32 DType = C.MPSGRAPH_DTYPE_FLOAT32
	DTypeFloat64 DType = C.MPSGRAPH_DTYPE_FLOAT64
	DTypeUint8   DType = C.MPSGRAPH_DTYPE_UINT8
	DTypeUint16  DType = C.MPSGRAPH_DTYPE_UINT16
	DTypeUint32  DType = C.MPSGRAPH_DTYPE_UINT32
	DTypeUint64  DType = C.MPSGRAPH_DTYPE_UINT64
)

// ReduceType mirrors the MPSGraphReduceType enum.
const (
	ReduceSum     = C.MPSGRAPH_REDUCE_SUM
	ReduceProduct = C.MPSGRAPH_REDUCE_PRODUCT
	ReduceMax     = C.MPSGRAPH_REDUCE_MAX
	ReduceMin     = C.MPSGRAPH_REDUCE_MIN
)

// ScatterMode for scatter operations — must match bridge.m switch cases.
const (
	ScatterModeSet = 0
	ScatterModeAdd = 1
	ScatterModeMax = 2
	ScatterModeMin = 3
)

// Tensor is an opaque handle to an MPSGraphTensor in the computation graph.
type Tensor = unsafe.Pointer

// Context wraps an MPSGraph context (graph + device + command queue).
type Context struct {
	handle C.MPSGraphContextHandle
}

// Exec wraps a compiled MPSGraphExecutable.
type Exec struct {
	handle C.MPSGraphExecHandle
}

// Buffer wraps an MTLBuffer with shared storage for unified memory access.
type Buffer struct {
	handle C.MTLBufferHandle
}

// extractError checks the error struct and returns a Go error if set.
func extractError(err C.MPSGraphError) error {
	if err.code == 0 {
		return nil
	}
	msg := "unknown error"
	if err.message != nil {
		msg = C.GoString(err.message)
		C.free(unsafe.Pointer(err.message))
	}
	return fmt.Errorf("mpsgraph: %s (code %d)", msg, err.code)
}

// ===========================================================================
// Context Lifecycle
// ===========================================================================

// NewContext creates a new MPSGraph context with Metal device and command queue.
func NewContext() (*Context, error) {
	var cErr C.MPSGraphError
	handle := C.mpsgraph_create_context(&cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return &Context{handle: handle}, nil
}

// NewContextWithDevice creates a new MPSGraph context reusing an existing Metal device.
// This avoids creating redundant MTLDevice objects when multiple builders share the same GPU.
func NewContextWithDevice(deviceHandle unsafe.Pointer) (*Context, error) {
	var cErr C.MPSGraphError
	handle := C.mpsgraph_create_context_with_device(deviceHandle, &cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return &Context{handle: handle}, nil
}

// DeviceHandle returns the raw MTLDevice pointer for sharing across contexts.
func (c *Context) DeviceHandle() unsafe.Pointer {
	return C.mpsgraph_device_handle(c.handle)
}

// Destroy releases the context and all associated MPSGraph resources.
func (c *Context) Destroy() {
	if c.handle != nil {
		C.mpsgraph_destroy_context(c.handle)
		c.handle = nil
	}
}

// DeviceName returns the Metal device name.
func (c *Context) DeviceName() string {
	cName := C.mpsgraph_device_name(c.handle)
	if cName == nil {
		return ""
	}
	name := C.GoString(cName)
	C.free(unsafe.Pointer(cName))
	return name
}

// ===========================================================================
// Tensor Creation
// ===========================================================================

// Placeholder creates a placeholder tensor for graph inputs.
func (c *Context) Placeholder(dtype DType, shape []int64) (Tensor, error) {
	var cErr C.MPSGraphError
	var shapePtr *C.int64_t
	if len(shape) > 0 {
		shapePtr = (*C.int64_t)(unsafe.Pointer(&shape[0]))
	}
	t := C.mpsgraph_placeholder(c.handle, C.int(dtype), shapePtr, C.int(len(shape)), &cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return Tensor(t), nil
}

// Constant creates a constant tensor from data.
func (c *Context) Constant(data unsafe.Pointer, nbytes int64, dtype DType, shape []int64) (Tensor, error) {
	var cErr C.MPSGraphError
	var shapePtr *C.int64_t
	if len(shape) > 0 {
		shapePtr = (*C.int64_t)(unsafe.Pointer(&shape[0]))
	}
	t := C.mpsgraph_constant(c.handle, data, C.int64_t(nbytes), C.int(dtype), shapePtr, C.int(len(shape)), &cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return Tensor(t), nil
}

// Iota creates a tensor with sequential values along the specified axis.
func (c *Context) Iota(dtype DType, shape []int64, axis int) (Tensor, error) {
	var cErr C.MPSGraphError
	var shapePtr *C.int64_t
	if len(shape) > 0 {
		shapePtr = (*C.int64_t)(unsafe.Pointer(&shape[0]))
	}
	t := C.mpsgraph_iota(c.handle, C.int(dtype), shapePtr, C.int(len(shape)), C.int(axis), &cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return Tensor(t), nil
}

// ===========================================================================
// Unary Operations - each calls its C function directly (CGo doesn't support
// passing C function pointers as Go function values).
// ===========================================================================

func (c *Context) callUnary(t C.MPSGraphTensorHandle, cErr C.MPSGraphError) (Tensor, error) {
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return Tensor(t), nil
}

func (c *Context) Abs(x Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callUnary(C.mpsgraph_abs(c.handle, C.MPSGraphTensorHandle(x), &e), e)
}
func (c *Context) Neg(x Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callUnary(C.mpsgraph_neg(c.handle, C.MPSGraphTensorHandle(x), &e), e)
}
func (c *Context) Sqrt(x Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callUnary(C.mpsgraph_sqrt(c.handle, C.MPSGraphTensorHandle(x), &e), e)
}
func (c *Context) Rsqrt(x Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callUnary(C.mpsgraph_rsqrt(c.handle, C.MPSGraphTensorHandle(x), &e), e)
}
func (c *Context) Exp(x Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callUnary(C.mpsgraph_exp(c.handle, C.MPSGraphTensorHandle(x), &e), e)
}
func (c *Context) Expm1(x Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callUnary(C.mpsgraph_expm1(c.handle, C.MPSGraphTensorHandle(x), &e), e)
}
func (c *Context) Log(x Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callUnary(C.mpsgraph_log(c.handle, C.MPSGraphTensorHandle(x), &e), e)
}
func (c *Context) Log1p(x Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callUnary(C.mpsgraph_log1p(c.handle, C.MPSGraphTensorHandle(x), &e), e)
}
func (c *Context) Sin(x Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callUnary(C.mpsgraph_sin(c.handle, C.MPSGraphTensorHandle(x), &e), e)
}
func (c *Context) Cos(x Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callUnary(C.mpsgraph_cos(c.handle, C.MPSGraphTensorHandle(x), &e), e)
}
func (c *Context) Tanh(x Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callUnary(C.mpsgraph_tanh(c.handle, C.MPSGraphTensorHandle(x), &e), e)
}
func (c *Context) Sigmoid(x Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callUnary(C.mpsgraph_sigmoid(c.handle, C.MPSGraphTensorHandle(x), &e), e)
}
func (c *Context) Erf(x Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callUnary(C.mpsgraph_erf(c.handle, C.MPSGraphTensorHandle(x), &e), e)
}
func (c *Context) Floor(x Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callUnary(C.mpsgraph_floor(c.handle, C.MPSGraphTensorHandle(x), &e), e)
}
func (c *Context) Ceil(x Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callUnary(C.mpsgraph_ceil(c.handle, C.MPSGraphTensorHandle(x), &e), e)
}
func (c *Context) Round(x Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callUnary(C.mpsgraph_round(c.handle, C.MPSGraphTensorHandle(x), &e), e)
}
func (c *Context) Sign(x Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callUnary(C.mpsgraph_sign(c.handle, C.MPSGraphTensorHandle(x), &e), e)
}
func (c *Context) LogicalNot(x Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callUnary(C.mpsgraph_logical_not(c.handle, C.MPSGraphTensorHandle(x), &e), e)
}
func (c *Context) BitwiseNot(x Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callUnary(C.mpsgraph_bitwise_not(c.handle, C.MPSGraphTensorHandle(x), &e), e)
}
func (c *Context) IsFinite(x Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callUnary(C.mpsgraph_is_finite(c.handle, C.MPSGraphTensorHandle(x), &e), e)
}
func (c *Context) IsNaN(x Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callUnary(C.mpsgraph_is_nan(c.handle, C.MPSGraphTensorHandle(x), &e), e)
}
func (c *Context) Identity(x Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callUnary(C.mpsgraph_identity(c.handle, C.MPSGraphTensorHandle(x), &e), e)
}

// ===========================================================================
// Binary Operations
// ===========================================================================

func (c *Context) callBinary(t C.MPSGraphTensorHandle, cErr C.MPSGraphError) (Tensor, error) {
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return Tensor(t), nil
}

func (c *Context) Add(lhs, rhs Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callBinary(C.mpsgraph_add(c.handle, C.MPSGraphTensorHandle(lhs), C.MPSGraphTensorHandle(rhs), &e), e)
}
func (c *Context) Sub(lhs, rhs Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callBinary(C.mpsgraph_sub(c.handle, C.MPSGraphTensorHandle(lhs), C.MPSGraphTensorHandle(rhs), &e), e)
}
func (c *Context) Mul(lhs, rhs Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callBinary(C.mpsgraph_mul(c.handle, C.MPSGraphTensorHandle(lhs), C.MPSGraphTensorHandle(rhs), &e), e)
}
func (c *Context) Div(lhs, rhs Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callBinary(C.mpsgraph_div(c.handle, C.MPSGraphTensorHandle(lhs), C.MPSGraphTensorHandle(rhs), &e), e)
}
func (c *Context) Rem(lhs, rhs Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callBinary(C.mpsgraph_rem(c.handle, C.MPSGraphTensorHandle(lhs), C.MPSGraphTensorHandle(rhs), &e), e)
}
func (c *Context) Pow(lhs, rhs Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callBinary(C.mpsgraph_pow(c.handle, C.MPSGraphTensorHandle(lhs), C.MPSGraphTensorHandle(rhs), &e), e)
}
func (c *Context) Max(lhs, rhs Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callBinary(C.mpsgraph_max(c.handle, C.MPSGraphTensorHandle(lhs), C.MPSGraphTensorHandle(rhs), &e), e)
}
func (c *Context) Min(lhs, rhs Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callBinary(C.mpsgraph_min(c.handle, C.MPSGraphTensorHandle(lhs), C.MPSGraphTensorHandle(rhs), &e), e)
}
func (c *Context) Atan2(lhs, rhs Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callBinary(C.mpsgraph_atan2(c.handle, C.MPSGraphTensorHandle(lhs), C.MPSGraphTensorHandle(rhs), &e), e)
}
func (c *Context) LogicalAnd(lhs, rhs Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callBinary(C.mpsgraph_logical_and(c.handle, C.MPSGraphTensorHandle(lhs), C.MPSGraphTensorHandle(rhs), &e), e)
}
func (c *Context) LogicalOr(lhs, rhs Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callBinary(C.mpsgraph_logical_or(c.handle, C.MPSGraphTensorHandle(lhs), C.MPSGraphTensorHandle(rhs), &e), e)
}
func (c *Context) LogicalXor(lhs, rhs Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callBinary(C.mpsgraph_logical_xor(c.handle, C.MPSGraphTensorHandle(lhs), C.MPSGraphTensorHandle(rhs), &e), e)
}
func (c *Context) BitwiseAnd(lhs, rhs Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callBinary(C.mpsgraph_bitwise_and(c.handle, C.MPSGraphTensorHandle(lhs), C.MPSGraphTensorHandle(rhs), &e), e)
}
func (c *Context) BitwiseOr(lhs, rhs Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callBinary(C.mpsgraph_bitwise_or(c.handle, C.MPSGraphTensorHandle(lhs), C.MPSGraphTensorHandle(rhs), &e), e)
}
func (c *Context) BitwiseXor(lhs, rhs Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callBinary(C.mpsgraph_bitwise_xor(c.handle, C.MPSGraphTensorHandle(lhs), C.MPSGraphTensorHandle(rhs), &e), e)
}
func (c *Context) ShiftLeft(lhs, rhs Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callBinary(C.mpsgraph_shift_left(c.handle, C.MPSGraphTensorHandle(lhs), C.MPSGraphTensorHandle(rhs), &e), e)
}
func (c *Context) ShiftRight(lhs, rhs Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callBinary(C.mpsgraph_shift_right(c.handle, C.MPSGraphTensorHandle(lhs), C.MPSGraphTensorHandle(rhs), &e), e)
}

// --- Comparison ---

func (c *Context) Equal(lhs, rhs Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callBinary(C.mpsgraph_equal(c.handle, C.MPSGraphTensorHandle(lhs), C.MPSGraphTensorHandle(rhs), &e), e)
}
func (c *Context) NotEqual(lhs, rhs Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callBinary(C.mpsgraph_not_equal(c.handle, C.MPSGraphTensorHandle(lhs), C.MPSGraphTensorHandle(rhs), &e), e)
}
func (c *Context) LessThan(lhs, rhs Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callBinary(C.mpsgraph_less_than(c.handle, C.MPSGraphTensorHandle(lhs), C.MPSGraphTensorHandle(rhs), &e), e)
}
func (c *Context) LessOrEqual(lhs, rhs Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callBinary(C.mpsgraph_less_or_equal(c.handle, C.MPSGraphTensorHandle(lhs), C.MPSGraphTensorHandle(rhs), &e), e)
}
func (c *Context) GreaterThan(lhs, rhs Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callBinary(C.mpsgraph_greater_than(c.handle, C.MPSGraphTensorHandle(lhs), C.MPSGraphTensorHandle(rhs), &e), e)
}
func (c *Context) GreaterOrEqual(lhs, rhs Tensor) (Tensor, error) {
	var e C.MPSGraphError; return c.callBinary(C.mpsgraph_greater_or_equal(c.handle, C.MPSGraphTensorHandle(lhs), C.MPSGraphTensorHandle(rhs), &e), e)
}

// ===========================================================================
// Shape Operations
// ===========================================================================

// Reshape changes the shape of a tensor without changing its data.
func (c *Context) Reshape(x Tensor, shape []int64) (Tensor, error) {
	var cErr C.MPSGraphError
	var shapePtr *C.int64_t
	if len(shape) > 0 {
		shapePtr = (*C.int64_t)(unsafe.Pointer(&shape[0]))
	}
	t := C.mpsgraph_reshape(c.handle, C.MPSGraphTensorHandle(x), shapePtr, C.int(len(shape)), &cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return Tensor(t), nil
}

// Transpose reorders the dimensions of a tensor according to the given permutation.
func (c *Context) Transpose(x Tensor, permutation []int) (Tensor, error) {
	var cErr C.MPSGraphError
	cPerm := make([]C.int, len(permutation))
	for i, v := range permutation {
		cPerm[i] = C.int(v)
	}
	t := C.mpsgraph_transpose(c.handle, C.MPSGraphTensorHandle(x), &cPerm[0], C.int(len(permutation)), &cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return Tensor(t), nil
}

// Cast converts a tensor to the specified data type.
func (c *Context) Cast(x Tensor, dtype DType) (Tensor, error) {
	var cErr C.MPSGraphError
	t := C.mpsgraph_cast(c.handle, C.MPSGraphTensorHandle(x), C.int(dtype), &cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return Tensor(t), nil
}

// BroadcastTo broadcasts a tensor to the given shape.
func (c *Context) BroadcastTo(x Tensor, shape []int64) (Tensor, error) {
	var cErr C.MPSGraphError
	var shapePtr *C.int64_t
	if len(shape) > 0 {
		shapePtr = (*C.int64_t)(unsafe.Pointer(&shape[0]))
	}
	t := C.mpsgraph_broadcast_to(c.handle, C.MPSGraphTensorHandle(x), shapePtr, C.int(len(shape)), &cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return Tensor(t), nil
}

// Slice extracts a sub-tensor using start/end/stride per dimension.
func (c *Context) Slice(x Tensor, starts, ends, strides []int64) (Tensor, error) {
	var cErr C.MPSGraphError
	rank := len(starts)
	t := C.mpsgraph_slice(c.handle, C.MPSGraphTensorHandle(x),
		(*C.int64_t)(unsafe.Pointer(&starts[0])),
		(*C.int64_t)(unsafe.Pointer(&ends[0])),
		(*C.int64_t)(unsafe.Pointer(&strides[0])),
		C.int(rank), &cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return Tensor(t), nil
}

// Concatenate joins tensors along the specified axis.
func (c *Context) Concatenate(tensors []Tensor, axis int) (Tensor, error) {
	var cErr C.MPSGraphError
	cTensors := make([]C.MPSGraphTensorHandle, len(tensors))
	for i, t := range tensors {
		cTensors[i] = C.MPSGraphTensorHandle(t)
	}
	result := C.mpsgraph_concatenate(c.handle, &cTensors[0], C.int(len(tensors)), C.int(axis), &cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return Tensor(result), nil
}

// Reverse reverses a tensor along the specified axes.
func (c *Context) Reverse(x Tensor, axes []int) (Tensor, error) {
	var cErr C.MPSGraphError
	cAxes := make([]C.int, len(axes))
	for i, a := range axes {
		cAxes[i] = C.int(a)
	}
	t := C.mpsgraph_reverse(c.handle, C.MPSGraphTensorHandle(x), &cAxes[0], C.int(len(axes)), &cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return Tensor(t), nil
}

// Pad pads a tensor with the given padding amounts. padValue is a scalar constant tensor.
func (c *Context) Pad(x, padValue Tensor, padBefore, padAfter []int64) (Tensor, error) {
	var cErr C.MPSGraphError
	rank := len(padBefore)
	t := C.mpsgraph_pad(c.handle, C.MPSGraphTensorHandle(x), C.MPSGraphTensorHandle(padValue),
		(*C.int64_t)(unsafe.Pointer(&padBefore[0])),
		(*C.int64_t)(unsafe.Pointer(&padAfter[0])),
		C.int(rank), &cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return Tensor(t), nil
}

// ===========================================================================
// Ternary / Selection
// ===========================================================================

// Where selects elements from onTrue or onFalse based on cond.
func (c *Context) Where(cond, onTrue, onFalse Tensor) (Tensor, error) {
	var cErr C.MPSGraphError
	t := C.mpsgraph_where(c.handle, C.MPSGraphTensorHandle(cond),
		C.MPSGraphTensorHandle(onTrue), C.MPSGraphTensorHandle(onFalse), &cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return Tensor(t), nil
}

// Clamp clamps x between minVal and maxVal.
func (c *Context) Clamp(minVal, x, maxVal Tensor) (Tensor, error) {
	var cErr C.MPSGraphError
	t := C.mpsgraph_clamp(c.handle, C.MPSGraphTensorHandle(minVal),
		C.MPSGraphTensorHandle(x), C.MPSGraphTensorHandle(maxVal), &cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return Tensor(t), nil
}

// ===========================================================================
// Matrix Operations
// ===========================================================================

// MatMul performs matrix multiplication (supports batched).
func (c *Context) MatMul(lhs, rhs Tensor) (Tensor, error) {
	var cErr C.MPSGraphError
	t := C.mpsgraph_matmul(c.handle, C.MPSGraphTensorHandle(lhs), C.MPSGraphTensorHandle(rhs), &cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return Tensor(t), nil
}

// ===========================================================================
// Reduction Operations
// ===========================================================================

// Reduce performs a reduction operation along the specified axes.
func (c *Context) Reduce(x Tensor, reduceType int, axes []int) (Tensor, error) {
	var cErr C.MPSGraphError
	cAxes := make([]C.int, len(axes))
	for i, a := range axes {
		cAxes[i] = C.int(a)
	}
	t := C.mpsgraph_reduce(c.handle, C.MPSGraphTensorHandle(x), C.int(reduceType),
		&cAxes[0], C.int(len(axes)), &cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return Tensor(t), nil
}

// ===========================================================================
// Gather / Scatter
// ===========================================================================

// GatherND performs an N-dimensional gather.
func (c *Context) GatherND(params, indices Tensor, batchDims int) (Tensor, error) {
	var cErr C.MPSGraphError
	t := C.mpsgraph_gather_nd(c.handle, C.MPSGraphTensorHandle(params),
		C.MPSGraphTensorHandle(indices), C.int(batchDims), &cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return Tensor(t), nil
}

// GatherAlongAxis gathers elements along a single axis using indices.
func (c *Context) GatherAlongAxis(x, indices Tensor, axis int) (Tensor, error) {
	var cErr C.MPSGraphError
	t := C.mpsgraph_gather_along_axis(c.handle, C.MPSGraphTensorHandle(x),
		C.MPSGraphTensorHandle(indices), C.int(axis), &cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return Tensor(t), nil
}

// ScatterND performs an N-dimensional scatter with the given mode.
func (c *Context) ScatterND(data, indices, updates Tensor, shape []int64, mode int) (Tensor, error) {
	var cErr C.MPSGraphError
	var shapePtr *C.int64_t
	if len(shape) > 0 {
		shapePtr = (*C.int64_t)(unsafe.Pointer(&shape[0]))
	}
	t := C.mpsgraph_scatter_nd(c.handle, C.MPSGraphTensorHandle(data),
		C.MPSGraphTensorHandle(indices), C.MPSGraphTensorHandle(updates),
		shapePtr, C.int(len(shape)), C.int(mode), &cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return Tensor(t), nil
}

// ===========================================================================
// ArgMin / ArgMax
// ===========================================================================

// ArgMin returns indices of minimum values along the specified axis.
func (c *Context) ArgMin(x Tensor, axis int, outputDtype DType) (Tensor, error) {
	var cErr C.MPSGraphError
	t := C.mpsgraph_argmin(c.handle, C.MPSGraphTensorHandle(x), C.int(axis), C.int(outputDtype), &cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return Tensor(t), nil
}

// ArgMax returns indices of maximum values along the specified axis.
func (c *Context) ArgMax(x Tensor, axis int, outputDtype DType) (Tensor, error) {
	var cErr C.MPSGraphError
	t := C.mpsgraph_argmax(c.handle, C.MPSGraphTensorHandle(x), C.int(axis), C.int(outputDtype), &cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return Tensor(t), nil
}

// ===========================================================================
// Batch Normalization
// ===========================================================================

// BatchNormInference performs batch normalization for inference.
func (c *Context) BatchNormInference(input, mean, variance, gamma, beta Tensor, epsilon float32, featureAxis int) (Tensor, error) {
	var cErr C.MPSGraphError
	t := C.mpsgraph_batch_norm_inference(c.handle,
		C.MPSGraphTensorHandle(input), C.MPSGraphTensorHandle(mean),
		C.MPSGraphTensorHandle(variance), C.MPSGraphTensorHandle(gamma),
		C.MPSGraphTensorHandle(beta), C.float(epsilon), C.int(featureAxis), &cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return Tensor(t), nil
}

// ===========================================================================
// Softmax
// ===========================================================================

// Softmax computes softmax along the given axis.
func (c *Context) Softmax(x Tensor, axis int) (Tensor, error) {
	var cErr C.MPSGraphError
	t := C.mpsgraph_softmax(c.handle, C.MPSGraphTensorHandle(x), C.int(axis), &cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return Tensor(t), nil
}

// ===========================================================================
// Pooling (ReduceWindow)
// ===========================================================================

// Pool2D performs 2D pooling (max or average).
// mode: 0=max, 1=avg
func (c *Context) Pool2D(x Tensor, mode int, windowDims, strides, padBefore, padAfter []int64) (Tensor, error) {
	var cErr C.MPSGraphError
	t := C.mpsgraph_pool2d(c.handle, C.MPSGraphTensorHandle(x), C.int(mode),
		(*C.int64_t)(unsafe.Pointer(&windowDims[0])),
		(*C.int64_t)(unsafe.Pointer(&strides[0])),
		(*C.int64_t)(unsafe.Pointer(&padBefore[0])),
		(*C.int64_t)(unsafe.Pointer(&padAfter[0])),
		&cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return Tensor(t), nil
}

// MaxPool2DGradient computes the gradient of max pooling 2D (SelectAndScatter for MaxPool backprop).
// gradient: incoming gradient (same shape as maxpool output, NCHW)
// source: original input to maxpool (NCHW)
// Returns: gradient with respect to source (same shape as source).
func (c *Context) MaxPool2DGradient(gradient, source Tensor, windowDims, strides, padBefore, padAfter []int64) (Tensor, error) {
	var cErr C.MPSGraphError
	t := C.mpsgraph_max_pool2d_gradient(c.handle,
		C.MPSGraphTensorHandle(gradient), C.MPSGraphTensorHandle(source),
		(*C.int64_t)(unsafe.Pointer(&windowDims[0])),
		(*C.int64_t)(unsafe.Pointer(&strides[0])),
		(*C.int64_t)(unsafe.Pointer(&padBefore[0])),
		(*C.int64_t)(unsafe.Pointer(&padAfter[0])),
		&cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return Tensor(t), nil
}

// ===========================================================================
// Random Number Generation
// ===========================================================================

// RandomUniform generates a tensor filled with uniform random values in [0, 1).
func (c *Context) RandomUniform(dtype DType, shape []int64) (Tensor, error) {
	var cErr C.MPSGraphError
	var shapePtr *C.int64_t
	if len(shape) > 0 {
		shapePtr = (*C.int64_t)(unsafe.Pointer(&shape[0]))
	}
	t := C.mpsgraph_random_uniform(c.handle, C.int(dtype), shapePtr, C.int(len(shape)), &cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return Tensor(t), nil
}

// ===========================================================================
// Convolution (General)
// ===========================================================================

// ConvGeneral performs a general N-D convolution (currently only 2D supported).
func (c *Context) ConvGeneral(input, kernel Tensor, numSpatialDims int,
	strides, dilations, padBefore, padAfter []int64, groups int) (Tensor, error) {
	var cErr C.MPSGraphError
	t := C.mpsgraph_conv_general(c.handle,
		C.MPSGraphTensorHandle(input), C.MPSGraphTensorHandle(kernel),
		C.int(numSpatialDims),
		(*C.int64_t)(unsafe.Pointer(&strides[0])),
		(*C.int64_t)(unsafe.Pointer(&dilations[0])),
		(*C.int64_t)(unsafe.Pointer(&padBefore[0])),
		(*C.int64_t)(unsafe.Pointer(&padAfter[0])),
		C.int(groups), &cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return Tensor(t), nil
}

// ===========================================================================
// Scatter Along Axis
// ===========================================================================

// ScatterAlongAxis scatters updates into data along the given axis using indices.
func (c *Context) ScatterAlongAxis(data, indices, updates Tensor, axis, mode int) (Tensor, error) {
	var cErr C.MPSGraphError
	t := C.mpsgraph_scatter_along_axis(c.handle,
		C.MPSGraphTensorHandle(data), C.MPSGraphTensorHandle(indices),
		C.MPSGraphTensorHandle(updates), C.int(axis), C.int(mode), &cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return Tensor(t), nil
}

// ===========================================================================
// Dynamic Slice
// ===========================================================================

// DynamicSlice performs a dynamic slice with runtime start indices.
func (c *Context) DynamicSlice(x Tensor, startIndices []Tensor, sliceSizes []int64) (Tensor, error) {
	var cErr C.MPSGraphError
	numIndices := len(startIndices)

	cStartIndices := make([]C.MPSGraphTensorHandle, numIndices)
	for i, idx := range startIndices {
		cStartIndices[i] = C.MPSGraphTensorHandle(idx)
	}

	t := C.mpsgraph_dynamic_slice(c.handle, C.MPSGraphTensorHandle(x),
		&cStartIndices[0], C.int(numIndices),
		(*C.int64_t)(unsafe.Pointer(&sliceSizes[0])), C.int(len(sliceSizes)), &cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return Tensor(t), nil
}

// DynamicUpdateSlice replaces a slice of the input with the update at the given start indices.
func (c *Context) DynamicUpdateSlice(x, update Tensor, startIndices []Tensor) (Tensor, error) {
	var cErr C.MPSGraphError
	numIndices := len(startIndices)

	cStartIndices := make([]C.MPSGraphTensorHandle, numIndices)
	for i, idx := range startIndices {
		cStartIndices[i] = C.MPSGraphTensorHandle(idx)
	}

	t := C.mpsgraph_dynamic_update_slice(c.handle, C.MPSGraphTensorHandle(x), C.MPSGraphTensorHandle(update),
		&cStartIndices[0], C.int(numIndices), &cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return Tensor(t), nil
}

// ===========================================================================
// Compilation & Execution
// ===========================================================================

// CompileInfo holds the information needed to compile a graph.
type CompileInfo struct {
	Feeds       []Tensor
	FeedDtypes  []DType
	FeedShapes  [][]int64
	Targets     []Tensor
}

// Compile compiles the graph into an executable.
func (c *Context) Compile(info CompileInfo) (*Exec, error) {
	var cErr C.MPSGraphError

	numFeeds := len(info.Feeds)
	numTargets := len(info.Targets)

	// Pin Go memory that will be referenced indirectly by C arrays.
	var pinner runtime.Pinner
	defer pinner.Unpin()

	// Prepare feeds.
	cFeeds := make([]C.MPSGraphTensorHandle, numFeeds)
	cFeedDtypes := make([]C.int, numFeeds)
	cFeedShapes := make([]*C.int64_t, numFeeds)
	cFeedRanks := make([]C.int, numFeeds)

	for i := range numFeeds {
		cFeeds[i] = C.MPSGraphTensorHandle(info.Feeds[i])
		cFeedDtypes[i] = C.int(info.FeedDtypes[i])
		if len(info.FeedShapes[i]) > 0 {
			pinner.Pin(&info.FeedShapes[i][0])
			cFeedShapes[i] = (*C.int64_t)(unsafe.Pointer(&info.FeedShapes[i][0]))
		}
		cFeedRanks[i] = C.int(len(info.FeedShapes[i]))
	}

	// Prepare targets.
	cTargets := make([]C.MPSGraphTensorHandle, numTargets)
	for i := range numTargets {
		cTargets[i] = C.MPSGraphTensorHandle(info.Targets[i])
	}

	var feedsPtr *C.MPSGraphTensorHandle
	var feedDtypesPtr *C.int
	var feedShapesPtr **C.int64_t
	var feedRanksPtr *C.int
	if numFeeds > 0 {
		feedsPtr = &cFeeds[0]
		feedDtypesPtr = &cFeedDtypes[0]
		feedShapesPtr = &cFeedShapes[0]
		feedRanksPtr = &cFeedRanks[0]
	}

	var targetsPtr *C.MPSGraphTensorHandle
	if numTargets > 0 {
		targetsPtr = &cTargets[0]
	}

	handle := C.mpsgraph_compile(c.handle,
		feedsPtr, feedDtypesPtr, feedShapesPtr, feedRanksPtr, C.int(numFeeds),
		targetsPtr, C.int(numTargets),
		&cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return &Exec{handle: handle}, nil
}

// ExecInput describes one input for execution.
type ExecInput struct {
	Data   unsafe.Pointer
	Size   int64
	DType  DType
	Shape  []int64
}

// ExecOutput describes one output buffer for execution.
type ExecOutput struct {
	Data   unsafe.Pointer
	Size   int64
	DType  DType
	Shape  []int64
}

// Execute runs the compiled graph with the given inputs and writes results to outputs.
func (e *Exec) Execute(inputs []ExecInput, outputs []ExecOutput) error {
	// Lock this goroutine to a single OS thread for the entire Metal execution.
	// This prevents the Go scheduler from moving us between threads mid-execution,
	// which can cause Metal/MPSGraph corruption.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.MPSGraphError
	numInputs := len(inputs)
	numOutputs := len(outputs)

	// Pin Go memory that will be referenced indirectly by C arrays.
	var pinner runtime.Pinner
	defer pinner.Unpin()

	// Prepare input arrays.
	cInputData := make([]unsafe.Pointer, numInputs)
	cInputSizes := make([]C.int64_t, numInputs)
	cInputDtypes := make([]C.int, numInputs)
	cInputShapes := make([]*C.int64_t, numInputs)
	cInputRanks := make([]C.int, numInputs)

	for i := range numInputs {
		if inputs[i].Data != nil {
			pinner.Pin(inputs[i].Data)
		}
		cInputData[i] = inputs[i].Data
		cInputSizes[i] = C.int64_t(inputs[i].Size)
		cInputDtypes[i] = C.int(inputs[i].DType)
		if len(inputs[i].Shape) > 0 {
			pinner.Pin(&inputs[i].Shape[0])
			cInputShapes[i] = (*C.int64_t)(unsafe.Pointer(&inputs[i].Shape[0]))
		}
		cInputRanks[i] = C.int(len(inputs[i].Shape))
	}

	// Prepare output arrays.
	cOutputData := make([]unsafe.Pointer, numOutputs)
	cOutputSizes := make([]C.int64_t, numOutputs)
	cOutputDtypes := make([]C.int, numOutputs)
	cOutputShapes := make([]*C.int64_t, numOutputs)
	cOutputRanks := make([]C.int, numOutputs)

	for i := range numOutputs {
		if outputs[i].Data != nil {
			pinner.Pin(outputs[i].Data)
		}
		cOutputData[i] = outputs[i].Data
		cOutputSizes[i] = C.int64_t(outputs[i].Size)
		cOutputDtypes[i] = C.int(outputs[i].DType)
		if len(outputs[i].Shape) > 0 {
			pinner.Pin(&outputs[i].Shape[0])
			cOutputShapes[i] = (*C.int64_t)(unsafe.Pointer(&outputs[i].Shape[0]))
		}
		cOutputRanks[i] = C.int(len(outputs[i].Shape))
	}

	var inputDataPtr *unsafe.Pointer
	var inputSizesPtr *C.int64_t
	var inputDtypesPtr *C.int
	var inputShapesPtr **C.int64_t
	var inputRanksPtr *C.int
	if numInputs > 0 {
		inputDataPtr = &cInputData[0]
		inputSizesPtr = &cInputSizes[0]
		inputDtypesPtr = &cInputDtypes[0]
		inputShapesPtr = &cInputShapes[0]
		inputRanksPtr = &cInputRanks[0]
	}

	var outputDataPtr *unsafe.Pointer
	var outputSizesPtr *C.int64_t
	var outputDtypesPtr *C.int
	var outputShapesPtr **C.int64_t
	var outputRanksPtr *C.int
	if numOutputs > 0 {
		outputDataPtr = &cOutputData[0]
		outputSizesPtr = &cOutputSizes[0]
		outputDtypesPtr = &cOutputDtypes[0]
		outputShapesPtr = &cOutputShapes[0]
		outputRanksPtr = &cOutputRanks[0]
	}

	ok := C.mpsgraph_execute(e.handle,
		inputDataPtr, inputSizesPtr, inputDtypesPtr, inputShapesPtr, inputRanksPtr, C.int(numInputs),
		outputDataPtr, outputSizesPtr, outputDtypesPtr, outputShapesPtr, outputRanksPtr, C.int(numOutputs),
		&cErr)
	if !ok {
		if err := extractError(cErr); err != nil {
			return err
		}
		return fmt.Errorf("mpsgraph: execution failed")
	}
	return nil
}

// Destroy releases the compiled executable.
func (e *Exec) Destroy() {
	if e.handle != nil {
		C.mpsgraph_destroy_exec(e.handle)
		e.handle = nil
	}
}

// ===========================================================================
// Buffer Management
// ===========================================================================

// NewBuffer creates a new MTLBuffer with shared storage for unified memory access.
func (c *Context) NewBuffer(nbytes int64) (*Buffer, error) {
	var cErr C.MPSGraphError
	handle := C.mpsgraph_buffer_create(c.handle, C.int64_t(nbytes), &cErr)
	if err := extractError(cErr); err != nil {
		return nil, err
	}
	return &Buffer{handle: handle}, nil
}

// Contents returns a pointer to the buffer's shared memory (valid for CPU and GPU).
func (b *Buffer) Contents() unsafe.Pointer {
	return C.mpsgraph_buffer_contents(b.handle)
}

// Length returns the buffer size in bytes.
func (b *Buffer) Length() int64 {
	return int64(C.mpsgraph_buffer_length(b.handle))
}

// Destroy releases the MTLBuffer.
func (b *Buffer) Destroy() {
	if b.handle != nil {
		C.mpsgraph_buffer_destroy(b.handle)
		b.handle = nil
	}
}
