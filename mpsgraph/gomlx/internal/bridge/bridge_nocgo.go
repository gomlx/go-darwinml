// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && !cgo

// Package bridge provides CGo bindings to Apple's MPSGraph framework.
// This file is a stub for when CGO is disabled.
package bridge

import (
	"fmt"
	"unsafe"
)

type DType = int

const (
	DTypeBool    DType = 0
	DTypeInt8    DType = 1
	DTypeInt16   DType = 2
	DTypeInt32   DType = 3
	DTypeInt64   DType = 4
	DTypeFloat16 DType = 5
	DTypeBF16    DType = 6
	DTypeFloat32 DType = 7
	DTypeFloat64 DType = 8
	DTypeUint8   DType = 9
	DTypeUint16  DType = 10
	DTypeUint32  DType = 11
	DTypeUint64  DType = 12
)

const (
	ReduceSum     = 0
	ReduceProduct = 1
	ReduceMax     = 2
	ReduceMin     = 3
)

const (
	ScatterModeSet = 0
	ScatterModeAdd = 1
	ScatterModeMax = 2
	ScatterModeMin = 3
)

type Tensor = unsafe.Pointer

type Context struct{}
type Exec struct{}
type Buffer struct{}

var errNoCGo = fmt.Errorf("mpsgraph: CGO is disabled; rebuild with CGO_ENABLED=1")

func NewContext() (*Context, error)                                                   { return nil, errNoCGo }
func NewContextWithDevice(deviceHandle unsafe.Pointer) (*Context, error)              { return nil, errNoCGo }
func (c *Context) Destroy()                                                           {}
func (c *Context) DeviceHandle() unsafe.Pointer                                       { return nil }
func (c *Context) DeviceName() string                                                 { return "" }
func (c *Context) Placeholder(dtype DType, shape []int64) (Tensor, error)   { return nil, errNoCGo }
func (c *Context) Constant(data unsafe.Pointer, nbytes int64, dtype DType, shape []int64) (Tensor, error) {
	return nil, errNoCGo
}
func (c *Context) Iota(dtype DType, shape []int64, axis int) (Tensor, error) { return nil, errNoCGo }

func (c *Context) Abs(x Tensor) (Tensor, error)        { return nil, errNoCGo }
func (c *Context) Neg(x Tensor) (Tensor, error)        { return nil, errNoCGo }
func (c *Context) Sqrt(x Tensor) (Tensor, error)       { return nil, errNoCGo }
func (c *Context) Rsqrt(x Tensor) (Tensor, error)      { return nil, errNoCGo }
func (c *Context) Exp(x Tensor) (Tensor, error)        { return nil, errNoCGo }
func (c *Context) Expm1(x Tensor) (Tensor, error)      { return nil, errNoCGo }
func (c *Context) Log(x Tensor) (Tensor, error)        { return nil, errNoCGo }
func (c *Context) Log1p(x Tensor) (Tensor, error)      { return nil, errNoCGo }
func (c *Context) Sin(x Tensor) (Tensor, error)        { return nil, errNoCGo }
func (c *Context) Cos(x Tensor) (Tensor, error)        { return nil, errNoCGo }
func (c *Context) Tanh(x Tensor) (Tensor, error)       { return nil, errNoCGo }
func (c *Context) Sigmoid(x Tensor) (Tensor, error)    { return nil, errNoCGo }
func (c *Context) Erf(x Tensor) (Tensor, error)        { return nil, errNoCGo }
func (c *Context) Floor(x Tensor) (Tensor, error)      { return nil, errNoCGo }
func (c *Context) Ceil(x Tensor) (Tensor, error)       { return nil, errNoCGo }
func (c *Context) Round(x Tensor) (Tensor, error)      { return nil, errNoCGo }
func (c *Context) Sign(x Tensor) (Tensor, error)       { return nil, errNoCGo }
func (c *Context) LogicalNot(x Tensor) (Tensor, error) { return nil, errNoCGo }
func (c *Context) BitwiseNot(x Tensor) (Tensor, error) { return nil, errNoCGo }
func (c *Context) IsFinite(x Tensor) (Tensor, error)   { return nil, errNoCGo }
func (c *Context) IsNaN(x Tensor) (Tensor, error)      { return nil, errNoCGo }
func (c *Context) Identity(x Tensor) (Tensor, error)   { return nil, errNoCGo }

func (c *Context) Add(lhs, rhs Tensor) (Tensor, error)          { return nil, errNoCGo }
func (c *Context) Sub(lhs, rhs Tensor) (Tensor, error)          { return nil, errNoCGo }
func (c *Context) Mul(lhs, rhs Tensor) (Tensor, error)          { return nil, errNoCGo }
func (c *Context) Div(lhs, rhs Tensor) (Tensor, error)          { return nil, errNoCGo }
func (c *Context) Rem(lhs, rhs Tensor) (Tensor, error)          { return nil, errNoCGo }
func (c *Context) Pow(lhs, rhs Tensor) (Tensor, error)          { return nil, errNoCGo }
func (c *Context) Max(lhs, rhs Tensor) (Tensor, error)          { return nil, errNoCGo }
func (c *Context) Min(lhs, rhs Tensor) (Tensor, error)          { return nil, errNoCGo }
func (c *Context) Atan2(lhs, rhs Tensor) (Tensor, error)        { return nil, errNoCGo }
func (c *Context) LogicalAnd(lhs, rhs Tensor) (Tensor, error)   { return nil, errNoCGo }
func (c *Context) LogicalOr(lhs, rhs Tensor) (Tensor, error)    { return nil, errNoCGo }
func (c *Context) LogicalXor(lhs, rhs Tensor) (Tensor, error)   { return nil, errNoCGo }
func (c *Context) BitwiseAnd(lhs, rhs Tensor) (Tensor, error)   { return nil, errNoCGo }
func (c *Context) BitwiseOr(lhs, rhs Tensor) (Tensor, error)    { return nil, errNoCGo }
func (c *Context) BitwiseXor(lhs, rhs Tensor) (Tensor, error)   { return nil, errNoCGo }
func (c *Context) ShiftLeft(lhs, rhs Tensor) (Tensor, error)    { return nil, errNoCGo }
func (c *Context) ShiftRight(lhs, rhs Tensor) (Tensor, error)   { return nil, errNoCGo }
func (c *Context) Equal(lhs, rhs Tensor) (Tensor, error)        { return nil, errNoCGo }
func (c *Context) NotEqual(lhs, rhs Tensor) (Tensor, error)     { return nil, errNoCGo }
func (c *Context) LessThan(lhs, rhs Tensor) (Tensor, error)     { return nil, errNoCGo }
func (c *Context) LessOrEqual(lhs, rhs Tensor) (Tensor, error)  { return nil, errNoCGo }
func (c *Context) GreaterThan(lhs, rhs Tensor) (Tensor, error)  { return nil, errNoCGo }
func (c *Context) GreaterOrEqual(lhs, rhs Tensor) (Tensor, error) { return nil, errNoCGo }

func (c *Context) Reshape(x Tensor, shape []int64) (Tensor, error)            { return nil, errNoCGo }
func (c *Context) Transpose(x Tensor, permutation []int) (Tensor, error)      { return nil, errNoCGo }
func (c *Context) Cast(x Tensor, dtype DType) (Tensor, error)                 { return nil, errNoCGo }
func (c *Context) BroadcastTo(x Tensor, shape []int64) (Tensor, error)        { return nil, errNoCGo }
func (c *Context) Slice(x Tensor, starts, ends, strides []int64) (Tensor, error) { return nil, errNoCGo }
func (c *Context) Concatenate(tensors []Tensor, axis int) (Tensor, error)      { return nil, errNoCGo }
func (c *Context) Reverse(x Tensor, axes []int) (Tensor, error)               { return nil, errNoCGo }
func (c *Context) Pad(x, padValue Tensor, padBefore, padAfter []int64) (Tensor, error) { return nil, errNoCGo }

func (c *Context) Where(cond, onTrue, onFalse Tensor) (Tensor, error)  { return nil, errNoCGo }
func (c *Context) Clamp(minVal, x, maxVal Tensor) (Tensor, error)      { return nil, errNoCGo }
func (c *Context) MatMul(lhs, rhs Tensor) (Tensor, error)              { return nil, errNoCGo }
func (c *Context) Reduce(x Tensor, reduceType int, axes []int) (Tensor, error) { return nil, errNoCGo }
func (c *Context) GatherND(params, indices Tensor, batchDims int) (Tensor, error) { return nil, errNoCGo }
func (c *Context) GatherAlongAxis(x, indices Tensor, axis int) (Tensor, error)    { return nil, errNoCGo }
func (c *Context) ScatterND(data, indices, updates Tensor, shape []int64, mode int) (Tensor, error) {
	return nil, errNoCGo
}
func (c *Context) ArgMin(x Tensor, axis int, outputDtype DType) (Tensor, error) { return nil, errNoCGo }
func (c *Context) ArgMax(x Tensor, axis int, outputDtype DType) (Tensor, error) { return nil, errNoCGo }
func (c *Context) Pool2D(x Tensor, mode int, windowDims, strides, padBefore, padAfter []int64) (Tensor, error) {
	return nil, errNoCGo
}
func (c *Context) MaxPool2DGradient(gradient, source Tensor, windowDims, strides, padBefore, padAfter []int64) (Tensor, error) {
	return nil, errNoCGo
}
func (c *Context) DynamicSlice(x Tensor, starts []Tensor, sizes []int64) (Tensor, error) {
	return nil, errNoCGo
}
func (c *Context) DynamicUpdateSlice(x, update Tensor, starts []Tensor) (Tensor, error) {
	return nil, errNoCGo
}
func (c *Context) RandomUniform(dtype DType, shape []int64) (Tensor, error) { return nil, errNoCGo }
func (c *Context) Softmax(x Tensor, axis int) (Tensor, error)              { return nil, errNoCGo }
func (c *Context) ConvGeneral(input, kernel Tensor, numSpatialDims int, strides, dilations, padBefore, padAfter []int64, groups int) (Tensor, error) {
	return nil, errNoCGo
}
func (c *Context) ScatterAlongAxis(data, indices, updates Tensor, axis, mode int) (Tensor, error) {
	return nil, errNoCGo
}
func (c *Context) BatchNormInference(input, mean, variance, gamma, beta Tensor, epsilon float32, featureAxis int) (Tensor, error) {
	return nil, errNoCGo
}

type CompileInfo struct {
	Feeds      []Tensor
	FeedDtypes []DType
	FeedShapes [][]int64
	Targets    []Tensor
}

func (c *Context) Compile(info CompileInfo) (*Exec, error)       { return nil, errNoCGo }
func (e *Exec) Execute(inputs []ExecInput, outputs []ExecOutput) error { return errNoCGo }
func (e *Exec) Destroy()                                               {}

type ExecInput struct {
	Data  unsafe.Pointer
	Size  int64
	DType DType
	Shape []int64
}

type ExecOutput struct {
	Data  unsafe.Pointer
	Size  int64
	DType DType
	Shape []int64
}

func (c *Context) NewBuffer(nbytes int64) (*Buffer, error) { return nil, errNoCGo }
func (b *Buffer) Contents() unsafe.Pointer                 { return nil }
func (b *Buffer) Length() int64                             { return 0 }
func (b *Buffer) Destroy()                                 {}
