// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && !cgo

// Package mlxbridge provides CGo bindings to Apple's MLX framework.
// This file is a stub for when CGO is disabled.
package mlxbridge

import (
	"fmt"
	"unsafe"
)

type DType = int

const (
	DTypeBool      DType = 0
	DTypeInt8      DType = 1
	DTypeInt16     DType = 2
	DTypeInt32     DType = 3
	DTypeInt64     DType = 4
	DTypeFloat16   DType = 5
	DTypeBF16      DType = 6
	DTypeFloat32   DType = 7
	DTypeFloat64   DType = 8
	DTypeUint8     DType = 9
	DTypeUint16    DType = 10
	DTypeUint32    DType = 11
	DTypeUint64    DType = 12
	DTypeComplex64 DType = 13
)

type UnaryOp = int

const (
	UnaryAbs        UnaryOp = 0
	UnaryNeg        UnaryOp = 1
	UnarySqrt       UnaryOp = 2
	UnaryRsqrt      UnaryOp = 3
	UnaryExp        UnaryOp = 4
	UnaryExpm1      UnaryOp = 5
	UnaryLog        UnaryOp = 6
	UnaryLog1p      UnaryOp = 7
	UnarySin        UnaryOp = 8
	UnaryCos        UnaryOp = 9
	UnaryTanh       UnaryOp = 10
	UnaryLogistic   UnaryOp = 11
	UnaryErf        UnaryOp = 12
	UnaryFloor      UnaryOp = 13
	UnaryCeil       UnaryOp = 14
	UnaryRound      UnaryOp = 15
	UnarySign       UnaryOp = 16
	UnaryLogicalNot UnaryOp = 17
	UnaryBitwiseNot UnaryOp = 18
	UnaryIsFinite   UnaryOp = 19
	UnaryIsNaN      UnaryOp = 20
	UnaryConj       UnaryOp = 21
	UnaryReal       UnaryOp = 22
	UnaryImag       UnaryOp = 23
	UnaryBitCount   UnaryOp = 24
	UnaryClz        UnaryOp = 25
	UnaryIdentity   UnaryOp = 26
)

type BinaryOp = int

const (
	BinaryAdd                  BinaryOp = 0
	BinarySub                  BinaryOp = 1
	BinaryMul                  BinaryOp = 2
	BinaryDiv                  BinaryOp = 3
	BinaryRem                  BinaryOp = 4
	BinaryPow                  BinaryOp = 5
	BinaryMax                  BinaryOp = 6
	BinaryMin                  BinaryOp = 7
	BinaryAtan2                BinaryOp = 8
	BinaryLogicalAnd           BinaryOp = 9
	BinaryLogicalOr            BinaryOp = 10
	BinaryLogicalXor           BinaryOp = 11
	BinaryBitwiseAnd           BinaryOp = 12
	BinaryBitwiseOr            BinaryOp = 13
	BinaryBitwiseXor           BinaryOp = 14
	BinaryShiftLeft            BinaryOp = 15
	BinaryShiftRightArithmetic BinaryOp = 16
	BinaryShiftRightLogical    BinaryOp = 17
	BinaryComplex              BinaryOp = 18
	BinaryEqual                BinaryOp = 19
	BinaryNotEqual             BinaryOp = 20
	BinaryLessThan             BinaryOp = 21
	BinaryLessOrEqual          BinaryOp = 22
	BinaryGreaterThan          BinaryOp = 23
	BinaryGreaterOrEqual       BinaryOp = 24
)

type ArrayHandle uintptr
type NodeID = int32

type Tape struct{}
type Exec struct{}

var errNoCGo = fmt.Errorf("mlx: CGO is disabled; rebuild with CGO_ENABLED=1")

func Init() error                                                           { return errNoCGo }
func DeviceName() (string, error)                                           { return "", errNoCGo }
func Version() (string, error)                                              { return "", errNoCGo }
func NewArray(nbytes int64, dtype DType, shape []int) (ArrayHandle, error) { return 0, errNoCGo }
func NewArrayFromData(data unsafe.Pointer, nbytes int64, dtype DType, shape []int) (ArrayHandle, error) {
	return 0, errNoCGo
}
func ArrayDataPtr(arr ArrayHandle) unsafe.Pointer { return nil }
func ArrayNBytes(arr ArrayHandle) int64           { return 0 }
func ArrayDType(arr ArrayHandle) DType            { return DTypeFloat32 }
func ArrayShape(arr ArrayHandle) []int            { return nil }
func ArrayIsRowContiguous(arr ArrayHandle) bool   { return false }
func ArrayEnsureContiguous(arr ArrayHandle) error { return errNoCGo }
func DestroyArray(arr ArrayHandle)                {}

func NewTape() (*Tape, error)                                         { return nil, errNoCGo }
func (t *Tape) Destroy()                                              {}
func (t *Tape) AddParameter(dtype DType, shape []int) (NodeID, error) { return -1, errNoCGo }
func (t *Tape) AddConstant(data unsafe.Pointer, nbytes int64, dtype DType, shape []int) (NodeID, error) {
	return -1, errNoCGo
}
func (t *Tape) AddUnary(op UnaryOp, input NodeID) (NodeID, error)             { return -1, errNoCGo }
func (t *Tape) AddBinary(op BinaryOp, lhs, rhs NodeID) (NodeID, error)        { return -1, errNoCGo }
func (t *Tape) AddConvertDType(input NodeID, targetDType DType) (NodeID, error) { return -1, errNoCGo }
func (t *Tape) AddBitcast(input NodeID, targetDType DType, outShape []int) (NodeID, error) {
	return -1, errNoCGo
}
func (t *Tape) AddIota(dtype DType, shape []int, iotaAxis int) (NodeID, error) { return -1, errNoCGo }
func (t *Tape) Compile(outputIDs []NodeID, compileGraph, eagerContiguous bool) (*Exec, error) {
	return nil, errNoCGo
}
func (e *Exec) Destroy() {}
func (e *Exec) Execute(inputs []ArrayHandle, donate []bool, numOutputs int) ([]ArrayHandle, error) {
	return nil, errNoCGo
}
