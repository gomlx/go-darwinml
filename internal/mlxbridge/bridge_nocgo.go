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

type ArrayHandle uintptr
type NodeID = int32

type Tape struct{}
type Exec struct{}

var errNoCGo = fmt.Errorf("mlx: CGO is disabled; rebuild with CGO_ENABLED=1")

func Init() error                                                                    { return errNoCGo }
func DeviceName() (string, error)                                                    { return "", errNoCGo }
func Version() (string, error)                                                       { return "", errNoCGo }
func NewArray(nbytes int64, dtype DType, shape []int) (ArrayHandle, error)          { return 0, errNoCGo }
func NewArrayFromData(data unsafe.Pointer, nbytes int64, dtype DType, shape []int) (ArrayHandle, error) {
	return 0, errNoCGo
}
func ArrayDataPtr(arr ArrayHandle) unsafe.Pointer        { return nil }
func ArrayNBytes(arr ArrayHandle) int64                  { return 0 }
func ArrayDType(arr ArrayHandle) DType                   { return DTypeFloat32 }
func ArrayShape(arr ArrayHandle) []int                   { return nil }
func ArrayIsRowContiguous(arr ArrayHandle) bool          { return false }
func ArrayEnsureContiguous(arr ArrayHandle) error        { return errNoCGo }
func DestroyArray(arr ArrayHandle)                       {}

func NewTape() (*Tape, error)                                                        { return nil, errNoCGo }
func (t *Tape) Destroy()                                                             {}
func (t *Tape) AddParameter(dtype DType, shape []int) (NodeID, error)               { return -1, errNoCGo }
func (t *Tape) AddConstant(data unsafe.Pointer, nbytes int64, dtype DType, shape []int) (NodeID, error) {
	return -1, errNoCGo
}
func (t *Tape) Compile(outputIDs []NodeID, compileGraph, eagerContiguous bool) (*Exec, error) {
	return nil, errNoCGo
}
func (e *Exec) Destroy() {}
func (e *Exec) Execute(inputs []ArrayHandle, donate []bool, numOutputs int) ([]ArrayHandle, error) {
	return nil, errNoCGo
}
