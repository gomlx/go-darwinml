// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mpsgraph

import (
	"fmt"
	"reflect"
	"unsafe"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

// gpuBuffer holds tensor data in Go-managed memory. During execution, data is
// copied to/from MPSGraphTensorData. Future optimization: use MTLBuffer-backed
// shared memory for zero-copy.
type gpuBuffer struct {
	shape shapes.Shape
	flat  any  // typed slice ([]float32, []int32, etc.)
	valid bool
}

// newBuffer allocates a new buffer for the given shape.
func newBuffer(shape shapes.Shape) *gpuBuffer {
	size := shape.Size()
	if size == 0 {
		size = 1 // Scalar
	}
	goType := shape.DType.GoType()
	flat := reflect.MakeSlice(reflect.SliceOf(goType), size, size).Interface()
	return &gpuBuffer{
		shape: shape,
		flat:  flat,
		valid: true,
	}
}

// flatDataPtr returns a raw pointer to the flat data and its byte size.
func (buf *gpuBuffer) flatDataPtr() (unsafe.Pointer, int64) {
	if !buf.valid || buf.flat == nil {
		return nil, 0
	}
	v := reflect.ValueOf(buf.flat)
	if v.Len() == 0 {
		return nil, 0
	}
	elemSize := int64(buf.shape.DType.Size())
	nbytes := int64(v.Len()) * elemSize
	return unsafe.Pointer(v.Pointer()), nbytes
}

// bufferFromFlat creates a buffer from a Go flat slice, copying the data.
func bufferFromFlat(flat any, shape shapes.Shape) (*gpuBuffer, error) {
	buf := newBuffer(shape)
	if err := copyFlat(buf.flat, flat); err != nil {
		return nil, errors.Wrapf(err, "bufferFromFlat")
	}
	return buf, nil
}

// bufferCopyToFlat copies buffer data to the given flat slice.
func bufferCopyToFlat(buf *gpuBuffer, flat any) error {
	if !buf.valid {
		return errors.New("bufferCopyToFlat: buffer has been finalized")
	}
	return copyFlat(flat, buf.flat)
}

// copyFlat copies data between two flat slices of the same type and length.
func copyFlat(dst, src any) error {
	dstVal := reflect.ValueOf(dst)
	srcVal := reflect.ValueOf(src)
	if dstVal.Kind() != reflect.Slice || srcVal.Kind() != reflect.Slice {
		return errors.Errorf("copyFlat: expected slices, got dst=%T src=%T", dst, src)
	}
	if dstVal.Type().Elem() != srcVal.Type().Elem() {
		return errors.Errorf("copyFlat: type mismatch dst=%v src=%v", dstVal.Type().Elem(), srcVal.Type().Elem())
	}
	if dstVal.Len() != srcVal.Len() {
		return errors.Errorf("copyFlat: length mismatch dst=%d src=%d", dstVal.Len(), srcVal.Len())
	}
	reflect.Copy(dstVal, srcVal)
	return nil
}

// dtypeToBridgeDType converts a GoMLX dtype to the bridge dtype constant.
func dtypeToBridgeDType(dt dtypes.DType) int {
	switch dt {
	case dtypes.Bool:
		return 0 // MPSGRAPH_DTYPE_BOOL
	case dtypes.Int8:
		return 1 // MPSGRAPH_DTYPE_INT8
	case dtypes.Int16:
		return 2 // MPSGRAPH_DTYPE_INT16
	case dtypes.Int32:
		return 3 // MPSGRAPH_DTYPE_INT32
	case dtypes.Int64:
		return 4 // MPSGRAPH_DTYPE_INT64
	case dtypes.Float16:
		return 5 // MPSGRAPH_DTYPE_FLOAT16
	case dtypes.BFloat16:
		return 6 // MPSGRAPH_DTYPE_BFLOAT16
	case dtypes.Float32:
		return 7 // MPSGRAPH_DTYPE_FLOAT32
	case dtypes.Uint8:
		return 9 // MPSGRAPH_DTYPE_UINT8
	case dtypes.Uint16:
		return 10 // MPSGRAPH_DTYPE_UINT16
	case dtypes.Uint32:
		return 11 // MPSGRAPH_DTYPE_UINT32
	case dtypes.Uint64:
		return 12 // MPSGRAPH_DTYPE_UINT64
	default:
		panic(fmt.Sprintf("dtypeToBridgeDType: unsupported dtype %v", dt))
	}
}
