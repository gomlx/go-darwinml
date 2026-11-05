// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mps

import (
	"fmt"
	"reflect"
	"runtime"
	"unsafe"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/go-darwinml/internal/bridge"
	"github.com/pkg/errors"
)

// gpuBuffer holds tensor data in an MTLBuffer with MTLResourceStorageModeShared
// (Apple Silicon unified memory). Both CPU (via a typed Go slice wrapping
// [mtlBuffer contents]) and GPU (via MPSGraphTensorData) access the same memory
// directly with zero copy.
type gpuBuffer struct {
	backend *Backend
	shape   shapes.Shape
	mtlBuf  *bridge.Buffer
	flat    any // typed slice ([]float32, []int32, etc.) backed by mtlBuf.Contents()
	cleanup runtime.Cleanup
	valid   bool
}

// Verify interface compliance.
var _ compute.Buffer = (*gpuBuffer)(nil)

// makeTypedSlice creates a typed Go slice backed directly by ptr with length size.
// If size == 0, ptr may be nil and an empty typed slice is returned.
func makeTypedSlice(dt dtypes.DType, ptr unsafe.Pointer, size int) any {
	switch dt {
	case dtypes.Bool:
		if size == 0 {
			return []bool{}
		}
		return unsafe.Slice((*bool)(ptr), size)
	case dtypes.Int8:
		if size == 0 {
			return []int8{}
		}
		return unsafe.Slice((*int8)(ptr), size)
	case dtypes.Int16:
		if size == 0 {
			return []int16{}
		}
		return unsafe.Slice((*int16)(ptr), size)
	case dtypes.Int32:
		if size == 0 {
			return []int32{}
		}
		return unsafe.Slice((*int32)(ptr), size)
	case dtypes.Int64:
		if size == 0 {
			return []int64{}
		}
		return unsafe.Slice((*int64)(ptr), size)
	case dtypes.Uint8:
		if size == 0 {
			return []uint8{}
		}
		return unsafe.Slice((*uint8)(ptr), size)
	case dtypes.Uint16:
		if size == 0 {
			return []uint16{}
		}
		return unsafe.Slice((*uint16)(ptr), size)
	case dtypes.Uint32:
		if size == 0 {
			return []uint32{}
		}
		return unsafe.Slice((*uint32)(ptr), size)
	case dtypes.Uint64:
		if size == 0 {
			return []uint64{}
		}
		return unsafe.Slice((*uint64)(ptr), size)
	case dtypes.Float16:
		if size == 0 {
			return []float16.Float16{}
		}
		return unsafe.Slice((*float16.Float16)(ptr), size)
	case dtypes.BFloat16:
		if size == 0 {
			return []bfloat16.BFloat16{}
		}
		return unsafe.Slice((*bfloat16.BFloat16)(ptr), size)
	case dtypes.Float32:
		if size == 0 {
			return []float32{}
		}
		return unsafe.Slice((*float32)(ptr), size)
	case dtypes.Float64:
		if size == 0 {
			return []float64{}
		}
		return unsafe.Slice((*float64)(ptr), size)
	default:
		goType := dt.GoType()
		return reflect.MakeSlice(reflect.SliceOf(goType), size, size).Interface()
	}
}

// newBuffer allocates a new MTLBuffer-backed shared buffer for the given shape.
func (b *Backend) newBuffer(shape shapes.Shape) *gpuBuffer {
	size := shape.Size()
	nbytes := int64(size) * int64(shape.DType.Size())

	var mtlBuf *bridge.Buffer
	if nbytes > 0 && b != nil && b.ctx != nil {
		mtlBuf, _ = b.ctx.NewBuffer(nbytes)
	}
	return b.newBufferWithMTL(shape, mtlBuf)
}

// newBufferWithMTL wraps an existing (or nil) bridge.Buffer into a gpuBuffer for the given shape.
func (b *Backend) newBufferWithMTL(shape shapes.Shape, mtlBuf *bridge.Buffer) *gpuBuffer {
	size := shape.Size()
	var ptr unsafe.Pointer
	if mtlBuf != nil {
		ptr = mtlBuf.Contents()
	}

	var flat any
	if ptr != nil || size == 0 {
		flat = makeTypedSlice(shape.DType, ptr, size)
	} else {
		goType := shape.DType.GoType()
		flat = reflect.MakeSlice(reflect.SliceOf(goType), size, size).Interface()
	}

	buf := &gpuBuffer{
		backend: b,
		shape:   shape.Clone(),
		mtlBuf:  mtlBuf,
		flat:    flat,
		valid:   true,
	}
	if mtlBuf != nil {
		buf.cleanup = runtime.AddCleanup(buf, func(m *bridge.Buffer) {
			m.Destroy()
		}, mtlBuf)
	}
	return buf
}

// Backend returns the compute.Backend that owns this buffer.
func (buf *gpuBuffer) Backend() compute.Backend {
	return buf.backend
}

// Finalize releases the buffer's resources immediately and cancels its GC cleanup.
func (buf *gpuBuffer) Finalize() error {
	if buf == nil {
		return nil
	}
	buf.valid = false
	buf.flat = nil
	buf.shape = shapes.Invalid()
	if buf.mtlBuf != nil {
		buf.cleanup.Stop()
		buf.mtlBuf.Destroy()
		buf.mtlBuf = nil
	}
	return nil
}

// Shape returns the shape of the buffer.
func (buf *gpuBuffer) Shape() (shapes.Shape, error) {
	if buf == nil || !buf.valid {
		return shapes.Invalid(), errors.New("Shape: buffer has been finalized")
	}
	return buf.shape.Clone(), nil
}

// DeviceNum returns the device number (always 0 for single device).
func (buf *gpuBuffer) DeviceNum() (compute.DeviceNum, error) {
	if buf == nil || !buf.valid {
		return 0, errors.New("DeviceNum: buffer has been finalized")
	}
	return 0, nil
}

// ToFlatData copies the buffer contents into the given Go flat slice.
func (buf *gpuBuffer) ToFlatData(flat any) error {
	if buf == nil || !buf.valid {
		return errors.New("ToFlatData: buffer has been finalized")
	}
	return copyFlat(flat, buf.flat)
}

// Data returns direct access to the buffer's flat data slice in unified memory.
func (buf *gpuBuffer) Data() (flat any, err error) {
	if buf == nil || !buf.valid {
		return nil, errors.New("Data: buffer has been finalized")
	}
	if buf.backend != nil && buf.backend.IsFinalized() {
		return nil, errors.New("Data: backend is already finalized")
	}
	return buf.flat, nil
}

// CopyToDevice copies the buffer to another device (not supported on single-device MPS).
func (buf *gpuBuffer) CopyToDevice(deviceNum compute.DeviceNum) (compute.Buffer, error) {
	return nil, errors.Errorf("CopyToDevice: backend %q only supports a single device", BackendName)
}

// flatDataPtr returns a raw pointer to the flat data and its byte size.
func (buf *gpuBuffer) flatDataPtr() (unsafe.Pointer, int64) {
	if !buf.valid || buf.flat == nil {
		return nil, 0
	}
	if buf.mtlBuf != nil {
		return buf.mtlBuf.Contents(), buf.mtlBuf.Length()
	}
	v := reflect.ValueOf(buf.flat)
	if v.Len() == 0 {
		return nil, 0
	}
	elemSize := int64(buf.shape.DType.Size())
	nbytes := int64(v.Len()) * elemSize
	return unsafe.Pointer(v.Pointer()), nbytes
}

// bufferFromFlat creates a buffer from a Go flat slice, copying the data into shared Metal memory.
func (b *Backend) bufferFromFlat(flat any, shape shapes.Shape) (*gpuBuffer, error) {
	if err := checkDTypeSupported(shape.DType); err != nil {
		return nil, err
	}
	buf := b.newBuffer(shape)
	if err := copyFlat(buf.flat, flat); err != nil {
		_ = buf.Finalize()
		return nil, errors.Wrapf(err, "bufferFromFlat")
	}
	return buf, nil
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

// checkDTypeSupported returns compute.ErrNotImplemented if dt is not supported by the MPS backend.
func checkDTypeSupported(dt dtypes.DType) error {
	if !backendCapabilities.DTypes[dt] {
		return errors.Wrapf(compute.ErrNotImplemented, "dtype %s not supported in %q backend", dt, BackendName)
	}
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
