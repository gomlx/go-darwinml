// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mlx

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/go-darwinml/internal/mlxbridge"
	"github.com/pkg/errors"
)

// arrayHandle wraps a C++ mlx::core::array* pointer in an atomic cell so that
// explicit Finalize(), donation in Execute(), and GC runtime.AddCleanup can
// never double-free the underlying C++ object.
type arrayHandle struct {
	ptr atomic.Uintptr // mlxbridge.ArrayHandle
}

func newArrayHandle(h mlxbridge.ArrayHandle) *arrayHandle {
	cell := &arrayHandle{}
	cell.ptr.Store(uintptr(h))
	return cell
}

func (h *arrayHandle) load() mlxbridge.ArrayHandle {
	if h == nil {
		return 0
	}
	return mlxbridge.ArrayHandle(h.ptr.Load())
}

// take detaches and returns the underlying handle without calling DestroyArray,
// used when donating the handle directly into mlx_exec_execute.
func (h *arrayHandle) take() mlxbridge.ArrayHandle {
	if h == nil {
		return 0
	}
	return mlxbridge.ArrayHandle(h.ptr.Swap(0))
}

func (h *arrayHandle) destroy() {
	if h == nil {
		return
	}
	if raw := h.ptr.Swap(0); raw != 0 {
		mlxbridge.DestroyArray(mlxbridge.ArrayHandle(raw))
	}
}

// Buffer holds tensor data in an mlx::core::array backed by Apple Silicon unified memory.
type Buffer struct {
	backend *Backend
	shape   shapes.Shape
	handle  *arrayHandle
	cleanup runtime.Cleanup

	mu    sync.Mutex
	flat  any // lazily constructed typed Go slice backed by unified memory
	valid bool
}

// Verify interface compliance.
var _ compute.Buffer = (*Buffer)(nil)

// newBuffer allocates a new uninitialized unified-memory MLX array for shape.
func (b *Backend) newBuffer(shape shapes.Shape) (*Buffer, error) {
	if err := b.checkDTypeSupported(shape.DType); err != nil {
		return nil, err
	}
	nbytes := int64(shape.ByteSize())
	bridgeDType := dtypeToBridgeDType(shape.DType)
	arr, err := mlxbridge.NewArray(nbytes, bridgeDType, shape.Dimensions)
	if err != nil {
		return nil, errors.Wrap(err, "newBuffer")
	}
	return b.newBufferFromHandle(shape, arr), nil
}

// newBufferFromHandle wraps an existing mlxbridge.ArrayHandle into a Buffer with GC cleanup.
// Note: flat slice construction is deferred lazily until Data() or ToFlatData() is called.
func (b *Backend) newBufferFromHandle(shape shapes.Shape, arr mlxbridge.ArrayHandle) *Buffer {
	cell := newArrayHandle(arr)
	buf := &Buffer{
		backend: b,
		shape:   shape.Clone(),
		handle:  cell,
		valid:   true,
	}
	if arr != 0 {
		buf.cleanup = runtime.AddCleanup(buf, (*arrayHandle).destroy, cell)
	}
	return buf
}

// Backend returns the compute.Backend that owns this buffer.
func (buf *Buffer) Backend() compute.Backend {
	return buf.backend
}

// Finalize releases the buffer's C++ MLX array immediately and cancels its GC cleanup.
// Safe to call multiple times or after donation.
func (buf *Buffer) Finalize() error {
	if buf == nil {
		return nil
	}
	buf.mu.Lock()
	defer buf.mu.Unlock()

	buf.valid = false
	buf.flat = nil
	buf.shape = shapes.Invalid()
	if buf.handle != nil {
		buf.cleanup.Stop()
		buf.handle.destroy()
	}
	return nil
}

// takeForDonation detaches the underlying C++ MLX array handle for in-place donation
// during Execute() and marks the Go Buffer finalized so GC cleanup won't touch it.
func (buf *Buffer) takeForDonation() mlxbridge.ArrayHandle {
	buf.mu.Lock()
	defer buf.mu.Unlock()

	buf.valid = false
	buf.flat = nil
	buf.shape = shapes.Invalid()
	if buf.handle == nil {
		return 0
	}
	buf.cleanup.Stop()
	return buf.handle.take()
}

// Shape returns the shape of the buffer.
func (buf *Buffer) Shape() (shapes.Shape, error) {
	if buf == nil {
		return shapes.Invalid(), errors.New("Shape: buffer is nil")
	}
	buf.mu.Lock()
	defer buf.mu.Unlock()
	if !buf.valid {
		return shapes.Invalid(), errors.New("Shape: buffer has been finalized")
	}
	return buf.shape.Clone(), nil
}

// DeviceNum returns the device number (always 0 for single device).
func (buf *Buffer) DeviceNum() (compute.DeviceNum, error) {
	if buf == nil {
		return 0, errors.New("DeviceNum: buffer is nil")
	}
	buf.mu.Lock()
	defer buf.mu.Unlock()
	if !buf.valid {
		return 0, errors.New("DeviceNum: buffer has been finalized")
	}
	return 0, nil
}

// ensureFlatLocked ensures the underlying MLX array is row-contiguous and evaluated,
// and lazily constructs buf.flat backed by unified memory. Caller must hold buf.mu.
func (buf *Buffer) ensureFlatLocked() error {
	if !buf.valid {
		return errors.New("buffer has been finalized")
	}
	if buf.backend != nil && buf.backend.IsFinalized() {
		return errors.New("backend is already finalized")
	}
	if buf.flat != nil {
		return nil
	}

	size := buf.shape.Size()
	if size == 0 {
		buf.flat = dtypes.MakeAnySlice(buf.shape.DType, 0)
		return nil
	}

	arr := buf.handle.load()
	if arr == 0 {
		return errors.New("buffer has nil MLX array handle")
	}

	// Lazily materialize a contiguous array if this buffer was produced as a non-contiguous view.
	if !mlxbridge.ArrayIsRowContiguous(arr) {
		if err := mlxbridge.ArrayEnsureContiguous(arr); err != nil {
			return errors.Wrap(err, "materializing contiguous buffer")
		}
	}

	ptr := mlxbridge.ArrayDataPtr(arr)
	if ptr == nil {
		return errors.New("buffer unified memory pointer is nil")
	}
	buf.flat = dtypes.UnsafeAnySliceFromBytes(ptr, buf.shape.DType, size)
	return nil
}

// Data returns direct access to the buffer's flat data slice in unified memory.
func (buf *Buffer) Data() (flat any, err error) {
	if buf == nil {
		return nil, errors.New("Data: buffer is nil")
	}
	buf.mu.Lock()
	defer buf.mu.Unlock()

	if err := buf.ensureFlatLocked(); err != nil {
		return nil, errors.Wrap(err, "Data")
	}
	return buf.flat, nil
}

// ToFlatData copies the buffer contents into the given Go flat slice.
func (buf *Buffer) ToFlatData(flat any) error {
	if buf == nil {
		return errors.New("ToFlatData: buffer is nil")
	}
	buf.mu.Lock()
	defer buf.mu.Unlock()

	if err := buf.ensureFlatLocked(); err != nil {
		return errors.Wrap(err, "ToFlatData")
	}
	return dtypes.SafeCopyAnySlice(flat, buf.flat)
}

// CopyToDevice copies the buffer to another device (not supported on single-device MLX).
func (buf *Buffer) CopyToDevice(deviceNum compute.DeviceNum) (compute.Buffer, error) {
	return nil, errors.Errorf("CopyToDevice: backend %q only supports a single device", BackendName)
}

// bufferFromFlat creates an MLX buffer initialized with the contents of flat.
func (b *Backend) bufferFromFlat(flat any, shape shapes.Shape) (*Buffer, error) {
	if err := b.checkDTypeSupported(shape.DType); err != nil {
		return nil, err
	}
	ptr, nbytes, err := flatSlicePtr(flat, shape)
	if err != nil {
		return nil, errors.Wrap(err, "bufferFromFlat")
	}
	arr, err := mlxbridge.NewArrayFromData(ptr, nbytes, dtypeToBridgeDType(shape.DType), shape.Dimensions)
	runtime.KeepAlive(flat)
	if err != nil {
		return nil, errors.Wrap(err, "bufferFromFlat")
	}
	return b.newBufferFromHandle(shape, arr), nil
}

// flatSlicePtr validates that flat is a slice matching shape and returns its raw pointer and byte length.
func flatSlicePtr(flat any, shape shapes.Shape) (unsafe.Pointer, int64, error) {
	rawBytes, err := dtypes.CheckAnySlice(flat, shape.DType, shape.Size())
	if err != nil {
		return nil, 0, err
	}
	if len(rawBytes) == 0 {
		return nil, 0, nil
	}
	return unsafe.Pointer(&rawBytes[0]), int64(len(rawBytes)), nil
}

// checkDTypeSupported returns compute.ErrNotImplemented if dt is not supported by the MLX backend.
func (b *Backend) checkDTypeSupported(dt dtypes.DType) error {
	if !b.capabilities.DTypes[dt] {
		return errors.Wrapf(compute.ErrNotImplemented, "dtype %s not supported in %q backend", dt, BackendName)
	}
	return nil
}

// dtypeToBridgeDType converts a GoMLX dtype to the mlxbridge DType constant.
func dtypeToBridgeDType(dt dtypes.DType) mlxbridge.DType {
	switch dt {
	case dtypes.Bool:
		return mlxbridge.DTypeBool
	case dtypes.Int8:
		return mlxbridge.DTypeInt8
	case dtypes.Int16:
		return mlxbridge.DTypeInt16
	case dtypes.Int32:
		return mlxbridge.DTypeInt32
	case dtypes.Int64:
		return mlxbridge.DTypeInt64
	case dtypes.Float16:
		return mlxbridge.DTypeFloat16
	case dtypes.BFloat16:
		return mlxbridge.DTypeBF16
	case dtypes.Float32:
		return mlxbridge.DTypeFloat32
	case dtypes.Float64:
		return mlxbridge.DTypeFloat64
	case dtypes.Uint8:
		return mlxbridge.DTypeUint8
	case dtypes.Uint16:
		return mlxbridge.DTypeUint16
	case dtypes.Uint32:
		return mlxbridge.DTypeUint32
	case dtypes.Uint64:
		return mlxbridge.DTypeUint64
	case dtypes.Complex64:
		return mlxbridge.DTypeComplex64
	default:
		panic(fmt.Sprintf("dtypeToBridgeDType: unsupported dtype %v", dt))
	}
}
