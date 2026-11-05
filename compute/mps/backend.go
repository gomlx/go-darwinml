// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

// Package mps implements a GoMLX compute backend using Apple's MPSGraph framework
// for GPU-accelerated tensor computation on Apple Silicon.
package mps

import (
	"sync"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/go-darwinml/internal/bridge"
	"github.com/pkg/errors"
)

// BackendName is the name used to register this backend.
const BackendName = "mps"

// maxIdlePoolSteps is the maximum number of backend Execute calls an Executable's
// ping-pong pool can go unused before its remaining un-stolen buffers are evicted.
const maxIdlePoolSteps = 32

// Backend implements the compute.Backend interface using Apple's MPSGraph.
type Backend struct {
	ctx         *bridge.Context
	deviceName  string
	mu          sync.RWMutex
	isFinalized bool

	poolMu   sync.Mutex
	pools    map[*pingPongPool]struct{}
	execStep uint64
}

// Verify interface compliance.
var _ compute.Backend = &Backend{}

// New creates a new MPS backend. config is currently unused.
func New(config string) (compute.Backend, error) {
	ctx, err := bridge.NewContext()
	if err != nil {
		return nil, errors.Wrap(err, "creating MPS backend")
	}
	b := &Backend{
		ctx:        ctx,
		deviceName: ctx.DeviceName(),
		pools:      make(map[*pingPongPool]struct{}),
	}
	return b, nil
}

// Name returns the backend name.
func (b *Backend) Name() string { return BackendName }

// String returns the backend name.
func (b *Backend) String() string { return b.Name() }

// Description returns a human-readable description.
func (b *Backend) Description() string {
	return "MPSGraph GPU backend (" + b.deviceName + ")"
}

// NumDevices returns 1 (single GPU on Apple Silicon).
func (b *Backend) NumDevices() int { return 1 }

// DeviceDescription returns the Metal device name.
func (b *Backend) DeviceDescription(deviceNum compute.DeviceNum) string {
	return b.deviceName
}

// Capabilities returns the set of supported ops and dtypes.
func (b *Backend) Capabilities() compute.Capabilities {
	return backendCapabilities
}

// Builder creates a new computation builder, reusing the backend's Metal device.
func (b *Backend) Builder(name string) compute.Builder {
	ctx, err := bridge.NewContextWithDevice(b.ctx.DeviceHandle())
	if err != nil {
		// Graph building functions panic on error per GoMLX convention.
		panic(errors.Wrapf(err, "creating MPS builder %q", name))
	}
	return newBuilder(b, name, ctx)
}

// Finalize releases all backend resources.
func (b *Backend) Finalize() {
	b.mu.Lock()
	if b.isFinalized {
		b.mu.Unlock()
		return
	}
	b.isFinalized = true
	b.mu.Unlock()

	b.poolMu.Lock()
	for pool := range b.pools {
		pool.destroyAllLocked()
		delete(b.pools, pool)
	}
	b.poolMu.Unlock()

	b.mu.Lock()
	if b.ctx != nil {
		b.ctx.Destroy()
		b.ctx = nil
	}
	b.mu.Unlock()
}

// registerPool registers an Executable's ping-pong pool with the Backend so other
// executables can steal matching buffers from it during multi-graph / multi-shape execution.
func (b *Backend) registerPool(pool *pingPongPool) {
	if pool == nil {
		return
	}
	b.poolMu.Lock()
	defer b.poolMu.Unlock()
	if !b.isFinalized {
		pool.lastUsedStep = b.execStep
		b.pools[pool] = struct{}{}
	}
}

// unregisterAndDestroyPool removes an Executable's pool from the Backend and destroys
// any buffers remaining in it.
func (b *Backend) unregisterAndDestroyPool(pool *pingPongPool) {
	if pool == nil {
		return
	}
	b.poolMu.Lock()
	delete(b.pools, pool)
	pool.destroyAllLocked()
	b.poolMu.Unlock()
}

// acquireOutputBuffers allocates output gpuBuffers for an execution using a mixed-scope strategy:
//  1. Prefer buffers from localPool (the executing Executable's own pool).
//  2. Fallback to stealing matching-size buffers from other Executables' pools on this Backend.
//  3. Allocate a new MTLBuffer if no pooled buffer of that byte size is available.
//
// After output buffers are acquired, any unused buffers remaining in localPool are immediately
// destroyed (Condition 3), and any other Executable's pool that has been idle for more than
// maxIdlePoolSteps backend executions also has its remaining buffers evicted.
func (b *Backend) acquireOutputBuffers(localPool *pingPongPool, outputShapes []shapes.Shape) []*gpuBuffer {
	poppedMTL := make([]*bridge.Buffer, len(outputShapes))

	b.poolMu.Lock()
	b.execStep++
	currentStep := b.execStep
	if localPool != nil {
		localPool.lastUsedStep = currentStep
	}

	for i, outShape := range outputShapes {
		nbytes := outShape.ByteSize()
		if nbytes <= 0 {
			continue
		}
		// 1. Check localPool first.
		if localPool != nil {
			if list := localPool.buffers[nbytes]; len(list) > 0 {
				poppedMTL[i] = list[len(list)-1]
				localPool.buffers[nbytes] = list[:len(list)-1]
				continue
			}
		}
		// 2. Fallback: steal from another Executable's pool on the same Backend.
		for otherPool := range b.pools {
			if otherPool == localPool {
				continue
			}
			if list := otherPool.buffers[nbytes]; len(list) > 0 {
				poppedMTL[i] = list[len(list)-1]
				otherPool.buffers[nbytes] = list[:len(list)-1]
				break
			}
		}
	}

	// Condition 3: Evict any buffers in localPool that were not used in this execution.
	if localPool != nil {
		localPool.destroyAllLocked()
	}

	// Evict buffers from other Executables' pools that have been idle for too long
	// (e.g. rare shape buckets).
	for otherPool := range b.pools {
		if otherPool != localPool && currentStep-otherPool.lastUsedStep > maxIdlePoolSteps {
			otherPool.destroyAllLocked()
		}
	}
	b.poolMu.Unlock()

	outputBuffers := make([]*gpuBuffer, len(outputShapes))
	for i, outShape := range outputShapes {
		if poppedMTL[i] != nil {
			outputBuffers[i] = b.newBufferWithMTL(outShape, poppedMTL[i])
		} else {
			outputBuffers[i] = b.newBuffer(outShape)
		}
	}
	return outputBuffers
}

// returnDonatedBuffers stores donated input MTLBuffers into localPool, unless the backend is finalized.
func (b *Backend) returnDonatedBuffers(localPool *pingPongPool, donatedBySize map[int64][]*bridge.Buffer) {
	if len(donatedBySize) == 0 {
		return
	}
	b.poolMu.Lock()
	defer b.poolMu.Unlock()

	if b.isFinalized || localPool == nil {
		for _, list := range donatedBySize {
			for _, buf := range list {
				buf.Destroy()
			}
		}
		return
	}
	for nbytes, list := range donatedBySize {
		localPool.buffers[nbytes] = append(localPool.buffers[nbytes], list...)
	}
}

// IsFinalized returns whether the backend has been finalized.
func (b *Backend) IsFinalized() bool {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.isFinalized
}

// --- DataInterface implementation ---

// BufferFromFlatData creates a buffer from a Go flat slice.
func (b *Backend) BufferFromFlatData(deviceNum compute.DeviceNum, flat any, shape shapes.Shape) (compute.Buffer, error) {
	if b.IsFinalized() {
		return nil, errors.New("BufferFromFlatData: backend is already finalized")
	}
	if deviceNum != 0 {
		return nil, errors.Errorf("BufferFromFlatData: backend %q only supports deviceNum 0, got %d", BackendName, deviceNum)
	}
	return b.bufferFromFlat(flat, shape)
}

// HasSharedBuffers returns true since we use Go-managed memory for buffers.
func (b *Backend) HasSharedBuffers() bool {
	return true
}

// NewSharedBuffer creates a new buffer with direct access to its memory.
func (b *Backend) NewSharedBuffer(deviceNum compute.DeviceNum, shape shapes.Shape) (buffer compute.Buffer, flat any, err error) {
	if b.IsFinalized() {
		return nil, nil, errors.New("NewSharedBuffer: backend is already finalized")
	}
	if deviceNum != 0 {
		return nil, nil, errors.Errorf("NewSharedBuffer: backend %q only supports deviceNum 0, got %d", BackendName, deviceNum)
	}
	if err := checkDTypeSupported(shape.DType); err != nil {
		return nil, nil, err
	}
	buf := b.newBuffer(shape)
	return buf, buf.flat, nil
}
