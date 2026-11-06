// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mlx_test

import (
	"runtime"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/go-darwinml/compute/mlx"
)

func TestBackendRegistrationAndConfig(t *testing.T) {
	backend, err := compute.NewWithConfig("mlx")
	if err != nil {
		t.Fatalf("compute.NewWithConfig(\"mlx\") failed: %v", err)
	}
	defer backend.Finalize()

	if backend.Name() != mlx.BackendName {
		t.Fatalf("expected backend name %q, got %q", mlx.BackendName, backend.Name())
	}
	t.Logf("Backend description: %s", backend.Description())
	t.Logf("Device 0: %s", backend.DeviceDescription(0))

	if backend.Capabilities().DynamicShapes != compute.DynamicShapesNone {
		t.Fatalf("expected default mlx DynamicShapesNone, got %v", backend.Capabilities().DynamicShapes)
	}

	// Test dynamic config.
	dynBackend, err := compute.NewWithConfig("mlx:dynamic,lazy_contiguous")
	if err != nil {
		t.Fatalf("compute.NewWithConfig(\"mlx:dynamic,lazy_contiguous\") failed: %v", err)
	}
	defer dynBackend.Finalize()
	if dynBackend.Capabilities().DynamicShapes != compute.DynamicShapesNative {
		t.Fatalf("expected mlx:dynamic DynamicShapesNative, got %v", dynBackend.Capabilities().DynamicShapes)
	}

	// Test invalid lazy_contiguous without dynamic.
	if _, err := mlx.New("lazy_contiguous"); err == nil {
		t.Fatalf("expected error when enabling lazy_contiguous without dynamic mode")
	}
}

func TestSharedBufferAndLifecycle(t *testing.T) {
	backend, err := mlx.New("")
	if err != nil {
		t.Fatalf("mlx.New failed: %v", err)
	}
	defer backend.Finalize()

	shape := shapes.Make(dtypes.Float32, 2, 3)
	buf, flat, err := backend.NewSharedBuffer(0, shape)
	if err != nil {
		t.Fatalf("NewSharedBuffer failed: %v", err)
	}

	data := flat.([]float32)
	for i := range data {
		data[i] = float32(i + 1)
	}

	dst := make([]float32, 6)
	if err := buf.ToFlatData(dst); err != nil {
		t.Fatalf("ToFlatData failed: %v", err)
	}
	for i, want := range data {
		if dst[i] != want {
			t.Fatalf("dst[%d] = %v, want %v", i, dst[i], want)
		}
	}

	// Test idempotent Finalize + GC cleanup.
	if err := buf.Finalize(); err != nil {
		t.Fatalf("Finalize failed: %v", err)
	}
	if err := buf.Finalize(); err != nil {
		t.Fatalf("second Finalize failed: %v", err)
	}
	runtime.GC()
}

func TestSimpleParameterAndConstantExecution(t *testing.T) {
	backend, err := mlx.New("")
	if err != nil {
		t.Fatalf("mlx.New failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_passthrough")

	mainFn := builder.Main()
	inShape := shapes.Make(dtypes.Float32, 4)
	x, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Parameter failed: %v", err)
	}
	c, err := mainFn.Constant([]float32{10, 20, 30, 40}, 4)
	if err != nil {
		t.Fatalf("Constant failed: %v", err)
	}
	if err := mainFn.Return([]compute.Value{x, c}, nil); err != nil {
		t.Fatalf("Return failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile failed: %v", err)
	}
	defer exec.Finalize()

	inBuf, err := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4}, inShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData failed: %v", err)
	}

	outs, err := exec.Execute([]compute.Buffer{inBuf}, []bool{true}, 0)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	defer func() {
		for _, o := range outs {
			_ = o.Finalize()
		}
	}()

	got0 := make([]float32, 4)
	if err := outs[0].ToFlatData(got0); err != nil {
		t.Fatalf("outs[0].ToFlatData failed: %v", err)
	}
	for i, want := range []float32{1, 2, 3, 4} {
		if got0[i] != want {
			t.Fatalf("got0[%d] = %v, want %v", i, got0[i], want)
		}
	}

	got1 := make([]float32, 4)
	if err := outs[1].ToFlatData(got1); err != nil {
		t.Fatalf("outs[1].ToFlatData failed: %v", err)
	}
	for i, want := range []float32{10, 20, 30, 40} {
		if got1[i] != want {
			t.Fatalf("got1[%d] = %v, want %v", i, got1[i], want)
		}
	}
}
