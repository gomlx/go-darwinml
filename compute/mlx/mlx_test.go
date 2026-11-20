// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mlx_test

import (
	"runtime"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/backendtest"
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

func TestDonatedBufferExecution(t *testing.T) {
	backend, err := mlx.New("")
	if err != nil {
		t.Fatalf("mlx.New failed: %v", err)
	}
	defer backend.Finalize()

	shape := shapes.Make(dtypes.Float32, 4)
	builder := backend.Builder("test_donate")
	mainFn := builder.Main()
	x, _ := mainFn.Parameter("x", shape, nil)
	one, _ := mainFn.Constant([]float32{1, 2, 3, 4}, 4)
	y, _ := mainFn.Add(x, one)
	_ = mainFn.Return([]compute.Value{y}, nil)
	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile failed: %v", err)
	}
	defer exec.Finalize()

	curBuf, err := backend.BufferFromFlatData(0, []float32{10, 20, 30, 40}, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData failed: %v", err)
	}
	for step := range 3 {
		prevBuf := curBuf
		outs, err := exec.Execute([]compute.Buffer{curBuf}, []bool{true}, 0)
		if err != nil {
			t.Fatalf("step %d Execute failed: %v", step, err)
		}
		// Donated buffer should be finalized and attempting ToFlatData on it should fail.
		tmp := make([]float32, 4)
		if err := prevBuf.ToFlatData(tmp); err == nil {
			t.Fatalf("step %d: expected ToFlatData on donated buffer to fail", step)
		}
		curBuf = outs[0]
	}
	defer curBuf.Finalize()

	got := make([]float32, 4)
	if err := curBuf.ToFlatData(got); err != nil {
		t.Fatalf("final ToFlatData failed: %v", err)
	}
	want := []float32{13, 26, 39, 52}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestSortCustomComparatorNotImplemented(t *testing.T) {
	backend, err := mlx.New("")
	if err != nil {
		t.Fatalf("mlx.New failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_custom_sort")
	mainFn := builder.Main()
	x, _ := mainFn.Parameter("x", shapes.Make(dtypes.Float32, 4), nil)
	comp, err := mainFn.Closure()
	if err != nil {
		t.Fatalf("Closure failed: %v", err)
	}
	lhs, _ := comp.Parameter("lhs", shapes.Make(dtypes.Float32), nil)
	rhs, _ := comp.Parameter("rhs", shapes.Make(dtypes.Float32), nil)
	absL, _ := comp.Abs(lhs)
	absR, _ := comp.Abs(rhs)
	lt, _ := comp.LessThan(absL, absR)
	_ = comp.Return([]compute.Value{lt}, nil)

	_, err = mainFn.Sort(comp, 0, false, x)
	if !compute.IsNotImplemented(err) {
		t.Fatalf("expected compute.ErrNotImplemented for custom Sort comparator, got: %v", err)
	}
}

func TestScatterDynamicShapes(t *testing.T) {
	backend, err := mlx.New("dynamic")
	if err != nil {
		t.Fatalf("mlx.New(\"dynamic\") failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("scatter_dyn_test")
	fn := builder.Main()

	opShape := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, shapes.DynamicDim}, []string{"rows", "cols"})
	idxShape := shapes.MakeDynamic(dtypes.Int32, []int{shapes.DynamicDim, shapes.DynamicDim, 1}, []string{shapes.UniqueAxisName("i0"), shapes.UniqueAxisName("i1"), ""})
	updShape := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, shapes.DynamicDim, shapes.DynamicDim}, []string{shapes.UniqueAxisName("u0"), shapes.UniqueAxisName("u1"), "cols"})

	op, err := fn.Parameter("op", opShape, nil)
	if err != nil {
		t.Fatalf("Parameter op: %v", err)
	}
	idx, err := fn.Parameter("idx", idxShape, nil)
	if err != nil {
		t.Fatalf("Parameter idx: %v", err)
	}
	upd, err := fn.Parameter("upd", updShape, nil)
	if err != nil {
		t.Fatalf("Parameter upd: %v", err)
	}
	out, err := fn.ScatterSum(op, idx, upd, 2, []int{2}, []int{0}, []int{0}, false, false)
	if err != nil {
		t.Fatalf("ScatterSum: %v", err)
	}
	if err := fn.Return([]compute.Value{out}, nil); err != nil {
		t.Fatalf("Return: %v", err)
	}
	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	defer exec.Finalize()

	// Valid empty batch (0, 2)
	opBuf, _ := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6}, shapes.Make(dtypes.Float32, 2, 3))
	idxBuf, _ := backend.BufferFromFlatData(0, []int32{}, shapes.Make(dtypes.Int32, 0, 2, 1))
	updBuf, _ := backend.BufferFromFlatData(0, []float32{}, shapes.Make(dtypes.Float32, 0, 2, 3))
	outs, err := exec.Execute([]compute.Buffer{opBuf, idxBuf, updBuf}, []bool{true, true, true}, 0)
	if err != nil {
		t.Fatalf("Execute empty batch failed: %v", err)
	}
	got := make([]float32, 6)
	if err := outs[0].ToFlatData(got); err != nil {
		t.Fatalf("ToFlatData failed: %v", err)
	}
	_ = outs[0].Finalize()
	for i, want := range []float32{1, 2, 3, 4, 5, 6} {
		if got[i] != want {
			t.Fatalf("empty batch got[%d]=%v, want %v", i, got[i], want)
		}
	}

	// Swapped batch dims (2, 3) vs (3, 2) have identical element count and must still error.
	opBuf2, _ := backend.BufferFromFlatData(0, make([]float32, 12), shapes.Make(dtypes.Float32, 4, 3))
	idxBuf2, _ := backend.BufferFromFlatData(0, make([]int32, 6), shapes.Make(dtypes.Int32, 2, 3, 1))
	updBuf2, _ := backend.BufferFromFlatData(0, make([]float32, 18), shapes.Make(dtypes.Float32, 3, 2, 3))
	outs2, err := exec.Execute([]compute.Buffer{opBuf2, idxBuf2, updBuf2}, []bool{true, true, true}, 0)
	for _, o := range outs2 {
		if o != nil {
			_ = o.Finalize()
		}
	}
	if err == nil {
		t.Fatalf("expected runtime error for swapped batch dims (2, 3) vs (3, 2), got nil")
	}
}

func TestCompliance(t *testing.T) {
	for _, cfg := range []string{"mlx", "mlx:dynamic"} {
		t.Run(cfg, func(t *testing.T) {
			b, err := compute.NewWithConfig(cfg)
			if err != nil {
				t.Fatalf("compute.NewWithConfig(%q) failed: %v", cfg, err)
			}
			defer b.Finalize()

			backendtest.RunAll(t, b, nil)
		})
	}
}

func TestFinalize(t *testing.T) {
	for _, cfg := range []string{"", "dynamic"} {
		name := cfg
		if name == "" {
			name = "static"
		}
		t.Run(name, func(t *testing.T) {
			backendtest.TestFinalize(t, func() (compute.Backend, error) {
				return mlx.New(cfg)
			})
		})
	}
}

// BenchmarkCompliance runs all compliance (backendtest) compute.Backend benchmarks on the "mlx" backend.
// To run:
//
//	$ go test -bench=. -benchmem ./compute/mlx
func BenchmarkCompliance(b *testing.B) {
	backend, err := mlx.New("")
	if err != nil {
		b.Fatalf("mlx.New(\"\") failed: %v", err)
	}
	defer backend.Finalize()

	backendtest.RunAllBenchmarks(b, backend)
}
