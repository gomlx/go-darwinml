// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mlx_test

import (
	"math"
	"runtime"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/backendtest"
	"github.com/gomlx/compute/support/testutil"
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

func TestComplianceStep2(t *testing.T) {
	for _, cfg := range []string{"", "dynamic", "dynamic,lazy_contiguous"} {
		name := cfg
		if name == "" {
			name = "compiled"
		}
		t.Run(name, func(t *testing.T) {
			b, err := mlx.New(cfg)
			if err != nil {
				t.Fatalf("mlx.New(%q) failed: %v", cfg, err)
			}
			defer b.Finalize()

			t.Run("BinaryOps", func(t *testing.T) { backendtest.TestBinaryOps(t, b) })
			t.Run("UnaryOps", func(t *testing.T) { backendtest.TestUnaryOps(t, b) })
			t.Run("ConvertDType", func(t *testing.T) { backendtest.TestConvertDType(t, b) })
			t.Run("Bitcast", func(t *testing.T) { backendtest.TestBitcast(t, b) })
			t.Run("Exec", func(t *testing.T) { backendtest.TestExec(t, b) })
			t.Run("Barriers", func(t *testing.T) { backendtest.TestBarriers(t, b) })
		})
	}
}

func TestStep2ExtraOps(t *testing.T) {
	b, err := mlx.New("")
	if err != nil {
		t.Fatalf("mlx.New failed: %v", err)
	}
	defer b.Finalize()

	t.Run("Iota", func(t *testing.T) {
		y0, err := testutil.Exec1(b, nil, func(f compute.Function, _ []compute.Value) (compute.Value, error) {
			return f.Iota(shapes.Make(dtypes.Int32, 2, 3), 1)
		})
		if err != nil {
			t.Fatalf("Iota axis 1 failed: %v", err)
		}
		if ok, diff := testutil.IsEqual([][]int32{{0, 1, 2}, {0, 1, 2}}, y0); !ok {
			t.Errorf("Iota axis 1 mismatch:\n%s", diff)
		}

		y1, err := testutil.Exec1(b, nil, func(f compute.Function, _ []compute.Value) (compute.Value, error) {
			return f.Iota(shapes.Make(dtypes.Float32, 3, 2), 0)
		})
		if err != nil {
			t.Fatalf("Iota axis 0 failed: %v", err)
		}
		if ok, diff := testutil.IsEqual([][]float32{{0, 0}, {1, 1}, {2, 2}}, y1); !ok {
			t.Errorf("Iota axis 0 mismatch:\n%s", diff)
		}
	})

	t.Run("RemNegative", func(t *testing.T) {
		gotInt, err := testutil.Exec1(b, []any{[]int32{-7, -9, 7, 9}, []int32{4, -4, -4, 4}},
			func(f compute.Function, p []compute.Value) (compute.Value, error) {
				return f.Rem(p[0], p[1])
			})
		if err != nil {
			t.Fatalf("Rem int32 failed: %v", err)
		}
		if ok, diff := testutil.IsEqual([]int32{-3, -1, 3, 1}, gotInt); !ok {
			t.Errorf("Rem int32 mismatch:\n%s", diff)
		}

		gotFloat, err := testutil.Exec1(b, []any{[]float32{-7.5, 7.5}, []float32{2.0, -2.0}},
			func(f compute.Function, p []compute.Value) (compute.Value, error) {
				return f.Rem(p[0], p[1])
			})
		if err != nil {
			t.Fatalf("Rem float32 failed: %v", err)
		}
		if ok, diff := testutil.IsEqual([]float32{-1.5, 1.5}, gotFloat); !ok {
			t.Errorf("Rem float32 mismatch:\n%s", diff)
		}
	})

	t.Run("Shifts", func(t *testing.T) {
		gotSRL, err := testutil.Exec1(b, []any{[]int32{-8, 8}, []int32{1, 1}},
			func(f compute.Function, p []compute.Value) (compute.Value, error) {
				return f.ShiftRightLogical(p[0], p[1])
			})
		if err != nil {
			t.Fatalf("ShiftRightLogical failed: %v", err)
		}
		if ok, diff := testutil.IsEqual([]int32{int32(uint32(0xFFFFFFF8) >> 1), 4}, gotSRL); !ok {
			t.Errorf("ShiftRightLogical mismatch:\n%s", diff)
		}

		gotSRA, err := testutil.Exec1(b, []any{[]int32{-8, 8}, []int32{1, 1}},
			func(f compute.Function, p []compute.Value) (compute.Value, error) {
				return f.ShiftRightArithmetic(p[0], p[1])
			})
		if err != nil {
			t.Fatalf("ShiftRightArithmetic failed: %v", err)
		}
		if ok, diff := testutil.IsEqual([]int32{-4, 4}, gotSRA); !ok {
			t.Errorf("ShiftRightArithmetic mismatch:\n%s", diff)
		}

		gotSL, err := testutil.Exec1(b, []any{[]int32{-1, 3}, []int32{1, 2}},
			func(f compute.Function, p []compute.Value) (compute.Value, error) {
				return f.ShiftLeft(p[0], p[1])
			})
		if err != nil {
			t.Fatalf("ShiftLeft failed: %v", err)
		}
		if ok, diff := testutil.IsEqual([]int32{-2, 12}, gotSL); !ok {
			t.Errorf("ShiftLeft mismatch:\n%s", diff)
		}
	})

	t.Run("BitCountAndClz64", func(t *testing.T) {
		gotBC, err := testutil.Exec1(b, []any{[]uint64{0, 1, 0xF0F0F0F00F0F0F0F, ^uint64(0)}},
			func(f compute.Function, p []compute.Value) (compute.Value, error) {
				return f.BitCount(p[0])
			})
		if err != nil {
			t.Fatalf("BitCount uint64 failed: %v", err)
		}
		if ok, diff := testutil.IsEqual([]uint64{0, 1, 32, 64}, gotBC); !ok {
			t.Errorf("BitCount uint64 mismatch:\n%s", diff)
		}

		gotClz, err := testutil.Exec1(b, []any{[]uint64{0, 1, 1 << 63, 1 << 32}},
			func(f compute.Function, p []compute.Value) (compute.Value, error) {
				return f.Clz(p[0])
			})
		if err != nil {
			t.Fatalf("Clz uint64 failed: %v", err)
		}
		if ok, diff := testutil.IsEqual([]uint64{64, 63, 0, 31}, gotClz); !ok {
			t.Errorf("Clz uint64 mismatch:\n%s", diff)
		}
	})

	t.Run("ComplexRealImagConj", func(t *testing.T) {
		re, im, err := testutil.Exec2(b, []any{[]float32{3, -5}, []float32{4, 12}},
			func(f compute.Function, p []compute.Value) (compute.Value, compute.Value, error) {
				c, err := f.Complex(p[0], p[1])
				if err != nil {
					return nil, nil, err
				}
				conj, err := f.Conj(c)
				if err != nil {
					return nil, nil, err
				}
				r, err := f.Real(conj)
				if err != nil {
					return nil, nil, err
				}
				i, err := f.Imag(conj)
				if err != nil {
					return nil, nil, err
				}
				return r, i, nil
			})
		if err != nil {
			t.Fatalf("Complex/Conj/Real/Imag failed: %v", err)
		}
		if ok, diff := testutil.IsEqual([]float32{3, -5}, re); !ok {
			t.Errorf("Real(Conj(Complex)) mismatch:\n%s", diff)
		}
		if ok, diff := testutil.IsEqual([]float32{-4, -12}, im); !ok {
			t.Errorf("Imag(Conj(Complex)) mismatch:\n%s", diff)
		}
	})

	t.Run("IsNaNAndIsFinite", func(t *testing.T) {
		nan := float32(math.NaN())
		inf := float32(math.Inf(1))
		gotNaN, gotFin, err := testutil.Exec2(b, []any{[]float32{1.0, nan, inf}},
			func(f compute.Function, p []compute.Value) (compute.Value, compute.Value, error) {
				n, err := f.IsNaN(p[0])
				if err != nil {
					return nil, nil, err
				}
				fin, err := f.IsFinite(p[0])
				if err != nil {
					return nil, nil, err
				}
				return n, fin, nil
			})
		if err != nil {
			t.Fatalf("IsNaN/IsFinite failed: %v", err)
		}
		if ok, diff := testutil.IsEqual([]bool{false, true, false}, gotNaN); !ok {
			t.Errorf("IsNaN mismatch:\n%s", diff)
		}
		if ok, diff := testutil.IsEqual([]bool{true, false, false}, gotFin); !ok {
			t.Errorf("IsFinite mismatch:\n%s", diff)
		}
	})
}
