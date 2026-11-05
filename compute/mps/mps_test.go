// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mps

import (
	"math"
	"runtime"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/backendtest"
	"github.com/gomlx/compute/support/testutil"
)

// TestCompliance runs the standard compute.Backend compliance test suite.
func TestCompliance(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	backendtest.RunAll(t, backend, nil)
}

// TestBackendCreation tests that the backend can be created.
func TestBackendCreation(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	if backend.Name() != BackendName {
		t.Errorf("Name() = %q, want %q", backend.Name(), BackendName)
	}
	if backend.NumDevices() != 1 {
		t.Errorf("NumDevices() = %d, want 1", backend.NumDevices())
	}
	t.Logf("Backend: %s — %s", backend.Name(), backend.Description())
}

// TestBufferOperations tests buffer creation, read-back, and finalize.
func TestBufferOperations(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	shape := shapes.Make(dtypes.Float32, 2, 3)
	inputData := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}

	buf, err := backend.BufferFromFlatData(0, inputData, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %+v", err)
	}

	gotShape, err := buf.Shape()
	if err != nil {
		t.Fatalf("buf.Shape() failed: %+v", err)
	}
	if !gotShape.Equal(shape) {
		t.Errorf("buf.Shape() = %v, want %v", gotShape, shape)
	}

	outputData := make([]float32, 6)
	if err := buf.ToFlatData(outputData); err != nil {
		t.Fatalf("buf.ToFlatData() failed: %+v", err)
	}
	for i := range inputData {
		if outputData[i] != inputData[i] {
			t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], inputData[i])
		}
	}

	if err := buf.Finalize(); err != nil {
		t.Fatalf("buf.Finalize() failed: %+v", err)
	}
}

// TestSharedBuffer tests shared buffer (Go-managed memory) functionality.
func TestSharedBuffer(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	if !backend.HasSharedBuffers() {
		t.Fatal("HasSharedBuffers() = false, want true")
	}

	shape := shapes.Make(dtypes.Float32, 4)
	buf, flat, err := backend.NewSharedBuffer(0, shape)
	if err != nil {
		t.Fatalf("NewSharedBuffer() failed: %+v", err)
	}

	flatData := flat.([]float32)
	for i := range flatData {
		flatData[i] = float32(i + 1)
	}

	gotFlat, err := buf.Data()
	if err != nil {
		t.Fatalf("buf.Data() failed: %+v", err)
	}
	gotData := gotFlat.([]float32)
	for i := range gotData {
		if gotData[i] != float32(i+1) {
			t.Errorf("gotData[%d] = %f, want %f", i, gotData[i], float32(i+1))
		}
	}
	_ = buf.Finalize()
}

// execUnaryOp is a test helper that builds, compiles, and executes a unary op.
func execUnaryOp(t *testing.T, backend compute.Backend, opFn func(compute.Function, compute.Value) (compute.Value, error),
	input []float32) []float32 {
	t.Helper()

	shape := shapes.Make(dtypes.Float32, len(input))
	builder := backend.Builder("test")
	mainFn := builder.Main()

	x, err := mainFn.Parameter("x", shape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}
	y, err := opFn(mainFn, x)
	if err != nil {
		t.Fatalf("opFn() failed: %+v", err)
	}
	if err := mainFn.Return([]compute.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}
	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	xBuf, err := backend.BufferFromFlatData(0, input, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %+v", err)
	}

	outputs, err := exec.Execute([]compute.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}
	result := make([]float32, len(input))
	if err := outputs[0].ToFlatData(result); err != nil {
		t.Fatalf("ToFlatData() failed: %+v", err)
	}
	return result
}

// execBinaryOp is a test helper that builds, compiles, and executes a binary op.
func execBinaryOp(t *testing.T, backend compute.Backend,
	opFn func(compute.Function, compute.Value, compute.Value) (compute.Value, error),
	shape shapes.Shape, lhsData, rhsData []float32) []float32 {
	t.Helper()

	builder := backend.Builder("test")
	mainFn := builder.Main()

	x, err := mainFn.Parameter("x", shape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}
	y, err := mainFn.Parameter("y", shape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}
	z, err := opFn(mainFn, x, y)
	if err != nil {
		t.Fatalf("opFn() failed: %+v", err)
	}
	if err := mainFn.Return([]compute.Value{z}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}
	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	xBuf, err := backend.BufferFromFlatData(0, lhsData, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData(x) failed: %+v", err)
	}
	yBuf, err := backend.BufferFromFlatData(0, rhsData, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData(y) failed: %+v", err)
	}

	outputs, err := exec.Execute([]compute.Buffer{xBuf, yBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}
	result := make([]float32, len(lhsData))
	if err := outputs[0].ToFlatData(result); err != nil {
		t.Fatalf("ToFlatData() failed: %+v", err)
	}
	return result
}

func assertClose(t *testing.T, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d, want %d", len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > tol {
			t.Errorf("[%d] = %g, want %g", i, got[i], want[i])
		}
	}
}

// TestAddOperation tests the full pipeline: parameter → add → compile → execute → read.
func TestAddOperation(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	shape := shapes.Make(dtypes.Float32, 2, 3)
	result := execBinaryOp(t, backend,
		func(f compute.Function, x, y compute.Value) (compute.Value, error) { return f.Add(x, y) },
		shape,
		[]float32{1, 2, 3, 4, 5, 6},
		[]float32{10, 20, 30, 40, 50, 60},
	)
	assertClose(t, result, []float32{11, 22, 33, 44, 55, 66}, 1e-5)
}

// TestUnaryOperations tests several unary math ops.
func TestUnaryOperations(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	tests := []struct {
		name     string
		opFn     func(compute.Function, compute.Value) (compute.Value, error)
		input    []float32
		expected []float32
		tol      float64
	}{
		{"Abs", func(f compute.Function, x compute.Value) (compute.Value, error) { return f.Abs(x) },
			[]float32{-1, 2, -3, 4}, []float32{1, 2, 3, 4}, 1e-5},
		{"Neg", func(f compute.Function, x compute.Value) (compute.Value, error) { return f.Neg(x) },
			[]float32{-1, 2, -3, 4}, []float32{1, -2, 3, -4}, 1e-5},
		{"Sqrt", func(f compute.Function, x compute.Value) (compute.Value, error) { return f.Sqrt(x) },
			[]float32{1, 4, 9, 16}, []float32{1, 2, 3, 4}, 1e-5},
		{"Exp", func(f compute.Function, x compute.Value) (compute.Value, error) { return f.Exp(x) },
			[]float32{0, 1}, []float32{1, float32(math.E)}, 1e-5},
		{"Log", func(f compute.Function, x compute.Value) (compute.Value, error) { return f.Log(x) },
			[]float32{1, float32(math.E)}, []float32{0, 1}, 1e-4},
		{"Tanh", func(f compute.Function, x compute.Value) (compute.Value, error) { return f.Tanh(x) },
			[]float32{0, 1}, []float32{0, float32(math.Tanh(1))}, 1e-5},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := execUnaryOp(t, backend, tc.opFn, tc.input)
			assertClose(t, got, tc.expected, tc.tol)
		})
	}
}

// TestBinaryOperations tests binary math ops.
func TestBinaryOperations(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	shape := shapes.Make(dtypes.Float32, 4)
	lhs := []float32{10, 20, 30, 40}
	rhs := []float32{3, 5, 10, 8}

	t.Run("Sub", func(t *testing.T) {
		got := execBinaryOp(t, backend,
			func(f compute.Function, x, y compute.Value) (compute.Value, error) { return f.Sub(x, y) },
			shape, lhs, rhs)
		assertClose(t, got, []float32{7, 15, 20, 32}, 1e-5)
	})

	t.Run("Mul", func(t *testing.T) {
		got := execBinaryOp(t, backend,
			func(f compute.Function, x, y compute.Value) (compute.Value, error) { return f.Mul(x, y) },
			shape, lhs, rhs)
		assertClose(t, got, []float32{30, 100, 300, 320}, 1e-5)
	})

	t.Run("Div", func(t *testing.T) {
		got := execBinaryOp(t, backend,
			func(f compute.Function, x, y compute.Value) (compute.Value, error) { return f.Div(x, y) },
			shape, lhs, rhs)
		assertClose(t, got, []float32{10.0 / 3, 4, 3, 5}, 1e-4)
	})

	t.Run("Max", func(t *testing.T) {
		got := execBinaryOp(t, backend,
			func(f compute.Function, x, y compute.Value) (compute.Value, error) { return f.Max(x, y) },
			shape, lhs, rhs)
		assertClose(t, got, []float32{10, 20, 30, 40}, 1e-5)
	})

	t.Run("Min", func(t *testing.T) {
		got := execBinaryOp(t, backend,
			func(f compute.Function, x, y compute.Value) (compute.Value, error) { return f.Min(x, y) },
			shape, lhs, rhs)
		assertClose(t, got, []float32{3, 5, 10, 8}, 1e-5)
	})
}

// TestReshape tests the Reshape operation.
func TestReshape(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_reshape")
	mainFn := builder.Main()

	inShape := shapes.Make(dtypes.Float32, 2, 3)
	x, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}
	y, err := mainFn.Reshape(x, 3, 2)
	if err != nil {
		t.Fatalf("Reshape() failed: %+v", err)
	}
	if err := mainFn.Return([]compute.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	outShapes := exec.Outputs()
	if len(outShapes) != 1 {
		t.Fatalf("Expected 1 output, got %d", len(outShapes))
	}
	expectedOutShape := shapes.Make(dtypes.Float32, 3, 2)
	if !outShapes[0].Equal(expectedOutShape) {
		t.Errorf("Output shape = %v, want %v", outShapes[0], expectedOutShape)
	}

	xData := []float32{1, 2, 3, 4, 5, 6}
	xBuf, err := backend.BufferFromFlatData(0, xData, inShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %+v", err)
	}

	outputs, err := exec.Execute([]compute.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	result := make([]float32, 6)
	if err := outputs[0].ToFlatData(result); err != nil {
		t.Fatalf("ToFlatData() failed: %+v", err)
	}
	assertClose(t, result, xData, 0)
}

// TestConstant tests that Constant values are compiled into the graph correctly.
func TestConstant(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_constant")
	mainFn := builder.Main()

	inShape := shapes.Make(dtypes.Float32, 3)
	x, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}

	c, err := mainFn.Constant([]float32{10, 20, 30}, 3)
	if err != nil {
		t.Fatalf("Constant() failed: %+v", err)
	}

	y, err := mainFn.Add(x, c)
	if err != nil {
		t.Fatalf("Add() failed: %+v", err)
	}

	if err := mainFn.Return([]compute.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	xBuf, err := backend.BufferFromFlatData(0, []float32{1, 2, 3}, inShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %+v", err)
	}

	outputs, err := exec.Execute([]compute.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	result := make([]float32, 3)
	if err := outputs[0].ToFlatData(result); err != nil {
		t.Fatalf("ToFlatData() failed: %+v", err)
	}
	assertClose(t, result, []float32{11, 22, 33}, 1e-5)
}

// TestReduceSum tests the ReduceSum operation.
func TestReduceSum(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_reduce_sum")
	mainFn := builder.Main()

	inShape := shapes.Make(dtypes.Float32, 2, 3)
	x, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}

	y, err := mainFn.ReduceSum(x, 1)
	if err != nil {
		t.Fatalf("ReduceSum() failed: %+v", err)
	}

	if err := mainFn.Return([]compute.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	xBuf, err := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6}, inShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %+v", err)
	}

	outputs, err := exec.Execute([]compute.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	result := make([]float32, 2)
	if err := outputs[0].ToFlatData(result); err != nil {
		t.Fatalf("ToFlatData() failed: %+v", err)
	}
	assertClose(t, result, []float32{6, 15}, 1e-5)
}

// TestDotGeneral tests the DotGeneral (batched matmul) operation.
func TestDotGeneral(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	t.Run("SimpleMatMul", func(t *testing.T) {
		builder := backend.Builder("test_dot_simple")
		mainFn := builder.Main()

		lhsShape := shapes.Make(dtypes.Float32, 2, 3)
		rhsShape := shapes.Make(dtypes.Float32, 3, 2)

		lhs, err := mainFn.Parameter("lhs", lhsShape, nil)
		if err != nil {
			t.Fatalf("Parameter() failed: %+v", err)
		}
		rhs, err := mainFn.Parameter("rhs", rhsShape, nil)
		if err != nil {
			t.Fatalf("Parameter() failed: %+v", err)
		}

		y, err := mainFn.DotGeneral(lhs, []int{1}, nil, rhs, []int{0}, nil, compute.DotGeneralConfig{})
		if err != nil {
			t.Fatalf("DotGeneral() failed: %+v", err)
		}

		if err := mainFn.Return([]compute.Value{y}, nil); err != nil {
			t.Fatalf("Return() failed: %+v", err)
		}

		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile() failed: %+v", err)
		}
		defer exec.Finalize()

		lhsBuf, _ := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6}, lhsShape)
		rhsBuf, _ := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6}, rhsShape)

		outputs, err := exec.Execute([]compute.Buffer{lhsBuf, rhsBuf}, nil, 0)
		if err != nil {
			t.Fatalf("Execute() failed: %+v", err)
		}

		result := make([]float32, 4)
		if err := outputs[0].ToFlatData(result); err != nil {
			t.Fatalf("ToFlatData() failed: %+v", err)
		}
		assertClose(t, result, []float32{22, 28, 49, 64}, 1e-4)
	})

	t.Run("BatchedMatMul", func(t *testing.T) {
		builder := backend.Builder("test_dot_batched")
		mainFn := builder.Main()

		lhsShape := shapes.Make(dtypes.Float32, 2, 2, 3)
		rhsShape := shapes.Make(dtypes.Float32, 2, 3, 2)

		lhs, err := mainFn.Parameter("lhs", lhsShape, nil)
		if err != nil {
			t.Fatalf("Parameter() failed: %+v", err)
		}
		rhs, err := mainFn.Parameter("rhs", rhsShape, nil)
		if err != nil {
			t.Fatalf("Parameter() failed: %+v", err)
		}

		y, err := mainFn.DotGeneral(lhs, []int{2}, []int{0}, rhs, []int{1}, []int{0}, compute.DotGeneralConfig{})
		if err != nil {
			t.Fatalf("DotGeneral() failed: %+v", err)
		}

		if err := mainFn.Return([]compute.Value{y}, nil); err != nil {
			t.Fatalf("Return() failed: %+v", err)
		}

		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile() failed: %+v", err)
		}
		defer exec.Finalize()

		lhsData := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
		rhsData := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
		lhsBuf, _ := backend.BufferFromFlatData(0, lhsData, lhsShape)
		rhsBuf, _ := backend.BufferFromFlatData(0, rhsData, rhsShape)

		outputs, err := exec.Execute([]compute.Buffer{lhsBuf, rhsBuf}, nil, 0)
		if err != nil {
			t.Fatalf("Execute() failed: %+v", err)
		}

		result := make([]float32, 8)
		if err := outputs[0].ToFlatData(result); err != nil {
			t.Fatalf("ToFlatData() failed: %+v", err)
		}
		assertClose(t, result, []float32{22, 28, 49, 64, 220, 244, 301, 334}, 1e-3)
	})
}

// TestGather tests the Gather operation (embedding lookup pattern).
func TestGather(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	t.Run("EmbeddingLookup", func(t *testing.T) {
		builder := backend.Builder("test_gather_embed")
		mainFn := builder.Main()

		operandShape := shapes.Make(dtypes.Float32, 4, 3)
		indicesShape := shapes.Make(dtypes.Int32, 2, 1)

		operand, err := mainFn.Parameter("operand", operandShape, nil)
		if err != nil {
			t.Fatalf("Parameter() failed: %+v", err)
		}
		indices, err := mainFn.Parameter("indices", indicesShape, nil)
		if err != nil {
			t.Fatalf("Parameter() failed: %+v", err)
		}

		result, err := mainFn.Gather(operand, indices,
			1,
			[]int{1},
			[]int{0},
			[]int{0},
			[]int{1, 3},
			false,
		)
		if err != nil {
			t.Fatalf("Gather() failed: %+v", err)
		}

		if err := mainFn.Return([]compute.Value{result}, nil); err != nil {
			t.Fatalf("Return() failed: %+v", err)
		}

		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile() failed: %+v", err)
		}
		defer exec.Finalize()

		operandData := []float32{10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42}
		indicesData := []int32{1, 3}

		operandBuf, err := backend.BufferFromFlatData(0, operandData, operandShape)
		if err != nil {
			t.Fatalf("BufferFromFlatData(operand) failed: %+v", err)
		}
		indicesBuf, err := backend.BufferFromFlatData(0, indicesData, indicesShape)
		if err != nil {
			t.Fatalf("BufferFromFlatData(indices) failed: %+v", err)
		}

		outputs, err := exec.Execute([]compute.Buffer{operandBuf, indicesBuf}, nil, 0)
		if err != nil {
			t.Fatalf("Execute() failed: %+v", err)
		}

		got := make([]float32, 6)
		if err := outputs[0].ToFlatData(got); err != nil {
			t.Fatalf("ToFlatData() failed: %+v", err)
		}
		assertClose(t, got, []float32{20, 21, 22, 40, 41, 42}, 1e-5)
	})
}

// TestFusedSoftmax tests the FusedSoftmax operation.
func TestFusedSoftmax(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_softmax")
	mainFn := builder.Main()

	inShape := shapes.Make(dtypes.Float32, 2, 3)
	x, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}

	y, err := mainFn.FusedSoftmax(x, 1)
	if err != nil {
		t.Fatalf("FusedSoftmax() failed: %+v", err)
	}

	if err := mainFn.Return([]compute.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	xBuf, err := backend.BufferFromFlatData(0, []float32{1, 2, 3, 1, 1, 1}, inShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %+v", err)
	}

	outputs, err := exec.Execute([]compute.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]float32, 6)
	if err := outputs[0].ToFlatData(got); err != nil {
		t.Fatalf("ToFlatData() failed: %+v", err)
	}

	expected := []float32{
		float32(math.Exp(1) / (math.Exp(1) + math.Exp(2) + math.Exp(3))),
		float32(math.Exp(2) / (math.Exp(1) + math.Exp(2) + math.Exp(3))),
		float32(math.Exp(3) / (math.Exp(1) + math.Exp(2) + math.Exp(3))),
		1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0,
	}
	assertClose(t, got, expected, 1e-4)
}

// TestFusedActivation tests the FusedActivation operation.
func TestFusedActivation(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	t.Run("Gelu", func(t *testing.T) {
		got, err := testutil.Exec1(backend, []any{[]float32{0.0, 1.0, -1.0, 3.0}}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.FusedActivation(params[0], compute.ActivationConfig{Type: compute.ActivationGelu})
		})
		if err != nil {
			t.Fatalf("FusedActivation(Gelu) failed: %+v", err)
		}
		assertClose(t, got.([]float32), []float32{0.0, 0.8413, -0.1587, 2.9960}, 1e-3)
	})

	t.Run("GeluApproximate", func(t *testing.T) {
		got, err := testutil.Exec1(backend, []any{[]float32{0.0, 1.0, -1.0, 3.0}}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.FusedActivation(params[0], compute.ActivationConfig{Type: compute.ActivationGeluApproximate})
		})
		if err != nil {
			t.Fatalf("FusedActivation(GeluApproximate) failed: %+v", err)
		}
		assertClose(t, got.([]float32), []float32{0.0, 0.8412, -0.1588, 2.9964}, 1e-3)
	})

	t.Run("Relu", func(t *testing.T) {
		got, err := testutil.Exec1(backend, []any{[]float32{-2.0, -1.0, 0.0, 1.5, 3.0}}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.FusedActivation(params[0], compute.ActivationConfig{Type: compute.ActivationRelu})
		})
		if err != nil {
			t.Fatalf("FusedActivation(Relu) failed: %+v", err)
		}
		assertClose(t, got.([]float32), []float32{0.0, 0.0, 0.0, 1.5, 3.0}, 1e-5)
	})
}

// TestFusedLayerNorm tests the FusedLayerNorm operation.
func TestFusedLayerNorm(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	t.Run("basic", func(t *testing.T) {
		x := [][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}}
		got, err := testutil.Exec1(backend, []any{x}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.FusedLayerNorm(params[0], []int{-1}, 1e-5, nil, nil)
		})
		if err != nil {
			t.Fatalf("FusedLayerNorm failed: %+v", err)
		}
		want := [][]float32{{-1.3416, -0.4472, 0.4472, 1.3416}, {-1.3416, -0.4472, 0.4472, 1.3416}}
		if ok, diff := testutil.IsInDelta(want, got, 1e-3); !ok {
			t.Errorf("Mismatch:\n%s", diff)
		}
	})

	t.Run("with_gamma_beta", func(t *testing.T) {
		x := [][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}}
		gamma := []float32{2, 2, 2, 2}
		beta := []float32{1, 1, 1, 1}
		got, err := testutil.Exec1(backend, []any{x, gamma, beta}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.FusedLayerNorm(params[0], []int{-1}, 1e-5, params[1], params[2])
		})
		if err != nil {
			t.Fatalf("FusedLayerNorm with gamma/beta failed: %+v", err)
		}
		want := [][]float32{{-1.6833, 0.1056, 1.8944, 3.6833}, {-1.6833, 0.1056, 1.8944, 3.6833}}
		if ok, diff := testutil.IsInDelta(want, got, 1e-3); !ok {
			t.Errorf("Mismatch:\n%s", diff)
		}
	})
}

// TestFusedDense tests the FusedDense operation (matmul + bias + activation).
func TestFusedDense(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	t.Run("no_activation_no_bias", func(t *testing.T) {
		x := [][]float32{{1, 2, 3}, {4, 5, 6}}
		w := [][]float32{
			{1, 0, 0, 0},
			{0, 1, 0, 0},
			{0, 0, 1, 0},
		}
		got, err := testutil.Exec1(backend, []any{x, w}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.FusedDense(params[0], params[1], nil, compute.DenseConfig{
				Activation:   compute.ActivationConfig{Type: compute.ActivationNone},
				WeightLayout: compute.DenseLayoutInputOutputs,
			})
		})
		if err != nil {
			t.Fatalf("FusedDense failed: %+v", err)
		}
		want := [][]float32{{1, 2, 3, 0}, {4, 5, 6, 0}}
		if ok, diff := testutil.IsInDelta(want, got, 1e-4); !ok {
			t.Errorf("Mismatch:\n%s", diff)
		}
	})

	t.Run("with_bias", func(t *testing.T) {
		x := [][]float32{{1, 0}, {0, 1}}
		w := [][]float32{{2, 0}, {0, 3}}
		b := []float32{10, 20}
		got, err := testutil.Exec1(backend, []any{x, w, b}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.FusedDense(params[0], params[1], params[2], compute.DenseConfig{
				Activation:   compute.ActivationConfig{Type: compute.ActivationNone},
				WeightLayout: compute.DenseLayoutInputOutputs,
			})
		})
		if err != nil {
			t.Fatalf("FusedDense failed: %+v", err)
		}
		want := [][]float32{{12, 20}, {10, 23}}
		if ok, diff := testutil.IsInDelta(want, got, 1e-4); !ok {
			t.Errorf("Mismatch:\n%s", diff)
		}
	})

	t.Run("DenseLayoutOutputsInput", func(t *testing.T) {
		x := [][]float32{{1, -2, 3}}
		w := [][]float32{{1, 0, 1}, {0, 1, 0}}
		bias := []float32{1, 2}
		got, err := testutil.Exec1(backend, []any{x, w, bias}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.FusedDense(params[0], params[1], params[2], compute.DenseConfig{
				Activation:   compute.ActivationConfig{Type: compute.ActivationNone},
				WeightLayout: compute.DenseLayoutOutputsInput,
			})
		})
		if err != nil {
			t.Fatalf("FusedDense failed: %+v", err)
		}
		want := [][]float32{{5, 0}}
		if ok, diff := testutil.IsInDelta(want, got, 1e-4); !ok {
			t.Errorf("Mismatch:\n%s", diff)
		}
	})

	t.Run("relu_activation", func(t *testing.T) {
		x := [][]float32{{1, -1}, {-1, 1}}
		w := [][]float32{{1, 0}, {0, 1}}
		got, err := testutil.Exec1(backend, []any{x, w}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.FusedDense(params[0], params[1], nil, compute.DenseConfig{
				Activation:   compute.ActivationConfig{Type: compute.ActivationRelu},
				WeightLayout: compute.DenseLayoutInputOutputs,
			})
		})
		if err != nil {
			t.Fatalf("FusedDense failed: %+v", err)
		}
		want := [][]float32{{1, 0}, {0, 1}}
		if ok, diff := testutil.IsInDelta(want, got, 1e-4); !ok {
			t.Errorf("Mismatch:\n%s", diff)
		}
	})
}

// TestFusedScaledDotProductAttention tests the SDPA fused op.
func TestFusedScaledDotProductAttention(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	t.Run("basic_BHSD", func(t *testing.T) {
		qkv := [][][][]float32{{{{1, 0}, {0, 1}}}}
		scale := 1.0 / math.Sqrt(2.0)

		got, err := testutil.Exec1(backend, []any{qkv, qkv, qkv}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			out, _, err := f.FusedScaledDotProductAttention(
				params[0], params[1], params[2], compute.AttentionAxesLayoutBHSD,
				&compute.ScaledDotProductAttentionConfig{Scale: scale})
			return out, err
		})
		if err != nil {
			t.Fatalf("SDPA failed: %+v", err)
		}
		out := got.([][][][]float32)
		for _, b := range out {
			for _, h := range b {
				for _, s := range h {
					for _, v := range s {
						if v < -0.1 || v > 1.1 {
							t.Errorf("Output = %f, expected in [0,1]", v)
						}
					}
				}
			}
		}
	})

	t.Run("causal_mask", func(t *testing.T) {
		q := [][][][]float32{{{{1, 0}, {0, 1}, {1, 1}}}}
		k := [][][][]float32{{{{1, 0}, {0, 1}, {1, 1}}}}
		v := [][][][]float32{{{{1, 0}, {0, 1}, {0.5, 0.5}}}}

		got, err := testutil.Exec1(backend, []any{q, k, v}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			out, _, err := f.FusedScaledDotProductAttention(
				params[0], params[1], params[2], compute.AttentionAxesLayoutBHSD,
				&compute.ScaledDotProductAttentionConfig{Scale: 1.0, Causal: true})
			return out, err
		})
		if err != nil {
			t.Fatalf("SDPA causal failed: %+v", err)
		}
		out := got.([][][][]float32)
		assertClose(t, out[0][0][0], []float32{1, 0}, 1e-3)
	})

	t.Run("BSHD_layout", func(t *testing.T) {
		// B=1, S=2, H=1, D=2
		qkv := [][][][]float32{{{{1, 0}}, {{0, 1}}}}
		scale := 1.0 / math.Sqrt(2.0)

		got, err := testutil.Exec1(backend, []any{qkv, qkv, qkv}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			out, _, err := f.FusedScaledDotProductAttention(
				params[0], params[1], params[2], compute.AttentionAxesLayoutBSHD,
				&compute.ScaledDotProductAttentionConfig{Scale: scale})
			return out, err
		})
		if err != nil {
			t.Fatalf("SDPA BSHD failed: %+v", err)
		}
		out := got.([][][][]float32)
		if len(out) != 1 || len(out[0]) != 2 || len(out[0][0]) != 1 || len(out[0][0][0]) != 2 {
			t.Errorf("Unexpected BSHD shape: [%d,%d,%d,%d], want [1,2,1,2]",
				len(out), len(out[0]), len(out[0][0]), len(out[0][0][0]))
		}
	})

	t.Run("boolean_mask", func(t *testing.T) {
		q := [][][][]float32{{{{1, 0}, {0, 1}}}}
		k := [][][][]float32{{{{1, 0}, {0, 1}}}}
		v := [][][][]float32{{{{10, 20}, {30, 40}}}}
		mask := [][][][]bool{{{{true, false}, {true, true}}}}

		got, err := testutil.Exec1(backend, []any{q, k, v, mask}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			out, _, err := f.FusedScaledDotProductAttention(
				params[0], params[1], params[2], compute.AttentionAxesLayoutBHSD,
				&compute.ScaledDotProductAttentionConfig{Scale: 1.0, Mask: params[3]})
			return out, err
		})
		if err != nil {
			t.Fatalf("SDPA boolean mask failed: %+v", err)
		}
		out := got.([][][][]float32)
		assertClose(t, out[0][0][0], []float32{10, 20}, 1e-3)
	})
}

// TestSharedBuffersAndGCCleanup verifies zero-copy MTLBuffer shared buffers and runtime.AddCleanup GC release.
func TestSharedBuffersAndGCCleanup(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	if !backend.HasSharedBuffers() {
		t.Fatal("HasSharedBuffers() should be true")
	}

	shape := shapes.Make(dtypes.Float32, 4)
	inBuf, flatAny, err := backend.NewSharedBuffer(0, shape)
	if err != nil {
		t.Fatalf("NewSharedBuffer() failed: %+v", err)
	}
	defer func() { _ = inBuf.Finalize() }()

	inSlice := flatAny.([]float32)
	copy(inSlice, []float32{1, 2, 3, 4})

	// Build a simple graph y = x * 2
	builder := backend.Builder("shared_buf_test")
	mainFn := builder.Main()
	x, err := mainFn.Parameter("x", shape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}
	two, err := mainFn.Constant([]float32{2, 2, 2, 2}, 4)
	if err != nil {
		t.Fatalf("Constant() failed: %+v", err)
	}
	y, err := mainFn.Mul(x, two)
	if err != nil {
		t.Fatalf("Mul() failed: %+v", err)
	}
	if err := mainFn.Return([]compute.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}
	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	// First execution
	outs, err := exec.Execute([]compute.Buffer{inBuf}, []bool{false}, 0)
	if err != nil {
		t.Fatalf("Execute() #1 failed: %+v", err)
	}
	outDataAny, err := outs[0].Data()
	if err != nil {
		t.Fatalf("outs[0].Data() failed: %+v", err)
	}
	assertClose(t, outDataAny.([]float32), []float32{2, 4, 6, 8}, 1e-5)
	_ = outs[0].Finalize()

	// Mutate the shared input slice in-place without creating a new buffer
	copy(inSlice, []float32{10, 20, 30, 40})
	outs2, err := exec.Execute([]compute.Buffer{inBuf}, []bool{false}, 0)
	if err != nil {
		t.Fatalf("Execute() #2 failed: %+v", err)
	}
	outDataAny2, err := outs2[0].Data()
	if err != nil {
		t.Fatalf("outs2[0].Data() failed: %+v", err)
	}
	assertClose(t, outDataAny2.([]float32), []float32{20, 40, 60, 80}, 1e-5)
	_ = outs2[0].Finalize()

	// Allocate buffers without calling Finalize() and trigger GC to exercise runtime.AddCleanup.
	func() {
		for range 16 {
			b, s, err := backend.NewSharedBuffer(0, shapes.Make(dtypes.Float32, 1024))
			if err != nil {
				t.Fatalf("NewSharedBuffer() failed: %+v", err)
			}
			s.([]float32)[0] = 42
			_ = b
		}
	}()
	runtime.GC()
	runtime.GC()
}

// TestDonatedBufferReuse verifies that donated input MTLBuffers are pooled and recycled
// across executions (ping-ponging without allocating new Metal buffers) while preserving
// correctness even for non-elementwise ops like Transpose, and respecting all pooling bounds.
func TestDonatedBufferReuse(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	t.Run("PingPongTranspose", func(t *testing.T) {
		const N = 512
		shape := shapes.Make(dtypes.Float32, N, N)

		builder := backend.Builder("transpose_donated")
		mainFn := builder.Main()
		x, _ := mainFn.Parameter("x", shape, nil)
		y, _ := mainFn.Transpose(x, 1, 0)
		_ = mainFn.Return([]compute.Value{y}, nil)
		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile() failed: %+v", err)
		}
		defer exec.Finalize()

		inBuf, inFlatAny, err := backend.NewSharedBuffer(0, shape)
		if err != nil {
			t.Fatalf("NewSharedBuffer() failed: %+v", err)
		}
		inFlat := inFlatAny.([]float32)
		for i := range inFlat {
			inFlat[i] = float32(i + 1)
		}
		ptr0, _ := inBuf.(*gpuBuffer).flatDataPtr()

		// Step 1: Donate inBuf (ptr0) -> produces outs1 (ptr1), and returns ptr0 to the executable's pool.
		outs1, err := exec.Execute([]compute.Buffer{inBuf}, []bool{true}, 0)
		if err != nil {
			t.Fatalf("Execute() #1 failed: %+v", err)
		}
		outFlatAny1, err := outs1[0].Data()
		if err != nil {
			t.Fatalf("Data() #1 failed: %+v", err)
		}
		outFlat1 := outFlatAny1.([]float32)
		for r := range N {
			for c := range N {
				want := float32(c*N + r + 1)
				if outFlat1[r*N+c] != want {
					t.Fatalf("Transpose #1 mismatch at (%d,%d): got %v, want %v", r, c, outFlat1[r*N+c], want)
				}
			}
		}

		// Step 2: Donate outs1[0] (ptr1) -> output buffer must recycle ptr0 from the pool!
		outs2, err := exec.Execute([]compute.Buffer{outs1[0]}, []bool{true}, 0)
		if err != nil {
			t.Fatalf("Execute() #2 failed: %+v", err)
		}
		defer func() { _ = outs2[0].Finalize() }()

		ptr2, _ := outs2[0].(*gpuBuffer).flatDataPtr()
		if ptr2 != ptr0 {
			t.Errorf("Expected Step 2 output to recycle donated Step 1 MTLBuffer pointer %p from pool, got %p", ptr0, ptr2)
		}
		outFlatAny2, err := outs2[0].Data()
		if err != nil {
			t.Fatalf("Data() #2 failed: %+v", err)
		}
		outFlat2 := outFlatAny2.([]float32)
		for i := range outFlat2 {
			if outFlat2[i] != float32(i+1) {
				t.Fatalf("Transpose #2 mismatch at %d: got %v, want %v", i, outFlat2[i], float32(i+1))
			}
		}
	})

	t.Run("PoolCapAndNonMatchingSizes", func(t *testing.T) {
		// Graph takes 2 inputs of shape [4] and 1 input of shape [2, 4] (different byte size),
		// and returns only 1 output of shape [4].
		// Even if all 3 inputs are donated:
		// - Only 1 buffer of shape [4] (16 bytes) may be pooled (matching the 1 output of 16 bytes).
		// - The [2, 4] input (32 bytes) must NOT be pooled because no output has 32 bytes.
		shape4 := shapes.Make(dtypes.Float32, 4)
		shape2x4 := shapes.Make(dtypes.Float32, 2, 4)

		builder := backend.Builder("pool_bounds")
		mainFn := builder.Main()
		a, _ := mainFn.Parameter("a", shape4, nil)
		b, _ := mainFn.Parameter("b", shape4, nil)
		c, _ := mainFn.Parameter("c", shape2x4, nil)
		cRed, _ := mainFn.ReduceSum(c, 0) // shape [4]
		ab, _ := mainFn.Add(a, b)
		out, _ := mainFn.Add(ab, cRed)
		_ = mainFn.Return([]compute.Value{out}, nil)
		execIface, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile() failed: %+v", err)
		}
		defer execIface.Finalize()
		exec := execIface.(*Executable)

		aBuf, _ := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4}, shape4)
		bBuf, _ := backend.BufferFromFlatData(0, []float32{10, 20, 30, 40}, shape4)
		cBuf, _ := backend.BufferFromFlatData(0, []float32{1, 1, 1, 1, 2, 2, 2, 2}, shape2x4)

		outs, err := exec.Execute([]compute.Buffer{aBuf, bBuf, cBuf}, []bool{true, true, true}, 0)
		if err != nil {
			t.Fatalf("Execute() failed: %+v", err)
		}
		defer func() { _ = outs[0].Finalize() }()

		if got := len(exec.pool.buffers[shape4.ByteSize()]); got != 1 {
			t.Errorf("Expected exactly 1 pooled buffer for size %d, got %d", shape4.ByteSize(), got)
		}
		if got := len(exec.pool.buffers[shape2x4.ByteSize()]); got != 0 {
			t.Errorf("Expected 0 pooled buffers for non-output size %d, got %d", shape2x4.ByteSize(), got)
		}
	})

	t.Run("CrossExecutableStealing", func(t *testing.T) {
		// Simulate multi-task / multi-shape training:
		// Executable A outputs a shared variable shape [8] (32 bytes) and a task-A-specific shape [3] (12 bytes).
		// Executable B outputs the shared variable shape [8] (32 bytes) only.
		sharedShape := shapes.Make(dtypes.Float32, 8)
		taskAShape := shapes.Make(dtypes.Float32, 3)

		// Build Executable A
		bA := backend.Builder("exec_A")
		fnA := bA.Main()
		wA, _ := fnA.Parameter("w", sharedShape, nil)
		tA, _ := fnA.Parameter("t", taskAShape, nil)
		wAOut, _ := fnA.Add(wA, wA)
		tAOut, _ := fnA.Add(tA, tA)
		_ = fnA.Return([]compute.Value{wAOut, tAOut}, nil)
		execAIface, err := bA.Compile()
		if err != nil {
			t.Fatalf("Compile(A) failed: %+v", err)
		}
		defer execAIface.Finalize()
		execA := execAIface.(*Executable)

		// Build Executable B
		bB := backend.Builder("exec_B")
		fnB := bB.Main()
		wB, _ := fnB.Parameter("w", sharedShape, nil)
		wBOut, _ := fnB.Neg(wB)
		_ = fnB.Return([]compute.Value{wBOut}, nil)
		execBIface, err := bB.Compile()
		if err != nil {
			t.Fatalf("Compile(B) failed: %+v", err)
		}
		defer execBIface.Finalize()
		execB := execBIface.(*Executable)

		w0, _ := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6, 7, 8}, sharedShape)
		t0, _ := backend.BufferFromFlatData(0, []float32{10, 20, 30}, taskAShape)
		w0Ptr, _ := w0.(*gpuBuffer).flatDataPtr()
		t0Ptr, _ := t0.(*gpuBuffer).flatDataPtr()

		// Step 1: Run A, donating w0 and t0 -> both are pooled into execA.pool.
		outsA1, err := execA.Execute([]compute.Buffer{w0, t0}, []bool{true, true}, 0)
		if err != nil {
			t.Fatalf("execA.Execute #1 failed: %+v", err)
		}

		// Step 2: Run B, donating outsA1[0] (w1).
		// B should steal w0Ptr from execA.pool for its output, and leave t0Ptr in execA.pool untouched!
		outsB1, err := execB.Execute([]compute.Buffer{outsA1[0]}, []bool{true}, 0)
		if err != nil {
			t.Fatalf("execB.Execute #1 failed: %+v", err)
		}
		outBPtr, _ := outsB1[0].(*gpuBuffer).flatDataPtr()
		if outBPtr != w0Ptr {
			t.Errorf("Expected execB to steal w0Ptr %p from execA.pool, got %p", w0Ptr, outBPtr)
		}
		if got := len(execA.pool.buffers[sharedShape.ByteSize()]); got != 0 {
			t.Errorf("Expected execA.pool to have 0 buffers of shared size after steal, got %d", got)
		}
		if got := len(execA.pool.buffers[taskAShape.ByteSize()]); got != 1 {
			t.Errorf("Expected execA.pool to retain 1 buffer of taskAShape size, got %d", got)
		}

		// Step 3: Run A again, donating outsB1[0] and outsA1[1].
		// A should steal w1 from execB.pool AND reuse t0Ptr from execA.pool!
		outsA2, err := execA.Execute([]compute.Buffer{outsB1[0], outsA1[1]}, []bool{true, true}, 0)
		if err != nil {
			t.Fatalf("execA.Execute #2 failed: %+v", err)
		}
		defer func() {
			_ = outsA2[0].Finalize()
			_ = outsA2[1].Finalize()
		}()
		outATaskPtr, _ := outsA2[1].(*gpuBuffer).flatDataPtr()
		if outATaskPtr != t0Ptr {
			t.Errorf("Expected execA #2 to reuse preserved taskA buffer %p from execA.pool, got %p", t0Ptr, outATaskPtr)
		}
	})
}


