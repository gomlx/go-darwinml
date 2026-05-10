//go:build darwin && cgo

package metal

import (
	"math"
	"math/rand/v2"
	"slices"
	"sort"
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/stretchr/testify/require"
	"github.com/x448/float16"
)

func TestMetalCapabilitiesIncludesControlFlow(t *testing.T) {
	require.True(t, MetalCapabilities.Operations[backends.OpTypeIf])
	require.True(t, MetalCapabilities.Operations[backends.OpTypeWhile])
	require.True(t, MetalCapabilities.Operations[backends.OpTypeSort])
	require.True(t, MetalCapabilities.Operations[backends.OpTypeRNGBitGenerator])
	require.True(t, MetalCapabilities.Operations[backends.OpTypeBatchNormForTraining])
	require.True(t, MetalCapabilities.Operations[backends.OpTypeBatchNormGradient])
	require.True(t, MetalCapabilities.Operations[backends.OpTypeDynamicUpdateSlice])
	require.True(t, MetalCapabilities.Operations[backends.OpTypeDynamicSlice])
	require.True(t, MetalCapabilities.Operations[backends.OpTypeShiftLeft])
	require.True(t, MetalCapabilities.Operations[backends.OpTypeSelectAndScatterMax])
	require.True(t, MetalCapabilities.DTypes[dtypes.BFloat16])
}

func TestExecuteDoubleReshapeOfScratchConstant(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("double_reshape")
	main := builder.Main()
	c, err := main.Constant([]float32{1, 2, 3, 4}, 4)
	require.NoError(t, err)
	r1, err := main.Reshape(c, 2, 2)
	require.NoError(t, err)
	r2, err := main.Reshape(r1, 4)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{r2}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)
	require.Len(t, outs, 1)

	buf := outs[0].(*Buffer)
	out := make([]float32, 4)
	require.NoError(t, b.BufferToFlatData(buf, out))
	require.Equal(t, []float32{1, 2, 3, 4}, out)
}

func TestWhereBoolPredicateFloat32(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("where_bool")
	main := builder.Main()
	x, err := main.Constant([]float32{1, 2, 3, 4}, 4)
	require.NoError(t, err)
	two, err := main.Constant([]float32{2, 2, 2, 2}, 4)
	require.NoError(t, err)
	pred, err := main.GreaterThan(x, two)
	require.NoError(t, err)
	ten, err := main.Constant([]float32{10, 10, 10, 10}, 4)
	require.NoError(t, err)
	z, err := main.Constant([]float32{0, 0, 0, 0}, 4)
	require.NoError(t, err)
	y, err := main.Where(pred, ten, z)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{y}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)
	require.Len(t, outs, 1)

	out := make([]float32, 4)
	require.NoError(t, b.BufferToFlatData(outs[0], out))
	require.Equal(t, []float32{0, 0, 10, 10}, out)
}

func TestParamIntermediateReshapeChainThenNeg(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("param_reshape_neg")
	main := builder.Main()
	p, err := main.Parameter("x", shapes.Make(dtypes.Float32, 4), nil)
	require.NoError(t, err)
	r1, err := main.Reshape(p, 2, 2)
	require.NoError(t, err)
	r2, err := main.Reshape(r1, 4)
	require.NoError(t, err)
	o, err := main.Neg(r2)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{o}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	in, err := b.BufferFromFlatData(0, []float32{1, 2, 3, 4}, shapes.Make(dtypes.Float32, 4))
	require.NoError(t, err)
	outs, err := exe.Execute([]backends.Buffer{in}, []bool{false}, 0)
	require.NoError(t, err)

	out := make([]float32, 4)
	require.NoError(t, b.BufferToFlatData(outs[0], out))
	require.Equal(t, []float32{-1, -2, -3, -4}, out)
}

func TestClampAndBatchNormInferenceF32(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("clamp_bn")
	main := builder.Main()
	x, err := main.Parameter("x", shapes.Make(dtypes.Float32, 2, 2), nil)
	require.NoError(t, err)
	lo, err := main.Constant([]float32{0, 0, 0, 0}, 2, 2)
	require.NoError(t, err)
	hi, err := main.Constant([]float32{2, 2, 2, 2}, 2, 2)
	require.NoError(t, err)
	c, err := main.Clamp(lo, x, hi)
	require.NoError(t, err)

	ch := 2
	scale, err := main.Constant([]float32{1, 1}, ch)
	require.NoError(t, err)
	offset, err := main.Constant([]float32{0, 0}, ch)
	require.NoError(t, err)
	mean, err := main.Constant([]float32{0, 0}, ch)
	require.NoError(t, err)
	variance, err := main.Constant([]float32{1, 1}, ch)
	require.NoError(t, err)
	y, err := main.BatchNormForInference(c, scale, offset, mean, variance, 1e-5, 1)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{y}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	in, err := b.BufferFromFlatData(0, []float32{-1, 3, 1, 1}, shapes.Make(dtypes.Float32, 2, 2))
	require.NoError(t, err)
	outs, err := exe.Execute([]backends.Buffer{in}, []bool{false}, 0)
	require.NoError(t, err)

	out := make([]float32, 4)
	require.NoError(t, b.BufferToFlatData(outs[0], out))
	// clamp -> {0,2,1,1}; BN is ~(identity) up to rsqrt numerics
	require.InDeltaSlice(t, []float32{0, 2, 1, 1}, out, 2e-4)
}

func TestRNGBitGeneratorAdvancesState(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("rng_bits")
	main := builder.Main()
	st, err := main.Parameter("s", backends.RNGStateShape, nil)
	require.NoError(t, err)
	ns, bits, err := main.RNGBitGenerator(st, shapes.Make(dtypes.Uint8, 32))
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{ns, bits}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	seed := make([]uint64, 3)
	seed[0] = 0x1234abcd
	seed[1] = 0xf00dcafe
	seed[2] = 0
	in, err := b.BufferFromFlatData(0, seed, backends.RNGStateShape)
	require.NoError(t, err)
	outs, err := exe.Execute([]backends.Buffer{in}, []bool{false}, 0)
	require.NoError(t, err)
	require.Len(t, outs, 2)

	outState := make([]uint64, 3)
	require.NoError(t, b.BufferToFlatData(outs[0], outState))
	require.NotEqual(t, seed[0], outState[0], "PCG state[0] should advance")
	outBits := make([]uint8, 32)
	require.NoError(t, b.BufferToFlatData(outs[1], outBits))
	nonZero := false

	for _, v := range outBits {
		if v != 0 {
			nonZero = true
			break
		}
	}

	require.True(t, nonZero, "expected some random bytes")
}

func TestBatchNormTrainingF32(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("bn_train")
	main := builder.Main()
	input, err := main.Iota(shapes.Make(dtypes.Float32, 7, 3), 0)
	require.NoError(t, err)
	scale, err := main.Constant([]float32{1, 2, 3}, 3)
	require.NoError(t, err)
	offset, err := main.Constant([]float32{10, 100, 1000}, 3)
	require.NoError(t, err)
	norm, bmean, bvar, err := main.BatchNormForTraining(input, scale, offset, 1e-7, -1)
	require.NoError(t, err)
	norm2, err := main.BatchNormForInference(input, scale, offset, bmean, bvar, 1e-7, -1)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{norm, norm2, bmean, bvar}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)
	require.Len(t, outs, 4)

	wantNorm := [][]float32{
		{8.5, 97, 995.5},
		{9, 98, 997},
		{9.5, 99, 998.5},
		{10, 100, 1000},
		{10.5, 101, 1001.5},
		{11, 102, 1003},
		{11.5, 103, 1004.5},
	}
	flatNorm := make([]float32, 7*3)
	require.NoError(t, b.BufferToFlatData(outs[0], flatNorm))

	for r := range wantNorm {
		require.InDeltaSlice(t, wantNorm[r], flatNorm[r*3:(r+1)*3], 1e-3)
	}

	flatNorm2 := make([]float32, 7*3)
	require.NoError(t, b.BufferToFlatData(outs[1], flatNorm2))
	require.InDeltaSlice(t, flatNorm, flatNorm2, 1e-3)

	meanOut := make([]float32, 3)
	require.NoError(t, b.BufferToFlatData(outs[2], meanOut))
	require.InDeltaSlice(t, []float32{3, 3, 3}, meanOut, 1e-4)

	varOut := make([]float32, 3)
	require.NoError(t, b.BufferToFlatData(outs[3], varOut))
	require.InDeltaSlice(t, []float32{4, 4, 4}, varOut, 1e-4)
}

func f16(v float32) float16.Float16 { return float16.Fromfloat32(v) }

func TestBatchNormTrainingF16(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("bn_train_f16")
	main := builder.Main()
	inputF32, err := main.Iota(shapes.Make(dtypes.Float32, 7, 3), 0)
	require.NoError(t, err)
	input, err := main.ConvertDType(inputF32, dtypes.Float16)
	require.NoError(t, err)
	scale, err := main.Constant([]float16.Float16{f16(1), f16(2), f16(3)}, 3)
	require.NoError(t, err)
	offset, err := main.Constant([]float16.Float16{f16(10), f16(100), f16(1000)}, 3)
	require.NoError(t, err)
	norm, bmean, bvar, err := main.BatchNormForTraining(input, scale, offset, 1e-7, -1)
	require.NoError(t, err)
	norm2, err := main.BatchNormForInference(input, scale, offset, bmean, bvar, 1e-7, -1)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{norm, norm2, bmean, bvar}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)
	require.Len(t, outs, 4)

	wantNorm := [][]float32{
		{8.5, 97, 995.5},
		{9, 98, 997},
		{9.5, 99, 998.5},
		{10, 100, 1000},
		{10.5, 101, 1001.5},
		{11, 102, 1003},
		{11.5, 103, 1004.5},
	}
	flatNorm := make([]float16.Float16, 7*3)
	require.NoError(t, b.BufferToFlatData(outs[0], flatNorm))

	for r := range wantNorm {
		got := flatNorm[r*3 : (r+1)*3]

		for c := range wantNorm[r] {
			require.InDelta(t, wantNorm[r][c], got[c].Float32(), 8e-2,
				"norm row %d col %d", r, c)
		}
	}

	flatNorm2 := make([]float16.Float16, 7*3)
	require.NoError(t, b.BufferToFlatData(outs[1], flatNorm2))

	for i := range flatNorm {
		require.InDelta(t, flatNorm[i].Float32(), flatNorm2[i].Float32(), 8e-2, "idx %d", i)
	}

	meanOut := make([]float16.Float16, 3)
	require.NoError(t, b.BufferToFlatData(outs[2], meanOut))

	for i, want := range []float32{3, 3, 3} {
		require.InDelta(t, want, meanOut[i].Float32(), 5e-2)
	}

	varOut := make([]float16.Float16, 3)
	require.NoError(t, b.BufferToFlatData(outs[3], varOut))

	for i, want := range []float32{4, 4, 4} {
		require.InDelta(t, want, varOut[i].Float32(), 5e-2)
	}
}

func TestScatterSumF16(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("scatter_sum_f16")
	main := builder.Main()
	zeros := make([]float16.Float16, 5)
	init, err := main.Constant(zeros, 5)
	require.NoError(t, err)
	flat, err := main.Constant([]float16.Float16{
		f16(1), f16(3), f16(5), f16(7), f16(11), f16(13),
	}, 6)
	require.NoError(t, err)
	idx, err := main.Constant([]int32{0, 0, 0, 1, 1, 3}, 6, 1)
	require.NoError(t, err)
	y, err := main.ScatterSum(init, idx, flat, 1, nil, []int{0}, []int{0}, true, false)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{y}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	got := make([]float16.Float16, 5)
	require.NoError(t, b.BufferToFlatData(outs[0], got))

	want := []float32{1 + 3 + 5, 7 + 11, 0, 13, 0}

	for i := range want {
		require.InDelta(t, want[i], got[i].Float32(), 5e-2, "i=%d", i)
	}
}

func TestScatterSumI64(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("scatter_sum_i64")
	main := builder.Main()
	init, err := main.Constant(make([]int64, 5), 5)
	require.NoError(t, err)
	flat, err := main.Constant([]int64{1, 3, 5, 7, 11, 13}, 6)
	require.NoError(t, err)
	idx, err := main.Constant([]int32{0, 0, 0, 1, 1, 3}, 6, 1)
	require.NoError(t, err)
	y, err := main.ScatterSum(init, idx, flat, 1, nil, []int{0}, []int{0}, true, false)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{y}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	got := make([]int64, 5)
	require.NoError(t, b.BufferToFlatData(outs[0], got))

	require.Equal(t, []int64{1 + 3 + 5, 7 + 11, 0, 13, 0}, got)
}

// Exercises int64 scatter fast path (many updates, radix + segmented scan).
func TestScatterSumI64ManyUpdates(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	const n = 2500
	const numOut = 4
	idx := make([]int32, n)
	flat := make([]int64, n)
	want := make([]int64, numOut)
	for i := 0; i < n; i++ {
		idx[i] = int32(i % numOut)
		flat[i] = int64(i + 1)
		want[idx[i]] += flat[i]
	}

	builder := b.Builder("scatter_sum_i64_many")
	main := builder.Main()
	init, err := main.Constant(make([]int64, numOut), numOut)
	require.NoError(t, err)
	updates, err := main.Constant(flat, n)
	require.NoError(t, err)
	ind, err := main.Constant(idx, n, 1)
	require.NoError(t, err)
	y, err := main.ScatterSum(init, ind, updates, 1, nil, []int{0}, []int{0}, true, false)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{y}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	got := make([]int64, numOut)
	require.NoError(t, b.BufferToFlatData(outs[0], got))

	require.Equal(t, want, got)
}

func TestFusedQuantizedDenseF16(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("qdense_f16")
	main := builder.Main()
	x, err := main.Constant([]float16.Float16{
		f16(1), f16(0), f16(0), f16(0),
		f16(0), f16(1), f16(0), f16(0),
	}, 2, 4)
	require.NoError(t, err)
	w, err := main.Constant([]int8{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	}, 4, 3)
	require.NoError(t, err)
	s, err := main.Constant([]float32{1, 1, 1, 1}, 4, 1)
	require.NoError(t, err)
	quant := &backends.Quantization{
		Scheme:    backends.QuantLinear,
		Scale:     s,
		BlockAxis: 1,
		BlockSize: 3,
	}

	y, err := main.FusedQuantizedDense(x, w, nil, quant, backends.ActivationNone)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{y}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	got := make([]float16.Float16, 6)
	require.NoError(t, b.BufferToFlatData(outs[0], got))

	want := []float32{1, 2, 3, 4, 5, 6}

	for i := range want {
		require.InDelta(t, want[i], got[i].Float32(), 5e-2, "i=%d", i)
	}
}

func TestFusedDenseRelu(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("fused_dense")
	main := builder.Main()
	x, err := main.Parameter("x", shapes.Make(dtypes.Float32, 1, 2), nil)
	require.NoError(t, err)
	w, err := main.Constant([]float32{1, 0, 0, 1}, 2, 2)
	require.NoError(t, err)
	y, err := main.FusedDense(x, w, nil, backends.ActivationRelu)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{y}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	in, err := b.BufferFromFlatData(0, []float32{-1, 2}, shapes.Make(dtypes.Float32, 1, 2))
	require.NoError(t, err)
	outs, err := exe.Execute([]backends.Buffer{in}, []bool{false}, 0)
	require.NoError(t, err)

	out := make([]float32, 2)
	require.NoError(t, b.BufferToFlatData(outs[0], out))

	require.Equal(t, []float32{0, 2}, out)
}

func TestFusedDenseSilu(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("fused_dense_silu")
	main := builder.Main()
	x, err := main.Parameter("x", shapes.Make(dtypes.Float32, 1, 2), nil)
	require.NoError(t, err)
	w, err := main.Constant([]float32{1, 0, 0, 1}, 2, 2)
	require.NoError(t, err)
	y, err := main.FusedDense(x, w, nil, backends.ActivationSilu)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{y}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	in, err := b.BufferFromFlatData(0, []float32{1, -1}, shapes.Make(dtypes.Float32, 1, 2))
	require.NoError(t, err)
	outs, err := exe.Execute([]backends.Buffer{in}, []bool{false}, 0)
	require.NoError(t, err)

	out := make([]float32, 2)
	require.NoError(t, b.BufferToFlatData(outs[0], out))

	require.InDelta(t, 1.0/(1.0+math.Exp(-1.0)), float64(out[0]), 1e-5)
	require.InDelta(t, -1.0/(1.0+math.Exp(1.0)), float64(out[1]), 1e-5)
}

func TestFusedAttentionQKVProjectionFusedKernel(t *testing.T) {
	b, err := New("")
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	defer b.Finalize()

	builder := b.Builder("qkv_fused")
	main := builder.Main()
	x, err := main.Constant([]float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
	}, 2, 4)
	require.NoError(t, err)
	wdat := make([]float32, 4*7)
	for i := range wdat {
		wdat[i] = float32(i) * 0.01
	}
	w, err := main.Constant(wdat, 4, 7)
	require.NoError(t, err)
	q, k, v, err := main.FusedAttentionQKVProjection(x, w, nil, nil, nil, 3, 2)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{q, k, v}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)
	require.Len(t, outs, 3)

	sh0, err := b.BufferShape(outs[0])
	require.NoError(t, err)
	sh1, err := b.BufferShape(outs[1])
	require.NoError(t, err)
	sh2, err := b.BufferShape(outs[2])
	require.NoError(t, err)

	require.Equal(t, shapes.Make(dtypes.Float32, 2, 3), sh0)
	require.Equal(t, shapes.Make(dtypes.Float32, 2, 2), sh1)
	require.Equal(t, shapes.Make(dtypes.Float32, 2, 2), sh2)

	qf := make([]float32, 6)
	require.NoError(t, b.BufferToFlatData(outs[0], qf))
	require.True(t, len(qf) == 6 && qf[0] == qf[0]) // finite
}

func TestExecuteInputShapeMismatch(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("param_neg")
	main := builder.Main()
	p, err := main.Parameter("x", shapes.Make(dtypes.Float32, 3), nil)
	require.NoError(t, err)
	n, err := main.Neg(p)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{n}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	wrong, err := b.BufferFromFlatData(0, []float32{1, 2, 3, 4}, shapes.Make(dtypes.Float32, 4))
	require.NoError(t, err)
	_, err = exe.Execute([]backends.Buffer{wrong}, []bool{false}, 0)
	require.Error(t, err)
}

func TestAddBroadcastsSameRankShapeOneDims(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("add_broadcast")
	main := builder.Main()
	a, err := main.Constant([]float32{1, 2, 3}, 3, 1)
	require.NoError(t, err)
	bv, err := main.Constant([]float32{10, 20}, 1, 2)
	require.NoError(t, err)
	sum, err := main.Add(a, bv)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{sum}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	got := make([]float32, 6)
	require.NoError(t, b.BufferToFlatData(outs[0], got))

	require.Equal(t, []float32{11, 21, 12, 22, 13, 23}, got)
}

func TestAddRejectsDifferentRankBroadcasting(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("add_rank_mismatch")
	main := builder.Main()
	a, err := main.Constant([]float32{1, 2}, 2, 1)
	require.NoError(t, err)
	bv, err := main.Constant([]float32{1, 2}, 2)
	require.NoError(t, err)
	_, err = main.Add(a, bv)
	require.Error(t, err)
}

func TestEqualTotalOrderFloat32Executes(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("total_order_eq")
	main := builder.Main()
	a, err := main.Constant([]float32{1, 2, 3}, 3)
	require.NoError(t, err)
	bv, err := main.Constant([]float32{1, 2, 4}, 3)
	require.NoError(t, err)
	eq, err := main.EqualTotalOrder(a, bv)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{eq}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)
	require.Len(t, outs, 1)

	out := make([]bool, 3)
	require.NoError(t, b.BufferToFlatData(outs[0], out))
	require.Equal(t, []bool{true, true, false}, out)
}

func TestWhereBroadcastsScalarPredicateAndValue(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("where_broadcast")
	main := builder.Main()
	pred, err := main.Constant([]bool{true})
	require.NoError(t, err)
	onTrue, err := main.Constant([]int32{1, 2, 3, 4}, 2, 2)
	require.NoError(t, err)
	onFalse, err := main.Constant([]int32{9})
	require.NoError(t, err)
	out, err := main.Where(pred, onTrue, onFalse)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{out}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	got := make([]int32, 4)
	require.NoError(t, b.BufferToFlatData(outs[0], got))

	require.Equal(t, []int32{1, 2, 3, 4}, got)
}

func TestLogicalAndBroadcastsBoolInputs(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("logical_and_broadcast")
	main := builder.Main()
	a, err := main.Constant([]bool{true, false, true}, 3, 1)
	require.NoError(t, err)
	bv, err := main.Constant([]bool{true, false}, 1, 2)
	require.NoError(t, err)
	out, err := main.LogicalAnd(a, bv)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{out}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	got := make([]bool, 6)
	require.NoError(t, b.BufferToFlatData(outs[0], got))

	require.Equal(t, []bool{true, false, false, false, true, false}, got)
}

func TestBitwiseOrBroadcastsUint32Inputs(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("bitwise_or_broadcast")
	main := builder.Main()
	a, err := main.Constant([]uint32{0b1100, 0b0011}, 2, 1)
	require.NoError(t, err)
	bv, err := main.Constant([]uint32{0b1010, 0b0101}, 1, 2)
	require.NoError(t, err)
	out, err := main.BitwiseOr(a, bv)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{out}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	got := make([]uint32, 4)
	require.NoError(t, b.BufferToFlatData(outs[0], got))

	require.Equal(t, []uint32{0b1110, 0b1101, 0b1011, 0b0111}, got)
}

func TestBroadcastReplicatesInt64Scalar(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("broadcast_i64_scalar")
	main := builder.Main()
	x, err := main.Constant([]int64{7})
	require.NoError(t, err)
	y, err := main.BroadcastInDim(x, shapes.Make(dtypes.Int64, 2, 3), nil)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{y}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	got := make([]int64, 6)
	require.NoError(t, b.BufferToFlatData(outs[0], got))

	require.Equal(t, []int64{7, 7, 7, 7, 7, 7}, got)
}

func TestBroadcastAddsPrefixDims(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("broadcast_prefix_dims")
	main := builder.Main().(*Function)
	x, err := main.Constant([]int64{1, 2}, 2)
	require.NoError(t, err)
	y, err := main.Broadcast(x, 3)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{y}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	got := make([]int64, 6)
	require.NoError(t, b.BufferToFlatData(outs[0], got))

	require.Equal(t, []int64{1, 2, 1, 2, 1, 2}, got)
}

func TestBroadcastRejectsNonPositivePrefixDim(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("broadcast_bad_prefix")
	main := builder.Main().(*Function)
	x, err := main.Constant([]float32{1, 2}, 2)
	require.NoError(t, err)
	_, err = main.Broadcast(x, 0)
	require.Error(t, err)
}

func TestBroadcastInDimExpandsUint8Axis(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("broadcast_in_dim_u8")
	main := builder.Main()
	x, err := main.Constant([]uint8{1, 2, 3}, 3, 1)
	require.NoError(t, err)
	y, err := main.BroadcastInDim(x, shapes.Make(dtypes.Uint8, 3, 2), []int{0, 1})
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{y}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	got := make([]uint8, 6)
	require.NoError(t, b.BufferToFlatData(outs[0], got))

	require.Equal(t, []uint8{1, 1, 2, 2, 3, 3}, got)
}

func TestWhereBroadcastsScalarUint8Value(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("where_broadcast_u8")
	main := builder.Main()
	pred, err := main.Constant([]bool{true, false, false, true}, 2, 2)
	require.NoError(t, err)
	onTrue, err := main.Constant([]uint8{1, 2, 3, 4}, 2, 2)
	require.NoError(t, err)
	onFalse, err := main.Constant([]uint8{9})
	require.NoError(t, err)
	out, err := main.Where(pred, onTrue, onFalse)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{out}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	got := make([]uint8, 4)
	require.NoError(t, b.BufferToFlatData(outs[0], got))

	require.Equal(t, []uint8{1, 9, 9, 4}, got)
}

func TestSliceBoolTensor(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("slice_bool")
	main := builder.Main()
	x, err := main.Constant([]bool{true, false, true, false, true, true}, 2, 3)
	require.NoError(t, err)
	y, err := main.Slice(x, []int{0, 1}, []int{2, 3}, []int{1, 1})
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{y}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	got := make([]bool, 4)
	require.NoError(t, b.BufferToFlatData(outs[0], got))

	require.Equal(t, []bool{false, true, true, true}, got)
}

func TestDynamicUpdateSlicePackedStartIndices(t *testing.T) {
	b, err := New("")
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	defer b.Finalize()

	builder := b.Builder("dus_packed")
	main := builder.Main()
	operand, err := main.Constant([]float32{0, 1, 2, 3, 4, 5}, 3, 2)
	require.NoError(t, err)
	updates, err := main.Constant([]float32{100, 100}, 2, 1)
	require.NoError(t, err)
	start, err := main.Constant([]int32{0, 1}, 2)
	require.NoError(t, err)
	out, err := main.DynamicUpdateSlice(operand, updates, []backends.Value{start})
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{out}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	got := make([]float32, 6)
	require.NoError(t, b.BufferToFlatData(outs[0], got))
	require.InDeltaSlice(t, []float32{0, 100, 2, 100, 4, 5}, got, 1e-6)
}

func TestDynamicUpdateSliceScalarStarts(t *testing.T) {
	b, err := New("")
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	defer b.Finalize()

	builder := b.Builder("dus_scalars")
	main := builder.Main()
	operand, err := main.Constant([]float32{0, 1, 2, 3, 4, 5}, 3, 2)
	require.NoError(t, err)
	updates, err := main.Constant([]float32{100, 100}, 2, 1)
	require.NoError(t, err)
	s0, err := main.Constant([]int32{0})
	require.NoError(t, err)
	s1, err := main.Constant([]int32{1})
	require.NoError(t, err)
	out, err := main.DynamicUpdateSlice(operand, updates, []backends.Value{s0, s1})
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{out}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	got := make([]float32, 6)
	require.NoError(t, b.BufferToFlatData(outs[0], got))
	require.InDeltaSlice(t, []float32{0, 100, 2, 100, 4, 5}, got, 1e-6)
}

func TestShiftLeftInt32(t *testing.T) {
	b, err := New("")
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	defer b.Finalize()

	builder := b.Builder("shift_i32")
	main := builder.Main()
	x, err := main.Constant([]int32{1, 2, -8, -1}, 4)
	require.NoError(t, err)
	one, err := main.Constant([]int32{1, 1, 1, 1}, 4)
	require.NoError(t, err)
	y, err := main.ShiftLeft(x, one)
	require.NoError(t, err)
	sra, err := main.ShiftRightArithmetic(x, one)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{y, sra}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	gotL := make([]int32, 4)
	require.NoError(t, b.BufferToFlatData(outs[0], gotL))
	require.Equal(t, []int32{2, 4, -16, -2}, gotL)

	gotA := make([]int32, 4)
	require.NoError(t, b.BufferToFlatData(outs[1], gotA))
	require.Equal(t, []int32{0, 1, -4, -1}, gotA)
}

func TestShiftLeftInt16Uint8(t *testing.T) {
	b, err := New("")
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	defer b.Finalize()

	builder := b.Builder("shift_narrow_int")
	main := builder.Main()
	x16, err := main.Constant([]int16{1, 2, -8, -1}, 4)
	require.NoError(t, err)
	one16, err := main.Constant([]int16{1, 1, 1, 1}, 4)
	require.NoError(t, err)
	y16, err := main.ShiftLeft(x16, one16)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{y16}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	got := make([]int16, 4)
	require.NoError(t, b.BufferToFlatData(outs[0], got))
	require.Equal(t, []int16{2, 4, -16, -2}, got)

	builder2 := b.Builder("shift_u8")
	main2 := builder2.Main()
	x8, err := main2.Constant([]uint8{1, 128, 255, 8}, 4)
	require.NoError(t, err)
	one8, err := main2.Constant([]uint8{1, 1, 1, 2}, 4)
	require.NoError(t, err)
	y8, err := main2.ShiftLeft(x8, one8)
	require.NoError(t, err)
	srl, err := main2.ShiftRightLogical(x8, one8)
	require.NoError(t, err)
	require.NoError(t, main2.Return([]backends.Value{y8, srl}, nil))

	exe2, err := builder2.Compile()
	require.NoError(t, err)
	defer exe2.Finalize()

	outs2, err := exe2.Execute(nil, nil, 0)
	require.NoError(t, err)

	gotL := make([]uint8, 4)
	require.NoError(t, b.BufferToFlatData(outs2[0], gotL))
	require.Equal(t, []uint8{2, 0, 254, 32}, gotL)
	gotR := make([]uint8, 4)
	require.NoError(t, b.BufferToFlatData(outs2[1], gotR))
	require.Equal(t, []uint8{0, 64, 127, 2}, gotR)
}

func TestShiftScalarBroadcastInt16(t *testing.T) {
	b, err := New("")
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	defer b.Finalize()

	builder := b.Builder("shift_bc_i16")
	main := builder.Main()
	x, err := main.Constant([]int16{2, 4, 8}, 3)
	require.NoError(t, err)
	s, err := main.Constant([]int16{1})
	require.NoError(t, err)
	y, err := main.ShiftLeft(x, s)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{y}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)
	got := make([]int16, 3)
	require.NoError(t, b.BufferToFlatData(outs[0], got))
	require.Equal(t, []int16{4, 8, 16}, got)
}

func TestShiftUint16ArithmeticRight(t *testing.T) {
	b, err := New("")
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	defer b.Finalize()

	builder := b.Builder("shift_u16_sra")
	main := builder.Main()
	x, err := main.Constant([]uint16{100, 7, 1}, 3)
	require.NoError(t, err)
	k, err := main.Constant([]uint16{1, 1, 0}, 3)
	require.NoError(t, err)
	y, err := main.ShiftRightArithmetic(x, k)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{y}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)
	got := make([]uint16, 3)
	require.NoError(t, b.BufferToFlatData(outs[0], got))
	require.Equal(t, []uint16{50, 3, 1}, got)
}

func TestShiftRightLogicalSignedInt16(t *testing.T) {
	b, err := New("")
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	defer b.Finalize()

	builder := b.Builder("shift_srl_i16")
	main := builder.Main()
	x, err := main.Constant([]int16{-8, -1}, 2)
	require.NoError(t, err)
	one, err := main.Constant([]int16{1, 1}, 2)
	require.NoError(t, err)
	y, err := main.ShiftRightLogical(x, one)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{y}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	logicalRightInt16 := func(v int16, bits uint16) int16 {
		return int16(uint16(v) >> bits)
	}
	want := []int16{logicalRightInt16(-8, 1), logicalRightInt16(-1, 1)}

	got := make([]int16, 2)
	require.NoError(t, b.BufferToFlatData(outs[0], got))
	require.Equal(t, want, got)
}

func TestShiftInt64Uint64NativeWidth(t *testing.T) {
	b, err := New("")
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	defer b.Finalize()

	large := int64(0x100000003)
	builder := b.Builder("shift_i64_native")
	main := builder.Main()
	x, err := main.Constant([]int64{7, -9, large}, 3)
	require.NoError(t, err)
	one, err := main.Constant([]int64{1, 1, 1}, 3)
	require.NoError(t, err)
	y, err := main.ShiftLeft(x, one)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{y}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)
	got := make([]int64, 3)
	require.NoError(t, b.BufferToFlatData(outs[0], got))
	require.Equal(t, []int64{14, -18, int64(0x200000006)}, got)

	builder2 := b.Builder("shift_u64_wide")
	main2 := builder2.Main()
	uh := uint64(1) << 63
	x2, err := main2.Constant([]uint64{uh, 100}, 2)
	require.NoError(t, err)
	k2, err := main2.Constant([]uint64{1, 2}, 2)
	require.NoError(t, err)
	y2, err := main2.ShiftRightLogical(x2, k2)
	require.NoError(t, err)
	require.NoError(t, main2.Return([]backends.Value{y2}, nil))

	exe2, err := builder2.Compile()
	require.NoError(t, err)
	defer exe2.Finalize()

	outs2, err := exe2.Execute(nil, nil, 0)
	require.NoError(t, err)
	got2 := make([]uint64, 2)
	require.NoError(t, b.BufferToFlatData(outs2[0], got2))
	require.Equal(t, []uint64{(uint64(1) << 62), 25}, got2)
}

func TestShiftUint64Small(t *testing.T) {
	b, err := New("")
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	defer b.Finalize()

	builder := b.Builder("shift_u64")
	main := builder.Main()
	x, err := main.Constant([]uint64{100, 1}, 2)
	require.NoError(t, err)
	k, err := main.Constant([]uint64{2, 0}, 2)
	require.NoError(t, err)
	y, err := main.ShiftLeft(x, k)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{y}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)
	got := make([]uint64, 2)
	require.NoError(t, b.BufferToFlatData(outs[0], got))
	require.Equal(t, []uint64{400, 1}, got)
}

func TestShiftLeftRejectsFloat(t *testing.T) {
	b, err := New("")
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	defer b.Finalize()

	builder := b.Builder("shift_float")
	main := builder.Main()
	x, err := main.Constant([]float32{1}, 1)
	require.NoError(t, err)
	y, err := main.Constant([]float32{1}, 1)
	require.NoError(t, err)
	_, err = main.ShiftLeft(x, y)
	require.Error(t, err)
	require.Contains(t, err.Error(), "integer")
}

func TestDynamicSlicePackedStarts(t *testing.T) {
	b, err := New("")
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	defer b.Finalize()

	builder := b.Builder("dyn_slice")
	main := builder.Main()
	x, err := main.Constant([]float32{0, 1, 2, 3, 4, 5}, 3, 2)
	require.NoError(t, err)
	start, err := main.Constant([]int32{1, 0}, 2)
	require.NoError(t, err)
	y, err := main.DynamicSlice(x, []backends.Value{start}, []int{2, 2})
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{y}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	got := make([]float32, 4)
	require.NoError(t, b.BufferToFlatData(outs[0], got))
	require.InDeltaSlice(t, []float32{2, 3, 4, 5}, got, 1e-6)
}

func TestDynamicSliceStartAfterGPUOp(t *testing.T) {
	b, err := New("")
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	defer b.Finalize()

	builder := b.Builder("dyn_slice_gpu_start")
	main := builder.Main()
	x, err := main.Constant([]float32{10, 20, 30, 40, 50}, 5)
	require.NoError(t, err)
	one, err := main.Constant([]int32{1})
	require.NoError(t, err)
	two, err := main.Constant([]int32{2})
	require.NoError(t, err)
	start, err := main.Add(one, two)
	require.NoError(t, err)
	y, err := main.DynamicSlice(x, []backends.Value{start}, []int{2})
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{y}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	got := make([]float32, 2)
	require.NoError(t, b.BufferToFlatData(outs[0], got))
	require.InDeltaSlice(t, []float32{40, 50}, got, 1e-6)
}

func TestSelectAndScatterMaxPoolGradNCHW(t *testing.T) {
	b, err := New("")
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	defer b.Finalize()

	builder := b.Builder("sas_max")
	main := builder.Main()
	// N=1,C=1,H=2,W=2 values; 2x2 max pool, stride 2, no pad -> 1 output
	op, err := main.Constant([]float32{1, 3, 2, 4}, 1, 1, 2, 2)
	require.NoError(t, err)
	dy, err := main.Constant([]float32{10}, 1, 1, 1, 1)
	require.NoError(t, err)
	out, err := main.SelectAndScatterMax(
		op, dy,
		[]int{1, 1, 2, 2},
		[]int{1, 1, 2, 2},
		[][2]int{{0, 0}, {0, 0}, {0, 0}, {0, 0}},
	)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{out}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	got := make([]float32, 4)
	require.NoError(t, b.BufferToFlatData(outs[0], got))
	require.InDeltaSlice(t, []float32{0, 0, 0, 10}, got, 1e-5)
}

func TestPadUint8Tensor(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("pad_u8")
	main := builder.Main()
	x, err := main.Constant([]uint8{1, 2}, 2)
	require.NoError(t, err)
	fill, err := main.Constant([]uint8{9})
	require.NoError(t, err)
	y, err := main.Pad(x, fill, backends.PadAxis{Start: 1, End: 2})
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{y}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	got := make([]uint8, 5)
	require.NoError(t, b.BufferToFlatData(outs[0], got))

	require.Equal(t, []uint8{9, 1, 2, 9, 9}, got)
}

func TestReverseInt64Tensor(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("reverse_i64")
	main := builder.Main()
	x, err := main.Constant([]int64{1, 2, 3, 4, 5, 6}, 2, 3)
	require.NoError(t, err)
	y, err := main.Reverse(x, 1)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{y}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	got := make([]int64, 6)
	require.NoError(t, b.BufferToFlatData(outs[0], got))

	require.Equal(t, []int64{3, 2, 1, 6, 5, 4}, got)
}

func TestTransposeUint8Tensor(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("transpose_u8")
	main := builder.Main()
	x, err := main.Constant([]uint8{1, 2, 3, 4, 5, 6}, 2, 3)
	require.NoError(t, err)
	y, err := main.Transpose(x, 1, 0)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{y}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	got := make([]uint8, 6)
	require.NoError(t, b.BufferToFlatData(outs[0], got))

	require.Equal(t, []uint8{1, 4, 2, 5, 3, 6}, got)
}

func TestConcatenateInt16Tensor(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("concat_i16")
	main := builder.Main()
	a, err := main.Constant([]int16{1, 2, 3, 4}, 2, 2)
	require.NoError(t, err)
	c, err := main.Constant([]int16{5, 6}, 2, 1)
	require.NoError(t, err)
	y, err := main.Concatenate(1, a, c)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{y}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	got := make([]int16, 6)
	require.NoError(t, b.BufferToFlatData(outs[0], got))

	require.Equal(t, []int16{1, 2, 5, 3, 4, 6}, got)
}

func TestIotaUint32(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("iota_u32")
	main := builder.Main()
	x, err := main.Iota(shapes.Make(dtypes.Uint32, 2, 4), 1)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{x}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	got := make([]uint32, 8)
	require.NoError(t, b.BufferToFlatData(outs[0], got))

	require.Equal(t, []uint32{0, 1, 2, 3, 0, 1, 2, 3}, got)
}

func TestIotaRejectsScalarShape(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("iota_scalar")
	main := builder.Main()
	_, err = main.Iota(shapes.Make(dtypes.Uint32), 0)
	require.Error(t, err)
}

func TestConvertDTypeUint32ToFloat32(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("convert_u32_to_f32")
	main := builder.Main()
	x, err := main.Constant([]uint32{1, 2, 7}, 3)
	require.NoError(t, err)
	y, err := main.ConvertDType(x, dtypes.Float32)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{y}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	got := make([]float32, 3)
	require.NoError(t, b.BufferToFlatData(outs[0], got))

	require.Equal(t, []float32{1, 2, 7}, got)
}

func TestBFloat16ConvertAddAndDot(t *testing.T) {
	b, err := New("")
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	defer b.Finalize()

	t.Run("add_double", func(t *testing.T) {
		builder := b.Builder("bf16_add")
		main := builder.Main()
		x, err := main.Constant([]float32{1, 2, 3, 4}, 4)
		require.NoError(t, err)
		bf, err := main.ConvertDType(x, dtypes.BFloat16)
		require.NoError(t, err)
		sum, err := main.Add(bf, bf)
		require.NoError(t, err)
		out, err := main.ConvertDType(sum, dtypes.Float32)
		require.NoError(t, err)
		require.NoError(t, main.Return([]backends.Value{out}, nil))

		exe, err := builder.Compile()
		require.NoError(t, err)
		defer exe.Finalize()

		outs, err := exe.Execute(nil, nil, 0)
		require.NoError(t, err)
		got := make([]float32, 4)
		require.NoError(t, b.BufferToFlatData(outs[0], got))
		require.InDeltaSlice(t, []float32{2, 4, 6, 8}, got, 0.15)
	})

	t.Run("dot", func(t *testing.T) {
		builder := b.Builder("bf16_dot")
		main := builder.Main()
		a, err := main.Constant([]float32{1, 2, 3, 4}, 2, 2)
		require.NoError(t, err)
		c, err := main.Constant([]float32{5, 6, 7, 8}, 2, 2)
		require.NoError(t, err)
		aBF, err := main.ConvertDType(a, dtypes.BFloat16)
		require.NoError(t, err)
		cBF, err := main.ConvertDType(c, dtypes.BFloat16)
		require.NoError(t, err)
		d, err := main.DotGeneral(aBF, []int{1}, nil, cBF, []int{0}, nil, backends.DotGeneralConfig{})
		require.NoError(t, err)
		out, err := main.ConvertDType(d, dtypes.Float32)
		require.NoError(t, err)
		require.NoError(t, main.Return([]backends.Value{out}, nil))

		exe, err := builder.Compile()
		require.NoError(t, err)
		defer exe.Finalize()

		outs, err := exe.Execute(nil, nil, 0)
		require.NoError(t, err)
		got := make([]float32, 4)
		require.NoError(t, b.BufferToFlatData(outs[0], got))
		require.InDeltaSlice(t, []float32{19, 22, 43, 50}, got, 1.5)
	})
}

func TestDotGeneralRank2ContractsAxis0(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("dotg_axis0")
	main := builder.Main()
	a, err := main.Constant([]float32{1, 2, 3, 4}, 2, 2)
	require.NoError(t, err)
	bMat, err := main.Constant([]float32{10, 20, 30, 40}, 2, 2)
	require.NoError(t, err)
	d, err := main.DotGeneral(a, []int{0}, nil, bMat, []int{0}, nil, backends.DotGeneralConfig{})
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{d}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)
	got := make([]float32, 4)
	require.NoError(t, b.BufferToFlatData(outs[0], got))
	require.InDeltaSlice(t, []float32{100, 140, 140, 200}, got, 1e-3)
}

func TestConvertDTypeBoolToInt32(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("convert_bool_to_i32")
	main := builder.Main()
	x, err := main.Constant([]bool{true, false, true}, 3)
	require.NoError(t, err)
	y, err := main.ConvertDType(x, dtypes.Int32)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{y}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	got := make([]int32, 3)
	require.NoError(t, b.BufferToFlatData(outs[0], got))

	require.Equal(t, []int32{1, 0, 1}, got)
}

func TestConvertDTypeUint8ToInt16(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("convert_u8_to_i16")
	main := builder.Main()
	x, err := main.Constant([]uint8{1, 2, 255}, 3)
	require.NoError(t, err)
	y, err := main.ConvertDType(x, dtypes.Int16)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{y}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	got := make([]int16, 3)
	require.NoError(t, b.BufferToFlatData(outs[0], got))

	require.Equal(t, []int16{1, 2, 255}, got)
}

func TestReduceProductFloat16(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("reduce_product_f16")
	main := builder.Main()
	x, err := main.Constant([]float16.Float16{f16(1), f16(2), f16(3), f16(4)}, 2, 2)
	require.NoError(t, err)
	y, err := main.ReduceProduct(x, 1)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{y}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	got := make([]float16.Float16, 2)
	require.NoError(t, b.BufferToFlatData(outs[0], got))

	require.Equal(t, []float16.Float16{f16(2), f16(12)}, got)
}

func TestReduceSumInt32(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("reduce_sum_i32")
	main := builder.Main()
	x, err := main.Constant([]int32{1, 2, 3, 4, 5, 6}, 2, 3)
	require.NoError(t, err)
	y, err := main.ReduceSum(x, 1)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{y}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	got := make([]int32, 2)
	require.NoError(t, b.BufferToFlatData(outs[0], got))

	require.Equal(t, []int32{6, 15}, got)
}

func TestReduceMaxUint32(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("reduce_max_u32")
	main := builder.Main()
	x, err := main.Constant([]uint32{1, 8, 3, 4, 2, 6}, 2, 3)
	require.NoError(t, err)
	y, err := main.ReduceMax(x, 1)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{y}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	got := make([]uint32, 2)
	require.NoError(t, b.BufferToFlatData(outs[0], got))

	require.Equal(t, []uint32{8, 6}, got)
}

func TestCallSiblingClosureSameBuilder(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("call_sibling")
	main := builder.Main()

	addOne, err := main.Closure()
	require.NoError(t, err)
	px, err := addOne.Parameter("x", shapes.Make(dtypes.Float32), nil)
	require.NoError(t, err)
	one, err := addOne.Constant([]float32{1})
	require.NoError(t, err)
	s, err := addOne.Add(px, one)
	require.NoError(t, err)
	require.NoError(t, addOne.Return([]backends.Value{s}, nil))

	caller, err := main.Closure()
	require.NoError(t, err)
	py, err := caller.Parameter("y", shapes.Make(dtypes.Float32), nil)
	require.NoError(t, err)
	out, err := caller.Call(addOne, py)
	require.NoError(t, err)
	require.Len(t, out, 1)
	require.NoError(t, caller.Return(out, nil))

	zConst, err := main.Constant([]float32{41})
	require.NoError(t, err)
	outs, err := main.Call(caller, zConst)
	require.NoError(t, err)
	require.Len(t, outs, 1)
	require.NoError(t, main.Return(outs, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	res, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	got := make([]float32, 1)
	require.NoError(t, b.BufferToFlatData(res[0], got))
	require.InEpsilon(t, float32(42), got[0], 1e-5)
}

func TestIfSelectsBranchByPredicate(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	run := func(name string, predVal bool, want float32) {
		t.Helper()
		builder := b.Builder("if_branch_" + name)
		main := builder.Main()
		pred, err := main.Constant([]bool{predVal})
		require.NoError(t, err)
		trueBr, err := main.Closure()
		require.NoError(t, err)
		tConst, err := trueBr.Constant([]float32{10})
		require.NoError(t, err)
		require.NoError(t, trueBr.Return([]backends.Value{tConst}, nil))
		falseBr, err := main.Closure()
		require.NoError(t, err)
		fConst, err := falseBr.Constant([]float32{20})
		require.NoError(t, err)
		require.NoError(t, falseBr.Return([]backends.Value{fConst}, nil))

		outs, err := main.If(pred, trueBr, falseBr)
		require.NoError(t, err)
		require.Len(t, outs, 1)
		require.NoError(t, main.Return(outs, nil))

		exe, err := builder.Compile()
		require.NoError(t, err)
		defer exe.Finalize()

		o, err := exe.Execute(nil, nil, 0)
		require.NoError(t, err)

		v := make([]float32, 1)
		require.NoError(t, b.BufferToFlatData(o[0], v))

		require.Equal(t, want, v[0])
	}

	t.Run("true", func(t *testing.T) { run("true", true, 10) })
	t.Run("false", func(t *testing.T) { run("false", false, 20) })
}

func TestCallDirectClosureMultiplies(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("call_double")
	main := builder.Main()
	doubleFn, err := main.Closure()
	require.NoError(t, err)
	dx, err := doubleFn.Parameter("x", shapes.Make(dtypes.Float32), nil)
	require.NoError(t, err)
	two, err := doubleFn.Constant([]float32{2})
	require.NoError(t, err)
	dy, err := doubleFn.Mul(dx, two)
	require.NoError(t, err)
	require.NoError(t, doubleFn.Return([]backends.Value{dy}, nil))

	px, err := main.Parameter("in", shapes.Make(dtypes.Float32), nil)
	require.NoError(t, err)
	couts, err := main.Call(doubleFn, px)
	require.NoError(t, err)
	require.Len(t, couts, 1)
	require.NoError(t, main.Return([]backends.Value{couts[0]}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	in, err := b.BufferFromFlatData(0, []float32{21}, shapes.Make(dtypes.Float32))
	require.NoError(t, err)
	outs, err := exe.Execute([]backends.Buffer{in}, []bool{false}, 0)
	require.NoError(t, err)

	v := make([]float32, 1)
	require.NoError(t, b.BufferToFlatData(outs[0], v))

	require.Equal(t, float32(42), v[0])
}

func TestCallIdentityIntermediateKeepsInputOwnedByCaller(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("call_identity_ownership")
	main := builder.Main()
	idFn, err := main.Closure()
	require.NoError(t, err)
	dx, err := idFn.Parameter("x", shapes.Make(dtypes.Float32), nil)
	require.NoError(t, err)
	require.NoError(t, idFn.Return([]backends.Value{dx}, nil))

	px, err := main.Parameter("in", shapes.Make(dtypes.Float32), nil)
	require.NoError(t, err)
	couts, err := main.Call(idFn, px)
	require.NoError(t, err)
	require.Len(t, couts, 1)
	neg, err := main.Neg(couts[0])
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{neg}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	inAny, err := b.BufferFromFlatData(0, []float32{7}, shapes.Make(dtypes.Float32))
	require.NoError(t, err)
	inBuf := inAny.(*Buffer)
	require.NotNil(t, inBuf.mtl)

	for range 2 {
		outs, err := exe.Execute([]backends.Buffer{inBuf}, []bool{false}, 0)
		require.NoError(t, err)

		v := make([]float32, 1)
		require.NoError(t, b.BufferToFlatData(outs[0], v))

		require.Equal(t, float32(-7), v[0])
	}

	require.NotNil(t, inBuf.mtl)
}

func TestWhileIncrementsToLimitFloat32(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("while_count_f32")
	main := builder.Main()

	cond, err := main.Closure()
	require.NoError(t, err)
	cc, err := cond.Parameter("c", shapes.Make(dtypes.Float32), nil)
	require.NoError(t, err)
	lim, err := cond.Constant([]float32{5})
	require.NoError(t, err)
	clt, err := cond.LessThan(cc, lim)
	require.NoError(t, err)
	require.NoError(t, cond.Return([]backends.Value{clt}, nil))

	body, err := main.Closure()
	require.NoError(t, err)
	bc, err := body.Parameter("c", shapes.Make(dtypes.Float32), nil)
	require.NoError(t, err)
	one, err := body.Constant([]float32{1})
	require.NoError(t, err)
	bn, err := body.Add(bc, one)
	require.NoError(t, err)
	require.NoError(t, body.Return([]backends.Value{bn}, nil))

	initC, err := main.Constant([]float32{0})
	require.NoError(t, err)
	wout, err := main.While(cond, body, initC)
	require.NoError(t, err)
	require.Len(t, wout, 1)
	require.NoError(t, main.Return(wout, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	v := make([]float32, 1)
	require.NoError(t, b.BufferToFlatData(outs[0], v))

	require.Equal(t, float32(5), v[0])
}

func TestWhileIncrementsToLimitInt32(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("while_count_i32")
	main := builder.Main()

	cond, err := main.Closure()
	require.NoError(t, err)
	cc, err := cond.Parameter("c", shapes.Make(dtypes.Int32), nil)
	require.NoError(t, err)
	lim, err := cond.Constant([]int32{5})
	require.NoError(t, err)
	clt, err := cond.LessThan(cc, lim)
	require.NoError(t, err)
	require.NoError(t, cond.Return([]backends.Value{clt}, nil))

	body, err := main.Closure()
	require.NoError(t, err)
	bc, err := body.Parameter("c", shapes.Make(dtypes.Int32), nil)
	require.NoError(t, err)
	one, err := body.Constant([]int32{1})
	require.NoError(t, err)
	bn, err := body.Add(bc, one)
	require.NoError(t, err)
	require.NoError(t, body.Return([]backends.Value{bn}, nil))

	initC, err := main.Constant([]int32{0})
	require.NoError(t, err)
	wout, err := main.While(cond, body, initC)
	require.NoError(t, err)
	require.Len(t, wout, 1)
	require.NoError(t, main.Return(wout, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	outs, err := exe.Execute(nil, nil, 0)
	require.NoError(t, err)

	v := make([]int32, 1)
	require.NoError(t, b.BufferToFlatData(outs[0], v))

	require.Equal(t, int32(5), v[0])
}

func TestSortFloatAxis0(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("sort_f32")
	main := builder.Main()

	comp, err := main.Closure()
	require.NoError(t, err)
	lhs, err := comp.Parameter("lhs", shapes.Make(dtypes.Float32), nil)
	require.NoError(t, err)
	rhs, err := comp.Parameter("rhs", shapes.Make(dtypes.Float32), nil)
	require.NoError(t, err)
	lt, err := comp.LessThan(lhs, rhs)
	require.NoError(t, err)
	require.NoError(t, comp.Return([]backends.Value{lt}, nil))

	in, err := main.Parameter("data", shapes.Make(dtypes.Float32, 5), nil)
	require.NoError(t, err)
	sout, err := main.Sort(comp, 0, false, in)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{sout[0]}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	inBuf, err := b.BufferFromFlatData(0, []float32{5, 2, 8, 1, 3}, shapes.Make(dtypes.Float32, 5))
	require.NoError(t, err)
	outs, err := exe.Execute([]backends.Buffer{inBuf}, []bool{false}, 0)
	require.NoError(t, err)

	got := make([]float32, 5)
	require.NoError(t, b.BufferToFlatData(outs[0], got))

	require.Equal(t, []float32{1, 2, 3, 5, 8}, got)
}

func TestSortStableGPUWithLexicographicTieBreak(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("sort_stable_lex")
	main := builder.Main()

	comp, err := main.Closure()
	require.NoError(t, err)
	lk, err := comp.Parameter("lk", shapes.Make(dtypes.Float32), nil)
	require.NoError(t, err)
	rk, err := comp.Parameter("rk", shapes.Make(dtypes.Float32), nil)
	require.NoError(t, err)
	ltb, err := comp.Parameter("lt", shapes.Make(dtypes.Float32), nil)
	require.NoError(t, err)
	rtb, err := comp.Parameter("rt", shapes.Make(dtypes.Float32), nil)
	require.NoError(t, err)
	kLess, err := comp.LessThan(lk, rk)
	require.NoError(t, err)
	rLess, err := comp.LessThan(rk, lk)
	require.NoError(t, err)
	nkLess, err := comp.LogicalNot(kLess)
	require.NoError(t, err)
	nrLess, err := comp.LogicalNot(rLess)
	require.NoError(t, err)
	eqK, err := comp.LogicalAnd(nkLess, nrLess)
	require.NoError(t, err)
	tLess, err := comp.LessThan(ltb, rtb)
	require.NoError(t, err)
	tiePart, err := comp.LogicalAnd(eqK, tLess)
	require.NoError(t, err)
	out, err := comp.LogicalOr(kLess, tiePart)
	require.NoError(t, err)
	require.NoError(t, comp.Return([]backends.Value{out}, nil))

	keys, err := main.Parameter("keys", shapes.Make(dtypes.Float32, 4), nil)
	require.NoError(t, err)
	ties, err := main.Parameter("ties", shapes.Make(dtypes.Float32, 4), nil)
	require.NoError(t, err)
	sout, err := main.Sort(comp, 0, true, keys, ties)
	require.NoError(t, err)
	require.Len(t, sout, 2)
	require.NoError(t, main.Return([]backends.Value{sout[0], sout[1]}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	// Two equal keys 1.0 at input positions 1 and 2; lower tie (10) must precede 20 for stable order.
	kb, err := b.BufferFromFlatData(0, []float32{2, 1, 1, 3}, shapes.Make(dtypes.Float32, 4))
	require.NoError(t, err)
	tb, err := b.BufferFromFlatData(0, []float32{0, 10, 20, 30}, shapes.Make(dtypes.Float32, 4))
	require.NoError(t, err)
	outs, err := exe.Execute([]backends.Buffer{kb, tb}, []bool{false, false}, 0)
	require.NoError(t, err)

	gotK := make([]float32, 4)
	gotT := make([]float32, 4)
	require.NoError(t, b.BufferToFlatData(outs[0], gotK))
	require.NoError(t, b.BufferToFlatData(outs[1], gotT))

	require.Equal(t, []float32{1, 1, 2, 3}, gotK)
	require.Equal(t, []float32{10, 20, 0, 30}, gotT)
}

func TestSortUnstableAxisLargerThan4096NonPow2(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	const n = 5003

	builder := b.Builder("sort_large_np2")
	main := builder.Main()

	comp, err := main.Closure()
	require.NoError(t, err)
	lhs, err := comp.Parameter("lhs", shapes.Make(dtypes.Float32), nil)
	require.NoError(t, err)
	rhs, err := comp.Parameter("rhs", shapes.Make(dtypes.Float32), nil)
	require.NoError(t, err)
	lt, err := comp.LessThan(lhs, rhs)
	require.NoError(t, err)
	require.NoError(t, comp.Return([]backends.Value{lt}, nil))

	in, err := main.Parameter("data", shapes.Make(dtypes.Float32, n), nil)
	require.NoError(t, err)
	sout, err := main.Sort(comp, 0, false, in)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{sout[0]}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	rng := rand.New(rand.NewPCG(1, 2))
	data := make([]float32, n)

	for i := range data {
		data[i] = rng.Float32()*200 - 100
	}

	want := slices.Clone(data)
	slices.Sort(want)

	inBuf, err := b.BufferFromFlatData(0, data, shapes.Make(dtypes.Float32, n))
	require.NoError(t, err)
	outs, err := exe.Execute([]backends.Buffer{inBuf}, []bool{false}, 0)
	require.NoError(t, err)

	got := make([]float32, n)
	require.NoError(t, b.BufferToFlatData(outs[0], got))
	require.Equal(t, want, got)
}

func TestSortStableLexicographicAxis2500(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	const n = 2500

	builder := b.Builder("sort_stable_large")
	main := builder.Main()

	comp, err := main.Closure()
	require.NoError(t, err)
	lk, err := comp.Parameter("lk", shapes.Make(dtypes.Float32), nil)
	require.NoError(t, err)
	rk, err := comp.Parameter("rk", shapes.Make(dtypes.Float32), nil)
	require.NoError(t, err)
	ltb, err := comp.Parameter("lt", shapes.Make(dtypes.Float32), nil)
	require.NoError(t, err)
	rtb, err := comp.Parameter("rt", shapes.Make(dtypes.Float32), nil)
	require.NoError(t, err)
	kLess, err := comp.LessThan(lk, rk)
	require.NoError(t, err)
	rLess, err := comp.LessThan(rk, lk)
	require.NoError(t, err)
	nkLess, err := comp.LogicalNot(kLess)
	require.NoError(t, err)
	nrLess, err := comp.LogicalNot(rLess)
	require.NoError(t, err)
	eqK, err := comp.LogicalAnd(nkLess, nrLess)
	require.NoError(t, err)
	tLess, err := comp.LessThan(ltb, rtb)
	require.NoError(t, err)
	tiePart, err := comp.LogicalAnd(eqK, tLess)
	require.NoError(t, err)
	out, err := comp.LogicalOr(kLess, tiePart)
	require.NoError(t, err)
	require.NoError(t, comp.Return([]backends.Value{out}, nil))

	keys, err := main.Parameter("keys", shapes.Make(dtypes.Float32, n), nil)
	require.NoError(t, err)
	ties, err := main.Parameter("ties", shapes.Make(dtypes.Float32, n), nil)
	require.NoError(t, err)
	sout, err := main.Sort(comp, 0, true, keys, ties)
	require.NoError(t, err)
	require.Len(t, sout, 2)
	require.NoError(t, main.Return([]backends.Value{sout[0], sout[1]}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	kb := make([]float32, n)
	tb := make([]float32, n)

	for i := range kb {
		kb[i] = float32((i * 17) % 41)
		tb[i] = float32(i)
	}

	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}

	sort.SliceStable(idx, func(i, j int) bool {
		ii, jj := idx[i], idx[j]

		if kb[ii] != kb[jj] {
			return kb[ii] < kb[jj]
		}

		return tb[ii] < tb[jj]
	})

	wantK := make([]float32, n)
	wantT := make([]float32, n)

	for pos, orig := range idx {
		wantK[pos] = kb[orig]
		wantT[pos] = tb[orig]
	}

	kBuf, err := b.BufferFromFlatData(0, kb, shapes.Make(dtypes.Float32, n))
	require.NoError(t, err)
	tBuf, err := b.BufferFromFlatData(0, tb, shapes.Make(dtypes.Float32, n))
	require.NoError(t, err)
	outs, err := exe.Execute([]backends.Buffer{kBuf, tBuf}, []bool{false, false}, 0)
	require.NoError(t, err)

	gotK := make([]float32, n)
	gotT := make([]float32, n)
	require.NoError(t, b.BufferToFlatData(outs[0], gotK))
	require.NoError(t, b.BufferToFlatData(outs[1], gotT))
	require.Equal(t, wantK, gotK)
	require.Equal(t, wantT, gotT)
}

func TestTransposeIdentityRank9Float32(t *testing.T) {
	b, err := New("")

	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	defer b.Finalize()

	builder := b.Builder("tp_rank9")
	main := builder.Main()
	dims := make([]int, 9)
	for i := range dims {
		dims[i] = 2
	}

	sh := shapes.Make(dtypes.Float32, dims...)
	p, err := main.Parameter("x", sh, nil)
	require.NoError(t, err)
	tr, err := main.Transpose(p, 0, 1, 2, 3, 4, 5, 6, 7, 8)
	require.NoError(t, err)
	require.NoError(t, main.Return([]backends.Value{tr}, nil))

	exe, err := builder.Compile()
	require.NoError(t, err)
	defer exe.Finalize()

	n := sh.Size()
	inFlat := make([]float32, n)
	for i := range inFlat {
		inFlat[i] = float32(i)
	}

	inBuf, err := b.BufferFromFlatData(0, inFlat, sh)
	require.NoError(t, err)
	outs, err := exe.Execute([]backends.Buffer{inBuf}, []bool{false}, 0)
	require.NoError(t, err)

	got := make([]float32, n)
	require.NoError(t, b.BufferToFlatData(outs[0], got))

	require.Equal(t, inFlat, got)
}
