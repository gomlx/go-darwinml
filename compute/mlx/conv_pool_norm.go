// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mlx

import (
	"math"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/go-darwinml/internal/mlxbridge"
	"github.com/pkg/errors"
)

// makeScalarConst creates a scalar constant node of the given dtype.
func (f *Function) makeScalarConst(val float64, dt dtypes.DType) (*graphNode, error) {
	var v compute.Value
	var err error
	switch dt {
	case dtypes.Float32:
		v, err = f.Constant([]float32{float32(val)})
	case dtypes.Float16:
		v, err = f.Constant([]float16.Float16{float16.FromFloat32(float32(val))})
	case dtypes.BFloat16:
		v, err = f.Constant([]bfloat16.BFloat16{bfloat16.FromFloat32(float32(val))})
	case dtypes.Int8:
		v, err = f.Constant([]int8{int8(val)})
	case dtypes.Int16:
		v, err = f.Constant([]int16{int16(val)})
	case dtypes.Int32:
		v, err = f.Constant([]int32{int32(val)})
	case dtypes.Int64:
		v, err = f.Constant([]int64{int64(val)})
	case dtypes.Uint8:
		v, err = f.Constant([]uint8{uint8(val)})
	case dtypes.Uint16:
		v, err = f.Constant([]uint16{uint16(val)})
	case dtypes.Uint32:
		v, err = f.Constant([]uint32{uint32(val)})
	case dtypes.Uint64:
		v, err = f.Constant([]uint64{uint64(val)})
	case dtypes.Bool:
		v, err = f.Constant([]bool{val != 0})
	case dtypes.Complex64:
		v, err = f.Constant([]complex64{complex(float32(val), 0)})
	default:
		return nil, errors.Wrapf(compute.ErrNotImplemented, "makeScalarConst: unsupported dtype %s", dt)
	}
	if err != nil {
		return nil, err
	}
	return v.(*graphNode), nil
}

// makeReduceIdentityConst returns the scalar identity element for the given reduction type and dtype.
func (f *Function) makeReduceIdentityConst(reductionType compute.ReduceOpType, dt dtypes.DType) (*graphNode, error) {
	switch reductionType {
	case compute.ReduceOpSum:
		return f.makeScalarConst(0, dt)
	case compute.ReduceOpProduct:
		return f.makeScalarConst(1, dt)
	case compute.ReduceOpMax:
		switch dt {
		case dtypes.Float32, dtypes.Float16, dtypes.BFloat16:
			return f.makeScalarConst(math.Inf(-1), dt)
		case dtypes.Int8:
			return f.makeScalarConst(float64(math.MinInt8), dt)
		case dtypes.Int16:
			return f.makeScalarConst(float64(math.MinInt16), dt)
		case dtypes.Int32:
			return f.makeScalarConst(float64(math.MinInt32), dt)
		case dtypes.Int64:
			v, err := f.Constant([]int64{math.MinInt64})
			if err != nil {
				return nil, err
			}
			return v.(*graphNode), nil
		case dtypes.Uint8, dtypes.Uint16, dtypes.Uint32, dtypes.Uint64, dtypes.Bool:
			return f.makeScalarConst(0, dt)
		default:
			return nil, errors.Wrapf(compute.ErrNotImplemented, "ReduceWindow: Max not supported for dtype %s", dt)
		}
	case compute.ReduceOpMin:
		switch dt {
		case dtypes.Float32, dtypes.Float16, dtypes.BFloat16:
			return f.makeScalarConst(math.Inf(1), dt)
		case dtypes.Int8:
			return f.makeScalarConst(float64(math.MaxInt8), dt)
		case dtypes.Int16:
			return f.makeScalarConst(float64(math.MaxInt16), dt)
		case dtypes.Int32:
			return f.makeScalarConst(float64(math.MaxInt32), dt)
		case dtypes.Int64:
			v, err := f.Constant([]int64{math.MaxInt64})
			if err != nil {
				return nil, err
			}
			return v.(*graphNode), nil
		case dtypes.Uint8:
			return f.makeScalarConst(float64(math.MaxUint8), dt)
		case dtypes.Uint16:
			return f.makeScalarConst(float64(math.MaxUint16), dt)
		case dtypes.Uint32:
			return f.makeScalarConst(float64(math.MaxUint32), dt)
		case dtypes.Uint64:
			v, err := f.Constant([]uint64{math.MaxUint64})
			if err != nil {
				return nil, err
			}
			return v.(*graphNode), nil
		case dtypes.Bool:
			return f.makeScalarConst(1, dt)
		default:
			return nil, errors.Wrapf(compute.ErrNotImplemented, "ReduceWindow: Min not supported for dtype %s", dt)
		}
	default:
		return nil, errors.Errorf("unsupported reduction type %v", reductionType)
	}
}

// ===========================================================================
// Convolution (ConvGeneral)
// ===========================================================================

// ConvGeneral performs a general N-D convolution (1D, 2D, or 3D) with arbitrary axis layouts,
// strides, padding, input/kernel dilations, and channel/batch grouping.
func (f *Function) ConvGeneral(
	input, kernel compute.Value,
	axes compute.ConvolveAxesConfig,
	strides []int,
	paddings [][2]int,
	inputDilations, kernelDilations []int,
	channelGroupCount, batchGroupCount int,
) (compute.Value, error) {
	inputNode, err := f.resolveNode(input)
	if err != nil {
		return nil, errors.Wrap(err, "ConvGeneral: input")
	}
	kernelNode, err := f.resolveNode(kernel)
	if err != nil {
		return nil, errors.Wrap(err, "ConvGeneral: kernel")
	}

	numSpatialDims := len(axes.InputSpatial)
	if strides == nil {
		strides = make([]int, numSpatialDims)
		for i := range strides {
			strides[i] = 1
		}
	}
	if inputDilations == nil {
		inputDilations = make([]int, numSpatialDims)
		for i := range inputDilations {
			inputDilations[i] = 1
		}
	}
	if kernelDilations == nil {
		kernelDilations = make([]int, numSpatialDims)
		for i := range kernelDilations {
			kernelDilations[i] = 1
		}
	}
	if channelGroupCount < 1 {
		channelGroupCount = 1
	}
	if batchGroupCount < 1 {
		batchGroupCount = 1
	}

	outShape, err := shapeinference.ConvGeneral(
		inputNode.shape, kernelNode.shape, axes,
		strides, paddings, inputDilations, kernelDilations,
		channelGroupCount, batchGroupCount)
	if err != nil {
		return nil, errors.Wrap(err, "ConvGeneral")
	}
	outShape = f.nameAnonymousDynamicAxes(outShape, "#conv")

	if !inputNode.shape.DType.IsFloat() {
		return nil, errors.Wrapf(compute.ErrNotImplemented,
			"ConvGeneral: only floating-point dtypes are supported in %q backend, got %s",
			BackendName, inputNode.shape.DType)
	}
	if kernelNode.shape.IsDynamic() || inputNode.shape.Dimensions[axes.InputChannels] == shapes.DynamicDim {
		return nil, errors.Wrapf(compute.ErrNotImplemented,
			"ConvGeneral: dynamic kernel or input channel dimensions are not supported in %q backend", BackendName)
	}
	if numSpatialDims < 1 || numSpatialDims > 3 {
		return nil, errors.Wrapf(compute.ErrNotImplemented,
			"ConvGeneral: only 1D, 2D, and 3D convolutions are supported in %q backend, got %d spatial dims",
			BackendName, numSpatialDims)
	}
	if !outShape.IsDynamic() && outShape.Size() == 0 {
		id, err := f.builder.tape.AddConstant(nil, 0, dtypeToBridgeDType(outShape.DType), outShape.Dimensions)
		if err != nil {
			return nil, errors.Wrap(err, "ConvGeneral: zero-sized output")
		}
		return &graphNode{id: id, shape: outShape, owner: f}, nil
	}

	padLo := make([]int, numSpatialDims)
	padHi := make([]int, numSpatialDims)
	inDil := make([]int, numSpatialDims)
	hasNegativePad := false
	for i := range numSpatialDims {
		inDil[i] = inputDilations[i]
		if paddings != nil && i < len(paddings) {
			padLo[i] = paddings[i][0]
			padHi[i] = paddings[i][1]
			if padLo[i] < 0 || padHi[i] < 0 {
				hasNegativePad = true
			}
		}
	}
	if inputNode.shape.IsDynamic() && (batchGroupCount > 1 || hasNegativePad) {
		return nil, errors.Wrapf(compute.ErrNotImplemented,
			"ConvGeneral: dynamic shapes with batchGroupCount > 1 or negative padding are not supported in %q backend", BackendName)
	}

	// Fast path for weight-gradient convolutions (convVJPWrtKernel) where the output spatial
	// filter size is small (e.g. 1x1, 3x3) and the contracted kernel spatial size is large:
	// use patch extraction + Split-K Matmul (matching MLX's internal conv_weight_backward_patches),
	// which also avoids transposing NHWC inputs/cotangents/outputs.
	if !inputNode.shape.IsDynamic() && channelGroupCount == 1 && batchGroupCount == 1 && !hasNegativePad {
		allInDilOne := true
		for _, d := range inDil {
			if d != 1 {
				allInDilOne = false
				break
			}
		}
		outSpatialProd := 1
		kSpatialProd := 1
		for i := range numSpatialDims {
			outSpatialProd *= outShape.Dimensions[axes.OutputSpatial[i]]
			kSpatialProd *= kernelNode.shape.Dimensions[axes.KernelSpatial[i]]
		}
		cIn := inputNode.shape.Dimensions[axes.InputBatch]
		nContract := inputNode.shape.Dimensions[axes.InputChannels]
		patchElems := nContract * kSpatialProd * outSpatialProd * cIn
		if allInDilOne && kSpatialProd > outSpatialProd && kSpatialProd >= 64 &&
			(outSpatialProd == 1 || (outSpatialProd <= 9 && cIn*outSpatialProd <= 512 && patchElems <= 40*1024*1024)) {
			return f.convWeightBackwardPatches(
				inputNode, kernelNode, outShape, axes, strides, padLo, padHi, kernelDilations)
		}
	}

	// Step 1: Transpose input to channels-last [N, spatial..., C_in].
	rank := inputNode.shape.Rank()
	inPerm := make([]int, rank)
	inPerm[0] = axes.InputBatch
	for i, ax := range axes.InputSpatial {
		inPerm[1+i] = ax
	}
	inPerm[rank-1] = axes.InputChannels

	var inVal compute.Value = inputNode
	if !isIdentityPerm(inPerm) {
		inVal, err = f.Transpose(inVal, inPerm...)
		if err != nil {
			return nil, errors.Wrap(err, "ConvGeneral: transpose input")
		}
	}

	// Step 2: Transpose kernel to MLX channels-last [C_out, spatial..., C_in].
	kPerm := make([]int, rank)
	kPerm[0] = axes.KernelOutputChannels
	for i, ax := range axes.KernelSpatial {
		kPerm[1+i] = ax
	}
	kPerm[rank-1] = axes.KernelInputChannels

	var kVal compute.Value = kernelNode
	if !isIdentityPerm(kPerm) {
		kVal, err = f.Transpose(kVal, kPerm...)
		if err != nil {
			return nil, errors.Wrap(err, "ConvGeneral: transpose kernel")
		}
	}

	// Step 3: If negative padding (cropping) is present, apply input dilation and padding via f.Pad.
	if hasNegativePad {
		zeroConst, err := f.makeScalarConst(0, inputNode.shape.DType)
		if err != nil {
			return nil, errors.Wrap(err, "ConvGeneral: zero constant")
		}
		padAxes := make([]compute.PadAxis, rank)
		for i := range numSpatialDims {
			padAxes[1+i] = compute.PadAxis{
				Start:    padLo[i],
				End:      padHi[i],
				Interior: inDil[i] - 1,
			}
			padLo[i] = 0
			padHi[i] = 0
			inDil[i] = 1
		}
		inVal, err = f.Pad(inVal, zeroConst, padAxes...)
		if err != nil {
			return nil, errors.Wrap(err, "ConvGeneral: negative pad")
		}
	}

	// Step 4: If batchGroupCount > 1, fold batch groups into input channels:
	// [N, spatial..., C_in] -> [G_b, N', spatial..., C_in] -> [N', spatial..., G_b, C_in] -> [N', spatial..., G_b * C_in].
	if batchGroupCount > 1 {
		inCur := inVal.(*graphNode)
		curDims := inCur.shape.Dimensions
		nPrime := curDims[0] / batchGroupCount
		cIn := curDims[rank-1]

		splitDims := make([]int, rank+1)
		splitDims[0] = batchGroupCount
		splitDims[1] = nPrime
		copy(splitDims[2:rank], curDims[1:rank-1])
		splitDims[rank] = cIn
		inVal, err = f.Reshape(inVal, splitDims...)
		if err != nil {
			return nil, errors.Wrap(err, "ConvGeneral: batchGroup split reshape")
		}

		bgPerm := make([]int, rank+1)
		for i := range rank - 1 {
			bgPerm[i] = i + 1
		}
		bgPerm[rank-1] = 0
		bgPerm[rank] = rank
		inVal, err = f.Transpose(inVal, bgPerm...)
		if err != nil {
			return nil, errors.Wrap(err, "ConvGeneral: batchGroup transpose")
		}

		mergedDims := make([]int, rank)
		mergedDims[0] = nPrime
		copy(mergedDims[1:rank-1], curDims[1:rank-1])
		mergedDims[rank-1] = batchGroupCount * cIn
		inVal, err = f.Reshape(inVal, mergedDims...)
		if err != nil {
			return nil, errors.Wrap(err, "ConvGeneral: batchGroup merge reshape")
		}
	}

	// Step 5: Run mlx::core::conv_general.
	groups := channelGroupCount * batchGroupCount
	convID, err := f.builder.tape.AddConvGeneral(
		inVal.(*graphNode).id,
		kVal.(*graphNode).id,
		strides, padLo, padHi, kernelDilations, inDil, groups)
	if err != nil {
		return nil, errors.Wrap(err, "ConvGeneral")
	}

	// Step 6: Transpose from channels-last [N_out, outSpatial..., C_out] to requested output layout.
	fwdPerm := make([]int, rank)
	fwdPerm[0] = axes.OutputBatch
	for i, ax := range axes.OutputSpatial {
		fwdPerm[1+i] = ax
	}
	fwdPerm[rank-1] = axes.OutputChannels

	invPerm := make([]int, rank)
	for i, v := range fwdPerm {
		invPerm[v] = i
	}
	if !isIdentityPerm(invPerm) {
		convID, err = f.builder.tape.AddTranspose(convID, invPerm)
		if err != nil {
			return nil, errors.Wrap(err, "ConvGeneral: transpose output")
		}
	}

	outNode := &graphNode{id: convID, shape: outShape, owner: f}
	f.registerDynamicAxes(outNode)
	return outNode, nil
}

// convWeightBackwardPatches computes a convolution where the contracted spatial dimensions
// (KernelSpatial, e.g. H_out x W_out) are much larger than the output spatial dimensions
// (OutputSpatial, e.g. 1x1 or 3x3 filter size in convVJPWrtKernel), using strided patch
// extraction + Split-K Matmul.
func (f *Function) convWeightBackwardPatches(
	inputNode, kernelNode *graphNode,
	outShape shapes.Shape,
	axes compute.ConvolveAxesConfig,
	strides, padLo, padHi, kernelDilations []int,
) (compute.Value, error) {
	rank := inputNode.shape.Rank()
	numSpatialDims := len(axes.InputSpatial)

	// Step 1: Transpose inputNode to [InputChannels, InputSpatial..., InputBatch] = [N_contract, H_in..., C_in].
	// For NHWC convVJPWrtKernel, this is the identity permutation [0, 1, 2, 3].
	inPerm := make([]int, rank)
	inPerm[0] = axes.InputChannels
	for i, ax := range axes.InputSpatial {
		inPerm[1+i] = ax
	}
	inPerm[rank-1] = axes.InputBatch

	var inVal compute.Value = inputNode
	var err error
	if !isIdentityPerm(inPerm) {
		inVal, err = f.Transpose(inVal, inPerm...)
		if err != nil {
			return nil, errors.Wrap(err, "convWeightBackwardPatches: transpose input")
		}
	}

	// Step 2: Transpose kernelNode (cotangent) to [KernelInputChannels, KernelSpatial..., KernelOutputChannels]
	// = [N_contract, K_spatial..., C_out]. For NHWC convVJPWrtKernel, this is also identity [0, 1, 2, 3].
	kPerm := make([]int, rank)
	kPerm[0] = axes.KernelInputChannels
	for i, ax := range axes.KernelSpatial {
		kPerm[1+i] = ax
	}
	kPerm[rank-1] = axes.KernelOutputChannels

	var kVal compute.Value = kernelNode
	if !isIdentityPerm(kPerm) {
		kVal, err = f.Transpose(kVal, kPerm...)
		if err != nil {
			return nil, errors.Wrap(err, "convWeightBackwardPatches: transpose kernel")
		}
	}

	// Step 3: Pad spatial dimensions 1..numSpatialDims of inVal if needed.
	needsPad := false
	for i := range numSpatialDims {
		if padLo[i] > 0 || padHi[i] > 0 {
			needsPad = true
			break
		}
	}
	if needsPad {
		zeroConst, err := f.makeScalarConst(0, inputNode.shape.DType)
		if err != nil {
			return nil, errors.Wrap(err, "convWeightBackwardPatches: zero constant")
		}
		padAxes := make([]compute.PadAxis, rank)
		for i := range numSpatialDims {
			padAxes[1+i] = compute.PadAxis{Start: padLo[i], End: padHi[i]}
		}
		inVal, err = f.Pad(inVal, zeroConst, padAxes...)
		if err != nil {
			return nil, errors.Wrap(err, "convWeightBackwardPatches: pad")
		}
	}

	inPaddedNode := inVal.(*graphNode)
	kCurNode := kVal.(*graphNode)
	pDims := inPaddedNode.shape.Dimensions
	kDims := kCurNode.shape.Dimensions

	nContract := pDims[0]
	cIn := pDims[rank-1]
	cOut := kDims[rank-1]

	mContract := nContract
	oProd := 1
	outSpatialDims := make([]int, numSpatialDims)
	for i := range numSpatialDims {
		mContract *= kDims[1+i]
		outSpatialDims[i] = outShape.Dimensions[axes.OutputSpatial[i]]
		oProd *= outSpatialDims[i]
	}

	// Step 4: Extract patches [N_contract, K_0..., O_0..., C_in] via AsStrided (or direct reshape if 1x1 stride-1).
	var patches2DID mlxbridge.NodeID
	isDirectReshape := oProd == 1
	if isDirectReshape {
		for i := range numSpatialDims {
			if kDims[1+i] != pDims[1+i] || kernelDilations[i] != 1 {
				isDirectReshape = false
				break
			}
		}
	}

	if isDirectReshape {
		patches2DID, err = f.builder.tape.AddReshape(inPaddedNode.id, []int{mContract, cIn})
		if err != nil {
			return nil, errors.Wrap(err, "convWeightBackwardPatches: reshape 1x1")
		}
	} else {
		rowStrides := make([]int64, rank)
		rowStrides[rank-1] = 1
		for i := rank - 2; i >= 0; i-- {
			rowStrides[i] = rowStrides[i+1] * int64(pDims[i+1])
		}

		patchShape := make([]int, 0, 2+2*numSpatialDims)
		patchStrides := make([]int64, 0, 2+2*numSpatialDims)
		patchShape = append(patchShape, nContract)
		patchStrides = append(patchStrides, rowStrides[0])
		for i := range numSpatialDims {
			patchShape = append(patchShape, kDims[1+i])
			patchStrides = append(patchStrides, rowStrides[1+i]*int64(kernelDilations[i]))
		}
		for i := range numSpatialDims {
			patchShape = append(patchShape, outSpatialDims[i])
			patchStrides = append(patchStrides, rowStrides[1+i]*int64(strides[i]))
		}
		patchShape = append(patchShape, cIn)
		patchStrides = append(patchStrides, rowStrides[rank-1])

		stridedID, err := f.builder.tape.AddAsStrided(inPaddedNode.id, patchShape, patchStrides, 0)
		if err != nil {
			return nil, errors.Wrap(err, "convWeightBackwardPatches: as_strided")
		}
		patches2DID, err = f.builder.tape.AddReshape(stridedID, []int{mContract, oProd * cIn})
		if err != nil {
			return nil, errors.Wrap(err, "convWeightBackwardPatches: reshape patches")
		}
	}

	// Step 5: Compute patches_t [O_prod * C_in, M_contract] @ cotan_2d [M_contract, C_out]
	// -> [O_prod * C_in, C_out] -> [O_0, ..., O_{D-1}, C_in, C_out].
	patchesTID, err := f.builder.tape.AddTranspose(patches2DID, []int{1, 0})
	if err != nil {
		return nil, errors.Wrap(err, "convWeightBackwardPatches: transpose patches")
	}
	cotan2DID, err := f.builder.tape.AddReshape(kCurNode.id, []int{mContract, cOut})
	if err != nil {
		return nil, errors.Wrap(err, "convWeightBackwardPatches: reshape cotan")
	}
	mmID, err := f.builder.tape.AddMatmul(patchesTID, cotan2DID)
	if err != nil {
		return nil, errors.Wrap(err, "convWeightBackwardPatches: matmul")
	}

	canonDims := make([]int, rank)
	copy(canonDims[:numSpatialDims], outSpatialDims)
	canonDims[numSpatialDims] = cIn
	canonDims[numSpatialDims+1] = cOut
	outID, err := f.builder.tape.AddReshape(mmID, canonDims)
	if err != nil {
		return nil, errors.Wrap(err, "convWeightBackwardPatches: reshape output")
	}

	// Step 6: Transpose from [OutputSpatial..., OutputBatch(C_in), OutputChannels(C_out)] to outShape.
	// For NHWC convVJPWrtKernel, invPerm is the identity [0, 1, 2, 3].
	fwdPerm := make([]int, rank)
	for i, ax := range axes.OutputSpatial {
		fwdPerm[i] = ax
	}
	fwdPerm[numSpatialDims] = axes.OutputBatch
	fwdPerm[numSpatialDims+1] = axes.OutputChannels

	invPerm := make([]int, rank)
	for i, v := range fwdPerm {
		invPerm[v] = i
	}
	if !isIdentityPerm(invPerm) {
		outID, err = f.builder.tape.AddTranspose(outID, invPerm)
		if err != nil {
			return nil, errors.Wrap(err, "convWeightBackwardPatches: transpose output")
		}
	}

	return &graphNode{id: outID, shape: outShape, owner: f}, nil
}

// ===========================================================================
// ReduceWindow & SelectAndScatter (Pooling and Pooling Gradients)
// ===========================================================================

// ReduceWindow applies a reduction function over sliding windows of x for any tensor rank.
func (f *Function) ReduceWindow(
	x compute.Value,
	reductionType compute.ReduceOpType,
	windowDimensions, strides, baseDilations, windowDilations []int,
	paddings [][2]int,
) (compute.Value, error) {
	opNode, err := f.resolveNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "ReduceWindow")
	}
	if opNode.shape.DType == dtypes.Complex64 && (reductionType == compute.ReduceOpMax || reductionType == compute.ReduceOpMin) {
		return nil, errors.Wrapf(compute.ErrNotImplemented,
			"ReduceWindow: Max/Min not supported for Complex64 in %q backend", BackendName)
	}

	outShape, err := shapeinference.ReduceWindow(
		opNode.shape, windowDimensions, strides, baseDilations, windowDilations, paddings)
	if err != nil {
		return nil, errors.Wrap(err, "ReduceWindow")
	}

	rank := opNode.shape.Rank()
	if rank == 0 {
		return f.Identity(opNode)
	}
	if !outShape.IsDynamic() && outShape.Size() == 0 {
		id, err := f.builder.tape.AddConstant(nil, 0, dtypeToBridgeDType(outShape.DType), outShape.Dimensions)
		if err != nil {
			return nil, errors.Wrap(err, "ReduceWindow: zero-sized output")
		}
		return &graphNode{id: id, shape: outShape, owner: f}, nil
	}

	winDims := make([]int, rank)
	winStrides := make([]int, rank)
	baseDil := make([]int, rank)
	winDil := make([]int, rank)
	for i := range rank {
		if len(windowDimensions) > 0 {
			winDims[i] = windowDimensions[i]
		} else {
			winDims[i] = 1
		}
		if len(strides) > 0 {
			winStrides[i] = strides[i]
		} else {
			winStrides[i] = winDims[i]
		}
		if len(baseDilations) > 0 {
			baseDil[i] = baseDilations[i]
		} else {
			baseDil[i] = 1
		}
		if len(windowDilations) > 0 {
			winDil[i] = windowDilations[i]
		} else {
			winDil[i] = 1
		}
	}

	// Step 1: Apply baseDilations and paddings via f.Pad if needed.
	needsPad := false
	padConfig := make([]compute.PadAxis, rank)
	for i := range rank {
		padConfig[i].Interior = baseDil[i] - 1
		if paddings != nil && i < len(paddings) {
			padConfig[i].Start = paddings[i][0]
			padConfig[i].End = paddings[i][1]
		}
		if padConfig[i].Start != 0 || padConfig[i].End != 0 || padConfig[i].Interior != 0 {
			needsPad = true
		}
	}

	paddedNode := opNode
	if needsPad {
		idConst, err := f.makeReduceIdentityConst(reductionType, opNode.shape.DType)
		if err != nil {
			return nil, errors.Wrap(err, "ReduceWindow: identity constant")
		}
		paddedVal, err := f.Pad(opNode, idConst, padConfig...)
		if err != nil {
			return nil, errors.Wrap(err, "ReduceWindow: pad")
		}
		paddedNode = paddedVal.(*graphNode)
	}

	pDims := paddedNode.shape.Dimensions
	var mlxOp mlxbridge.ReduceOp
	switch reductionType {
	case compute.ReduceOpSum:
		mlxOp = mlxbridge.ReduceSum
	case compute.ReduceOpProduct:
		mlxOp = mlxbridge.ReduceProduct
	case compute.ReduceOpMax:
		mlxOp = mlxbridge.ReduceMax
	case compute.ReduceOpMin:
		mlxOp = mlxbridge.ReduceMin
	default:
		return nil, errors.Errorf("ReduceWindow: unsupported reduction type %v", reductionType)
	}

	// Fast path for non-overlapping windows with unit window dilation:
	// use a zero-copy Reshape splitting each active window axis (O_i * W_i) -> (O_i, W_i)
	// followed by Reduce, avoiding AsStrided so mlx::core::compile can fuse across pooling.
	nonOverlapping := true
	needsSlice := false
	coveredDims := make([]int, rank)
	for i := range rank {
		if winStrides[i] != winDims[i] || winDil[i] != 1 {
			nonOverlapping = false
			break
		}
		if pDims[i] == shapes.DynamicDim {
			coveredDims[i] = -1
			continue
		}
		coveredDims[i] = outShape.Dimensions[i] * winDims[i]
		if coveredDims[i] > pDims[i] {
			nonOverlapping = false
			break
		}
		if coveredDims[i] < pDims[i] {
			needsSlice = true
		}
	}

	if nonOverlapping {
		curID := paddedNode.id
		if needsSlice {
			starts := make([]int, rank)
			stridesOnes := make([]int, rank)
			for i := range rank {
				stridesOnes[i] = 1
			}
			curID, err = f.builder.tape.AddSlice(curID, starts, coveredDims, stridesOnes)
			if err != nil {
				return nil, errors.Wrap(err, "ReduceWindow: slice covered")
			}
		}
		splitShape := make([]int, 0, 2*rank)
		reduceAxes := make([]int, 0, rank)
		for i := range rank {
			outDim := outShape.Dimensions[i]
			if outDim == shapes.DynamicDim {
				outDim = -(2 + int(paddedNode.id)*16 + i)
			}
			splitShape = append(splitShape, outDim)
			if winDims[i] > 1 {
				reduceAxes = append(reduceAxes, len(splitShape))
				splitShape = append(splitShape, winDims[i])
			}
		}
		if len(reduceAxes) == 0 {
			return &graphNode{id: curID, shape: outShape, owner: f}, nil
		}
		reshapedID, err := f.builder.tape.AddReshape(curID, splitShape)
		if err != nil {
			return nil, errors.Wrap(err, "ReduceWindow: reshape")
		}
		redID, err := f.builder.tape.AddReduce(mlxOp, reshapedID, reduceAxes)
		if err != nil {
			return nil, errors.Wrap(err, "ReduceWindow: reduce")
		}
		return &graphNode{id: redID, shape: outShape, owner: f}, nil
	}

	// Step 2: General overlapping or dilated windows — construct an interleaved strided view
	// where window axes W_i (only for winDims[i] > 1) are placed immediately after O_i,
	// preserving stride-1 inner channel axes at the end for coalesced GPU reductions.
	for i := 1; i < rank; i++ {
		if pDims[i] == shapes.DynamicDim {
			return nil, errors.Wrapf(compute.ErrNotImplemented,
				"ReduceWindow: overlapping windows with dynamic non-leading axis %d are not supported in %q backend",
				i, BackendName)
		}
	}
	rowStrides := make([]int64, rank)
	rowStrides[rank-1] = 1
	for i := rank - 2; i >= 0; i-- {
		rowStrides[i] = rowStrides[i+1] * int64(pDims[i+1])
	}

	viewShape := make([]int, 0, 2*rank)
	viewStrides := make([]int64, 0, 2*rank)
	reduceAxes := make([]int, 0, rank)
	for i := range rank {
		outDim := outShape.Dimensions[i]
		if outDim == shapes.DynamicDim {
			outDim = -(2 + int(paddedNode.id)*16 + i)
		}
		viewShape = append(viewShape, outDim)
		viewStrides = append(viewStrides, rowStrides[i]*int64(winStrides[i]))
		if winDims[i] > 1 {
			reduceAxes = append(reduceAxes, len(viewShape))
			viewShape = append(viewShape, winDims[i])
			viewStrides = append(viewStrides, rowStrides[i]*int64(winDil[i]))
		}
	}

	stridedID, err := f.builder.tape.AddAsStrided(paddedNode.id, viewShape, viewStrides, 0)
	if err != nil {
		return nil, errors.Wrap(err, "ReduceWindow: as_strided")
	}
	if len(reduceAxes) == 0 {
		return &graphNode{id: stridedID, shape: outShape, owner: f}, nil
	}
	redID, err := f.builder.tape.AddReduce(mlxOp, stridedID, reduceAxes)
	if err != nil {
		return nil, errors.Wrap(err, "ReduceWindow: reduce")
	}
	return &graphNode{id: redID, shape: outShape, owner: f}, nil
}

// selectAndScatter implements SelectAndScatterMax and SelectAndScatterMin for any tensor rank.
func (f *Function) selectAndScatter(
	opName string,
	isMax bool,
	operand, source compute.Value,
	windowDimensions, windowStrides []int,
	paddings [][2]int,
) (compute.Value, error) {
	opNode, err := f.resolveNode(operand)
	if err != nil {
		return nil, errors.Wrapf(err, "%s: operand", opName)
	}
	srcNode, err := f.resolveNode(source)
	if err != nil {
		return nil, errors.Wrapf(err, "%s: source", opName)
	}
	if opNode.shape.IsDynamic() || srcNode.shape.IsDynamic() {
		return nil, errors.Wrapf(compute.ErrNotImplemented,
			"%s: dynamic shapes are not supported in %q backend", opName, BackendName)
	}
	if opNode.shape.DType == dtypes.Complex64 {
		return nil, errors.Wrapf(compute.ErrNotImplemented,
			"%s: Complex64 is not supported in %q backend", opName, BackendName)
	}

	rank := opNode.shape.Rank()
	winDims := make([]int, rank)
	winStrides := make([]int, rank)
	padLo := make([]int, rank)
	padHi := make([]int, rank)
	for i := range rank {
		if len(windowDimensions) > 0 {
			winDims[i] = windowDimensions[i]
		} else {
			winDims[i] = 1
		}
		if len(windowStrides) > 0 {
			winStrides[i] = windowStrides[i]
		} else {
			winStrides[i] = winDims[i]
		}
		if paddings != nil && i < len(paddings) {
			padLo[i] = paddings[i][0]
			padHi[i] = paddings[i][1]
			if padLo[i] < 0 || padHi[i] < 0 {
				return nil, errors.Wrapf(compute.ErrNotImplemented,
					"%s: negative padding is not supported in %q backend", opName, BackendName)
			}
		}
	}

	outShape, err := shapeinference.SelectAndScatter(
		opNode.shape, srcNode.shape, winDims, winStrides, paddings)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}

	id, err := f.builder.tape.AddSelectAndScatter(
		opNode.id, srcNode.id, isMax, winDims, winStrides, padLo, padHi)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}
	return &graphNode{id: id, shape: outShape, owner: f}, nil
}

// SelectAndScatterMax selects the maximum in each window of operand and scatters source into a zero tensor.
func (f *Function) SelectAndScatterMax(operand, source compute.Value, windowDimensions, windowStrides []int, paddings [][2]int) (compute.Value, error) {
	return f.selectAndScatter("SelectAndScatterMax", true, operand, source, windowDimensions, windowStrides, paddings)
}

// SelectAndScatterMin selects the minimum in each window of operand and scatters source into a zero tensor.
func (f *Function) SelectAndScatterMin(operand, source compute.Value, windowDimensions, windowStrides []int, paddings [][2]int) (compute.Value, error) {
	return f.selectAndScatter("SelectAndScatterMin", false, operand, source, windowDimensions, windowStrides, paddings)
}

// ===========================================================================
// Batch Normalization
// ===========================================================================

// validateBatchNormParams validates operand and 1D feature parameter shapes for BatchNorm operations.
func validateBatchNormParams(opName string, operand shapes.Shape, featureAxis int, params ...shapes.Shape) error {
	if !operand.DType.IsFloat() {
		return errors.Wrapf(compute.ErrNotImplemented,
			"%s: only floating-point dtypes are supported, got %s", opName, operand.DType)
	}
	if featureAxis < 0 || featureAxis >= operand.Rank() {
		return errors.Errorf("%s: featureAxis %d out of range for operand %s", opName, featureAxis, operand)
	}
	featureDim := operand.Dimensions[featureAxis]
	for i, p := range params {
		if p.DType != operand.DType {
			return errors.Errorf("%s: param #%d dtype %s does not match operand dtype %s", opName, i, p.DType, operand.DType)
		}
		if p.Rank() != 1 || p.Dimensions[0] != featureDim {
			return errors.Errorf("%s: param #%d shape %s must be 1D with dimension %d", opName, i, p, featureDim)
		}
	}
	return nil
}

// BatchNormForInference applies batch normalization using pre-computed mean and variance.
func (f *Function) BatchNormForInference(
	operand, scale, offset, mean, variance compute.Value,
	epsilon float32,
	featureAxis int,
) (compute.Value, error) {
	nodes, err := f.resolveNodes("BatchNormForInference", operand, scale, offset, mean, variance)
	if err != nil {
		return nil, err
	}
	opNode, scaleNode, offsetNode, meanNode, varNode := nodes[0], nodes[1], nodes[2], nodes[3], nodes[4]
	if featureAxis < 0 {
		featureAxis += opNode.shape.Rank()
	}
	if err := validateBatchNormParams("BatchNormForInference", opNode.shape, featureAxis,
		scaleNode.shape, offsetNode.shape, meanNode.shape, varNode.shape); err != nil {
		return nil, err
	}
	outShape := opNode.shape.Clone()

	dt := opNode.shape.DType
	epsNode, err := f.makeScalarConst(float64(epsilon), dt)
	if err != nil {
		return nil, errors.Wrap(err, "BatchNormForInference: epsilon")
	}

	// invStd = rsqrt(variance + epsilon) * scale
	varPlusEps, err := f.Add(varNode, epsNode)
	if err != nil {
		return nil, err
	}
	invStd, err := f.Rsqrt(varPlusEps)
	if err != nil {
		return nil, err
	}
	scaledInvStd, err := f.Mul(invStd, scaleNode)
	if err != nil {
		return nil, err
	}

	bMean, err := f.BroadcastInDim(meanNode, outShape, []int{featureAxis})
	if err != nil {
		return nil, err
	}
	bScaledInvStd, err := f.BroadcastInDim(scaledInvStd, outShape, []int{featureAxis})
	if err != nil {
		return nil, err
	}
	bOffset, err := f.BroadcastInDim(offsetNode, outShape, []int{featureAxis})
	if err != nil {
		return nil, err
	}

	centered, err := f.Sub(opNode, bMean)
	if err != nil {
		return nil, err
	}
	scaled, err := f.Mul(centered, bScaledInvStd)
	if err != nil {
		return nil, err
	}
	return f.Add(scaled, bOffset)
}

// BatchNormForTraining applies batch normalization while computing batchMean and batchVariance.
func (f *Function) BatchNormForTraining(
	operand, scale, offset compute.Value,
	epsilon float32,
	featureAxis int,
) (normalized, batchMean, batchVariance compute.Value, err error) {
	nodes, err := f.resolveNodes("BatchNormForTraining", operand, scale, offset)
	if err != nil {
		return nil, nil, nil, err
	}
	opNode, scaleNode, offsetNode := nodes[0], nodes[1], nodes[2]
	if featureAxis < 0 {
		featureAxis += opNode.shape.Rank()
	}
	if err := validateBatchNormParams("BatchNormForTraining", opNode.shape, featureAxis,
		scaleNode.shape, offsetNode.shape); err != nil {
		return nil, nil, nil, err
	}
	outShape := opNode.shape.Clone()
	if opNode.shape.IsDynamic() {
		return nil, nil, nil, errors.Wrapf(compute.ErrNotImplemented,
			"BatchNormForTraining: dynamic shapes are not supported in %q backend", BackendName)
	}

	rank := opNode.shape.Rank()
	dt := opNode.shape.DType
	var batchAxes []int
	batchCount := int64(1)
	for ax := range rank {
		if ax != featureAxis {
			batchAxes = append(batchAxes, ax)
			batchCount *= int64(opNode.shape.Dimensions[ax])
		}
	}

	countConst, err := f.makeScalarConst(float64(batchCount), dt)
	if err != nil {
		return nil, nil, nil, err
	}

	if len(batchAxes) == 0 {
		batchMean = opNode
	} else {
		sumVal, err := f.ReduceSum(opNode, batchAxes...)
		if err != nil {
			return nil, nil, nil, err
		}
		batchMean, err = f.Div(sumVal, countConst)
		if err != nil {
			return nil, nil, nil, err
		}
	}

	bMean, err := f.BroadcastInDim(batchMean, outShape, []int{featureAxis})
	if err != nil {
		return nil, nil, nil, err
	}
	diff, err := f.Sub(opNode, bMean)
	if err != nil {
		return nil, nil, nil, err
	}
	diffSq, err := f.Mul(diff, diff)
	if err != nil {
		return nil, nil, nil, err
	}

	if len(batchAxes) == 0 {
		zeroConst, err := f.makeScalarConst(0, dt)
		if err != nil {
			return nil, nil, nil, err
		}
		batchVariance, err = f.BroadcastInDim(zeroConst, scaleNode.shape, nil)
		if err != nil {
			return nil, nil, nil, err
		}
	} else {
		varSum, err := f.ReduceSum(diffSq, batchAxes...)
		if err != nil {
			return nil, nil, nil, err
		}
		batchVariance, err = f.Div(varSum, countConst)
		if err != nil {
			return nil, nil, nil, err
		}
	}

	epsNode, err := f.makeScalarConst(float64(epsilon), dt)
	if err != nil {
		return nil, nil, nil, err
	}
	varPlusEps, err := f.Add(batchVariance, epsNode)
	if err != nil {
		return nil, nil, nil, err
	}
	invStd, err := f.Rsqrt(varPlusEps)
	if err != nil {
		return nil, nil, nil, err
	}
	scaledInvStd, err := f.Mul(invStd, scaleNode)
	if err != nil {
		return nil, nil, nil, err
	}
	bScaledInvStd, err := f.BroadcastInDim(scaledInvStd, outShape, []int{featureAxis})
	if err != nil {
		return nil, nil, nil, err
	}
	bOffset, err := f.BroadcastInDim(offsetNode, outShape, []int{featureAxis})
	if err != nil {
		return nil, nil, nil, err
	}
	normScaled, err := f.Mul(diff, bScaledInvStd)
	if err != nil {
		return nil, nil, nil, err
	}
	normalized, err = f.Add(normScaled, bOffset)
	if err != nil {
		return nil, nil, nil, err
	}
	return normalized, batchMean, batchVariance, nil
}

// BatchNormGradient computes the gradients of BatchNormForTraining with respect to operand, scale, and offset.
func (f *Function) BatchNormGradient(
	operand, scale, mean, variance, gradOutput compute.Value,
	epsilon float32,
	featureAxis int,
) (gradOperand, gradScale, gradOffset compute.Value, err error) {
	nodes, err := f.resolveNodes("BatchNormGradient", operand, scale, mean, variance, gradOutput)
	if err != nil {
		return nil, nil, nil, err
	}
	opNode, scaleNode, meanNode, varNode, gradOutNode := nodes[0], nodes[1], nodes[2], nodes[3], nodes[4]
	if featureAxis < 0 {
		featureAxis += opNode.shape.Rank()
	}
	if err := validateBatchNormParams("BatchNormGradient", opNode.shape, featureAxis,
		scaleNode.shape, meanNode.shape, varNode.shape); err != nil {
		return nil, nil, nil, err
	}
	if !gradOutNode.shape.Equal(opNode.shape) {
		return nil, nil, nil, errors.Errorf("BatchNormGradient: gradOutput shape %s must match operand shape %s",
			gradOutNode.shape, opNode.shape)
	}
	if opNode.shape.IsDynamic() {
		return nil, nil, nil, errors.Wrapf(compute.ErrNotImplemented,
			"BatchNormGradient: dynamic shapes are not supported in %q backend", BackendName)
	}

	rank := opNode.shape.Rank()
	dt := opNode.shape.DType
	var batchAxes []int
	batchCount := int64(1)
	for ax := range rank {
		if ax != featureAxis {
			batchAxes = append(batchAxes, ax)
			batchCount *= int64(opNode.shape.Dimensions[ax])
		}
	}

	epsNode, err := f.makeScalarConst(float64(epsilon), dt)
	if err != nil {
		return nil, nil, nil, err
	}
	varPlusEps, err := f.Add(varNode, epsNode)
	if err != nil {
		return nil, nil, nil, err
	}
	invStd1D, err := f.Rsqrt(varPlusEps)
	if err != nil {
		return nil, nil, nil, err
	}

	bMean, err := f.BroadcastInDim(meanNode, opNode.shape, []int{featureAxis})
	if err != nil {
		return nil, nil, nil, err
	}
	bInvStd, err := f.BroadcastInDim(invStd1D, opNode.shape, []int{featureAxis})
	if err != nil {
		return nil, nil, nil, err
	}

	centered, err := f.Sub(opNode, bMean)
	if err != nil {
		return nil, nil, nil, err
	}
	xHat, err := f.Mul(centered, bInvStd)
	if err != nil {
		return nil, nil, nil, err
	}

	gradOutXHat, err := f.Mul(gradOutNode, xHat)
	if err != nil {
		return nil, nil, nil, err
	}
	if len(batchAxes) == 0 {
		gradOffset = gradOutNode
		gradScale = gradOutXHat
	} else {
		gradOffset, err = f.ReduceSum(gradOutNode, batchAxes...)
		if err != nil {
			return nil, nil, nil, err
		}
		gradScale, err = f.ReduceSum(gradOutXHat, batchAxes...)
		if err != nil {
			return nil, nil, nil, err
		}
	}

	nConst, err := f.makeScalarConst(float64(batchCount), dt)
	if err != nil {
		return nil, nil, nil, err
	}
	invNConst, err := f.makeScalarConst(1.0/float64(batchCount), dt)
	if err != nil {
		return nil, nil, nil, err
	}

	bGradOffset, err := f.BroadcastInDim(gradOffset, opNode.shape, []int{featureAxis})
	if err != nil {
		return nil, nil, nil, err
	}
	bGradScale, err := f.BroadcastInDim(gradScale, opNode.shape, []int{featureAxis})
	if err != nil {
		return nil, nil, nil, err
	}

	// term = N * gradOutput - gradOffset - xHat * gradScale
	term1, err := f.Mul(gradOutNode, nConst)
	if err != nil {
		return nil, nil, nil, err
	}
	term2, err := f.Sub(term1, bGradOffset)
	if err != nil {
		return nil, nil, nil, err
	}
	term3, err := f.Mul(xHat, bGradScale)
	if err != nil {
		return nil, nil, nil, err
	}
	term, err := f.Sub(term2, term3)
	if err != nil {
		return nil, nil, nil, err
	}

	// factor = (1/N) * scale * invStd
	scaleInvStd1D, err := f.Mul(scaleNode, invStd1D)
	if err != nil {
		return nil, nil, nil, err
	}
	factor1D, err := f.Mul(scaleInvStd1D, invNConst)
	if err != nil {
		return nil, nil, nil, err
	}
	bFactor, err := f.BroadcastInDim(factor1D, opNode.shape, []int{featureAxis})
	if err != nil {
		return nil, nil, nil, err
	}
	gradOperand, err = f.Mul(term, bFactor)
	if err != nil {
		return nil, nil, nil, err
	}
	return gradOperand, gradScale, gradOffset, nil
}

// ===========================================================================
// Random Number Generation (RNGBitGenerator)
// ===========================================================================

// RNGBitGenerator generates a tensor of random bits/values of the given shape and advances the PRNG state.
func (f *Function) RNGBitGenerator(state compute.Value, shape shapes.Shape) (newState, values compute.Value, err error) {
	if err := f.builder.backend.checkDTypeSupported(shape.DType); err != nil {
		return nil, nil, errors.Wrap(err, "RNGBitGenerator")
	}
	if shape.IsDynamic() {
		return nil, nil, errors.Wrapf(compute.ErrNotImplemented,
			"RNGBitGenerator: dynamic shape %s is not supported in %q backend", shape, BackendName)
	}
	stateNode, err := f.resolveNode(state)
	if err != nil {
		return nil, nil, errors.Wrap(err, "RNGBitGenerator: state")
	}
	if !stateNode.shape.Equal(compute.RNGStateShape) {
		return nil, nil, errors.Errorf("RNGBitGenerator: expected state shape %s, got %s",
			compute.RNGStateShape, stateNode.shape)
	}

	newStateID, valuesID, err := f.builder.tape.AddRNGBitGenerator(
		stateNode.id, dtypeToBridgeDType(shape.DType), shape.Dimensions)
	if err != nil {
		return nil, nil, errors.Wrap(err, "RNGBitGenerator")
	}
	newState = &graphNode{id: newStateID, shape: stateNode.shape.Clone(), owner: f}
	values = &graphNode{id: valuesID, shape: shape.Clone(), owner: f}
	return newState, values, nil
}
