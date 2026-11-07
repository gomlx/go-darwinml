// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mlx

import (
	"math"
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/sets"
	"github.com/gomlx/go-darwinml/internal/mlxbridge"
	"github.com/pkg/errors"
)

// FusedSoftmax computes softmax(x) along axis using MLX's native precise softmax primitive.
func (f *Function) FusedSoftmax(x compute.Value, axis int) (compute.Value, error) {
	node, err := f.resolveNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "FusedSoftmax")
	}
	if !node.shape.DType.IsFloat() {
		return nil, errors.Wrapf(compute.ErrNotImplemented,
			"FusedSoftmax: only floating-point dtypes are supported, got %s", node.shape.DType)
	}
	if axis < 0 || axis >= node.shape.Rank() {
		return nil, errors.Errorf("FusedSoftmax: axis %d out of range for shape %s", axis, node.shape)
	}
	outShape := node.shape.Clone()
	id, err := f.builder.tape.AddSoftmax(node.id, axis)
	if err != nil {
		return nil, errors.Wrap(err, "FusedSoftmax")
	}
	return &graphNode{id: id, shape: outShape, owner: f}, nil
}

// FusedLayerNorm applies layer normalization over the specified axes.
func (f *Function) FusedLayerNorm(x compute.Value, axes []int, epsilon float64, gamma, beta compute.Value) (compute.Value, error) {
	node, err := f.resolveNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "FusedLayerNorm")
	}
	if !node.shape.DType.IsFloat() {
		return nil, errors.Wrapf(compute.ErrNotImplemented,
			"FusedLayerNorm: only floating-point dtypes are supported, got %s", node.shape.DType)
	}
	if len(axes) == 0 {
		return nil, errors.Errorf("FusedLayerNorm: axes cannot be empty")
	}

	rank := node.shape.Rank()
	normAxes := make([]int, len(axes))
	for i, ax := range axes {
		if ax < 0 {
			ax += rank
		}
		if ax < 0 || ax >= rank {
			return nil, errors.Errorf("FusedLayerNorm: axis %d out of range for rank %d", axes[i], rank)
		}
		normAxes[i] = ax
	}

	var gammaNode, betaNode *graphNode
	if gamma != nil {
		gammaNode, err = f.resolveNode(gamma)
		if err != nil {
			return nil, errors.Wrap(err, "FusedLayerNorm: gamma")
		}
		if gammaNode.shape.DType != node.shape.DType {
			return nil, errors.Errorf("FusedLayerNorm: gamma dtype %s does not match x dtype %s",
				gammaNode.shape.DType, node.shape.DType)
		}
	}
	if beta != nil {
		betaNode, err = f.resolveNode(beta)
		if err != nil {
			return nil, errors.Wrap(err, "FusedLayerNorm: beta")
		}
		if betaNode.shape.DType != node.shape.DType {
			return nil, errors.Errorf("FusedLayerNorm: beta dtype %s does not match x dtype %s",
				betaNode.shape.DType, node.shape.DType)
		}
	}
	outShape := node.shape.Clone()

	// Fast path: single trailing axis normalization uses mlx::core::fast::layer_norm
	// when gamma/beta are 1D (or nil). When gamma/beta are already broadcast-shaped
	// (e.g., [1, 1, 1, C] from norm.LayerNorm), using the decomposed path below allows
	// mlx::core::compile CSE to share the mean/variance reductions with the decomposed VJP.
	if len(normAxes) == 1 && normAxes[0] == rank-1 &&
		(gammaNode == nil || gammaNode.shape.Rank() == 1) &&
		(betaNode == nil || betaNode.shape.Rank() == 1) {
		gammaID := mlxbridge.NodeID(-1)
		if gammaNode != nil {
			gammaID = gammaNode.id
		}
		betaID := mlxbridge.NodeID(-1)
		if betaNode != nil {
			betaID = betaNode.id
		}
		id, err := f.builder.tape.AddLayerNorm(node.id, gammaID, betaID, float32(epsilon))
		if err != nil {
			return nil, errors.Wrap(err, "FusedLayerNorm")
		}
		return &graphNode{id: id, shape: outShape, owner: f}, nil
	}

	if node.shape.IsDynamic() {
		return nil, errors.Wrapf(compute.ErrNotImplemented,
			"FusedLayerNorm: dynamic shapes on non-trailing axes are not supported in %q backend", BackendName)
	}

	normSet := sets.MakeWith(normAxes...)
	var nonNormAxes []int
	numElements := int64(1)
	for ax := range rank {
		if normSet.Has(ax) {
			numElements *= int64(node.shape.Dimensions[ax])
		} else {
			nonNormAxes = append(nonNormAxes, ax)
		}
	}

	dt := node.shape.DType
	countConst, err := f.makeScalarConst(float64(numElements), dt)
	if err != nil {
		return nil, err
	}
	epsConst, err := f.makeScalarConst(epsilon, dt)
	if err != nil {
		return nil, err
	}

	sumVal, err := f.ReduceSum(node, normAxes...)
	if err != nil {
		return nil, err
	}
	meanVal, err := f.Div(sumVal, countConst)
	if err != nil {
		return nil, err
	}
	bMean, err := f.BroadcastInDim(meanVal, outShape, nonNormAxes)
	if err != nil {
		return nil, err
	}
	diff, err := f.Sub(node, bMean)
	if err != nil {
		return nil, err
	}
	diffSq, err := f.Mul(diff, diff)
	if err != nil {
		return nil, err
	}
	varSum, err := f.ReduceSum(diffSq, normAxes...)
	if err != nil {
		return nil, err
	}
	varVal, err := f.Div(varSum, countConst)
	if err != nil {
		return nil, err
	}
	varPlusEps, err := f.Add(varVal, epsConst)
	if err != nil {
		return nil, err
	}
	invStd, err := f.Rsqrt(varPlusEps)
	if err != nil {
		return nil, err
	}
	bInvStd, err := f.BroadcastInDim(invStd, outShape, nonNormAxes)
	if err != nil {
		return nil, err
	}
	normalized, err := f.Mul(diff, bInvStd)
	if err != nil {
		return nil, err
	}

	sortedNormAxes := slices.Clone(normAxes)
	slices.Sort(sortedNormAxes)

	if gammaNode != nil {
		var bGamma compute.Value = gammaNode
		if gammaNode.shape.Rank() < rank {
			bGamma, err = f.BroadcastInDim(gammaNode, outShape, sortedNormAxes)
			if err != nil {
				return nil, err
			}
		}
		normalized, err = f.Mul(normalized, bGamma)
		if err != nil {
			return nil, err
		}
	}
	if betaNode != nil {
		var bBeta compute.Value = betaNode
		if betaNode.shape.Rank() < rank {
			bBeta, err = f.BroadcastInDim(betaNode, outShape, sortedNormAxes)
			if err != nil {
				return nil, err
			}
		}
		normalized, err = f.Add(normalized, bBeta)
		if err != nil {
			return nil, err
		}
	}
	return normalized, nil
}

// FusedActivation applies a fused activation function.
func (f *Function) FusedActivation(x compute.Value, cfg compute.ActivationConfig) (compute.Value, error) {
	node, err := f.resolveNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "FusedActivation")
	}
	outShape, err := shapeinference.FusedActivation(node.shape, cfg)
	if err != nil {
		return nil, errors.Wrap(err, "FusedActivation")
	}
	dt := node.shape.DType

	switch cfg.Type {
	case compute.ActivationNone:
		return f.Identity(node)
	case compute.ActivationGelu:
		return f.fusedGelu(node, true)
	case compute.ActivationGeluApproximate:
		return f.fusedGelu(node, false)
	case compute.ActivationRelu:
		zeroConst, err := f.makeScalarConst(0, dt)
		if err != nil {
			return nil, err
		}
		return f.Max(node, zeroConst)
	case compute.ActivationSigmoid:
		return f.Logistic(node)
	case compute.ActivationHardSigmoid:
		c02, err := f.makeScalarConst(0.2, dt)
		if err != nil {
			return nil, err
		}
		c05, err := f.makeScalarConst(0.5, dt)
		if err != nil {
			return nil, err
		}
		zeroConst, err := f.makeScalarConst(0, dt)
		if err != nil {
			return nil, err
		}
		oneConst, err := f.makeScalarConst(1, dt)
		if err != nil {
			return nil, err
		}
		scaled, err := f.Mul(node, c02)
		if err != nil {
			return nil, err
		}
		shifted, err := f.Add(scaled, c05)
		if err != nil {
			return nil, err
		}
		return f.Clamp(zeroConst, shifted, oneConst)
	case compute.ActivationLeakyRelu:
		zeroConst, err := f.makeScalarConst(0, dt)
		if err != nil {
			return nil, err
		}
		alphaConst, err := f.makeScalarConst(0.3, dt)
		if err != nil {
			return nil, err
		}
		cond, err := f.GreaterOrEqual(node, zeroConst)
		if err != nil {
			return nil, err
		}
		negBranch, err := f.Mul(node, alphaConst)
		if err != nil {
			return nil, err
		}
		return f.Where(cond, node, negBranch)
	case compute.ActivationSelu:
		const seluAlpha = 1.6732632423543772848170429916717
		const seluScale = 1.0507009873554804934193349852946
		zeroConst, err := f.makeScalarConst(0, dt)
		if err != nil {
			return nil, err
		}
		alphaConst, err := f.makeScalarConst(seluAlpha, dt)
		if err != nil {
			return nil, err
		}
		scaleConst, err := f.makeScalarConst(seluScale, dt)
		if err != nil {
			return nil, err
		}
		cond, err := f.GreaterThan(node, zeroConst)
		if err != nil {
			return nil, err
		}
		expm1Val, err := f.Expm1(node)
		if err != nil {
			return nil, err
		}
		negBranch, err := f.Mul(alphaConst, expm1Val)
		if err != nil {
			return nil, err
		}
		eluVal, err := f.Where(cond, node, negBranch)
		if err != nil {
			return nil, err
		}
		return f.Mul(scaleConst, eluVal)
	case compute.ActivationSilu:
		sig, err := f.Logistic(node)
		if err != nil {
			return nil, err
		}
		return f.Mul(node, sig)
	case compute.ActivationHardSwish:
		c3, err := f.makeScalarConst(3.0, dt)
		if err != nil {
			return nil, err
		}
		c6, err := f.makeScalarConst(6.0, dt)
		if err != nil {
			return nil, err
		}
		zeroConst, err := f.makeScalarConst(0, dt)
		if err != nil {
			return nil, err
		}
		xPlus3, err := f.Add(node, c3)
		if err != nil {
			return nil, err
		}
		relu6, err := f.Clamp(zeroConst, xPlus3, c6)
		if err != nil {
			return nil, err
		}
		prod, err := f.Mul(node, relu6)
		if err != nil {
			return nil, err
		}
		return f.Div(prod, c6)
	case compute.ActivationTanh:
		return f.Tanh(node)
	case compute.ActivationSwiGLU:
		rank := node.shape.Rank()
		lastDim := node.shape.Dimensions[rank-1]
		if lastDim == shapes.DynamicDim {
			return nil, errors.Wrapf(compute.ErrNotImplemented,
				"FusedActivation: SwiGLU with dynamic last dimension is not supported in %q backend", BackendName)
		}
		halfDim := lastDim / 2
		startsGate := make([]int, rank)
		limitsGate := slices.Clone(node.shape.Dimensions)
		startsVal := make([]int, rank)
		limitsVal := slices.Clone(node.shape.Dimensions)
		strides := make([]int, rank)
		for i := range rank {
			strides[i] = 1
		}
		limitsGate[rank-1] = halfDim
		startsVal[rank-1] = halfDim

		gate, err := f.Slice(node, startsGate, limitsGate, strides)
		if err != nil {
			return nil, err
		}
		val, err := f.Slice(node, startsVal, limitsVal, strides)
		if err != nil {
			return nil, err
		}
		sigGate, err := f.Logistic(gate)
		if err != nil {
			return nil, err
		}
		siluGate, err := f.Mul(gate, sigGate)
		if err != nil {
			return nil, err
		}
		res, err := f.Mul(siluGate, val)
		if err != nil {
			return nil, err
		}
		res.(*graphNode).shape = outShape
		return res, nil
	default:
		return nil, errors.Wrapf(compute.ErrNotImplemented,
			"FusedActivation: activation %v not implemented in %q backend", cfg.Type, BackendName)
	}
}

func (f *Function) fusedGelu(node *graphNode, exact bool) (compute.Value, error) {
	dt := node.shape.DType
	half, err := f.makeScalarConst(0.5, dt)
	if err != nil {
		return nil, err
	}
	one, err := f.makeScalarConst(1.0, dt)
	if err != nil {
		return nil, err
	}

	if exact {
		invSqrt2, err := f.makeScalarConst(0.7071067811865475, dt)
		if err != nil {
			return nil, err
		}
		xScaled, err := f.Mul(node, invSqrt2)
		if err != nil {
			return nil, err
		}
		erfVal, err := f.Erf(xScaled)
		if err != nil {
			return nil, err
		}
		onePlusErf, err := f.Add(one, erfVal)
		if err != nil {
			return nil, err
		}
		halfX, err := f.Mul(half, node)
		if err != nil {
			return nil, err
		}
		return f.Mul(halfX, onePlusErf)
	}

	coeff, err := f.makeScalarConst(0.044715, dt)
	if err != nil {
		return nil, err
	}
	sqrtTwoPi, err := f.makeScalarConst(0.7978845608028654, dt)
	if err != nil {
		return nil, err
	}
	x2, err := f.Mul(node, node)
	if err != nil {
		return nil, err
	}
	x3, err := f.Mul(x2, node)
	if err != nil {
		return nil, err
	}
	cx3, err := f.Mul(coeff, x3)
	if err != nil {
		return nil, err
	}
	inner, err := f.Add(node, cx3)
	if err != nil {
		return nil, err
	}
	scaled, err := f.Mul(sqrtTwoPi, inner)
	if err != nil {
		return nil, err
	}
	tanhVal, err := f.Tanh(scaled)
	if err != nil {
		return nil, err
	}
	onePlusTanh, err := f.Add(one, tanhVal)
	if err != nil {
		return nil, err
	}
	halfX, err := f.Mul(half, node)
	if err != nil {
		return nil, err
	}
	return f.Mul(halfX, onePlusTanh)
}

// FusedDense performs a fused dense (linear) projection with optional bias and activation.
func (f *Function) FusedDense(x, weight, bias compute.Value, options compute.DenseConfig) (compute.Value, error) {
	if options.Activation.Type == compute.ActivationSwiGLU {
		return nil, errors.Wrapf(compute.ErrNotImplemented, "FusedDense: ActivationSwiGLU is not supported in FusedDense")
	}

	xNode, err := f.resolveNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "FusedDense: x")
	}
	wNode, err := f.resolveNode(weight)
	if err != nil {
		return nil, errors.Wrap(err, "FusedDense: weight")
	}
	var biasShape shapes.Shape
	var biasNode *graphNode
	if bias != nil {
		biasNode, err = f.resolveNode(bias)
		if err != nil {
			return nil, errors.Wrap(err, "FusedDense: bias")
		}
		biasShape = biasNode.shape
	}

	if _, err := shapeinference.FusedDense(xNode.shape, wNode.shape, biasShape, options); err != nil {
		return nil, errors.Wrap(err, "FusedDense")
	}

	wContractAxis := 0
	if options.WeightLayout == compute.DenseLayoutOutputsInput {
		wContractAxis = wNode.shape.Rank() - 1
	}

	result, err := f.DotGeneral(xNode, []int{xNode.shape.Rank() - 1}, nil, wNode, []int{wContractAxis}, nil, compute.DotGeneralConfig{})
	if err != nil {
		return nil, errors.Wrap(err, "FusedDense: DotGeneral")
	}

	if biasNode != nil {
		resNode := result.(*graphNode)
		broadcastAxes := make([]int, biasNode.shape.Rank())
		offset := resNode.shape.Rank() - biasNode.shape.Rank()
		for i := range broadcastAxes {
			broadcastAxes[i] = offset + i
		}
		bBias, err := f.BroadcastInDim(biasNode, resNode.shape, broadcastAxes)
		if err != nil {
			return nil, errors.Wrap(err, "FusedDense: broadcast bias")
		}
		result, err = f.Add(resNode, bBias)
		if err != nil {
			return nil, errors.Wrap(err, "FusedDense: add bias")
		}
	}

	if options.Activation.Type == compute.ActivationNone {
		return result, nil
	}
	return f.FusedActivation(result, options.Activation)
}

// FusedAttentionQKVProjection performs a fused QKV projection and splits the output into query, key, and value.
func (f *Function) FusedAttentionQKVProjection(
	x, wQKV, biasQ, biasK, biasV compute.Value,
	queryDim, keyValueDim int,
) (query, key, value compute.Value, err error) {
	xNode, err := f.resolveNode(x)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "FusedAttentionQKVProjection: x")
	}
	wNode, err := f.resolveNode(wQKV)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "FusedAttentionQKVProjection: wQKV")
	}
	var bQShape, bKShape, bVShape shapes.Shape
	if biasQ != nil {
		n, err := f.resolveNode(biasQ)
		if err != nil {
			return nil, nil, nil, errors.Wrap(err, "FusedAttentionQKVProjection: biasQ")
		}
		bQShape = n.shape
	}
	if biasK != nil {
		n, err := f.resolveNode(biasK)
		if err != nil {
			return nil, nil, nil, errors.Wrap(err, "FusedAttentionQKVProjection: biasK")
		}
		bKShape = n.shape
	}
	if biasV != nil {
		n, err := f.resolveNode(biasV)
		if err != nil {
			return nil, nil, nil, errors.Wrap(err, "FusedAttentionQKVProjection: biasV")
		}
		bVShape = n.shape
	}

	if xNode.shape.Rank() < 1 || wNode.shape.Rank() != 2 {
		return nil, nil, nil, errors.Errorf("FusedAttentionQKVProjection: x must have rank >= 1 and wQKV must have rank 2, got %s and %s",
			xNode.shape, wNode.shape)
	}
	if queryDim <= 0 || keyValueDim <= 0 {
		return nil, nil, nil, errors.Errorf("FusedAttentionQKVProjection: queryDim (%d) and keyValueDim (%d) must be > 0",
			queryDim, keyValueDim)
	}
	if wNode.shape.Dimensions[1] != queryDim+2*keyValueDim {
		return nil, nil, nil, errors.Errorf("FusedAttentionQKVProjection: wQKV output dim %d must equal queryDim + 2*keyValueDim (%d)",
			wNode.shape.Dimensions[1], queryDim+2*keyValueDim)
	}
	for _, b := range []struct {
		name  string
		shape shapes.Shape
		dim   int
	}{
		{"biasQ", bQShape, queryDim},
		{"biasK", bKShape, keyValueDim},
		{"biasV", bVShape, keyValueDim},
	} {
		if b.shape.Ok() && (b.shape.DType != xNode.shape.DType || b.shape.Rank() != 1 || b.shape.Dimensions[0] != b.dim) {
			return nil, nil, nil, errors.Errorf("FusedAttentionQKVProjection: %s shape %s does not match expected [1D %d %s]",
				b.name, b.shape, b.dim, xNode.shape.DType)
		}
	}

	combined, err := f.DotGeneral(xNode, []int{xNode.shape.Rank() - 1}, nil, wNode, []int{0}, nil, compute.DotGeneralConfig{})
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "FusedAttentionQKVProjection: DotGeneral")
	}
	combinedNode := combined.(*graphNode)
	rank := combinedNode.shape.Rank()

	makeSlice := func(startLast, limitLast int) (compute.Value, error) {
		starts := make([]int, rank)
		limits := slices.Clone(combinedNode.shape.Dimensions)
		strides := make([]int, rank)
		for i := range rank {
			strides[i] = 1
		}
		starts[rank-1] = startLast
		limits[rank-1] = limitLast
		return f.Slice(combinedNode, starts, limits, strides)
	}

	query, err = makeSlice(0, queryDim)
	if err != nil {
		return nil, nil, nil, err
	}
	key, err = makeSlice(queryDim, queryDim+keyValueDim)
	if err != nil {
		return nil, nil, nil, err
	}
	value, err = makeSlice(queryDim+keyValueDim, queryDim+2*keyValueDim)
	if err != nil {
		return nil, nil, nil, err
	}

	addBias := func(val, bias compute.Value) (compute.Value, error) {
		if bias == nil {
			return val, nil
		}
		valNode := val.(*graphNode)
		biasNode := bias.(*graphNode)
		broadcastAxes := make([]int, biasNode.shape.Rank())
		offset := valNode.shape.Rank() - biasNode.shape.Rank()
		for i := range broadcastAxes {
			broadcastAxes[i] = offset + i
		}
		bBias, err := f.BroadcastInDim(biasNode, valNode.shape, broadcastAxes)
		if err != nil {
			return nil, err
		}
		return f.Add(valNode, bBias)
	}

	if query, err = addBias(query, biasQ); err != nil {
		return nil, nil, nil, err
	}
	if key, err = addBias(key, biasK); err != nil {
		return nil, nil, nil, err
	}
	if value, err = addBias(value, biasV); err != nil {
		return nil, nil, nil, err
	}
	return query, key, value, nil
}

// FusedScaledDotProductAttention computes scaled dot-product attention with support for BHSD/BSHD layouts,
// grouped-query attention (GQA), causal masking, boolean/additive masks, and additive attention bias.
func (f *Function) FusedScaledDotProductAttention(
	query, key, value compute.Value,
	axesLayout compute.AttentionAxesLayout,
	options *compute.ScaledDotProductAttentionConfig,
) (output compute.Value, statesForVJP []compute.Value, err error) {
	if options != nil && (options.QuerySeqLen != nil || options.KeyValueSeqLen != nil) {
		return nil, nil, errors.Wrapf(compute.ErrNotImplemented,
			"FusedScaledDotProductAttention: QuerySeqLen/KeyValueSeqLen not implemented in %q backend", BackendName)
	}

	qNode, err := f.resolveNode(query)
	if err != nil {
		return nil, nil, errors.Wrap(err, "SDPA: query")
	}
	kNode, err := f.resolveNode(key)
	if err != nil {
		return nil, nil, errors.Wrap(err, "SDPA: key")
	}
	vNode, err := f.resolveNode(value)
	if err != nil {
		return nil, nil, errors.Wrap(err, "SDPA: value")
	}

	if qNode.shape.Rank() != 4 || kNode.shape.Rank() != 4 || vNode.shape.Rank() != 4 {
		return nil, nil, errors.Errorf("SDPA: query, key, and value must have rank 4, got %s, %s, %s",
			qNode.shape, kNode.shape, vNode.shape)
	}
	if qNode.shape.IsDynamic() || kNode.shape.IsDynamic() || vNode.shape.IsDynamic() {
		return nil, nil, errors.Wrapf(compute.ErrNotImplemented,
			"SDPA: dynamic shapes are not supported in %q backend", BackendName)
	}

	headsAxis := axesLayout.HeadsAxis()
	numHeads := qNode.shape.Dimensions[headsAxis]
	numKVHeads := kNode.shape.Dimensions[headsAxis]
	headDim := qNode.shape.Dimensions[3]
	if numHeads <= 0 || numKVHeads <= 0 || numHeads%numKVHeads != 0 {
		return nil, nil, errors.Errorf("SDPA: numHeads (%d) must be positive and divisible by numKVHeads (%d)",
			numHeads, numKVHeads)
	}

	var scale float64
	var causal bool
	var mask, bias compute.Value
	if options != nil {
		scale = options.Scale
		causal = options.Causal
		mask = options.Mask
		bias = options.Bias
	}
	if scale == 0 {
		scale = 1.0 / math.Sqrt(float64(headDim))
	}

	isBSHD := axesLayout == compute.AttentionAxesLayoutBSHD
	if isBSHD {
		query, err = f.Transpose(query, 0, 2, 1, 3)
		if err != nil {
			return nil, nil, err
		}
		key, err = f.Transpose(key, 0, 2, 1, 3)
		if err != nil {
			return nil, nil, err
		}
		value, err = f.Transpose(value, 0, 2, 1, 3)
		if err != nil {
			return nil, nil, err
		}
		if mask != nil {
			maskNode, err := f.resolveNode(mask)
			if err != nil {
				return nil, nil, err
			}
			if maskNode.shape.Rank() == 4 {
				mask, err = f.Transpose(mask, 0, 2, 1, 3)
				if err != nil {
					return nil, nil, err
				}
			}
		}
		qNode = query.(*graphNode)
		kNode = key.(*graphNode)
		vNode = value.(*graphNode)
	}

	kvKey := key
	kvValue := value
	if numKVHeads < numHeads {
		repeats := numHeads / numKVHeads
		kvKey, err = f.repeatHeads(kNode, 1, repeats)
		if err != nil {
			return nil, nil, err
		}
		kvValue, err = f.repeatHeads(vNode, 1, repeats)
		if err != nil {
			return nil, nil, err
		}
	}

	scores, err := f.DotGeneral(
		query, []int{3}, []int{0, 1},
		kvKey, []int{3}, []int{0, 1},
		compute.DotGeneralConfig{})
	if err != nil {
		return nil, nil, err
	}

	dt := qNode.shape.DType
	scaleConst, err := f.makeScalarConst(scale, dt)
	if err != nil {
		return nil, nil, err
	}
	scores, err = f.Mul(scores, scaleConst)
	if err != nil {
		return nil, nil, err
	}

	scoresShape := scores.(*graphNode).shape

	if bias != nil {
		bBias, err := f.alignTrailingRank(bias, scoresShape.Rank())
		if err != nil {
			return nil, nil, err
		}
		scores, err = f.Add(scores, bBias)
		if err != nil {
			return nil, nil, err
		}
	}

	if mask != nil {
		mask, err = f.alignTrailingRank(mask, scoresShape.Rank())
		if err != nil {
			return nil, nil, err
		}
	}

	if causal {
		seqLen := qNode.shape.Dimensions[2]
		kvSeqLen := kNode.shape.Dimensions[2]
		maskShape := shapes.Make(dtypes.Int32, seqLen, kvSeqLen)
		rowIota, err := f.Iota(maskShape, 0)
		if err != nil {
			return nil, nil, err
		}
		colIota, err := f.Iota(maskShape, 1)
		if err != nil {
			return nil, nil, err
		}
		causalMask, err := f.GreaterOrEqual(rowIota, colIota)
		if err != nil {
			return nil, nil, err
		}
		causalMask, err = f.Reshape(causalMask, 1, 1, seqLen, kvSeqLen)
		if err != nil {
			return nil, nil, err
		}

		if mask != nil {
			maskNode, err := f.resolveNode(mask)
			if err != nil {
				return nil, nil, err
			}
			if maskNode.shape.DType == dtypes.Bool {
				mask, err = f.LogicalAnd(mask, causalMask)
				if err != nil {
					return nil, nil, err
				}
			} else {
				negInfConst, err := f.makeScalarConst(math.Inf(-1), dt)
				if err != nil {
					return nil, nil, err
				}
				zeroConst, err := f.makeScalarConst(0, dt)
				if err != nil {
					return nil, nil, err
				}
				addCausal, err := f.Where(causalMask, zeroConst, negInfConst)
				if err != nil {
					return nil, nil, err
				}
				mask, err = f.Add(mask, addCausal)
				if err != nil {
					return nil, nil, err
				}
			}
		} else {
			mask = causalMask
		}
	}

	if mask != nil {
		maskNode, err := f.resolveNode(mask)
		if err != nil {
			return nil, nil, err
		}
		if maskNode.shape.DType == dtypes.Bool {
			negInfConst, err := f.makeScalarConst(math.Inf(-1), dt)
			if err != nil {
				return nil, nil, err
			}
			scores, err = f.Where(maskNode, scores, negInfConst)
			if err != nil {
				return nil, nil, err
			}
		} else {
			scores, err = f.Add(scores, maskNode)
			if err != nil {
				return nil, nil, err
			}
		}
	}

	scores, err = f.FusedSoftmax(scores, 3)
	if err != nil {
		return nil, nil, err
	}

	output, err = f.DotGeneral(
		scores, []int{3}, []int{0, 1},
		kvValue, []int{2}, []int{0, 1},
		compute.DotGeneralConfig{})
	if err != nil {
		return nil, nil, err
	}

	if isBSHD {
		output, err = f.Transpose(output, 0, 2, 1, 3)
		if err != nil {
			return nil, nil, err
		}
	}
	return output, nil, nil
}

// alignTrailingRank prepends size-1 dimensions to v if its rank is smaller than targetRank,
// enabling GoMLX's same-rank binary and Where operations to broadcast trailing dimensions.
func (f *Function) alignTrailingRank(v compute.Value, targetRank int) (compute.Value, error) {
	n, err := f.resolveNode(v)
	if err != nil {
		return nil, err
	}
	if n.shape.Rank() == 0 || n.shape.Rank() >= targetRank {
		return n, nil
	}
	newDims := make([]int, targetRank)
	offset := targetRank - n.shape.Rank()
	for i := range offset {
		newDims[i] = 1
	}
	copy(newDims[offset:], n.shape.Dimensions)
	return f.Reshape(n, newDims...)
}

func (f *Function) repeatHeads(node *graphNode, headsAxis, repeats int) (compute.Value, error) {
	dims := node.shape.Dimensions
	rank := node.shape.Rank()
	numKVHeads := dims[headsAxis]

	newDims := make([]int, rank+1)
	copy(newDims[:headsAxis+1], dims[:headsAxis+1])
	newDims[headsAxis+1] = 1
	copy(newDims[headsAxis+2:], dims[headsAxis+1:])
	reshaped, err := f.Reshape(node, newDims...)
	if err != nil {
		return nil, err
	}

	broadcastDims := slices.Clone(newDims)
	broadcastDims[headsAxis+1] = repeats
	broadcastShape := shapes.Make(node.shape.DType, broadcastDims...)
	axes := make([]int, rank+1)
	for i := range axes {
		axes[i] = i
	}
	broadcasted, err := f.BroadcastInDim(reshaped, broadcastShape, axes)
	if err != nil {
		return nil, err
	}

	finalDims := slices.Clone(dims)
	finalDims[headsAxis] = numKVHeads * repeats
	return f.Reshape(broadcasted, finalDims...)
}
