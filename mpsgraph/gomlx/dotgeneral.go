// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mpsgraph

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

// dotGeneralOutputShape computes the output shape of a DotGeneral operation.
// Output dimensions are: [batch dims..., lhs cross dims..., rhs cross dims...].
func dotGeneralOutputShape(
	lhsShape shapes.Shape, lhsContractingAxes, lhsBatchAxes []int,
	rhsShape shapes.Shape, rhsContractingAxes, rhsBatchAxes []int,
) (shapes.Shape, error) {
	lhsSet := make(map[int]bool)
	for _, a := range lhsContractingAxes {
		lhsSet[a] = true
	}
	for _, a := range lhsBatchAxes {
		lhsSet[a] = true
	}
	rhsSet := make(map[int]bool)
	for _, a := range rhsContractingAxes {
		rhsSet[a] = true
	}
	for _, a := range rhsBatchAxes {
		rhsSet[a] = true
	}

	var dims []int
	// Batch dimensions (from lhs).
	for _, a := range lhsBatchAxes {
		dims = append(dims, lhsShape.Dimensions[a])
	}
	// LHS cross (free) dimensions.
	for i := range lhsShape.Rank() {
		if !lhsSet[i] {
			dims = append(dims, lhsShape.Dimensions[i])
		}
	}
	// RHS cross (free) dimensions.
	for i := range rhsShape.Rank() {
		if !rhsSet[i] {
			dims = append(dims, rhsShape.Dimensions[i])
		}
	}

	return shapes.Make(lhsShape.DType, dims...), nil
}

// DotGeneral implements the generalized dot product (einsum) operation.
//
// Strategy: decompose into transpose + reshape + matmul + reshape, following
// the same normalization approach used by SimpleGo's dgNormalizePrepare.
//
// 1. Classify each axis as batch, contracting, or cross.
// 2. Transpose both inputs to [batch..., cross..., contracting...] order.
// 3. Reshape to rank-3: [B, M, K] and [B, K, N].
// 4. Call MPSGraph matrixMultiplication (supports batched matmul).
// 5. Reshape output from [B, M, N] back to the full output dimensions.
func (f *Function) DotGeneral(
	lhs backends.Value, lhsContractingAxes []int, lhsBatchAxes []int,
	rhs backends.Value, rhsContractingAxes []int, rhsBatchAxes []int,
	config backends.DotGeneralConfig,
) (backends.Value, error) {
	lhsNode, err := f.resolveNode(lhs)
	if err != nil {
		return nil, errors.Wrap(err, "DotGeneral: lhs")
	}
	rhsNode, err := f.resolveNode(rhs)
	if err != nil {
		return nil, errors.Wrap(err, "DotGeneral: rhs")
	}

	// Compute output shape: [batch dims..., lhs cross dims..., rhs cross dims...].
	outShape, err := dotGeneralOutputShape(
		lhsNode.shape, lhsContractingAxes, lhsBatchAxes,
		rhsNode.shape, rhsContractingAxes, rhsBatchAxes)
	if err != nil {
		return nil, errors.Wrap(err, "DotGeneral")
	}

	// Fast path: simple matmul [M,K] x [K,N] → [M,N]
	if isSimpleMatMul(lhsNode.shape, lhsContractingAxes, lhsBatchAxes,
		rhsNode.shape, rhsContractingAxes, rhsBatchAxes) {
		tensor, err := f.ctx().MatMul(lhsNode.tensor, rhsNode.tensor)
		if err != nil {
			return nil, errors.Wrap(err, "DotGeneral: fast matmul")
		}
		return &graphNode{tensor: tensor, shape: outShape, owner: f}, nil
	}

	// General path: normalize → rank-3 matmul → reshape

	// Classify axes for both sides.
	lhsBatchSize, lhsCrossSize, contractingSize, lhsPerm := classifyAxes(
		lhsNode.shape, lhsBatchAxes, lhsContractingAxes)
	_, rhsCrossSize, _, rhsPerm := classifyAxes(
		rhsNode.shape, rhsBatchAxes, rhsContractingAxes)

	batchSize := max(lhsBatchSize, 1)

	// Transpose lhs to [batch, cross, contracting] layout.
	lhsTensor := lhsNode.tensor
	if !isIdentityPerm(lhsPerm) {
		lhsTensor, err = f.ctx().Transpose(lhsTensor, lhsPerm)
		if err != nil {
			return nil, errors.Wrap(err, "DotGeneral: lhs transpose")
		}
	}

	// Transpose rhs to [batch, cross, contracting] layout.
	rhsTensor := rhsNode.tensor
	if !isIdentityPerm(rhsPerm) {
		rhsTensor, err = f.ctx().Transpose(rhsTensor, rhsPerm)
		if err != nil {
			return nil, errors.Wrap(err, "DotGeneral: rhs transpose")
		}
	}

	// Reshape lhs to [B, M, K].
	lhsTensor, err = f.ctx().Reshape(lhsTensor, []int64{
		int64(batchSize), int64(lhsCrossSize), int64(contractingSize),
	})
	if err != nil {
		return nil, errors.Wrap(err, "DotGeneral: lhs reshape")
	}

	// Reshape rhs to [B, N, K] (after transpose, rhs is [batch, cross, contracting]).
	rhsTensor, err = f.ctx().Reshape(rhsTensor, []int64{
		int64(batchSize), int64(rhsCrossSize), int64(contractingSize),
	})
	if err != nil {
		return nil, errors.Wrap(err, "DotGeneral: rhs reshape")
	}

	// Transpose rhs from [B, N, K] to [B, K, N] for matmul.
	rhsTensor, err = f.ctx().Transpose(rhsTensor, []int{0, 2, 1})
	if err != nil {
		return nil, errors.Wrap(err, "DotGeneral: rhs transpose for matmul")
	}

	// Batched matmul: [B, M, K] x [B, K, N] → [B, M, N].
	result, err := f.ctx().MatMul(lhsTensor, rhsTensor)
	if err != nil {
		return nil, errors.Wrap(err, "DotGeneral: matmul")
	}

	// Reshape from [B, M, N] to the full output shape.
	outDims := make([]int64, outShape.Rank())
	for i, d := range outShape.Dimensions {
		outDims[i] = int64(d)
	}
	result, err = f.ctx().Reshape(result, outDims)
	if err != nil {
		return nil, errors.Wrap(err, "DotGeneral: output reshape")
	}

	return &graphNode{tensor: result, shape: outShape, owner: f}, nil
}

// isSimpleMatMul detects the common case: [M,K] x [K,N] with axis 1 contracting on lhs, axis 0 on rhs.
func isSimpleMatMul(
	lhsShape shapes.Shape, lhsContractingAxes, lhsBatchAxes []int,
	rhsShape shapes.Shape, rhsContractingAxes, rhsBatchAxes []int,
) bool {
	if len(lhsBatchAxes) != 0 || len(rhsBatchAxes) != 0 {
		return false
	}
	if lhsShape.Rank() != 2 || rhsShape.Rank() != 2 {
		return false
	}
	if len(lhsContractingAxes) != 1 || len(rhsContractingAxes) != 1 {
		return false
	}
	return lhsContractingAxes[0] == 1 && rhsContractingAxes[0] == 0
}

// classifyAxes determines which axes are batch, cross (free), and contracting,
// and returns the permutation to reorder to [batch, cross, contracting] layout.
// Also returns the product sizes for each category.
func classifyAxes(shape shapes.Shape, batchAxes, contractingAxes []int) (batchSize, crossSize, contractingSize int, perm []int) {
	rank := shape.Rank()

	// Identify cross axes (neither batch nor contracting).
	batchSet := make(map[int]bool, len(batchAxes))
	for _, a := range batchAxes {
		batchSet[a] = true
	}
	contractSet := make(map[int]bool, len(contractingAxes))
	for _, a := range contractingAxes {
		contractSet[a] = true
	}

	var crossAxes []int
	for i := range rank {
		if !batchSet[i] && !contractSet[i] {
			crossAxes = append(crossAxes, i)
		}
	}

	// Build permutation: batch, cross, contracting.
	perm = make([]int, 0, rank)
	perm = append(perm, batchAxes...)
	perm = append(perm, crossAxes...)
	perm = append(perm, contractingAxes...)

	// Compute sizes.
	batchSize = 1
	for _, a := range batchAxes {
		batchSize *= shape.Dimensions[a]
	}
	crossSize = 1
	for _, a := range crossAxes {
		crossSize *= shape.Dimensions[a]
	}
	contractingSize = 1
	for _, a := range contractingAxes {
		contractingSize *= shape.Dimensions[a]
	}

	return
}

// isIdentityPerm checks if a permutation is the identity (0, 1, 2, ...).
func isIdentityPerm(perm []int) bool {
	for i, v := range perm {
		if i != v {
			return false
		}
	}
	return true
}
