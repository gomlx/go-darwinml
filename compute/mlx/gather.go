// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mlx

import (
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/sets"
	"github.com/gomlx/go-darwinml/internal/mlxbridge"
	"github.com/pkg/errors"
)

// Gather extracts slices from operand at the given startIndices.
func (f *Function) Gather(
	operand, startIndices compute.Value,
	indexVectorAxis int,
	offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes []int,
	indicesAreSorted bool,
) (compute.Value, error) {
	opNode, err := f.resolveNode(operand)
	if err != nil {
		return nil, errors.Wrap(err, "Gather: operand")
	}
	idxNode, err := f.resolveNode(startIndices)
	if err != nil {
		return nil, errors.Wrap(err, "Gather: startIndices")
	}

	outShape, err := shapeinference.Gather(
		opNode.shape, idxNode.shape, indexVectorAxis,
		offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes, indicesAreSorted)
	if err != nil {
		return nil, errors.Wrap(err, "Gather")
	}

	if !outShape.IsDynamic() && outShape.Size() == 0 {
		id, err := f.builder.tape.AddConstant(nil, 0, dtypeToBridgeDType(outShape.DType), outShape.Dimensions)
		if err != nil {
			return nil, errors.Wrap(err, "Gather: zero-sized output")
		}
		return &graphNode{id: id, shape: outShape, owner: f}, nil
	}

	// Compute batch dimensions (startIndices dimensions excluding indexVectorAxis),
	// encoding any dynamic batch dimensions as references to idxNode.
	var batchDims []int
	for ax, d := range idxNode.shape.Dimensions {
		if ax != indexVectorAxis {
			if d == shapes.DynamicDim {
				d = -(2 + int(idxNode.id)*16 + ax)
			}
			batchDims = append(batchDims, d)
		}
	}

	// Step 1: Run mlx::core::gather, which produces shape [batchDims..., sliceSizes[0], ..., sliceSizes[R-1]].
	curID, err := f.builder.tape.AddGather(opNode.id, idxNode.id, indexVectorAxis, startIndexMap, sliceSizes)
	if err != nil {
		return nil, errors.Wrap(err, "Gather")
	}

	// Step 2: Collapse collapsedSliceAxes if any.
	collapsedSet := sets.MakeWith(collapsedSliceAxes...)
	var nonCollapsedSliceDims []int
	for ax, s := range sliceSizes {
		if !collapsedSet.Has(ax) {
			nonCollapsedSliceDims = append(nonCollapsedSliceDims, s)
		}
	}
	if len(collapsedSliceAxes) > 0 || opNode.shape.Rank() == 0 {
		reshapedDims := make([]int, 0, len(batchDims)+len(nonCollapsedSliceDims))
		reshapedDims = append(reshapedDims, batchDims...)
		reshapedDims = append(reshapedDims, nonCollapsedSliceDims...)
		curID, err = f.builder.tape.AddReshape(curID, reshapedDims)
		if err != nil {
			return nil, errors.Wrap(err, "Gather: collapse slice axes")
		}
	}

	// Step 3: Transpose from [batchDims..., nonCollapsedSliceDims...] to outShape layout if needed.
	outRank := outShape.Rank()
	if outRank > 1 {
		perm := make([]int, outRank)
		isOffset := make([]bool, outRank)
		for k, outAx := range offsetOutputAxes {
			isOffset[outAx] = true
			perm[outAx] = len(batchDims) + k
		}
		batchIdx := 0
		for outAx := range outRank {
			if !isOffset[outAx] {
				perm[outAx] = batchIdx
				batchIdx++
			}
		}
		if !isIdentityPerm(perm) {
			curID, err = f.builder.tape.AddTranspose(curID, perm)
			if err != nil {
				return nil, errors.Wrap(err, "Gather: transpose output")
			}
		}
	}

	return &graphNode{id: curID, shape: outShape, owner: f}, nil
}

// scatterOp implements ScatterSum, ScatterMax, and ScatterMin.
func (f *Function) scatterOp(
	opName string,
	mlxOp mlxbridge.ScatterOp,
	operand, scatterIndices, updates compute.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
) (compute.Value, error) {
	opNode, err := f.resolveNode(operand)
	if err != nil {
		return nil, errors.Wrapf(err, "%s: operand", opName)
	}
	idxNode, err := f.resolveNode(scatterIndices)
	if err != nil {
		return nil, errors.Wrapf(err, "%s: scatterIndices", opName)
	}
	updNode, err := f.resolveNode(updates)
	if err != nil {
		return nil, errors.Wrapf(err, "%s: updates", opName)
	}

	outShape, err := shapeinference.Scatter(
		opNode.shape, idxNode.shape, updNode.shape,
		indexVectorAxis, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}

	if opNode.shape.IsDynamic() || idxNode.shape.IsDynamic() || updNode.shape.IsDynamic() {
		return nil, errors.Wrapf(compute.ErrNotImplemented,
			"%s: dynamic shapes are not supported in %q backend", opName, BackendName)
	}
	if opNode.shape.DType == dtypes.Complex64 && mlxOp != mlxbridge.ScatterSum {
		return nil, errors.Wrapf(compute.ErrNotImplemented,
			"%s: Complex64 is not supported for min/max scatter in %q backend", opName, BackendName)
	}
	if updNode.shape.Size() == 0 || outShape.Size() == 0 {
		return f.Identity(opNode)
	}

	// Compute batch dimensions (scatterIndices dimensions excluding indexVectorAxis).
	var batchDims []int
	for ax, d := range idxNode.shape.Dimensions {
		if ax != indexVectorAxis {
			batchDims = append(batchDims, d)
		}
	}

	// Step 1: Transpose updates so that batch axes come first and updateWindowAxes come last.
	updID := updNode.id
	updRank := updNode.shape.Rank()
	winAxesSet := sets.MakeWith(updateWindowAxes...)
	perm := make([]int, 0, updRank)
	for ax := range updRank {
		if !winAxesSet.Has(ax) {
			perm = append(perm, ax)
		}
	}
	perm = append(perm, updateWindowAxes...)
	if !isIdentityPerm(perm) {
		updID, err = f.builder.tape.AddTranspose(updID, perm)
		if err != nil {
			return nil, errors.Wrapf(err, "%s: transpose updates", opName)
		}
	}

	// Step 2: Reshape updates to [batchDims..., win_0, ..., win_{opRank-1}], inserting size-1 dims
	// at insertedWindowAxes as expected by mlx::core::scatter_*.
	opRank := opNode.shape.Rank()
	insertedSet := sets.MakeWith(insertedWindowAxes...)
	mlxUpdDims := make([]int, 0, len(batchDims)+opRank)
	mlxUpdDims = append(mlxUpdDims, batchDims...)
	winIdx := 0
	for d := range opRank {
		if insertedSet.Has(d) {
			mlxUpdDims = append(mlxUpdDims, 1)
		} else {
			mlxUpdDims = append(mlxUpdDims, updNode.shape.Dimensions[updateWindowAxes[winIdx]])
			winIdx++
		}
	}
	if !slices.Equal(updNode.shape.Dimensions, mlxUpdDims) || !isIdentityPerm(perm) {
		updID, err = f.builder.tape.AddReshape(updID, mlxUpdDims)
		if err != nil {
			return nil, errors.Wrapf(err, "%s: reshape updates", opName)
		}
	}

	// Step 3: Record Scatter op in tape.
	id, err := f.builder.tape.AddScatter(
		mlxOp, opNode.id, idxNode.id, updID, indexVectorAxis, scatterAxesToOperandAxes)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}
	return &graphNode{id: id, shape: outShape, owner: f}, nil
}

// ScatterSum scatters updates into operand using addition.
func (f *Function) ScatterSum(
	operand, scatterIndices, updates compute.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	_ bool, _ bool,
) (compute.Value, error) {
	return f.scatterOp("ScatterSum", mlxbridge.ScatterSum,
		operand, scatterIndices, updates, indexVectorAxis,
		updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes)
}

// ScatterMax scatters updates into operand using element-wise maximum.
func (f *Function) ScatterMax(
	operand, scatterIndices, updates compute.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	_ bool, _ bool,
) (compute.Value, error) {
	return f.scatterOp("ScatterMax", mlxbridge.ScatterMax,
		operand, scatterIndices, updates, indexVectorAxis,
		updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes)
}

// ScatterMin scatters updates into operand using element-wise minimum.
func (f *Function) ScatterMin(
	operand, scatterIndices, updates compute.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	_ bool, _ bool,
) (compute.Value, error) {
	return f.scatterOp("ScatterMin", mlxbridge.ScatterMin,
		operand, scatterIndices, updates, indexVectorAxis,
		updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes)
}

// packStartIndices validates and packs startIndices (either 1D tensor of length rank or rank scalar values)
// into a single 1D tape node of length rank.
func (f *Function) packStartIndices(opName string, startIndices []compute.Value, rank int) (mlxbridge.NodeID, error) {
	nodes, err := f.resolveNodes(opName+": startIndices", startIndices...)
	if err != nil {
		return -1, err
	}
	if rank == 0 {
		if len(nodes) != 0 {
			return -1, errors.Errorf("%s: scalar operand (rank 0) expects 0 startIndices, got %d", opName, len(nodes))
		}
		return -1, nil
	}
	if len(nodes) == 1 && nodes[0].shape.Rank() == 1 {
		if !nodes[0].shape.DType.IsInt() {
			return -1, errors.Errorf("%s: startIndices must be integer dtype, got %s", opName, nodes[0].shape)
		}
		if nodes[0].shape.Dimensions[0] != rank {
			return -1, errors.Errorf("%s: 1D startIndices length %d must match operand rank %d",
				opName, nodes[0].shape.Dimensions[0], rank)
		}
		return nodes[0].id, nil
	}
	if len(nodes) != rank {
		return -1, errors.Errorf("%s: expected %d startIndices (or a single 1D tensor of length %d), got %d",
			opName, rank, rank, len(nodes))
	}
	ids := make([]mlxbridge.NodeID, len(nodes))
	for i, n := range nodes {
		if !n.shape.DType.IsInt() || n.shape.Size() != 1 {
			return -1, errors.Errorf("%s: startIndices[%d] must be an integer scalar, got %s", opName, i, n.shape)
		}
		id := n.id
		if n.shape.DType != nodes[0].shape.DType {
			id, err = f.builder.tape.AddConvertDType(id, mlxbridge.DTypeInt32)
			if err != nil {
				return -1, err
			}
		}
		if n.shape.Rank() != 1 {
			id, err = f.builder.tape.AddReshape(id, []int{1})
			if err != nil {
				return -1, err
			}
		}
		ids[i] = id
	}
	if len(ids) == 1 {
		return ids[0], nil
	}
	catID, err := f.builder.tape.AddConcatenate(ids, 0)
	if err != nil {
		return -1, err
	}
	return catID, nil
}

// DynamicSlice extracts a sub-array from operand at dynamic startIndices with static sliceSizes.
func (f *Function) DynamicSlice(operand compute.Value, startIndices []compute.Value, sliceSizes []int) (compute.Value, error) {
	opNode, err := f.resolveNode(operand)
	if err != nil {
		return nil, errors.Wrap(err, "DynamicSlice: operand")
	}
	if opNode.shape.IsDynamic() {
		return nil, errors.Wrapf(compute.ErrNotImplemented,
			"DynamicSlice: dynamic operand shape %s is not supported in %q backend", opNode.shape, BackendName)
	}
	startID, err := f.packStartIndices("DynamicSlice", startIndices, opNode.shape.Rank())
	if err != nil {
		return nil, err
	}
	outShape, err := shapeinference.DynamicSlice(opNode.shape, sliceSizes)
	if err != nil {
		return nil, errors.Wrap(err, "DynamicSlice")
	}
	if opNode.shape.Rank() == 0 {
		return f.Identity(opNode)
	}
	if outShape.Size() == 0 {
		id, err := f.builder.tape.AddConstant(nil, 0, dtypeToBridgeDType(outShape.DType), outShape.Dimensions)
		if err != nil {
			return nil, errors.Wrap(err, "DynamicSlice: zero-sized output")
		}
		return &graphNode{id: id, shape: outShape, owner: f}, nil
	}
	id, err := f.builder.tape.AddDynamicSlice(opNode.id, startID, sliceSizes)
	if err != nil {
		return nil, errors.Wrap(err, "DynamicSlice")
	}
	return &graphNode{id: id, shape: outShape, owner: f}, nil
}

// DynamicUpdateSlice writes update into operand at dynamic startIndices.
func (f *Function) DynamicUpdateSlice(operand, update compute.Value, startIndices []compute.Value) (compute.Value, error) {
	opNode, err := f.resolveNode(operand)
	if err != nil {
		return nil, errors.Wrap(err, "DynamicUpdateSlice: operand")
	}
	updNode, err := f.resolveNode(update)
	if err != nil {
		return nil, errors.Wrap(err, "DynamicUpdateSlice: update")
	}
	if opNode.shape.IsDynamic() || updNode.shape.IsDynamic() {
		return nil, errors.Wrapf(compute.ErrNotImplemented,
			"DynamicUpdateSlice: dynamic shapes are not supported in %q backend", BackendName)
	}
	startID, err := f.packStartIndices("DynamicUpdateSlice", startIndices, opNode.shape.Rank())
	if err != nil {
		return nil, err
	}
	outShape, err := shapeinference.DynamicUpdateSlice(opNode.shape, updNode.shape)
	if err != nil {
		return nil, errors.Wrap(err, "DynamicUpdateSlice")
	}
	if opNode.shape.Rank() == 0 {
		return f.Identity(updNode)
	}
	if updNode.shape.Size() == 0 {
		return f.Identity(opNode)
	}
	id, err := f.builder.tape.AddDynamicUpdateSlice(opNode.id, updNode.id, startID, opNode.shape.Rank())
	if err != nil {
		return nil, errors.Wrap(err, "DynamicUpdateSlice")
	}
	return &graphNode{id: id, shape: outShape, owner: f}, nil
}
