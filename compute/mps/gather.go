// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mps

import (
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/go-darwinml/internal/bridge"
	"github.com/pkg/errors"
)

// Gather implements the XLA-style Gather operation.
//
// Strategy: decompose the general XLA Gather into MPSGraph operations.
//
// For the common embedding-lookup pattern (single indexed axis, gather full slices),
// we use MPSGraph's gatherAlongAxis. For the general case, we decompose into
// reshape + gatherND + reshape.
func (f *Function) Gather(
	operand, startIndices compute.Value,
	indexVectorAxis int,
	offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes []int,
	indicesAreSorted bool,
) (compute.Value, error) {
	operandNode, err := f.resolveNode(operand)
	if err != nil {
		return nil, errors.Wrap(err, "Gather: operand")
	}
	indicesNode, err := f.resolveNode(startIndices)
	if err != nil {
		return nil, errors.Wrap(err, "Gather: startIndices")
	}

	outShape, err := shapeinference.Gather(
		operandNode.shape, indicesNode.shape,
		indexVectorAxis, offsetOutputAxes, collapsedSliceAxes,
		startIndexMap, sliceSizes, indicesAreSorted)
	if err != nil {
		return nil, errors.Wrap(err, "Gather")
	}

	// Try fast path: simple embedding lookup along a single axis.
	if result, ok := f.gatherEmbeddingLookup(operandNode, indicesNode, indexVectorAxis,
		offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes, outShape); ok {
		return result, nil
	}

	// General path: decompose XLA Gather into MPSGraph operations.
	return f.gatherGeneral(operandNode, indicesNode, indexVectorAxis,
		offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes, outShape)
}

// gatherEmbeddingLookup handles the common pattern: gather full slices along one axis.
// Pattern: startIndexMap = [axis], sliceSizes[axis] = 1, collapsedSliceAxes = [axis],
// and all other sliceSizes are the full dimension.
func (f *Function) gatherEmbeddingLookup(
	operandNode, indicesNode *graphNode,
	indexVectorAxis int,
	offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes []int,
	outShape shapes.Shape,
) (compute.Value, bool) {
	// Must have exactly one axis in startIndexMap.
	if len(startIndexMap) != 1 {
		return nil, false
	}
	gatherAxis := startIndexMap[0]

	// sliceSizes[gatherAxis] must be 1.
	if sliceSizes[gatherAxis] != 1 {
		return nil, false
	}

	// gatherAxis must be in collapsedSliceAxes.
	if !slices.Contains(collapsedSliceAxes, gatherAxis) {
		return nil, false
	}

	// All other sliceSizes must be the full dimension.
	for i, sz := range sliceSizes {
		if i == gatherAxis {
			continue
		}
		if sz != operandNode.shape.Dimensions[i] {
			return nil, false
		}
	}

	// offsetOutputAxes must be contiguous ascending.
	for i := 1; i < len(offsetOutputAxes); i++ {
		if offsetOutputAxes[i] != offsetOutputAxes[i-1]+1 {
			return nil, false
		}
	}

	// indexVectorAxis should be at the end (or startIndices.Rank() for implicit size-1 axis).
	idxRank := indicesNode.shape.Rank()
	if indexVectorAxis != idxRank-1 && indexVectorAxis != idxRank {
		return nil, false
	}

	// Remove the indexVector axis (size 1) from indices to get batch dims.
	idxTensor := indicesNode.tensor
	var err error

	// Compute batch dimensions (all index dims except indexVectorAxis).
	batchDims := make([]int64, 0, idxRank)
	for i, d := range indicesNode.shape.Dimensions {
		if i != indexVectorAxis || indexVectorAxis >= idxRank {
			batchDims = append(batchDims, int64(d))
		}
	}
	if len(batchDims) == 0 {
		batchDims = []int64{1} // Scalar index becomes [1].
	}

	if indexVectorAxis < idxRank {
		// Remove the indexVector axis (size 1) by reshaping.
		idxTensor, err = f.ctx().Reshape(idxTensor, batchDims)
		if err != nil {
			return nil, false
		}
	}

	// MPSGraph's gatherAlongAxis requires indices to have the same rank as operand.
	// We need to reshape indices to match operand rank, inserting size-1 for non-gather axes,
	// then broadcast to match the full gather output shape for gatherAlongAxis.
	operandRank := operandNode.shape.Rank()

	// Build the shape for indices matching operand rank: batch dims along gatherAxis,
	// size-1 for all other axes.
	idxFullShape := make([]int64, operandRank)
	for i := range operandRank {
		if i == gatherAxis {
			// Flatten all batch dims into this axis.
			batchSize := int64(1)
			for _, d := range batchDims {
				batchSize *= d
			}
			idxFullShape[i] = batchSize
		} else {
			idxFullShape[i] = 1
		}
	}
	idxTensor, err = f.ctx().Reshape(idxTensor, idxFullShape)
	if err != nil {
		return nil, false
	}

	// Broadcast indices to match the gatherAlongAxis output shape:
	// same as operand but with gatherAxis replaced by the batch size.
	broadcastShape := make([]int64, operandRank)
	for i, d := range operandNode.shape.Dimensions {
		if i == gatherAxis {
			broadcastShape[i] = idxFullShape[i]
		} else {
			broadcastShape[i] = int64(d)
		}
	}
	idxTensor, err = f.ctx().BroadcastTo(idxTensor, broadcastShape)
	if err != nil {
		return nil, false
	}

	// Use gatherAlongAxis — indices and operand now have the same rank.
	result, err := f.ctx().GatherAlongAxis(operandNode.tensor, idxTensor, gatherAxis)
	if err != nil {
		return nil, false
	}

	// After gatherAlongAxis, the result has the gathered dimension at gatherAxis.
	// For the output, batch dims should come first. If gatherAxis > 0, we need
	// to transpose the gatherAxis to position 0, then reshape to output shape.
	if gatherAxis > 0 {
		// Build permutation: move gatherAxis to front, keep rest in order.
		perm := make([]int, operandRank)
		perm[0] = gatherAxis
		idx := 1
		for i := range operandRank {
			if i != gatherAxis {
				perm[idx] = i
				idx++
			}
		}
		result, err = f.ctx().Transpose(result, perm)
		if err != nil {
			return nil, false
		}
	}

	// Reshape to the expected output shape.
	outDims := toInt64Slice(outShape.Dimensions)
	result, err = f.ctx().Reshape(result, outDims)
	if err != nil {
		return nil, false
	}

	return &graphNode{tensor: result, shape: outShape, owner: f}, true
}

// gatherGeneral implements the full XLA Gather semantics by decomposing into
// flatten indices → gatherND → reshape output.
func (f *Function) gatherGeneral(
	operandNode, indicesNode *graphNode,
	indexVectorAxis int,
	offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes []int,
	outShape shapes.Shape,
) (compute.Value, error) {
	operandShape := operandNode.shape
	indicesShape := indicesNode.shape
	operandRank := operandShape.Rank()

	// Step 1: Normalize indexVectorAxis to trailing position.
	idxTensor, batchSize, err := f.normalizeIndexVectorAxis(
		indicesNode.tensor, indicesShape, indexVectorAxis, "Gather")
	if err != nil {
		return nil, err
	}

	indexVectorSize := len(startIndexMap)

	collapsedSet := make(map[int]bool)
	for _, a := range collapsedSliceAxes {
		collapsedSet[a] = true
	}
	indexedSet := make(map[int]bool)
	for _, a := range startIndexMap {
		indexedSet[a] = true
	}

	// Reshape indices to [batchSize, indexVectorSize].
	// indexVectorAxis is already in trailing position after the transpose above.
	idxTensor, err = f.ctx().Reshape(idxTensor, []int64{int64(batchSize), int64(indexVectorSize)})
	if err != nil {
		return nil, errors.Wrap(err, "Gather: reshape indices")
	}

	// General strategy: transpose operand so that startIndexMap axes come first,
	// then use gatherND on the transposed operand.
	perm := make([]int, operandRank)
	copy(perm, startIndexMap)
	idx := len(startIndexMap)
	for i := range operandRank {
		if !indexedSet[i] {
			perm[idx] = i
			idx++
		}
	}

	operandTensor := operandNode.tensor
	needsTranspose := false
	for i, v := range perm {
		if v != i {
			needsTranspose = true
			break
		}
	}
	if needsTranspose {
		operandTensor, err = f.ctx().Transpose(operandTensor, perm)
		if err != nil {
			return nil, errors.Wrap(err, "Gather: transpose operand")
		}
	}

	// Now the first len(startIndexMap) axes of operand are the indexed ones.
	// Use gatherND with the indices.
	result, err := f.ctx().GatherND(operandTensor, idxTensor, 0)
	if err != nil {
		return nil, errors.Wrap(err, "Gather: gatherND general")
	}

	// Compute the batch dimensions of startIndices (all dims except indexVectorAxis).
	var batchDims []int
	for i := range indicesShape.Rank() {
		if i != indexVectorAxis || indexVectorAxis == indicesShape.Rank() {
			batchDims = append(batchDims, indicesShape.Dimensions[i])
		}
	}

	// Post-GatherND: trim non-collapsed, non-indexed axes where sliceSizes < full dimension.
	// The GatherND result has shape [batchSize, <remaining transposed operand dims>].
	// The remaining axes correspond to perm[len(startIndexMap):].
	needsSlice := false
	for k := len(startIndexMap); k < operandRank; k++ {
		origAxis := perm[k]
		if !collapsedSet[origAxis] && sliceSizes[origAxis] < operandShape.Dimensions[origAxis] {
			needsSlice = true
			break
		}
	}
	if needsSlice {
		// Build starts/ends/strides for the slice.
		// Axis 0 is the batch dimension, kept fully.
		sliceRank := 1 + (operandRank - len(startIndexMap))
		starts := make([]int64, sliceRank)
		ends := make([]int64, sliceRank)
		strides := make([]int64, sliceRank)
		starts[0] = 0
		ends[0] = int64(batchSize)
		strides[0] = 1
		for k := len(startIndexMap); k < operandRank; k++ {
			si := 1 + k - len(startIndexMap)
			origAxis := perm[k]
			starts[si] = 0
			strides[si] = 1
			if collapsedSet[origAxis] {
				ends[si] = 1
			} else {
				ends[si] = int64(sliceSizes[origAxis])
			}
		}
		result, err = f.ctx().Slice(result, starts, ends, strides)
		if err != nil {
			return nil, errors.Wrap(err, "Gather: post-gatherND slice")
		}
	}

	// Currently result has shape [batchSize, rem_0, rem_1, ...] where rem_j corresponds
	// to original operand axis perm[len(startIndexMap)+j].
	// First, reshape to expand batchDims and drop any collapsedSliceAxes among the remaining axes,
	// while keeping non-collapsed offset axes in their original operand order.
	var offsetOrigAxes []int
	for i := range operandRank {
		if !collapsedSet[i] {
			offsetOrigAxes = append(offsetOrigAxes, i)
		}
	}

	// Check if remaining non-collapsed axes in perm[len(startIndexMap):] are already in ascending order.
	// Note: any axis in startIndexMap that is NOT in collapsedSet was consumed by GatherND (since GatherND collapses indexed axes),
	// so if any axis in startIndexMap is not collapsed and has sliceSizes[a] == 1, it is a size-1 offset axis.
	var gatherNDOffsetOrigAxes []int
	for k := len(startIndexMap); k < operandRank; k++ {
		origAxis := perm[k]
		if !collapsedSet[origAxis] {
			gatherNDOffsetOrigAxes = append(gatherNDOffsetOrigAxes, origAxis)
		}
	}

	// Reshape result from [batchSize, ...] to [batchDims..., gatherNDOffsetDims...].
	intermediateDims := make([]int64, 0, len(batchDims)+len(offsetOrigAxes))
	for _, d := range batchDims {
		intermediateDims = append(intermediateDims, int64(d))
	}
	for _, origAxis := range gatherNDOffsetOrigAxes {
		intermediateDims = append(intermediateDims, int64(sliceSizes[origAxis]))
	}
	// If there are indexed axes that were NOT collapsed (so they have sliceSizes == 1), append size 1 for them,
	// and record their original axis so we can permute them into place.
	for _, origAxis := range startIndexMap {
		if !collapsedSet[origAxis] {
			if sliceSizes[origAxis] > 1 {
				return nil, errors.Wrapf(compute.ErrNotImplemented, "Gather: sliceSizes[%d]=%d > 1 on indexed axis not supported in %q backend", origAxis, sliceSizes[origAxis], BackendName)
			}
			gatherNDOffsetOrigAxes = append(gatherNDOffsetOrigAxes, origAxis)
			intermediateDims = append(intermediateDims, 1)
		}
	}
	result, err = f.ctx().Reshape(result, intermediateDims)
	if err != nil {
		return nil, errors.Wrap(err, "Gather: reshape intermediate")
	}

	// Now result has axes:
	// - 0 .. len(batchDims)-1: batch dimensions
	// - len(batchDims) + m: offset dimension for original operand axis gatherNDOffsetOrigAxes[m]
	// In the final output (rank = outShape.Rank()), the offset axes (in ascending order of original operand axes,
	// i.e., offsetOrigAxes[k]) go to output positions offsetOutputAxes[k], and batchDims go to the remaining output positions in order.
	outPerm := make([]int, outShape.Rank())
	isOffsetOutAxis := make(map[int]bool, len(offsetOutputAxes))
	for k, outAx := range offsetOutputAxes {
		isOffsetOutAxis[outAx] = true
		origAx := offsetOrigAxes[k]
		// Find where origAx is in gatherNDOffsetOrigAxes.
		foundIdx := -1
		for m, a := range gatherNDOffsetOrigAxes {
			if a == origAx {
				foundIdx = m
				break
			}
		}
		outPerm[outAx] = len(batchDims) + foundIdx
	}
	batchIdx := 0
	for outAx := range outShape.Rank() {
		if !isOffsetOutAxis[outAx] {
			outPerm[outAx] = batchIdx
			batchIdx++
		}
	}

	if !isIdentityPerm(outPerm) {
		result, err = f.ctx().Transpose(result, outPerm)
		if err != nil {
			return nil, errors.Wrap(err, "Gather: transpose output axes")
		}
	}

	// Reshape to output shape (handles scalar/empty edge cases).
	outDims := toInt64Slice(outShape.Dimensions)
	result, err = f.ctx().Reshape(result, outDims)
	if err != nil {
		return nil, errors.Wrap(err, "Gather: reshape output general")
	}

	return &graphNode{tensor: result, shape: outShape, owner: f}, nil
}

// normalizeIndexVectorAxis handles the common pattern of ensuring the indexVectorAxis
// is expanded (if implicit) and transposed to the trailing position.
// Returns the (possibly modified) tensor and the computed batchSize.
func (f *Function) normalizeIndexVectorAxis(
	idxTensor bridge.Tensor, indicesShape shapes.Shape,
	indexVectorAxis int, opName string,
) (bridge.Tensor, int, error) {
	var err error

	// If indexVectorAxis == rank, there's an implicit axis of size 1.
	if indexVectorAxis == indicesShape.Rank() {
		newDims := append(toInt64Slice(indicesShape.Dimensions), 1)
		idxTensor, err = f.ctx().Reshape(idxTensor, newDims)
		if err != nil {
			return nil, 0, errors.Wrap(err, opName+": reshape indices for implicit axis")
		}
	}

	// Compute batchSize (product of all dims except indexVectorAxis).
	batchSize := 1
	for i := range indicesShape.Rank() {
		if i != indexVectorAxis || indexVectorAxis == indicesShape.Rank() {
			batchSize *= indicesShape.Dimensions[i]
		}
	}

	// Transpose indexVectorAxis to trailing position if needed.
	effectiveRank := indicesShape.Rank()
	if indexVectorAxis == indicesShape.Rank() {
		effectiveRank = indicesShape.Rank() + 1
	}
	if indexVectorAxis != effectiveRank-1 {
		perm := make([]int, effectiveRank)
		pi := 0
		for i := 0; i < effectiveRank; i++ {
			if i != indexVectorAxis {
				perm[pi] = i
				pi++
			}
		}
		perm[effectiveRank-1] = indexVectorAxis
		idxTensor, err = f.ctx().Transpose(idxTensor, perm)
		if err != nil {
			return nil, 0, errors.Wrap(err, opName+": transposing index vector axis to trailing position")
		}
	}

	return idxTensor, batchSize, nil
}

// ===========================================================================
// Scatter Operations
// ===========================================================================

// scatterImpl is the common implementation for ScatterSum, ScatterMax, ScatterMin.
func (f *Function) scatterImpl(
	opName string,
	operandOp, scatterIndicesOp, updatesOp compute.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
	scatterMode int,
) (compute.Value, error) {
	operandNode, err := f.resolveNode(operandOp)
	if err != nil {
		return nil, errors.Wrap(err, opName+": operand")
	}
	indicesNode, err := f.resolveNode(scatterIndicesOp)
	if err != nil {
		return nil, errors.Wrap(err, opName+": indices")
	}
	updatesNode, err := f.resolveNode(updatesOp)
	if err != nil {
		return nil, errors.Wrap(err, opName+": updates")
	}

	outShape, err := shapeinference.Scatter(
		operandNode.shape, indicesNode.shape, updatesNode.shape,
		indexVectorAxis, updateWindowAxes, insertedWindowAxes,
		scatterAxesToOperandAxes)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}

	// Try simple scatter along a single axis (common case for embedding gradients).
	// Only valid when batch axes come first in updates (i.e. updateWindowAxes are trailing axes).
	canUseSimpleScatter := len(scatterAxesToOperandAxes) == 1 && len(insertedWindowAxes) == 1 && scatterAxesToOperandAxes[0] == 0
	for i, uwAx := range updateWindowAxes {
		if uwAx != (updatesNode.shape.Rank()-len(updateWindowAxes))+i {
			canUseSimpleScatter = false
			break
		}
	}
	if canUseSimpleScatter {
		axis := scatterAxesToOperandAxes[0]

		// Flatten indices to remove indexVectorAxis, yielding batch dimensions.
		idxTensor := indicesNode.tensor
		idxRank := indicesNode.shape.Rank()
		batchDims := make([]int64, 0, idxRank)
		for i, d := range indicesNode.shape.Dimensions {
			if i != indexVectorAxis || indexVectorAxis >= idxRank {
				batchDims = append(batchDims, int64(d))
			}
		}
		if len(batchDims) == 0 {
			batchDims = []int64{1}
		}
		if indexVectorAxis < idxRank {
			idxTensor, err = f.ctx().Reshape(idxTensor, batchDims)
			if err != nil {
				return nil, errors.Wrap(err, opName+": reshape indices")
			}
		}

		// MPSGraph's scatterAlongAxis requires indices rank == updates rank.
		// Reshape indices to match updates rank: batch dims go into the scatter axis,
		// size-1 for all other axes, then broadcast to updates shape.
		updatesRank := updatesNode.shape.Rank()
		batchSize := int64(1)
		for _, d := range batchDims {
			batchSize *= d
		}
		idxFullShape := make([]int64, updatesRank)
		for i := range updatesRank {
			if i == axis {
				idxFullShape[i] = batchSize
			} else {
				idxFullShape[i] = 1
			}
		}
		idxTensor, err = f.ctx().Reshape(idxTensor, idxFullShape)
		if err != nil {
			return nil, errors.Wrap(err, opName+": reshape indices to updates rank")
		}

		// Broadcast to match updates shape.
		broadcastShape := make([]int64, updatesRank)
		for i, d := range updatesNode.shape.Dimensions {
			if i == axis {
				broadcastShape[i] = batchSize
			} else {
				broadcastShape[i] = int64(d)
			}
		}
		idxTensor, err = f.ctx().BroadcastTo(idxTensor, broadcastShape)
		if err != nil {
			return nil, errors.Wrap(err, opName+": broadcast indices")
		}

		result, err := f.ctx().ScatterAlongAxis(operandNode.tensor, idxTensor, updatesNode.tensor, axis, scatterMode)
		if err != nil {
			return nil, errors.Wrap(err, opName+": scatter along axis")
		}
		return &graphNode{tensor: result, shape: outShape, owner: f}, nil
	}

	// General case: use scatterND.
	// Reshape indices for scatterND.
	indicesShape := indicesNode.shape
	indexVectorSize := len(scatterAxesToOperandAxes)

	idxTensor, batchSize, err := f.normalizeIndexVectorAxis(
		indicesNode.tensor, indicesShape, indexVectorAxis, opName)
	if err != nil {
		return nil, err
	}

	// indexVectorAxis is now in trailing position; reshape to [batchSize, indexVectorSize].
	idxTensor, err = f.ctx().Reshape(idxTensor, []int64{int64(batchSize), int64(indexVectorSize)})
	if err != nil {
		return nil, errors.Wrap(err, opName+": reshape indices for scatterND")
	}

	// Transpose operand so scatterAxesToOperandAxes axes come first,
	// matching the index columns.
	operandRank := operandNode.shape.Rank()
	scatterIndexedSet := make(map[int]bool)
	for _, a := range scatterAxesToOperandAxes {
		scatterIndexedSet[a] = true
	}
	scatterPerm := make([]int, operandRank)
	copy(scatterPerm, scatterAxesToOperandAxes)
	si := len(scatterAxesToOperandAxes)
	for i := range operandRank {
		if !scatterIndexedSet[i] {
			scatterPerm[si] = i
			si++
		}
	}

	operandTensor := operandNode.tensor
	needsScatterTranspose := false
	for i, v := range scatterPerm {
		if v != i {
			needsScatterTranspose = true
			break
		}
	}
	if needsScatterTranspose {
		operandTensor, err = f.ctx().Transpose(operandTensor, scatterPerm)
		if err != nil {
			return nil, errors.Wrap(err, opName+": transpose operand for scatterND")
		}
	}

	// For scatterND, determine the output shape from the (possibly transposed) operand.
	transposedDims := make([]int64, operandRank)
	for i, p := range scatterPerm {
		transposedDims[i] = int64(operandNode.shape.Dimensions[p])
	}

	// Normalize updates tensor so batch axes come first (matching flattened batchSize)
	// and updateWindowAxes come last, matching transposedDims[indexVectorSize:].
	updatesRank := updatesNode.shape.Rank()
	isUpdateWindowAxis := make(map[int]bool, len(updateWindowAxes))
	for _, a := range updateWindowAxes {
		isUpdateWindowAxis[a] = true
	}
	var updatesBatchAxes []int
	for i := range updatesRank {
		if !isUpdateWindowAxis[i] {
			updatesBatchAxes = append(updatesBatchAxes, i)
		}
	}

	updatesPerm := make([]int, 0, updatesRank)
	updatesPerm = append(updatesPerm, updatesBatchAxes...)
	updatesPerm = append(updatesPerm, updateWindowAxes...)

	updatesTensor := updatesNode.tensor
	if !isIdentityPerm(updatesPerm) {
		updatesTensor, err = f.ctx().Transpose(updatesTensor, updatesPerm)
		if err != nil {
			return nil, errors.Wrap(err, opName+": transpose updates for scatterND")
		}
	}

	// Reshape updates to [batchSize, transposedDims[indexVectorSize:]...].
	updatesNDDims := make([]int64, 1+(operandRank-indexVectorSize))
	updatesNDDims[0] = int64(batchSize)
	copy(updatesNDDims[1:], transposedDims[indexVectorSize:])
	updatesTensor, err = f.ctx().Reshape(updatesTensor, updatesNDDims)
	if err != nil {
		return nil, errors.Wrap(err, opName+": reshape updates for scatterND")
	}

	result, err := f.ctx().ScatterND(operandTensor, idxTensor, updatesTensor, transposedDims, scatterMode)
	if err != nil {
		return nil, errors.Wrap(err, opName+": scatterND")
	}

	// Transpose result back if we transposed the operand.
	if needsScatterTranspose {
		inversePerm := make([]int, operandRank)
		for i, v := range scatterPerm {
			inversePerm[v] = i
		}
		result, err = f.ctx().Transpose(result, inversePerm)
		if err != nil {
			return nil, errors.Wrap(err, opName+": inverse transpose after scatterND")
		}
	}

	return &graphNode{tensor: result, shape: outShape, owner: f}, nil
}

func (f *Function) ScatterSum(
	operand, scatterIndices, updates compute.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
) (compute.Value, error) {
	return f.scatterImpl("ScatterSum", operand, scatterIndices, updates,
		indexVectorAxis, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes,
		indicesAreSorted, uniqueIndices, bridge.ScatterModeAdd)
}

func (f *Function) ScatterMax(
	operand, scatterIndices, updates compute.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
) (compute.Value, error) {
	return f.scatterImpl("ScatterMax", operand, scatterIndices, updates,
		indexVectorAxis, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes,
		indicesAreSorted, uniqueIndices, bridge.ScatterModeMax)
}

func (f *Function) ScatterMin(
	operand, scatterIndices, updates compute.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
) (compute.Value, error) {
	return f.scatterImpl("ScatterMin", operand, scatterIndices, updates,
		indexVectorAxis, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes,
		indicesAreSorted, uniqueIndices, bridge.ScatterModeMin)
}
