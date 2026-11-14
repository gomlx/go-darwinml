// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mlx

import (
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/go-darwinml/internal/mlxbridge"
	"github.com/pkg/errors"
)

// DotGeneral implements the generalized dot product (einsum) operation.
func (f *Function) DotGeneral(
	lhs compute.Value, lhsContractingAxes []int, lhsBatchAxes []int,
	rhs compute.Value, rhsContractingAxes []int, rhsBatchAxes []int,
	config compute.DotGeneralConfig,
) (compute.Value, error) {
	lhsNode, err := f.resolveNode(lhs)
	if err != nil {
		return nil, errors.Wrap(err, "DotGeneral: lhs")
	}
	rhsNode, err := f.resolveNode(rhs)
	if err != nil {
		return nil, errors.Wrap(err, "DotGeneral: rhs")
	}
	if lhsNode.shape.DType != rhsNode.shape.DType {
		return nil, errors.Errorf("DotGeneral: lhs dtype (%s) and rhs dtype (%s) must match",
			lhsNode.shape.DType, rhsNode.shape.DType)
	}
	inputDType := lhsNode.shape.DType
	if err := f.builder.backend.checkDTypeSupported(inputDType); err != nil {
		return nil, errors.Wrap(err, "DotGeneral")
	}
	if config.AccumulatorDType != dtypes.InvalidDType {
		if err := f.builder.backend.checkDTypeSupported(config.AccumulatorDType); err != nil {
			return nil, errors.Wrap(err, "DotGeneral: AccumulatorDType")
		}
	}
	outDType := inputDType
	if config.OutputDType != dtypes.InvalidDType {
		if err := f.builder.backend.checkDTypeSupported(config.OutputDType); err != nil {
			return nil, errors.Wrap(err, "DotGeneral: OutputDType")
		}
		outDType = config.OutputDType
	}

	lhsContracting, err := normalizeAxes("DotGeneral: lhsContractingAxes", lhsContractingAxes, lhsNode.shape.Rank())
	if err != nil {
		return nil, err
	}
	lhsBatch, err := normalizeAxes("DotGeneral: lhsBatchAxes", lhsBatchAxes, lhsNode.shape.Rank())
	if err != nil {
		return nil, err
	}
	rhsContracting, err := normalizeAxes("DotGeneral: rhsContractingAxes", rhsContractingAxes, rhsNode.shape.Rank())
	if err != nil {
		return nil, err
	}
	rhsBatch, err := normalizeAxes("DotGeneral: rhsBatchAxes", rhsBatchAxes, rhsNode.shape.Rank())
	if err != nil {
		return nil, err
	}

	outShape, err := dotGeneralOutputShape(
		lhsNode.shape, lhsContracting, lhsBatch,
		rhsNode.shape, rhsContracting, rhsBatch,
		outDType,
	)
	if err != nil {
		return nil, errors.Wrap(err, "DotGeneral")
	}
	makeOut := func(id mlxbridge.NodeID) *graphNode {
		n := &graphNode{id: id, shape: outShape, owner: f}
		f.registerDynamicAxes(n)
		return n
	}

	// Determine computation dtype.
	computeDType := inputDType
	if config.AccumulatorDType != dtypes.InvalidDType && config.AccumulatorDType != inputDType {
		computeDType = config.AccumulatorDType
	}

	lhsID := lhsNode.id
	rhsID := rhsNode.id
	if computeDType != inputDType {
		lhsID, err = f.builder.tape.AddConvertDType(lhsID, dtypeToBridgeDType(computeDType))
		if err != nil {
			return nil, errors.Wrap(err, "DotGeneral: convert lhs to AccumulatorDType")
		}
		rhsID, err = f.builder.tape.AddConvertDType(rhsID, dtypeToBridgeDType(computeDType))
		if err != nil {
			return nil, errors.Wrap(err, "DotGeneral: convert rhs to AccumulatorDType")
		}
	}

	// Fast path: standard 2D/batched matmul [B..., M, K] x [B..., K, N] -> [B..., M, N]
	// (including when rhs is [B..., N, K] with rhsContracting == [rhsRank-1]).
	if isSimpleMatMul(lhsNode.shape, lhsContracting, lhsBatch, rhsNode.shape, rhsContracting, rhsBatch) {
		resID, err := f.builder.tape.AddMatmul(lhsID, rhsID)
		if err != nil {
			return nil, errors.Wrap(err, "DotGeneral: matmul")
		}
		if outDType != computeDType {
			resID, err = f.builder.tape.AddConvertDType(resID, dtypeToBridgeDType(outDType))
			if err != nil {
				return nil, errors.Wrap(err, "DotGeneral: convert output")
			}
		}
		return makeOut(resID), nil
	}
	if isTransposedRhsMatMul(lhsNode.shape, lhsContracting, lhsBatch, rhsNode.shape, rhsContracting, rhsBatch) {
		rRank := rhsNode.shape.Rank()
		perm := make([]int, rRank)
		for i := range rRank {
			perm[i] = i
		}
		perm[rRank-2], perm[rRank-1] = rRank-1, rRank-2
		rhsT, err := f.builder.tape.AddTranspose(rhsID, perm)
		if err != nil {
			return nil, errors.Wrap(err, "DotGeneral: rhs transpose")
		}
		resID, err := f.builder.tape.AddMatmul(lhsID, rhsT)
		if err != nil {
			return nil, errors.Wrap(err, "DotGeneral: matmul")
		}
		if outDType != computeDType {
			resID, err = f.builder.tape.AddConvertDType(resID, dtypeToBridgeDType(outDType))
			if err != nil {
				return nil, errors.Wrap(err, "DotGeneral: convert output")
			}
		}
		return makeOut(resID), nil
	}

	// General path: normalize to 2D [M, K] x [K, N] (when len(lhsBatch) == 0)
	// or 3D [B, M, K] x [B, K, N] (when len(lhsBatch) > 0).
	// When an operand's axes are already ordered as [batch..., contracting..., cross...]
	// (e.g., weight gradients dW = X^T @ dY), reshape first (zero-copy!) and transpose the
	// 2D/3D matrix rather than transposing the N-D tensor before reshape.
	lhsBatchSize, lhsCrossSize, contractingSize, lhsPermMK, lhsPermKM := classifyAxes(lhsNode.shape, lhsBatch, lhsContracting)
	_, rhsCrossSize, _, rhsPermNK, rhsPermKN := classifyAxes(rhsNode.shape, rhsBatch, rhsContracting)

	// Fast path when both operands have 1 cross axis and 1 contracting axis
	// (e.g., BSHD attention [B, S, H, D] with batch=[0, 2]): transpose directly to
	// [batch..., M, K] and [batch..., K, N] and call Matmul without reshaping batch dims.
	if len(lhsContracting) == 1 && len(rhsContracting) == 1 &&
		lhsNode.shape.Rank() == len(lhsBatch)+2 && rhsNode.shape.Rank() == len(rhsBatch)+2 {
		if !isIdentityPerm(lhsPermMK) {
			lhsID, err = f.builder.tape.AddTranspose(lhsID, lhsPermMK)
			if err != nil {
				return nil, errors.Wrap(err, "DotGeneral: lhs transpose")
			}
		}
		if !isIdentityPerm(rhsPermKN) {
			rhsID, err = f.builder.tape.AddTranspose(rhsID, rhsPermKN)
			if err != nil {
				return nil, errors.Wrap(err, "DotGeneral: rhs transpose")
			}
		}
		resID, err := f.builder.tape.AddMatmul(lhsID, rhsID)
		if err != nil {
			return nil, errors.Wrap(err, "DotGeneral: matmul")
		}
		if outDType != computeDType {
			resID, err = f.builder.tape.AddConvertDType(resID, dtypeToBridgeDType(outDType))
			if err != nil {
				return nil, errors.Wrap(err, "DotGeneral: convert output")
			}
		}
		return makeOut(resID), nil
	}

	hasBatch := len(lhsBatch) > 0
	batchSize := lhsBatchSize

	// Prepare lhs in [M, K] or [B, M, K].
	if isIdentityPerm(lhsPermKM) && !isIdentityPerm(lhsPermMK) {
		if hasBatch {
			lhsID, err = f.builder.tape.AddReshape(lhsID, []int{batchSize, contractingSize, lhsCrossSize})
		} else {
			lhsID, err = f.builder.tape.AddReshape(lhsID, []int{contractingSize, lhsCrossSize})
		}
		if err != nil {
			return nil, errors.Wrap(err, "DotGeneral: lhs reshape KM")
		}
		if hasBatch {
			lhsID, err = f.builder.tape.AddTranspose(lhsID, []int{0, 2, 1})
		} else {
			lhsID, err = f.builder.tape.AddTranspose(lhsID, []int{1, 0})
		}
		if err != nil {
			return nil, errors.Wrap(err, "DotGeneral: lhs transpose KM->MK")
		}
	} else {
		if !isIdentityPerm(lhsPermMK) {
			lhsID, err = f.builder.tape.AddTranspose(lhsID, lhsPermMK)
			if err != nil {
				return nil, errors.Wrap(err, "DotGeneral: lhs transpose")
			}
		}
		if hasBatch {
			lhsID, err = f.builder.tape.AddReshape(lhsID, []int{batchSize, lhsCrossSize, contractingSize})
		} else {
			lhsID, err = f.builder.tape.AddReshape(lhsID, []int{lhsCrossSize, contractingSize})
		}
		if err != nil {
			return nil, errors.Wrap(err, "DotGeneral: lhs reshape")
		}
	}

	// Prepare rhs in [K, N] or [B, K, N].
	if isIdentityPerm(rhsPermKN) {
		if hasBatch {
			rhsID, err = f.builder.tape.AddReshape(rhsID, []int{batchSize, contractingSize, rhsCrossSize})
		} else {
			rhsID, err = f.builder.tape.AddReshape(rhsID, []int{contractingSize, rhsCrossSize})
		}
		if err != nil {
			return nil, errors.Wrap(err, "DotGeneral: rhs reshape KN")
		}
	} else if isIdentityPerm(rhsPermNK) {
		if hasBatch {
			rhsID, err = f.builder.tape.AddReshape(rhsID, []int{batchSize, rhsCrossSize, contractingSize})
		} else {
			rhsID, err = f.builder.tape.AddReshape(rhsID, []int{rhsCrossSize, contractingSize})
		}
		if err != nil {
			return nil, errors.Wrap(err, "DotGeneral: rhs reshape NK")
		}
		if hasBatch {
			rhsID, err = f.builder.tape.AddTranspose(rhsID, []int{0, 2, 1})
		} else {
			rhsID, err = f.builder.tape.AddTranspose(rhsID, []int{1, 0})
		}
		if err != nil {
			return nil, errors.Wrap(err, "DotGeneral: rhs transpose NK->KN")
		}
	} else {
		rhsID, err = f.builder.tape.AddTranspose(rhsID, rhsPermKN)
		if err != nil {
			return nil, errors.Wrap(err, "DotGeneral: rhs transpose")
		}
		if hasBatch {
			rhsID, err = f.builder.tape.AddReshape(rhsID, []int{batchSize, contractingSize, rhsCrossSize})
		} else {
			rhsID, err = f.builder.tape.AddReshape(rhsID, []int{contractingSize, rhsCrossSize})
		}
		if err != nil {
			return nil, errors.Wrap(err, "DotGeneral: rhs reshape")
		}
	}

	// Matmul: [M, K] x [K, N] -> [M, N] or [B, M, K] x [B, K, N] -> [B, M, N].
	resID, err := f.builder.tape.AddMatmul(lhsID, rhsID)
	if err != nil {
		return nil, errors.Wrap(err, "DotGeneral: matmul")
	}

	// Reshape to outShape.Dimensions, encoding any dynamic dimensions
	// by referencing the corresponding lhs or rhs input node and axis.
	if hasBatch || outShape.Rank() != 2 || outShape.IsDynamic() || !slices.Equal(outShape.Dimensions, []int{lhsCrossSize, rhsCrossSize}) {
		outReshapeDims := encodeDotGeneralOutputDims(
			outShape.Dimensions,
			lhsNode, lhsBatch, lhsContracting,
			rhsNode, rhsBatch, rhsContracting)
		resID, err = f.builder.tape.AddReshape(resID, outReshapeDims)
		if err != nil {
			return nil, errors.Wrap(err, "DotGeneral: output reshape")
		}
	}

	if outDType != computeDType {
		resID, err = f.builder.tape.AddConvertDType(resID, dtypeToBridgeDType(outDType))
		if err != nil {
			return nil, errors.Wrap(err, "DotGeneral: convert output")
		}
	}

	return makeOut(resID), nil
}

func normalizeAxes(opName string, axes []int, rank int) ([]int, error) {
	if len(axes) == 0 {
		return nil, nil
	}
	res := make([]int, len(axes))
	for i, a := range axes {
		if a < 0 {
			a += rank
		}
		if a < 0 || a >= rank {
			return nil, errors.Errorf("%s: axis %d out of bounds for rank %d", opName, axes[i], rank)
		}
		res[i] = a
	}
	return res, nil
}

func dotGeneralOutputShape(
	lhsShape shapes.Shape, lhsContractingAxes, lhsBatchAxes []int,
	rhsShape shapes.Shape, rhsContractingAxes, rhsBatchAxes []int,
	outDType dtypes.DType,
) (shapes.Shape, error) {
	if len(lhsBatchAxes) != len(rhsBatchAxes) {
		return shapes.Invalid(), errors.Errorf("DotGeneral: number of batch axes must match (%d vs %d)",
			len(lhsBatchAxes), len(rhsBatchAxes))
	}
	if len(lhsContractingAxes) != len(rhsContractingAxes) {
		return shapes.Invalid(), errors.Errorf("DotGeneral: number of contracting axes must match (%d vs %d)",
			len(lhsContractingAxes), len(rhsContractingAxes))
	}

	lhsSet := make(map[int]bool, len(lhsContractingAxes)+len(lhsBatchAxes))
	for _, a := range lhsContractingAxes {
		if lhsSet[a] {
			return shapes.Invalid(), errors.Errorf("DotGeneral: duplicate lhs axis %d", a)
		}
		lhsSet[a] = true
	}
	for _, a := range lhsBatchAxes {
		if lhsSet[a] {
			return shapes.Invalid(), errors.Errorf("DotGeneral: duplicate lhs axis %d", a)
		}
		lhsSet[a] = true
	}

	rhsSet := make(map[int]bool, len(rhsContractingAxes)+len(rhsBatchAxes))
	for _, a := range rhsContractingAxes {
		if rhsSet[a] {
			return shapes.Invalid(), errors.Errorf("DotGeneral: duplicate rhs axis %d", a)
		}
		rhsSet[a] = true
	}
	for _, a := range rhsBatchAxes {
		if rhsSet[a] {
			return shapes.Invalid(), errors.Errorf("DotGeneral: duplicate rhs axis %d", a)
		}
		rhsSet[a] = true
	}

	for i, la := range lhsBatchAxes {
		ra := rhsBatchAxes[i]
		ld := lhsShape.Dimensions[la]
		rd := rhsShape.Dimensions[ra]
		if ld == shapes.DynamicDim && rd == shapes.DynamicDim {
			if !shapes.AxisNamesCompatible(lhsShape.AxisName(la), rhsShape.AxisName(ra)) {
				return shapes.Invalid(), errors.Errorf("DotGeneral: batch axis %d axis name conflict (%q vs %q)", i, lhsShape.AxisName(la), rhsShape.AxisName(ra))
			}
		} else if ld != shapes.DynamicDim && rd != shapes.DynamicDim && ld != rd {
			return shapes.Invalid(), errors.Errorf("DotGeneral: batch axis %d dimension mismatch (%d vs %d)", i, ld, rd)
		}
	}
	for i, la := range lhsContractingAxes {
		ra := rhsContractingAxes[i]
		ld := lhsShape.Dimensions[la]
		rd := rhsShape.Dimensions[ra]
		if ld == shapes.DynamicDim && rd == shapes.DynamicDim {
			if !shapes.AxisNamesCompatible(lhsShape.AxisName(la), rhsShape.AxisName(ra)) {
				return shapes.Invalid(), errors.Errorf("DotGeneral: contracting axis %d axis name conflict (%q vs %q)", i, lhsShape.AxisName(la), rhsShape.AxisName(ra))
			}
		} else if ld != shapes.DynamicDim && rd != shapes.DynamicDim && ld != rd {
			return shapes.Invalid(), errors.Errorf("DotGeneral: contracting axis %d dimension mismatch (%d vs %d)", i, ld, rd)
		}
	}

	var dims []int
	var names []string
	hasNames := lhsShape.AxisNames != nil || rhsShape.AxisNames != nil

	for i, a := range lhsBatchAxes {
		ra := rhsBatchAxes[i]
		if lhsShape.Dimensions[a] == shapes.DynamicDim || rhsShape.Dimensions[ra] == shapes.DynamicDim {
			dims = append(dims, shapes.DynamicDim)
		} else {
			dims = append(dims, lhsShape.Dimensions[a])
		}
		if hasNames {
			unified, err := shapes.UnifyAxisName(lhsShape.AxisName(a), rhsShape.AxisName(ra))
			if err != nil {
				return shapes.Invalid(), errors.Wrapf(err, "DotGeneral batch axis %d", i)
			}
			names = append(names, unified)
		}
	}
	for i := range lhsShape.Rank() {
		if !lhsSet[i] {
			dims = append(dims, lhsShape.Dimensions[i])
			if hasNames {
				names = append(names, lhsShape.AxisName(i))
			}
		}
	}
	for i := range rhsShape.Rank() {
		if !rhsSet[i] {
			dims = append(dims, rhsShape.Dimensions[i])
			if hasNames {
				names = append(names, rhsShape.AxisName(i))
			}
		}
	}

	if hasNames && slices.ContainsFunc(names, func(s string) bool { return s != "" }) {
		return shapes.MakeDynamic(outDType, dims, names), nil
	}
	return shapes.Shape{DType: outDType, Dimensions: dims}, nil
}

func isSimpleMatMul(
	lhsShape shapes.Shape, lhsContractingAxes, lhsBatchAxes []int,
	rhsShape shapes.Shape, rhsContractingAxes, rhsBatchAxes []int,
) bool {
	if len(lhsContractingAxes) != 1 || len(rhsContractingAxes) != 1 {
		return false
	}
	b := len(lhsBatchAxes)
	if b != len(rhsBatchAxes) {
		return false
	}
	if b == 0 {
		return lhsShape.Rank() >= 2 && rhsShape.Rank() == 2 &&
			lhsContractingAxes[0] == lhsShape.Rank()-1 && rhsContractingAxes[0] == 0
	}
	if lhsShape.Rank() != b+2 || rhsShape.Rank() != b+2 {
		return false
	}
	for i := range b {
		if lhsBatchAxes[i] != i || rhsBatchAxes[i] != i {
			return false
		}
	}
	return lhsContractingAxes[0] == b+1 && rhsContractingAxes[0] == b
}

func isTransposedRhsMatMul(
	lhsShape shapes.Shape, lhsContractingAxes, lhsBatchAxes []int,
	rhsShape shapes.Shape, rhsContractingAxes, rhsBatchAxes []int,
) bool {
	if len(lhsContractingAxes) != 1 || len(rhsContractingAxes) != 1 {
		return false
	}
	b := len(lhsBatchAxes)
	if b != len(rhsBatchAxes) {
		return false
	}
	if b == 0 {
		return lhsShape.Rank() >= 2 && rhsShape.Rank() == 2 &&
			lhsContractingAxes[0] == lhsShape.Rank()-1 && rhsContractingAxes[0] == 1
	}
	if lhsShape.Rank() != b+2 || rhsShape.Rank() != b+2 {
		return false
	}
	for i := range b {
		if lhsBatchAxes[i] != i || rhsBatchAxes[i] != i {
			return false
		}
	}
	return lhsContractingAxes[0] == b+1 && rhsContractingAxes[0] == b+1
}

func classifyAxes(shape shapes.Shape, batchAxes, contractingAxes []int) (batchSize, crossSize, contractingSize int, permCrossFirst, permContractFirst []int) {
	rank := shape.Rank()
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

	permCrossFirst = make([]int, 0, rank)
	permCrossFirst = append(permCrossFirst, batchAxes...)
	permCrossFirst = append(permCrossFirst, crossAxes...)
	permCrossFirst = append(permCrossFirst, contractingAxes...)

	permContractFirst = make([]int, 0, rank)
	permContractFirst = append(permContractFirst, batchAxes...)
	permContractFirst = append(permContractFirst, contractingAxes...)
	permContractFirst = append(permContractFirst, crossAxes...)

	prod := func(axes []int) int {
		p := 1
		for _, a := range axes {
			d := shape.Dimensions[a]
			if d == shapes.DynamicDim {
				return -1
			}
			p *= d
		}
		return p
	}

	batchSize = prod(batchAxes)
	crossSize = prod(crossAxes)
	contractingSize = prod(contractingAxes)
	return
}

func isIdentityPerm(perm []int) bool {
	for i, v := range perm {
		if i != v {
			return false
		}
	}
	return true
}

func encodeDotGeneralOutputDims(
	outDims []int,
	lhsNode *graphNode, lhsBatchAxes, lhsContractingAxes []int,
	rhsNode *graphNode, rhsBatchAxes, rhsContractingAxes []int,
) []int {
	if !slices.Contains(outDims, shapes.DynamicDim) {
		return outDims
	}
	lhsSet := make(map[int]bool, len(lhsBatchAxes)+len(lhsContractingAxes))
	for _, a := range lhsBatchAxes {
		lhsSet[a] = true
	}
	for _, a := range lhsContractingAxes {
		lhsSet[a] = true
	}
	rhsSet := make(map[int]bool, len(rhsBatchAxes)+len(rhsContractingAxes))
	for _, a := range rhsBatchAxes {
		rhsSet[a] = true
	}
	for _, a := range rhsContractingAxes {
		rhsSet[a] = true
	}

	res := make([]int, 0, len(outDims))
	enc := func(node *graphNode, axis int) int {
		d := node.shape.Dimensions[axis]
		if d != shapes.DynamicDim {
			return d
		}
		return -(2 + int(node.id)*16 + axis)
	}
	for _, a := range lhsBatchAxes {
		res = append(res, enc(lhsNode, a))
	}
	for i := range lhsNode.shape.Rank() {
		if !lhsSet[i] {
			res = append(res, enc(lhsNode, i))
		}
	}
	for i := range rhsNode.shape.Rank() {
		if !rhsSet[i] {
			res = append(res, enc(rhsNode, i))
		}
	}
	return res
}
