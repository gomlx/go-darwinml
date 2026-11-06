// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mlx

import (
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
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

	// Fast path: standard 2D matmul [M, K] x [K, N] -> [M, N].
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
		return &graphNode{id: resID, shape: outShape, owner: f}, nil
	}

	// General path: normalize to rank-3 [B, M, K] x [B, K, N] -> [B, M, N] -> reshape to outShape.
	lhsBatchSize, lhsCrossSize, contractingSize, lhsPerm := classifyAxes(lhsNode.shape, lhsBatch, lhsContracting)
	_, rhsCrossSize, _, rhsPerm := classifyAxes(rhsNode.shape, rhsBatch, rhsContracting)

	batchSize := lhsBatchSize
	if len(lhsBatch) == 0 {
		batchSize = 1
	}

	if !isIdentityPerm(lhsPerm) {
		lhsID, err = f.builder.tape.AddTranspose(lhsID, lhsPerm)
		if err != nil {
			return nil, errors.Wrap(err, "DotGeneral: lhs transpose")
		}
	}
	if !isIdentityPerm(rhsPerm) {
		rhsID, err = f.builder.tape.AddTranspose(rhsID, rhsPerm)
		if err != nil {
			return nil, errors.Wrap(err, "DotGeneral: rhs transpose")
		}
	}

	// Reshape lhs to [B, M, K].
	lhsID, err = f.builder.tape.AddReshape(lhsID, []int{batchSize, lhsCrossSize, contractingSize})
	if err != nil {
		return nil, errors.Wrap(err, "DotGeneral: lhs reshape")
	}

	// Reshape rhs to [B, N, K] (since rhsPerm orders [batch, cross, contracting]).
	rhsID, err = f.builder.tape.AddReshape(rhsID, []int{batchSize, rhsCrossSize, contractingSize})
	if err != nil {
		return nil, errors.Wrap(err, "DotGeneral: rhs reshape")
	}

	// Transpose rhs from [B, N, K] to [B, K, N].
	rhsID, err = f.builder.tape.AddTranspose(rhsID, []int{0, 2, 1})
	if err != nil {
		return nil, errors.Wrap(err, "DotGeneral: rhs transpose for matmul")
	}

	// Batched matmul: [B, M, K] x [B, K, N] -> [B, M, N].
	resID, err := f.builder.tape.AddMatmul(lhsID, rhsID)
	if err != nil {
		return nil, errors.Wrap(err, "DotGeneral: matmul")
	}

	// Reshape from [B, M, N] to outShape.Dimensions, encoding any dynamic dimensions
	// by referencing the corresponding lhs or rhs input node and axis.
	outReshapeDims := encodeDotGeneralOutputDims(
		outShape.Dimensions,
		lhsNode, lhsBatch, lhsContracting,
		rhsNode, rhsBatch, rhsContracting)
	resID, err = f.builder.tape.AddReshape(resID, outReshapeDims)
	if err != nil {
		return nil, errors.Wrap(err, "DotGeneral: output reshape")
	}

	if outDType != computeDType {
		resID, err = f.builder.tape.AddConvertDType(resID, dtypeToBridgeDType(outDType))
		if err != nil {
			return nil, errors.Wrap(err, "DotGeneral: convert output")
		}
	}

	return &graphNode{id: resID, shape: outShape, owner: f}, nil
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
		if ld != shapes.DynamicDim && rd != shapes.DynamicDim && ld != rd {
			return shapes.Invalid(), errors.Errorf("DotGeneral: batch axis %d dimension mismatch (%d vs %d)", i, ld, rd)
		}
	}
	for i, la := range lhsContractingAxes {
		ra := rhsContractingAxes[i]
		ld := lhsShape.Dimensions[la]
		rd := rhsShape.Dimensions[ra]
		if ld != shapes.DynamicDim && rd != shapes.DynamicDim && ld != rd {
			return shapes.Invalid(), errors.Errorf("DotGeneral: contracting axis %d dimension mismatch (%d vs %d)", i, ld, rd)
		}
	}

	var dims []int
	var names []string
	hasNames := lhsShape.AxisNames != nil || rhsShape.AxisNames != nil

	for _, a := range lhsBatchAxes {
		dims = append(dims, lhsShape.Dimensions[a])
		if hasNames {
			names = append(names, lhsShape.AxisName(a))
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
	if len(lhsBatchAxes) != 0 || len(rhsBatchAxes) != 0 {
		return false
	}
	if lhsShape.Rank() < 2 || rhsShape.Rank() != 2 {
		return false
	}
	if len(lhsContractingAxes) != 1 || len(rhsContractingAxes) != 1 {
		return false
	}
	return lhsContractingAxes[0] == lhsShape.Rank()-1 && rhsContractingAxes[0] == 0
}

func classifyAxes(shape shapes.Shape, batchAxes, contractingAxes []int) (batchSize, crossSize, contractingSize int, perm []int) {
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

	perm = make([]int, 0, rank)
	perm = append(perm, batchAxes...)
	perm = append(perm, crossAxes...)
	perm = append(perm, contractingAxes...)

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
