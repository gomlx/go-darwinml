// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mlx

import (
	"runtime"
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/notimplemented"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/sets"
	"github.com/gomlx/go-darwinml/internal/mlxbridge"
	"github.com/pkg/errors"
)

// graphNode represents a value recorded in the MLX computation tape.
type graphNode struct {
	id    mlxbridge.NodeID
	shape shapes.Shape
	name  string
	owner *Function

	// Optional op metadata used to pattern-match simple comparator closures in Sort:
	hasUnaryOp bool
	unaryOp    mlxbridge.UnaryOp
	unaryIn    *graphNode

	hasBinOp bool
	binaryOp mlxbridge.BinaryOp
	lhs, rhs *graphNode
}

type dynamicAxisSource struct {
	nodeID mlxbridge.NodeID
	axis   int
}

// Function implements compute.Function for the MLX backend.
type Function struct {
	notimplemented.Function

	builder            *Builder
	name               string
	parent             *Function
	returned           bool
	params             []*graphNode
	outputs            []*graphNode
	knownDynamicAxes   sets.Set[string]
	dynamicAxisSources map[string]dynamicAxisSource
}

// Verify interface compliance.
var _ compute.Function = (*Function)(nil)

func newFunction(builder *Builder, name string, parent *Function) *Function {
	fn := &Function{
		builder: builder,
		name:    name,
		parent:  parent,
	}
	fn.Function.ErrFn = func(op compute.OpType) error {
		return errors.Wrapf(compute.ErrNotImplemented, "op %s not implemented in %q backend", op, BackendName)
	}
	return fn
}

// Builder returns the Builder that created this function.
func (f *Function) Builder() compute.Builder { return f.builder }

// Shape returns the shape of a value computed in this function.
func (f *Function) Shape(v compute.Value) (shapes.Shape, error) {
	return f.builder.OpShape(v)
}

// Name returns the function name.
func (f *Function) Name() string { return f.name }

// Parent returns the parent function.
func (f *Function) Parent() compute.Function {
	if f.parent == nil {
		return nil
	}
	return f.parent
}

// Closure creates a closure function.
func (f *Function) Closure() (compute.Function, error) {
	return newFunction(f.builder, "", f), nil
}

// Parameter creates an input parameter for this function.
func (f *Function) Parameter(name string, shape shapes.Shape, _ *compute.ShardingSpec) (compute.Value, error) {
	if err := f.builder.backend.checkDTypeSupported(shape.DType); err != nil {
		return nil, err
	}
	if shape.IsDynamic() && !f.builder.backend.dynamicShapes {
		return nil, errors.Wrapf(compute.ErrNotImplemented,
			"Parameter(%q): dynamic shape %s is not supported in %q backend (use %q to enable dynamic shapes)",
			name, shape, BackendName, "mlx:dynamic")
	}
	var id mlxbridge.NodeID
	var err error
	if f == f.builder.mainFn {
		id, err = f.builder.tape.AddParameter(dtypeToBridgeDType(shape.DType), shape.Dimensions)
	} else {
		id, err = f.builder.tape.AddSubParameter(dtypeToBridgeDType(shape.DType), shape.Dimensions)
	}
	if err != nil {
		return nil, errors.Wrapf(err, "Parameter(%q)", name)
	}
	node := &graphNode{
		id:    id,
		shape: shape.Clone(),
		name:  name,
		owner: f,
	}
	f.params = append(f.params, node)
	f.registerDynamicAxes(node)
	return node, nil
}

// Constant creates a constant tensor in the function from a flat Go slice and dimensions.
func (f *Function) Constant(flat any, dimensions ...int) (compute.Value, error) {
	shape, err := shapeFromFlat(flat, dimensions)
	if err != nil {
		return nil, errors.Wrap(err, "Constant")
	}
	if err := f.builder.backend.checkDTypeSupported(shape.DType); err != nil {
		return nil, err
	}
	ptr, nbytes, err := flatSlicePtr(flat, shape)
	if err != nil {
		return nil, errors.Wrap(err, "Constant")
	}
	id, err := f.builder.tape.AddConstant(ptr, nbytes, dtypeToBridgeDType(shape.DType), shape.Dimensions)
	runtime.KeepAlive(flat)
	if err != nil {
		return nil, errors.Wrap(err, "Constant")
	}
	return &graphNode{
		id:    id,
		shape: shape,
		owner: f,
	}, nil
}

// Return marks the outputs of this function.
func (f *Function) Return(outputs []compute.Value, _ []*compute.ShardingSpec) error {
	if f.returned {
		return errors.New("Return: function already returned")
	}
	if len(outputs) == 0 {
		return errors.New("Return: at least one output is required")
	}
	nodes, err := f.resolveNodes("Return", outputs...)
	if err != nil {
		return err
	}
	f.outputs = nodes
	f.returned = true
	return nil
}

// Call invokes a target sub-function by inlining its recorded operations into the caller's tape.
func (f *Function) Call(target compute.Function, inputs ...compute.Value) ([]compute.Value, error) {
	targetFn, ok := target.(*Function)
	if !ok || targetFn == nil {
		return nil, errors.Errorf("Call: expected *mlx.Function, got %T", target)
	}
	if targetFn.builder != f.builder {
		return nil, errors.New("Call: target function belongs to a different Builder")
	}
	if targetFn.parent != nil {
		return nil, errors.New("Call: cannot call a closure; use Builder.NewFunction for callable functions")
	}
	if targetFn == f.builder.mainFn {
		return nil, errors.New("Call: cannot call Main function")
	}
	if !targetFn.returned {
		return nil, errors.Errorf("Call: target function %q has not called Return()", targetFn.name)
	}
	if len(inputs) != len(targetFn.params) {
		return nil, errors.Errorf("Call(%q): expected %d inputs, got %d",
			targetFn.name, len(targetFn.params), len(inputs))
	}

	argNodes, err := f.resolveNodes("Call", inputs...)
	if err != nil {
		return nil, err
	}

	paramIDs := make([]mlxbridge.NodeID, len(targetFn.params))
	argIDs := make([]mlxbridge.NodeID, len(argNodes))
	for i, p := range targetFn.params {
		if !argNodes[i].shape.Equal(p.shape) {
			return nil, errors.Errorf("Call(%q): input #%d shape %s does not match parameter %q shape %s",
				targetFn.name, i, argNodes[i].shape, p.name, p.shape)
		}
		paramIDs[i] = p.id
		argIDs[i] = argNodes[i].id
	}

	outIDs := make([]mlxbridge.NodeID, len(targetFn.outputs))
	for i, out := range targetFn.outputs {
		outIDs[i] = out.id
	}

	clonedIDs, err := f.builder.tape.Inline(paramIDs, argIDs, outIDs)
	if err != nil {
		return nil, errors.Wrapf(err, "Call(%q)", targetFn.name)
	}

	results := make([]compute.Value, len(clonedIDs))
	for i, id := range clonedIDs {
		results[i] = &graphNode{
			id:    id,
			shape: targetFn.outputs[i].shape.Clone(),
			owner: f,
		}
	}
	return results, nil
}

// matchSortComparator inspects a Sort comparator closure and returns (ascending, ok).
// It recognizes standard ascending (<, <=) and descending (>, >=) comparisons on the first input pair
// (params[0] vs params[1]), including optional Identity or LogicalNot wrappers.
func matchSortComparator(compFn *Function) (ascending bool, ok bool) {
	if len(compFn.params) < 2 || len(compFn.outputs) != 1 {
		return false, false
	}
	p0, p1 := compFn.params[0], compFn.params[1]
	cur := compFn.outputs[0]
	invert := false
	for cur != nil && cur.hasUnaryOp {
		switch cur.unaryOp {
		case mlxbridge.UnaryIdentity:
			cur = cur.unaryIn
		case mlxbridge.UnaryLogicalNot:
			invert = !invert
			cur = cur.unaryIn
		default:
			return false, false
		}
	}
	if cur == nil || !cur.hasBinOp {
		return false, false
	}
	var asc bool
	if cur.lhs == p0 && cur.rhs == p1 {
		switch cur.binaryOp {
		case mlxbridge.BinaryLessThan, mlxbridge.BinaryLessOrEqual:
			asc = true
		case mlxbridge.BinaryGreaterThan, mlxbridge.BinaryGreaterOrEqual:
			asc = false
		default:
			return false, false
		}
	} else if cur.lhs == p1 && cur.rhs == p0 {
		switch cur.binaryOp {
		case mlxbridge.BinaryGreaterThan, mlxbridge.BinaryGreaterOrEqual:
			asc = true
		case mlxbridge.BinaryLessThan, mlxbridge.BinaryLessOrEqual:
			asc = false
		default:
			return false, false
		}
	} else {
		return false, false
	}
	if invert {
		asc = !asc
	}
	return asc, true
}

// Sort sorts one or more tensors along axis using comparator.
// Because MLX does not support arbitrary user-defined Metal comparator kernels, the MLX backend
// pattern-matches standard ascending/descending comparators on the primary key tensor and lowers
// them to native GPU mlx::core::sort or mlx::core::argsort + mlx::core::take_along_axis.
func (f *Function) Sort(comparator compute.Function, axis int, _ bool, inputs ...compute.Value) ([]compute.Value, error) {
	if len(inputs) == 0 {
		return nil, errors.New("Sort: at least one input tensor is required")
	}
	compFn, ok := comparator.(*Function)
	if !ok || compFn == nil {
		return nil, errors.Errorf("Sort: expected *mlx.Function comparator, got %T", comparator)
	}
	if compFn.parent == nil {
		return nil, errors.New("Sort: comparator must be a closure created with Closure()")
	}
	if !compFn.returned {
		return nil, errors.New("Sort: comparator closure has not called Return()")
	}

	inNodes, err := f.resolveNodes("Sort", inputs...)
	if err != nil {
		return nil, err
	}

	rank := inNodes[0].shape.Rank()
	if axis < 0 {
		axis += rank
	}
	if axis < 0 || axis >= rank {
		return nil, errors.Errorf("Sort: axis %d out of bounds for shape %s", axis, inNodes[0].shape)
	}
	for i, n := range inNodes {
		if n.shape.IsDynamic() {
			return nil, errors.Wrapf(compute.ErrNotImplemented,
				"Sort: dynamic shape %s is not supported in %q backend", n.shape, BackendName)
		}
		if !n.shape.EqualDimensions(inNodes[0].shape) {
			return nil, errors.Errorf("Sort: input #%d shape %s does not match input #0 dimensions %s",
				i, n.shape, inNodes[0].shape)
		}
	}
	if len(compFn.params) != 2*len(inNodes) {
		return nil, errors.Errorf("Sort: comparator has %d parameters, expected %d (2 per input tensor)",
			len(compFn.params), 2*len(inNodes))
	}
	for i, n := range inNodes {
		pLHS := compFn.params[2*i]
		pRHS := compFn.params[2*i+1]
		if !pLHS.shape.IsScalar() || pLHS.shape.DType != n.shape.DType ||
			!pRHS.shape.IsScalar() || pRHS.shape.DType != n.shape.DType {
			return nil, errors.Errorf("Sort: comparator parameters %d,%d must be scalars of dtype %s, got %s and %s",
				2*i, 2*i+1, n.shape.DType, pLHS.shape, pRHS.shape)
		}
	}
	if len(compFn.outputs) != 1 || !compFn.outputs[0].shape.IsScalar() || compFn.outputs[0].shape.DType != dtypes.Bool {
		return nil, errors.New("Sort: comparator must return a single scalar Bool value")
	}

	keyDType := inNodes[0].shape.DType
	if keyDType.IsComplex() {
		return nil, errors.Wrapf(compute.ErrNotImplemented,
			"Sort: Complex64 key tensor is not supported in %q backend", BackendName)
	}

	ascending, matched := matchSortComparator(compFn)
	if !matched {
		return nil, errors.Wrapf(compute.ErrNotImplemented,
			"Sort: %q backend only supports standard ascending (<, <=) or descending (>, >=) comparators on the primary key tensor; please refactor custom comparator logic",
			BackendName)
	}

	if inNodes[0].shape.Dimensions[axis] <= 1 {
		outs := make([]compute.Value, len(inNodes))
		for i, n := range inNodes {
			outs[i], err = f.Identity(n)
			if err != nil {
				return nil, err
			}
		}
		return outs, nil
	}

	// Single-tensor ascending sort: use mlx::core::sort directly.
	if ascending && len(inNodes) == 1 {
		keyID := inNodes[0].id
		if keyDType == dtypes.Bool {
			keyID, err = f.builder.tape.AddConvertDType(keyID, mlxbridge.DTypeUint8)
			if err != nil {
				return nil, errors.Wrap(err, "Sort")
			}
		}
		sortedID, err := f.builder.tape.AddSort(keyID, axis)
		if err != nil {
			return nil, errors.Wrap(err, "Sort")
		}
		if keyDType == dtypes.Bool {
			sortedID, err = f.builder.tape.AddConvertDType(sortedID, mlxbridge.DTypeBool)
			if err != nil {
				return nil, errors.Wrap(err, "Sort")
			}
		}
		return []compute.Value{&graphNode{id: sortedID, shape: inNodes[0].shape.Clone(), owner: f}}, nil
	}

	// General path (descending or multi-tensor sort): compute stable argsort indices on key
	// (applying an order-reversing bijection for descending sort so tie stability is preserved).
	keyID := inNodes[0].id
	if !ascending {
		switch {
		case keyDType.IsFloat():
			keyID, err = f.builder.tape.AddUnary(mlxbridge.UnaryNeg, keyID)
		case keyDType.IsInt():
			keyID, err = f.builder.tape.AddUnary(mlxbridge.UnaryBitwiseNot, keyID)
		case keyDType == dtypes.Bool:
			keyID, err = f.builder.tape.AddUnary(mlxbridge.UnaryLogicalNot, keyID)
		}
		if err != nil {
			return nil, errors.Wrap(err, "Sort: key transform")
		}
	}
	if keyDType == dtypes.Bool {
		keyID, err = f.builder.tape.AddConvertDType(keyID, mlxbridge.DTypeUint8)
		if err != nil {
			return nil, errors.Wrap(err, "Sort: bool cast")
		}
	}

	indicesID, err := f.builder.tape.AddArgsort(keyID, axis)
	if err != nil {
		return nil, errors.Wrap(err, "Sort: argsort")
	}

	outs := make([]compute.Value, len(inNodes))
	for i, n := range inNodes {
		inID := n.id
		if n.shape.DType == dtypes.Complex64 {
			reVal, err := f.Real(n)
			if err != nil {
				return nil, err
			}
			imVal, err := f.Imag(n)
			if err != nil {
				return nil, err
			}
			reSortedID, err := f.builder.tape.AddTakeAlongAxis(reVal.(*graphNode).id, indicesID, axis)
			if err != nil {
				return nil, err
			}
			imSortedID, err := f.builder.tape.AddTakeAlongAxis(imVal.(*graphNode).id, indicesID, axis)
			if err != nil {
				return nil, err
			}
			cID, err := f.builder.tape.AddBinary(mlxbridge.BinaryComplex, reSortedID, imSortedID)
			if err != nil {
				return nil, err
			}
			outs[i] = &graphNode{id: cID, shape: n.shape.Clone(), owner: f}
			continue
		}
		sortedID, err := f.builder.tape.AddTakeAlongAxis(inID, indicesID, axis)
		if err != nil {
			return nil, errors.Wrapf(err, "Sort: take_along_axis input #%d", i)
		}
		outs[i] = &graphNode{id: sortedID, shape: n.shape.Clone(), owner: f}
	}
	return outs, nil
}

// resolveNode validates that v is a *graphNode belonging to this function or an ancestor.
func (f *Function) resolveNode(v compute.Value) (*graphNode, error) {
	node, ok := v.(*graphNode)
	if !ok || node == nil {
		return nil, errors.Errorf("expected *graphNode, got %T", v)
	}
	if node.owner == nil || node.owner == f {
		return node, nil
	}
	for p := f.parent; p != nil; p = p.parent {
		if node.owner == p {
			return node, nil
		}
	}
	return nil, errors.New("value belongs to an unrelated function scope")
}

// resolveNodes validates a slice of compute.Value as *graphNodes.
func (f *Function) resolveNodes(opName string, values ...compute.Value) ([]*graphNode, error) {
	nodes := make([]*graphNode, len(values))
	for i, v := range values {
		n, err := f.resolveNode(v)
		if err != nil {
			return nil, errors.Wrapf(err, "%s: input #%d", opName, i)
		}
		nodes[i] = n
	}
	return nodes, nil
}

// shapeFromFlat infers the shape (dtype + dimensions) from a flat Go slice.
func shapeFromFlat(flat any, dimensions []int) (shapes.Shape, error) {
	dt, _, err := dtypes.FromAnySlice(flat)
	if err != nil {
		return shapes.Invalid(), err
	}
	return shapes.Make(dt, dimensions...), nil
}

// ===========================================================================
// Identity, Barriers, Type Conversions, and Iota
// ===========================================================================

// Identity returns an op whose output is identical to its input.
func (f *Function) Identity(x compute.Value) (compute.Value, error) {
	node, err := f.resolveNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "Identity")
	}
	id, err := f.builder.tape.AddUnary(mlxbridge.UnaryIdentity, node.id)
	if err != nil {
		return nil, errors.Wrap(err, "Identity")
	}
	return &graphNode{
		id:         id,
		shape:      node.shape.Clone(),
		owner:      f,
		hasUnaryOp: true,
		unaryOp:    mlxbridge.UnaryIdentity,
		unaryIn:    node,
	}, nil
}

// OptimizationBarrier returns values identical to the operands.
func (f *Function) OptimizationBarrier(operands ...compute.Value) ([]compute.Value, error) {
	nodes, err := f.resolveNodes("OptimizationBarrier", operands...)
	if err != nil {
		return nil, err
	}
	outs := make([]compute.Value, len(nodes))
	for i, n := range nodes {
		out, err := f.Identity(n)
		if err != nil {
			return nil, errors.Wrapf(err, "OptimizationBarrier: operand #%d", i)
		}
		outs[i] = out
	}
	return outs, nil
}

// SchedulingBarrier returns a value identical to operand after validating dependencies.
func (f *Function) SchedulingBarrier(operand compute.Value, dependencies ...compute.Value) (compute.Value, error) {
	if _, err := f.resolveNodes("SchedulingBarrier", dependencies...); err != nil {
		return nil, err
	}
	return f.Identity(operand)
}

// ConvertDType converts x to target dtype.
func (f *Function) ConvertDType(x compute.Value, dtype dtypes.DType) (compute.Value, error) {
	if err := f.builder.backend.checkDTypeSupported(dtype); err != nil {
		return nil, errors.Wrap(err, "ConvertDType")
	}
	node, err := f.resolveNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "ConvertDType")
	}
	outShape := node.shape.Clone()
	outShape.DType = dtype
	if node.shape.DType == dtype {
		return f.Identity(node)
	}
	id, err := f.builder.tape.AddConvertDType(node.id, dtypeToBridgeDType(dtype))
	if err != nil {
		return nil, errors.Wrap(err, "ConvertDType")
	}
	return &graphNode{id: id, shape: outShape, owner: f}, nil
}

// Bitcast reinterprets the underlying bits of operand as targetDType.
func (f *Function) Bitcast(operand compute.Value, targetDType dtypes.DType) (compute.Value, error) {
	if err := f.builder.backend.checkDTypeSupported(targetDType); err != nil {
		return nil, errors.Wrap(err, "Bitcast")
	}
	node, err := f.resolveNode(operand)
	if err != nil {
		return nil, errors.Wrap(err, "Bitcast")
	}
	outShape, err := shapeinference.Bitcast(node.shape, targetDType)
	if err != nil {
		return nil, errors.Wrap(err, "Bitcast")
	}
	if outShape.IsDynamic() {
		return nil, errors.Wrapf(compute.ErrNotImplemented,
			"Bitcast: dynamic output shape %s is not supported in %q backend", outShape, BackendName)
	}
	id, err := f.builder.tape.AddBitcast(node.id, dtypeToBridgeDType(targetDType), outShape.Dimensions)
	if err != nil {
		return nil, errors.Wrap(err, "Bitcast")
	}
	return &graphNode{id: id, shape: outShape, owner: f}, nil
}

// Iota creates a tensor of the given shape with increasing values starting at 0 along iotaAxis.
func (f *Function) Iota(shape shapes.Shape, iotaAxis int) (compute.Value, error) {
	if err := f.builder.backend.checkDTypeSupported(shape.DType); err != nil {
		return nil, errors.Wrap(err, "Iota")
	}
	if shape.IsDynamic() {
		return nil, errors.Wrapf(compute.ErrNotImplemented,
			"Iota: dynamic shape %s is not supported in %q backend", shape, BackendName)
	}
	if shape.Rank() == 0 {
		return nil, errors.New("Iota: shape must have rank >= 1")
	}
	if iotaAxis < 0 {
		iotaAxis += shape.Rank()
	}
	if iotaAxis < 0 || iotaAxis >= shape.Rank() {
		return nil, errors.Errorf("Iota: iotaAxis %d out of bounds for rank %d", iotaAxis, shape.Rank())
	}
	id, err := f.builder.tape.AddIota(dtypeToBridgeDType(shape.DType), shape.Dimensions, iotaAxis)
	if err != nil {
		return nil, errors.Wrap(err, "Iota")
	}
	return &graphNode{id: id, shape: shape.Clone(), owner: f}, nil
}

// ===========================================================================
// Unary Operations
// ===========================================================================

func (f *Function) unaryOp(opName string, opType compute.OpType, mlxOp mlxbridge.UnaryOp, x compute.Value) (compute.Value, error) {
	node, err := f.resolveNode(x)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}
	outShape, err := shapeinference.UnaryOp(opType, node.shape)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}
	id, err := f.builder.tape.AddUnary(mlxOp, node.id)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}
	return &graphNode{
		id:         id,
		shape:      outShape.Clone(),
		owner:      f,
		hasUnaryOp: true,
		unaryOp:    mlxOp,
		unaryIn:    node,
	}, nil
}

func (f *Function) Abs(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Abs", compute.OpTypeAbs, mlxbridge.UnaryAbs, x)
}

func (f *Function) Neg(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Neg", compute.OpTypeNeg, mlxbridge.UnaryNeg, x)
}

func (f *Function) Sqrt(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Sqrt", compute.OpTypeSqrt, mlxbridge.UnarySqrt, x)
}

func (f *Function) Rsqrt(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Rsqrt", compute.OpTypeRsqrt, mlxbridge.UnaryRsqrt, x)
}

func (f *Function) Exp(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Exp", compute.OpTypeExp, mlxbridge.UnaryExp, x)
}

func (f *Function) Expm1(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Expm1", compute.OpTypeExpm1, mlxbridge.UnaryExpm1, x)
}

func (f *Function) Log(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Log", compute.OpTypeLog, mlxbridge.UnaryLog, x)
}

func (f *Function) Log1p(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Log1p", compute.OpTypeLog1p, mlxbridge.UnaryLog1p, x)
}

func (f *Function) Sin(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Sin", compute.OpTypeSin, mlxbridge.UnarySin, x)
}

func (f *Function) Cos(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Cos", compute.OpTypeCos, mlxbridge.UnaryCos, x)
}

func (f *Function) Tanh(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Tanh", compute.OpTypeTanh, mlxbridge.UnaryTanh, x)
}

func (f *Function) Logistic(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Logistic", compute.OpTypeLogistic, mlxbridge.UnaryLogistic, x)
}

func (f *Function) Erf(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Erf", compute.OpTypeErf, mlxbridge.UnaryErf, x)
}

func (f *Function) Floor(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Floor", compute.OpTypeFloor, mlxbridge.UnaryFloor, x)
}

func (f *Function) Ceil(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Ceil", compute.OpTypeCeil, mlxbridge.UnaryCeil, x)
}

func (f *Function) Round(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Round", compute.OpTypeRound, mlxbridge.UnaryRound, x)
}

func (f *Function) Sign(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Sign", compute.OpTypeSign, mlxbridge.UnarySign, x)
}

func (f *Function) LogicalNot(x compute.Value) (compute.Value, error) {
	return f.unaryOp("LogicalNot", compute.OpTypeLogicalNot, mlxbridge.UnaryLogicalNot, x)
}

func (f *Function) BitwiseNot(x compute.Value) (compute.Value, error) {
	return f.unaryOp("BitwiseNot", compute.OpTypeBitwiseNot, mlxbridge.UnaryBitwiseNot, x)
}

func (f *Function) BitCount(operand compute.Value) (compute.Value, error) {
	return f.unaryOp("BitCount", compute.OpTypeBitCount, mlxbridge.UnaryBitCount, operand)
}

func (f *Function) Clz(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Clz", compute.OpTypeClz, mlxbridge.UnaryClz, x)
}

func (f *Function) IsFinite(x compute.Value) (compute.Value, error) {
	node, err := f.resolveNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "IsFinite")
	}
	if !node.shape.DType.IsFloat() && !node.shape.DType.IsComplex() {
		return nil, errors.Errorf("IsFinite requires a float or complex input, got %s", node.shape)
	}
	outShape := node.shape.Clone()
	outShape.DType = dtypes.Bool
	id, err := f.builder.tape.AddUnary(mlxbridge.UnaryIsFinite, node.id)
	if err != nil {
		return nil, errors.Wrap(err, "IsFinite")
	}
	return &graphNode{id: id, shape: outShape, owner: f}, nil
}

func (f *Function) IsNaN(x compute.Value) (compute.Value, error) {
	node, err := f.resolveNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "IsNaN")
	}
	if !node.shape.DType.IsFloat() && !node.shape.DType.IsComplex() {
		return nil, errors.Errorf("IsNaN requires a float or complex input, got %s", node.shape)
	}
	outShape := node.shape.Clone()
	outShape.DType = dtypes.Bool
	id, err := f.builder.tape.AddUnary(mlxbridge.UnaryIsNaN, node.id)
	if err != nil {
		return nil, errors.Wrap(err, "IsNaN")
	}
	return &graphNode{id: id, shape: outShape, owner: f}, nil
}

func (f *Function) Conj(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Conj", compute.OpTypeConj, mlxbridge.UnaryConj, x)
}

func (f *Function) Real(x compute.Value) (compute.Value, error) {
	node, err := f.resolveNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "Real")
	}
	outShape, err := shapeinference.UnaryOp(compute.OpTypeReal, node.shape)
	if err != nil {
		return nil, errors.Wrap(err, "Real")
	}
	outShape = outShape.Clone()
	outShape.DType = outShape.DType.RealDType()
	id, err := f.builder.tape.AddUnary(mlxbridge.UnaryReal, node.id)
	if err != nil {
		return nil, errors.Wrap(err, "Real")
	}
	return &graphNode{id: id, shape: outShape, owner: f}, nil
}

func (f *Function) Imag(x compute.Value) (compute.Value, error) {
	node, err := f.resolveNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "Imag")
	}
	outShape, err := shapeinference.UnaryOp(compute.OpTypeImag, node.shape)
	if err != nil {
		return nil, errors.Wrap(err, "Imag")
	}
	outShape = outShape.Clone()
	outShape.DType = outShape.DType.RealDType()
	id, err := f.builder.tape.AddUnary(mlxbridge.UnaryImag, node.id)
	if err != nil {
		return nil, errors.Wrap(err, "Imag")
	}
	return &graphNode{id: id, shape: outShape, owner: f}, nil
}

// ===========================================================================
// Binary and Comparison Operations
// ===========================================================================

func (f *Function) binaryOp(opName string, opType compute.OpType, mlxOp mlxbridge.BinaryOp, lhs, rhs compute.Value) (compute.Value, error) {
	nodes, err := f.resolveNodes(opName, lhs, rhs)
	if err != nil {
		return nil, err
	}
	outShape, err := shapeinference.BinaryOp(opType, nodes[0].shape, nodes[1].shape)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}
	id, err := f.builder.tape.AddBinary(mlxOp, nodes[0].id, nodes[1].id)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}
	return &graphNode{id: id, shape: outShape, owner: f}, nil
}

func (f *Function) Add(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("Add", compute.OpTypeAdd, mlxbridge.BinaryAdd, lhs, rhs)
}

func (f *Function) Sub(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("Sub", compute.OpTypeSub, mlxbridge.BinarySub, lhs, rhs)
}

func (f *Function) Mul(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("Mul", compute.OpTypeMul, mlxbridge.BinaryMul, lhs, rhs)
}

func (f *Function) Div(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("Div", compute.OpTypeDiv, mlxbridge.BinaryDiv, lhs, rhs)
}

func (f *Function) Rem(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("Rem", compute.OpTypeRem, mlxbridge.BinaryRem, lhs, rhs)
}

func (f *Function) Pow(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("Pow", compute.OpTypePow, mlxbridge.BinaryPow, lhs, rhs)
}

func (f *Function) Max(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("Max", compute.OpTypeMax, mlxbridge.BinaryMax, lhs, rhs)
}

func (f *Function) Min(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("Min", compute.OpTypeMin, mlxbridge.BinaryMin, lhs, rhs)
}

func (f *Function) Atan2(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("Atan2", compute.OpTypeAtan2, mlxbridge.BinaryAtan2, lhs, rhs)
}

func (f *Function) LogicalAnd(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("LogicalAnd", compute.OpTypeLogicalAnd, mlxbridge.BinaryLogicalAnd, lhs, rhs)
}

func (f *Function) LogicalOr(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("LogicalOr", compute.OpTypeLogicalOr, mlxbridge.BinaryLogicalOr, lhs, rhs)
}

func (f *Function) LogicalXor(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("LogicalXor", compute.OpTypeLogicalXor, mlxbridge.BinaryLogicalXor, lhs, rhs)
}

func (f *Function) BitwiseAnd(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("BitwiseAnd", compute.OpTypeBitwiseAnd, mlxbridge.BinaryBitwiseAnd, lhs, rhs)
}

func (f *Function) BitwiseOr(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("BitwiseOr", compute.OpTypeBitwiseOr, mlxbridge.BinaryBitwiseOr, lhs, rhs)
}

func (f *Function) BitwiseXor(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("BitwiseXor", compute.OpTypeBitwiseXor, mlxbridge.BinaryBitwiseXor, lhs, rhs)
}

func (f *Function) ShiftLeft(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("ShiftLeft", compute.OpTypeShiftLeft, mlxbridge.BinaryShiftLeft, lhs, rhs)
}

func (f *Function) ShiftRightArithmetic(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("ShiftRightArithmetic", compute.OpTypeShiftRightArithmetic, mlxbridge.BinaryShiftRightArithmetic, lhs, rhs)
}

func (f *Function) ShiftRightLogical(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("ShiftRightLogical", compute.OpTypeShiftRightLogical, mlxbridge.BinaryShiftRightLogical, lhs, rhs)
}

func (f *Function) Complex(lhs, rhs compute.Value) (compute.Value, error) {
	nodes, err := f.resolveNodes("Complex", lhs, rhs)
	if err != nil {
		return nil, err
	}
	if nodes[0].shape.DType != dtypes.Float32 || nodes[1].shape.DType != dtypes.Float32 {
		return nil, errors.Errorf("Complex requires Float32 real and imaginary parts in %q backend, got %s and %s",
			BackendName, nodes[0].shape, nodes[1].shape)
	}
	outShape, err := shapeinference.BinaryOp(compute.OpTypeAdd, nodes[0].shape, nodes[1].shape)
	if err != nil {
		return nil, errors.Wrap(err, "Complex")
	}
	outShape = outShape.Clone()
	outShape.DType = dtypes.Complex64
	id, err := f.builder.tape.AddBinary(mlxbridge.BinaryComplex, nodes[0].id, nodes[1].id)
	if err != nil {
		return nil, errors.Wrap(err, "Complex")
	}
	return &graphNode{id: id, shape: outShape, owner: f}, nil
}

func (f *Function) comparisonOp(opName string, opType compute.OpType, mlxOp mlxbridge.BinaryOp, lhs, rhs compute.Value) (compute.Value, error) {
	nodes, err := f.resolveNodes(opName, lhs, rhs)
	if err != nil {
		return nil, err
	}
	outShape, err := shapeinference.ComparisonOp(opType, nodes[0].shape, nodes[1].shape)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}
	id, err := f.builder.tape.AddBinary(mlxOp, nodes[0].id, nodes[1].id)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}
	return &graphNode{
		id:       id,
		shape:    outShape,
		owner:    f,
		hasBinOp: true,
		binaryOp: mlxOp,
		lhs:      nodes[0],
		rhs:      nodes[1],
	}, nil
}

func (f *Function) Equal(lhs, rhs compute.Value) (compute.Value, error) {
	return f.comparisonOp("Equal", compute.OpTypeEqual, mlxbridge.BinaryEqual, lhs, rhs)
}

func (f *Function) NotEqual(lhs, rhs compute.Value) (compute.Value, error) {
	return f.comparisonOp("NotEqual", compute.OpTypeNotEqual, mlxbridge.BinaryNotEqual, lhs, rhs)
}

func (f *Function) LessThan(lhs, rhs compute.Value) (compute.Value, error) {
	return f.comparisonOp("LessThan", compute.OpTypeLessThan, mlxbridge.BinaryLessThan, lhs, rhs)
}

func (f *Function) LessOrEqual(lhs, rhs compute.Value) (compute.Value, error) {
	return f.comparisonOp("LessOrEqual", compute.OpTypeLessOrEqual, mlxbridge.BinaryLessOrEqual, lhs, rhs)
}

func (f *Function) GreaterThan(lhs, rhs compute.Value) (compute.Value, error) {
	return f.comparisonOp("GreaterThan", compute.OpTypeGreaterThan, mlxbridge.BinaryGreaterThan, lhs, rhs)
}

func (f *Function) GreaterOrEqual(lhs, rhs compute.Value) (compute.Value, error) {
	return f.comparisonOp("GreaterOrEqual", compute.OpTypeGreaterOrEqual, mlxbridge.BinaryGreaterOrEqual, lhs, rhs)
}

func (f *Function) EqualTotalOrder(lhs, rhs compute.Value) (compute.Value, error) {
	return f.Equal(lhs, rhs)
}

func (f *Function) NotEqualTotalOrder(lhs, rhs compute.Value) (compute.Value, error) {
	return f.NotEqual(lhs, rhs)
}

func (f *Function) LessThanTotalOrder(lhs, rhs compute.Value) (compute.Value, error) {
	return f.LessThan(lhs, rhs)
}

func (f *Function) LessOrEqualTotalOrder(lhs, rhs compute.Value) (compute.Value, error) {
	return f.LessOrEqual(lhs, rhs)
}

func (f *Function) GreaterThanTotalOrder(lhs, rhs compute.Value) (compute.Value, error) {
	return f.GreaterThan(lhs, rhs)
}

func (f *Function) GreaterOrEqualTotalOrder(lhs, rhs compute.Value) (compute.Value, error) {
	return f.GreaterOrEqual(lhs, rhs)
}

// ===========================================================================
// Shape Manipulation Operations
// ===========================================================================

// rootFunction returns the root Function of this function's scope hierarchy.
func (f *Function) rootFunction() *Function {
	root := f
	for root.parent != nil {
		root = root.parent
	}
	return root
}

// registerDynamicAxes records named dynamic axes introduced by n on the root function.
func (f *Function) registerDynamicAxes(n *graphNode) {
	if n == nil || !n.shape.IsDynamic() {
		return
	}
	root := f.rootFunction()
	if root.knownDynamicAxes == nil {
		root.knownDynamicAxes = sets.Make[string]()
	}
	if root.dynamicAxisSources == nil {
		root.dynamicAxisSources = make(map[string]dynamicAxisSource)
	}
	for axis, dim := range n.shape.Dimensions {
		if dim != shapes.DynamicDim {
			continue
		}
		name := n.shape.AxisName(axis)
		if name == "" {
			continue
		}
		root.knownDynamicAxes.Insert(name)
		if _, exists := root.dynamicAxisSources[name]; !exists {
			root.dynamicAxisSources[name] = dynamicAxisSource{nodeID: n.id, axis: axis}
		}
	}
}

// knownDynamicAxisNames returns all named dynamic axes from this function's parameters and derived ops.
func (f *Function) knownDynamicAxisNames() sets.Set[string] {
	root := f.rootFunction()
	if root.knownDynamicAxes == nil {
		root.knownDynamicAxes = sets.Make[string]()
	}
	for cur := f; cur != nil; cur = cur.parent {
		for _, p := range cur.params {
			for axis, d := range p.shape.Dimensions {
				if d == shapes.DynamicDim {
					name := p.shape.AxisName(axis)
					if name != "" {
						root.knownDynamicAxes.Insert(name)
					}
				}
			}
		}
	}
	return root.knownDynamicAxes
}

// findDynamicAxisSource returns the node ID and axis index that defines the given dynamic axis name.
func (f *Function) findDynamicAxisSource(axisName string) (mlxbridge.NodeID, int, bool) {
	if axisName == "" {
		return -1, -1, false
	}
	root := f.rootFunction()
	if src, ok := root.dynamicAxisSources[axisName]; ok {
		return src.nodeID, src.axis, true
	}
	for cur := f; cur != nil; cur = cur.parent {
		for _, p := range cur.params {
			for axis, d := range p.shape.Dimensions {
				if d == shapes.DynamicDim && p.shape.AxisName(axis) == axisName {
					return p.id, axis, true
				}
			}
		}
	}
	return -1, -1, false
}

// Reshape reshapes x to the given dimensions.
func (f *Function) Reshape(x compute.Value, dimensions ...int) (compute.Value, error) {
	node, err := f.resolveNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "Reshape")
	}
	outShape, err := shapeinference.Reshape(node.shape, dimensions)
	if err != nil {
		return nil, errors.Wrap(err, "Reshape")
	}
	if outShape.IsDynamic() && !f.builder.backend.dynamicShapes {
		return nil, errors.Wrapf(compute.ErrNotImplemented,
			"Reshape: dynamic shape %s is not supported in %q backend (use %q to enable dynamic shapes)",
			outShape, BackendName, "mlx:dynamic")
	}
	id, err := f.builder.tape.AddReshape(node.id, outShape.Dimensions)
	if err != nil {
		return nil, errors.Wrap(err, "Reshape")
	}
	return &graphNode{id: id, shape: outShape, owner: f}, nil
}

// Transpose permutes the dimensions of x according to permutation.
func (f *Function) Transpose(x compute.Value, permutation ...int) (compute.Value, error) {
	node, err := f.resolveNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "Transpose")
	}
	outShape, err := shapeinference.Transpose(node.shape, permutation)
	if err != nil {
		return nil, errors.Wrap(err, "Transpose")
	}
	id, err := f.builder.tape.AddTranspose(node.id, permutation)
	if err != nil {
		return nil, errors.Wrap(err, "Transpose")
	}
	return &graphNode{id: id, shape: outShape, owner: f}, nil
}

// BroadcastInDim broadcasts operand to outputShape mapping operand's i-th axis to broadcastAxes[i].
func (f *Function) BroadcastInDim(operand compute.Value, outputShape shapes.Shape, broadcastAxes []int) (compute.Value, error) {
	if err := f.builder.backend.checkDTypeSupported(outputShape.DType); err != nil {
		return nil, errors.Wrap(err, "BroadcastInDim")
	}
	if outputShape.IsDynamic() && !f.builder.backend.dynamicShapes {
		return nil, errors.Wrapf(compute.ErrNotImplemented,
			"BroadcastInDim: dynamic shape %s is not supported in %q backend (use %q to enable dynamic shapes)",
			outputShape, BackendName, "mlx:dynamic")
	}
	node, err := f.resolveNode(operand)
	if err != nil {
		return nil, errors.Wrap(err, "BroadcastInDim")
	}
	if node.shape.DType != outputShape.DType {
		return nil, errors.Errorf("BroadcastInDim: operand dtype (%s) and outputShape dtype (%s) must match",
			node.shape.DType, outputShape.DType)
	}
	if err := shapeinference.BroadcastInDim(node.shape, outputShape, broadcastAxes, f.knownDynamicAxisNames()); err != nil {
		return nil, errors.Wrap(err, "BroadcastInDim")
	}

	// Build target shape dimensions for the tape, encoding newly introduced dynamic axes
	// as references to their source parameter node and axis.
	targetDims := slices.Clone(outputShape.Dimensions)
	if outputShape.IsDynamic() {
		preservedDynamic := make(map[int]bool, len(broadcastAxes))
		for inAx, outAx := range broadcastAxes {
			if node.shape.Dimensions[inAx] == shapes.DynamicDim {
				preservedDynamic[outAx] = true
			}
		}
		for outAx, d := range targetDims {
			if d == shapes.DynamicDim && !preservedDynamic[outAx] {
				name := outputShape.AxisName(outAx)
				paramID, paramAx, ok := f.findDynamicAxisSource(name)
				if !ok {
					return nil, errors.Errorf("BroadcastInDim: unknown dynamic axis %q at output axis %d", name, outAx)
				}
				targetDims[outAx] = -(2 + int(paramID)*16 + paramAx)
			}
		}
	}

	id, err := f.builder.tape.AddBroadcastInDim(node.id, targetDims, broadcastAxes)
	if err != nil {
		return nil, errors.Wrap(err, "BroadcastInDim")
	}
	return &graphNode{id: id, shape: outputShape.Clone(), owner: f}, nil
}

// Slice extracts a slice from operand specified by starts, limits, and strides.
func (f *Function) Slice(operand compute.Value, starts, limits, strides []int) (compute.Value, error) {
	node, err := f.resolveNode(operand)
	if err != nil {
		return nil, errors.Wrap(err, "Slice")
	}
	outShape, err := shapeinference.Slice(node.shape, starts, limits, strides)
	if err != nil {
		return nil, errors.Wrap(err, "Slice")
	}
	id, err := f.builder.tape.AddSlice(node.id, starts, limits, strides)
	if err != nil {
		return nil, errors.Wrap(err, "Slice")
	}
	return &graphNode{id: id, shape: outShape, owner: f}, nil
}

// Concatenate joins operands along axis.
func (f *Function) Concatenate(axis int, operands ...compute.Value) (compute.Value, error) {
	nodes, err := f.resolveNodes("Concatenate", operands...)
	if err != nil {
		return nil, err
	}
	if len(nodes) == 1 {
		return f.Identity(nodes[0])
	}
	inputShapes := make([]shapes.Shape, len(nodes))
	inputIDs := make([]mlxbridge.NodeID, len(nodes))
	for i, n := range nodes {
		inputShapes[i] = n.shape
		inputIDs[i] = n.id
	}
	outShape, err := shapeinference.Concatenate(inputShapes, axis)
	if err != nil {
		return nil, errors.Wrap(err, "Concatenate")
	}
	id, err := f.builder.tape.AddConcatenate(inputIDs, axis)
	if err != nil {
		return nil, errors.Wrap(err, "Concatenate")
	}
	return &graphNode{id: id, shape: outShape, owner: f}, nil
}

// Reverse reverses the elements of x along the specified axes.
func (f *Function) Reverse(x compute.Value, axes ...int) (compute.Value, error) {
	node, err := f.resolveNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "Reverse")
	}
	rank := node.shape.Rank()
	normAxes := make([]int, len(axes))
	seen := sets.Make[int](len(axes))
	for i, a := range axes {
		if a < 0 {
			a += rank
		}
		if a < 0 || a >= rank {
			return nil, errors.Errorf("Reverse: axis %d out of bounds for shape %s", axes[i], node.shape)
		}
		if seen.Has(a) {
			return nil, errors.Errorf("Reverse: duplicate axis %d", a)
		}
		seen.Insert(a)
		normAxes[i] = a
	}
	if len(normAxes) == 0 {
		return f.Identity(node)
	}
	id, err := f.builder.tape.AddReverse(node.id, normAxes)
	if err != nil {
		return nil, errors.Wrap(err, "Reverse")
	}
	return &graphNode{id: id, shape: node.shape.Clone(), owner: f}, nil
}

// Pad injects padding on the start, end, or interior of operand.
func (f *Function) Pad(operand, fillValue compute.Value, axesConfig ...compute.PadAxis) (compute.Value, error) {
	opNode, err := f.resolveNode(operand)
	if err != nil {
		return nil, errors.Wrap(err, "Pad: operand")
	}
	fillNode, err := f.resolveNode(fillValue)
	if err != nil {
		return nil, errors.Wrap(err, "Pad: fillValue")
	}
	if fillNode.shape.Rank() != 0 || fillNode.shape.DType != opNode.shape.DType {
		return nil, errors.Errorf("Pad: fillValue must be scalar of dtype %s, got %s", opNode.shape.DType, fillNode.shape)
	}
	outShape, err := shapeinference.Pad(opNode.shape, axesConfig...)
	if err != nil {
		return nil, errors.Wrap(err, "Pad")
	}
	if !outShape.IsDynamic() && outShape.Size() == 0 {
		id, err := f.builder.tape.AddConstant(nil, 0, dtypeToBridgeDType(outShape.DType), outShape.Dimensions)
		if err != nil {
			return nil, errors.Wrap(err, "Pad: zero-sized output")
		}
		return &graphNode{id: id, shape: outShape, owner: f}, nil
	}
	if !opNode.shape.IsDynamic() && opNode.shape.Size() == 0 {
		return f.BroadcastInDim(fillNode, outShape, nil)
	}

	rank := opNode.shape.Rank()
	fullConfig := make([]compute.PadAxis, rank)
	copy(fullConfig, axesConfig)

	curID := opNode.id
	curShape := slices.Clone(opNode.shape.Dimensions)
	didWork := false

	// Step 1: Handle interior padding per axis by reshaping [N] -> [N, 1],
	// padding the new axis with ac.Interior, flattening to [N*(ac.Interior+1)],
	// and slicing off the trailing ac.Interior elements to [(N-1)*ac.Interior + N].
	for i, ac := range fullConfig {
		if ac.Interior > 0 && curShape[i] > 1 {
			n := curShape[i]
			dilation := ac.Interior + 1

			reshapeDims := make([]int, rank+1)
			copy(reshapeDims[:i], curShape[:i])
			reshapeDims[i] = n
			reshapeDims[i+1] = 1
			copy(reshapeDims[i+2:], curShape[i+1:])

			curID, err = f.builder.tape.AddReshape(curID, reshapeDims)
			if err != nil {
				return nil, errors.Wrapf(err, "Pad: interior reshape axis %d", i)
			}

			curID, err = f.builder.tape.AddPad(curID, fillNode.id, []int{i + 1}, []int{0}, []int{ac.Interior})
			if err != nil {
				return nil, errors.Wrapf(err, "Pad: interior pad axis %d", i)
			}

			flatDims := slices.Clone(curShape)
			flatDims[i] = n * dilation
			curID, err = f.builder.tape.AddReshape(curID, flatDims)
			if err != nil {
				return nil, errors.Wrapf(err, "Pad: interior flatten axis %d", i)
			}

			dilatedSize := n + (n-1)*ac.Interior
			starts := make([]int, rank)
			limits := slices.Clone(flatDims)
			strides := make([]int, rank)
			for ax := range rank {
				strides[ax] = 1
			}
			limits[i] = dilatedSize
			curID, err = f.builder.tape.AddSlice(curID, starts, limits, strides)
			if err != nil {
				return nil, errors.Wrapf(err, "Pad: interior slice axis %d", i)
			}

			curShape[i] = dilatedSize
			didWork = true
		}
	}

	// Step 2: Handle positive Start / End padding via Pad first so cropping
	// never produces a zero-sized or negative intermediate tensor when all input
	// elements on an axis are cropped out into opposite-side padding.
	var padAxes, lowPads, highPads []int
	for i, ac := range fullConfig {
		low := max(0, ac.Start)
		high := max(0, ac.End)
		if low > 0 || high > 0 {
			padAxes = append(padAxes, i)
			lowPads = append(lowPads, low)
			highPads = append(highPads, high)
			if curShape[i] != shapes.DynamicDim {
				curShape[i] += low + high
			}
		}
	}
	if len(padAxes) > 0 {
		curID, err = f.builder.tape.AddPad(curID, fillNode.id, padAxes, lowPads, highPads)
		if err != nil {
			return nil, errors.Wrap(err, "Pad: edge pad")
		}
		didWork = true
	}

	// Step 3: Handle negative Start / End padding (cropping) via Slice.
	hasCrop := false
	cropStarts := make([]int, rank)
	cropLimits := slices.Clone(curShape)
	cropStrides := make([]int, rank)
	for i, ac := range fullConfig {
		cropStrides[i] = 1
		if ac.Start < 0 {
			cropStarts[i] = -ac.Start
			hasCrop = true
		}
		if ac.End < 0 {
			cropLimits[i] = curShape[i] + ac.End
			hasCrop = true
		}
	}
	if hasCrop {
		curID, err = f.builder.tape.AddSlice(curID, cropStarts, cropLimits, cropStrides)
		if err != nil {
			return nil, errors.Wrap(err, "Pad: negative padding slice")
		}
		for i := range rank {
			if curShape[i] != shapes.DynamicDim {
				curShape[i] = cropLimits[i] - cropStarts[i]
			}
		}
		didWork = true
	}

	if !didWork {
		return f.Identity(opNode)
	}
	return &graphNode{id: curID, shape: outShape, owner: f}, nil
}

// ===========================================================================
// Selection Operations (Where, Clamp)
// ===========================================================================

// Where selects elements from onTrue or onFalse depending on condition.
func (f *Function) Where(condition, onTrue, onFalse compute.Value) (compute.Value, error) {
	nodes, err := f.resolveNodes("Where", condition, onTrue, onFalse)
	if err != nil {
		return nil, err
	}
	outShape, err := shapeinference.Where(nodes[0].shape, nodes[1].shape, nodes[2].shape)
	if err != nil {
		return nil, errors.Wrap(err, "Where")
	}
	id, err := f.builder.tape.AddWhere(nodes[0].id, nodes[1].id, nodes[2].id)
	if err != nil {
		return nil, errors.Wrap(err, "Where")
	}
	return &graphNode{id: id, shape: outShape, owner: f}, nil
}

// Clamp clamps x element-wise between min and max.
func (f *Function) Clamp(minVal, x, maxVal compute.Value) (compute.Value, error) {
	nodes, err := f.resolveNodes("Clamp", minVal, x, maxVal)
	if err != nil {
		return nil, err
	}
	dt := nodes[1].shape.DType
	if nodes[0].shape.DType != dt || nodes[2].shape.DType != dt {
		return nil, errors.Errorf("Clamp: min (%s), x (%s), and max (%s) must have matching dtypes",
			nodes[0].shape, nodes[1].shape, nodes[2].shape)
	}
	if !dt.IsFloat() && !dt.IsInt() {
		return nil, errors.Errorf("Clamp: requires float or integer dtype, got %s", dt)
	}
	if !nodes[0].shape.IsScalar() && !nodes[0].shape.EqualDimensions(nodes[1].shape) {
		if _, err := shapeinference.BinaryOp(compute.OpTypeMax, nodes[0].shape, nodes[1].shape); err != nil {
			return nil, errors.Wrap(err, "Clamp: min shape incompatible with x")
		}
	}
	if !nodes[2].shape.IsScalar() && !nodes[2].shape.EqualDimensions(nodes[1].shape) {
		if _, err := shapeinference.BinaryOp(compute.OpTypeMin, nodes[1].shape, nodes[2].shape); err != nil {
			return nil, errors.Wrap(err, "Clamp: max shape incompatible with x")
		}
	}
	id, err := f.builder.tape.AddClamp(nodes[0].id, nodes[1].id, nodes[2].id)
	if err != nil {
		return nil, errors.Wrap(err, "Clamp")
	}
	return &graphNode{id: id, shape: nodes[1].shape.Clone(), owner: f}, nil
}

// ===========================================================================
// Reduction Operations
// ===========================================================================

func (f *Function) reduceOp(opName string, mlxOp mlxbridge.ReduceOp, x compute.Value, axes ...int) (compute.Value, error) {
	node, err := f.resolveNode(x)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}
	dt := node.shape.DType
	switch mlxOp {
	case mlxbridge.ReduceLogicalAnd, mlxbridge.ReduceLogicalOr, mlxbridge.ReduceLogicalXor:
		if dt != dtypes.Bool {
			return nil, errors.Errorf("%s: requires Bool operand, got %s", opName, node.shape)
		}
	case mlxbridge.ReduceBitwiseAnd, mlxbridge.ReduceBitwiseOr, mlxbridge.ReduceBitwiseXor:
		if !dt.IsInt() {
			return nil, errors.Errorf("%s: requires integer operand, got %s", opName, node.shape)
		}
	case mlxbridge.ReduceMax, mlxbridge.ReduceMin:
		if !dt.IsFloat() && !dt.IsInt() && dt != dtypes.Bool {
			return nil, errors.Errorf("%s: requires float, integer, or bool operand, got %s", opName, node.shape)
		}
	case mlxbridge.ReduceSum, mlxbridge.ReduceProduct:
		if !dt.IsFloat() && !dt.IsInt() && !dt.IsComplex() {
			return nil, errors.Errorf("%s: requires numeric operand, got %s", opName, node.shape)
		}
	}

	rank := node.shape.Rank()
	if len(axes) == 0 {
		if rank == 0 {
			return f.Identity(node)
		}
		axes = make([]int, rank)
		for i := range axes {
			axes[i] = i
		}
	} else {
		norm := make([]int, len(axes))
		seen := sets.Make[int](len(axes))
		for i, a := range axes {
			if a < 0 {
				a += rank
			}
			if a < 0 || a >= rank {
				return nil, errors.Errorf("%s: axis %d out of bounds for shape %s", opName, axes[i], node.shape)
			}
			if seen.Has(a) {
				return nil, errors.Errorf("%s: duplicate reduction axis %d", opName, a)
			}
			seen.Insert(a)
			norm[i] = a
		}
		axes = norm
	}

	outShape, err := shapeinference.Reduce(node.shape, axes)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}
	id, err := f.builder.tape.AddReduce(mlxOp, node.id, axes)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}
	return &graphNode{id: id, shape: outShape, owner: f}, nil
}

func (f *Function) ReduceSum(x compute.Value, axes ...int) (compute.Value, error) {
	return f.reduceOp("ReduceSum", mlxbridge.ReduceSum, x, axes...)
}

func (f *Function) ReduceProduct(x compute.Value, axes ...int) (compute.Value, error) {
	return f.reduceOp("ReduceProduct", mlxbridge.ReduceProduct, x, axes...)
}

func (f *Function) ReduceMax(x compute.Value, axes ...int) (compute.Value, error) {
	return f.reduceOp("ReduceMax", mlxbridge.ReduceMax, x, axes...)
}

func (f *Function) ReduceMin(x compute.Value, axes ...int) (compute.Value, error) {
	return f.reduceOp("ReduceMin", mlxbridge.ReduceMin, x, axes...)
}

func (f *Function) ReduceLogicalAnd(x compute.Value, axes ...int) (compute.Value, error) {
	return f.reduceOp("ReduceLogicalAnd", mlxbridge.ReduceLogicalAnd, x, axes...)
}

func (f *Function) ReduceLogicalOr(x compute.Value, axes ...int) (compute.Value, error) {
	return f.reduceOp("ReduceLogicalOr", mlxbridge.ReduceLogicalOr, x, axes...)
}

func (f *Function) ReduceLogicalXor(x compute.Value, axes ...int) (compute.Value, error) {
	return f.reduceOp("ReduceLogicalXor", mlxbridge.ReduceLogicalXor, x, axes...)
}

func (f *Function) ReduceBitwiseAnd(x compute.Value, axes ...int) (compute.Value, error) {
	return f.reduceOp("ReduceBitwiseAnd", mlxbridge.ReduceBitwiseAnd, x, axes...)
}

func (f *Function) ReduceBitwiseOr(x compute.Value, axes ...int) (compute.Value, error) {
	return f.reduceOp("ReduceBitwiseOr", mlxbridge.ReduceBitwiseOr, x, axes...)
}

func (f *Function) ReduceBitwiseXor(x compute.Value, axes ...int) (compute.Value, error) {
	return f.reduceOp("ReduceBitwiseXor", mlxbridge.ReduceBitwiseXor, x, axes...)
}

// ===========================================================================
// ArgMinMax & Cumulative Operations
// ===========================================================================

// ArgMinMax calculates the argmin or argmax across an axis of x.
func (f *Function) ArgMinMax(x compute.Value, axis int, outputDType dtypes.DType, isMin bool) (compute.Value, error) {
	if err := f.builder.backend.checkDTypeSupported(outputDType); err != nil {
		return nil, errors.Wrap(err, "ArgMinMax")
	}
	node, err := f.resolveNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "ArgMinMax")
	}
	if axis < 0 {
		axis += node.shape.Rank()
	}
	outShape, err := shapeinference.ArgMinMax(node.shape, axis, outputDType)
	if err != nil {
		return nil, errors.Wrap(err, "ArgMinMax")
	}
	id, err := f.builder.tape.AddArgMinMax(node.id, axis, isMin, dtypeToBridgeDType(outputDType))
	if err != nil {
		return nil, errors.Wrap(err, "ArgMinMax")
	}
	return &graphNode{id: id, shape: outShape, owner: f}, nil
}

// CumSum returns the cumulative sum of elements along axis.
func (f *Function) CumSum(operand compute.Value, axis int, options compute.CumSumOptions) (compute.Value, error) {
	node, err := f.resolveNode(operand)
	if err != nil {
		return nil, errors.Wrap(err, "CumSum")
	}
	if axis < 0 {
		axis += node.shape.Rank()
	}
	outShape, err := shapeinference.CumSum(node.shape, axis)
	if err != nil {
		return nil, errors.Wrap(err, "CumSum")
	}
	id, err := f.builder.tape.AddCumulative(mlxbridge.CumulativeSum, node.id, axis, options.Reverse, options.Exclusive)
	if err != nil {
		return nil, errors.Wrap(err, "CumSum")
	}
	return &graphNode{id: id, shape: outShape, owner: f}, nil
}

// CumMax returns the cumulative maximum of elements along axis.
func (f *Function) CumMax(operand compute.Value, axis int, options compute.CumSumOptions) (compute.Value, error) {
	node, err := f.resolveNode(operand)
	if err != nil {
		return nil, errors.Wrap(err, "CumMax")
	}
	if axis < 0 {
		axis += node.shape.Rank()
	}
	outShape, err := shapeinference.CumMax(node.shape, axis)
	if err != nil {
		return nil, errors.Wrap(err, "CumMax")
	}
	id, err := f.builder.tape.AddCumulative(mlxbridge.CumulativeMax, node.id, axis, options.Reverse, options.Exclusive)
	if err != nil {
		return nil, errors.Wrap(err, "CumMax")
	}
	return &graphNode{id: id, shape: outShape, owner: f}, nil
}

// ===========================================================================
// Data-Dependent Dynamic Operations (NonZero)
// ===========================================================================

// NonZero returns row-major coordinates of nonzero elements as an int64 tensor [count, rank]
// (or [rank, count] if cfg.Transpose is true).
func (f *Function) NonZero(operand compute.Value, cfg compute.NonZeroConfig) (compute.Value, error) {
	if !f.builder.backend.dynamicShapes {
		return nil, errors.Wrapf(compute.ErrNotImplemented,
			"NonZero: dynamic shapes are not supported in %q backend (use %q to enable dynamic shapes)",
			BackendName, "mlx:dynamic")
	}
	opNode, err := f.resolveNode(operand)
	if err != nil {
		return nil, errors.Wrap(err, "NonZero")
	}
	if err := f.builder.backend.checkDTypeSupported(opNode.shape.DType); err != nil {
		return nil, errors.Wrap(err, "NonZero")
	}

	knownAxisNames := f.knownDynamicAxisNames()
	if cfg.CountAxisName == "" {
		for {
			cfg.CountAxisName = shapes.UniqueAxisName("#nonzeros")
			if !knownAxisNames.Has(cfg.CountAxisName) {
				break
			}
		}
	} else if knownAxisNames.Has(cfg.CountAxisName) {
		return nil, errors.Errorf("NonZero: count axis name %q is already in use", cfg.CountAxisName)
	}

	outShape, err := shapeinference.NonZero(opNode.shape, cfg)
	if err != nil {
		return nil, errors.Wrap(err, "NonZero")
	}

	id, err := f.builder.tape.AddNonZero(opNode.id, cfg.Transpose)
	if err != nil {
		return nil, errors.Wrap(err, "NonZero")
	}
	outNode := &graphNode{id: id, shape: outShape, owner: f}
	f.registerDynamicAxes(outNode)
	return outNode, nil
}
