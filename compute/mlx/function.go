// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mlx

import (
	"runtime"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/notimplemented"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/go-darwinml/internal/mlxbridge"
	"github.com/pkg/errors"
)

// graphNode represents a value recorded in the MLX computation tape.
type graphNode struct {
	id    mlxbridge.NodeID
	shape shapes.Shape
	name  string
	owner *Function
}

// Function implements compute.Function for the MLX backend.
type Function struct {
	notimplemented.Function

	builder  *Builder
	name     string
	parent   *Function
	returned bool
	params   []*graphNode
	outputs  []*graphNode
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
			"Parameter(%q): dynamic shape %s requires %q backend configuration", name, shape, "mlx:dynamic")
	}
	id, err := f.builder.tape.AddParameter(dtypeToBridgeDType(shape.DType), shape.Dimensions)
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
	return &graphNode{id: id, shape: node.shape.Clone(), owner: f}, nil
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
	return &graphNode{id: id, shape: outShape.Clone(), owner: f}, nil
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
	return &graphNode{id: id, shape: outShape, owner: f}, nil
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
