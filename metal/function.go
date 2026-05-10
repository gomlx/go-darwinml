//go:build darwin && cgo

package metal

import (
	"slices"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/notimplemented"
	"github.com/gomlx/gomlx/backends/shapeinference"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
	"github.com/x448/float16"
)

// Function implements backends.Function for the Metal backend.
type Function struct {
	notimplemented.Function

	builder    *Builder
	name       string
	parent     *Function
	returned   bool
	nodes      []*Node
	outputs    []*Node
	parameters []*Node

	capturedParentNodes []*Node
	capturedLocalNodes  []*Node
	compiled            *functionExecutable
}

var _ backends.Function = (*Function)(nil)

func newFunction(b *Builder, name string, parent *Function) *Function {
	return &Function{
		Function: notimplemented.Function{
			ErrFn: notImplementedError,
		},
		builder: b,
		name:    name,
		parent:  parent,
	}
}

func (f *Function) Name() string { return f.name }

func (f *Function) Parent() backends.Function {
	if f.parent == nil {
		return nil
	}
	return f.parent
}

func (f *Function) Closure() (backends.Function, error) {
	if err := f.CheckValid(); err != nil {
		return nil, err
	}
	return newFunction(f.builder, "", f), nil
}

func (f *Function) CheckValid() error {
	if f == nil || f.builder == nil {
		return errors.New("metal: function or builder is nil")
	}
	if f.builder.compiled {
		return errors.Errorf("cannot modify function %q: builder already compiled", f.name)
	}
	return nil
}

func (f *Function) IsAncestorOf(leaf *Function) bool {
	for g := leaf; g != nil; g = g.parent {
		if g == f {
			return true
		}
	}
	return false
}

func (f *Function) newMultiOutputsNode(opType backends.OpType, outputShapes []shapes.Shape, data any, inputs ...*Node) *Node {
	node := f.addNode(opType, shapes.Invalid(), inputs, data)
	node.multiOutputsShapes = outputShapes
	node.multiOutputsNodes = make([]*Node, len(outputShapes))
	for i, sh := range outputShapes {
		child := &Node{
			idx:                len(f.nodes),
			opType:             opType,
			shape:              sh,
			inputs:             []*Node{node},
			function:           f,
			builder:            f.builder,
			isNodeSelectOutput: true,
			selectOutputIdx:    i,
		}
		node.multiOutputsNodes[i] = child
		f.nodes = append(f.nodes, child)
	}
	return node
}

func (f *Function) verifyAndCastValues(name string, values ...backends.Value) ([]*Node, error) {
	if err := f.CheckValid(); err != nil {
		return nil, err
	}
	nodes, err := f.builder.checkValues(name, values...)
	if err != nil {
		return nil, err
	}
	for idx, node := range nodes {
		if node.function == nil {
			return nil, errors.Errorf("%s: input #%d has nil function", name, idx)
		}
		if node.function == f {
			continue
		}
		fromAncestor := false
		for a := f.parent; a != nil; a = a.parent {
			if node.function == a {
				fromAncestor = true
				break
			}
		}
		if fromAncestor {
			nodes[idx] = f.getOrCreateCaptureNode(node)
		} else {
			return nil, errors.Errorf("%s: input #%d is not from this function or an ancestor", name, idx)
		}
	}
	return nodes, nil
}

// addNode appends a node to this function and returns it.
func (f *Function) addNode(opType backends.OpType, shape shapes.Shape, inputs []*Node, data any) *Node {
	node := &Node{
		idx:      len(f.nodes),
		opType:   opType,
		shape:    shape,
		inputs:   inputs,
		function: f,
		builder:  f.builder,
		data:     data,
	}
	f.nodes = append(f.nodes, node)
	return node
}

// checkValues validates inputs, rejects post-compile mutations, and canonicalizes
// ancestor references into captured nodes for closures.
func (f *Function) checkValues(opName string, values ...backends.Value) ([]*Node, error) {
	return f.verifyAndCastValues(opName, values...)
}

// ─── Parameter & Constant ───────────────────────────────────────────────────

type nodeParameter struct {
	name string
}

func (f *Function) Parameter(name string, shape shapes.Shape, spec *backends.ShardingSpec) (backends.Value, error) {
	if err := f.CheckValid(); err != nil {
		return nil, err
	}
	if spec != nil {
		return nil, errors.Wrap(notimplemented.NotImplementedError, "metal: sharding not supported")
	}
	node := f.addNode(backends.OpTypeParameter, shape, nil, &nodeParameter{name: name})
	f.parameters = append(f.parameters, node)
	return node, nil
}

func (f *Function) Constant(flat any, dims ...int) (backends.Value, error) {
	if err := f.CheckValid(); err != nil {
		return nil, err
	}
	dtype, nelem, err := checkFlat(flat)
	if err != nil {
		return nil, err
	}
	shape := shapes.Make(dtype, dims...)
	if shape.Size() != nelem {
		return nil, errors.Errorf("Constant: flat has %d elements, shape %s has %d",
			nelem, shape, shape.Size())
	}
	node := f.addNode(backends.OpTypeConstant, shape, nil, flat)
	return node, nil
}

func (f *Function) Return(outputs []backends.Value, shardings []*backends.ShardingSpec) error {
	if len(shardings) != 0 {
		return errors.New("metal Return: sharding not supported")
	}
	nodes, err := f.verifyAndCastValues("Return", outputs...)
	if err != nil {
		return err
	}
	for _, node := range nodes {
		if len(node.multiOutputsShapes) > 0 {
			return errors.Errorf("Return: cannot use internal multi-output node %s as output", node.opType)
		}
	}
	f.outputs = nodes
	f.returned = true
	if f.parent != nil || f.name != backends.MainName {
		compiled, err := newFunctionExecutable(f)
		if err != nil {
			return errors.WithMessagef(err, "compile function %q", f.name)
		}
		f.compiled = compiled
	}
	return nil
}

func (f *Function) Identity(x backends.Value) (backends.Value, error) {
	inputs, err := f.checkValues("Identity", x)
	if err != nil {
		return nil, err
	}
	return f.addNode(backends.OpTypeIdentity, inputs[0].shape, inputs, nil), nil
}

// ─── Unary ops ──────────────────────────────────────────────────────────────

func (f *Function) unaryOp(opType backends.OpType, name string, x backends.Value) (backends.Value, error) {
	inputs, err := f.checkValues(name, x)
	if err != nil {
		return nil, err
	}
	outShape, err := shapeinference.UnaryOp(opType, inputs[0].shape)
	if err != nil {
		return nil, errors.WithMessagef(err, "metal: %s", name)
	}
	return f.addNode(opType, outShape, inputs, nil), nil
}

func (f *Function) Abs(x backends.Value) (backends.Value, error) {
	return f.unaryOp(backends.OpTypeAbs, "Abs", x)
}
func (f *Function) Neg(x backends.Value) (backends.Value, error) {
	return f.unaryOp(backends.OpTypeNeg, "Neg", x)
}
func (f *Function) Ceil(x backends.Value) (backends.Value, error) {
	return f.unaryOp(backends.OpTypeCeil, "Ceil", x)
}
func (f *Function) Floor(x backends.Value) (backends.Value, error) {
	return f.unaryOp(backends.OpTypeFloor, "Floor", x)
}
func (f *Function) Round(x backends.Value) (backends.Value, error) {
	return f.unaryOp(backends.OpTypeRound, "Round", x)
}
func (f *Function) Sign(x backends.Value) (backends.Value, error) {
	return f.unaryOp(backends.OpTypeSign, "Sign", x)
}
func (f *Function) Sqrt(x backends.Value) (backends.Value, error) {
	return f.unaryOp(backends.OpTypeSqrt, "Sqrt", x)
}
func (f *Function) Rsqrt(x backends.Value) (backends.Value, error) {
	return f.unaryOp(backends.OpTypeRsqrt, "Rsqrt", x)
}
func (f *Function) Exp(x backends.Value) (backends.Value, error) {
	return f.unaryOp(backends.OpTypeExp, "Exp", x)
}
func (f *Function) Expm1(x backends.Value) (backends.Value, error) {
	return f.unaryOp(backends.OpTypeExpm1, "Expm1", x)
}
func (f *Function) Log(x backends.Value) (backends.Value, error) {
	return f.unaryOp(backends.OpTypeLog, "Log", x)
}
func (f *Function) Log1p(x backends.Value) (backends.Value, error) {
	return f.unaryOp(backends.OpTypeLog1p, "Log1p", x)
}
func (f *Function) Sin(x backends.Value) (backends.Value, error) {
	return f.unaryOp(backends.OpTypeSin, "Sin", x)
}
func (f *Function) Cos(x backends.Value) (backends.Value, error) {
	return f.unaryOp(backends.OpTypeCos, "Cos", x)
}
func (f *Function) Tanh(x backends.Value) (backends.Value, error) {
	return f.unaryOp(backends.OpTypeTanh, "Tanh", x)
}
func (f *Function) Erf(x backends.Value) (backends.Value, error) {
	return f.unaryOp(backends.OpTypeErf, "Erf", x)
}
func (f *Function) Logistic(x backends.Value) (backends.Value, error) {
	return f.unaryOp(backends.OpTypeLogistic, "Logistic", x)
}
func (f *Function) IsFinite(x backends.Value) (backends.Value, error) {
	inputs, err := f.checkValues("IsFinite", x)
	if err != nil {
		return nil, err
	}
	outShape := shapes.Make(dtypes.Bool, inputs[0].shape.Dimensions...)
	return f.addNode(backends.OpTypeIsFinite, outShape, inputs, nil), nil
}
func (f *Function) IsNaN(x backends.Value) (backends.Value, error) {
	inputs, err := f.checkValues("IsNaN", x)
	if err != nil {
		return nil, err
	}
	outShape := shapes.Make(dtypes.Bool, inputs[0].shape.Dimensions...)
	return f.addNode(backends.OpTypeIsNaN, outShape, inputs, nil), nil
}
func (f *Function) LogicalNot(x backends.Value) (backends.Value, error) {
	return f.unaryOp(backends.OpTypeLogicalNot, "LogicalNot", x)
}
func (f *Function) BitwiseNot(x backends.Value) (backends.Value, error) {
	return f.unaryOp(backends.OpTypeBitwiseNot, "BitwiseNot", x)
}
func (f *Function) Clz(x backends.Value) (backends.Value, error) {
	return f.unaryOp(backends.OpTypeClz, "Clz", x)
}
func (f *Function) BitCount(x backends.Value) (backends.Value, error) {
	return f.unaryOp(backends.OpTypeBitCount, "BitCount", x)
}

// ─── Binary ops ─────────────────────────────────────────────────────────────

func supportedElementwiseAutoBroadcastDType(dt dtypes.DType) bool {
	switch dt {
	case dtypes.Bool, dtypes.Float16, dtypes.BFloat16, dtypes.Float32,
		dtypes.Int8, dtypes.Int16, dtypes.Int32, dtypes.Int64,
		dtypes.Uint8, dtypes.Uint16, dtypes.Uint32, dtypes.Uint64:
		return true
	default:
		return false
	}
}

func supportedWhereAutoBroadcastDType(dt dtypes.DType) bool {
	return wherePredValueKind(dt) >= 0
}

func supportedIntegerShiftBroadcastDType(dt dtypes.DType) bool {
	switch dt {
	case dtypes.Int8, dtypes.Int16, dtypes.Int32, dtypes.Int64,
		dtypes.Uint8, dtypes.Uint16, dtypes.Uint32, dtypes.Uint64:
		return true
	default:
		return false
	}
}

func (f *Function) broadcastNodeToShapeWithSupport(opName string, input *Node, outShape shapes.Shape, supported func(dtypes.DType) bool) (*Node, error) {
	if input.shape.Equal(outShape) {
		return input, nil
	}
	if input.shape.DType != outShape.DType {
		return nil, errors.Errorf("metal: %s cannot broadcast %s to %s with mismatched dtypes",
			opName, input.shape, outShape)
	}
	if !supported(outShape.DType) {
		return nil, errors.Errorf("metal: %s auto-broadcast for dtype %s is not implemented",
			opName, outShape.DType)
	}
	if input.shape.IsScalar() {
		v, err := f.Broadcast(input, outShape.Dimensions...)
		if err != nil {
			return nil, errors.WithMessagef(err, "metal: %s scalar broadcast", opName)
		}
		return v.(*Node), nil
	}
	if input.shape.Rank() != outShape.Rank() {
		return nil, errors.Errorf("metal: %s cannot broadcast rank %d to rank %d",
			opName, input.shape.Rank(), outShape.Rank())
	}
	axes := make([]int, input.shape.Rank())
	for i := range axes {
		axes[i] = i
	}
	v, err := f.BroadcastInDim(input, outShape, axes)
	if err != nil {
		return nil, errors.WithMessagef(err, "metal: %s broadcast_in_dim", opName)
	}
	return v.(*Node), nil
}

func (f *Function) broadcastNodeToShape(opName string, input *Node, outShape shapes.Shape) (*Node, error) {
	return f.broadcastNodeToShapeWithSupport(opName, input, outShape, supportedElementwiseAutoBroadcastDType)
}

func (f *Function) binaryInputsWithBroadcast(opType backends.OpType, name string, lhs, rhs backends.Value) ([]*Node, shapes.Shape, error) {
	inputs, err := f.checkValues(name, lhs, rhs)
	if err != nil {
		return nil, shapes.Invalid(), err
	}
	outShape, err := shapeinference.BinaryOp(opType, inputs[0].shape, inputs[1].shape)
	if err != nil {
		return nil, shapes.Invalid(), errors.WithMessagef(err, "metal: %s", name)
	}
	lhsNode, err := f.broadcastNodeToShape(name, inputs[0], outShape)
	if err != nil {
		return nil, shapes.Invalid(), err
	}
	rhsNode, err := f.broadcastNodeToShape(name, inputs[1], outShape)
	if err != nil {
		return nil, shapes.Invalid(), err
	}
	return []*Node{lhsNode, rhsNode}, outShape, nil
}

func (f *Function) binaryOp(opType backends.OpType, name string, lhs, rhs backends.Value) (backends.Value, error) {
	inputs, outShape, err := f.binaryInputsWithBroadcast(opType, name, lhs, rhs)
	if err != nil {
		return nil, err
	}
	return f.addNode(opType, outShape, inputs, nil), nil
}

func (f *Function) binaryShiftInputsWithBroadcast(opType backends.OpType, name string, lhs, rhs backends.Value) ([]*Node, shapes.Shape, error) {
	inputs, err := f.checkValues(name, lhs, rhs)
	if err != nil {
		return nil, shapes.Invalid(), err
	}
	if !inputs[0].shape.DType.IsInt() {
		return nil, shapes.Invalid(), errors.Errorf("metal: %s requires integer operands, got %s", name, inputs[0].shape.DType)
	}
	outShape, err := shapeinference.BinaryOp(opType, inputs[0].shape, inputs[1].shape)
	if err != nil {
		return nil, shapes.Invalid(), errors.WithMessagef(err, "metal: %s", name)
	}
	lhsNode, err := f.broadcastNodeToShapeWithSupport(name, inputs[0], outShape, supportedIntegerShiftBroadcastDType)
	if err != nil {
		return nil, shapes.Invalid(), err
	}
	rhsNode, err := f.broadcastNodeToShapeWithSupport(name, inputs[1], outShape, supportedIntegerShiftBroadcastDType)
	if err != nil {
		return nil, shapes.Invalid(), err
	}
	return []*Node{lhsNode, rhsNode}, outShape, nil
}

func (f *Function) binaryShiftOp(opType backends.OpType, name string, lhs, rhs backends.Value) (backends.Value, error) {
	inputs, outShape, err := f.binaryShiftInputsWithBroadcast(opType, name, lhs, rhs)
	if err != nil {
		return nil, err
	}
	return f.addNode(opType, outShape, inputs, nil), nil
}

func (f *Function) Add(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp(backends.OpTypeAdd, "Add", lhs, rhs)
}
func (f *Function) Sub(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp(backends.OpTypeSub, "Sub", lhs, rhs)
}
func (f *Function) Mul(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp(backends.OpTypeMul, "Mul", lhs, rhs)
}
func (f *Function) Div(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp(backends.OpTypeDiv, "Div", lhs, rhs)
}
func (f *Function) Pow(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp(backends.OpTypePow, "Pow", lhs, rhs)
}
func (f *Function) Rem(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp(backends.OpTypeRem, "Rem", lhs, rhs)
}
func (f *Function) Max(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp(backends.OpTypeMax, "Max", lhs, rhs)
}
func (f *Function) Min(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp(backends.OpTypeMin, "Min", lhs, rhs)
}
func (f *Function) Atan2(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp(backends.OpTypeAtan2, "Atan2", lhs, rhs)
}

func (f *Function) BitwiseAnd(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp(backends.OpTypeBitwiseAnd, "BitwiseAnd", lhs, rhs)
}
func (f *Function) BitwiseOr(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp(backends.OpTypeBitwiseOr, "BitwiseOr", lhs, rhs)
}
func (f *Function) BitwiseXor(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp(backends.OpTypeBitwiseXor, "BitwiseXor", lhs, rhs)
}

func (f *Function) ShiftLeft(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryShiftOp(backends.OpTypeShiftLeft, "ShiftLeft", lhs, rhs)
}

func (f *Function) ShiftRightArithmetic(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryShiftOp(backends.OpTypeShiftRightArithmetic, "ShiftRightArithmetic", lhs, rhs)
}

func (f *Function) ShiftRightLogical(lhs, rhs backends.Value) (backends.Value, error) {
	lhsNodes, err := f.checkValues("ShiftRightLogical", lhs)
	if err != nil {
		return nil, err
	}
	dt := lhsNodes[0].shape.DType

	unsignedDT, isSigned := signedToUnsigned(dt)
	if !isSigned {
		return f.binaryShiftOp(backends.OpTypeShiftRightLogical, "ShiftRightLogical", lhs, rhs)
	}

	castLHS, err := f.ConvertDType(lhs, unsignedDT)
	if err != nil {
		return nil, errors.Wrap(err, "ShiftRightLogical: cast lhs to unsigned")
	}
	castRHS, err := f.ConvertDType(rhs, unsignedDT)
	if err != nil {
		return nil, errors.Wrap(err, "ShiftRightLogical: cast rhs to unsigned")
	}
	shifted, err := f.binaryShiftOp(backends.OpTypeShiftRightLogical, "ShiftRightLogical", castLHS, castRHS)
	if err != nil {
		return nil, errors.Wrap(err, "ShiftRightLogical: shift")
	}
	return f.ConvertDType(shifted, dt)
}

// signedToUnsigned returns the unsigned counterpart of a signed integer dtype.
func signedToUnsigned(dt dtypes.DType) (dtypes.DType, bool) {
	switch dt {
	case dtypes.Int8:
		return dtypes.Uint8, true
	case dtypes.Int16:
		return dtypes.Uint16, true
	case dtypes.Int32:
		return dtypes.Uint32, true
	case dtypes.Int64:
		return dtypes.Uint64, true
	default:
		return dt, false
	}
}

func (f *Function) LogicalAnd(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp(backends.OpTypeLogicalAnd, "LogicalAnd", lhs, rhs)
}
func (f *Function) LogicalOr(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp(backends.OpTypeLogicalOr, "LogicalOr", lhs, rhs)
}
func (f *Function) LogicalXor(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp(backends.OpTypeLogicalXor, "LogicalXor", lhs, rhs)
}

// Comparison ops — output same shape as broadcast result with Bool dtype
func (f *Function) cmpOp(opType backends.OpType, name string, lhs, rhs backends.Value) (backends.Value, error) {
	inputs, err := f.checkValues(name, lhs, rhs)
	if err != nil {
		return nil, err
	}
	boolShape, err := shapeinference.ComparisonOp(opType, inputs[0].shape, inputs[1].shape)
	if err != nil {
		return nil, errors.WithMessagef(err, "metal: %s", name)
	}
	operandShape := shapes.Make(inputs[0].shape.DType, boolShape.Dimensions...)
	lhsNode, err := f.broadcastNodeToShape(name, inputs[0], operandShape)
	if err != nil {
		return nil, err
	}
	rhsNode, err := f.broadcastNodeToShape(name, inputs[1], operandShape)
	if err != nil {
		return nil, err
	}
	return f.addNode(opType, boolShape, []*Node{lhsNode, rhsNode}, nil), nil
}

func (f *Function) Equal(lhs, rhs backends.Value) (backends.Value, error) {
	return f.cmpOp(backends.OpTypeEqual, "Equal", lhs, rhs)
}
func (f *Function) NotEqual(lhs, rhs backends.Value) (backends.Value, error) {
	return f.cmpOp(backends.OpTypeNotEqual, "NotEqual", lhs, rhs)
}
func (f *Function) LessThan(lhs, rhs backends.Value) (backends.Value, error) {
	return f.cmpOp(backends.OpTypeLessThan, "LessThan", lhs, rhs)
}
func (f *Function) LessOrEqual(lhs, rhs backends.Value) (backends.Value, error) {
	return f.cmpOp(backends.OpTypeLessOrEqual, "LessOrEqual", lhs, rhs)
}
func (f *Function) GreaterThan(lhs, rhs backends.Value) (backends.Value, error) {
	return f.cmpOp(backends.OpTypeGreaterThan, "GreaterThan", lhs, rhs)
}
func (f *Function) GreaterOrEqual(lhs, rhs backends.Value) (backends.Value, error) {
	return f.cmpOp(backends.OpTypeGreaterOrEqual, "GreaterOrEqual", lhs, rhs)
}

// ─── Reduce ops ─────────────────────────────────────────────────────────────

type reduceData struct {
	axes []int
}

func (f *Function) reduceOp(opType backends.OpType, name string, x backends.Value, axes []int) (backends.Value, error) {
	inputs, err := f.checkValues(name, x)
	if err != nil {
		return nil, err
	}
	inShape := inputs[0].shape
	// Build output dimensions by removing reduced axes
	axisSet := make(map[int]bool, len(axes))
	for _, a := range axes {
		axisSet[a] = true
	}
	var outDims []int
	for i, d := range inShape.Dimensions {
		if !axisSet[i] {
			outDims = append(outDims, d)
		}
	}
	if len(outDims) == 0 {
		outDims = []int{} // scalar
	}
	outShape := shapes.Make(inShape.DType, outDims...)
	return f.addNode(opType, outShape, inputs, &reduceData{axes: axes}), nil
}

func (f *Function) ReduceSum(x backends.Value, axes ...int) (backends.Value, error) {
	return f.reduceOp(backends.OpTypeReduceSum, "ReduceSum", x, axes)
}
func (f *Function) ReduceProduct(x backends.Value, axes ...int) (backends.Value, error) {
	return f.reduceOp(backends.OpTypeReduceProduct, "ReduceProduct", x, axes)
}
func (f *Function) ReduceMax(x backends.Value, axes ...int) (backends.Value, error) {
	return f.reduceOp(backends.OpTypeReduceMax, "ReduceMax", x, axes)
}
func (f *Function) ReduceMin(x backends.Value, axes ...int) (backends.Value, error) {
	return f.reduceOp(backends.OpTypeReduceMin, "ReduceMin", x, axes)
}

func (f *Function) ReduceBitwiseAnd(x backends.Value, axes ...int) (backends.Value, error) {
	return f.reduceOp(backends.OpTypeReduceBitwiseAnd, "ReduceBitwiseAnd", x, axes)
}

func (f *Function) ReduceBitwiseOr(x backends.Value, axes ...int) (backends.Value, error) {
	return f.reduceOp(backends.OpTypeReduceBitwiseOr, "ReduceBitwiseOr", x, axes)
}

func (f *Function) ReduceBitwiseXor(x backends.Value, axes ...int) (backends.Value, error) {
	return f.reduceOp(backends.OpTypeReduceBitwiseXor, "ReduceBitwiseXor", x, axes)
}

func (f *Function) ReduceLogicalAnd(x backends.Value, axes ...int) (backends.Value, error) {
	return f.reduceOp(backends.OpTypeReduceLogicalAnd, "ReduceLogicalAnd", x, axes)
}

func (f *Function) ReduceLogicalOr(x backends.Value, axes ...int) (backends.Value, error) {
	return f.reduceOp(backends.OpTypeReduceLogicalOr, "ReduceLogicalOr", x, axes)
}

func (f *Function) ReduceLogicalXor(x backends.Value, axes ...int) (backends.Value, error) {
	return f.reduceOp(backends.OpTypeReduceLogicalXor, "ReduceLogicalXor", x, axes)
}

// ─── Reshape / Transpose / Broadcast ────────────────────────────────────────

func (f *Function) Reshape(x backends.Value, dims ...int) (backends.Value, error) {
	inputs, err := f.checkValues("Reshape", x)
	if err != nil {
		return nil, err
	}
	outShape := shapes.Make(inputs[0].shape.DType, dims...)
	return f.addNode(backends.OpTypeReshape, outShape, inputs, nil), nil
}

type transposeData struct {
	permutation []int
}

func (f *Function) Transpose(x backends.Value, permutation ...int) (backends.Value, error) {
	inputs, err := f.checkValues("Transpose", x)
	if err != nil {
		return nil, err
	}
	inShape := inputs[0].shape
	outDims := make([]int, len(permutation))
	for i, p := range permutation {
		outDims[i] = inShape.Dimensions[p]
	}
	outShape := shapes.Make(inShape.DType, outDims...)
	return f.addNode(backends.OpTypeTranspose, outShape, inputs, &transposeData{permutation: permutation}), nil
}

func (f *Function) Broadcast(x backends.Value, dims ...int) (backends.Value, error) {
	inputs, err := f.checkValues("Broadcast", x)
	if err != nil {
		return nil, err
	}
	outShape, err := shapeinference.BroadcastOp(inputs[0].shape, dims)
	if err != nil {
		return nil, err
	}
	prefixDims := slices.Clone(dims)
	return f.addNode(backends.OpTypeBroadcast, outShape, inputs, prefixDims), nil
}

func (f *Function) BroadcastInDim(x backends.Value, outputShape shapes.Shape, broadcastAxes []int) (backends.Value, error) {
	inputs, err := f.checkValues("BroadcastInDim", x)
	if err != nil {
		return nil, err
	}
	if err := shapeinference.BroadcastInDimOp(inputs[0].shape, outputShape, broadcastAxes); err != nil {
		return nil, err
	}
	axes := slices.Clone(broadcastAxes)
	return f.addNode(backends.OpTypeBroadcastInDim, outputShape, inputs, axes), nil
}

// ─── Where ──────────────────────────────────────────────────────────────────

func (f *Function) Where(pred, onTrue, onFalse backends.Value) (backends.Value, error) {
	inputs, err := f.checkValues("Where", pred, onTrue, onFalse)
	if err != nil {
		return nil, err
	}
	outShape, err := shapeinference.WhereOp(inputs[0].shape, inputs[1].shape, inputs[2].shape)
	if err != nil {
		return nil, errors.WithMessage(err, "metal: Where")
	}
	predShape := shapes.Make(dtypes.Bool, outShape.Dimensions...)
	predNode, err := f.broadcastNodeToShapeWithSupport("Where", inputs[0], predShape, supportedWhereAutoBroadcastDType)
	if err != nil {
		return nil, err
	}
	trueNode, err := f.broadcastNodeToShapeWithSupport("Where", inputs[1], outShape, supportedWhereAutoBroadcastDType)
	if err != nil {
		return nil, err
	}
	falseNode, err := f.broadcastNodeToShapeWithSupport("Where", inputs[2], outShape, supportedWhereAutoBroadcastDType)
	if err != nil {
		return nil, err
	}
	return f.addNode(backends.OpTypeWhere, outShape, []*Node{predNode, trueNode, falseNode}, nil), nil
}

// ─── Concatenate ────────────────────────────────────────────────────────────

type concatData struct {
	axis int
}

func (f *Function) Concatenate(axis int, operands ...backends.Value) (backends.Value, error) {
	if len(operands) == 0 {
		return nil, errors.New("Concatenate: no operands")
	}
	nodes, err := f.verifyAndCastValues("Concatenate", operands...)
	if err != nil {
		return nil, err
	}
	inputShapes := make([]shapes.Shape, len(nodes))
	for i, n := range nodes {
		inputShapes[i] = n.shape
	}
	outShape, err := shapeinference.ConcatenateOp(inputShapes, axis)
	if err != nil {
		return nil, err
	}
	return f.addNode(backends.OpTypeConcatenate, outShape, nodes, &concatData{axis: axis}), nil
}

// ─── Slice ──────────────────────────────────────────────────────────────────

type sliceData struct {
	starts, limits, strides []int
}

func (f *Function) Slice(x backends.Value, starts, limits, strides []int) (backends.Value, error) {
	inputs, err := f.checkValues("Slice", x)
	if err != nil {
		return nil, err
	}
	outDims := make([]int, len(starts))
	for i := range starts {
		s := 1
		if strides != nil && i < len(strides) {
			s = strides[i]
		}
		outDims[i] = (limits[i] - starts[i] + s - 1) / s
	}
	outShape := shapes.Make(inputs[0].shape.DType, outDims...)
	return f.addNode(backends.OpTypeSlice, outShape, inputs, &sliceData{starts, limits, strides}), nil
}

const maxDynamicUpdateSliceRank = 16

type dynamicUpdateSliceData struct {
	packedStartIndices bool
}

func (f *Function) DynamicUpdateSlice(operand, update backends.Value, startIndices []backends.Value) (backends.Value, error) {
	if err := f.CheckValid(); err != nil {
		return nil, err
	}
	nodes, err := f.verifyAndCastValues("DynamicUpdateSlice", append([]backends.Value{operand, update}, startIndices...)...)
	if err != nil {
		return nil, err
	}
	opN, upN := nodes[0], nodes[1]
	oShape, uShape := opN.shape, upN.shape
	if oShape.DType != uShape.DType {
		return nil, errors.Errorf("DynamicUpdateSlice: operand dtype %s != update dtype %s", oShape.DType, uShape.DType)
	}
	if oShape.Rank() != uShape.Rank() {
		return nil, errors.Errorf("DynamicUpdateSlice: operand rank %d != update rank %d", oShape.Rank(), uShape.Rank())
	}
	for i := range oShape.Dimensions {
		if uShape.Dimensions[i] > oShape.Dimensions[i] {
			return nil, errors.Errorf(
				"DynamicUpdateSlice: update dim %d size %d exceeds operand %d",
				i, uShape.Dimensions[i], oShape.Dimensions[i])
		}
	}
	rank := oShape.Rank()
	if rank > maxDynamicUpdateSliceRank {
		return nil, errors.Errorf(
			"DynamicUpdateSlice: rank %d exceeds metal backend limit %d", rank, maxDynamicUpdateSliceRank)
	}
	startNodes := nodes[2:]
	var packed bool
	if rank == 0 {
		if len(startNodes) != 0 {
			return nil, errors.New("DynamicUpdateSlice: rank-0 operand expects no start indices")
		}
		return f.addNode(backends.OpTypeDynamicUpdateSlice, oShape, []*Node{opN, upN},
			&dynamicUpdateSliceData{}), nil
	}
	if len(startNodes) == 1 && startNodes[0].shape.Rank() == 1 && !startNodes[0].shape.IsScalar() {
		if startNodes[0].shape.Dimensions[0] != rank {
			return nil, errors.Errorf(
				"DynamicUpdateSlice: packed start length %d != operand rank %d",
				startNodes[0].shape.Dimensions[0], rank)
		}
		packed = true
	} else {
		if len(startNodes) != rank {
			return nil, errors.Errorf(
				"DynamicUpdateSlice: want %d scalar start indices, got %d", rank, len(startNodes))
		}
		packed = false
	}
	for _, sn := range startNodes {
		dt := sn.shape.DType
		if !dt.IsInt() && !dt.IsUnsigned() {
			return nil, errors.Errorf(
				"DynamicUpdateSlice: start index dtype %s not supported", dt)
		}
	}
	inputs := append([]*Node{opN, upN}, startNodes...)
	return f.addNode(backends.OpTypeDynamicUpdateSlice, oShape, inputs,
		&dynamicUpdateSliceData{packedStartIndices: packed}), nil
}

const maxDynamicSliceRank = 16

type dynamicSliceData struct {
	sliceSizes         []int
	packedStartIndices bool
}

func (f *Function) DynamicSlice(operand backends.Value, startIndices []backends.Value, sliceSizes []int) (backends.Value, error) {
	nodes, err := f.verifyAndCastValues("DynamicSlice", append([]backends.Value{operand}, startIndices...)...)
	if err != nil {
		return nil, err
	}
	opN := nodes[0]
	oShape := opN.shape
	rank := oShape.Rank()
	if rank != len(sliceSizes) {
		return nil, errors.Errorf("DynamicSlice: sliceSizes length %d must match operand rank %d",
			len(sliceSizes), rank)
	}
	for i, sz := range sliceSizes {
		if sz < 0 {
			return nil, errors.Errorf("DynamicSlice: sliceSizes[%d] = %d invalid", i, sz)
		}
		if sz > oShape.Dimensions[i] {
			return nil, errors.Errorf(
				"DynamicSlice: sliceSizes[%d]=%d exceeds operand dim %d",
				i, sz, oShape.Dimensions[i])
		}
	}
	if rank > maxDynamicSliceRank {
		return nil, errors.Errorf(
			"DynamicSlice: rank %d exceeds metal backend limit %d", rank, maxDynamicSliceRank)
	}

	startNodes := nodes[1:]
	var packed bool
	if rank == 0 {
		return f.addNode(backends.OpTypeDynamicSlice, oShape,
			[]*Node{opN}, &dynamicSliceData{sliceSizes: append([]int(nil), sliceSizes...)}), nil
	}
	if len(startNodes) == 1 && startNodes[0].shape.Rank() == 1 && !startNodes[0].shape.IsScalar() {
		if startNodes[0].shape.Dimensions[0] != rank {
			return nil, errors.Errorf(
				"DynamicSlice: packed start length %d != operand rank %d",
				startNodes[0].shape.Dimensions[0], rank)
		}
		packed = true
	} else {
		if len(startNodes) != rank {
			return nil, errors.Errorf(
				"DynamicSlice: want %d scalar start indices, got %d", rank, len(startNodes))
		}
		packed = false
	}
	for _, sn := range startNodes {
		dt := sn.shape.DType
		if !dt.IsInt() && !dt.IsUnsigned() {
			return nil, errors.Errorf(
				"DynamicSlice: start index dtype %s not supported", dt)
		}
	}

	outShape := shapes.Make(oShape.DType, sliceSizes...)
	inputs := append([]*Node{opN}, startNodes...)
	return f.addNode(backends.OpTypeDynamicSlice, outShape, inputs,
		&dynamicSliceData{
			sliceSizes:         append([]int(nil), sliceSizes...),
			packedStartIndices: packed,
		}), nil
}

// ─── Iota ───────────────────────────────────────────────────────────────────

type iotaData struct {
	iotaDimension int
}

func (f *Function) Iota(shape shapes.Shape, iotaDimension int) (backends.Value, error) {
	if err := f.CheckValid(); err != nil {
		return nil, err
	}
	if shape.Rank() == 0 {
		return nil, errors.Errorf("Iota: shape must have at least one dimension")
	}
	if iotaDimension < 0 || iotaDimension >= shape.Rank() {
		return nil, errors.Errorf("Iota: iotaDimension (%d) must be in the range [0,%d)", iotaDimension, shape.Rank())
	}
	if dtypeToMetalIota(shape.DType) < 0 && shape.DType != dtypes.BFloat16 {
		return nil, errors.Errorf("Iota: metal supports float16/float32/bfloat16/int32/int64/uint32/uint64, got %s", shape.DType)
	}
	return f.addNode(backends.OpTypeIota, shape, nil, &iotaData{iotaDimension}), nil
}

// ─── ConvertDType ───────────────────────────────────────────────────────────

func (f *Function) ConvertDType(x backends.Value, dtype dtypes.DType) (backends.Value, error) {
	inputs, err := f.checkValues("ConvertDType", x)
	if err != nil {
		return nil, err
	}
	outShape := shapes.Make(dtype, inputs[0].shape.Dimensions...)
	return f.addNode(backends.OpTypeConvertDType, outShape, inputs, nil), nil
}

// ─── Dot / DotGeneral ───────────────────────────────────────────────────────

type dotGeneralData struct {
	lhsContractingAxes []int
	rhsContractingAxes []int
	lhsBatchAxes       []int
	rhsBatchAxes       []int
	config             backends.DotGeneralConfig
}

func (f *Function) Dot(lhs, rhs backends.Value) (backends.Value, error) {
	inputs, err := f.checkValues("Dot", lhs, rhs)
	if err != nil {
		return nil, err
	}
	lShape := inputs[0].shape
	rShape := inputs[1].shape
	// Standard matmul: [m, k] @ [k, n] -> [m, n]
	m := lShape.Dimensions[0]
	n := rShape.Dimensions[1]
	outShape := shapes.Make(lShape.DType, m, n)
	return f.addNode(backends.OpTypeDot, outShape, inputs, nil), nil
}

func (f *Function) DotGeneral(
	lhsOp backends.Value,
	lhsContractingAxes, lhsBatchAxes []int,
	rhsOp backends.Value,
	rhsContractingAxes, rhsBatchAxes []int,
	config backends.DotGeneralConfig,
) (backends.Value, error) {
	inputs, err := f.checkValues("DotGeneral", lhsOp, rhsOp)
	if err != nil {
		return nil, err
	}

	dtype := inputs[0].shape.DType
	if dtype != inputs[1].shape.DType {
		return nil, errors.Errorf(
			"DotGeneral lhs and rhs dtype mismatch: %s vs %s", dtype, inputs[1].shape.DType)
	}

	isHalfPrecisionWithFloat32 := dtype.IsHalfPrecision() && config.AccumulatorDType == dtypes.Float32
	if !isHalfPrecisionWithFloat32 && config.AccumulatorDType != dtypes.InvalidDType && config.AccumulatorDType != dtype {
		lhsOp, err = f.ConvertDType(lhsOp, config.AccumulatorDType)
		if err != nil {
			return nil, err
		}
		rhsOp, err = f.ConvertDType(rhsOp, config.AccumulatorDType)
		if err != nil {
			return nil, err
		}
		inputs, err = f.checkValues("DotGeneral", lhsOp, rhsOp)
		if err != nil {
			return nil, err
		}
		dtype = config.AccumulatorDType
	}

	if len(lhsContractingAxes) != len(rhsContractingAxes) {
		return nil, errors.Errorf(
			"DotGeneral contracting axes: lhs has %d, rhs has %d",
			len(lhsContractingAxes), len(rhsContractingAxes))
	}
	if len(lhsBatchAxes) != len(rhsBatchAxes) {
		return nil, errors.Errorf(
			"DotGeneral batch axes: lhs has %d, rhs has %d",
			len(lhsBatchAxes), len(rhsBatchAxes))
	}

	lShape := inputs[0].shape
	rShape := inputs[1].shape

	// Compute output shape: batch dims + lhs free dims + rhs free dims.
	contractSet := make(map[int]bool)
	batchSetL := make(map[int]bool)
	for _, a := range lhsContractingAxes {
		contractSet[a] = true
	}
	for _, a := range lhsBatchAxes {
		batchSetL[a] = true
	}
	contractSetR := make(map[int]bool)
	batchSetR := make(map[int]bool)
	for _, a := range rhsContractingAxes {
		contractSetR[a] = true
	}
	for _, a := range rhsBatchAxes {
		batchSetR[a] = true
	}

	var outDims []int
	// Batch dims (from lhs)
	for _, a := range lhsBatchAxes {
		outDims = append(outDims, lShape.Dimensions[a])
	}
	// LHS free dims
	for i, d := range lShape.Dimensions {
		if !contractSet[i] && !batchSetL[i] {
			outDims = append(outDims, d)
		}
	}
	// RHS free dims
	for i, d := range rShape.Dimensions {
		if !contractSetR[i] && !batchSetR[i] {
			outDims = append(outDims, d)
		}
	}
	outShape := shapes.Make(lShape.DType, outDims...)

	data := &dotGeneralData{
		lhsContractingAxes: lhsContractingAxes,
		rhsContractingAxes: rhsContractingAxes,
		lhsBatchAxes:       lhsBatchAxes,
		rhsBatchAxes:       rhsBatchAxes,
		config:             config,
	}
	node := f.addNode(backends.OpTypeDotGeneral, outShape, inputs, data)
	if config.OutputDType != dtypes.InvalidDType && config.OutputDType != dtype {
		return f.ConvertDType(node, config.OutputDType)
	}
	return node, nil
}

// ─── Fused Ops ──────────────────────────────────────────────────────────────

type fusedSoftmaxData struct {
	axis int
}

func (f *Function) FusedSoftmax(x backends.Value, axis int) (backends.Value, error) {
	inputs, err := f.checkValues("FusedSoftmax", x)
	if err != nil {
		return nil, err
	}
	rank := inputs[0].shape.Rank()
	if axis < 0 {
		axis += rank
	}
	if axis != rank-1 {
		return nil, errors.New("metal: FusedSoftmax kernel uses row-major contiguous rows; only the innermost axis is supported")
	}
	return f.addNode(backends.OpTypeFusedSoftmax, inputs[0].shape, inputs, &fusedSoftmaxData{axis: axis}), nil
}

type fusedGeluData struct {
	exact bool
}

func (f *Function) FusedGelu(x backends.Value, exact bool) (backends.Value, error) {
	inputs, err := f.checkValues("FusedGelu", x)
	if err != nil {
		return nil, err
	}
	return f.addNode(backends.OpTypeFusedGelu, inputs[0].shape, inputs, &fusedGeluData{exact: exact}), nil
}

type fusedLayerNormData struct {
	axes    []int
	epsilon float64
}

func (f *Function) FusedLayerNorm(x backends.Value, axes []int, epsilon float64, gamma, beta backends.Value) (backends.Value, error) {
	vals := []backends.Value{x}
	if gamma != nil {
		vals = append(vals, gamma)
	}
	if beta != nil {
		vals = append(vals, beta)
	}
	inputs, err := f.checkValues("FusedLayerNorm", vals...)
	if err != nil {
		return nil, err
	}
	inRank := inputs[0].shape.Rank()
	if len(axes) == 0 || len(axes) > inRank {
		return nil, errors.Errorf("metal: FusedLayerNorm needs 1..rank axes, got %v", axes)
	}
	for i := 1; i < len(axes); i++ {
		if axes[i] != axes[i-1]+1 {
			return nil, errors.New("metal: FusedLayerNorm requires a contiguous increasing axis range")
		}
	}
	if axes[len(axes)-1] != inRank-1 || axes[0] != inRank-len(axes) {
		return nil, errors.New("metal: FusedLayerNorm kernel expects normalized axes to be the tensor tail dimensions only")
	}
	return f.addNode(backends.OpTypeFusedLayerNorm, inputs[0].shape, inputs, &fusedLayerNormData{axes: axes, epsilon: epsilon}), nil
}

type fusedSDPAData struct {
	numHeads   int
	numKVHeads int
	axesLayout backends.AxesLayout
	scale      float64
	causal     bool
	options    *backends.ScaledDotProductAttentionConfig
}

/*
fusedDenseData configures the single-pass Metal kernel for FusedDense
(matmul → bias → activation).
*/

type fusedDenseData struct {
	activation backends.ActivationType
	/* geluExact selects erf GELU vs tanh approx; FusedDense follows applyDenseActivation (approx GELU). */
	geluExact bool
}

/*
fusedQKVData configures fused q×W projection writing Q/K/V without a full combined buffer.
*/

type fusedQKVData struct {
	QueryDim    int
	KeyValueDim int
	HasBiasQ    bool
	HasBiasK    bool
	HasBiasV    bool
}

func (f *Function) FusedScaledDotProductAttention(
	query, key, value, mask backends.Value,
	numHeads, numKVHeads int,
	axesLayout backends.AxesLayout,
	scale float64,
	causal bool,
	options *backends.ScaledDotProductAttentionConfig) (backends.Value, error) {

	vals := []backends.Value{query, key, value}
	if mask != nil {
		vals = append(vals, mask)
	}
	inputs, err := f.checkValues("FusedScaledDotProductAttention", vals...)
	if err != nil {
		return nil, err
	}
	data := &fusedSDPAData{
		numHeads:   numHeads,
		numKVHeads: numKVHeads,
		axesLayout: axesLayout,
		scale:      scale,
		causal:     causal,
		options:    options,
	}
	return f.addNode(backends.OpTypeFusedScaledDotProductAttention, inputs[0].shape, inputs, data), nil
}

// ─── Gather ─────────────────────────────────────────────────────────────────

type gatherData struct {
	indexVectorAxis    int
	offsetOutputAxes   []int
	collapsedSliceAxes []int
	startIndexMap      []int
	sliceSizes         []int
	indicesAreSorted   bool
}

func (f *Function) Gather(operand, startIndices backends.Value,
	indexVectorAxis int,
	offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes []int,
	indicesAreSorted bool) (backends.Value, error) {

	inputs, err := f.checkValues("Gather", operand, startIndices)
	if err != nil {
		return nil, err
	}
	if inputs[1].shape.DType != dtypes.Int32 {
		return nil, errors.Errorf("Gather: indices must be Int32, got %s", inputs[1].shape.DType)
	}
	if inputs[0].shape.DType.Size() <= 0 || inputs[0].shape.DType.Size() > 16 {
		return nil, errors.Errorf("Gather: metal gather_bytes supports element sizes 1..16 bytes, got %s", inputs[0].shape.DType)
	}

	// Compute output shape per XLA gather semantics:
	// batch dims from indices (all dims except indexVectorAxis) +
	// offset dims from sliceSizes (non-collapsed)
	idxShape := inputs[1].shape
	var batchDims []int
	for i := 0; i < idxShape.Rank(); i++ {
		if i != indexVectorAxis {
			batchDims = append(batchDims, idxShape.Dimensions[i])
		}
	}
	offsetSet := make(map[int]bool)
	for _, a := range collapsedSliceAxes {
		offsetSet[a] = true
	}
	var offsetDims []int
	for i, s := range sliceSizes {
		if !offsetSet[i] {
			offsetDims = append(offsetDims, s)
		}
	}

	// Insert offset dims at the right positions
	outRank := len(batchDims) + len(offsetDims)
	outDims := make([]int, outRank)
	oSet := make(map[int]bool)
	for _, a := range offsetOutputAxes {
		oSet[a] = true
	}
	bi, oi := 0, 0
	for i := 0; i < outRank; i++ {
		if oSet[i] {
			outDims[i] = offsetDims[oi]
			oi++
		} else {
			outDims[i] = batchDims[bi]
			bi++
		}
	}

	outShape := shapes.Make(inputs[0].shape.DType, outDims...)
	data := &gatherData{
		indexVectorAxis:    indexVectorAxis,
		offsetOutputAxes:   offsetOutputAxes,
		collapsedSliceAxes: collapsedSliceAxes,
		startIndexMap:      startIndexMap,
		sliceSizes:         sliceSizes,
		indicesAreSorted:   indicesAreSorted,
	}
	return f.addNode(backends.OpTypeGather, outShape, inputs, data), nil
}

// ─── Scatter ────────────────────────────────────────────────────────────────

type scatterData struct {
	indexVectorAxis          int
	updateWindowAxes         []int
	insertedWindowAxes       []int
	scatterAxesToOperandAxes []int
	indicesAreSorted         bool
	uniqueIndices            bool
}

func (f *Function) scatterOp(opType backends.OpType, name string,
	operand, scatterIndices, updates backends.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool) (backends.Value, error) {

	inputs, err := f.checkValues(name, operand, scatterIndices, updates)
	if err != nil {
		return nil, err
	}
	if inputs[1].shape.DType != dtypes.Int32 {
		return nil, errors.Errorf("%s: scatter indices must be Int32, got %s", name, inputs[1].shape.DType)
	}
	if inputs[0].shape.DType != inputs[2].shape.DType {
		return nil, errors.Errorf("%s: operand dtype %s != updates dtype %s",
			name, inputs[0].shape.DType, inputs[2].shape.DType)
	}
	dt := inputs[0].shape.DType
	if scatterElemKind(dt) < 0 && dt != dtypes.Float16 && dt != dtypes.BFloat16 {
		return nil, errors.Errorf("%s: metal scatter supports float16/bfloat16/float32/int32/uint32/int64/uint64, got %s",
			name, dt)
	}
	// Scatter output has same shape as operand
	data := &scatterData{
		indexVectorAxis:          indexVectorAxis,
		updateWindowAxes:         updateWindowAxes,
		insertedWindowAxes:       insertedWindowAxes,
		scatterAxesToOperandAxes: scatterAxesToOperandAxes,
		indicesAreSorted:         indicesAreSorted,
		uniqueIndices:            uniqueIndices,
	}
	return f.addNode(opType, inputs[0].shape, inputs, data), nil
}

func (f *Function) ScatterSum(operand, scatterIndices, updates backends.Value,
	indexVectorAxis int, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool) (backends.Value, error) {
	return f.scatterOp(backends.OpTypeScatterSum, "ScatterSum", operand, scatterIndices, updates,
		indexVectorAxis, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes,
		indicesAreSorted, uniqueIndices)
}

func (f *Function) ScatterMax(operand, scatterIndices, updates backends.Value,
	indexVectorAxis int, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool) (backends.Value, error) {
	return f.scatterOp(backends.OpTypeScatterMax, "ScatterMax", operand, scatterIndices, updates,
		indexVectorAxis, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes,
		indicesAreSorted, uniqueIndices)
}

func (f *Function) ScatterMin(operand, scatterIndices, updates backends.Value,
	indexVectorAxis int, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool) (backends.Value, error) {
	return f.scatterOp(backends.OpTypeScatterMin, "ScatterMin", operand, scatterIndices, updates,
		indexVectorAxis, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes,
		indicesAreSorted, uniqueIndices)
}

// ─── Pad ────────────────────────────────────────────────────────────────────

type padData struct {
	axesConfig []backends.PadAxis
}

func (f *Function) Pad(x, fillValue backends.Value, axesConfig ...backends.PadAxis) (backends.Value, error) {
	inputs, err := f.checkValues("Pad", x, fillValue)
	if err != nil {
		return nil, err
	}
	inShape := inputs[0].shape
	outDims := make([]int, inShape.Rank())
	for i, d := range inShape.Dimensions {
		p := axesConfig[i]
		// output_dim = pad_low + (input_dim - 1) * (interior + 1) + 1 + pad_end
		outDims[i] = p.Start + d + (d-1)*p.Interior + p.End
	}
	outShape := shapes.Make(inShape.DType, outDims...)
	return f.addNode(backends.OpTypePad, outShape, inputs, &padData{axesConfig: axesConfig}), nil
}

// ─── Reverse ────────────────────────────────────────────────────────────────

type reverseData struct {
	axes []int
}

func (f *Function) Reverse(x backends.Value, axes ...int) (backends.Value, error) {
	inputs, err := f.checkValues("Reverse", x)
	if err != nil {
		return nil, err
	}
	return f.addNode(backends.OpTypeReverse, inputs[0].shape, inputs, &reverseData{axes: axes}), nil
}

// ─── ArgMinMax ──────────────────────────────────────────────────────────────

type argMinMaxData struct {
	axis        int
	outputDType dtypes.DType
	isMin       bool
}

func (f *Function) ArgMinMax(x backends.Value, axis int, outputDType dtypes.DType, isMin bool) (backends.Value, error) {
	inputs, err := f.checkValues("ArgMinMax", x)
	if err != nil {
		return nil, err
	}
	if outputDType != dtypes.Int32 && outputDType != dtypes.Int64 {
		return nil, errors.Errorf("metal: ArgMinMax supports int32/int64 indices, got %s", outputDType)
	}
	inShape := inputs[0].shape
	if inShape.DType != dtypes.Float32 && inShape.DType != dtypes.Float16 && inShape.DType != dtypes.BFloat16 {
		return nil, errors.Errorf("metal: ArgMinMax supports bfloat16/float16/float32 input only, got %s", inShape.DType)
	}
	// Output has the reduce axis removed, with outputDType
	var outDims []int
	for i, d := range inShape.Dimensions {
		if i != axis {
			outDims = append(outDims, d)
		}
	}
	if len(outDims) == 0 {
		outDims = []int{}
	}
	outShape := shapes.Make(outputDType, outDims...)
	return f.addNode(backends.OpTypeArgMinMax, outShape, inputs, &argMinMaxData{axis: axis, outputDType: outputDType, isMin: isMin}), nil
}

// ─── ConvGeneral ────────────────────────────────────────────────────────────

type convGeneralData struct {
	axes              backends.ConvolveAxesConfig
	strides           []int
	paddings          [][2]int
	inputDilations    []int
	kernelDilations   []int
	channelGroupCount int
	batchGroupCount   int
}

func (f *Function) ConvGeneral(input, kernel backends.Value,
	axes backends.ConvolveAxesConfig,
	strides []int, paddings [][2]int,
	inputDilations, kernelDilations []int,
	channelGroupCount, batchGroupCount int) (backends.Value, error) {

	inputs, err := f.checkValues("ConvGeneral", input, kernel)
	if err != nil {
		return nil, err
	}
	idt, kdt := inputs[0].shape.DType, inputs[1].shape.DType
	if idt != dtypes.Float32 && idt != dtypes.Float16 && idt != dtypes.BFloat16 {
		return nil, errors.Errorf("ConvGeneral: metal supports float16/bfloat16/float32 input, got %s", idt)
	}
	if kdt != idt {
		return nil, errors.Errorf("ConvGeneral: kernel dtype %s must match input %s", kdt, idt)
	}
	inShape := inputs[0].shape
	kShape := inputs[1].shape
	spatialRank := len(axes.InputSpatial)

	// Compute output spatial dimensions
	outDims := make([]int, inShape.Rank())
	outDims[axes.OutputBatch] = inShape.Dimensions[axes.InputBatch]
	outDims[axes.OutputChannels] = kShape.Dimensions[axes.KernelOutputChannels]

	for i := 0; i < spatialRank; i++ {
		inDim := inShape.Dimensions[axes.InputSpatial[i]]
		kDim := kShape.Dimensions[axes.KernelSpatial[i]]
		// Apply input dilation
		if inputDilations != nil && i < len(inputDilations) {
			inDim = (inDim-1)*inputDilations[i] + 1
		}
		// Apply kernel dilation
		effectiveK := kDim
		if kernelDilations != nil && i < len(kernelDilations) {
			effectiveK = (kDim-1)*kernelDilations[i] + 1
		}
		// Add padding
		padLow, padHigh := 0, 0
		if paddings != nil && i < len(paddings) {
			padLow = paddings[i][0]
			padHigh = paddings[i][1]
		}
		s := 1
		if strides != nil && i < len(strides) {
			s = strides[i]
		}
		outDims[axes.OutputSpatial[i]] = (inDim+padLow+padHigh-effectiveK)/s + 1
	}

	outShape := shapes.Make(inShape.DType, outDims...)
	data := &convGeneralData{
		axes:              axes,
		strides:           strides,
		paddings:          paddings,
		inputDilations:    inputDilations,
		kernelDilations:   kernelDilations,
		channelGroupCount: channelGroupCount,
		batchGroupCount:   batchGroupCount,
	}
	return f.addNode(backends.OpTypeConvGeneral, outShape, inputs, data), nil
}

// ─── ReduceWindow ───────────────────────────────────────────────────────────

type reduceWindowData struct {
	reductionType    backends.ReduceOpType
	windowDimensions []int
	strides          []int
	baseDilations    []int
	windowDilations  []int
	paddings         [][2]int
}

func (f *Function) ReduceWindow(x backends.Value,
	reductionType backends.ReduceOpType,
	windowDimensions, strides, baseDilations, windowDilations []int,
	paddings [][2]int) (backends.Value, error) {

	inputs, err := f.checkValues("ReduceWindow", x)
	if err != nil {
		return nil, err
	}
	idt := inputs[0].shape.DType
	if idt != dtypes.Float32 && idt != dtypes.Float16 && idt != dtypes.BFloat16 {
		return nil, errors.Errorf("ReduceWindow: metal supports float16/bfloat16/float32, got %s", idt)
	}
	inShape := inputs[0].shape
	rank := inShape.Rank()

	outDims := make([]int, rank)
	for i := 0; i < rank; i++ {
		inDim := inShape.Dimensions[i]
		// Apply base dilation
		if baseDilations != nil && i < len(baseDilations) {
			inDim = (inDim-1)*baseDilations[i] + 1
		}
		// Apply padding
		padLow, padHigh := 0, 0
		if paddings != nil && i < len(paddings) {
			padLow = paddings[i][0]
			padHigh = paddings[i][1]
		}
		// Effective window
		wDim := windowDimensions[i]
		if windowDilations != nil && i < len(windowDilations) {
			wDim = (windowDimensions[i]-1)*windowDilations[i] + 1
		}
		s := 1
		if strides != nil && i < len(strides) {
			s = strides[i]
		}
		outDims[i] = (inDim+padLow+padHigh-wDim)/s + 1
	}

	outShape := shapes.Make(inShape.DType, outDims...)
	data := &reduceWindowData{
		reductionType:    reductionType,
		windowDimensions: windowDimensions,
		strides:          strides,
		baseDilations:    baseDilations,
		windowDilations:  windowDilations,
		paddings:         paddings,
	}
	return f.addNode(backends.OpTypeReduceWindow, outShape, inputs, data), nil
}

// poolAxesInfo describes how a 4D pooling layout maps to NCHW.
type poolAxesInfo struct {
	spatialAxes    [2]int
	nonSpatialAxes [2]int
	toNCHW         []int
	fromNCHW       []int
	needsTranspose bool
	spatialWindow  [2]int64
	spatialStrides [2]int64
	padBefore      [2]int64
	padAfter       [2]int64
}

func detectPoolAxes(windowDimensions, windowStrides []int, paddings [][2]int) (poolAxesInfo, error) {
	var info poolAxesInfo

	var spatialAxes, nonSpatialAxes []int
	for i := range 4 {
		isSpatial := false
		if windowDimensions[i] > 1 {
			isSpatial = true
		}
		if windowStrides != nil && i < len(windowStrides) && windowStrides[i] > 1 {
			isSpatial = true
		}
		if paddings != nil && i < len(paddings) && (paddings[i][0] != 0 || paddings[i][1] != 0) {
			isSpatial = true
		}
		if isSpatial {
			spatialAxes = append(spatialAxes, i)
		} else {
			nonSpatialAxes = append(nonSpatialAxes, i)
		}
	}

	if len(spatialAxes) != 2 || len(nonSpatialAxes) != 2 {
		return info, errors.Errorf("expected exactly 2 spatial axes (window > 1), got %d spatial %v, %d non-spatial %v",
			len(spatialAxes), spatialAxes, len(nonSpatialAxes), nonSpatialAxes)
	}

	info.spatialAxes = [2]int{spatialAxes[0], spatialAxes[1]}
	info.nonSpatialAxes = [2]int{nonSpatialAxes[0], nonSpatialAxes[1]}

	info.toNCHW = []int{nonSpatialAxes[0], nonSpatialAxes[1], spatialAxes[0], spatialAxes[1]}
	info.needsTranspose = info.toNCHW[0] != 0 || info.toNCHW[1] != 1 || info.toNCHW[2] != 2 || info.toNCHW[3] != 3

	info.fromNCHW = make([]int, 4)
	for i, v := range info.toNCHW {
		info.fromNCHW[v] = i
	}

	info.spatialWindow = [2]int64{int64(windowDimensions[spatialAxes[0]]), int64(windowDimensions[spatialAxes[1]])}
	info.spatialStrides = [2]int64{1, 1}
	if windowStrides != nil {
		info.spatialStrides = [2]int64{int64(windowStrides[spatialAxes[0]]), int64(windowStrides[spatialAxes[1]])}
	}
	if paddings != nil {
		info.padBefore = [2]int64{int64(paddings[spatialAxes[0]][0]), int64(paddings[spatialAxes[1]][0])}
		info.padAfter = [2]int64{int64(paddings[spatialAxes[0]][1]), int64(paddings[spatialAxes[1]][1])}
	}

	return info, nil
}

type selectAndScatterMaxData struct {
	windowDimensions []int
	strides          []int
	baseDilations    []int
	windowDilations  []int
	paddings         [][2]int
}

func (f *Function) SelectAndScatterMax(operand, source backends.Value, windowDimensions, windowStrides []int, paddings [][2]int) (backends.Value, error) {
	inputs, err := f.checkValues("SelectAndScatterMax", operand, source)
	if err != nil {
		return nil, err
	}
	opN, srcN := inputs[0], inputs[1]
	if opN.shape.Rank() != 4 {
		return nil, errors.Errorf("SelectAndScatterMax: only 4D tensors supported, got rank %d", opN.shape.Rank())
	}
	dt := opN.shape.DType
	if dt != dtypes.Float32 && dt != dtypes.Float16 && dt != dtypes.BFloat16 {
		return nil, errors.Errorf("SelectAndScatterMax: metal supports float16/bfloat16/float32, got %s", dt)
	}
	exp, err := shapeinference.ReduceWindowOp(opN.shape, windowDimensions, windowStrides, nil, nil, paddings)
	if err != nil {
		return nil, errors.WithMessage(err, "SelectAndScatterMax")
	}
	if !srcN.shape.Equal(exp) {
		return nil, errors.Errorf("SelectAndScatterMax: source shape %s, want pool gradient shape %s",
			srcN.shape, exp)
	}

	axesInfo, err := detectPoolAxes(windowDimensions, windowStrides, paddings)
	if err != nil {
		return nil, errors.WithMessage(err, "SelectAndScatterMax")
	}

	opNCHW, srcNCHW := opN, srcN
	if axesInfo.needsTranspose {
		tOp, err := f.Transpose(operand, axesInfo.toNCHW...)
		if err != nil {
			return nil, errors.Wrap(err, "SelectAndScatterMax: transpose operand to NCHW")
		}
		opNCHW = tOp.(*Node)
		tSrc, err := f.Transpose(source, axesInfo.toNCHW...)
		if err != nil {
			return nil, errors.Wrap(err, "SelectAndScatterMax: transpose source to NCHW")
		}
		srcNCHW = tSrc.(*Node)
	}

	wd := []int{1, 1, int(axesInfo.spatialWindow[0]), int(axesInfo.spatialWindow[1])}
	st := []int{1, 1, int(axesInfo.spatialStrides[0]), int(axesInfo.spatialStrides[1])}
	pad := make([][2]int, 4)
	pad[2] = [2]int{int(axesInfo.padBefore[0]), int(axesInfo.padAfter[0])}
	pad[3] = [2]int{int(axesInfo.padBefore[1]), int(axesInfo.padAfter[1])}

	data := &selectAndScatterMaxData{
		windowDimensions: wd,
		strides:          st,
		baseDilations:    []int{1, 1, 1, 1},
		windowDilations:  []int{1, 1, 1, 1},
		paddings:         pad,
	}

	inner := f.addNode(backends.OpTypeSelectAndScatterMax, opNCHW.shape, []*Node{opNCHW, srcNCHW}, data)
	if axesInfo.needsTranspose {
		out, err := f.Transpose(inner, axesInfo.fromNCHW...)
		if err != nil {
			return nil, errors.Wrap(err, "SelectAndScatterMax: transpose from NCHW")
		}
		return out, nil
	}
	return inner, nil
}

// ─── RNGBitGenerator ────────────────────────────────────────────────────────

func (f *Function) RNGBitGenerator(stateOp backends.Value, shape shapes.Shape) (newState, values backends.Value, err error) {
	opType := backends.OpTypeRNGBitGenerator
	inputs, err := f.verifyAndCastValues(opType.String(), stateOp)
	if err != nil {
		return nil, nil, err
	}
	state := inputs[0]
	if !state.shape.Equal(backends.RNGStateShape) {
		return nil, nil, errors.Errorf(
			"expected random state to be shaped %s, got state.shape=%s instead for RNGBitGenerator",
			backends.RNGStateShape,
			state.shape,
		)
	}
	outputShapes := []shapes.Shape{
		state.shape.Clone(),
		shape.Clone(),
	}
	node := f.newMultiOutputsNode(opType, outputShapes, nil, state)
	newState = node.multiOutputsNodes[0]
	values = node.multiOutputsNodes[1]
	return
}

// ─── Clamp / IsNaN / comparisons ─────────────────────────────────────────────

func (f *Function) Clamp(minV, x, maxV backends.Value) (backends.Value, error) {
	v, err := f.Max(minV, x)
	if err != nil {
		return nil, errors.WithMessage(err, "metal Clamp")
	}
	return f.Min(v, maxV)
}

func (f *Function) totalOrderComparison(totalOp, plainOp backends.OpType, name string, lhs, rhs backends.Value) (backends.Value, error) {
	inputs, err := f.checkValues(name, lhs, rhs)
	if err != nil {
		return nil, err
	}
	dt := inputs[0].shape.DType
	switch dt {
	case dtypes.Float32, dtypes.Float16, dtypes.BFloat16:
		boolShape, err := shapeinference.ComparisonOp(totalOp, inputs[0].shape, inputs[1].shape)
		if err != nil {
			return nil, errors.WithMessagef(err, "metal: %s", name)
		}
		operandShape := shapes.Make(dt, boolShape.Dimensions...)
		lhsNode, err := f.broadcastNodeToShape(name, inputs[0], operandShape)
		if err != nil {
			return nil, err
		}
		rhsNode, err := f.broadcastNodeToShape(name, inputs[1], operandShape)
		if err != nil {
			return nil, err
		}
		return f.addNode(totalOp, boolShape, []*Node{lhsNode, rhsNode}, nil), nil
	default:
		return f.cmpOp(plainOp, name, lhs, rhs)
	}
}

func (f *Function) EqualTotalOrder(lhs, rhs backends.Value) (backends.Value, error) {
	return f.totalOrderComparison(backends.OpTypeEqualTotalOrder, backends.OpTypeEqual, "EqualTotalOrder", lhs, rhs)
}
func (f *Function) NotEqualTotalOrder(lhs, rhs backends.Value) (backends.Value, error) {
	return f.totalOrderComparison(backends.OpTypeNotEqualTotalOrder, backends.OpTypeNotEqual, "NotEqualTotalOrder", lhs, rhs)
}
func (f *Function) LessThanTotalOrder(lhs, rhs backends.Value) (backends.Value, error) {
	return f.totalOrderComparison(backends.OpTypeLessThanTotalOrder, backends.OpTypeLessThan, "LessThanTotalOrder", lhs, rhs)
}
func (f *Function) LessOrEqualTotalOrder(lhs, rhs backends.Value) (backends.Value, error) {
	return f.totalOrderComparison(backends.OpTypeLessOrEqualTotalOrder, backends.OpTypeLessOrEqual, "LessOrEqualTotalOrder", lhs, rhs)
}
func (f *Function) GreaterThanTotalOrder(lhs, rhs backends.Value) (backends.Value, error) {
	return f.totalOrderComparison(backends.OpTypeGreaterThanTotalOrder, backends.OpTypeGreaterThan, "GreaterThanTotalOrder", lhs, rhs)
}
func (f *Function) GreaterOrEqualTotalOrder(lhs, rhs backends.Value) (backends.Value, error) {
	return f.totalOrderComparison(backends.OpTypeGreaterOrEqualTotalOrder, backends.OpTypeGreaterOrEqual, "GreaterOrEqualTotalOrder", lhs, rhs)
}

// ─── Collectives ────────────────────────────────────────────────────────────

func (f *Function) AllReduce(operands []backends.Value, _ backends.ReduceOpType, replicaGroups [][]int) ([]backends.Value, error) {
	if len(operands) == 0 {
		return nil, errors.New("AllReduce: need at least one operand")
	}
	for gi, g := range replicaGroups {
		for _, id := range g {
			if id != 0 {
				return nil, errors.Errorf("metal AllReduce: only local device index 0 exists, got %d in group %d", id, gi)
			}
		}
	}
	out := make([]backends.Value, len(operands))
	for i, o := range operands {
		v, err := f.Identity(o)
		if err != nil {
			return nil, err
		}
		out[i] = v
	}
	return out, nil
}

// ─── Batch normalization (inference via primitive ops) ───────────────────────

func (f *Function) batchNormBroadcastParam(param *Node, operandShape shapes.Shape, featureAxis int) (*Node, error) {
	fa := featureAxis
	if fa < 0 {
		fa += operandShape.Rank()
	}
	if fa < 0 || fa >= operandShape.Rank() {
		return nil, errors.Errorf("metal BatchNorm: feature axis %d invalid for rank %d", featureAxis, operandShape.Rank())
	}
	v, err := f.BroadcastInDim(param, operandShape, []int{fa})
	if err != nil {
		return nil, err
	}
	return v.(*Node), nil
}

// BatchNormForInference implements inference BN using elementwise ops (float16/float32).
func (f *Function) BatchNormForInference(operand, scale, offset, mean, variance backends.Value, epsilon float32, featureAxis int) (backends.Value, error) {
	inputs, err := f.checkValues("BatchNormForInference", operand, scale, offset, mean, variance)
	if err != nil {
		return nil, err
	}
	op := inputs[0]
	dt := op.shape.DType
	if dt != dtypes.Float16 && dt != dtypes.Float32 && dt != dtypes.BFloat16 {
		return nil, errors.Errorf("metal BatchNormForInference: float16/bfloat16/float32 only, got %s", dt)
	}
	for i, n := range inputs[1:] {
		if n.shape.DType != dt {
			return nil, errors.Errorf("metal BatchNormForInference: input %d dtype %s != operand %s", i+1, n.shape.DType, dt)
		}
	}
	meanB, err := f.batchNormBroadcastParam(inputs[3], op.shape, featureAxis)
	if err != nil {
		return nil, err
	}
	varB, err := f.batchNormBroadcastParam(inputs[4], op.shape, featureAxis)
	if err != nil {
		return nil, err
	}
	scaleB, err := f.batchNormBroadcastParam(inputs[1], op.shape, featureAxis)
	if err != nil {
		return nil, err
	}
	offsetB, err := f.batchNormBroadcastParam(inputs[2], op.shape, featureAxis)
	if err != nil {
		return nil, err
	}
	epsNode, err := f.Constant([]float32{epsilon})
	if err != nil {
		return nil, err
	}
	if dt == dtypes.Float16 || dt == dtypes.BFloat16 {
		epsNode, err = f.ConvertDType(epsNode, dt)
		if err != nil {
			return nil, err
		}
	}
	epsB, err := f.Broadcast(epsNode, op.shape.Dimensions...)
	if err != nil {
		return nil, err
	}
	varEps, err := f.Add(varB, epsB)
	if err != nil {
		return nil, err
	}
	invStd, err := f.Rsqrt(varEps)
	if err != nil {
		return nil, err
	}
	xCent, err := f.Sub(op, meanB)
	if err != nil {
		return nil, err
	}
	norm, err := f.Mul(xCent, invStd)
	if err != nil {
		return nil, err
	}
	scaled, err := f.Mul(norm, scaleB)
	if err != nil {
		return nil, err
	}
	return f.Add(scaled, offsetB)
}

type batchNormTrainingData struct {
	epsilon     float32
	featureAxis int
}

type batchNormGradientData struct {
	epsilon     float32
	featureAxis int
}

// BatchNormForTraining implements XLA batch-norm training on GPU (float16/float32).
func (f *Function) BatchNormForTraining(operand, scale, offset backends.Value, epsilon float32, featureAxis int) (
	normalized, batchMean, batchVariance backends.Value, err error) {
	inputs, err := f.checkValues("BatchNormForTraining", operand, scale, offset)
	if err != nil {
		return nil, nil, nil, err
	}
	op := inputs[0]
	if op.shape.DType != dtypes.Float16 && op.shape.DType != dtypes.Float32 && op.shape.DType != dtypes.BFloat16 {
		return nil, nil, nil, errors.Errorf("metal BatchNormForTraining: float16/bfloat16/float32 only, got %s", op.shape.DType)
	}
	dt := op.shape.DType
	for i, n := range inputs[1:] {
		if n.shape.DType != dt {
			return nil, nil, nil, errors.Errorf("metal BatchNormForTraining: input %d dtype %s != operand %s", i+1, n.shape.DType, dt)
		}
	}
	meanVarShape := inputs[1].shape.Clone()
	outputShapes := []shapes.Shape{
		op.shape.Clone(),
		meanVarShape,
		meanVarShape.Clone(),
	}
	node := f.newMultiOutputsNode(backends.OpTypeBatchNormForTraining, outputShapes,
		&batchNormTrainingData{epsilon: epsilon, featureAxis: featureAxis}, inputs...)
	return node.multiOutputsNodes[0], node.multiOutputsNodes[1], node.multiOutputsNodes[2], nil
}

// BatchNormGradient implements XLA batch-norm gradient on GPU (float16/float32).
func (f *Function) BatchNormGradient(operand, scale, mean, variance, gradOutput backends.Value, epsilon float32, featureAxis int) (
	gradOperand, gradScale, gradOffset backends.Value, err error) {
	inputs, err := f.checkValues("BatchNormGradient", operand, scale, mean, variance, gradOutput)
	if err != nil {
		return nil, nil, nil, err
	}
	op := inputs[0]
	if op.shape.DType != dtypes.Float16 && op.shape.DType != dtypes.Float32 && op.shape.DType != dtypes.BFloat16 {
		return nil, nil, nil, errors.Errorf("metal BatchNormGradient: float16/bfloat16/float32 only, got %s", op.shape.DType)
	}
	dt := op.shape.DType
	for i, n := range inputs[1:] {
		if n.shape.DType != dt {
			return nil, nil, nil, errors.Errorf("metal BatchNormGradient: input %d dtype %s != operand %s", i+1, n.shape.DType, dt)
		}
	}
	outShapes := []shapes.Shape{
		op.shape.Clone(),
		inputs[1].shape.Clone(),
		inputs[1].shape.Clone(), // gradOffset matches scale (same as inferred offset param shape)
	}
	node := f.newMultiOutputsNode(backends.OpTypeBatchNormGradient, outShapes,
		&batchNormGradientData{epsilon: epsilon, featureAxis: featureAxis}, inputs...)
	return node.multiOutputsNodes[0], node.multiOutputsNodes[1], node.multiOutputsNodes[2], nil
}

// ─── FusedDense / FusedAttentionQKVProjection / FusedQuantizedDense ─────────

func (f *Function) applyDenseActivation(x backends.Value, act backends.ActivationType) (backends.Value, error) {
	inputs, err := f.checkValues("activation", x)
	if err != nil {
		return nil, err
	}
	sh := inputs[0].shape
	switch act {
	case backends.ActivationNone:
		return x, nil
	case backends.ActivationRelu:
		z, err := f.scalarConstantForDType(sh.DType, 0)
		if err != nil {
			return nil, err
		}
		zb, err := f.Broadcast(z, sh.Dimensions...)
		if err != nil {
			return nil, err
		}
		return f.Max(x, zb)
	case backends.ActivationTanh:
		return f.Tanh(x)
	case backends.ActivationGelu:
		return f.FusedGelu(x, false)
	case backends.ActivationSilu:
		sig, err := f.Logistic(x)
		if err != nil {
			return nil, err
		}
		return f.Mul(x, sig)
	case backends.ActivationHardSwish:
		// Match simplego fusedDenseApplyActivation (scale 1/6, bias 0.5, clamp to [0,1], then x * clamped).
		xN := inputs[0]
		sh := xN.shape
		scaleConst, err := f.scalarConstantForDType(sh.DType, float32(1.0/6.0))
		if err != nil {
			return nil, err
		}
		biasConst, err := f.scalarConstantForDType(sh.DType, 0.5)
		if err != nil {
			return nil, err
		}
		zeroConst, err := f.scalarConstantForDType(sh.DType, 0)
		if err != nil {
			return nil, err
		}
		oneConst, err := f.scalarConstantForDType(sh.DType, 1)
		if err != nil {
			return nil, err
		}
		scaleB, err := f.Broadcast(scaleConst, sh.Dimensions...)
		if err != nil {
			return nil, err
		}
		biasB, err := f.Broadcast(biasConst, sh.Dimensions...)
		if err != nil {
			return nil, err
		}
		zeroB, err := f.Broadcast(zeroConst, sh.Dimensions...)
		if err != nil {
			return nil, err
		}
		oneB, err := f.Broadcast(oneConst, sh.Dimensions...)
		if err != nil {
			return nil, err
		}
		scaled, err := f.Mul(xN, scaleB)
		if err != nil {
			return nil, err
		}
		shifted, err := f.Add(scaled, biasB)
		if err != nil {
			return nil, err
		}
		clamped, err := f.Clamp(zeroB, shifted, oneB)
		if err != nil {
			return nil, err
		}
		return f.Mul(xN, clamped)
	default:
		return nil, errors.Errorf("metal FusedDense: unknown activation %v", act)
	}
}

func (f *Function) scalarConstantForDType(dtype dtypes.DType, value float32) (backends.Value, error) {
	switch dtype {
	case dtypes.Float16:
		return f.Constant([]float16.Float16{float16.Fromfloat32(value)})
	case dtypes.BFloat16:
		return f.Constant([]bfloat16.BFloat16{bfloat16.FromFloat32(value)})
	case dtypes.Float32:
		return f.Constant([]float32{value})
	default:
		return nil, errors.Errorf("metal activation: unsupported scalar dtype %s", dtype)
	}
}

// FusedDense decomposes to matmul + optional bias + activation (no dedicated fused kernel).
func fusedDenseMetalEligible(xN, wN *Node, biasNode *Node) bool {
	if wN.shape.Rank() != 2 {
		return false
	}
	dt := xN.shape.DType
	if wN.shape.DType != dt {
		return false
	}
	if biasNode != nil && biasNode.shape.DType != dt {
		return false
	}
	switch dt {
	case dtypes.Float16, dtypes.Float32, dtypes.BFloat16:
		return true
	default:
		return false
	}
}

func fusedQKVMetalEligible(xN, wN *Node, biasQ, biasK, biasV *Node) bool {
	if wN.shape.Rank() != 2 {
		return false
	}
	dt := xN.shape.DType
	if wN.shape.DType != dt {
		return false
	}
	checkBias := func(bn *Node) bool {
		if bn == nil {
			return true
		}
		return bn.shape.DType == dt
	}
	if !checkBias(biasQ) || !checkBias(biasK) || !checkBias(biasV) {
		return false
	}
	switch dt {
	case dtypes.Float16, dtypes.Float32, dtypes.BFloat16:
		return true
	default:
		return false
	}
}

// FusedDense applies a single-pass matmul → optional bias → activation on GPU when the weight is rank-2
// and dtypes are float16/float32/bfloat16.
func (f *Function) FusedDense(x, weight, bias backends.Value, activation backends.ActivationType) (backends.Value, error) {
	values := []backends.Value{x, weight}
	if bias != nil {
		values = append(values, bias)
	}
	inputs, err := f.checkValues("FusedDense", values...)
	if err != nil {
		return nil, err
	}
	xNode := inputs[0]
	wNode := inputs[1]
	var biasNode *Node
	if len(inputs) > 2 {
		biasNode = inputs[2]
	}
	if xNode.shape.Rank() < 1 || wNode.shape.Rank() < 2 {
		return nil, errors.Errorf("FusedDense: x rank >= 1 and weight rank >= 2 required, got %d and %d",
			xNode.shape.Rank(), wNode.shape.Rank())
	}
	inFeatures := xNode.shape.Dimensions[xNode.shape.Rank()-1]
	if inFeatures != wNode.shape.Dimensions[0] {
		return nil, errors.Errorf("FusedDense: last dim of x (%d) must match first dim of weight (%d)",
			inFeatures, wNode.shape.Dimensions[0])
	}

	if fusedDenseMetalEligible(xNode, wNode, biasNode) {
		outDims := slices.Clone(xNode.shape.Dimensions[:xNode.shape.Rank()-1])
		outDims = append(outDims, wNode.shape.Dimensions[1])
		outShape := shapes.Make(xNode.shape.DType, outDims...)
		var in []*Node
		if biasNode != nil {
			in = []*Node{xNode, wNode, biasNode}
		} else {
			in = []*Node{xNode, wNode}
		}
		geluExact := false
		return f.addNode(backends.OpTypeFusedDense, outShape, in, &fusedDenseData{
			activation: activation,
			geluExact:  geluExact,
		}), nil
	}

	y, err := f.DotGeneral(xNode, []int{xNode.shape.Rank() - 1}, nil, wNode, []int{0}, nil, backends.DotGeneralConfig{})
	if err != nil {
		return nil, err
	}
	if biasNode != nil {
		yn := y.(*Node)
		bb, err := f.BroadcastInDim(biasNode, yn.shape, []int{yn.shape.Rank() - 1})
		if err != nil {
			return nil, err
		}
		y, err = f.Add(yn, bb)
		if err != nil {
			return nil, err
		}
	}
	return f.applyDenseActivation(y, activation)
}

// FusedAttentionQKVProjection fuses x @ W into separate Q/K/V outputs without materializing the concatenated tensor.
func (f *Function) FusedAttentionQKVProjection(x, wQKV, biasQ, biasK, biasV backends.Value, queryDim, keyValueDim int) (
	queryOut, keyOut, valueOut backends.Value, err error) {

	values := []backends.Value{x, wQKV}
	if biasQ != nil {
		values = append(values, biasQ)
	}
	if biasK != nil {
		values = append(values, biasK)
	}
	if biasV != nil {
		values = append(values, biasV)
	}
	inputs, err := f.checkValues("FusedAttentionQKVProjection", values...)
	if err != nil {
		return nil, nil, nil, err
	}
	xNode := inputs[0]
	wNode := inputs[1]
	if xNode.shape.Rank() < 1 {
		return nil, nil, nil, errors.New("FusedAttentionQKVProjection: x must have rank >= 1")
	}
	rank := xNode.shape.Rank()
	last := xNode.shape.Dimensions[rank-1]
	if last != wNode.shape.Dimensions[0] {
		return nil, nil, nil, errors.Errorf("FusedAttentionQKVProjection: dim mismatch x last %d vs w first %d",
			last, wNode.shape.Dimensions[0])
	}
	combinedDim := queryDim + 2*keyValueDim
	if wNode.shape.Dimensions[1] != combinedDim {
		return nil, nil, nil, errors.Errorf("FusedAttentionQKVProjection: w second dim must be %d, got %d",
			combinedDim, wNode.shape.Dimensions[1])
	}

	var bQ, bK, bV *Node
	idx := 2
	if biasQ != nil {
		bQ = inputs[idx]
		idx++
	}
	if biasK != nil {
		bK = inputs[idx]
		idx++
	}
	if biasV != nil {
		bV = inputs[idx]
		idx++
	}

	if fusedQKVMetalEligible(xNode, wNode, bQ, bK, bV) {
		dt := xNode.shape.DType
		lead := slices.Clone(xNode.shape.Dimensions[:rank-1])
		outQ := shapes.Make(dt, append(slices.Clone(lead), queryDim)...)
		outK := shapes.Make(dt, append(slices.Clone(lead), keyValueDim)...)
		outV := shapes.Make(dt, append(slices.Clone(lead), keyValueDim)...)

		in := []*Node{xNode, wNode}
		if bQ != nil {
			in = append(in, bQ)
		}
		if bK != nil {
			in = append(in, bK)
		}
		if bV != nil {
			in = append(in, bV)
		}

		internal := f.newMultiOutputsNode(backends.OpTypeFusedAttentionQKVProjection,
			[]shapes.Shape{outQ, outK, outV},
			&fusedQKVData{
				QueryDim:    queryDim,
				KeyValueDim: keyValueDim,
				HasBiasQ:    biasQ != nil,
				HasBiasK:    biasK != nil,
				HasBiasV:    biasV != nil,
			},
			in...)
		return internal.multiOutputsNodes[0], internal.multiOutputsNodes[1], internal.multiOutputsNodes[2], nil
	}

	combinedAny, err := f.DotGeneral(xNode, []int{rank - 1}, nil, wNode, []int{0}, nil, backends.DotGeneralConfig{})
	if err != nil {
		return nil, nil, nil, err
	}
	combined := combinedAny.(*Node)

	starts := make([]int, rank)
	limits := make([]int, rank)
	copy(limits, combined.shape.Dimensions)
	strides := make([]int, rank)
	for i := range strides {
		strides[i] = 1
	}

	limits[rank-1] = queryDim
	qAny, err := f.Slice(combined, starts, limits, strides)
	if err != nil {
		return nil, nil, nil, err
	}
	qN := qAny.(*Node)

	starts[rank-1] = queryDim
	limits[rank-1] = queryDim + keyValueDim
	kAny, err := f.Slice(combined, starts, limits, strides)
	if err != nil {
		return nil, nil, nil, err
	}
	kN := kAny.(*Node)

	starts[rank-1] = queryDim + keyValueDim
	limits[rank-1] = queryDim + 2*keyValueDim
	vAny, err := f.Slice(combined, starts, limits, strides)
	if err != nil {
		return nil, nil, nil, err
	}
	vN := vAny.(*Node)

	biasIdx := 2
	if biasQ != nil {
		bb, err := f.BroadcastInDim(inputs[biasIdx], qN.shape, []int{qN.shape.Rank() - 1})
		if err != nil {
			return nil, nil, nil, err
		}
		queryOut, err = f.Add(qN, bb)
		if err != nil {
			return nil, nil, nil, err
		}
		qN = queryOut.(*Node)
		biasIdx++
	} else {
		queryOut = qN
	}
	if biasK != nil {
		bb, err := f.BroadcastInDim(inputs[biasIdx], kN.shape, []int{kN.shape.Rank() - 1})
		if err != nil {
			return nil, nil, nil, err
		}
		keyOut, err = f.Add(kN, bb)
		if err != nil {
			return nil, nil, nil, err
		}
		kN = keyOut.(*Node)
		biasIdx++
	} else {
		keyOut = kN
	}
	if biasV != nil {
		bb, err := f.BroadcastInDim(inputs[biasIdx], vN.shape, []int{vN.shape.Rank() - 1})
		if err != nil {
			return nil, nil, nil, err
		}
		valueOut, err = f.Add(vN, bb)
		if err != nil {
			return nil, nil, nil, err
		}
	} else {
		valueOut = vN
	}
	return queryOut, keyOut, valueOut, nil
}

type nodeFusedQuantizedDense struct {
	scheme       backends.QuantizationScheme
	blockAxis    int
	blockSize    int
	activation   backends.ActivationType
	hasZeroPoint bool
	hasBias      bool
}

// FusedQuantizedDense builds a quantized matmul (+ bias + activation); execution runs on the GPU
// (kernels/quantized_dense.metal).
func (f *Function) FusedQuantizedDense(x, weights, bias backends.Value,
	weightsQuantization *backends.Quantization,
	activation backends.ActivationType) (backends.Value, error) {
	if weightsQuantization == nil {
		return nil, errors.New("FusedQuantizedDense: weightsQuantization must not be nil")
	}

	scales := weightsQuantization.Scale
	zeroPoints := weightsQuantization.ZeroPoint
	blockAxis := weightsQuantization.BlockAxis
	blockSize := weightsQuantization.BlockSize
	scheme := weightsQuantization.Scheme

	values := []backends.Value{x, weights, scales}
	if zeroPoints != nil {
		values = append(values, zeroPoints)
	}
	if bias != nil {
		values = append(values, bias)
	}
	inputs, err := f.checkValues("FusedQuantizedDense", values...)
	if err != nil {
		return nil, err
	}
	xNode := inputs[0]
	wNode := inputs[1]
	sNode := inputs[2]

	if xNode.shape.DType != dtypes.Float32 && xNode.shape.DType != dtypes.Float16 && xNode.shape.DType != dtypes.BFloat16 {
		return nil, errors.Errorf("FusedQuantizedDense: x must be float32, float16, or bfloat16, got %s", xNode.shape.DType)
	}
	if xNode.shape.Rank() < 1 {
		return nil, errors.Errorf("FusedQuantizedDense: x must have rank >= 1, got %d", xNode.shape.Rank())
	}
	K := xNode.shape.Dimensions[xNode.shape.Rank()-1]
	if wNode.shape.Rank() != 2 || wNode.shape.Dimensions[0] != K {
		return nil, errors.Errorf("FusedQuantizedDense: weights must be [%d, N], got %v", K, wNode.shape.Dimensions)
	}
	N := wNode.shape.Dimensions[1]
	numBlocks := (N + blockSize - 1) / blockSize
	if sNode.shape.Rank() != 2 || sNode.shape.Dimensions[0] != K || sNode.shape.Dimensions[1] != numBlocks {
		return nil, errors.Errorf("FusedQuantizedDense: scales must be [%d, %d], got %v",
			K, numBlocks, sNode.shape.Dimensions)
	}

	outDims := make([]int, xNode.shape.Rank())
	copy(outDims, xNode.shape.Dimensions[:xNode.shape.Rank()-1])
	outDims[xNode.shape.Rank()-1] = N
	outShape := shapes.Make(xNode.shape.DType, outDims...)

	if blockAxis != 1 {
		return nil, errors.Errorf("FusedQuantizedDense: only Axis=1 is supported, got %d", blockAxis)
	}
	if scheme == backends.QuantNF4 && zeroPoints != nil {
		return nil, errors.Errorf("FusedQuantizedDense: ZeroPoint must be nil for NF4 quantization scheme")
	}

	data := &nodeFusedQuantizedDense{
		scheme:       scheme,
		blockAxis:    blockAxis,
		blockSize:    blockSize,
		activation:   activation,
		hasZeroPoint: zeroPoints != nil,
		hasBias:      bias != nil,
	}
	return f.addNode(backends.OpTypeFusedQuantizedDense, outShape, inputs, data), nil
}

// ─── Bitcast ────────────────────────────────────────────────────────────────

func (f *Function) Bitcast(x backends.Value, targetDType dtypes.DType) (backends.Value, error) {
	inputs, err := f.checkValues("Bitcast", x)
	if err != nil {
		return nil, err
	}
	inShape := inputs[0].shape
	if inShape.Rank() == 0 {
		return nil, errors.New("Bitcast: rank-0 tensors are not supported")
	}
	srcSize := int(inShape.DType.Size())
	dstSize := int(targetDType.Size())
	var outDims []int
	if srcSize == dstSize {
		outDims = inShape.Dimensions
	} else if srcSize > dstSize {
		if srcSize%dstSize != 0 {
			return nil, errors.Errorf("Bitcast: source element size %d is not divisible by target size %d", srcSize, dstSize)
		}
		// Last dim expands: e.g. f32 -> f16 doubles last dim
		outDims = make([]int, inShape.Rank())
		copy(outDims, inShape.Dimensions)
		factor := srcSize / dstSize
		outDims[len(outDims)-1] *= factor
	} else {
		if dstSize%srcSize != 0 {
			return nil, errors.Errorf("Bitcast: target element size %d is not divisible by source size %d", dstSize, srcSize)
		}
		// Last dim contracts
		outDims = make([]int, inShape.Rank())
		copy(outDims, inShape.Dimensions)
		factor := dstSize / srcSize
		last := len(outDims) - 1
		if outDims[last]%factor != 0 {
			return nil, errors.Errorf("Bitcast: last dimension %d is not divisible by contraction factor %d",
				outDims[last], factor)
		}
		outDims[last] /= factor
	}
	outShape := shapes.Make(targetDType, outDims...)
	// Bitcast is zero-copy — same bits, different type interpretation
	return f.addNode(backends.OpTypeBitcast, outShape, inputs, nil), nil
}
