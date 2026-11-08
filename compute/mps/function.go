// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mps

import (
	"math"
	"reflect"
	"runtime"
	"unsafe"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/notimplemented"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/go-darwinml/internal/bridge"
	"github.com/pkg/errors"
)

// toInt64Slice converts []int to []int64.
func toInt64Slice(s []int) []int64 {
	out := make([]int64, len(s))
	for i, v := range s {
		out[i] = int64(v)
	}
	return out
}

// graphNode represents a value in the computation graph.
type graphNode struct {
	tensor bridge.Tensor // MPSGraphTensor handle
	shape  shapes.Shape
	name   string    // Optional name (for parameters)
	owner  *Function // Which function created this node
}

// Function implements compute.Function for the MPSGraph backend.
type Function struct {
	notimplemented.Function // Bootstrap: unimplemented ops return ErrNotImplemented.

	builder  *Builder
	name     string
	parent   *Function
	returned bool
	params   []*graphNode // Input placeholders
	outputs  []*graphNode // Return values

	// ownCtx is a separate bridge.Context for closures. Nil for the main function.
	ownCtx *bridge.Context

	// Closure capture tracking: capturedParentNodes[i] in the parent maps to
	// capturedLocalNodes[i] (a placeholder) in this closure's context.
	capturedParentNodes []*graphNode
	capturedLocalNodes  []*graphNode

	// controlFlowStep records a control flow operation (While/If/Sort/Call).
	// Currently only one CF op per function is supported; it must be the
	// last operation before Return().
	controlFlowStep *controlFlowStep
}

// controlFlowStep records a pending control flow operation.
type controlFlowStep struct {
	opType       compute.OpType
	inputs       []*graphNode   // Inputs to the CF op (from the current graph)
	outputShapes []shapes.Shape // Output shapes of the CF op
	outputNodes  []*graphNode   // Virtual output nodes (no tensor)
	whileData    *whileStepData
	ifData       *ifStepData
	sortData     *sortStepData
	callData     *callStepData
}

type whileStepData struct {
	condFn *Function
	bodyFn *Function
}

type ifStepData struct {
	trueFn  *Function
	falseFn *Function
}

type sortStepData struct {
	comparatorFn *Function
	axis         int
	isStable     bool
}

type callStepData struct {
	targetFn *Function
}

// Verify interface compliance.
var _ compute.Function = &Function{}

// newFunction creates a new function in the computation graph.
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

// Closure creates a closure function with its own MPSGraph context.
func (f *Function) Closure() (compute.Function, error) {
	if f.builder.backend.ctx == nil {
		return nil, errors.New("Closure: backend context is nil")
	}
	ctx, err := bridge.NewContextWithDevice(f.builder.backend.ctx.DeviceHandle())
	if err != nil {
		return nil, errors.Wrap(err, "Closure: creating context")
	}
	closure := newFunction(f.builder, "", f)
	closure.ownCtx = ctx
	return closure, nil
}

// ctx returns the bridge context for this function.
// Closures use their own context; the main function uses the builder's.
func (f *Function) ctx() *bridge.Context {
	if f.ownCtx != nil {
		return f.ownCtx
	}
	return f.builder.ctx
}

// getOrCreateCaptureNode returns a local placeholder for a parent scope's node.
// If the node was already captured, returns the existing placeholder.
// For nested closures (grandparent captures), propagates through intermediates.
func (f *Function) getOrCreateCaptureNode(parentNode *graphNode) (*graphNode, error) {
	// Check if already captured.
	for i, pn := range f.capturedParentNodes {
		if pn == parentNode {
			return f.capturedLocalNodes[i], nil
		}
	}

	// If parentNode's owner is not our direct parent, propagate through intermediates.
	nodeToCapture := parentNode
	if parentNode.owner != f.parent {
		intermediate, err := f.parent.getOrCreateCaptureNode(parentNode)
		if err != nil {
			return nil, err
		}
		nodeToCapture = intermediate
	}

	// Create a placeholder in our own context for the captured value.
	dims := toInt64Slice(nodeToCapture.shape.Dimensions)
	dtype := dtypeToBridgeDType(nodeToCapture.shape.DType)
	tensor, err := f.ctx().Placeholder(dtype, dims)
	if err != nil {
		return nil, errors.Wrap(err, "capture: creating placeholder")
	}
	node := &graphNode{tensor: tensor, shape: nodeToCapture.shape, owner: f}

	f.capturedParentNodes = append(f.capturedParentNodes, nodeToCapture)
	f.capturedLocalNodes = append(f.capturedLocalNodes, node)

	return node, nil
}

// resolveNode converts a compute.Value to a graphNode, capturing parent values for closures.
func (f *Function) resolveNode(v compute.Value) (*graphNode, error) {
	node, ok := v.(*graphNode)
	if !ok {
		return nil, errors.Errorf("expected *graphNode, got %T", v)
	}
	if node.tensor == nil {
		return nil, errors.Wrap(compute.ErrNotImplemented, "mps: operations on control-flow outputs or zero-sized tensors are not supported")
	}
	// Fast path: no closure context or same owner — no capture needed.
	if f.ownCtx == nil || node.owner == nil || node.owner == f {
		return node, nil
	}
	// Check if from an ancestor function.
	for p := f.parent; p != nil; p = p.parent {
		if node.owner == p {
			return f.getOrCreateCaptureNode(node)
		}
	}
	return nil, errors.Errorf("node from unrelated function scope")
}

// resolveNodes converts multiple compute.Value to graphNodes, handling closure captures.
func (f *Function) resolveNodes(name string, values ...compute.Value) ([]*graphNode, error) {
	nodes := make([]*graphNode, len(values))
	for i, v := range values {
		n, err := f.resolveNode(v)
		if err != nil {
			return nil, errors.Wrapf(err, "%s: input #%d", name, i)
		}
		nodes[i] = n
	}
	return nodes, nil
}

// makeScalarConst creates a scalar constant of the given dtype with the given float64 value.
// It creates the constant as float32 and then casts to the target dtype if needed.
func (f *Function) makeScalarConst(val float64, dt dtypes.DType) (bridge.Tensor, error) {
	f32Val := float32(val)
	tensor, err := f.ctx().Constant(unsafe.Pointer(&f32Val), 4, dtypeToBridgeDType(dtypes.Float32), []int64{1})
	if err != nil {
		return nil, err
	}
	// Reshape to scalar
	tensor, err = f.ctx().Reshape(tensor, nil)
	if err != nil {
		return nil, err
	}
	// Cast to target dtype if not already float32
	if dt != dtypes.Float32 {
		tensor, err = f.ctx().Cast(tensor, dtypeToBridgeDType(dt))
		if err != nil {
			return nil, err
		}
	}
	return tensor, nil
}

// validateClosure validates that a compute.Function is a compiled closure of the current function.
func (f *Function) validateClosure(opName, closureName string, closure compute.Function) (*Function, error) {
	fn, ok := closure.(*Function)
	if !ok {
		return nil, errors.Errorf("%s: %s must be a *mpsgraph.Function, got %T", opName, closureName, closure)
	}
	if fn.parent != f {
		return nil, errors.Errorf("%s: %s must be a closure of the current function", opName, closureName)
	}
	if !fn.returned {
		return nil, errors.Errorf("%s: %s must have Return() called", opName, closureName)
	}
	return fn, nil
}

// compileClosure compiles a closure Function into an Executable.
// The closure's feeds are: parameters first, then captured local nodes.
func (f *Function) compileClosure() (*Executable, error) {
	// Determine which context to compile from.
	// Closures have their own context; named functions (for Call) use the builder's context.
	ctx := f.ownCtx
	if ctx == nil {
		ctx = f.builder.ctx
	}
	if ctx == nil {
		return nil, errors.New("compileClosure: no context available")
	}

	allFeeds := make([]*graphNode, 0, len(f.params)+len(f.capturedLocalNodes))
	allFeeds = append(allFeeds, f.params...)
	allFeeds = append(allFeeds, f.capturedLocalNodes...)

	info := buildCompileInfo(allFeeds, f.outputs)
	exec, err := ctx.Compile(info)
	if err != nil {
		return nil, errors.Wrap(err, "compileClosure")
	}

	inputNames, inputShapes := collectParamInfo(allFeeds)
	return newExecutable(
		f.builder.backend,
		exec,
		inputNames,
		inputShapes,
		collectOutputShapes(f.outputs),
	), nil
}



// ===========================================================================
// Lifecycle: Parameter, Constant, Return
// ===========================================================================

// Parameter creates an input placeholder.
func (f *Function) Parameter(name string, shape shapes.Shape, sharding *compute.ShardingSpec) (compute.Value, error) {
	if err := checkDTypeSupported(shape.DType); err != nil {
		return nil, errors.Wrapf(err, "Parameter(%s)", name)
	}
	dims := toInt64Slice(shape.Dimensions)
	dtype := dtypeToBridgeDType(shape.DType)
	tensor, err := f.ctx().Placeholder(dtype, dims)
	if err != nil {
		return nil, errors.Wrapf(err, "Parameter(%s)", name)
	}
	node := &graphNode{tensor: tensor, shape: shape, name: name, owner: f}
	f.params = append(f.params, node)
	return node, nil
}

// Constant creates a constant tensor.
func (f *Function) Constant(flat any, dims ...int) (compute.Value, error) {
	flatVal := reflect.ValueOf(flat)
	if flatVal.Kind() != reflect.Slice {
		return nil, errors.Errorf("Constant: expected slice, got %T", flat)
	}
	dt := dtypes.FromGoType(flatVal.Type().Elem())
	if err := checkDTypeSupported(dt); err != nil {
		return nil, errors.Wrap(err, "Constant")
	}
	shape := shapes.Make(dt, dims...)

	bridgeDType := dtypeToBridgeDType(dt)

	var dataPtr unsafe.Pointer
	var nbytes int64
	if flatVal.Len() > 0 {
		dataPtr = unsafe.Pointer(flatVal.Pointer())
		nbytes = int64(flatVal.Len()) * int64(dt.Size())
	}

	// MPSGraph requires shape.count > 0 (no rank-0 tensors for constants).
	// For scalars, create as [1] and reshape to scalar.
	isScalar := len(dims) == 0
	shapeDims := toInt64Slice(dims)
	if isScalar {
		shapeDims = []int64{1}
	}

	tensor, err := f.ctx().Constant(dataPtr, nbytes, bridgeDType, shapeDims)
	runtime.KeepAlive(flat) // Prevent GC from collecting flat during CGo call
	if err != nil {
		return nil, errors.Wrap(err, "Constant")
	}

	if isScalar {
		tensor, err = f.ctx().Reshape(tensor, nil)
		if err != nil {
			return nil, errors.Wrap(err, "Constant: reshape to scalar")
		}
	}

	return &graphNode{tensor: tensor, shape: shape, owner: f}, nil
}

// Return marks the function outputs.
func (f *Function) Return(outputs []compute.Value, shardings []*compute.ShardingSpec) error {
	nodes := make([]*graphNode, len(outputs))
	for i, v := range outputs {
		node, ok := v.(*graphNode)
		if !ok {
			return errors.Errorf("Return: input #%d: expected *graphNode, got %T", i, v)
		}
		if node.tensor != nil {
			var err error
			node, err = f.resolveNode(v)
			if err != nil {
				return errors.Wrapf(err, "Return: input #%d", i)
			}
		}
		nodes[i] = node
	}
	f.outputs = nodes
	f.returned = true
	return nil
}

// Call calls another function with the given inputs.
func (f *Function) Call(fn compute.Function, inputs ...compute.Value) ([]compute.Value, error) {
	if f.controlFlowStep != nil {
		return nil, errors.Wrap(compute.ErrNotImplemented, "Call: only one control flow operation per function is supported")
	}
	inputNodes, err := f.resolveNodes("Call", inputs...)
	if err != nil {
		return nil, err
	}

	targetFn, ok := fn.(*Function)
	if !ok {
		return nil, errors.Errorf("Call: target must be *mpsgraph.Function, got %T", fn)
	}
	if targetFn.builder != f.builder {
		return nil, errors.New("Call: target function must be from the same builder")
	}
	if !targetFn.returned {
		return nil, errors.Errorf("Call: target function %q must have Return() called", targetFn.name)
	}

	// Validate inputs match target parameters.
	if len(inputNodes) != len(targetFn.params) {
		return nil, errors.Errorf("Call: function %q expects %d parameters, got %d inputs",
			targetFn.name, len(targetFn.params), len(inputNodes))
	}
	for i, param := range targetFn.params {
		if !param.shape.Equal(inputNodes[i].shape) {
			return nil, errors.Errorf("Call: parameter %d shape mismatch: expected %s, got %s",
				i, param.shape, inputNodes[i].shape)
		}
	}

	outputShapes := make([]shapes.Shape, len(targetFn.outputs))
	for i, out := range targetFn.outputs {
		outputShapes[i] = out.shape.Clone()
	}

	outputNodes := make([]*graphNode, len(outputShapes))
	results := make([]compute.Value, len(outputShapes))
	for i, s := range outputShapes {
		n := &graphNode{shape: s, owner: f}
		outputNodes[i] = n
		results[i] = n
	}

	f.controlFlowStep = &controlFlowStep{
		opType:       compute.OpTypeCall,
		inputs:       inputNodes,
		outputShapes: outputShapes,
		outputNodes:  outputNodes,
		callData:     &callStepData{targetFn: targetFn},
	}

	return results, nil
}

// While executes a loop while a condition is true.
func (f *Function) While(cond, body compute.Function, initialState ...compute.Value) ([]compute.Value, error) {
	if f.controlFlowStep != nil {
		return nil, errors.New("While: only one control flow operation per function is supported")
	}
	if len(initialState) == 0 {
		return nil, errors.New("While: requires at least one initial state value")
	}

	stateNodes, err := f.resolveNodes("While", initialState...)
	if err != nil {
		return nil, err
	}

	condFn, err := f.validateClosure("While", "cond", cond)
	if err != nil {
		return nil, err
	}
	bodyFn, err := f.validateClosure("While", "body", body)
	if err != nil {
		return nil, err
	}

	// Validate cond: params match state shapes, returns scalar bool.
	if len(condFn.params) != len(stateNodes) {
		return nil, errors.Errorf("While: cond must have %d parameters, got %d",
			len(stateNodes), len(condFn.params))
	}
	for i, param := range condFn.params {
		if !param.shape.Equal(stateNodes[i].shape) {
			return nil, errors.Errorf("While: cond parameter %d shape %s doesn't match state shape %s",
				i, param.shape, stateNodes[i].shape)
		}
	}
	if len(condFn.outputs) != 1 {
		return nil, errors.Errorf("While: cond must return exactly one value, got %d", len(condFn.outputs))
	}
	if condFn.outputs[0].shape.Rank() != 0 || condFn.outputs[0].shape.DType != dtypes.Bool {
		return nil, errors.Errorf("While: cond must return scalar bool, got %s", condFn.outputs[0].shape)
	}

	// Validate body: params match state shapes, returns same shapes.
	if len(bodyFn.params) != len(stateNodes) {
		return nil, errors.Errorf("While: body must have %d parameters, got %d",
			len(stateNodes), len(bodyFn.params))
	}
	for i, param := range bodyFn.params {
		if !param.shape.Equal(stateNodes[i].shape) {
			return nil, errors.Errorf("While: body parameter %d shape %s doesn't match state shape %s",
				i, param.shape, stateNodes[i].shape)
		}
	}
	if len(bodyFn.outputs) != len(stateNodes) {
		return nil, errors.Errorf("While: body must return %d values, got %d",
			len(stateNodes), len(bodyFn.outputs))
	}
	for i, out := range bodyFn.outputs {
		if !out.shape.Equal(stateNodes[i].shape) {
			return nil, errors.Errorf("While: body output %d shape %s must match state shape %s",
				i, out.shape, stateNodes[i].shape)
		}
	}

	outputShapes := make([]shapes.Shape, len(stateNodes))
	for i, n := range stateNodes {
		outputShapes[i] = n.shape.Clone()
	}

	outputNodes := make([]*graphNode, len(outputShapes))
	results := make([]compute.Value, len(outputShapes))
	for i, s := range outputShapes {
		n := &graphNode{shape: s, owner: f}
		outputNodes[i] = n
		results[i] = n
	}

	f.controlFlowStep = &controlFlowStep{
		opType:       compute.OpTypeWhile,
		inputs:       stateNodes,
		outputShapes: outputShapes,
		outputNodes:  outputNodes,
		whileData:    &whileStepData{condFn: condFn, bodyFn: bodyFn},
	}

	return results, nil
}

// If executes one of two branches based on a boolean predicate.
func (f *Function) If(pred compute.Value, trueBranch, falseBranch compute.Function) ([]compute.Value, error) {
	if f.controlFlowStep != nil {
		return nil, errors.New("If: only one control flow operation per function is supported")
	}

	predNode, err := f.resolveNode(pred)
	if err != nil {
		return nil, errors.Wrap(err, "If: pred")
	}
	if predNode.shape.Rank() != 0 || predNode.shape.DType != dtypes.Bool {
		return nil, errors.Errorf("If: pred must be scalar bool, got %s", predNode.shape)
	}

	trueFn, err := f.validateClosure("If", "trueBranch", trueBranch)
	if err != nil {
		return nil, err
	}
	falseFn, err := f.validateClosure("If", "falseBranch", falseBranch)
	if err != nil {
		return nil, err
	}

	// If branches take no parameters (they capture values from parent scope).
	if len(trueFn.params) != 0 {
		return nil, errors.Errorf("If: trueBranch must have no parameters, got %d", len(trueFn.params))
	}
	if len(falseFn.params) != 0 {
		return nil, errors.Errorf("If: falseBranch must have no parameters, got %d", len(falseFn.params))
	}

	// Both branches must return same number of outputs with matching shapes.
	if len(trueFn.outputs) != len(falseFn.outputs) {
		return nil, errors.Errorf("If: branches must return same number of outputs (true=%d, false=%d)",
			len(trueFn.outputs), len(falseFn.outputs))
	}
	for i := range trueFn.outputs {
		if !trueFn.outputs[i].shape.Equal(falseFn.outputs[i].shape) {
			return nil, errors.Errorf("If: output %d shapes must match (true=%s, false=%s)",
				i, trueFn.outputs[i].shape, falseFn.outputs[i].shape)
		}
	}

	outputShapes := make([]shapes.Shape, len(trueFn.outputs))
	for i, out := range trueFn.outputs {
		outputShapes[i] = out.shape.Clone()
	}

	outputNodes := make([]*graphNode, len(outputShapes))
	results := make([]compute.Value, len(outputShapes))
	for i, s := range outputShapes {
		n := &graphNode{shape: s, owner: f}
		outputNodes[i] = n
		results[i] = n
	}

	f.controlFlowStep = &controlFlowStep{
		opType:       compute.OpTypeIf,
		inputs:       []*graphNode{predNode},
		outputShapes: outputShapes,
		outputNodes:  outputNodes,
		ifData:       &ifStepData{trueFn: trueFn, falseFn: falseFn},
	}

	return results, nil
}

// Sort sorts one or more tensors along an axis using a comparator closure.
func (f *Function) Sort(comparator compute.Function, axis int, isStable bool, inputs ...compute.Value) ([]compute.Value, error) {
	if f.controlFlowStep != nil {
		return nil, errors.New("Sort: only one control flow operation per function is supported")
	}
	if len(inputs) == 0 {
		return nil, errors.New("Sort: requires at least one input tensor")
	}

	inputNodes, err := f.resolveNodes("Sort", inputs...)
	if err != nil {
		return nil, err
	}

	compFn, err := f.validateClosure("Sort", "comparator", comparator)
	if err != nil {
		return nil, err
	}

	// All inputs must have the same dimensions.
	firstShape := inputNodes[0].shape
	for i, n := range inputNodes[1:] {
		if firstShape.Rank() != n.shape.Rank() {
			return nil, errors.Errorf("Sort: all inputs must have same rank, input 0 rank=%d, input %d rank=%d",
				firstShape.Rank(), i+1, n.shape.Rank())
		}
		for j := range firstShape.Dimensions {
			if firstShape.Dimensions[j] != n.shape.Dimensions[j] {
				return nil, errors.Errorf("Sort: all inputs must have same dimensions")
			}
		}
	}

	// Normalize axis.
	rank := firstShape.Rank()
	if axis < 0 {
		axis = rank + axis
	}
	if axis < 0 || axis >= rank {
		return nil, errors.Errorf("Sort: axis %d out of range for rank %d", axis, rank)
	}

	// Verify comparator: 2*N scalar parameters, returns scalar bool.
	expectedParams := 2 * len(inputNodes)
	if len(compFn.params) != expectedParams {
		return nil, errors.Errorf("Sort: comparator must have %d parameters, got %d",
			expectedParams, len(compFn.params))
	}
	for i, n := range inputNodes {
		for j := range 2 {
			paramIdx := 2*i + j
			param := compFn.params[paramIdx]
			if param.shape.Rank() != 0 {
				return nil, errors.Errorf("Sort: comparator parameter %d must be scalar, got %s",
					paramIdx, param.shape)
			}
			if param.shape.DType != n.shape.DType {
				return nil, errors.Errorf("Sort: comparator parameter %d dtype %s must match input dtype %s",
					paramIdx, param.shape.DType, n.shape.DType)
			}
		}
	}
	if len(compFn.outputs) != 1 {
		return nil, errors.Errorf("Sort: comparator must return exactly one value, got %d", len(compFn.outputs))
	}
	if compFn.outputs[0].shape.Rank() != 0 || compFn.outputs[0].shape.DType != dtypes.Bool {
		return nil, errors.Errorf("Sort: comparator must return scalar bool, got %s", compFn.outputs[0].shape)
	}

	outputShapes := make([]shapes.Shape, len(inputNodes))
	for i, n := range inputNodes {
		outputShapes[i] = n.shape.Clone()
	}

	outputNodes := make([]*graphNode, len(outputShapes))
	results := make([]compute.Value, len(outputShapes))
	for i, s := range outputShapes {
		n := &graphNode{shape: s, owner: f}
		outputNodes[i] = n
		results[i] = n
	}

	f.controlFlowStep = &controlFlowStep{
		opType:       compute.OpTypeSort,
		inputs:       inputNodes,
		outputShapes: outputShapes,
		outputNodes:  outputNodes,
		sortData:     &sortStepData{comparatorFn: compFn, axis: axis, isStable: isStable},
	}

	return results, nil
}

// ===========================================================================
// Unary Operations
// ===========================================================================

func (f *Function) unaryOp(opName string, opType compute.OpType, bridgeFn func(bridge.Tensor) (bridge.Tensor, error), x compute.Value) (compute.Value, error) {
	node, err := f.resolveNode(x)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}
	outShape, err := shapeinference.UnaryOp(opType, node.shape)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}
	tensor, err := bridgeFn(node.tensor)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}
	return &graphNode{tensor: tensor, shape: outShape, owner: f}, nil
}

func (f *Function) Abs(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Abs", compute.OpTypeAbs, f.ctx().Abs, x)
}

func (f *Function) Neg(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Neg", compute.OpTypeNeg, f.ctx().Neg, x)
}

func (f *Function) Sqrt(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Sqrt", compute.OpTypeSqrt, f.ctx().Sqrt, x)
}

func (f *Function) Rsqrt(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Rsqrt", compute.OpTypeRsqrt, f.ctx().Rsqrt, x)
}

func (f *Function) Exp(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Exp", compute.OpTypeExp, f.ctx().Exp, x)
}

func (f *Function) Expm1(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Expm1", compute.OpTypeExpm1, f.ctx().Expm1, x)
}

func (f *Function) Log(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Log", compute.OpTypeLog, f.ctx().Log, x)
}

func (f *Function) Log1p(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Log1p", compute.OpTypeLog1p, f.ctx().Log1p, x)
}

func (f *Function) Sin(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Sin", compute.OpTypeSin, f.ctx().Sin, x)
}

func (f *Function) Cos(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Cos", compute.OpTypeCos, f.ctx().Cos, x)
}

func (f *Function) Tanh(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Tanh", compute.OpTypeTanh, f.ctx().Tanh, x)
}

func (f *Function) Logistic(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Logistic", compute.OpTypeLogistic, f.ctx().Sigmoid, x)
}

func (f *Function) Erf(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Erf", compute.OpTypeErf, f.ctx().Erf, x)
}

func (f *Function) Floor(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Floor", compute.OpTypeFloor, f.ctx().Floor, x)
}

func (f *Function) Ceil(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Ceil", compute.OpTypeCeil, f.ctx().Ceil, x)
}

func (f *Function) Round(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Round", compute.OpTypeRound, f.ctx().Round, x)
}

func (f *Function) Sign(x compute.Value) (compute.Value, error) {
	return f.unaryOp("Sign", compute.OpTypeSign, f.ctx().Sign, x)
}

func (f *Function) LogicalNot(x compute.Value) (compute.Value, error) {
	return f.unaryOp("LogicalNot", compute.OpTypeLogicalNot, f.ctx().LogicalNot, x)
}

func (f *Function) BitwiseNot(x compute.Value) (compute.Value, error) {
	return f.unaryOp("BitwiseNot", compute.OpTypeBitwiseNot, f.ctx().BitwiseNot, x)
}

func (f *Function) IsFinite(x compute.Value) (compute.Value, error) {
	node, err := f.resolveNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "IsFinite")
	}
	outShape := shapes.Make(dtypes.Bool, node.shape.Dimensions...)
	tensor, err := f.ctx().IsFinite(node.tensor)
	if err != nil {
		return nil, errors.Wrap(err, "IsFinite")
	}
	return &graphNode{tensor: tensor, shape: outShape, owner: f}, nil
}

func (f *Function) IsNaN(x compute.Value) (compute.Value, error) {
	node, err := f.resolveNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "IsNaN")
	}
	outShape := shapes.Make(dtypes.Bool, node.shape.Dimensions...)
	tensor, err := f.ctx().IsNaN(node.tensor)
	if err != nil {
		return nil, errors.Wrap(err, "IsNaN")
	}
	return &graphNode{tensor: tensor, shape: outShape, owner: f}, nil
}

func (f *Function) Identity(x compute.Value) (compute.Value, error) {
	node, err := f.resolveNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "Identity")
	}
	tensor, err := f.ctx().Identity(node.tensor)
	if err != nil {
		return nil, errors.Wrap(err, "Identity")
	}
	return &graphNode{tensor: tensor, shape: node.shape, owner: f}, nil
}

// ===========================================================================
// Binary Operations
// ===========================================================================

func (f *Function) binaryOp(opName string, opType compute.OpType, bridgeFn func(bridge.Tensor, bridge.Tensor) (bridge.Tensor, error), lhs, rhs compute.Value) (compute.Value, error) {
	nodes, err := f.resolveNodes(opName, lhs, rhs)
	if err != nil {
		return nil, err
	}
	outShape, err := shapeinference.BinaryOp(opType, nodes[0].shape, nodes[1].shape)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}
	tensor, err := bridgeFn(nodes[0].tensor, nodes[1].tensor)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}
	return &graphNode{tensor: tensor, shape: outShape, owner: f}, nil
}

func (f *Function) Add(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("Add", compute.OpTypeAdd, f.ctx().Add, lhs, rhs)
}

func (f *Function) Sub(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("Sub", compute.OpTypeSub, f.ctx().Sub, lhs, rhs)
}

func (f *Function) Mul(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("Mul", compute.OpTypeMul, f.ctx().Mul, lhs, rhs)
}

func (f *Function) Div(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("Div", compute.OpTypeDiv, f.ctx().Div, lhs, rhs)
}

func (f *Function) Rem(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("Rem", compute.OpTypeRem, f.ctx().Rem, lhs, rhs)
}

func (f *Function) Pow(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("Pow", compute.OpTypePow, f.ctx().Pow, lhs, rhs)
}

func (f *Function) Max(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("Max", compute.OpTypeMax, f.ctx().Max, lhs, rhs)
}

func (f *Function) Min(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("Min", compute.OpTypeMin, f.ctx().Min, lhs, rhs)
}

func (f *Function) Atan2(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("Atan2", compute.OpTypeAtan2, f.ctx().Atan2, lhs, rhs)
}

func (f *Function) LogicalAnd(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("LogicalAnd", compute.OpTypeLogicalAnd, f.ctx().LogicalAnd, lhs, rhs)
}

func (f *Function) LogicalOr(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("LogicalOr", compute.OpTypeLogicalOr, f.ctx().LogicalOr, lhs, rhs)
}

func (f *Function) LogicalXor(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("LogicalXor", compute.OpTypeLogicalXor, f.ctx().LogicalXor, lhs, rhs)
}

func (f *Function) BitwiseAnd(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("BitwiseAnd", compute.OpTypeBitwiseAnd, f.ctx().BitwiseAnd, lhs, rhs)
}

func (f *Function) BitwiseOr(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("BitwiseOr", compute.OpTypeBitwiseOr, f.ctx().BitwiseOr, lhs, rhs)
}

func (f *Function) BitwiseXor(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("BitwiseXor", compute.OpTypeBitwiseXor, f.ctx().BitwiseXor, lhs, rhs)
}

func (f *Function) ShiftLeft(lhs, rhs compute.Value) (compute.Value, error) {
	return f.binaryOp("ShiftLeft", compute.OpTypeShiftLeft, f.ctx().ShiftLeft, lhs, rhs)
}

func (f *Function) ShiftRightArithmetic(lhs, rhs compute.Value) (compute.Value, error) {
	// MPSGraph's shiftRight is arithmetic for signed types.
	return f.binaryOp("ShiftRightArithmetic", compute.OpTypeShiftRightArithmetic, f.ctx().ShiftRight, lhs, rhs)
}

func (f *Function) ShiftRightLogical(lhs, rhs compute.Value) (compute.Value, error) {
	lhsNode, err := f.resolveNode(lhs)
	if err != nil {
		return nil, errors.Wrap(err, "ShiftRightLogical")
	}
	dt := lhsNode.shape.DType

	// For unsigned types, arithmetic shift right IS logical shift right.
	unsignedDT, isSigned := signedToUnsigned(dt)
	if !isSigned {
		return f.binaryOp("ShiftRightLogical", compute.OpTypeShiftRightLogical, f.ctx().ShiftRight, lhs, rhs)
	}

	// For signed types: cast to unsigned, shift, cast back.
	castLHS, err := f.ConvertDType(lhs, unsignedDT)
	if err != nil {
		return nil, errors.Wrap(err, "ShiftRightLogical: cast to unsigned")
	}
	castRHS, err := f.ConvertDType(rhs, unsignedDT)
	if err != nil {
		return nil, errors.Wrap(err, "ShiftRightLogical: cast rhs to unsigned")
	}
	shifted, err := f.binaryOp("ShiftRightLogical", compute.OpTypeShiftRightLogical, f.ctx().ShiftRight, castLHS, castRHS)
	if err != nil {
		return nil, errors.Wrap(err, "ShiftRightLogical: shift")
	}
	return f.ConvertDType(shifted, dt)
}

// signedToUnsigned returns the unsigned equivalent of a signed integer dtype.
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

// --- Comparison ---

func (f *Function) Equal(lhs, rhs compute.Value) (compute.Value, error) {
	return f.comparisonOp("Equal", compute.OpTypeEqual, f.ctx().Equal, lhs, rhs)
}

func (f *Function) NotEqual(lhs, rhs compute.Value) (compute.Value, error) {
	return f.comparisonOp("NotEqual", compute.OpTypeNotEqual, f.ctx().NotEqual, lhs, rhs)
}

func (f *Function) GreaterThan(lhs, rhs compute.Value) (compute.Value, error) {
	return f.comparisonOp("GreaterThan", compute.OpTypeGreaterThan, f.ctx().GreaterThan, lhs, rhs)
}

func (f *Function) GreaterOrEqual(lhs, rhs compute.Value) (compute.Value, error) {
	return f.comparisonOp("GreaterOrEqual", compute.OpTypeGreaterOrEqual, f.ctx().GreaterOrEqual, lhs, rhs)
}

func (f *Function) LessThan(lhs, rhs compute.Value) (compute.Value, error) {
	return f.comparisonOp("LessThan", compute.OpTypeLessThan, f.ctx().LessThan, lhs, rhs)
}

func (f *Function) LessOrEqual(lhs, rhs compute.Value) (compute.Value, error) {
	return f.comparisonOp("LessOrEqual", compute.OpTypeLessOrEqual, f.ctx().LessOrEqual, lhs, rhs)
}

func (f *Function) comparisonOp(opName string, opType compute.OpType, bridgeFn func(bridge.Tensor, bridge.Tensor) (bridge.Tensor, error), lhs, rhs compute.Value) (compute.Value, error) {
	nodes, err := f.resolveNodes(opName, lhs, rhs)
	if err != nil {
		return nil, err
	}
	// Comparison output shape: broadcast shape of inputs, dtype Bool.
	outShape, err := shapeinference.ComparisonOp(opType, nodes[0].shape, nodes[1].shape)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}
	tensor, err := bridgeFn(nodes[0].tensor, nodes[1].tensor)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}
	return &graphNode{tensor: tensor, shape: outShape, owner: f}, nil
}

// ===========================================================================
// Shape Operations
// ===========================================================================

func (f *Function) Reshape(x compute.Value, dimensions ...int) (compute.Value, error) {
	node, err := f.resolveNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "Reshape")
	}
	outShape := shapes.Make(node.shape.DType, dimensions...)
	dims := toInt64Slice(dimensions)
	tensor, err := f.ctx().Reshape(node.tensor, dims)
	if err != nil {
		return nil, errors.Wrap(err, "Reshape")
	}
	return &graphNode{tensor: tensor, shape: outShape, owner: f}, nil
}

func (f *Function) Transpose(x compute.Value, permutation ...int) (compute.Value, error) {
	node, err := f.resolveNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "Transpose")
	}
	outShape, err := shapeinference.Transpose(node.shape, permutation)
	if err != nil {
		return nil, errors.Wrap(err, "Transpose")
	}
	tensor, err := f.ctx().Transpose(node.tensor, permutation)
	if err != nil {
		return nil, errors.Wrap(err, "Transpose")
	}
	return &graphNode{tensor: tensor, shape: outShape, owner: f}, nil
}

func (f *Function) ConvertDType(x compute.Value, dtype dtypes.DType) (compute.Value, error) {
	if err := checkDTypeSupported(dtype); err != nil {
		return nil, errors.Wrap(err, "ConvertDType")
	}
	node, err := f.resolveNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "ConvertDType")
	}
	outShape := shapes.Make(dtype, node.shape.Dimensions...)
	bridgeDType := dtypeToBridgeDType(dtype)
	tensor, err := f.ctx().Cast(node.tensor, bridgeDType)
	if err != nil {
		return nil, errors.Wrap(err, "ConvertDType")
	}
	return &graphNode{tensor: tensor, shape: outShape, owner: f}, nil
}

func (f *Function) BroadcastInDim(x compute.Value, outputShape shapes.Shape, broadcastAxes []int) (compute.Value, error) {
	node, err := f.resolveNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "BroadcastInDim")
	}

	// BroadcastInDim: x's i-th axis maps to broadcastAxes[i]-th axis of output.
	// We need to reshape x by inserting size-1 dimensions, then broadcast.
	intermediateShape := make([]int64, outputShape.Rank())
	for i := range intermediateShape {
		intermediateShape[i] = 1
	}
	for i, outAxis := range broadcastAxes {
		intermediateShape[outAxis] = int64(node.shape.Dimensions[i])
	}

	// Reshape to intermediate (with 1s in non-broadcast dims).
	reshaped, err := f.ctx().Reshape(node.tensor, intermediateShape)
	if err != nil {
		return nil, errors.Wrap(err, "BroadcastInDim: reshape")
	}

	// Broadcast to output shape.
	outDims := toInt64Slice(outputShape.Dimensions)
	tensor, err := f.ctx().BroadcastTo(reshaped, outDims)
	if err != nil {
		return nil, errors.Wrap(err, "BroadcastInDim: broadcast")
	}
	return &graphNode{tensor: tensor, shape: outputShape, owner: f}, nil
}

func (f *Function) Where(condition, onTrue, onFalse compute.Value) (compute.Value, error) {
	nodes, err := f.resolveNodes("Where", condition, onTrue, onFalse)
	if err != nil {
		return nil, err
	}
	outShape, err := shapeinference.Where(nodes[0].shape, nodes[1].shape, nodes[2].shape)
	if err != nil {
		return nil, errors.Wrap(err, "Where")
	}
	tensor, err := f.ctx().Where(nodes[0].tensor, nodes[1].tensor, nodes[2].tensor)
	if err != nil {
		return nil, errors.Wrap(err, "Where")
	}
	return &graphNode{tensor: tensor, shape: outShape, owner: f}, nil
}

func (f *Function) Clamp(min, x, max compute.Value) (compute.Value, error) {
	nodes, err := f.resolveNodes("Clamp", min, x, max)
	if err != nil {
		return nil, err
	}
	// Output shape is the same as x's shape.
	outShape := nodes[1].shape
	tensor, err := f.ctx().Clamp(nodes[0].tensor, nodes[1].tensor, nodes[2].tensor)
	if err != nil {
		return nil, errors.Wrap(err, "Clamp")
	}
	return &graphNode{tensor: tensor, shape: outShape, owner: f}, nil
}

func (f *Function) Slice(operand compute.Value, starts, limits, strides []int) (compute.Value, error) {
	node, err := f.resolveNode(operand)
	if err != nil {
		return nil, errors.Wrap(err, "Slice")
	}
	outShape, err := shapeinference.Slice(node.shape, starts, limits, strides)
	if err != nil {
		return nil, errors.Wrap(err, "Slice")
	}
	if outShape.Size() == 0 {
		return &graphNode{tensor: nil, shape: outShape, owner: f}, nil
	}
	starts64 := toInt64Slice(starts)
	ends64 := toInt64Slice(limits)
	strides64 := toInt64Slice(strides)
	tensor, err := f.ctx().Slice(node.tensor, starts64, ends64, strides64)
	if err != nil {
		return nil, errors.Wrap(err, "Slice")
	}
	return &graphNode{tensor: tensor, shape: outShape, owner: f}, nil
}

func (f *Function) Concatenate(axis int, operands ...compute.Value) (compute.Value, error) {
	nodes, err := f.resolveNodes("Concatenate", operands...)
	if err != nil {
		return nil, err
	}
	inputShapes := make([]shapes.Shape, len(nodes))
	tensors := make([]bridge.Tensor, len(nodes))
	for i, n := range nodes {
		inputShapes[i] = n.shape
		tensors[i] = n.tensor
	}
	outShape, err := shapeinference.Concatenate(inputShapes, axis)
	if err != nil {
		return nil, errors.Wrap(err, "Concatenate")
	}
	tensor, err := f.ctx().Concatenate(tensors, axis)
	if err != nil {
		return nil, errors.Wrap(err, "Concatenate")
	}
	return &graphNode{tensor: tensor, shape: outShape, owner: f}, nil
}

func (f *Function) Reverse(x compute.Value, axes ...int) (compute.Value, error) {
	node, err := f.resolveNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "Reverse")
	}
	tensor, err := f.ctx().Reverse(node.tensor, axes)
	if err != nil {
		return nil, errors.Wrap(err, "Reverse")
	}
	return &graphNode{tensor: tensor, shape: node.shape, owner: f}, nil
}

func (f *Function) Iota(shape shapes.Shape, iotaAxis int) (compute.Value, error) {
	if err := checkDTypeSupported(shape.DType); err != nil {
		return nil, errors.Wrap(err, "Iota")
	}
	dims := toInt64Slice(shape.Dimensions)
	dtype := dtypeToBridgeDType(shape.DType)
	tensor, err := f.ctx().Iota(dtype, dims, iotaAxis)
	if err != nil {
		return nil, errors.Wrap(err, "Iota")
	}
	return &graphNode{tensor: tensor, shape: shape, owner: f}, nil
}

// ===========================================================================
// Matrix Operations
// ===========================================================================

// DotGeneral is implemented in dotgeneral.go.

// ===========================================================================
// Reduction Operations
// ===========================================================================

func (f *Function) reduceOp(opName string, opType compute.OpType, reduceType int, x compute.Value, axes ...int) (compute.Value, error) {
	node, err := f.resolveNode(x)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}
	// Empty axes means reduce all dimensions.
	if len(axes) == 0 {
		axes = make([]int, node.shape.Rank())
		for i := range axes {
			axes[i] = i
		}
	}
	outShape, err := shapeinference.Reduce(node.shape, axes)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}
	tensor, err := f.ctx().Reduce(node.tensor, reduceType, axes)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}
	// MPSGraph reductions keep reduced dims as size 1 — reshape to squeeze them.
	if outShape.Rank() > 0 {
		outDims := toInt64Slice(outShape.Dimensions)
		tensor, err = f.ctx().Reshape(tensor, outDims)
		if err != nil {
			return nil, errors.Wrap(err, opName+": reshape after reduce")
		}
	} else {
		// Scalar output: reshape to rank 0.
		tensor, err = f.ctx().Reshape(tensor, nil)
		if err != nil {
			return nil, errors.Wrap(err, opName+": reshape to scalar")
		}
	}
	return &graphNode{tensor: tensor, shape: outShape, owner: f}, nil
}

func (f *Function) ReduceSum(x compute.Value, axes ...int) (compute.Value, error) {
	return f.reduceOp("ReduceSum", compute.OpTypeReduceSum, bridge.ReduceSum, x, axes...)
}

func (f *Function) ReduceMax(x compute.Value, axes ...int) (compute.Value, error) {
	return f.reduceOp("ReduceMax", compute.OpTypeReduceMax, bridge.ReduceMax, x, axes...)
}

func (f *Function) ReduceMin(x compute.Value, axes ...int) (compute.Value, error) {
	return f.reduceOp("ReduceMin", compute.OpTypeReduceMin, bridge.ReduceMin, x, axes...)
}

func (f *Function) ReduceProduct(x compute.Value, axes ...int) (compute.Value, error) {
	return f.reduceOp("ReduceProduct", compute.OpTypeReduceProduct, bridge.ReduceProduct, x, axes...)
}

// ===========================================================================
// ArgMin/ArgMax
// ===========================================================================

func (f *Function) ArgMinMax(x compute.Value, axis int, outputDType dtypes.DType, isMin bool) (compute.Value, error) {
	if err := checkDTypeSupported(outputDType); err != nil {
		return nil, errors.Wrap(err, "ArgMinMax")
	}
	node, err := f.resolveNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "ArgMinMax")
	}
	// Output shape: same as input but with the reduced axis removed.
	outDims := make([]int, 0, node.shape.Rank()-1)
	for i, d := range node.shape.Dimensions {
		if i != axis {
			outDims = append(outDims, d)
		}
	}
	outShape := shapes.Make(outputDType, outDims...)

	bridgeDType := dtypeToBridgeDType(outputDType)
	var tensor bridge.Tensor
	if isMin {
		tensor, err = f.ctx().ArgMin(node.tensor, axis, bridgeDType)
	} else {
		tensor, err = f.ctx().ArgMax(node.tensor, axis, bridgeDType)
	}
	if err != nil {
		return nil, errors.Wrap(err, "ArgMinMax")
	}
	// MPSGraph keeps reduced dim as size 1 — reshape to squeeze it.
	if outShape.Rank() > 0 {
		squeezeDims := toInt64Slice(outShape.Dimensions)
		tensor, err = f.ctx().Reshape(tensor, squeezeDims)
		if err != nil {
			return nil, errors.Wrap(err, "ArgMinMax: reshape")
		}
	} else {
		tensor, err = f.ctx().Reshape(tensor, nil)
		if err != nil {
			return nil, errors.Wrap(err, "ArgMinMax: reshape to scalar")
		}
	}
	return &graphNode{tensor: tensor, shape: outShape, owner: f}, nil
}

// ===========================================================================
// Batch Normalization
// ===========================================================================

func (f *Function) BatchNormForInference(operand, scale, offset, mean, variance compute.Value, epsilon float32, featureAxis int) (compute.Value, error) {
	nodes, err := f.resolveNodes("BatchNormForInference", operand, scale, offset, mean, variance)
	if err != nil {
		return nil, err
	}
	tensor, err := f.ctx().BatchNormInference(
		nodes[0].tensor, nodes[3].tensor, nodes[4].tensor,
		nodes[1].tensor, nodes[2].tensor,
		float32(epsilon), featureAxis)
	if err != nil {
		return nil, errors.Wrap(err, "BatchNormForInference")
	}
	return &graphNode{tensor: tensor, shape: nodes[0].shape, owner: f}, nil
}

func (f *Function) BatchNormForTraining(
	operand, scale, offset compute.Value,
	epsilon float32,
	featureAxis int,
) (normalized compute.Value, batchMean compute.Value, batchVariance compute.Value, err error) {
	opNode, err := f.resolveNode(operand)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormForTraining: operand")
	}
	scaleNode, err := f.resolveNode(scale)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormForTraining: scale")
	}
	offsetNode, err := f.resolveNode(offset)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormForTraining: offset")
	}

	rank := opNode.shape.Rank()
	dt := opNode.shape.DType

	// Batch axes = all axes except featureAxis.
	var batchAxes []int
	for i := range rank {
		if i != featureAxis {
			batchAxes = append(batchAxes, i)
		}
	}

	// Compute batch mean: reduce over batch axes, keep feature axis.
	meanTensor, err := f.ctx().Reduce(opNode.tensor, bridge.ReduceSum, batchAxes)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormForTraining: reduce for mean")
	}
	batchSize := int64(1)
	for _, ax := range batchAxes {
		batchSize *= int64(opNode.shape.Dimensions[ax])
	}
	countTensor, err := f.makeScalarConst(float64(batchSize), dt)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormForTraining: count constant")
	}
	meanTensor, err = f.ctx().Div(meanTensor, countTensor)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormForTraining: mean div")
	}

	// Center: operand - mean (broadcast automatically via MPSGraph).
	diff, err := f.ctx().Sub(opNode.tensor, meanTensor)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormForTraining: subtract mean")
	}

	// Variance: mean((operand - mean)^2) over batch axes.
	diffSq, err := f.ctx().Mul(diff, diff)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormForTraining: diff squared")
	}
	varTensor, err := f.ctx().Reduce(diffSq, bridge.ReduceSum, batchAxes)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormForTraining: reduce for variance")
	}
	varTensor, err = f.ctx().Div(varTensor, countTensor)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormForTraining: variance div")
	}

	// Normalize: (operand - mean) / sqrt(variance + epsilon)
	epsTensor, err := f.makeScalarConst(float64(epsilon), dt)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormForTraining: epsilon constant")
	}
	varPlusEps, err := f.ctx().Add(varTensor, epsTensor)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormForTraining: var + eps")
	}
	invStd, err := f.ctx().Rsqrt(varPlusEps)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormForTraining: rsqrt")
	}
	normalizedTensor, err := f.ctx().Mul(diff, invStd)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormForTraining: normalize")
	}

	// Apply scale and offset: normalized * scale + offset.
	// Scale and offset have shape [featureDim] — need to broadcast to operand shape.
	normalizedTensor, err = f.ctx().Mul(normalizedTensor, scaleNode.tensor)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormForTraining: apply scale")
	}
	normalizedTensor, err = f.ctx().Add(normalizedTensor, offsetNode.tensor)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormForTraining: apply offset")
	}

	// Squeeze mean and variance to 1D [featureDim] shape.
	featureDim := opNode.shape.Dimensions[featureAxis]
	featureShape := shapes.Make(dt, featureDim)
	meanTensor, err = f.ctx().Reshape(meanTensor, []int64{int64(featureDim)})
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormForTraining: reshape mean")
	}
	varTensor, err = f.ctx().Reshape(varTensor, []int64{int64(featureDim)})
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormForTraining: reshape variance")
	}

	normalized = &graphNode{tensor: normalizedTensor, shape: opNode.shape, owner: f}
	batchMean = &graphNode{tensor: meanTensor, shape: featureShape, owner: f}
	batchVariance = &graphNode{tensor: varTensor, shape: featureShape, owner: f}
	return normalized, batchMean, batchVariance, nil
}

func (f *Function) BatchNormGradient(
	operand, scale, mean, variance, gradOutput compute.Value,
	epsilon float32,
	featureAxis int,
) (gradOperand compute.Value, gradScale compute.Value, gradOffset compute.Value, err error) {
	opNode, err := f.resolveNode(operand)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormGradient: operand")
	}
	scaleNode, err := f.resolveNode(scale)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormGradient: scale")
	}
	meanNode, err := f.resolveNode(mean)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormGradient: mean")
	}
	varNode, err := f.resolveNode(variance)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormGradient: variance")
	}
	gradOutNode, err := f.resolveNode(gradOutput)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormGradient: gradOutput")
	}

	rank := opNode.shape.Rank()
	dt := opNode.shape.DType

	// Batch axes = all axes except featureAxis.
	var batchAxes []int
	for i := range rank {
		if i != featureAxis {
			batchAxes = append(batchAxes, i)
		}
	}

	batchSize := int64(1)
	for _, ax := range batchAxes {
		batchSize *= int64(opNode.shape.Dimensions[ax])
	}

	// Broadcast mean and variance from [featureDim] to operand shape for element-wise ops.
	// MPSGraph will broadcast automatically since mean/variance have shape [featureDim]
	// aligned with the featureAxis.

	// invStd = 1 / sqrt(variance + epsilon)
	epsTensor, err := f.makeScalarConst(float64(epsilon), dt)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormGradient: epsilon constant")
	}
	varPlusEps, err := f.ctx().Add(varNode.tensor, epsTensor)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormGradient: var + eps")
	}
	invStd, err := f.ctx().Rsqrt(varPlusEps)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormGradient: rsqrt")
	}

	// xhat = (operand - mean) * invStd (normalized input without scale/offset)
	centered, err := f.ctx().Sub(opNode.tensor, meanNode.tensor)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormGradient: center")
	}
	xhat, err := f.ctx().Mul(centered, invStd)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormGradient: xhat")
	}

	// gradOffset = sum(gradOutput, batchAxes) — gradient w.r.t. offset/bias
	gradOffsetTensor, err := f.ctx().Reduce(gradOutNode.tensor, bridge.ReduceSum, batchAxes)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormGradient: gradOffset reduce")
	}

	// gradScale = sum(gradOutput * xhat, batchAxes) — gradient w.r.t. scale
	gradOutTimesXhat, err := f.ctx().Mul(gradOutNode.tensor, xhat)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormGradient: gradOutput * xhat")
	}
	gradScaleTensor, err := f.ctx().Reduce(gradOutTimesXhat, bridge.ReduceSum, batchAxes)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormGradient: gradScale reduce")
	}

	// gradOperand = (1/N) * scale * invStd * (N * gradOutput - sum(gradOutput) - xhat * sum(gradOutput * xhat))
	// This is the standard batch norm gradient formula.
	nTensor, err := f.makeScalarConst(float64(batchSize), dt)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormGradient: N constant")
	}
	invNTensor, err := f.makeScalarConst(1.0/float64(batchSize), dt)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormGradient: 1/N constant")
	}

	// term1 = N * gradOutput
	term1, err := f.ctx().Mul(nTensor, gradOutNode.tensor)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormGradient: N * gradOutput")
	}

	// term2 = sum(gradOutput, batchAxes) — this is gradOffset, broadcast back
	// (MPSGraph broadcasts automatically from reduced shape)
	term1, err = f.ctx().Sub(term1, gradOffsetTensor)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormGradient: term1 - gradOffset")
	}

	// term3 = xhat * sum(gradOutput * xhat, batchAxes) = xhat * gradScale
	term3, err := f.ctx().Mul(xhat, gradScaleTensor)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormGradient: xhat * gradScale")
	}
	term1, err = f.ctx().Sub(term1, term3)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormGradient: term1 - term3")
	}

	// gradOperand = (1/N) * scale * invStd * (N*gradOutput - gradOffset - xhat*gradScale)
	gradOpTensor, err := f.ctx().Mul(invNTensor, term1)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormGradient: 1/N * terms")
	}
	gradOpTensor, err = f.ctx().Mul(scaleNode.tensor, gradOpTensor)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormGradient: scale * terms")
	}
	gradOpTensor, err = f.ctx().Mul(invStd, gradOpTensor)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormGradient: invStd * terms")
	}

	// Squeeze gradScale and gradOffset to 1D [featureDim].
	featureDim := opNode.shape.Dimensions[featureAxis]
	featureShape := shapes.Make(dt, featureDim)
	gradScaleTensor, err = f.ctx().Reshape(gradScaleTensor, []int64{int64(featureDim)})
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormGradient: reshape gradScale")
	}
	gradOffsetTensor, err = f.ctx().Reshape(gradOffsetTensor, []int64{int64(featureDim)})
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "BatchNormGradient: reshape gradOffset")
	}

	gradOperand = &graphNode{tensor: gradOpTensor, shape: opNode.shape, owner: f}
	gradScale = &graphNode{tensor: gradScaleTensor, shape: featureShape, owner: f}
	gradOffset = &graphNode{tensor: gradOffsetTensor, shape: featureShape, owner: f}
	return gradOperand, gradScale, gradOffset, nil
}

// ===========================================================================
// Pad
// ===========================================================================

func (f *Function) Pad(operand, fillValue compute.Value, axesConfig ...compute.PadAxis) (compute.Value, error) {
	opNode, err := f.resolveNode(operand)
	if err != nil {
		return nil, errors.Wrap(err, "Pad")
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
	if outShape.Size() == 0 {
		return &graphNode{tensor: nil, shape: outShape, owner: f}, nil
	}
	if opNode.shape.Size() == 0 {
		tensor, err := f.ctx().BroadcastTo(fillNode.tensor, toInt64Slice(outShape.Dimensions))
		if err != nil {
			return nil, errors.Wrap(err, "Pad: broadcast fillValue for empty operand")
		}
		return &graphNode{tensor: tensor, shape: outShape, owner: f}, nil
	}

	rank := len(axesConfig)
	tensor := opNode.tensor
	currentShape := toInt64Slice(opNode.shape.Dimensions)

	// Step 1: Handle interior padding per axis by reshaping [N] -> [N, 1],
	// padding the new axis with ac.Interior, flattening to [N*(ac.Interior+1)],
	// and slicing to [(N-1)*ac.Interior + N].
	for i, ac := range axesConfig {
		if ac.Interior > 0 {
			n := currentShape[i]
			if n > 1 {
				dilation := int64(ac.Interior) + 1

				// Reshape to split axis i into [n, 1].
				reshapeDims := make([]int64, rank+1)
				copy(reshapeDims[:i], currentShape[:i])
				reshapeDims[i] = n
				reshapeDims[i+1] = 1
				copy(reshapeDims[i+2:], currentShape[i+1:])

				tensor, err = f.ctx().Reshape(tensor, reshapeDims)
				if err != nil {
					return nil, errors.Wrapf(err, "Pad: interior reshape axis %d", i)
				}

				// Pad axis i+1 with ac.Interior after.
				padBefore := make([]int64, rank+1)
				padAfter := make([]int64, rank+1)
				padAfter[i+1] = int64(ac.Interior)
				tensor, err = f.ctx().Pad(tensor, fillNode.tensor, padBefore, padAfter)
				if err != nil {
					return nil, errors.Wrapf(err, "Pad: interior pad axis %d", i)
				}

				// Reshape to flatten axis i back to [n * dilation].
				flatDims := make([]int64, rank)
				copy(flatDims, currentShape)
				flatDims[i] = n * dilation
				tensor, err = f.ctx().Reshape(tensor, flatDims)
				if err != nil {
					return nil, errors.Wrapf(err, "Pad: interior flatten axis %d", i)
				}

				// Slice off trailing ac.Interior elements on axis i.
				dilatedSize := n + (n-1)*int64(ac.Interior)
				starts := make([]int64, rank)
				ends := make([]int64, rank)
				strides := make([]int64, rank)
				for ax := range rank {
					ends[ax] = flatDims[ax]
					strides[ax] = 1
				}
				ends[i] = dilatedSize
				tensor, err = f.ctx().Slice(tensor, starts, ends, strides)
				if err != nil {
					return nil, errors.Wrapf(err, "Pad: interior slice axis %d", i)
				}

				currentShape[i] = dilatedSize
			}
		}
	}

	// Step 2: Handle positive Start / End padding via Pad first so cropping
	// never produces a zero-sized intermediate tensor when all input elements
	// on an axis are cropped out.
	hasPad := false
	padBefore := make([]int64, rank)
	padAfter := make([]int64, rank)
	for i, ac := range axesConfig {
		if ac.Start > 0 {
			padBefore[i] = int64(ac.Start)
			hasPad = true
		}
		if ac.End > 0 {
			padAfter[i] = int64(ac.End)
			hasPad = true
		}
	}
	if hasPad {
		tensor, err = f.ctx().Pad(tensor, fillNode.tensor, padBefore, padAfter)
		if err != nil {
			return nil, errors.Wrap(err, "Pad")
		}
		for i := range rank {
			currentShape[i] += padBefore[i] + padAfter[i]
		}
	}

	// Step 3: Handle negative Start / End padding (cropping) via Slice.
	hasCrop := false
	cropStarts := make([]int64, rank)
	cropEnds := make([]int64, rank)
	cropStrides := make([]int64, rank)
	for i, ac := range axesConfig {
		cropStrides[i] = 1
		cropEnds[i] = currentShape[i]
		if ac.Start < 0 {
			cropStarts[i] = int64(-ac.Start)
			hasCrop = true
		}
		if ac.End < 0 {
			cropEnds[i] = currentShape[i] + int64(ac.End)
			hasCrop = true
		}
	}
	if hasCrop {
		tensor, err = f.ctx().Slice(tensor, cropStarts, cropEnds, cropStrides)
		if err != nil {
			return nil, errors.Wrap(err, "Pad: negative padding slice")
		}
	}

	return &graphNode{tensor: tensor, shape: outShape, owner: f}, nil
}

// ===========================================================================
// DynamicSlice
// ===========================================================================

func (f *Function) normalizeDynamicStartIndices(opName string, rank int, startIndicesValues []compute.Value) ([]bridge.Tensor, error) {
	startNodes := make([]*graphNode, len(startIndicesValues))
	for i, v := range startIndicesValues {
		n, err := f.resolveNode(v)
		if err != nil {
			return nil, errors.Wrapf(err, "%s: startIndex[%d]", opName, i)
		}
		startNodes[i] = n
	}
	if len(startNodes) == 1 && rank > 1 && startNodes[0].shape.Rank() == 1 && startNodes[0].shape.Dimensions[0] == rank {
		vec := startNodes[0].tensor
		tensors := make([]bridge.Tensor, rank)
		for i := range rank {
			s, err := f.ctx().Slice(vec, []int64{int64(i)}, []int64{int64(i + 1)}, []int64{1})
			if err != nil {
				return nil, errors.Wrapf(err, "%s: slice startIndices[%d]", opName, i)
			}
			tensors[i] = s
		}
		return tensors, nil
	}
	tensors := make([]bridge.Tensor, len(startNodes))
	for i, n := range startNodes {
		tensors[i] = n.tensor
	}
	return tensors, nil
}

func (f *Function) DynamicSlice(operand compute.Value, startIndicesValues []compute.Value, sliceSizes []int) (compute.Value, error) {
	opNode, err := f.resolveNode(operand)
	if err != nil {
		return nil, errors.Wrap(err, "DynamicSlice: operand")
	}

	startIndicesTensors, err := f.normalizeDynamicStartIndices("DynamicSlice", opNode.shape.Rank(), startIndicesValues)
	if err != nil {
		return nil, err
	}
	outShape, err := shapeinference.DynamicSlice(opNode.shape, sliceSizes)
	if err != nil {
		return nil, errors.Wrap(err, "DynamicSlice")
	}

	sliceSizes64 := toInt64Slice(sliceSizes)
	tensor, err := f.ctx().DynamicSlice(opNode.tensor, startIndicesTensors, sliceSizes64)
	if err != nil {
		return nil, errors.Wrap(err, "DynamicSlice")
	}
	return &graphNode{tensor: tensor, shape: outShape, owner: f}, nil
}

func (f *Function) DynamicUpdateSlice(operand, update compute.Value, startIndicesValues []compute.Value) (compute.Value, error) {
	opNode, err := f.resolveNode(operand)
	if err != nil {
		return nil, errors.Wrap(err, "DynamicUpdateSlice: operand")
	}
	updNode, err := f.resolveNode(update)
	if err != nil {
		return nil, errors.Wrap(err, "DynamicUpdateSlice: update")
	}

	startIndicesTensors, err := f.normalizeDynamicStartIndices("DynamicUpdateSlice", opNode.shape.Rank(), startIndicesValues)
	if err != nil {
		return nil, err
	}
	outShape, err := shapeinference.DynamicUpdateSlice(opNode.shape, updNode.shape)
	if err != nil {
		return nil, errors.Wrap(err, "DynamicUpdateSlice")
	}

	tensor, err := f.ctx().DynamicUpdateSlice(opNode.tensor, updNode.tensor, startIndicesTensors)
	if err != nil {
		return nil, errors.Wrap(err, "DynamicUpdateSlice")
	}
	return &graphNode{tensor: tensor, shape: outShape, owner: f}, nil
}

// ===========================================================================
// RNG
// ===========================================================================

func (f *Function) RNGBitGenerator(state compute.Value, shape shapes.Shape) (newState, values compute.Value, err error) {
	stateNode, err := f.resolveNode(state)
	if err != nil {
		return nil, nil, errors.Wrap(err, "RNGBitGenerator: state")
	}

	// Validate state shape: expects [3]uint64 per GoMLX convention.
	expectedStateShape := compute.RNGStateShape
	if !stateNode.shape.Equal(expectedStateShape) {
		return nil, nil, errors.Errorf("RNGBitGenerator: expected state shape %s, got %s",
			expectedStateShape, stateNode.shape)
	}
	if err := checkDTypeSupported(shape.DType); err != nil {
		return nil, nil, errors.Wrap(err, "RNGBitGenerator")
	}

	dims := toInt64Slice(shape.Dimensions)

	// GoMLX's RandomUniform calls RNGBitGenerator with Uint32 dtype to get random bits,
	// then converts to float: ConvertDType(bits, Float32) * (1/2^32).
	// MPSGraph only supports float types for random generation (float16, bfloat16, float32).
	// Strategy: generate Float32 uniform [0, 1), scale to [0, 2^32) so the subsequent
	// ConvertDType(Uint32→Float32) is a no-op cast and MulScalar(1/2^32) produces [0, 1).
	var valuesTensor bridge.Tensor
	if shape.DType.IsFloat() {
		// Direct generation for float types.
		bridgeDType := dtypeToBridgeDType(shape.DType)
		valuesTensor, err = f.ctx().RandomUniform(bridgeDType, dims)
		if err != nil {
			return nil, nil, errors.Wrap(err, "RNGBitGenerator")
		}
	} else if shape.DType == dtypes.Uint64 || shape.DType == dtypes.Int64 {
		// For 64-bit integer types, generate two 32-bit uniform random chunks and combine (hi << 32) | lo.
		dims2 := append([]int64{2}, dims...)
		u, err := f.ctx().RandomUniform(dtypeToBridgeDType(dtypes.Float32), dims2)
		if err != nil {
			return nil, nil, errors.Wrap(err, "RNGBitGenerator: RandomUniform(Float32)")
		}
		scaleTensor, err := f.makeScalarConst(4294967296.0, dtypes.Float32) // 2^32
		if err != nil {
			return nil, nil, errors.Wrap(err, "RNGBitGenerator: scale constant")
		}
		u, err = f.ctx().Mul(u, scaleTensor)
		if err != nil {
			return nil, nil, errors.Wrap(err, "RNGBitGenerator: scale")
		}
		u32, err := f.ctx().Cast(u, dtypeToBridgeDType(dtypes.Uint32))
		if err != nil {
			return nil, nil, errors.Wrap(err, "RNGBitGenerator: cast Uint32")
		}
		u64, err := f.ctx().Cast(u32, dtypeToBridgeDType(dtypes.Uint64))
		if err != nil {
			return nil, nil, errors.Wrap(err, "RNGBitGenerator: cast Uint64")
		}
		starts0 := make([]int64, len(dims2))
		ends0 := make([]int64, len(dims2))
		starts1 := make([]int64, len(dims2))
		ends1 := make([]int64, len(dims2))
		strides := make([]int64, len(dims2))
		for i, d := range dims2 {
			strides[i] = 1
			ends0[i] = d
			ends1[i] = d
		}
		ends0[0] = 1
		starts1[0] = 1
		ends1[0] = 2
		hi, err := f.ctx().Slice(u64, starts0, ends0, strides)
		if err != nil {
			return nil, nil, errors.Wrap(err, "RNGBitGenerator: slice hi")
		}
		lo, err := f.ctx().Slice(u64, starts1, ends1, strides)
		if err != nil {
			return nil, nil, errors.Wrap(err, "RNGBitGenerator: slice lo")
		}
		if len(dims) > 0 {
			hi, err = f.ctx().Reshape(hi, dims)
			if err != nil {
				return nil, nil, errors.Wrap(err, "RNGBitGenerator: reshape hi")
			}
			lo, err = f.ctx().Reshape(lo, dims)
			if err != nil {
				return nil, nil, errors.Wrap(err, "RNGBitGenerator: reshape lo")
			}
		} else {
			hi, err = f.ctx().Reshape(hi, []int64{1})
			if err != nil {
				return nil, nil, errors.Wrap(err, "RNGBitGenerator: reshape hi")
			}
			lo, err = f.ctx().Reshape(lo, []int64{1})
			if err != nil {
				return nil, nil, errors.Wrap(err, "RNGBitGenerator: reshape lo")
			}
		}
		shift32, err := f.makeScalarConst(32, dtypes.Uint64)
		if err != nil {
			return nil, nil, errors.Wrap(err, "RNGBitGenerator: shift32")
		}
		hiShifted, err := f.ctx().ShiftLeft(hi, shift32)
		if err != nil {
			return nil, nil, errors.Wrap(err, "RNGBitGenerator: ShiftLeft")
		}
		valuesTensor, err = f.ctx().BitwiseOr(hiShifted, lo)
		if err != nil {
			return nil, nil, errors.Wrap(err, "RNGBitGenerator: BitwiseOr")
		}
		if shape.DType != dtypes.Uint64 {
			valuesTensor, err = f.ctx().Cast(valuesTensor, dtypeToBridgeDType(shape.DType))
			if err != nil {
				return nil, nil, errors.Wrap(err, "RNGBitGenerator: final cast")
			}
		}
	} else {
		// Integer type (typically Uint32): generate Float32 uniform, scale to [0, 2^32), and cast.
		valuesTensor, err = f.ctx().RandomUniform(dtypeToBridgeDType(dtypes.Float32), dims)
		if err != nil {
			return nil, nil, errors.Wrap(err, "RNGBitGenerator: RandomUniform(Float32)")
		}
		scaleTensor, err := f.makeScalarConst(4294967296.0, dtypes.Float32) // 2^32
		if err != nil {
			return nil, nil, errors.Wrap(err, "RNGBitGenerator: scale constant")
		}
		valuesTensor, err = f.ctx().Mul(valuesTensor, scaleTensor)
		if err != nil {
			return nil, nil, errors.Wrap(err, "RNGBitGenerator: scale")
		}
		valuesTensor, err = f.ctx().Cast(valuesTensor, dtypeToBridgeDType(dtypes.Uint32))
		if err != nil {
			return nil, nil, errors.Wrap(err, "RNGBitGenerator: cast Uint32")
		}
		if shape.DType != dtypes.Uint32 {
			valuesTensor, err = f.ctx().Cast(valuesTensor, dtypeToBridgeDType(shape.DType))
			if err != nil {
				return nil, nil, errors.Wrap(err, "RNGBitGenerator: final cast")
			}
		}
	}

	// Advance the RNG state by adding 1 to the state tensor, so subsequent calls
	// produce different random values.
	one, err := f.makeScalarConst(1.0, stateNode.shape.DType)
	if err != nil {
		return nil, nil, errors.Wrap(err, "RNGBitGenerator: creating state increment")
	}
	// Broadcast to state shape.
	stateDims := toInt64Slice(stateNode.shape.Dimensions)
	one, err = f.ctx().BroadcastTo(one, stateDims)
	if err != nil {
		return nil, nil, errors.Wrap(err, "RNGBitGenerator: broadcasting state increment")
	}
	newStateTensor, err := f.ctx().Add(stateNode.tensor, one)
	if err != nil {
		return nil, nil, errors.Wrap(err, "RNGBitGenerator: advancing state")
	}
	newStateNode := &graphNode{tensor: newStateTensor, shape: stateNode.shape, owner: f}
	valuesNode := &graphNode{tensor: valuesTensor, shape: shape, owner: f}
	return newStateNode, valuesNode, nil
}

// ===========================================================================
// Convolution
// ===========================================================================

func (f *Function) ConvGeneral(
	input, kernel compute.Value,
	axes compute.ConvolveAxesConfig,
	strides []int, paddings [][2]int,
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

	// Default nil strides to 1.
	if strides == nil {
		strides = make([]int, numSpatialDims)
		for i := range strides {
			strides[i] = 1
		}
	}

	// Default nil dilations to 1.
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

	// Default group counts.
	if channelGroupCount < 1 {
		channelGroupCount = 1
	}
	if batchGroupCount < 1 {
		batchGroupCount = 1
	}

	outputShape, err := shapeinference.ConvGeneral(
		inputNode.shape, kernelNode.shape, axes, strides, paddings,
		inputDilations, kernelDilations, channelGroupCount, batchGroupCount)
	if err != nil {
		return nil, errors.Wrap(err, "ConvGeneral")
	}

	if !inputNode.shape.DType.IsFloat() {
		return nil, errors.Wrapf(compute.ErrNotImplemented, "ConvGeneral: only float dtypes supported in %q backend, got %s", BackendName, inputNode.shape.DType)
	}
	if batchGroupCount > 1 {
		return nil, errors.Wrapf(compute.ErrNotImplemented, "ConvGeneral: batchGroupCount > 1 not supported in %q backend, got %d", BackendName, batchGroupCount)
	}
	if numSpatialDims != 2 {
		return nil, errors.Wrapf(compute.ErrNotImplemented, "ConvGeneral: only 2D convolution supported in %q backend, got %d spatial dims", BackendName, numSpatialDims)
	}

	// Check if input dilation is needed (values > 1).
	hasInputDilation := false
	for _, d := range inputDilations {
		if d > 1 {
			hasInputDilation = true
			break
		}
	}

	// Transpose input and kernel to NCHW / OIHW layout expected by MPSGraph.
	inputTensor, err := f.transposeToNCHW(inputNode, axes.InputBatch, axes.InputChannels, axes.InputSpatial)
	if err != nil {
		return nil, errors.Wrap(err, "ConvGeneral: transpose input")
	}

	// Handle input dilation by inserting zeros between input elements.
	// Input dilation of D for an axis means: between each pair of values, insert (D-1) zeros.
	// This expands a dimension of size N to (N-1)*D + 1.
	if hasInputDilation {
		inputTensor, err = f.dilateInput(inputTensor, inputNode.shape, axes.InputBatch, axes.InputChannels, axes.InputSpatial, inputDilations)
		if err != nil {
			return nil, errors.Wrap(err, "ConvGeneral: input dilation")
		}
	}

	kernelTensor, err := f.transposeToOIHW(kernelNode, axes.KernelOutputChannels, axes.KernelInputChannels, axes.KernelSpatial)
	if err != nil {
		return nil, errors.Wrap(err, "ConvGeneral: transpose kernel")
	}

	// Prepare strides, dilations, padding.
	strideArr := make([]int64, numSpatialDims)
	dilationArr := make([]int64, numSpatialDims)
	padBeforeArr := make([]int64, numSpatialDims)
	padAfterArr := make([]int64, numSpatialDims)
	for i := range numSpatialDims {
		strideArr[i] = int64(strides[i])
		dilationArr[i] = int64(kernelDilations[i])
		if paddings != nil && i < len(paddings) {
			padBeforeArr[i] = int64(paddings[i][0])
			padAfterArr[i] = int64(paddings[i][1])
		}
	}

	groups := channelGroupCount * batchGroupCount
	result, err := f.ctx().ConvGeneral(inputTensor, kernelTensor, numSpatialDims,
		strideArr, dilationArr, padBeforeArr, padAfterArr, groups)
	if err != nil {
		return nil, errors.Wrap(err, "ConvGeneral")
	}

	// Transpose output from NCHW back to the requested layout.
	result, err = f.transposeFromNCHW(result, outputShape, axes.OutputBatch, axes.OutputChannels, axes.OutputSpatial)
	if err != nil {
		return nil, errors.Wrap(err, "ConvGeneral: transpose output")
	}

	return &graphNode{tensor: result, shape: outputShape, owner: f}, nil
}

// dilateInput inserts zeros between input elements for input dilation.
// Input is already in NCHW layout. dilations are per spatial axis.
func (f *Function) dilateInput(tensor bridge.Tensor, origShape shapes.Shape, batchAxis, channelAxis int, spatialAxes []int, dilations []int) (bridge.Tensor, error) {
	// After transpose to NCHW, spatial dims are at indices 2 and 3.
	// Dilation of D on an axis with size N -> new size = (N-1)*D + 1.
	// Approach per axis: reshape to split axis into [N, 1], pad to [N, D],
	// reshape to flatten [N*D], then slice to [(N-1)*D+1].
	origDims := origShape.Dimensions
	spatialSizes := make([]int64, len(spatialAxes))
	for i, ax := range spatialAxes {
		spatialSizes[i] = int64(origDims[ax])
	}

	batchSize := int64(origDims[batchAxis])
	channelSize := int64(origDims[channelAxis])
	dilatedH := (spatialSizes[0]-1)*int64(dilations[0]) + 1

	result := tensor
	var err error

	// Dilate height (axis 2 in NCHW).
	if dilations[0] > 1 {
		result, err = f.dilateAxis(result, 2, spatialSizes[0], int64(dilations[0]),
			[]int64{batchSize, channelSize, spatialSizes[0], spatialSizes[1]}, origShape.DType)
		if err != nil {
			return nil, errors.Wrap(err, "dilateInput: dilate H")
		}
		spatialSizes[0] = dilatedH
	}

	// Dilate width (axis 3 in NCHW).
	if dilations[1] > 1 {
		result, err = f.dilateAxis(result, 3, spatialSizes[1], int64(dilations[1]),
			[]int64{batchSize, channelSize, dilatedH, spatialSizes[1]}, origShape.DType)
		if err != nil {
			return nil, errors.Wrap(err, "dilateInput: dilate W")
		}
	}

	return result, nil
}

// dilateAxis dilates a single axis by inserting (dilation-1) zeros between elements.
// Approach: reshape to insert a new dim, pad that dim, reshape to flatten, then slice.
func (f *Function) dilateAxis(tensor bridge.Tensor, axis int, axisSize, dilation int64, currentShape []int64, dtype dtypes.DType) (bridge.Tensor, error) {
	rank := len(currentShape)

	// Step 1: Reshape to split the target axis into [axisSize, 1].
	reshapeDims := make([]int64, rank+1)
	for i := 0; i < axis; i++ {
		reshapeDims[i] = currentShape[i]
	}
	reshapeDims[axis] = axisSize
	reshapeDims[axis+1] = 1
	for i := axis + 1; i < rank; i++ {
		reshapeDims[i+1] = currentShape[i]
	}
	result, err := f.ctx().Reshape(tensor, reshapeDims)
	if err != nil {
		return nil, err
	}

	// Step 2: Pad the new axis (axis+1) with (dilation-1) zeros after.
	padBefore := make([]int64, rank+1)
	padAfter := make([]int64, rank+1)
	padAfter[axis+1] = dilation - 1

	result, err = f.ctx().Pad(result, nil, padBefore, padAfter)
	if err != nil {
		return nil, err
	}

	// Step 3: Reshape to flatten the axis back: [axisSize * dilation].
	flatDims := make([]int64, rank)
	for i := 0; i < axis; i++ {
		flatDims[i] = currentShape[i]
	}
	flatDims[axis] = axisSize * dilation
	for i := axis + 1; i < rank; i++ {
		flatDims[i] = currentShape[i]
	}
	result, err = f.ctx().Reshape(result, flatDims)
	if err != nil {
		return nil, err
	}

	// Step 4: Slice to remove trailing (dilation-1) zeros.
	// New size = (axisSize-1)*dilation + 1.
	dilatedSize := (axisSize-1)*dilation + 1
	starts := make([]int64, rank)
	ends := make([]int64, rank)
	strides := make([]int64, rank)
	for i := range rank {
		ends[i] = flatDims[i]
		strides[i] = 1
	}
	ends[axis] = dilatedSize
	result, err = f.ctx().Slice(result, starts, ends, strides)
	if err != nil {
		return nil, err
	}

	return result, nil
}

// transposeToNCHW transposes a tensor from arbitrary axis layout to NCHW.
func (f *Function) transposeToNCHW(node *graphNode, batchAxis, channelAxis int, spatialAxes []int) (bridge.Tensor, error) {
	rank := node.shape.Rank()
	perm := make([]int, rank)
	perm[0] = batchAxis
	perm[1] = channelAxis
	for i, a := range spatialAxes {
		perm[2+i] = a
	}
	if isIdentityPerm(perm) {
		return node.tensor, nil
	}
	return f.ctx().Transpose(node.tensor, perm)
}

// transposeToOIHW transposes a kernel from arbitrary layout to OIHW.
func (f *Function) transposeToOIHW(node *graphNode, outChannelAxis, inChannelAxis int, spatialAxes []int) (bridge.Tensor, error) {
	rank := node.shape.Rank()
	perm := make([]int, rank)
	perm[0] = outChannelAxis
	perm[1] = inChannelAxis
	for i, a := range spatialAxes {
		perm[2+i] = a
	}
	if isIdentityPerm(perm) {
		return node.tensor, nil
	}
	return f.ctx().Transpose(node.tensor, perm)
}

// transposeFromNCHW transposes from NCHW layout back to the target layout.
func (f *Function) transposeFromNCHW(tensor bridge.Tensor, targetShape shapes.Shape, batchAxis, channelAxis int, spatialAxes []int) (bridge.Tensor, error) {
	rank := targetShape.Rank()
	// Build the inverse permutation: from NCHW position to target position.
	fwdPerm := make([]int, rank)
	fwdPerm[0] = batchAxis
	fwdPerm[1] = channelAxis
	for i, a := range spatialAxes {
		fwdPerm[2+i] = a
	}
	// Compute inverse.
	invPerm := make([]int, rank)
	for i, v := range fwdPerm {
		invPerm[v] = i
	}
	if isIdentityPerm(invPerm) {
		return tensor, nil
	}
	return f.ctx().Transpose(tensor, invPerm)
}

// ===========================================================================
// ReduceWindow (Pooling)
// ===========================================================================

// poolAxesInfo holds axis mapping for pool operations.
// MPSGraph pool2d expects NCHW layout; this detects the actual layout
// and provides permutations for transposing to/from NCHW.
type poolAxesInfo struct {
	spatialAxes    [2]int   // Indices of spatial axes in original layout.
	nonSpatialAxes [2]int   // Indices of batch/channel axes in original layout.
	toNCHW         []int    // Permutation from original layout to NCHW.
	fromNCHW       []int    // Permutation from NCHW back to original layout.
	needsTranspose bool     // Whether transposition is needed.
	spatialWindow  [2]int64 // Window sizes for spatial dims.
	spatialStrides [2]int64 // Strides for spatial dims.
	padBefore      [2]int64 // Padding before for spatial dims.
	padAfter       [2]int64 // Padding after for spatial dims.
}

// detectPoolAxes detects spatial axes from windowDimensions and builds
// transposition info. Spatial axes are those with window > 1 or stride > 1
// or non-zero padding.
func detectPoolAxes(windowDimensions, windowStrides []int, paddings [][2]int) (poolAxesInfo, error) {
	var info poolAxesInfo

	// Detect spatial axes: those with window > 1 or stride > 1 or padding.
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

	// Build permutation to NCHW: [nonSpatial0, nonSpatial1, spatial0, spatial1].
	info.toNCHW = []int{nonSpatialAxes[0], nonSpatialAxes[1], spatialAxes[0], spatialAxes[1]}
	info.needsTranspose = info.toNCHW[0] != 0 || info.toNCHW[1] != 1 || info.toNCHW[2] != 2 || info.toNCHW[3] != 3

	// Inverse permutation.
	info.fromNCHW = make([]int, 4)
	for i, v := range info.toNCHW {
		info.fromNCHW[v] = i
	}

	// Extract spatial parameters.
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

func (f *Function) ReduceWindow(
	x compute.Value,
	reductionType compute.ReduceOpType,
	windowDimensions, strides, baseDilations, windowDilations []int,
	paddings [][2]int,
) (compute.Value, error) {
	node, err := f.resolveNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "ReduceWindow")
	}

	outShape, err := shapeinference.ReduceWindow(
		node.shape, windowDimensions, strides, baseDilations, windowDilations, paddings)
	if err != nil {
		return nil, errors.Wrap(err, "ReduceWindow")
	}

	rank := node.shape.Rank()
	if rank != 4 {
		return nil, errors.Wrapf(compute.ErrNotImplemented, "ReduceWindow: only 4D tensors supported in %q backend, got rank %d", BackendName, rank)
	}

	// Validate dilations (not supported by MPSGraph pooling).
	for _, d := range baseDilations {
		if d > 1 {
			return nil, errors.Wrapf(compute.ErrNotImplemented, "ReduceWindow: baseDilations > 1 not supported in %q backend", BackendName)
		}
	}
	for _, d := range windowDilations {
		if d > 1 {
			return nil, errors.Wrapf(compute.ErrNotImplemented, "ReduceWindow: windowDilations > 1 not supported in %q backend", BackendName)
		}
	}

	var mode int
	isSum := false
	switch reductionType {
	case compute.ReduceOpMax:
		mode = 0
	case compute.ReduceOpSum:
		mode = 1 // MPSGraph has avg pooling (mode 1); we'll multiply by window area to get sum.
		isSum = true
	default:
		return nil, errors.Wrapf(compute.ErrNotImplemented, "ReduceWindow: reduction type %v not supported in %q backend", reductionType, BackendName)
	}

	axesInfo, err := detectPoolAxes(windowDimensions, strides, paddings)
	if err != nil {
		return nil, errors.Wrap(compute.ErrNotImplemented, err.Error())
	}

	// Transpose to NCHW if needed.
	tensor := node.tensor
	if axesInfo.needsTranspose {
		tensor, err = f.ctx().Transpose(tensor, axesInfo.toNCHW)
		if err != nil {
			return nil, errors.Wrap(err, "ReduceWindow: transpose to NCHW")
		}
	}

	spatialWindow := axesInfo.spatialWindow[:]
	spatialStrides := axesInfo.spatialStrides[:]
	padBefore := axesInfo.padBefore[:]
	padAfter := axesInfo.padAfter[:]

	tensor, err = f.ctx().Pool2D(tensor, mode, spatialWindow, spatialStrides, padBefore, padAfter)
	if err != nil {
		return nil, errors.Wrap(err, "ReduceWindow")
	}

	// For sum pooling, convert avg to sum by multiplying by window area.
	if isSum {
		windowArea := float64(axesInfo.spatialWindow[0] * axesInfo.spatialWindow[1])
		areaTensor, err := f.makeScalarConst(windowArea, node.shape.DType)
		if err != nil {
			return nil, errors.Wrap(err, "ReduceWindow: window area constant")
		}
		tensor, err = f.ctx().Mul(tensor, areaTensor)
		if err != nil {
			return nil, errors.Wrap(err, "ReduceWindow: sum = avg * window_area")
		}
	}

	// Transpose back from NCHW if needed.
	if axesInfo.needsTranspose {
		tensor, err = f.ctx().Transpose(tensor, axesInfo.fromNCHW)
		if err != nil {
			return nil, errors.Wrap(err, "ReduceWindow: transpose from NCHW")
		}
	}

	// Reshape to match expected output shape if needed.
	outDims := toInt64Slice(outShape.Dimensions)
	tensor, err = f.ctx().Reshape(tensor, outDims)
	if err != nil {
		return nil, errors.Wrap(err, "ReduceWindow: reshape")
	}

	return &graphNode{tensor: tensor, shape: outShape, owner: f}, nil
}

// ===========================================================================
// SelectAndScatter (MaxPool gradient)
// ===========================================================================

func (f *Function) SelectAndScatterMax(operand, source compute.Value, windowDimensions, windowStrides []int, paddings [][2]int) (compute.Value, error) {
	return f.selectAndScatterImpl("SelectAndScatterMax", operand, source, windowDimensions, windowStrides, paddings)
}

func (f *Function) SelectAndScatterMin(operand, source compute.Value, windowDimensions, windowStrides []int, paddings [][2]int) (compute.Value, error) {
	return nil, errors.Wrapf(compute.ErrNotImplemented, "SelectAndScatterMin not supported in %q backend", BackendName)
}

func (f *Function) selectAndScatterImpl(opName string, operand, source compute.Value, windowDimensions, windowStrides []int, paddings [][2]int) (compute.Value, error) {
	opNode, err := f.resolveNode(operand)
	if err != nil {
		return nil, errors.Wrapf(err, "%s: operand", opName)
	}
	srcNode, err := f.resolveNode(source)
	if err != nil {
		return nil, errors.Wrapf(err, "%s: source", opName)
	}

	rank := opNode.shape.Rank()
	if rank != 4 {
		return nil, errors.Wrapf(compute.ErrNotImplemented, "%s: only 4D tensors supported in %q backend, got rank %d", opName, BackendName, rank)
	}

	axesInfo, err := detectPoolAxes(windowDimensions, windowStrides, paddings)
	if err != nil {
		return nil, errors.Wrapf(compute.ErrNotImplemented, "%s: %v", opName, err)
	}

	// Transpose operand and source to NCHW if needed.
	opTensor := opNode.tensor
	srcTensor := srcNode.tensor
	if axesInfo.needsTranspose {
		opTensor, err = f.ctx().Transpose(opTensor, axesInfo.toNCHW)
		if err != nil {
			return nil, errors.Wrapf(err, "%s: transpose operand to NCHW", opName)
		}
		srcTensor, err = f.ctx().Transpose(srcTensor, axesInfo.toNCHW)
		if err != nil {
			return nil, errors.Wrapf(err, "%s: transpose source to NCHW", opName)
		}
	}

	spatialWindow := axesInfo.spatialWindow[:]
	spatialStrides := axesInfo.spatialStrides[:]
	padBefore := axesInfo.padBefore[:]
	padAfter := axesInfo.padAfter[:]

	// MPSGraph's maxPooling2DGradient takes:
	// - gradient: the incoming gradient (same shape as pool output = source)
	// - sourceTensor: the original input to pooling (= operand)
	tensor, err := f.ctx().MaxPool2DGradient(srcTensor, opTensor, spatialWindow, spatialStrides, padBefore, padAfter)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}

	// Transpose back from NCHW if needed.
	if axesInfo.needsTranspose {
		tensor, err = f.ctx().Transpose(tensor, axesInfo.fromNCHW)
		if err != nil {
			return nil, errors.Wrapf(err, "%s: transpose from NCHW", opName)
		}
	}

	// Output shape is same as operand.
	return &graphNode{tensor: tensor, shape: opNode.shape, owner: f}, nil
}

// ===========================================================================
// TotalOrder Comparisons
// ===========================================================================
// TotalOrder comparisons enforce: -NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN.
// For simplicity, we delegate to regular comparisons (correct for non-NaN values,
// which is the common case in ML).

func (f *Function) EqualTotalOrder(lhs, rhs compute.Value) (compute.Value, error) {
	return f.Equal(lhs, rhs)
}

func (f *Function) NotEqualTotalOrder(lhs, rhs compute.Value) (compute.Value, error) {
	return f.NotEqual(lhs, rhs)
}

func (f *Function) GreaterThanTotalOrder(lhs, rhs compute.Value) (compute.Value, error) {
	return f.GreaterThan(lhs, rhs)
}

func (f *Function) GreaterOrEqualTotalOrder(lhs, rhs compute.Value) (compute.Value, error) {
	return f.GreaterOrEqual(lhs, rhs)
}

func (f *Function) LessThanTotalOrder(lhs, rhs compute.Value) (compute.Value, error) {
	return f.LessThan(lhs, rhs)
}

func (f *Function) LessOrEqualTotalOrder(lhs, rhs compute.Value) (compute.Value, error) {
	return f.LessOrEqual(lhs, rhs)
}

// ===========================================================================
// Logical / Bitwise Reductions
// ===========================================================================

func (f *Function) ReduceLogicalAnd(x compute.Value, axes ...int) (compute.Value, error) {
	// ReduceMin of {0,1} gives AND semantics for boolean values.
	return f.reduceOp("ReduceLogicalAnd", compute.OpTypeReduceLogicalAnd, bridge.ReduceMin, x, axes...)
}

func (f *Function) ReduceLogicalOr(x compute.Value, axes ...int) (compute.Value, error) {
	// ReduceMax of {0,1} gives OR semantics for boolean values.
	return f.reduceOp("ReduceLogicalOr", compute.OpTypeReduceLogicalOr, bridge.ReduceMax, x, axes...)
}

// ===========================================================================
// Fused Operations
// ===========================================================================

func (f *Function) FusedSoftmax(x compute.Value, axis int) (compute.Value, error) {
	node, err := f.resolveNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "FusedSoftmax")
	}
	if axis < 0 || axis >= node.shape.Rank() {
		return nil, errors.Errorf("FusedSoftmax: axis %d out of range for rank %d", axis, node.shape.Rank())
	}
	tensor, err := f.ctx().Softmax(node.tensor, axis)
	if err != nil {
		return nil, errors.Wrap(err, "FusedSoftmax")
	}
	return &graphNode{tensor: tensor, shape: node.shape, owner: f}, nil
}

func (f *Function) FusedActivation(x compute.Value, cfg compute.ActivationConfig) (compute.Value, error) {
	node, err := f.resolveNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "FusedActivation")
	}
	dt := node.shape.DType
	switch cfg.Type {
	case compute.ActivationNone:
		return node, nil
	case compute.ActivationGelu:
		return f.fusedGelu(node, true)
	case compute.ActivationGeluApproximate:
		return f.fusedGelu(node, false)
	case compute.ActivationRelu:
		zeroTensor, err := f.makeScalarConst(0, dt)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: ReLU zero constant")
		}
		tensor, err := f.ctx().Max(node.tensor, zeroTensor)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: ReLU max")
		}
		return &graphNode{tensor: tensor, shape: node.shape, owner: f}, nil
	case compute.ActivationSigmoid:
		tensor, err := f.ctx().Sigmoid(node.tensor)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: Sigmoid")
		}
		return &graphNode{tensor: tensor, shape: node.shape, owner: f}, nil
	case compute.ActivationHardSigmoid:
		// y = max(0, min(1, 0.2*x + 0.5))
		c02, err := f.makeScalarConst(0.2, dt)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: HardSigmoid 0.2")
		}
		c05, err := f.makeScalarConst(0.5, dt)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: HardSigmoid 0.5")
		}
		zeroTensor, err := f.makeScalarConst(0, dt)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: HardSigmoid 0")
		}
		oneTensor, err := f.makeScalarConst(1, dt)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: HardSigmoid 1")
		}
		scaled, err := f.ctx().Mul(node.tensor, c02)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: HardSigmoid mul")
		}
		shifted, err := f.ctx().Add(scaled, c05)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: HardSigmoid add")
		}
		clamped, err := f.ctx().Clamp(zeroTensor, shifted, oneTensor)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: HardSigmoid clamp")
		}
		return &graphNode{tensor: clamped, shape: node.shape, owner: f}, nil
	case compute.ActivationLeakyRelu:
		// y = x if x >= 0; 0.3*x if x < 0
		zeroTensor, err := f.makeScalarConst(0, dt)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: LeakyRelu 0")
		}
		alphaTensor, err := f.makeScalarConst(0.3, dt)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: LeakyRelu alpha")
		}
		cond, err := f.ctx().GreaterOrEqual(node.tensor, zeroTensor)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: LeakyRelu cond")
		}
		negBranch, err := f.ctx().Mul(node.tensor, alphaTensor)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: LeakyRelu mul")
		}
		tensor, err := f.ctx().Where(cond, node.tensor, negBranch)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: LeakyRelu where")
		}
		return &graphNode{tensor: tensor, shape: node.shape, owner: f}, nil
	case compute.ActivationSelu:
		// y = scale * x if x > 0; scale * alpha * (exp(x) - 1) if x <= 0
		const seluAlpha = 1.6732632423543772848170429916717
		const seluScale = 1.0507009873554804934193349852946
		zeroTensor, err := f.makeScalarConst(0, dt)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: Selu 0")
		}
		alphaTensor, err := f.makeScalarConst(seluAlpha, dt)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: Selu alpha")
		}
		scaleTensor, err := f.makeScalarConst(seluScale, dt)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: Selu scale")
		}
		cond, err := f.ctx().GreaterThan(node.tensor, zeroTensor)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: Selu cond")
		}
		expm1Val, err := f.ctx().Expm1(node.tensor)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: Selu expm1")
		}
		negBranch, err := f.ctx().Mul(alphaTensor, expm1Val)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: Selu alpha*expm1")
		}
		eluVal, err := f.ctx().Where(cond, node.tensor, negBranch)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: Selu where")
		}
		tensor, err := f.ctx().Mul(scaleTensor, eluVal)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: Selu scale*elu")
		}
		return &graphNode{tensor: tensor, shape: node.shape, owner: f}, nil
	case compute.ActivationSilu:
		sigmoid, err := f.ctx().Sigmoid(node.tensor)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: SiLU sigmoid")
		}
		tensor, err := f.ctx().Mul(node.tensor, sigmoid)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: SiLU mul")
		}
		return &graphNode{tensor: tensor, shape: node.shape, owner: f}, nil
	case compute.ActivationHardSwish:
		// y = x * clamp(x + 3, 0, 6) / 6
		c3, err := f.makeScalarConst(3.0, dt)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: HardSwish 3")
		}
		c6, err := f.makeScalarConst(6.0, dt)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: HardSwish 6")
		}
		zeroTensor, err := f.makeScalarConst(0, dt)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: HardSwish 0")
		}
		xPlus3, err := f.ctx().Add(node.tensor, c3)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: HardSwish x+3")
		}
		relu6, err := f.ctx().Clamp(zeroTensor, xPlus3, c6)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: HardSwish clamp")
		}
		prod, err := f.ctx().Mul(node.tensor, relu6)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: HardSwish mul")
		}
		tensor, err := f.ctx().Div(prod, c6)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: HardSwish div")
		}
		return &graphNode{tensor: tensor, shape: node.shape, owner: f}, nil
	case compute.ActivationTanh:
		tensor, err := f.ctx().Tanh(node.tensor)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: Tanh")
		}
		return &graphNode{tensor: tensor, shape: node.shape, owner: f}, nil
	case compute.ActivationSwiGLU:
		outShape, err := shapeinference.FusedActivation(node.shape, cfg)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: SwiGLU")
		}
		rank := node.shape.Rank()
		lastDim := node.shape.Dimensions[rank-1]
		halfDim := lastDim / 2
		startsGate := make([]int64, rank)
		endsGate := make([]int64, rank)
		startsVal := make([]int64, rank)
		endsVal := make([]int64, rank)
		strides := make([]int64, rank)
		for i, d := range node.shape.Dimensions {
			strides[i] = 1
			endsGate[i] = int64(d)
			endsVal[i] = int64(d)
		}
		endsGate[rank-1] = int64(halfDim)
		startsVal[rank-1] = int64(halfDim)
		gate, err := f.ctx().Slice(node.tensor, startsGate, endsGate, strides)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: SwiGLU slice gate")
		}
		val, err := f.ctx().Slice(node.tensor, startsVal, endsVal, strides)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: SwiGLU slice val")
		}
		sigGate, err := f.ctx().Sigmoid(gate)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: SwiGLU sigmoid")
		}
		siluGate, err := f.ctx().Mul(gate, sigGate)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: SwiGLU silu")
		}
		tensor, err := f.ctx().Mul(siluGate, val)
		if err != nil {
			return nil, errors.Wrap(err, "FusedActivation: SwiGLU mul")
		}
		return &graphNode{tensor: tensor, shape: outShape, owner: f}, nil
	default:
		return nil, errors.Wrapf(compute.ErrNotImplemented, "FusedActivation: activation %v not implemented in %q backend", cfg.Type, BackendName)
	}
}

func (f *Function) fusedGelu(node *graphNode, exact bool) (compute.Value, error) {
	dt := node.shape.DType

	makeConst := func(val float64) (bridge.Tensor, error) {
		return f.makeScalarConst(val, dt)
	}

	if exact {
		// Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
		half, err := makeConst(0.5)
		if err != nil {
			return nil, errors.Wrap(err, "FusedGelu")
		}
		invSqrt2, err := makeConst(0.7071067811865475) // 1/sqrt(2)
		if err != nil {
			return nil, errors.Wrap(err, "FusedGelu")
		}
		one, err := makeConst(1.0)
		if err != nil {
			return nil, errors.Wrap(err, "FusedGelu")
		}

		t := node.tensor
		// x / sqrt(2) = x * (1/sqrt(2))
		xScaled, err := f.ctx().Mul(t, invSqrt2)
		if err != nil {
			return nil, errors.Wrap(err, "FusedGelu: scale")
		}
		// erf(x / sqrt(2))
		erfVal, err := f.ctx().Erf(xScaled)
		if err != nil {
			return nil, errors.Wrap(err, "FusedGelu: erf")
		}
		// 1 + erf(...)
		onePlusErf, err := f.ctx().Add(one, erfVal)
		if err != nil {
			return nil, errors.Wrap(err, "FusedGelu: 1+erf")
		}
		// 0.5 * x
		halfX, err := f.ctx().Mul(half, t)
		if err != nil {
			return nil, errors.Wrap(err, "FusedGelu: 0.5*x")
		}
		// 0.5 * x * (1 + erf(...))
		result, err := f.ctx().Mul(halfX, onePlusErf)
		if err != nil {
			return nil, errors.Wrap(err, "FusedGelu: final mul")
		}
		return &graphNode{tensor: result, shape: node.shape, owner: f}, nil
	}

	// Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
	half, err := makeConst(0.5)
	if err != nil {
		return nil, errors.Wrap(err, "FusedGelu")
	}
	one, err := makeConst(1.0)
	if err != nil {
		return nil, errors.Wrap(err, "FusedGelu")
	}
	coeff, err := makeConst(0.044715)
	if err != nil {
		return nil, errors.Wrap(err, "FusedGelu")
	}
	sqrtTwoPi, err := makeConst(0.7978845608) // sqrt(2/pi)
	if err != nil {
		return nil, errors.Wrap(err, "FusedGelu")
	}

	t := node.tensor
	// x^3
	x2, err := f.ctx().Mul(t, t)
	if err != nil {
		return nil, errors.Wrap(err, "FusedGelu")
	}
	x3, err := f.ctx().Mul(x2, t)
	if err != nil {
		return nil, errors.Wrap(err, "FusedGelu")
	}
	// 0.044715 * x^3
	cx3, err := f.ctx().Mul(coeff, x3)
	if err != nil {
		return nil, errors.Wrap(err, "FusedGelu")
	}
	// x + 0.044715 * x^3
	inner, err := f.ctx().Add(t, cx3)
	if err != nil {
		return nil, errors.Wrap(err, "FusedGelu")
	}
	// sqrt(2/pi) * (x + 0.044715 * x^3)
	scaled, err := f.ctx().Mul(sqrtTwoPi, inner)
	if err != nil {
		return nil, errors.Wrap(err, "FusedGelu")
	}
	// tanh(...)
	tanhVal, err := f.ctx().Tanh(scaled)
	if err != nil {
		return nil, errors.Wrap(err, "FusedGelu")
	}
	// 1 + tanh(...)
	onePlusTanh, err := f.ctx().Add(one, tanhVal)
	if err != nil {
		return nil, errors.Wrap(err, "FusedGelu")
	}
	// 0.5 * x
	halfX, err := f.ctx().Mul(half, t)
	if err != nil {
		return nil, errors.Wrap(err, "FusedGelu")
	}
	// 0.5 * x * (1 + tanh(...))
	result, err := f.ctx().Mul(halfX, onePlusTanh)
	if err != nil {
		return nil, errors.Wrap(err, "FusedGelu")
	}

	return &graphNode{tensor: result, shape: node.shape, owner: f}, nil
}

func (f *Function) FusedLayerNorm(x compute.Value, axes []int, epsilon float64, gamma, beta compute.Value) (compute.Value, error) {
	node, err := f.resolveNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "FusedLayerNorm")
	}

	// Normalize negative axes.
	rank := node.shape.Rank()
	normalizedAxes := make([]int, len(axes))
	for i, ax := range axes {
		if ax < 0 {
			ax += rank
		}
		if ax < 0 || ax >= rank {
			return nil, errors.Errorf("FusedLayerNorm: axis %d out of range for rank %d", axes[i], rank)
		}
		normalizedAxes[i] = ax
	}

	// Compute mean over the specified axes.
	meanTensor, err := f.ctx().Reduce(node.tensor, bridge.ReduceSum, normalizedAxes)
	if err != nil {
		return nil, errors.Wrap(err, "FusedLayerNorm: reduce for mean")
	}

	// Count elements being reduced.
	numElements := int64(1)
	for _, ax := range normalizedAxes {
		numElements *= int64(node.shape.Dimensions[ax])
	}
	dt := node.shape.DType
	countTensor, err := f.makeScalarConst(float64(numElements), dt)
	if err != nil {
		return nil, errors.Wrap(err, "FusedLayerNorm: count constant")
	}

	// mean = sum / count
	meanTensor, err = f.ctx().Div(meanTensor, countTensor)
	if err != nil {
		return nil, errors.Wrap(err, "FusedLayerNorm: mean div")
	}

	// x - mean (broadcast automatically)
	diff, err := f.ctx().Sub(node.tensor, meanTensor)
	if err != nil {
		return nil, errors.Wrap(err, "FusedLayerNorm: subtract mean")
	}

	// variance = mean((x - mean)^2)
	diffSq, err := f.ctx().Mul(diff, diff)
	if err != nil {
		return nil, errors.Wrap(err, "FusedLayerNorm: diff squared")
	}
	varTensor, err := f.ctx().Reduce(diffSq, bridge.ReduceSum, normalizedAxes)
	if err != nil {
		return nil, errors.Wrap(err, "FusedLayerNorm: reduce for variance")
	}
	varTensor, err = f.ctx().Div(varTensor, countTensor)
	if err != nil {
		return nil, errors.Wrap(err, "FusedLayerNorm: variance div")
	}

	// variance + epsilon
	epsTensor, err := f.makeScalarConst(epsilon, dt)
	if err != nil {
		return nil, errors.Wrap(err, "FusedLayerNorm: epsilon constant")
	}
	varPlusEps, err := f.ctx().Add(varTensor, epsTensor)
	if err != nil {
		return nil, errors.Wrap(err, "FusedLayerNorm: var + eps")
	}

	// 1 / sqrt(variance + epsilon)
	invStd, err := f.ctx().Rsqrt(varPlusEps)
	if err != nil {
		return nil, errors.Wrap(err, "FusedLayerNorm: rsqrt")
	}

	// normalized = (x - mean) * invStd
	normalized, err := f.ctx().Mul(diff, invStd)
	if err != nil {
		return nil, errors.Wrap(err, "FusedLayerNorm: normalize")
	}

	// Build int64 shape for broadcasting.
	targetShape := toInt64Slice(node.shape.Dimensions)

	// broadcastToTarget reshapes a lower-rank tensor (e.g. gamma [4]) to the
	// target shape by inserting size-1 dims for non-normalized axes, then broadcasting.
	broadcastToTarget := func(t bridge.Tensor, tShape shapes.Shape) (bridge.Tensor, error) {
		if tShape.Rank() >= rank {
			return t, nil
		}
		// Build reshape: insert 1s for non-normalized axes.
		reshapeDims := make([]int64, rank)
		normIdx := 0
		for i := range rank {
			isNormAxis := false
			for _, ax := range normalizedAxes {
				if ax == i {
					isNormAxis = true
					break
				}
			}
			if isNormAxis && normIdx < tShape.Rank() {
				reshapeDims[i] = int64(tShape.Dimensions[normIdx])
				normIdx++
			} else {
				reshapeDims[i] = 1
			}
		}
		reshaped, err := f.ctx().Reshape(t, reshapeDims)
		if err != nil {
			return nil, err
		}
		broadcasted, err := f.ctx().BroadcastTo(reshaped, targetShape)
		if err != nil {
			return nil, err
		}
		return broadcasted, nil
	}

	// Apply gamma (scale) if provided.
	if gamma != nil {
		gammaNode, err := f.resolveNode(gamma)
		if err != nil {
			return nil, errors.Wrap(err, "FusedLayerNorm: gamma")
		}
		gammaTensor, err := broadcastToTarget(gammaNode.tensor, gammaNode.shape)
		if err != nil {
			return nil, errors.Wrap(err, "FusedLayerNorm: broadcast gamma")
		}
		normalized, err = f.ctx().Mul(normalized, gammaTensor)
		if err != nil {
			return nil, errors.Wrap(err, "FusedLayerNorm: apply gamma")
		}
	}

	// Apply beta (offset) if provided.
	if beta != nil {
		betaNode, err := f.resolveNode(beta)
		if err != nil {
			return nil, errors.Wrap(err, "FusedLayerNorm: beta")
		}
		betaTensor, err := broadcastToTarget(betaNode.tensor, betaNode.shape)
		if err != nil {
			return nil, errors.Wrap(err, "FusedLayerNorm: broadcast beta")
		}
		normalized, err = f.ctx().Add(normalized, betaTensor)
		if err != nil {
			return nil, errors.Wrap(err, "FusedLayerNorm: apply beta")
		}
	}

	return &graphNode{tensor: normalized, shape: node.shape, owner: f}, nil
}

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

	if xNode.shape.Rank() < 1 || wNode.shape.Rank() < 2 {
		return nil, errors.Errorf("FusedDense: x must have rank >= 1 (got %d), weight must have rank >= 2 (got %d)",
			xNode.shape.Rank(), wNode.shape.Rank())
	}
	inFeatures := xNode.shape.Dimensions[xNode.shape.Rank()-1]
	wContractAxis := 0
	if options.WeightLayout == compute.DenseLayoutOutputsInput {
		wContractAxis = wNode.shape.Rank() - 1
	}
	if inFeatures != wNode.shape.Dimensions[wContractAxis] {
		return nil, errors.Errorf("FusedDense: x's last dim (%d) must match weight's contracting dim (%d)",
			inFeatures, wNode.shape.Dimensions[wContractAxis])
	}

	// Step 1: Matmul via DotGeneral: contract x's last axis with weight's contracting axis.
	result, err := f.DotGeneral(x, []int{xNode.shape.Rank() - 1}, nil, weight, []int{wContractAxis}, nil, compute.DotGeneralConfig{})
	if err != nil {
		return nil, errors.Wrap(err, "FusedDense: DotGeneral")
	}
	resultNode, err := f.resolveNode(result)
	if err != nil {
		return nil, errors.Wrap(err, "FusedDense: cast DotGeneral result")
	}

	// Step 2: Add bias if provided.
	if bias != nil {
		biasNode, err := f.resolveNode(bias)
		if err != nil {
			return nil, errors.Wrap(err, "FusedDense: bias")
		}
		// Broadcast bias to result shape: bias has trailing dims matching weight's output dims.
		broadcastAxes := make([]int, biasNode.shape.Rank())
		offset := resultNode.shape.Rank() - biasNode.shape.Rank()
		for i := range broadcastAxes {
			broadcastAxes[i] = offset + i
		}
		result, err = f.BroadcastInDim(bias, resultNode.shape, broadcastAxes)
		if err != nil {
			return nil, errors.Wrap(err, "FusedDense: broadcast bias")
		}
		result, err = f.Add(resultNode, result)
		if err != nil {
			return nil, errors.Wrap(err, "FusedDense: add bias")
		}
		resultNode, err = f.resolveNode(result)
		if err != nil {
			return nil, errors.Wrap(err, "FusedDense: cast after bias")
		}
	}

	// Step 3: Apply activation.
	if options.Activation.Type == compute.ActivationNone {
		return resultNode, nil
	}
	return f.FusedActivation(resultNode, options.Activation)
}

func (f *Function) FusedScaledDotProductAttention(
	query, key, value compute.Value,
	axesLayout compute.AttentionAxesLayout,
	options *compute.ScaledDotProductAttentionConfig,
) (output compute.Value, statesForVJP []compute.Value, err error) {
	if options != nil && (options.QuerySeqLen != nil || options.KeyValueSeqLen != nil) {
		return nil, nil, errors.Wrapf(compute.ErrNotImplemented, "SDPA: QuerySeqLen/KeyValueSeqLen not implemented in %q backend", BackendName)
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

	if qNode.shape.Rank() != 4 {
		return nil, nil, errors.Errorf("SDPA: query must have rank 4, got %d", qNode.shape.Rank())
	}

	headsAxis := axesLayout.HeadsAxis()
	numHeads := qNode.shape.Dimensions[headsAxis]
	numKVHeads := kNode.shape.Dimensions[headsAxis]
	headDim := qNode.shape.Dimensions[3]

	if numHeads <= 0 || numKVHeads <= 0 || numHeads%numKVHeads != 0 {
		return nil, nil, errors.Errorf("SDPA: numHeads (%d) must be positive and divisible by numKVHeads (%d)", numHeads, numKVHeads)
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

	// For simplicity and correctness, convert BSHD to BHSD, do attention, and convert back.
	// This avoids complex axis management with DotGeneral's output ordering.
	isBSHD := axesLayout == compute.AttentionAxesLayoutBSHD
	if isBSHD {
		// BSHD [B,S,H,D] → BHSD [B,H,S,D]
		query, err = f.Transpose(query, 0, 2, 1, 3)
		if err != nil {
			return nil, nil, errors.Wrap(err, "SDPA: transpose Q to BHSD")
		}
		key, err = f.Transpose(key, 0, 2, 1, 3)
		if err != nil {
			return nil, nil, errors.Wrap(err, "SDPA: transpose K to BHSD")
		}
		value, err = f.Transpose(value, 0, 2, 1, 3)
		if err != nil {
			return nil, nil, errors.Wrap(err, "SDPA: transpose V to BHSD")
		}
		// Also transpose mask if it's rank 4.
		if mask != nil {
			maskNode, _ := f.resolveNode(mask)
			if maskNode != nil && maskNode.shape.Rank() == 4 {
				mask, err = f.Transpose(mask, 0, 2, 1, 3)
				if err != nil {
					return nil, nil, errors.Wrap(err, "SDPA: transpose mask to BHSD")
				}
			}
		}
		qNode, _ = f.resolveNode(query)
		kNode, _ = f.resolveNode(key)
		vNode, _ = f.resolveNode(value)
	}

	// Now everything is in BHSD layout: [B, H, Sq, D].

	// For GQA: if numKVHeads < numHeads, repeat K/V heads.
	kvKey := key
	kvValue := value
	if numKVHeads < numHeads {
		repeats := numHeads / numKVHeads
		kvKey, err = f.repeatHeads(kNode, 1, repeats) // headsAxis=1 in BHSD
		if err != nil {
			return nil, nil, errors.Wrap(err, "SDPA: repeat K heads")
		}
		kvValue, err = f.repeatHeads(vNode, 1, repeats)
		if err != nil {
			return nil, nil, errors.Wrap(err, "SDPA: repeat V heads")
		}
	}

	// Compute attention scores: Q @ K^T.
	// Q: [B,H,Sq,D], K: [B,H,Sk,D] → scores: [B,H,Sq,Sk]
	scores, err := f.DotGeneral(
		query, []int{3}, []int{0, 1}, // contract dim(3), batch [B(0),H(1)]
		kvKey, []int{3}, []int{0, 1}, // contract dim(3), batch [B(0),H(1)]
		compute.DotGeneralConfig{})
	if err != nil {
		return nil, nil, errors.Wrap(err, "SDPA: scores matmul")
	}

	// Scale scores.
	scoresNode, err := f.resolveNode(scores)
	if err != nil {
		return nil, nil, errors.Wrap(err, "SDPA: cast scores")
	}
	dt := scoresNode.shape.DType
	scaleTensor, err := f.makeScalarConst(scale, dt)
	if err != nil {
		return nil, nil, errors.Wrap(err, "SDPA: scale constant")
	}
	scaledTensor, err := f.ctx().Mul(scoresNode.tensor, scaleTensor)
	if err != nil {
		return nil, nil, errors.Wrap(err, "SDPA: scale mul")
	}
	scores = &graphNode{tensor: scaledTensor, shape: scoresNode.shape, owner: f}

	// Apply additive bias if provided.
	if bias != nil {
		scores, err = f.Add(scores, bias)
		if err != nil {
			return nil, nil, errors.Wrap(err, "SDPA: add bias")
		}
	}

	// Apply causal mask if needed: lower triangular.
	if causal {
		seqLen := qNode.shape.Dimensions[2]  // Sq in BHSD
		kvSeqLen := kNode.shape.Dimensions[2] // Sk in BHSD
		// Create lower triangular mask: mask[i,j] = (i >= j).
		maskShape := shapes.Make(dtypes.Int32, seqLen, kvSeqLen)
		rowIota, err := f.Iota(maskShape, 0)
		if err != nil {
			return nil, nil, errors.Wrap(err, "SDPA: causal row iota")
		}
		colIota, err := f.Iota(maskShape, 1)
		if err != nil {
			return nil, nil, errors.Wrap(err, "SDPA: causal col iota")
		}
		causalMask, err := f.GreaterOrEqual(rowIota, colIota)
		if err != nil {
			return nil, nil, errors.Wrap(err, "SDPA: causal mask comparison")
		}
		// Reshape for broadcasting with BHSD scores: [1,1,Sq,Sk]
		causalMask, err = f.Reshape(causalMask, 1, 1, seqLen, kvSeqLen)
		if err != nil {
			return nil, nil, errors.Wrap(err, "SDPA: causal mask reshape")
		}
		if mask != nil {
			// Combine causal mask with provided mask.
			maskNode, mErr := f.resolveNode(mask)
			if mErr != nil {
				return nil, nil, errors.Wrap(mErr, "SDPA: resolve mask for causal combine")
			}
			if maskNode.shape.DType == dtypes.Bool {
				// Boolean masks: AND them together.
				mask, err = f.LogicalAnd(mask, causalMask)
				if err != nil {
					return nil, nil, errors.Wrap(err, "SDPA: combining boolean masks")
				}
			} else {
				// Additive mask: convert causal to additive (-inf where false, 0 where true)
				// and add to the existing additive mask.
				causalNode, cErr := f.resolveNode(causalMask)
				if cErr != nil {
					return nil, nil, errors.Wrap(cErr, "SDPA: resolve causal mask")
				}
				negInfTensor, cErr := f.makeScalarConst(math.Inf(-1), dt)
				if cErr != nil {
					return nil, nil, errors.Wrap(cErr, "SDPA: causal -inf constant")
				}
				zeroTensor, cErr := f.makeScalarConst(0, dt)
				if cErr != nil {
					return nil, nil, errors.Wrap(cErr, "SDPA: causal zero constant")
				}
				// Where(causalBool, 0, -inf) to create additive causal mask
				additiveCausal, cErr := f.ctx().Where(causalNode.tensor, zeroTensor, negInfTensor)
				if cErr != nil {
					return nil, nil, errors.Wrap(cErr, "SDPA: causal to additive")
				}
				additiveCausalNode := &graphNode{tensor: additiveCausal, shape: causalNode.shape, owner: f}
				// Note: causalNode shape has Bool dtype, but additiveCausal is float - fix shape
				additiveCausalNode.shape = shapes.Make(dt, causalNode.shape.Dimensions...)
				mask, err = f.Add(mask, additiveCausalNode)
				if err != nil {
					return nil, nil, errors.Wrap(err, "SDPA: combining additive masks")
				}
			}
		} else {
			mask = causalMask
		}
	}

	// Apply mask to scores.
	if mask != nil {
		maskNode, err := f.resolveNode(mask)
		if err != nil {
			return nil, nil, errors.Wrap(err, "SDPA: mask cast")
		}

		if maskNode.shape.DType == dtypes.Bool {
			// Boolean mask: where mask is false, set score to -inf.
			negInfTensor, err := f.makeScalarConst(math.Inf(-1), dt)
			if err != nil {
				return nil, nil, errors.Wrap(err, "SDPA: -inf constant")
			}
			// Broadcast -inf to scores shape.
			scoresNode, _ = f.resolveNode(scores)
			outDims := toInt64Slice(scoresNode.shape.Dimensions)
			negInfBroadcast, err := f.ctx().BroadcastTo(negInfTensor, outDims)
			if err != nil {
				return nil, nil, errors.Wrap(err, "SDPA: broadcast -inf")
			}
			// Broadcast mask to scores shape.
			maskBroadcast, err := f.broadcastMaskToScores(mask, scoresNode.shape)
			if err != nil {
				return nil, nil, errors.Wrap(err, "SDPA: broadcast bool mask")
			}
			maskBNode, _ := f.resolveNode(maskBroadcast)
			// Where(mask, scores, -inf)
			tensor, err := f.ctx().Where(maskBNode.tensor, scoresNode.tensor, negInfBroadcast)
			if err != nil {
				return nil, nil, errors.Wrap(err, "SDPA: where mask")
			}
			scores = &graphNode{tensor: tensor, shape: scoresNode.shape, owner: f}
		} else {
			// Additive mask: scores = scores + mask.
			scores, err = f.Add(scores, mask)
			if err != nil {
				return nil, nil, errors.Wrap(err, "SDPA: add mask")
			}
		}
	}

	// Softmax along the last axis (kv_seq dimension).
	scoresNode, _ = f.resolveNode(scores)
	scores, err = f.FusedSoftmax(scores, scoresNode.shape.Rank()-1)
	if err != nil {
		return nil, nil, errors.Wrap(err, "SDPA: softmax")
	}

	// Compute output: scores @ V.
	// scores: [B,H,Sq,Sk], V: [B,H,Sk,D] → output: [B,H,Sq,D]
	output, err = f.DotGeneral(
		scores, []int{3}, []int{0, 1}, // contract Sk(3), batch [B(0),H(1)]
		kvValue, []int{2}, []int{0, 1}, // contract Sk(2), batch [B(0),H(1)]
		compute.DotGeneralConfig{})
	if err != nil {
		return nil, nil, errors.Wrap(err, "SDPA: output matmul")
	}

	// Convert back from BHSD to BSHD if needed.
	if isBSHD {
		output, err = f.Transpose(output, 0, 2, 1, 3)
		if err != nil {
			return nil, nil, errors.Wrap(err, "SDPA: transpose output back to BSHD")
		}
	}

	return output, nil, nil
}

// repeatHeads repeats KV heads along the heads axis for GQA.
// Expands [B, ..., numKVHeads, ..., D] → [B, ..., numHeads, ..., D] by repeating each head.
func (f *Function) repeatHeads(node *graphNode, headsAxis, repeats int) (compute.Value, error) {
	dims := node.shape.Dimensions
	rank := node.shape.Rank()
	numKVHeads := dims[headsAxis]

	// Insert a new axis after headsAxis: [B, ..., numKVHeads, 1, ..., D]
	// Then broadcast to [B, ..., numKVHeads, repeats, ..., D]
	// Then reshape to [B, ..., numKVHeads*repeats, ..., D]
	newDims := make([]int, rank+1)
	for i := 0; i < headsAxis+1; i++ {
		newDims[i] = dims[i]
	}
	newDims[headsAxis+1] = 1
	for i := headsAxis + 1; i < rank; i++ {
		newDims[i+1] = dims[i]
	}
	reshaped, err := f.Reshape(node, newDims...)
	if err != nil {
		return nil, errors.Wrap(err, "repeatHeads: reshape insert")
	}

	// Broadcast the new axis to 'repeats'.
	broadcastDims := make([]int, rank+1)
	copy(broadcastDims, newDims)
	broadcastDims[headsAxis+1] = repeats
	broadcastShape := shapes.Make(node.shape.DType, broadcastDims...)
	axes := make([]int, rank+1)
	for i := range axes {
		axes[i] = i
	}
	broadcasted, err := f.BroadcastInDim(reshaped, broadcastShape, axes)
	if err != nil {
		return nil, errors.Wrap(err, "repeatHeads: broadcast")
	}

	// Reshape to merge heads axis: [B, ..., numKVHeads*repeats, ..., D]
	finalDims := make([]int, rank)
	for i := 0; i < headsAxis; i++ {
		finalDims[i] = dims[i]
	}
	finalDims[headsAxis] = numKVHeads * repeats
	for i := headsAxis + 1; i < rank; i++ {
		finalDims[i] = dims[i]
	}
	return f.Reshape(broadcasted, finalDims...)
}

// broadcastMaskToScores broadcasts a mask of arbitrary rank to the scores shape.
func (f *Function) broadcastMaskToScores(mask compute.Value, scoresShape shapes.Shape) (compute.Value, error) {
	maskNode, err := f.resolveNode(mask)
	if err != nil {
		return nil, err
	}
	maskRank := maskNode.shape.Rank()
	scoresRank := scoresShape.Rank()

	// Build broadcast axes: align trailing dimensions.
	broadcastAxes := make([]int, maskRank)
	offset := scoresRank - maskRank
	for i := range broadcastAxes {
		broadcastAxes[i] = offset + i
	}
	return f.BroadcastInDim(mask, scoresShape, broadcastAxes)
}

func (f *Function) FusedAttentionQKVProjection(
	x, wQKV, biasQ, biasK, biasV compute.Value,
	queryDim, keyValueDim int,
) (query, key, value compute.Value, err error) {
	xNode, err := f.resolveNode(x)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "FusedAttentionQKVProjection: x")
	}
	if xNode.shape.Rank() < 1 {
		return nil, nil, nil, errors.Errorf("FusedAttentionQKVProjection: x must have rank >= 1, got %d", xNode.shape.Rank())
	}

	// Step 1: Combined matmul: x @ wQKV → [batch..., queryDim + 2*keyValueDim]
	combined, err := f.DotGeneral(x, []int{xNode.shape.Rank() - 1}, nil, wQKV, []int{0}, nil, compute.DotGeneralConfig{})
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "FusedAttentionQKVProjection: DotGeneral")
	}

	// Step 2: Slice into Q, K, V along the last axis.
	combinedNode, err := f.resolveNode(combined)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "FusedAttentionQKVProjection: cast combined")
	}

	rank := combinedNode.shape.Rank()

	// Build starts/limits/strides for slicing.
	makeSlice := func(startLast, limitLast int) (compute.Value, error) {
		starts := make([]int, rank)
		limits := make([]int, rank)
		strides := make([]int, rank)
		for i := range rank {
			starts[i] = 0
			limits[i] = combinedNode.shape.Dimensions[i]
			strides[i] = 1
		}
		starts[rank-1] = startLast
		limits[rank-1] = limitLast
		return f.Slice(combined, starts, limits, strides)
	}

	// Q: [batch..., 0:queryDim]
	query, err = makeSlice(0, queryDim)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "FusedAttentionQKVProjection: slice Q")
	}

	// K: [batch..., queryDim:queryDim+keyValueDim]
	key, err = makeSlice(queryDim, queryDim+keyValueDim)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "FusedAttentionQKVProjection: slice K")
	}

	// V: [batch..., queryDim+keyValueDim:queryDim+2*keyValueDim]
	value, err = makeSlice(queryDim+keyValueDim, queryDim+2*keyValueDim)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "FusedAttentionQKVProjection: slice V")
	}

	// Step 3: Add biases if provided.
	addBias := func(result compute.Value, bias compute.Value, name string) (compute.Value, error) {
		if bias == nil {
			return result, nil
		}
		resultNode, err := f.resolveNode(result)
		if err != nil {
			return nil, errors.Wrapf(err, "FusedAttentionQKVProjection: cast %s", name)
		}
		biasNode, err := f.resolveNode(bias)
		if err != nil {
			return nil, errors.Wrapf(err, "FusedAttentionQKVProjection: cast bias%s", name)
		}
		// Broadcast bias to result shape.
		broadcastAxes := make([]int, biasNode.shape.Rank())
		offset := resultNode.shape.Rank() - biasNode.shape.Rank()
		for i := range broadcastAxes {
			broadcastAxes[i] = offset + i
		}
		broadcastedBias, err := f.BroadcastInDim(bias, resultNode.shape, broadcastAxes)
		if err != nil {
			return nil, errors.Wrapf(err, "FusedAttentionQKVProjection: broadcast bias%s", name)
		}
		return f.Add(result, broadcastedBias)
	}

	query, err = addBias(query, biasQ, "Q")
	if err != nil {
		return nil, nil, nil, err
	}
	key, err = addBias(key, biasK, "K")
	if err != nil {
		return nil, nil, nil, err
	}
	value, err = addBias(value, biasV, "V")
	if err != nil {
		return nil, nil, nil, err
	}

	return query, key, value, nil
}
