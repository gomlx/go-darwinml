// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mps

import (
	"sync"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/notimplemented"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/go-darwinml/internal/bridge"
	"github.com/pkg/errors"
)

// Builder implements compute.Builder for the MPS backend.
type Builder struct {
	notimplemented.Builder // Bootstrap: unimplemented ops return ErrNotImplemented.

	mu       sync.Mutex
	backend  *Backend
	name     string
	ctx      *bridge.Context
	mainFn   *Function
	compiled bool
	untrack  func()
}

// Verify interface compliance.
var _ compute.Builder = &Builder{}

// newBuilder creates a builder with its own MPSGraph context.
func newBuilder(backend *Backend, name string, ctx *bridge.Context) *Builder {
	return &Builder{
		backend: backend,
		name:    name,
		ctx:     ctx,
	}
}

// Name returns the builder name.
func (b *Builder) Name() string { return b.name }

// Finalize releases the builder's MPSGraph context.
func (b *Builder) Finalize() {
	if b == nil {
		return
	}
	b.mu.Lock()
	untrack := b.untrack
	b.untrack = nil
	ctx := b.ctx
	b.ctx = nil
	b.mu.Unlock()
	if untrack != nil {
		untrack()
	}
	if ctx != nil {
		ctx.Destroy()
	}
}

// Main returns the main function, creating it lazily.
func (b *Builder) Main() compute.Function {
	if b.mainFn == nil {
		b.mainFn = newFunction(b, compute.MainName, nil)
	}
	return b.mainFn
}

// NewFunction creates a named sub-function.
func (b *Builder) NewFunction(name string) (compute.Function, error) {
	if name == compute.MainName {
		return nil, errors.Errorf("cannot create function with reserved name %q", compute.MainName)
	}
	return newFunction(b, name, nil), nil
}

// OpShape returns the shape of a computation graph value.
func (b *Builder) OpShape(op compute.Value) (shapes.Shape, error) {
	node, ok := op.(*graphNode)
	if !ok {
		return shapes.Invalid(), errors.Errorf("OpShape: expected *graphNode, got %T", op)
	}
	return node.shape, nil
}

// Compile compiles the computation graph into an executable.
func (b *Builder) Compile() (compute.Executable, error) {
	if b == nil || b.backend == nil {
		return nil, errors.New("Builder is nil")
	}
	if !b.backend.lifecycle.Enter() {
		return nil, errors.Wrapf(compute.ErrBackendFinalized, "backend %q: Builder.Compile(%q)", BackendName, b.name)
	}
	defer b.backend.lifecycle.Leave()
	if b.ctx == nil {
		return nil, errors.New("Builder has been finalized")
	}
	if b.compiled {
		return nil, errors.New("Builder already compiled")
	}
	if b.mainFn == nil {
		return nil, errors.New("Builder has no main function")
	}
	if !b.mainFn.returned {
		return nil, errors.New("Main function has no Return() call")
	}

	b.compiled = true

	// If the main function has a control flow step, use CF-aware compilation.
	if b.mainFn.controlFlowStep != nil {
		return b.compileWithControlFlow()
	}

	return b.compileSimple()
}

// compileTargets filters out zero-sized outputs, wraps any pass-through feed
// tensors with Identity so MPSGraph doesn't treat a placeholder directly as a
// target, and compiles the remaining non-zero targets. It returns the compiled
// executable (nil when all outputs are zero-sized) and the indices into outputs
// corresponding to the executable's targets.
func compileTargets(ctx *bridge.Context, owner *Function, feeds, outputs []*graphNode) (*bridge.Exec, []int, error) {
	nonZeroTargets := make([]*graphNode, 0, len(outputs))
	nonZeroIndices := make([]int, 0, len(outputs))
	for i, out := range outputs {
		if out.tensor == nil || out.shape.Size() == 0 {
			continue
		}
		target := out
		if containsNode(feeds, out) {
			idTensor, err := ctx.Identity(out.tensor)
			if err != nil {
				return nil, nil, errors.Wrap(err, "identity for pass-through output")
			}
			target = &graphNode{tensor: idTensor, shape: out.shape, owner: owner}
		}
		nonZeroTargets = append(nonZeroTargets, target)
		nonZeroIndices = append(nonZeroIndices, i)
	}
	if len(nonZeroTargets) == 0 {
		return nil, nil, nil
	}
	info := buildCompileInfo(feeds, nonZeroTargets)
	exec, err := ctx.Compile(info)
	if err != nil {
		return nil, nil, err
	}
	return exec, nonZeroIndices, nil
}

// compileSimple compiles a function without control flow into a single Executable.
func (b *Builder) compileSimple() (compute.Executable, error) {
	exec, nonZeroIndices, err := compileTargets(b.ctx, b.mainFn, b.mainFn.params, b.mainFn.outputs)
	if err != nil {
		return nil, errors.Wrap(err, "compiling MPSGraph")
	}

	inputNames, inputShapes := collectParamInfo(b.mainFn.params)
	return newExecutable(
		b.backend,
		exec,
		inputNames,
		inputShapes,
		collectOutputShapes(b.mainFn.outputs),
		nonZeroIndices,
	), nil
}

// compileWithControlFlow compiles a function containing a control flow operation.
// The pre-CF graph is compiled into one Exec, and closures are compiled into
// separate Execs. The resulting ExecutableWithCF orchestrates execution.
func (b *Builder) compileWithControlFlow() (compute.Executable, error) {
	cf := b.mainFn.controlFlowStep

	// Gather all nodes that need to be outputs of the pre-CF graph:
	// 1. CF inputs (initial state, predicate, etc.)
	// 2. Captured values from closures
	// 3. Any Return outputs that are NOT CF results (direct pre-CF values)
	preGraphTargets := make([]*graphNode, 0)
	preGraphTargets = append(preGraphTargets, cf.inputs...)

	// Gather all captured parent nodes from closures.
	allClosureFns := b.gatherClosureFunctions(cf)
	for _, closureFn := range allClosureFns {
		for _, captured := range closureFn.capturedParentNodes {
			// Only add if it's from the main function and not already in targets.
			if captured.owner == b.mainFn && !containsNode(preGraphTargets, captured) {
				preGraphTargets = append(preGraphTargets, captured)
			}
		}
	}

	// Check if any Return outputs are direct pre-CF values (not CF results).
	for _, out := range b.mainFn.outputs {
		if !containsNode(cf.outputNodes, out) && out.tensor != nil {
			if !containsNode(preGraphTargets, out) {
				preGraphTargets = append(preGraphTargets, out)
			}
		}
	}

	// Compile the pre-CF graph.
	var preExec *bridge.Exec
	if len(preGraphTargets) > 0 && preGraphTargets[0].tensor != nil {
		info := buildCompileInfo(b.mainFn.params, preGraphTargets)
		var err error
		preExec, err = b.ctx.Compile(info)
		if err != nil {
			return nil, errors.Wrap(err, "compiling pre-CF graph")
		}
	}

	// Compile closures.
	closureExecs := make(map[*Function]*Executable)
	for _, fn := range allClosureFns {
		exec, err := fn.compileClosure()
		if err != nil {
			return nil, errors.Wrapf(err, "compiling closure %q", fn.name)
		}
		closureExecs[fn] = exec
	}

	// Build output mapping: for each Return output, record where it comes from.
	outputMapping := make([]outputSource, len(b.mainFn.outputs))
	for i, out := range b.mainFn.outputs {
		// Check if this output is a CF result.
		cfIdx := indexOfNode(cf.outputNodes, out)
		if cfIdx >= 0 {
			outputMapping[i] = outputSource{fromCF: true, cfIndex: cfIdx}
			continue
		}
		// Otherwise it's from the pre-CF graph.
		preIdx := indexOfNode(preGraphTargets, out)
		if preIdx >= 0 {
			outputMapping[i] = outputSource{fromCF: false, preIndex: preIdx}
		}
	}

	inputNames, inputShapes := collectParamInfo(b.mainFn.params)

	cfExec := &ExecutableWithCF{
		backend:         b.backend,
		preExec:         preExec,
		preGraphTargets: preGraphTargets,
		cfStep:          cf,
		closureExecs:    closureExecs,
		closureFns:      allClosureFns,
		outputMapping:   outputMapping,
		inputNames:      inputNames,
		inputShapes:     inputShapes,
		outputShapes:    collectOutputShapes(b.mainFn.outputs),
	}
	cfExec.untrack = b.backend.cfExecTracker.Track(cfExec).Untrack
	return cfExec, nil
}

// buildCompileInfo creates a CompileInfo from parameter nodes and target nodes.
func buildCompileInfo(params, targets []*graphNode) bridge.CompileInfo {
	info := bridge.CompileInfo{
		Feeds:      make([]bridge.Tensor, len(params)),
		FeedDtypes: make([]int, len(params)),
		FeedShapes: make([][]int64, len(params)),
		Targets:    make([]bridge.Tensor, len(targets)),
	}
	for i, p := range params {
		info.Feeds[i] = p.tensor
		info.FeedDtypes[i] = dtypeToBridgeDType(p.shape.DType)
		dims := p.shape.Dimensions
		info.FeedShapes[i] = make([]int64, len(dims))
		for j, d := range dims {
			info.FeedShapes[i][j] = int64(d)
		}
	}
	for i, t := range targets {
		info.Targets[i] = t.tensor
	}
	return info
}

// collectParamInfo extracts names and shapes from parameter nodes.
func collectParamInfo(params []*graphNode) ([]string, []shapes.Shape) {
	names := make([]string, len(params))
	paramShapes := make([]shapes.Shape, len(params))
	for i, p := range params {
		names[i] = p.name
		paramShapes[i] = p.shape
	}
	return names, paramShapes
}

// collectOutputShapes extracts shapes from output nodes.
func collectOutputShapes(outputs []*graphNode) []shapes.Shape {
	outShapes := make([]shapes.Shape, len(outputs))
	for i, out := range outputs {
		outShapes[i] = out.shape
	}
	return outShapes
}

// gatherClosureFunctions collects all closure Functions used by a CF step.
func (b *Builder) gatherClosureFunctions(cf *controlFlowStep) []*Function {
	var fns []*Function
	switch cf.opType {
	case compute.OpTypeWhile:
		fns = append(fns, cf.whileData.condFn, cf.whileData.bodyFn)
	case compute.OpTypeIf:
		fns = append(fns, cf.ifData.trueFn, cf.ifData.falseFn)
	case compute.OpTypeSort:
		fns = append(fns, cf.sortData.comparatorFn)
	case compute.OpTypeCall:
		fns = append(fns, cf.callData.targetFn)
	}
	return fns
}

// containsNode checks if a node is in a slice.
func containsNode(nodes []*graphNode, n *graphNode) bool {
	for _, node := range nodes {
		if node == n {
			return true
		}
	}
	return false
}

// indexOfNode returns the index of a node in a slice, or -1.
func indexOfNode(nodes []*graphNode, n *graphNode) int {
	for i, node := range nodes {
		if node == n {
			return i
		}
	}
	return -1
}
