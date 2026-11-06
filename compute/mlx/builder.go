// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mlx

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/notimplemented"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/go-darwinml/internal/mlxbridge"
	"github.com/pkg/errors"
)

// Builder implements compute.Builder for the MLX backend.
type Builder struct {
	notimplemented.Builder

	backend  *Backend
	name     string
	tape     *mlxbridge.Tape
	mainFn   *Function
	compiled bool
}

// Verify interface compliance.
var _ compute.Builder = (*Builder)(nil)

func newBuilder(backend *Backend, name string, tape *mlxbridge.Tape) *Builder {
	return &Builder{
		backend: backend,
		name:    name,
		tape:    tape,
	}
}

// Name returns the builder name.
func (b *Builder) Name() string { return b.name }

// Finalize releases the builder's tape.
func (b *Builder) Finalize() {
	if b.tape != nil {
		b.tape.Destroy()
		b.tape = nil
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

// Compile compiles the recorded computation tape into an Executable.
func (b *Builder) Compile() (compute.Executable, error) {
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

	outputIDs := make([]mlxbridge.NodeID, len(b.mainFn.outputs))
	outputShapes := make([]shapes.Shape, len(b.mainFn.outputs))
	for i, out := range b.mainFn.outputs {
		outputIDs[i] = out.id
		outputShapes[i] = out.shape
	}

	compileGraph := !b.backend.dynamicShapes
	exec, err := b.tape.Compile(outputIDs, compileGraph, b.backend.eagerContiguous)
	if err != nil {
		return nil, errors.Wrap(err, "compiling MLX tape")
	}

	inputNames := make([]string, len(b.mainFn.params))
	inputShapes := make([]shapes.Shape, len(b.mainFn.params))
	for i, p := range b.mainFn.params {
		inputNames[i] = p.name
		inputShapes[i] = p.shape
	}

	return newExecutable(b.backend, exec, inputNames, inputShapes, outputShapes), nil
}
