// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mlx

import (
	"runtime"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/notimplemented"
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
	nodes := make([]*graphNode, len(outputs))
	for i, out := range outputs {
		node, ok := out.(*graphNode)
		if !ok {
			return errors.Errorf("Return: output #%d is not a *graphNode, got %T", i, out)
		}
		nodes[i] = node
	}
	f.outputs = nodes
	f.returned = true
	return nil
}

// shapeFromFlat infers the shape (dtype + dimensions) from a flat Go slice.
func shapeFromFlat(flat any, dimensions []int) (shapes.Shape, error) {
	dt, _, err := dtypes.FromAnySlice(flat)
	if err != nil {
		return shapes.Invalid(), err
	}
	return shapes.Make(dt, dimensions...), nil
}
