// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mlx

import (
	"sync"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/go-darwinml/internal/mlxbridge"
	"github.com/pkg/errors"
)

// Executable implements compute.Executable for the MLX backend.
type Executable struct {
	backend      *Backend
	exec         *mlxbridge.Exec
	inputNames   []string
	inputShapes  []shapes.Shape
	outputShapes []shapes.Shape
	mu           sync.Mutex
}

// Verify interface compliance.
var _ compute.Executable = (*Executable)(nil)

func newExecutable(
	backend *Backend,
	exec *mlxbridge.Exec,
	inputNames []string,
	inputShapes, outputShapes []shapes.Shape,
) *Executable {
	return &Executable{
		backend:      backend,
		exec:         exec,
		inputNames:   inputNames,
		inputShapes:  inputShapes,
		outputShapes: outputShapes,
	}
}

// Finalize releases the compiled MLX executable.
func (e *Executable) Finalize() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.exec != nil {
		e.exec.Destroy()
		e.exec = nil
	}
}

// Inputs returns the parameter names and shapes.
func (e *Executable) Inputs() (names []string, inputShapes []shapes.Shape) {
	return e.inputNames, e.inputShapes
}

// Outputs returns the output shapes.
func (e *Executable) Outputs() (outputShapes []shapes.Shape) {
	return e.outputShapes
}

// Execute runs the compiled or dynamic tape on the given input buffers.
func (e *Executable) Execute(inputs []compute.Buffer, donate []bool, defaultDevice compute.DeviceNum) ([]compute.Buffer, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.exec == nil {
		return nil, errors.New("Execute: executable has been finalized")
	}
	if len(inputs) != len(e.inputShapes) {
		return nil, errors.Errorf("Execute: expected %d inputs, got %d", len(e.inputShapes), len(inputs))
	}

	inBuffers := make([]*Buffer, len(inputs))
	inHandles := make([]mlxbridge.ArrayHandle, len(inputs))
	donateFlags := make([]bool, len(inputs))

	var boundAxes map[string]int
	for i, in := range inputs {
		buf, ok := in.(*Buffer)
		if !ok || buf == nil {
			return nil, errors.Errorf("Execute: input #%d is not a *mlx.Buffer, got %T", i, in)
		}
		bufShape, err := buf.Shape()
		if err != nil {
			return nil, errors.Wrapf(err, "Execute: input #%d", i)
		}
		expected := e.inputShapes[i]
		if bufShape.DType != expected.DType || bufShape.Rank() != expected.Rank() {
			return nil, errors.Errorf("Execute: input #%d shape mismatch: expected %s, got %s",
				i, expected, bufShape)
		}
		for ax, expDim := range expected.Dimensions {
			gotDim := bufShape.Dimensions[ax]
			if expDim != shapes.DynamicDim && gotDim != expDim {
				return nil, errors.Errorf("Execute: input #%d dimension %d mismatch: expected %s, got %s",
					i, ax, expected, bufShape)
			}
			if name := expected.AxisName(ax); name != "" && name != shapes.AnonymousAxis {
				if boundAxes == nil {
					boundAxes = make(map[string]int)
				}
				if prev, exists := boundAxes[name]; exists && prev != gotDim {
					return nil, errors.Errorf("Execute: input #%d dynamic axis %q mismatch: previously bound to %d, got %d",
						i, name, prev, gotDim)
				}
				boundAxes[name] = gotDim
			}
		}
		inBuffers[i] = buf
		if i < len(donate) && donate[i] {
			donateFlags[i] = true
		}
	}

	// Detach donated input handles so C++ can delete them right before mlx::core::eval()
	// without racing with Go GC cleanup.
	for i, buf := range inBuffers {
		if donateFlags[i] {
			inHandles[i] = buf.takeForDonation()
		} else {
			inHandles[i] = buf.handle.load()
		}
	}

	outHandles, err := e.exec.Execute(inHandles, donateFlags, len(e.outputShapes))
	if err != nil {
		// If execution failed before C++ consumed donated handles, clean up any remaining handles.
		for i, d := range donateFlags {
			if d && inHandles[i] != 0 {
				// Note: mlx_exec_execute only deletes inputs[i] right before eval().
			}
		}
		return nil, errors.Wrap(err, "Execute")
	}

	results := make([]compute.Buffer, len(outHandles))
	for i, h := range outHandles {
		outShape := e.outputShapes[i]
		if e.backend.dynamicShapes {
			dims := mlxbridge.ArrayShape(h)
			outShape = shapes.Make(outShape.DType, dims...)
		}
		results[i] = e.backend.newBufferFromHandle(outShape, h)
	}
	return results, nil
}
