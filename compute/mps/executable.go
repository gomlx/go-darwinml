// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mps

import (
	"runtime"
	"sort"
	"sync"
	"unsafe"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/go-darwinml/internal/bridge"
	"github.com/pkg/errors"
)

// pingPongPool holds donated MTLBuffers of sizes that match the Executable's outputs
// so that the next execution of this Executable (or another Executable on the same Backend)
// can reuse them as output buffers without allocating new Metal buffers.
// Access to pingPongPool fields is guarded by Backend.poolMu.
type pingPongPool struct {
	buffers      map[int64][]*bridge.Buffer
	lastUsedStep uint64
}

func (p *pingPongPool) destroyAllLocked() {
	if p == nil {
		return
	}
	for size, list := range p.buffers {
		for _, buf := range list {
			buf.Destroy()
		}
		delete(p.buffers, size)
	}
}

// Executable implements compute.Executable for the MPS backend.
type Executable struct {
	backend      *Backend
	exec         *bridge.Exec
	inputNames   []string
	inputShapes  []shapes.Shape
	outputShapes []shapes.Shape
	pool         *pingPongPool
	cleanup      runtime.Cleanup
	mu           sync.Mutex // Serialize execution for safety.
}

// Verify interface compliance.
var _ compute.Executable = &Executable{}

// newExecutable constructs an Executable with a mixed-scope ping-pong buffer pool and GC cleanup.
func newExecutable(backend *Backend, exec *bridge.Exec, inputNames []string, inputShapes, outputShapes []shapes.Shape) *Executable {
	pool := &pingPongPool{
		buffers: make(map[int64][]*bridge.Buffer),
	}
	backend.registerPool(pool)
	e := &Executable{
		backend:      backend,
		exec:         exec,
		inputNames:   inputNames,
		inputShapes:  inputShapes,
		outputShapes: outputShapes,
		pool:         pool,
	}
	e.cleanup = runtime.AddCleanup(e, backend.unregisterAndDestroyPool, pool)
	return e
}

// Finalize releases the compiled executable and any pooled ping-pong buffers.
func (e *Executable) Finalize() {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.cleanup.Stop()
	if e.pool != nil {
		e.backend.unregisterAndDestroyPool(e.pool)
		e.pool = nil
	}
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

// Execute runs the compiled graph with the given input buffers.
func (e *Executable) Execute(inputs []compute.Buffer, donate []bool, defaultDevice compute.DeviceNum) ([]compute.Buffer, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if len(inputs) != len(e.inputShapes) {
		return nil, errors.Errorf("Execute: expected %d inputs, got %d", len(e.inputShapes), len(inputs))
	}

	// Prepare input data.
	execInputs := make([]bridge.ExecInput, len(inputs))
	for i, input := range inputs {
		buf, ok := input.(*gpuBuffer)
		if !ok {
			return nil, errors.Errorf("Execute: input #%d is not a *gpuBuffer, got %T", i, input)
		}
		if !buf.shape.Equal(e.inputShapes[i]) {
			return nil, errors.Errorf("Execute: input #%d shape mismatch: expected %s, got %s",
				i, e.inputShapes[i], buf.shape)
		}
		ptr, nbytes := buf.flatDataPtr()
		dims := make([]int64, buf.shape.Rank())
		for j, d := range buf.shape.Dimensions {
			dims[j] = int64(d)
		}
		execInputs[i] = bridge.ExecInput{
			Buffer: buf.mtlBuf,
			Data:   ptr,
			Size:   nbytes,
			DType:  dtypeToBridgeDType(buf.shape.DType),
			Shape:  dims,
		}
	}

	// Prepare output buffers using mixed-scope ping-pong pooling (local pool first, then
	// cross-executable stealing, then fresh allocation) and evicting unused local buffers.
	outputBuffers := e.backend.acquireOutputBuffers(e.pool, e.outputShapes)
	execOutputs := make([]bridge.ExecOutput, len(e.outputShapes))
	for i, outShape := range e.outputShapes {
		buf := outputBuffers[i]
		ptr, nbytes := buf.flatDataPtr()
		dims := make([]int64, outShape.Rank())
		for j, d := range outShape.Dimensions {
			dims[j] = int64(d)
		}
		execOutputs[i] = bridge.ExecOutput{
			Buffer: buf.mtlBuf,
			Data:   ptr,
			Size:   nbytes,
			DType:  dtypeToBridgeDType(outShape.DType),
			Shape:  dims,
		}
	}

	// Execute.
	if e.exec != nil {
		if err := e.exec.Execute(execInputs, execOutputs); err != nil {
			for _, outBuf := range outputBuffers {
				_ = outBuf.Finalize()
			}
			return nil, errors.Wrap(err, "Execute")
		}
	}

	// Count how many output buffers of each byte size were produced by this execution
	// (Conditions 2 & 2.a: only pool donated inputs that match an output size, and at most
	// as many as there are outputs of that size per execution).
	var availPoolSlots map[int64]int
	if e.pool != nil && !e.backend.IsFinalized() && len(donate) > 0 {
		for _, outShape := range e.outputShapes {
			if nbytes := outShape.ByteSize(); nbytes > 0 {
				if availPoolSlots == nil {
					availPoolSlots = make(map[int64]int, len(e.outputShapes))
				}
				availPoolSlots[nbytes]++
			}
		}
	}

	// Finalize donated inputs if requested (Condition 1: only donated inputs can be pooled).
	var donatedBySize map[int64][]*bridge.Buffer
	for i, d := range donate {
		if !d || inputs[i] == nil {
			continue
		}
		inBuf, ok := inputs[i].(*gpuBuffer)
		if !ok {
			_ = inputs[i].Finalize()
			continue
		}
		nbytes := inBuf.shape.ByteSize()
		if inBuf.valid && inBuf.mtlBuf != nil && nbytes > 0 && availPoolSlots[nbytes] > 0 {
			availPoolSlots[nbytes]--
			// Detach mtlBuf from inBuf so inBuf.Finalize() marks inBuf invalid without
			// destroying the recycled Metal buffer.
			inBuf.cleanup.Stop()
			mtlBuf := inBuf.mtlBuf
			inBuf.mtlBuf = nil
			if donatedBySize == nil {
				donatedBySize = make(map[int64][]*bridge.Buffer)
			}
			donatedBySize[nbytes] = append(donatedBySize[nbytes], mtlBuf)
		}
		_ = inBuf.Finalize()
	}
	if len(donatedBySize) > 0 {
		e.backend.returnDonatedBuffers(e.pool, donatedBySize)
	}

	// Convert to compute.Buffer interface.
	result := make([]compute.Buffer, len(outputBuffers))
	for i, buf := range outputBuffers {
		result[i] = buf
	}
	return result, nil
}

// ===========================================================================
// Control Flow Executable
// ===========================================================================

// outputSource describes where an output comes from.
type outputSource struct {
	fromCF   bool // true: from CF result; false: from pre-CF graph
	cfIndex  int  // index into CF results (when fromCF)
	preIndex int  // index into preGraphTargets (when !fromCF)
}

// ExecutableWithCF handles functions containing a control flow operation.
// It executes: pre-CF graph → control flow (CPU-orchestrated) → assemble outputs.
type ExecutableWithCF struct {
	backend         *Backend
	preExec         *bridge.Exec // Pre-CF graph executable (may be nil if CF inputs are all params)
	preGraphTargets []*graphNode // Nodes that are outputs of pre-CF graph
	cfStep          *controlFlowStep
	closureExecs    map[*Function]*Executable // Compiled closures
	closureFns      []*Function               // All closure functions used
	outputMapping   []outputSource            // Where each output comes from
	inputNames      []string
	inputShapes     []shapes.Shape
	outputShapes    []shapes.Shape
	mu              sync.Mutex
}

var _ compute.Executable = &ExecutableWithCF{}

func (e *ExecutableWithCF) Finalize() {
	if e.preExec != nil {
		e.preExec.Destroy()
		e.preExec = nil
	}
	for _, exec := range e.closureExecs {
		exec.Finalize()
	}
}

func (e *ExecutableWithCF) Inputs() ([]string, []shapes.Shape) {
	return e.inputNames, e.inputShapes
}

func (e *ExecutableWithCF) Outputs() []shapes.Shape {
	return e.outputShapes
}

func (e *ExecutableWithCF) Execute(inputs []compute.Buffer, donate []bool, defaultDevice compute.DeviceNum) ([]compute.Buffer, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if len(inputs) != len(e.inputShapes) {
		return nil, errors.Errorf("ExecuteWithCF: expected %d inputs, got %d", len(e.inputShapes), len(inputs))
	}

	// Step 1: Run the pre-CF graph to produce CF inputs and captured values.
	var preOutputs []compute.Buffer
	if e.preExec != nil {
		preInputs := prepareBridgeInputs(inputs)
		preOutputShapes := make([]shapes.Shape, len(e.preGraphTargets))
		for i, t := range e.preGraphTargets {
			preOutputShapes[i] = t.shape
		}
		var err error
		preOutputs, err = e.backend.executeBridgeExec(e.preExec, preInputs, preOutputShapes)
		if err != nil {
			return nil, errors.Wrap(err, "ExecuteWithCF: pre-CF graph")
		}
	}

	// Step 2: Execute the control flow operation.
	cf := e.cfStep
	var cfResults []compute.Buffer
	var err error

	switch cf.opType {
	case compute.OpTypeWhile:
		cfResults, err = e.execWhile(cf, preOutputs)
	case compute.OpTypeIf:
		cfResults, err = e.execIf(cf, preOutputs)
	case compute.OpTypeSort:
		cfResults, err = e.execSort(cf, preOutputs)
	case compute.OpTypeCall:
		cfResults, err = e.execCall(cf, preOutputs)
	default:
		err = errors.Errorf("unsupported control flow op: %v", cf.opType)
	}
	if err != nil {
		return nil, err
	}

	// Finalize donated inputs if requested.
	for i, d := range donate {
		if d && inputs[i] != nil {
			_ = inputs[i].Finalize()
		}
	}

	// Step 3: Assemble final outputs from CF results and/or pre-CF outputs.
	results := make([]compute.Buffer, len(e.outputMapping))
	for i, src := range e.outputMapping {
		if src.fromCF {
			results[i] = cfResults[src.cfIndex]
		} else {
			results[i] = preOutputs[src.preIndex]
		}
	}

	return results, nil
}

// execWhile executes a While loop with CPU orchestration.
func (e *ExecutableWithCF) execWhile(cf *controlFlowStep, preOutputs []compute.Buffer) ([]compute.Buffer, error) {
	wd := cf.whileData
	condExec := e.closureExecs[wd.condFn]
	bodyExec := e.closureExecs[wd.bodyFn]

	// Initial state = CF inputs from pre-graph outputs.
	stateCount := len(cf.inputs)
	state := make([]compute.Buffer, stateCount)
	for i := range stateCount {
		idx := indexOfNode(e.preGraphTargets, cf.inputs[i])
		if idx < 0 {
			return nil, errors.Errorf("While: CF input %d not found in pre-graph targets", i)
		}
		state[i] = e.backend.cloneBuffer(preOutputs[idx].(*gpuBuffer))
	}

	// Prepare captured values for cond and body closures.
	condCaptures := e.gatherCapturedBuffers(wd.condFn, preOutputs)
	bodyCaptures := e.gatherCapturedBuffers(wd.bodyFn, preOutputs)

	const maxIterations = 1000000
	loopCompleted := false
	for iter := range maxIterations {
		// Run cond with current state + captures.
		condInputs := make([]compute.Buffer, stateCount+len(condCaptures))
		copy(condInputs, state)
		copy(condInputs[stateCount:], condCaptures)
		condDonate := make([]bool, len(condInputs))

		condResults, err := condExec.Execute(condInputs, condDonate, 0)
		if err != nil {
			return nil, errors.Wrapf(err, "While: cond iteration %d", iter)
		}

		// Read the scalar bool result.
		condBuf := condResults[0].(*gpuBuffer)
		condPtr, _ := condBuf.flatDataPtr()
		condValue := *(*bool)(condPtr)
		if !condValue {
			loopCompleted = true
			break // Loop done.
		}

		// Run body with current state + captures.
		bodyInputs := make([]compute.Buffer, stateCount+len(bodyCaptures))
		copy(bodyInputs, state)
		copy(bodyInputs[stateCount:], bodyCaptures)
		bodyDonate := make([]bool, len(bodyInputs))

		newState, err := bodyExec.Execute(bodyInputs, bodyDonate, 0)
		if err != nil {
			return nil, errors.Wrapf(err, "While: body iteration %d", iter)
		}

		// Update state.
		state = newState
	}

	if !loopCompleted {
		return nil, errors.Errorf("While: exceeded maximum iterations (%d)", maxIterations)
	}

	return state, nil
}

// execIf executes an If branch selection.
func (e *ExecutableWithCF) execIf(cf *controlFlowStep, preOutputs []compute.Buffer) ([]compute.Buffer, error) {
	id := cf.ifData

	// Read the predicate.
	predIdx := indexOfNode(e.preGraphTargets, cf.inputs[0])
	if predIdx < 0 {
		return nil, errors.Errorf("If: predicate not found in pre-graph targets")
	}
	predBuf := preOutputs[predIdx].(*gpuBuffer)
	predPtr, _ := predBuf.flatDataPtr()
	predValue := *(*bool)(predPtr)

	// Select branch.
	var branchFn *Function
	if predValue {
		branchFn = id.trueFn
	} else {
		branchFn = id.falseFn
	}
	branchExec := e.closureExecs[branchFn]

	// If branches take no parameters, only captured values.
	captures := e.gatherCapturedBuffers(branchFn, preOutputs)
	donate := make([]bool, len(captures))

	results, err := branchExec.Execute(captures, donate, 0)
	if err != nil {
		return nil, errors.Wrap(err, "If: branch execution")
	}

	return results, nil
}

// execSort sorts tensors using a comparator closure.
func (e *ExecutableWithCF) execSort(cf *controlFlowStep, preOutputs []compute.Buffer) ([]compute.Buffer, error) {
	sd := cf.sortData
	compExec := e.closureExecs[sd.comparatorFn]
	compCaptures := e.gatherCapturedBuffers(sd.comparatorFn, preOutputs)

	// Get input tensors from pre-graph outputs.
	inputCount := len(cf.inputs)
	inputBufs := make([]*gpuBuffer, inputCount)
	for i := range inputCount {
		idx := indexOfNode(e.preGraphTargets, cf.inputs[i])
		if idx < 0 {
			return nil, errors.Errorf("Sort: input %d not found in pre-graph targets", i)
		}
		inputBufs[i] = preOutputs[idx].(*gpuBuffer)
	}

	firstShape := inputBufs[0].shape
	rank := firstShape.Rank()
	axis := sd.axis

	// Compute outer/inner/axis sizes.
	outerSize := 1
	for i := range axis {
		outerSize *= firstShape.Dimensions[i]
	}
	axisSize := firstShape.Dimensions[axis]
	innerSize := 1
	for i := axis + 1; i < rank; i++ {
		innerSize *= firstShape.Dimensions[i]
	}

	// Create output buffers (clones of inputs).
	outputBufs := make([]*gpuBuffer, inputCount)
	for i, buf := range inputBufs {
		outputBufs[i] = e.backend.cloneBuffer(buf)
	}

	// Sort each "row" along the axis.
	for outer := range outerSize {
		for inner := range innerSize {
			// Create index array for this row.
			indices := make([]int, axisSize)
			for k := range axisSize {
				indices[k] = k
			}

			var sortErr error

			sortFn := func(a, b int) bool {
				if sortErr != nil {
					return false // Already errored, just finish quickly.
				}
				// Build comparator inputs: lhs_0, rhs_0, lhs_1, rhs_1, ...
				compInputs := make([]compute.Buffer, 2*inputCount+len(compCaptures))
				for t := range inputCount {
					elemSize := int(inputBufs[t].shape.DType.Size())
					baseOffset := (outer*axisSize*innerSize + inner) * elemSize
					aOffset := baseOffset + indices[a]*innerSize*elemSize
					bOffset := baseOffset + indices[b]*innerSize*elemSize

					outPtr, _ := outputBufs[t].flatDataPtr()
					srcData := unsafe.Pointer(uintptr(outPtr) + uintptr(aOffset))
					lhsBuf := e.backend.newBuffer(shapes.Make(inputBufs[t].shape.DType))
					lhsPtr, _ := lhsBuf.flatDataPtr()
					copyBytes(lhsPtr, srcData, elemSize)
					compInputs[2*t] = lhsBuf

					srcData = unsafe.Pointer(uintptr(outPtr) + uintptr(bOffset))
					rhsBuf := e.backend.newBuffer(shapes.Make(inputBufs[t].shape.DType))
					rhsPtr, _ := rhsBuf.flatDataPtr()
					copyBytes(rhsPtr, srcData, elemSize)
					compInputs[2*t+1] = rhsBuf
				}
				copy(compInputs[2*inputCount:], compCaptures)
				donate := make([]bool, len(compInputs))

				results, err := compExec.Execute(compInputs, donate, 0)
				if err != nil {
					sortErr = err
					return false
				}
				resultBuf := results[0].(*gpuBuffer)
				resPtr, _ := resultBuf.flatDataPtr()
				return *(*bool)(resPtr)
			}

			if sd.isStable {
				sort.SliceStable(indices, sortFn)
			} else {
				sort.Slice(indices, sortFn)
			}

			if sortErr != nil {
				return nil, errors.Wrap(sortErr, "Sort: comparator execution failed")
			}

			// Apply permutation to outputs.
			for t := range inputCount {
				elemSize := int(inputBufs[t].shape.DType.Size())
				stride := innerSize * elemSize
				baseOffset := (outer*axisSize*innerSize + inner) * elemSize

				// Copy original scalars from input buffer.
				inPtr, _ := inputBufs[t].flatDataPtr()
				origData := make([]byte, axisSize*elemSize)
				for k := range axisSize {
					srcOffset := baseOffset + k*stride
					src := unsafe.Pointer(uintptr(inPtr) + uintptr(srcOffset))
					copyBytesToSlice(origData[k*elemSize:(k+1)*elemSize], src, elemSize)
				}

				// Write permuted scalars to output buffer.
				outPtr, _ := outputBufs[t].flatDataPtr()
				for k := range axisSize {
					srcIdx := indices[k]
					dstOffset := baseOffset + k*stride
					dst := unsafe.Pointer(uintptr(outPtr) + uintptr(dstOffset))
					copyBytesFromSlice(dst, origData[srcIdx*elemSize:(srcIdx+1)*elemSize], elemSize)
				}
			}
		}
	}

	results := make([]compute.Buffer, inputCount)
	for i, buf := range outputBufs {
		results[i] = buf
	}
	return results, nil
}

// execCall executes a function call.
func (e *ExecutableWithCF) execCall(cf *controlFlowStep, preOutputs []compute.Buffer) ([]compute.Buffer, error) {
	cd := cf.callData
	targetExec := e.closureExecs[cd.targetFn]

	// Gather inputs from pre-graph outputs.
	callInputs := make([]compute.Buffer, len(cf.inputs))
	for i, input := range cf.inputs {
		idx := indexOfNode(e.preGraphTargets, input)
		if idx < 0 {
			return nil, errors.Errorf("Call: input %d not found in pre-graph targets", i)
		}
		callInputs[i] = preOutputs[idx]
	}

	// Add captured values if the target function has captures.
	captures := e.gatherCapturedBuffers(cd.targetFn, preOutputs)
	allInputs := make([]compute.Buffer, len(callInputs)+len(captures))
	copy(allInputs, callInputs)
	copy(allInputs[len(callInputs):], captures)
	donate := make([]bool, len(allInputs))

	results, err := targetExec.Execute(allInputs, donate, 0)
	if err != nil {
		return nil, errors.Wrap(err, "Call")
	}

	return results, nil
}

// gatherCapturedBuffers collects the captured values for a closure from pre-graph outputs.
func (e *ExecutableWithCF) gatherCapturedBuffers(fn *Function, preOutputs []compute.Buffer) []compute.Buffer {
	captures := make([]compute.Buffer, len(fn.capturedParentNodes))
	for i, captured := range fn.capturedParentNodes {
		idx := indexOfNode(e.preGraphTargets, captured)
		if idx >= 0 {
			captures[i] = preOutputs[idx]
		}
	}
	return captures
}

// ===========================================================================
// Helper functions for buffer operations
// ===========================================================================

// prepareBridgeInputs converts compute.Buffer to bridge.ExecInput.
func prepareBridgeInputs(inputs []compute.Buffer) []bridge.ExecInput {
	execInputs := make([]bridge.ExecInput, len(inputs))
	for i, input := range inputs {
		buf := input.(*gpuBuffer)
		ptr, nbytes := buf.flatDataPtr()
		dims := make([]int64, buf.shape.Rank())
		for j, d := range buf.shape.Dimensions {
			dims[j] = int64(d)
		}
		execInputs[i] = bridge.ExecInput{
			Buffer: buf.mtlBuf,
			Data:   ptr,
			Size:   nbytes,
			DType:  dtypeToBridgeDType(buf.shape.DType),
			Shape:  dims,
		}
	}
	return execInputs
}

// executeBridgeExec runs a bridge.Exec and returns gpuBuffer results.
func (b *Backend) executeBridgeExec(exec *bridge.Exec, inputs []bridge.ExecInput, outputShapes []shapes.Shape) ([]compute.Buffer, error) {
	outputBuffers := make([]*gpuBuffer, len(outputShapes))
	execOutputs := make([]bridge.ExecOutput, len(outputShapes))
	for i, outShape := range outputShapes {
		buf := b.newBuffer(outShape)
		outputBuffers[i] = buf
		ptr, nbytes := buf.flatDataPtr()
		dims := make([]int64, outShape.Rank())
		for j, d := range outShape.Dimensions {
			dims[j] = int64(d)
		}
		execOutputs[i] = bridge.ExecOutput{
			Buffer: buf.mtlBuf,
			Data:   ptr,
			Size:   nbytes,
			DType:  dtypeToBridgeDType(outShape.DType),
			Shape:  dims,
		}
	}

	if err := exec.Execute(inputs, execOutputs); err != nil {
		return nil, err
	}

	results := make([]compute.Buffer, len(outputBuffers))
	for i, buf := range outputBuffers {
		results[i] = buf
	}
	return results, nil
}

// cloneBuffer creates a copy of a gpuBuffer.
func (b *Backend) cloneBuffer(src *gpuBuffer) *gpuBuffer {
	dst := b.newBuffer(src.shape)
	srcPtr, nbytes := src.flatDataPtr()
	if nbytes > 0 {
		dstPtr, _ := dst.flatDataPtr()
		copyBytes(dstPtr, srcPtr, int(nbytes))
	}
	return dst
}

// copyBytes copies n bytes from src to dst.
func copyBytes(dst, src unsafe.Pointer, n int) {
	dstSlice := unsafe.Slice((*byte)(dst), n)
	srcSlice := unsafe.Slice((*byte)(src), n)
	copy(dstSlice, srcSlice)
}

// copyBytesToSlice copies n bytes from src pointer to dst byte slice.
func copyBytesToSlice(dst []byte, src unsafe.Pointer, n int) {
	srcSlice := unsafe.Slice((*byte)(src), n)
	copy(dst, srcSlice)
}

// copyBytesFromSlice copies n bytes from src byte slice to dst pointer.
func copyBytesFromSlice(dst unsafe.Pointer, src []byte, n int) {
	dstSlice := unsafe.Slice((*byte)(dst), n)
	copy(dstSlice, src)
}
