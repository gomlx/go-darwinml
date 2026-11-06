// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0
//
// MLX bridge for GoMLX - C-compatible API wrapping Apple's MLX C++ framework.

#ifndef MLX_BRIDGE_H
#define MLX_BRIDGE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle types
typedef void* MLXArrayHandle; // mlx::core::array* (heap-allocated)
typedef void* MLXTapeHandle;  // Recorded computation tape (MLXTape*)
typedef void* MLXExecHandle;  // Compiled or replayable executable (MLXExec*)

// Error reporting. Caller must free(error->message) if non-NULL.
typedef struct {
    int code;
    const char* message;
} MLXError;

// Data types matching GoMLX dtypes.
typedef enum {
    MLX_DTYPE_BOOL      = 0,
    MLX_DTYPE_INT8      = 1,
    MLX_DTYPE_INT16     = 2,
    MLX_DTYPE_INT32     = 3,
    MLX_DTYPE_INT64     = 4,
    MLX_DTYPE_FLOAT16   = 5,
    MLX_DTYPE_BFLOAT16  = 6,
    MLX_DTYPE_FLOAT32   = 7,
    MLX_DTYPE_FLOAT64   = 8,
    MLX_DTYPE_UINT8     = 9,
    MLX_DTYPE_UINT16    = 10,
    MLX_DTYPE_UINT32    = 11,
    MLX_DTYPE_UINT64    = 12,
    MLX_DTYPE_COMPLEX64 = 13,
} MLXDType;

// --- Initialization & Device Info ---

// Initializes the MLX bridge, dynamically verifying that libmlx.dylib is installed
// and that the Metal GPU backend is available.
bool mlx_bridge_init(MLXError* error);

// Returns the Metal GPU device name (e.g. "Apple M3 Max"). Caller must free().
const char* mlx_device_name(MLXError* error);

// Returns the MLX version string (e.g. "0.32.4"). Caller must free().
const char* mlx_version_string(MLXError* error);

// --- Buffer / Array Management ---

// Allocates an uninitialized unified-memory MLX array of the given shape and dtype.
MLXArrayHandle mlx_array_create(int64_t nbytes, int dtype,
    const int32_t* shape, int rank, MLXError* error);

// Creates an MLX array initialized with a copy of data (nbytes) in unified memory.
MLXArrayHandle mlx_array_from_data(const void* data, int64_t nbytes, int dtype,
    const int32_t* shape, int rank, MLXError* error);

// Returns the shared CPU/GPU unified memory pointer for an evaluated array.
void* mlx_array_data_ptr(MLXArrayHandle arr);

// Returns the byte size of the array.
int64_t mlx_array_nbytes(MLXArrayHandle arr);

// Returns the bridge MLXDType of the array.
int mlx_array_dtype(MLXArrayHandle arr);

// Returns the rank (ndim) of the array.
int mlx_array_ndim(MLXArrayHandle arr);

// Copies the array's shape dimensions into out_shape (must have room for ndim int32_t elements).
void mlx_array_shape(MLXArrayHandle arr, int32_t* out_shape);

// Returns true if the array is row-contiguous and its underlying buffer size matches its logical element count.
bool mlx_array_is_row_contiguous(MLXArrayHandle arr);

// Ensures the array is evaluated and row-contiguous in-place so mlx_array_data_ptr() is flat and valid.
bool mlx_array_ensure_contiguous(MLXArrayHandle arr, MLXError* error);

// Destroys an MLXArrayHandle and releases its reference to the underlying unified memory buffer.
void mlx_array_destroy(MLXArrayHandle arr);

// --- Tape Construction & Execution ---

MLXTapeHandle mlx_tape_create(MLXError* error);
void mlx_tape_destroy(MLXTapeHandle tape);

// Adds an input parameter placeholder to the tape and returns its node ID (>= 0), or -1 on error.
int32_t mlx_tape_add_parameter(MLXTapeHandle tape, int dtype,
    const int32_t* shape, int rank, MLXError* error);

// Materializes a constant array in unified memory owned by the tape and returns its node ID (>= 0), or -1 on error.
int32_t mlx_tape_add_constant(MLXTapeHandle tape, const void* data, int64_t nbytes,
    int dtype, const int32_t* shape, int rank, MLXError* error);

// Compiles a tape into an executable.
// If compile_graph is true, traces and compiles the tape with mlx::core::compile.
// If eager_contiguous is true, appends mlx::core::contiguous() to every output before eval().
MLXExecHandle mlx_tape_compile(MLXTapeHandle tape,
    const int32_t* output_ids, int num_outputs,
    bool compile_graph, bool eager_contiguous, MLXError* error);

void mlx_exec_destroy(MLXExecHandle exec);

// Executes a compiled or dynamic tape.
// For any input i where donate[i] is true, inputs[i] is deleted AFTER constructing the lazy output
// graph and BEFORE calling mlx::core::eval(outputs), enabling MLX's in-place buffer reuse.
bool mlx_exec_execute(MLXExecHandle exec,
    MLXArrayHandle* inputs, const bool* donate, int num_inputs,
    MLXArrayHandle* outputs, int num_outputs, MLXError* error);

#ifdef __cplusplus
}
#endif

#endif // MLX_BRIDGE_H
