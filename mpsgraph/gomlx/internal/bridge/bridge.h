// MPSGraph bridge for GoMLX - C-compatible API wrapping Apple's MPSGraph framework.
// Generated for go-coreml MPSGraph backend.

#ifndef MPSGRAPH_BRIDGE_H
#define MPSGRAPH_BRIDGE_H

#include <stdint.h>
#include <stdbool.h>

// Opaque handle types
typedef void* MPSGraphContextHandle;  // MPSGraph + MTLDevice + MTLCommandQueue
typedef void* MPSGraphTensorHandle;   // MPSGraphTensor* (non-owning, graph retains)
typedef void* MPSGraphExecHandle;     // Compiled MPSGraphExecutable wrapper
typedef void* MTLBufferHandle;        // id<MTLBuffer> with shared storage

// Error reporting. Caller must free(error->message) if non-NULL.
typedef struct {
    int code;
    const char* message;
} MPSGraphError;

// Data types matching GoMLX dtypes.
typedef enum {
    MPSGRAPH_DTYPE_BOOL    = 0,
    MPSGRAPH_DTYPE_INT8    = 1,
    MPSGRAPH_DTYPE_INT16   = 2,
    MPSGRAPH_DTYPE_INT32   = 3,
    MPSGRAPH_DTYPE_INT64   = 4,
    MPSGRAPH_DTYPE_FLOAT16 = 5,
    MPSGRAPH_DTYPE_BFLOAT16 = 6,
    MPSGRAPH_DTYPE_FLOAT32 = 7,
    MPSGRAPH_DTYPE_FLOAT64 = 8,
    MPSGRAPH_DTYPE_UINT8   = 9,
    MPSGRAPH_DTYPE_UINT16  = 10,
    MPSGRAPH_DTYPE_UINT32  = 11,
    MPSGRAPH_DTYPE_UINT64  = 12,
} MPSGraphDType;

// Reduction types.
typedef enum {
    MPSGRAPH_REDUCE_SUM     = 0,
    MPSGRAPH_REDUCE_PRODUCT = 1,
    MPSGRAPH_REDUCE_MAX     = 2,
    MPSGRAPH_REDUCE_MIN     = 3,
} MPSGraphReduceType;

// --- Context Lifecycle ---
MPSGraphContextHandle mpsgraph_create_context(MPSGraphError* error);
MPSGraphContextHandle mpsgraph_create_context_with_device(void* deviceHandle, MPSGraphError* error);
void* mpsgraph_device_handle(MPSGraphContextHandle ctx);
void mpsgraph_destroy_context(MPSGraphContextHandle ctx);
const char* mpsgraph_device_name(MPSGraphContextHandle ctx);  // Caller must free()

// --- Tensor Creation ---
MPSGraphTensorHandle mpsgraph_placeholder(MPSGraphContextHandle ctx, int dtype,
    int64_t* shape, int rank, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_constant(MPSGraphContextHandle ctx, void* data,
    int64_t nbytes, int dtype, int64_t* shape, int rank, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_iota(MPSGraphContextHandle ctx, int dtype,
    int64_t* shape, int rank, int axis, MPSGraphError* error);

// --- Unary Operations ---
MPSGraphTensorHandle mpsgraph_abs(MPSGraphContextHandle ctx, MPSGraphTensorHandle x, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_neg(MPSGraphContextHandle ctx, MPSGraphTensorHandle x, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_sqrt(MPSGraphContextHandle ctx, MPSGraphTensorHandle x, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_rsqrt(MPSGraphContextHandle ctx, MPSGraphTensorHandle x, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_exp(MPSGraphContextHandle ctx, MPSGraphTensorHandle x, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_expm1(MPSGraphContextHandle ctx, MPSGraphTensorHandle x, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_log(MPSGraphContextHandle ctx, MPSGraphTensorHandle x, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_log1p(MPSGraphContextHandle ctx, MPSGraphTensorHandle x, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_sin(MPSGraphContextHandle ctx, MPSGraphTensorHandle x, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_cos(MPSGraphContextHandle ctx, MPSGraphTensorHandle x, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_tanh(MPSGraphContextHandle ctx, MPSGraphTensorHandle x, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_sigmoid(MPSGraphContextHandle ctx, MPSGraphTensorHandle x, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_erf(MPSGraphContextHandle ctx, MPSGraphTensorHandle x, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_floor(MPSGraphContextHandle ctx, MPSGraphTensorHandle x, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_ceil(MPSGraphContextHandle ctx, MPSGraphTensorHandle x, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_round(MPSGraphContextHandle ctx, MPSGraphTensorHandle x, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_sign(MPSGraphContextHandle ctx, MPSGraphTensorHandle x, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_logical_not(MPSGraphContextHandle ctx, MPSGraphTensorHandle x, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_bitwise_not(MPSGraphContextHandle ctx, MPSGraphTensorHandle x, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_is_finite(MPSGraphContextHandle ctx, MPSGraphTensorHandle x, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_is_nan(MPSGraphContextHandle ctx, MPSGraphTensorHandle x, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_identity(MPSGraphContextHandle ctx, MPSGraphTensorHandle x, MPSGraphError* error);

// --- Binary Operations ---
MPSGraphTensorHandle mpsgraph_add(MPSGraphContextHandle ctx, MPSGraphTensorHandle lhs, MPSGraphTensorHandle rhs, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_sub(MPSGraphContextHandle ctx, MPSGraphTensorHandle lhs, MPSGraphTensorHandle rhs, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_mul(MPSGraphContextHandle ctx, MPSGraphTensorHandle lhs, MPSGraphTensorHandle rhs, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_div(MPSGraphContextHandle ctx, MPSGraphTensorHandle lhs, MPSGraphTensorHandle rhs, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_rem(MPSGraphContextHandle ctx, MPSGraphTensorHandle lhs, MPSGraphTensorHandle rhs, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_pow(MPSGraphContextHandle ctx, MPSGraphTensorHandle lhs, MPSGraphTensorHandle rhs, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_max(MPSGraphContextHandle ctx, MPSGraphTensorHandle lhs, MPSGraphTensorHandle rhs, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_min(MPSGraphContextHandle ctx, MPSGraphTensorHandle lhs, MPSGraphTensorHandle rhs, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_atan2(MPSGraphContextHandle ctx, MPSGraphTensorHandle lhs, MPSGraphTensorHandle rhs, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_logical_and(MPSGraphContextHandle ctx, MPSGraphTensorHandle lhs, MPSGraphTensorHandle rhs, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_logical_or(MPSGraphContextHandle ctx, MPSGraphTensorHandle lhs, MPSGraphTensorHandle rhs, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_logical_xor(MPSGraphContextHandle ctx, MPSGraphTensorHandle lhs, MPSGraphTensorHandle rhs, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_bitwise_and(MPSGraphContextHandle ctx, MPSGraphTensorHandle lhs, MPSGraphTensorHandle rhs, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_bitwise_or(MPSGraphContextHandle ctx, MPSGraphTensorHandle lhs, MPSGraphTensorHandle rhs, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_bitwise_xor(MPSGraphContextHandle ctx, MPSGraphTensorHandle lhs, MPSGraphTensorHandle rhs, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_shift_left(MPSGraphContextHandle ctx, MPSGraphTensorHandle lhs, MPSGraphTensorHandle rhs, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_shift_right(MPSGraphContextHandle ctx, MPSGraphTensorHandle lhs, MPSGraphTensorHandle rhs, MPSGraphError* error);

// --- Comparison Operations ---
MPSGraphTensorHandle mpsgraph_equal(MPSGraphContextHandle ctx, MPSGraphTensorHandle lhs, MPSGraphTensorHandle rhs, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_not_equal(MPSGraphContextHandle ctx, MPSGraphTensorHandle lhs, MPSGraphTensorHandle rhs, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_less_than(MPSGraphContextHandle ctx, MPSGraphTensorHandle lhs, MPSGraphTensorHandle rhs, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_less_or_equal(MPSGraphContextHandle ctx, MPSGraphTensorHandle lhs, MPSGraphTensorHandle rhs, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_greater_than(MPSGraphContextHandle ctx, MPSGraphTensorHandle lhs, MPSGraphTensorHandle rhs, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_greater_or_equal(MPSGraphContextHandle ctx, MPSGraphTensorHandle lhs, MPSGraphTensorHandle rhs, MPSGraphError* error);

// --- Shape Operations ---
MPSGraphTensorHandle mpsgraph_reshape(MPSGraphContextHandle ctx, MPSGraphTensorHandle x,
    int64_t* shape, int rank, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_transpose(MPSGraphContextHandle ctx, MPSGraphTensorHandle x,
    int* permutation, int rank, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_cast(MPSGraphContextHandle ctx, MPSGraphTensorHandle x,
    int dtype, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_broadcast_to(MPSGraphContextHandle ctx, MPSGraphTensorHandle x,
    int64_t* shape, int rank, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_slice(MPSGraphContextHandle ctx, MPSGraphTensorHandle x,
    int64_t* starts, int64_t* ends, int64_t* strides, int rank, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_concatenate(MPSGraphContextHandle ctx,
    MPSGraphTensorHandle* tensors, int numTensors, int axis, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_reverse(MPSGraphContextHandle ctx, MPSGraphTensorHandle x,
    int* axes, int numAxes, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_pad(MPSGraphContextHandle ctx, MPSGraphTensorHandle x,
    MPSGraphTensorHandle padValue, int64_t* padBefore, int64_t* padAfter, int rank,
    MPSGraphError* error);

// --- Ternary / Selection ---
MPSGraphTensorHandle mpsgraph_where(MPSGraphContextHandle ctx, MPSGraphTensorHandle cond,
    MPSGraphTensorHandle onTrue, MPSGraphTensorHandle onFalse, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_clamp(MPSGraphContextHandle ctx, MPSGraphTensorHandle minVal,
    MPSGraphTensorHandle x, MPSGraphTensorHandle maxVal, MPSGraphError* error);

// --- Matrix Operations ---
MPSGraphTensorHandle mpsgraph_matmul(MPSGraphContextHandle ctx,
    MPSGraphTensorHandle lhs, MPSGraphTensorHandle rhs, MPSGraphError* error);

// --- Reduction Operations ---
MPSGraphTensorHandle mpsgraph_reduce(MPSGraphContextHandle ctx, MPSGraphTensorHandle x,
    int reduceType, int* axes, int numAxes, MPSGraphError* error);

// --- Gather/Scatter ---
MPSGraphTensorHandle mpsgraph_gather_nd(MPSGraphContextHandle ctx,
    MPSGraphTensorHandle params, MPSGraphTensorHandle indices,
    int batchDims, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_gather_along_axis(MPSGraphContextHandle ctx,
    MPSGraphTensorHandle x, MPSGraphTensorHandle indices, int axis,
    MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_scatter_nd(MPSGraphContextHandle ctx,
    MPSGraphTensorHandle data, MPSGraphTensorHandle indices, MPSGraphTensorHandle updates,
    int64_t* shape, int rank, int mode, MPSGraphError* error);

// --- Compilation & Execution ---
// Compile graph into executable. feeds/targets are arrays of tensor handles.
MPSGraphExecHandle mpsgraph_compile(MPSGraphContextHandle ctx,
    MPSGraphTensorHandle* feeds, int* feedDtypes, int64_t** feedShapes, int* feedRanks, int numFeeds,
    MPSGraphTensorHandle* targets, int numTargets,
    MPSGraphError* error);
void mpsgraph_destroy_exec(MPSGraphExecHandle exec);

// Execute compiled graph. inputData/outputData are raw byte pointers.
// inputSizes/outputSizes are in bytes. Returns output data in pre-allocated outputData buffers.
bool mpsgraph_execute(MPSGraphExecHandle exec,
    void** inputData, int64_t* inputSizes, int* inputDtypes,
    int64_t** inputShapes, int* inputRanks, int numInputs,
    void** outputData, int64_t* outputSizes, int* outputDtypes,
    int64_t** outputShapes, int* outputRanks, int numOutputs,
    MPSGraphError* error);

// --- Buffer Management ---
MTLBufferHandle mpsgraph_buffer_create(MPSGraphContextHandle ctx, int64_t nbytes, MPSGraphError* error);
void* mpsgraph_buffer_contents(MTLBufferHandle buffer);
int64_t mpsgraph_buffer_length(MTLBufferHandle buffer);
void mpsgraph_buffer_destroy(MTLBufferHandle buffer);

// --- Dynamic Slice/Update ---
MPSGraphTensorHandle mpsgraph_dynamic_slice(MPSGraphContextHandle ctx,
    MPSGraphTensorHandle x, MPSGraphTensorHandle* startIndices, int numIndices,
    int64_t* sliceSizes, int rank, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_dynamic_update_slice(MPSGraphContextHandle ctx,
    MPSGraphTensorHandle x, MPSGraphTensorHandle update,
    MPSGraphTensorHandle* startIndices, int numIndices, MPSGraphError* error);

// --- Batch Normalization ---
MPSGraphTensorHandle mpsgraph_batch_norm_inference(MPSGraphContextHandle ctx,
    MPSGraphTensorHandle input, MPSGraphTensorHandle mean, MPSGraphTensorHandle variance,
    MPSGraphTensorHandle gamma, MPSGraphTensorHandle beta,
    float epsilon, int featureAxis, MPSGraphError* error);

// --- ArgMin/ArgMax ---
MPSGraphTensorHandle mpsgraph_argmin(MPSGraphContextHandle ctx, MPSGraphTensorHandle x,
    int axis, int outputDtype, MPSGraphError* error);
MPSGraphTensorHandle mpsgraph_argmax(MPSGraphContextHandle ctx, MPSGraphTensorHandle x,
    int axis, int outputDtype, MPSGraphError* error);

// --- Softmax ---
MPSGraphTensorHandle mpsgraph_softmax(MPSGraphContextHandle ctx,
    MPSGraphTensorHandle x, int axis, MPSGraphError* error);

// --- Random Number Generation ---
MPSGraphTensorHandle mpsgraph_random_uniform(MPSGraphContextHandle ctx,
    int dtype, int64_t* shape, int rank, MPSGraphError* error);

// --- Pooling (ReduceWindow) ---
// mode: 0=max, 1=avg
MPSGraphTensorHandle mpsgraph_pool2d(MPSGraphContextHandle ctx,
    MPSGraphTensorHandle x, int mode,
    int64_t* windowDims, int64_t* strides, int64_t* padBefore, int64_t* padAfter,
    MPSGraphError* error);

// --- Max Pool 2D Gradient (SelectAndScatter for MaxPool backprop) ---
// gradient: incoming gradient (same shape as maxpool output)
// source: original input to maxpool (operand in XLA terms)
MPSGraphTensorHandle mpsgraph_max_pool2d_gradient(MPSGraphContextHandle ctx,
    MPSGraphTensorHandle gradient, MPSGraphTensorHandle source,
    int64_t* windowDims, int64_t* strides, int64_t* padBefore, int64_t* padAfter,
    MPSGraphError* error);

// --- General Convolution with axis transposition ---
MPSGraphTensorHandle mpsgraph_conv_general(MPSGraphContextHandle ctx,
    MPSGraphTensorHandle input, MPSGraphTensorHandle kernel,
    int numSpatialDims,
    int64_t* strides, int64_t* dilations,
    int64_t* padBefore, int64_t* padAfter,
    int groups, MPSGraphError* error);

// --- Scatter with reduction modes ---
// mode: 0=set, 1=add, 2=max, 3=min
MPSGraphTensorHandle mpsgraph_scatter_along_axis(MPSGraphContextHandle ctx,
    MPSGraphTensorHandle data, MPSGraphTensorHandle indices, MPSGraphTensorHandle updates,
    int axis, int mode, MPSGraphError* error);

#endif // MPSGRAPH_BRIDGE_H
