// MPSGraph bridge implementation - Objective-C++ wrapping Apple's MPSGraph framework.
// Generated for go-coreml MPSGraph backend.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include "bridge.h"
#include <string.h>

// --- Context object holding MPSGraph + Metal device + command queue ---

@interface MPSGraphContext : NSObject
@property (nonatomic, strong) MPSGraph* graph;
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> commandQueue;
// Track placeholders in insertion order for compilation.
@property (nonatomic, strong) NSMutableArray<MPSGraphTensor*>* placeholders;
@end

@implementation MPSGraphContext
@end

// --- Executable wrapper ---

@interface MPSGraphExecWrapper : NSObject
@property (nonatomic, strong) MPSGraphExecutable* executable;
@property (nonatomic, strong) MPSGraph* graph;  // Original graph for non-compiled execution
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property (nonatomic, strong) NSArray<MPSGraphTensor*>* feedTensors;
@property (nonatomic, strong) NSArray<MPSGraphTensor*>* targetTensors;
// Original feed/target tensors in our order (for non-compiled fallback).
@property (nonatomic, strong) NSArray<MPSGraphTensor*>* origFeedTensors;
@property (nonatomic, strong) NSArray<MPSGraphTensor*>* origTargetTensors;
// Permutation from our feed order to executable's feed order.
@property (nonatomic, strong) NSArray<NSNumber*>* feedPermutation;
@property (nonatomic, strong) NSArray<NSNumber*>* targetPermutation;
@end

@implementation MPSGraphExecWrapper
@end

// --- Helper: set error message ---

static void setError(MPSGraphError* error, int code, NSString* msg) {
    if (error) {
        error->code = code;
        error->message = strdup([msg UTF8String]);
    }
}

static void clearError(MPSGraphError* error) {
    if (error) {
        error->code = 0;
        error->message = NULL;
    }
}

// --- Helper: convert dtype enum to MPSDataType ---

static MPSDataType toMPSDataType(int dtype) {
    switch (dtype) {
        case MPSGRAPH_DTYPE_BOOL:     return MPSDataTypeBool;
        case MPSGRAPH_DTYPE_INT8:     return MPSDataTypeInt8;
        case MPSGRAPH_DTYPE_INT16:    return MPSDataTypeInt16;
        case MPSGRAPH_DTYPE_INT32:    return MPSDataTypeInt32;
        case MPSGRAPH_DTYPE_INT64:    return MPSDataTypeInt64;
        case MPSGRAPH_DTYPE_FLOAT16:  return MPSDataTypeFloat16;
        case MPSGRAPH_DTYPE_BFLOAT16: return MPSDataTypeBFloat16;
        case MPSGRAPH_DTYPE_FLOAT32:  return MPSDataTypeFloat32;
        case MPSGRAPH_DTYPE_FLOAT64:  return MPSDataTypeFloat32; // Float64 not supported; fall back to Float32
        case MPSGRAPH_DTYPE_UINT8:    return MPSDataTypeUInt8;
        case MPSGRAPH_DTYPE_UINT16:   return MPSDataTypeUInt16;
        case MPSGRAPH_DTYPE_UINT32:   return MPSDataTypeUInt32;
        case MPSGRAPH_DTYPE_UINT64:   return MPSDataTypeUInt64;
        default:                      return MPSDataTypeFloat32;
    }
}

// --- Helper: build NSArray<NSNumber*> from int64_t* shape ---

static NSArray<NSNumber*>* shapeArray(int64_t* shape, int rank) {
    NSMutableArray<NSNumber*>* arr = [NSMutableArray arrayWithCapacity:rank];
    for (int i = 0; i < rank; i++) {
        [arr addObject:@(shape[i])];
    }
    return arr;
}

// --- Helper: convert int scatter mode to MPSGraphScatterMode ---

static bool toScatterMode(int mode, MPSGraphScatterMode* out) {
    switch (mode) {
        case 0: *out = MPSGraphScatterModeSet; return true;
        case 1: *out = MPSGraphScatterModeAdd; return true;
        case 2: *out = MPSGraphScatterModeMax; return true;
        case 3: *out = MPSGraphScatterModeMin; return true;
        default: return false;
    }
}

// --- Helper: build NSArray<NSNumber*> from int* array ---

static NSArray<NSNumber*>* intArray(int* values, int count) {
    NSMutableArray<NSNumber*>* arr = [NSMutableArray arrayWithCapacity:count];
    for (int i = 0; i < count; i++) {
        [arr addObject:@(values[i])];
    }
    return arr;
}

// ===========================================================================
// Context Lifecycle
// ===========================================================================

MPSGraphContextHandle mpsgraph_create_context(MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            setError(error, 1, @"Failed to create Metal device");
            return NULL;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            setError(error, 2, @"Failed to create Metal command queue");
            return NULL;
        }

        MPSGraphContext* ctx = [[MPSGraphContext alloc] init];
        ctx.device = device;
        ctx.commandQueue = queue;
        ctx.graph = [[MPSGraph alloc] init];
        ctx.placeholders = [NSMutableArray array];

        return (__bridge_retained void*)ctx;
    }
}

MPSGraphContextHandle mpsgraph_create_context_with_device(void* deviceHandle, MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);

        id<MTLDevice> device = (__bridge id<MTLDevice>)deviceHandle;
        if (!device) {
            setError(error, 1, @"NULL device handle passed to create_context_with_device");
            return NULL;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            setError(error, 2, @"Failed to create Metal command queue");
            return NULL;
        }

        MPSGraphContext* ctx = [[MPSGraphContext alloc] init];
        ctx.device = device;
        ctx.commandQueue = queue;
        ctx.graph = [[MPSGraph alloc] init];
        ctx.placeholders = [NSMutableArray array];

        return (__bridge_retained void*)ctx;
    }
}

void* mpsgraph_device_handle(MPSGraphContextHandle handle) {
    if (!handle) return NULL;
    MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
    return (__bridge void*)ctx.device;
}

void mpsgraph_destroy_context(MPSGraphContextHandle handle) {
    if (handle) {
        @autoreleasepool {
            MPSGraphContext* ctx = (__bridge_transfer MPSGraphContext*)handle;
            (void)ctx; // ARC releases
        }
    }
}

const char* mpsgraph_device_name(MPSGraphContextHandle handle) {
    @autoreleasepool {
        if (!handle) return strdup("unknown");
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        if (!ctx.device) return strdup("unknown");
        return strdup([ctx.device.name UTF8String]);
    }
}

// ===========================================================================
// Tensor Creation
// ===========================================================================

MPSGraphTensorHandle mpsgraph_placeholder(MPSGraphContextHandle handle, int dtype,
    int64_t* shape, int rank, MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        NSArray<NSNumber*>* shapeArr = shapeArray(shape, rank);
        MPSDataType mpsType = toMPSDataType(dtype);

        MPSGraphTensor* tensor = [ctx.graph placeholderWithShape:shapeArr
                                                        dataType:mpsType
                                                            name:nil];
        if (!tensor) {
            setError(error, 10, @"Failed to create placeholder tensor");
            return NULL;
        }
        [ctx.placeholders addObject:tensor];
        return (__bridge void*)tensor;
    }
}

MPSGraphTensorHandle mpsgraph_constant(MPSGraphContextHandle handle, void* data,
    int64_t nbytes, int dtype, int64_t* shape, int rank, MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        NSArray<NSNumber*>* shapeArr = shapeArray(shape, rank);
        MPSDataType mpsType = toMPSDataType(dtype);
        NSData* nsData = [NSData dataWithBytes:data length:nbytes];

        MPSGraphTensor* tensor = [ctx.graph constantWithData:nsData
                                                       shape:shapeArr
                                                    dataType:mpsType];
        if (!tensor) {
            setError(error, 11, @"Failed to create constant tensor");
            return NULL;
        }
        return (__bridge void*)tensor;
    }
}

MPSGraphTensorHandle mpsgraph_iota(MPSGraphContextHandle handle, int dtype,
    int64_t* shape, int rank, int axis, MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        NSArray<NSNumber*>* shapeArr = shapeArray(shape, rank);
        MPSDataType mpsType = toMPSDataType(dtype);

        // Create coordinate tensor along the specified axis.
        MPSGraphTensor* tensor = [ctx.graph coordinateAlongAxis:axis
                                                      withShape:shapeArr
                                                           name:nil];
        // coordinateAlongAxis returns Int32; cast if needed.
        if (mpsType != MPSDataTypeInt32) {
            tensor = [ctx.graph castTensor:tensor toType:mpsType name:nil];
        }
        if (!tensor) {
            setError(error, 12, @"Failed to create iota tensor");
            return NULL;
        }
        return (__bridge void*)tensor;
    }
}

// ===========================================================================
// Unary Operations - using macros for conciseness
// ===========================================================================

#define IMPL_UNARY_OP(cname, method) \
MPSGraphTensorHandle cname(MPSGraphContextHandle handle, MPSGraphTensorHandle x, MPSGraphError* error) { \
    @autoreleasepool { \
        clearError(error); \
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle; \
        MPSGraphTensor* input = (__bridge MPSGraphTensor*)x; \
        MPSGraphTensor* result = [ctx.graph method##WithTensor:input name:nil]; \
        if (!result) { \
            setError(error, 20, @"Unary op " #method " failed"); \
            return NULL; \
        } \
        return (__bridge void*)result; \
    } \
}

IMPL_UNARY_OP(mpsgraph_abs,         absolute)
IMPL_UNARY_OP(mpsgraph_neg,         negative)
IMPL_UNARY_OP(mpsgraph_sqrt,        squareRoot)
IMPL_UNARY_OP(mpsgraph_rsqrt,       reciprocalSquareRoot)
IMPL_UNARY_OP(mpsgraph_exp,         exponent)
IMPL_UNARY_OP(mpsgraph_log,         logarithm)
IMPL_UNARY_OP(mpsgraph_sin,         sin)
IMPL_UNARY_OP(mpsgraph_cos,         cos)
IMPL_UNARY_OP(mpsgraph_tanh,        tanh)
IMPL_UNARY_OP(mpsgraph_sigmoid,     sigmoid)
IMPL_UNARY_OP(mpsgraph_erf,         erf)
IMPL_UNARY_OP(mpsgraph_floor,       floor)
IMPL_UNARY_OP(mpsgraph_ceil,        ceil)
IMPL_UNARY_OP(mpsgraph_round,       rint)
IMPL_UNARY_OP(mpsgraph_sign,        sign)
// logicalNOT: MPSGraph uses notWithTensor:name:
MPSGraphTensorHandle mpsgraph_logical_not(MPSGraphContextHandle handle, MPSGraphTensorHandle x, MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        MPSGraphTensor* input = (__bridge MPSGraphTensor*)x;
        MPSGraphTensor* result = [ctx.graph notWithTensor:input name:nil];
        if (!result) {
            setError(error, 20, @"logicalNOT failed");
            return NULL;
        }
        return (__bridge void*)result;
    }
}
IMPL_UNARY_OP(mpsgraph_bitwise_not, bitwiseNOT)
IMPL_UNARY_OP(mpsgraph_is_finite,   isFinite)
IMPL_UNARY_OP(mpsgraph_is_nan,      isNaN)
IMPL_UNARY_OP(mpsgraph_identity,    identity)

// expm1 = exp(x) - 1: MPSGraph has exponentMinusOne (macOS 15+), but we use a safe decomposition.
MPSGraphTensorHandle mpsgraph_expm1(MPSGraphContextHandle handle, MPSGraphTensorHandle x, MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        MPSGraphTensor* input = (__bridge MPSGraphTensor*)x;
        MPSGraphTensor* expX = [ctx.graph exponentWithTensor:input name:nil];
        MPSGraphTensor* one = [ctx.graph constantWithScalar:1.0
                                                      shape:@[@1]
                                                   dataType:input.dataType];
        MPSGraphTensor* result = [ctx.graph subtractionWithPrimaryTensor:expX
                                                        secondaryTensor:one
                                                                   name:nil];
        if (!result) {
            setError(error, 20, @"expm1 failed");
            return NULL;
        }
        return (__bridge void*)result;
    }
}

// log1p = log(1 + x)
MPSGraphTensorHandle mpsgraph_log1p(MPSGraphContextHandle handle, MPSGraphTensorHandle x, MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        MPSGraphTensor* input = (__bridge MPSGraphTensor*)x;
        MPSGraphTensor* one = [ctx.graph constantWithScalar:1.0
                                                      shape:@[@1]
                                                   dataType:input.dataType];
        MPSGraphTensor* onePlusX = [ctx.graph additionWithPrimaryTensor:one
                                                       secondaryTensor:input
                                                                  name:nil];
        MPSGraphTensor* result = [ctx.graph logarithmWithTensor:onePlusX name:nil];
        if (!result) {
            setError(error, 20, @"log1p failed");
            return NULL;
        }
        return (__bridge void*)result;
    }
}

// ===========================================================================
// Binary Operations
// ===========================================================================

#define IMPL_BINARY_OP(cname, method) \
MPSGraphTensorHandle cname(MPSGraphContextHandle handle, MPSGraphTensorHandle lhs, MPSGraphTensorHandle rhs, MPSGraphError* error) { \
    @autoreleasepool { \
        clearError(error); \
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle; \
        MPSGraphTensor* l = (__bridge MPSGraphTensor*)lhs; \
        MPSGraphTensor* r = (__bridge MPSGraphTensor*)rhs; \
        MPSGraphTensor* result = [ctx.graph method##WithPrimaryTensor:l secondaryTensor:r name:nil]; \
        if (!result) { \
            setError(error, 21, @"Binary op " #method " failed"); \
            return NULL; \
        } \
        return (__bridge void*)result; \
    } \
}

IMPL_BINARY_OP(mpsgraph_add,             addition)
IMPL_BINARY_OP(mpsgraph_sub,             subtraction)
IMPL_BINARY_OP(mpsgraph_mul,             multiplication)
IMPL_BINARY_OP(mpsgraph_div,             division)
IMPL_BINARY_OP(mpsgraph_rem,             modulo)
IMPL_BINARY_OP(mpsgraph_pow,             power)
IMPL_BINARY_OP(mpsgraph_max,             maximum)
IMPL_BINARY_OP(mpsgraph_min,             minimum)
IMPL_BINARY_OP(mpsgraph_atan2,           atan2)
IMPL_BINARY_OP(mpsgraph_logical_and,     logicalAND)
IMPL_BINARY_OP(mpsgraph_logical_or,      logicalOR)
IMPL_BINARY_OP(mpsgraph_logical_xor,     logicalXOR)
IMPL_BINARY_OP(mpsgraph_bitwise_and,     bitwiseAND)
IMPL_BINARY_OP(mpsgraph_bitwise_or,      bitwiseOR)
IMPL_BINARY_OP(mpsgraph_bitwise_xor,     bitwiseXOR)
IMPL_BINARY_OP(mpsgraph_shift_left,      bitwiseLeftShift)
IMPL_BINARY_OP(mpsgraph_shift_right,     bitwiseRightShift)

// --- Comparison Operations ---

IMPL_BINARY_OP(mpsgraph_equal,            equal)
IMPL_BINARY_OP(mpsgraph_not_equal,        notEqual)
IMPL_BINARY_OP(mpsgraph_less_than,        lessThan)
IMPL_BINARY_OP(mpsgraph_less_or_equal,    lessThanOrEqualTo)
IMPL_BINARY_OP(mpsgraph_greater_than,     greaterThan)
IMPL_BINARY_OP(mpsgraph_greater_or_equal, greaterThanOrEqualTo)

// ===========================================================================
// Shape Operations
// ===========================================================================

MPSGraphTensorHandle mpsgraph_reshape(MPSGraphContextHandle handle, MPSGraphTensorHandle x,
    int64_t* shape, int rank, MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        MPSGraphTensor* input = (__bridge MPSGraphTensor*)x;
        NSArray<NSNumber*>* shapeArr = shapeArray(shape, rank);
        MPSGraphTensor* result = [ctx.graph reshapeTensor:input
                                                withShape:shapeArr
                                                     name:nil];
        if (!result) {
            setError(error, 30, @"reshape failed");
            return NULL;
        }
        return (__bridge void*)result;
    }
}

MPSGraphTensorHandle mpsgraph_transpose(MPSGraphContextHandle handle, MPSGraphTensorHandle x,
    int* permutation, int rank, MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        MPSGraphTensor* input = (__bridge MPSGraphTensor*)x;

        // Apply transposition by chaining pairwise swaps to achieve the target permutation.
        // We build the permutation incrementally: for each position i, find where the target
        // axis currently is and swap it into place.
        int current[rank];
        for (int i = 0; i < rank; i++) current[i] = i;

        MPSGraphTensor* result = input;
        for (int i = 0; i < rank; i++) {
            int target = permutation[i];
            if (current[i] == target) continue;

            // Find where target currently is.
            int j = -1;
            for (int k = i; k < rank; k++) {
                if (current[k] == target) { j = k; break; }
            }
            if (j < 0) {
                setError(error, 31, @"invalid permutation in transpose");
                return NULL;
            }

            result = [ctx.graph transposeTensor:result
                                      dimension:(NSUInteger)i
                                  withDimension:(NSUInteger)j
                                           name:nil];
            // Update tracking.
            int tmp = current[i];
            current[i] = current[j];
            current[j] = tmp;
        }

        return (__bridge void*)result;
    }
}

MPSGraphTensorHandle mpsgraph_cast(MPSGraphContextHandle handle, MPSGraphTensorHandle x,
    int dtype, MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        MPSGraphTensor* input = (__bridge MPSGraphTensor*)x;
        MPSDataType mpsType = toMPSDataType(dtype);
        MPSGraphTensor* result = [ctx.graph castTensor:input toType:mpsType name:nil];
        if (!result) {
            setError(error, 32, @"cast failed");
            return NULL;
        }
        return (__bridge void*)result;
    }
}

MPSGraphTensorHandle mpsgraph_broadcast_to(MPSGraphContextHandle handle, MPSGraphTensorHandle x,
    int64_t* shape, int rank, MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        MPSGraphTensor* input = (__bridge MPSGraphTensor*)x;
        NSArray<NSNumber*>* shapeArr = shapeArray(shape, rank);
        MPSGraphTensor* result = [ctx.graph broadcastTensor:input
                                                    toShape:shapeArr
                                                       name:nil];
        if (!result) {
            setError(error, 33, @"broadcast_to failed");
            return NULL;
        }
        return (__bridge void*)result;
    }
}

MPSGraphTensorHandle mpsgraph_slice(MPSGraphContextHandle handle, MPSGraphTensorHandle x,
    int64_t* starts, int64_t* ends, int64_t* strides, int rank, MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        MPSGraphTensor* input = (__bridge MPSGraphTensor*)x;

        NSMutableArray<NSNumber*>* startsArr = [NSMutableArray arrayWithCapacity:rank];
        NSMutableArray<NSNumber*>* endsArr   = [NSMutableArray arrayWithCapacity:rank];
        NSMutableArray<NSNumber*>* stridesArr = [NSMutableArray arrayWithCapacity:rank];
        for (int i = 0; i < rank; i++) {
            [startsArr addObject:@(starts[i])];
            [endsArr addObject:@(ends[i])];
            [stridesArr addObject:@(strides[i])];
        }

        MPSGraphTensor* result = [ctx.graph sliceTensor:input
                                                 starts:startsArr
                                                   ends:endsArr
                                                strides:stridesArr
                                                   name:nil];
        if (!result) {
            setError(error, 34, @"slice failed");
            return NULL;
        }
        return (__bridge void*)result;
    }
}

MPSGraphTensorHandle mpsgraph_concatenate(MPSGraphContextHandle handle,
    MPSGraphTensorHandle* tensors, int numTensors, int axis, MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        NSMutableArray<MPSGraphTensor*>* arr = [NSMutableArray arrayWithCapacity:numTensors];
        for (int i = 0; i < numTensors; i++) {
            [arr addObject:(__bridge MPSGraphTensor*)tensors[i]];
        }
        MPSGraphTensor* result = [ctx.graph concatTensors:arr
                                                dimension:axis
                                                     name:nil];
        if (!result) {
            setError(error, 35, @"concatenate failed");
            return NULL;
        }
        return (__bridge void*)result;
    }
}

MPSGraphTensorHandle mpsgraph_reverse(MPSGraphContextHandle handle, MPSGraphTensorHandle x,
    int* axes, int numAxes, MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        MPSGraphTensor* input = (__bridge MPSGraphTensor*)x;
        NSArray<NSNumber*>* axesArr = intArray(axes, numAxes);
        MPSGraphTensor* result = [ctx.graph reverseTensor:input
                                                     axes:axesArr
                                                     name:nil];
        if (!result) {
            setError(error, 36, @"reverse failed");
            return NULL;
        }
        return (__bridge void*)result;
    }
}

MPSGraphTensorHandle mpsgraph_pad(MPSGraphContextHandle handle, MPSGraphTensorHandle x,
    MPSGraphTensorHandle padValue, int64_t* padBefore, int64_t* padAfter, int rank,
    MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        MPSGraphTensor* input = (__bridge MPSGraphTensor*)x;
        MPSGraphTensor* fillTensor = (__bridge MPSGraphTensor*)padValue;

        // Use padTensor with constant padding mode.
        MPSGraphPaddingMode mode = MPSGraphPaddingModeConstant;
        NSMutableArray<NSNumber*>* leftPad = [NSMutableArray arrayWithCapacity:rank];
        NSMutableArray<NSNumber*>* rightPad = [NSMutableArray arrayWithCapacity:rank];
        for (int i = 0; i < rank; i++) {
            [leftPad addObject:@(padBefore[i])];
            [rightPad addObject:@(padAfter[i])];
        }

        // Step 1: Pad the input with constant 0.0.
        MPSGraphTensor* paddedInput = [ctx.graph padTensor:input
                                           withPaddingMode:mode
                                               leftPadding:leftPad
                                              rightPadding:rightPad
                                             constantValue:0.0
                                                      name:nil];
        if (!paddedInput) {
            setError(error, 37, @"pad failed");
            return NULL;
        }

        // Step 2: Build a mask to identify padding positions.
        // Create a ones tensor with the input shape, pad it with 0.0.
        // Result: 1 where original data, 0 where padding.
        MPSGraphTensor* ones = [ctx.graph constantWithScalar:1.0
                                                       shape:input.shape
                                                    dataType:input.dataType];
        MPSGraphTensor* mask = [ctx.graph padTensor:ones
                                    withPaddingMode:mode
                                        leftPadding:leftPad
                                       rightPadding:rightPad
                                      constantValue:0.0
                                               name:nil];

        // Step 3: Invert the mask: 1 where padding, 0 where data.
        MPSGraphTensor* onesLike = [ctx.graph constantWithScalar:1.0
                                                           shape:paddedInput.shape
                                                        dataType:input.dataType];
        MPSGraphTensor* invertedMask = [ctx.graph subtractionWithPrimaryTensor:onesLike
                                                              secondaryTensor:mask
                                                                         name:nil];

        // Step 4: Multiply inverted mask by fillTensor (broadcast).
        MPSGraphTensor* fillBroadcast = [ctx.graph multiplicationWithPrimaryTensor:invertedMask
                                                                  secondaryTensor:fillTensor
                                                                             name:nil];

        // Step 5: Add to padded input.
        MPSGraphTensor* result = [ctx.graph additionWithPrimaryTensor:paddedInput
                                                     secondaryTensor:fillBroadcast
                                                                name:nil];
        if (!result) {
            setError(error, 37, @"pad with fill value failed");
            return NULL;
        }
        return (__bridge void*)result;
    }
}

// ===========================================================================
// Ternary / Selection
// ===========================================================================

MPSGraphTensorHandle mpsgraph_where(MPSGraphContextHandle handle, MPSGraphTensorHandle cond,
    MPSGraphTensorHandle onTrue, MPSGraphTensorHandle onFalse, MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        MPSGraphTensor* c = (__bridge MPSGraphTensor*)cond;
        MPSGraphTensor* t = (__bridge MPSGraphTensor*)onTrue;
        MPSGraphTensor* f = (__bridge MPSGraphTensor*)onFalse;
        MPSGraphTensor* result = [ctx.graph selectWithPredicateTensor:c
                                                 truePredicateTensor:t
                                                falsePredicateTensor:f
                                                                name:nil];
        if (!result) {
            setError(error, 40, @"where failed");
            return NULL;
        }
        return (__bridge void*)result;
    }
}

MPSGraphTensorHandle mpsgraph_clamp(MPSGraphContextHandle handle, MPSGraphTensorHandle minVal,
    MPSGraphTensorHandle x, MPSGraphTensorHandle maxVal, MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        MPSGraphTensor* mn = (__bridge MPSGraphTensor*)minVal;
        MPSGraphTensor* input = (__bridge MPSGraphTensor*)x;
        MPSGraphTensor* mx = (__bridge MPSGraphTensor*)maxVal;
        MPSGraphTensor* result = [ctx.graph clampWithTensor:input
                                             minValueTensor:mn
                                             maxValueTensor:mx
                                                       name:nil];
        if (!result) {
            setError(error, 41, @"clamp failed");
            return NULL;
        }
        return (__bridge void*)result;
    }
}

// ===========================================================================
// Matrix Operations
// ===========================================================================

MPSGraphTensorHandle mpsgraph_matmul(MPSGraphContextHandle handle,
    MPSGraphTensorHandle lhs, MPSGraphTensorHandle rhs, MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        MPSGraphTensor* l = (__bridge MPSGraphTensor*)lhs;
        MPSGraphTensor* r = (__bridge MPSGraphTensor*)rhs;
        MPSGraphTensor* result = [ctx.graph matrixMultiplicationWithPrimaryTensor:l
                                                                 secondaryTensor:r
                                                                            name:nil];
        if (!result) {
            setError(error, 50, @"matmul failed");
            return NULL;
        }
        return (__bridge void*)result;
    }
}

// ===========================================================================
// Reduction Operations
// ===========================================================================

MPSGraphTensorHandle mpsgraph_reduce(MPSGraphContextHandle handle, MPSGraphTensorHandle x,
    int reduceType, int* axes, int numAxes, MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        MPSGraphTensor* input = (__bridge MPSGraphTensor*)x;
        NSArray<NSNumber*>* axesArr = intArray(axes, numAxes);

        MPSGraphTensor* result = nil;
        switch (reduceType) {
            case MPSGRAPH_REDUCE_SUM:
                result = [ctx.graph reductionSumWithTensor:input axes:axesArr name:nil];
                break;
            case MPSGRAPH_REDUCE_PRODUCT:
                result = [ctx.graph reductionProductWithTensor:input axes:axesArr name:nil];
                break;
            case MPSGRAPH_REDUCE_MAX:
                result = [ctx.graph reductionMaximumWithTensor:input axes:axesArr name:nil];
                break;
            case MPSGRAPH_REDUCE_MIN:
                result = [ctx.graph reductionMinimumWithTensor:input axes:axesArr name:nil];
                break;
            default:
                setError(error, 60, @"unknown reduction type");
                return NULL;
        }
        if (!result) {
            setError(error, 61, @"reduction failed");
            return NULL;
        }
        return (__bridge void*)result;
    }
}

// ===========================================================================
// Gather / Scatter
// ===========================================================================

MPSGraphTensorHandle mpsgraph_gather_nd(MPSGraphContextHandle handle,
    MPSGraphTensorHandle params, MPSGraphTensorHandle indices,
    int batchDims, MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        MPSGraphTensor* p = (__bridge MPSGraphTensor*)params;
        MPSGraphTensor* idx = (__bridge MPSGraphTensor*)indices;
        MPSGraphTensor* result = [ctx.graph gatherNDWithUpdatesTensor:p
                                                        indicesTensor:idx
                                                       batchDimensions:batchDims
                                                                 name:nil];
        if (!result) {
            setError(error, 70, @"gather_nd failed");
            return NULL;
        }
        return (__bridge void*)result;
    }
}

MPSGraphTensorHandle mpsgraph_gather_along_axis(MPSGraphContextHandle handle,
    MPSGraphTensorHandle x, MPSGraphTensorHandle indices, int axis,
    MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        MPSGraphTensor* input = (__bridge MPSGraphTensor*)x;
        MPSGraphTensor* idx = (__bridge MPSGraphTensor*)indices;
        MPSGraphTensor* result = [ctx.graph gatherAlongAxis:axis
                                          withUpdatesTensor:input
                                              indicesTensor:idx
                                                       name:nil];
        if (!result) {
            setError(error, 71, @"gather_along_axis failed");
            return NULL;
        }
        return (__bridge void*)result;
    }
}

// NOTE: The `data` parameter is accepted for API consistency with ScatterAlongAxis but is
// not used by scatterNDWithUpdatesTensor — that API always scatters into a zero-initialized
// tensor (for Add mode). This is correct for GoMLX's usage where the operand is always zeros
// (autodiff scatter gradients). If non-zero operand support is needed in the general path,
// an additional Add/Max/Min of the operand with the scatter result would be required.
MPSGraphTensorHandle mpsgraph_scatter_nd(MPSGraphContextHandle handle,
    MPSGraphTensorHandle data, MPSGraphTensorHandle indices, MPSGraphTensorHandle updates,
    int64_t* shape, int rank, int mode, MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        (void)data;  // See NOTE above.
        MPSGraphTensor* idx = (__bridge MPSGraphTensor*)indices;
        MPSGraphTensor* upd = (__bridge MPSGraphTensor*)updates;
        NSArray<NSNumber*>* shapeArr = shapeArray(shape, rank);

        MPSGraphScatterMode scatterMode;
        if (!toScatterMode(mode, &scatterMode)) {
            scatterMode = MPSGraphScatterModeSet;
        }

        MPSGraphTensor* result = [ctx.graph scatterNDWithUpdatesTensor:upd
                                                        indicesTensor:idx
                                                                shape:shapeArr
                                                       batchDimensions:0
                                                                  mode:scatterMode
                                                                  name:nil];
        if (!result) {
            setError(error, 72, @"scatter_nd failed");
            return NULL;
        }
        return (__bridge void*)result;
    }
}

// ===========================================================================
// ArgMin / ArgMax
// ===========================================================================

MPSGraphTensorHandle mpsgraph_argmin(MPSGraphContextHandle handle, MPSGraphTensorHandle x,
    int axis, int outputDtype, MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        MPSGraphTensor* input = (__bridge MPSGraphTensor*)x;
        MPSGraphTensor* result = [ctx.graph reductionArgMinimumWithTensor:input
                                                                     axis:axis
                                                                     name:nil];
        MPSDataType mpsOutType = toMPSDataType(outputDtype);
        if (result.dataType != mpsOutType) {
            result = [ctx.graph castTensor:result toType:mpsOutType name:nil];
        }
        if (!result) {
            setError(error, 73, @"argmin failed");
            return NULL;
        }
        return (__bridge void*)result;
    }
}

MPSGraphTensorHandle mpsgraph_argmax(MPSGraphContextHandle handle, MPSGraphTensorHandle x,
    int axis, int outputDtype, MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        MPSGraphTensor* input = (__bridge MPSGraphTensor*)x;
        MPSGraphTensor* result = [ctx.graph reductionArgMaximumWithTensor:input
                                                                     axis:axis
                                                                     name:nil];
        MPSDataType mpsOutType = toMPSDataType(outputDtype);
        if (result.dataType != mpsOutType) {
            result = [ctx.graph castTensor:result toType:mpsOutType name:nil];
        }
        if (!result) {
            setError(error, 74, @"argmax failed");
            return NULL;
        }
        return (__bridge void*)result;
    }
}

// ===========================================================================
// Batch Normalization
// ===========================================================================

MPSGraphTensorHandle mpsgraph_batch_norm_inference(MPSGraphContextHandle handle,
    MPSGraphTensorHandle input, MPSGraphTensorHandle mean, MPSGraphTensorHandle variance,
    MPSGraphTensorHandle gamma, MPSGraphTensorHandle beta,
    float epsilon, int featureAxis, MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)input;
        MPSGraphTensor* m = (__bridge MPSGraphTensor*)mean;
        MPSGraphTensor* v = (__bridge MPSGraphTensor*)variance;
        MPSGraphTensor* g = gamma ? (__bridge MPSGraphTensor*)gamma : nil;
        MPSGraphTensor* b = beta ? (__bridge MPSGraphTensor*)beta : nil;

        // (x - mean) / sqrt(variance + epsilon) * gamma + beta
        MPSGraphTensor* epsTensor = [ctx.graph constantWithScalar:epsilon
                                                            shape:@[@1]
                                                         dataType:v.dataType];
        MPSGraphTensor* varEps = [ctx.graph additionWithPrimaryTensor:v
                                                     secondaryTensor:epsTensor
                                                                name:nil];
        MPSGraphTensor* stddev = [ctx.graph squareRootWithTensor:varEps name:nil];
        MPSGraphTensor* xCentered = [ctx.graph subtractionWithPrimaryTensor:x
                                                            secondaryTensor:m
                                                                       name:nil];
        MPSGraphTensor* result = [ctx.graph divisionWithPrimaryTensor:xCentered
                                                     secondaryTensor:stddev
                                                                name:nil];
        if (g) {
            result = [ctx.graph multiplicationWithPrimaryTensor:result
                                               secondaryTensor:g
                                                          name:nil];
        }
        if (b) {
            result = [ctx.graph additionWithPrimaryTensor:result
                                         secondaryTensor:b
                                                    name:nil];
        }
        if (!result) {
            setError(error, 80, @"batch_norm_inference failed");
            return NULL;
        }
        return (__bridge void*)result;
    }
}

// ===========================================================================
// Dynamic Slice / Update
// ===========================================================================

MPSGraphTensorHandle mpsgraph_dynamic_slice(MPSGraphContextHandle handle,
    MPSGraphTensorHandle x, MPSGraphTensorHandle* startIndices, int numIndices,
    int64_t* sliceSizes, int rank, MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        MPSGraphTensor* result = (__bridge MPSGraphTensor*)x;

        // Per-axis dynamic slicing using gather.
        // For each axis: create indices [start, start+1, ..., start+size-1],
        // then gatherAlongAxis to select those elements.
        for (int i = 0; i < numIndices; i++) {
            NSUInteger length = (NSUInteger)sliceSizes[i];
            NSArray<NSNumber*>* resultShape = result.shape;
            NSUInteger axisDim = resultShape[i].unsignedLongValue;

            // Skip if slice size equals the full dimension (no-op for this axis).
            if (length == axisDim) continue;

            MPSGraphTensor* startIdx = (__bridge MPSGraphTensor*)startIndices[i];

            // Create range [0, 1, ..., length-1] as Int32.
            NSMutableArray<NSNumber*>* rangeShape = [NSMutableArray arrayWithObject:@(length)];
            MPSGraphTensor* range = [ctx.graph coordinateAlongAxis:0
                                                        withShape:rangeShape
                                                             name:nil];
            // Cast range to match startIdx type.
            range = [ctx.graph castTensor:range toType:startIdx.dataType name:nil];

            // Reshape startIdx to scalar-compatible shape for broadcasting.
            startIdx = [ctx.graph reshapeTensor:startIdx withShape:@[@1] name:nil];

            // indices = range + startIdx → [start, start+1, ..., start+length-1]
            MPSGraphTensor* indices = [ctx.graph additionWithPrimaryTensor:range
                                                          secondaryTensor:startIdx
                                                                     name:nil];

            // Expand indices to match result rank for gatherAlongAxis.
            // gatherAlongAxis requires indices rank == data rank.
            NSUInteger resultRank = resultShape.count;
            NSMutableArray<NSNumber*>* expandedShape = [NSMutableArray arrayWithCapacity:resultRank];
            for (NSUInteger d = 0; d < resultRank; d++) {
                if (d == (NSUInteger)i) {
                    [expandedShape addObject:@(length)];
                } else {
                    [expandedShape addObject:@1];
                }
            }
            indices = [ctx.graph reshapeTensor:indices withShape:expandedShape name:nil];

            // Broadcast indices to match result shape (with sliced axis = length).
            NSMutableArray<NSNumber*>* broadcastShape = [NSMutableArray arrayWithArray:resultShape];
            broadcastShape[i] = @(length);
            indices = [ctx.graph broadcastTensor:indices toShape:broadcastShape name:nil];

            result = [ctx.graph gatherAlongAxis:(NSInteger)i
                              withUpdatesTensor:result
                                  indicesTensor:indices
                                           name:nil];
            if (!result) {
                setError(error, 100, [NSString stringWithFormat:@"dynamic_slice gather failed on axis %d", i]);
                return NULL;
            }
        }

        return (__bridge void*)result;
    }
}

MPSGraphTensorHandle mpsgraph_dynamic_update_slice(MPSGraphContextHandle handle,
    MPSGraphTensorHandle x, MPSGraphTensorHandle updateH,
    MPSGraphTensorHandle* startIndices, int numIndices, MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        MPSGraphTensor* input = (__bridge MPSGraphTensor*)x;
        MPSGraphTensor* update = (__bridge MPSGraphTensor*)updateH;
        NSArray<NSNumber*>* inputShape = input.shape;

        // Strategy: mask + gather to position update + where to combine.
        // 1. Build boolean mask: true where the update region is.
        // 2. Build padded_update: a full-size tensor with update values at the right positions.
        // 3. Result = where(mask, padded_update, input).

        // Step 1: Build mask by ANDing per-axis range conditions.
        MPSGraphTensor* mask = nil;
        for (int axis = 0; axis < numIndices; axis++) {
            MPSGraphTensor* startIdx = (__bridge MPSGraphTensor*)startIndices[axis];
            startIdx = [ctx.graph reshapeTensor:startIdx withShape:@[@1] name:nil];

            // range = [0, 1, ..., dim-1] along this axis, broadcast to input shape.
            MPSGraphTensor* range = [ctx.graph coordinateAlongAxis:axis
                                                        withShape:inputShape
                                                             name:nil];
            // Cast range to match startIdx type for comparison.
            range = [ctx.graph castTensor:range toType:startIdx.dataType name:nil];

            MPSGraphTensor* lower = [ctx.graph greaterThanOrEqualToWithPrimaryTensor:range
                                                                    secondaryTensor:startIdx
                                                                               name:nil];
            NSUInteger updateSize = update.shape[axis].unsignedLongValue;
            MPSGraphTensor* endIdx = [ctx.graph additionWithPrimaryTensor:startIdx
                                                         secondaryTensor:[ctx.graph constantWithScalar:updateSize
                                                                                             dataType:startIdx.dataType]
                                                                    name:nil];
            MPSGraphTensor* upper = [ctx.graph lessThanWithPrimaryTensor:range
                                                        secondaryTensor:endIdx
                                                                   name:nil];
            MPSGraphTensor* axisMask = [ctx.graph logicalANDWithPrimaryTensor:lower
                                                             secondaryTensor:upper
                                                                        name:nil];
            mask = (mask == nil) ? axisMask :
                [ctx.graph logicalANDWithPrimaryTensor:mask secondaryTensor:axisMask name:nil];
        }

        if (!mask) {
            setError(error, 101, @"dynamic_update_slice: failed to build mask");
            return NULL;
        }

        // Step 2: Build padded_update by gathering update values into input-sized tensor.
        // For each axis: create reverse indices (pos - start, clamped to [0, updateDim-1]),
        // then gatherAlongAxis from the update.
        MPSGraphTensor* paddedUpdate = update;
        for (int axis = 0; axis < numIndices; axis++) {
            MPSGraphTensor* startIdx = (__bridge MPSGraphTensor*)startIndices[axis];
            startIdx = [ctx.graph reshapeTensor:startIdx withShape:@[@1] name:nil];

            NSInteger inputDim = inputShape[axis].integerValue;
            NSInteger updateDim = update.shape[axis].integerValue;

            if (inputDim == updateDim) continue; // No expansion needed on this axis.

            // Create range [0, ..., inputDim-1].
            NSMutableArray<NSNumber*>* rangeShape = [NSMutableArray arrayWithObject:@(inputDim)];
            MPSGraphTensor* range = [ctx.graph coordinateAlongAxis:0 withShape:rangeShape name:nil];
            range = [ctx.graph castTensor:range toType:startIdx.dataType name:nil];

            // reverseIdx = range - start (maps input positions to update positions).
            MPSGraphTensor* reverseIdx = [ctx.graph subtractionWithPrimaryTensor:range
                                                                secondaryTensor:startIdx
                                                                           name:nil];
            // Clamp to [0, updateDim-1] to avoid out-of-bounds gather.
            // Values outside the update region will be masked away by where().
            MPSGraphTensor* zero = [ctx.graph constantWithScalar:0 dataType:reverseIdx.dataType];
            MPSGraphTensor* maxIdx = [ctx.graph constantWithScalar:(updateDim - 1) dataType:reverseIdx.dataType];
            reverseIdx = [ctx.graph clampWithTensor:reverseIdx
                                        minValueTensor:zero
                                        maxValueTensor:maxIdx
                                                  name:nil];
            // Cast to Int32 for gather indices.
            reverseIdx = [ctx.graph castTensor:reverseIdx toType:MPSDataTypeInt32 name:nil];

            // Expand reverseIdx to match paddedUpdate rank.
            NSArray<NSNumber*>* curShape = paddedUpdate.shape;
            NSUInteger curRank = curShape.count;
            NSMutableArray<NSNumber*>* expandedShape = [NSMutableArray arrayWithCapacity:curRank];
            for (NSUInteger d = 0; d < curRank; d++) {
                [expandedShape addObject:(d == (NSUInteger)axis) ? @(inputDim) : @1];
            }
            reverseIdx = [ctx.graph reshapeTensor:reverseIdx withShape:expandedShape name:nil];

            // Broadcast to full shape.
            NSMutableArray<NSNumber*>* broadcastShape = [NSMutableArray arrayWithArray:curShape];
            broadcastShape[axis] = @(inputDim);
            reverseIdx = [ctx.graph broadcastTensor:reverseIdx toShape:broadcastShape name:nil];

            paddedUpdate = [ctx.graph gatherAlongAxis:(NSInteger)axis
                                    withUpdatesTensor:paddedUpdate
                                        indicesTensor:reverseIdx
                                                 name:nil];
        }

        // Step 3: where(mask, paddedUpdate, input)
        // Cast paddedUpdate to input dtype if needed.
        if (paddedUpdate.dataType != input.dataType) {
            paddedUpdate = [ctx.graph castTensor:paddedUpdate toType:input.dataType name:nil];
        }
        MPSGraphTensor* result = [ctx.graph selectWithPredicateTensor:mask
                                                 truePredicateTensor:paddedUpdate
                                                falsePredicateTensor:input
                                                                name:nil];
        if (!result) {
            setError(error, 102, @"dynamic_update_slice: where failed");
            return NULL;
        }
        return (__bridge void*)result;
    }
}

// ===========================================================================
// Compilation & Execution
// ===========================================================================

MPSGraphExecHandle mpsgraph_compile(MPSGraphContextHandle handle,
    MPSGraphTensorHandle* feeds, int* feedDtypes, int64_t** feedShapes, int* feedRanks, int numFeeds,
    MPSGraphTensorHandle* targets, int numTargets,
    MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;

        // Build feeds dictionary: MPSGraphTensor* -> MPSGraphShapedType*
        NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*>* feedsDict =
            [NSMutableDictionary dictionaryWithCapacity:numFeeds];
        NSMutableArray<MPSGraphTensor*>* feedTensors = [NSMutableArray arrayWithCapacity:numFeeds];

        for (int i = 0; i < numFeeds; i++) {
            MPSGraphTensor* tensor = (__bridge MPSGraphTensor*)feeds[i];
            MPSDataType mpsType = toMPSDataType(feedDtypes[i]);
            NSArray<NSNumber*>* shape = shapeArray(feedShapes[i], feedRanks[i]);
            MPSGraphShapedType* shapedType = [[MPSGraphShapedType alloc] initWithShape:shape
                                                                              dataType:mpsType];
            feedsDict[tensor] = shapedType;
            [feedTensors addObject:tensor];
        }

        // Build targets array.
        NSMutableArray<MPSGraphTensor*>* targetTensors = [NSMutableArray arrayWithCapacity:numTargets];
        for (int i = 0; i < numTargets; i++) {
            [targetTensors addObject:(__bridge MPSGraphTensor*)targets[i]];
        }

        // Compile.
        MPSGraphCompilationDescriptor* compDesc = [[MPSGraphCompilationDescriptor alloc] init];
        MPSGraphExecutable* exec = nil;
        @try {
            exec = [ctx.graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:ctx.device]
                                                              feeds:feedsDict
                                                      targetTensors:targetTensors
                                                   targetOperations:nil
                                              compilationDescriptor:compDesc];
        } @catch (NSException* exception) {
            NSString* msg = [NSString stringWithFormat:@"Compilation threw: %@ - %@", exception.name, exception.reason];
            setError(error, 210, msg);
            return NULL;
        }
        if (!exec) {
            setError(error, 110, @"MPSGraph compilation failed");
            return NULL;
        }

        // Wrap in our context object.
        MPSGraphExecWrapper* wrapper = [[MPSGraphExecWrapper alloc] init];
        wrapper.executable = exec;
        wrapper.graph = ctx.graph;  // Keep reference for non-compiled fallback
        wrapper.device = ctx.device;
        wrapper.commandQueue = ctx.commandQueue;
        // Store original feed/target tensors in our order for non-compiled fallback.
        wrapper.origFeedTensors = [feedTensors copy];
        wrapper.origTargetTensors = [targetTensors copy];

        // Build permutation from our feed order to executable's feed order.
        // The NSDictionary used for compilation has undefined iteration order,
        // so the compiled executable's feedTensors may differ from our order.
        NSArray<MPSGraphTensor*>* execFeeds = exec.feedTensors;
        if (execFeeds && execFeeds.count == (NSUInteger)numFeeds) {
            NSMutableArray<NSNumber*>* perm = [NSMutableArray arrayWithCapacity:numFeeds];
            bool needsReorder = false;
            for (NSUInteger i = 0; i < execFeeds.count; i++) {
                MPSGraphTensor* t = execFeeds[i];
                NSUInteger ourIdx = [feedTensors indexOfObjectIdenticalTo:t];
                if (ourIdx == NSNotFound) {
                    perm = nil;
                    break;
                }
                [perm addObject:@(ourIdx)];
                if (ourIdx != i) needsReorder = true;
            }
            wrapper.feedPermutation = needsReorder ? [perm copy] : nil;
            wrapper.feedTensors = [execFeeds copy];
        } else {
            wrapper.feedTensors = [feedTensors copy];
            wrapper.feedPermutation = nil;
            // exec.feedTensors count mismatch — use our original order.
        }

        // Build target permutation similarly.
        NSArray<MPSGraphTensor*>* execTargets = exec.targetTensors;
        if (execTargets && execTargets.count == (NSUInteger)numTargets) {
            NSMutableArray<NSNumber*>* tPerm = [NSMutableArray arrayWithCapacity:numTargets];
            bool needsReorder = false;
            for (NSUInteger i = 0; i < execTargets.count; i++) {
                MPSGraphTensor* t = execTargets[i];
                NSUInteger ourIdx = [targetTensors indexOfObjectIdenticalTo:t];
                if (ourIdx == NSNotFound) {
                    tPerm = nil;
                    break;
                }
                [tPerm addObject:@(ourIdx)];
                if (ourIdx != i) needsReorder = true;
            }
            wrapper.targetPermutation = needsReorder ? [tPerm copy] : nil;
            wrapper.targetTensors = [execTargets copy];
        } else {
            wrapper.targetTensors = [targetTensors copy];
            wrapper.targetPermutation = nil;
        }

        // DEBUG: Log compilation info for large graphs.
#ifdef DEBUG
        if (numFeeds > 10) {
            fprintf(stderr, "[COMPILE] numFeeds=%d, numTargets=%d\n", numFeeds, numTargets);
            fprintf(stderr, "[COMPILE] execFeeds.count=%lu, execTargets.count=%lu\n",
                (unsigned long)(exec.feedTensors ? exec.feedTensors.count : 0),
                (unsigned long)(exec.targetTensors ? exec.targetTensors.count : 0));
            fprintf(stderr, "[COMPILE] feedPermutation=%s, targetPermutation=%s\n",
                wrapper.feedPermutation ? "YES" : "nil",
                wrapper.targetPermutation ? "YES" : "nil");
            if (wrapper.feedPermutation) {
                fprintf(stderr, "[COMPILE] feedPerm: ");
                for (NSUInteger pi = 0; pi < wrapper.feedPermutation.count; pi++) {
                    fprintf(stderr, "%d ", [wrapper.feedPermutation[pi] intValue]);
                }
                fprintf(stderr, "\n");
            }
            fflush(stderr);
        }
#endif

        return (__bridge_retained void*)wrapper;
    }
}

void mpsgraph_destroy_exec(MPSGraphExecHandle handle) {
    if (handle) {
        @autoreleasepool {
            MPSGraphExecWrapper* wrapper = (__bridge_transfer MPSGraphExecWrapper*)handle;
            (void)wrapper; // ARC releases
        }
    }
}

bool mpsgraph_execute(MPSGraphExecHandle handle,
    void** inputData, int64_t* inputSizes, int* inputDtypes,
    int64_t** inputShapes, int* inputRanks, int numInputs,
    void** outputData, int64_t* outputSizes, int* outputDtypes,
    int64_t** outputShapes, int* outputRanks, int numOutputs,
    MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphExecWrapper* wrapper = (__bridge MPSGraphExecWrapper*)handle;

        @try {

        // Use the compiled MPSGraphExecutable for execution.
        MPSGraphDevice* graphDevice = [MPSGraphDevice deviceWithMTLDevice:wrapper.device];

        // Verify feed/target counts match.
        if ((NSUInteger)numInputs != wrapper.origFeedTensors.count) {
            NSString* msg = [NSString stringWithFormat:@"Input count mismatch: got %d inputs, graph expects %lu feeds",
                numInputs, (unsigned long)wrapper.origFeedTensors.count];
            setError(error, 115, msg);
            return false;
        }
        if ((NSUInteger)numOutputs != wrapper.origTargetTensors.count) {
            NSString* msg = [NSString stringWithFormat:@"Output count mismatch: got %d outputs, graph expects %lu targets",
                numOutputs, (unsigned long)wrapper.origTargetTensors.count];
            setError(error, 116, msg);
            return false;
        }

        // Build MPSGraphTensorData array for inputs in our caller order.
        NSMutableArray<MPSGraphTensorData*>* callerInputs = [NSMutableArray arrayWithCapacity:numInputs];
        for (int i = 0; i < numInputs; i++) {
            if (inputData[i] == NULL) {
                NSString* msg = [NSString stringWithFormat:@"Input %d has NULL data pointer (size=%lld bytes)", i, inputSizes[i]];
                setError(error, 113, msg);
                return false;
            }

            MPSDataType mpsType = toMPSDataType(inputDtypes[i]);
            NSArray<NSNumber*>* shape = shapeArray(inputShapes[i], inputRanks[i]);

            // Copy input data to NSData. Metal needs its own copy for GPU access.
            NSData* data = [NSData dataWithBytes:inputData[i] length:(NSUInteger)inputSizes[i]];
            MPSGraphTensorData* tensorData = [[MPSGraphTensorData alloc] initWithDevice:graphDevice
                                                                                   data:data
                                                                                  shape:shape
                                                                               dataType:mpsType];
            [callerInputs addObject:tensorData];
        }

        // Reorder inputs from caller order to executable's feed order using feedPermutation.
        // feedPermutation[i] = index in caller's array for executable's i-th feed.
        NSMutableArray<MPSGraphTensorData*>* execInputs = [NSMutableArray arrayWithCapacity:numInputs];
        if (wrapper.feedPermutation) {
            for (NSUInteger i = 0; i < (NSUInteger)numInputs; i++) {
                NSUInteger callerIdx = [wrapper.feedPermutation[i] unsignedIntegerValue];
                [execInputs addObject:callerInputs[callerIdx]];
            }
        } else {
            [execInputs addObjectsFromArray:callerInputs];
        }

#ifdef DEBUG
        if (numInputs > 10) {
            int64_t totalBytes = 0;
            for (int i = 0; i < numInputs; i++) totalBytes += inputSizes[i];
            fprintf(stderr, "[EXEC] encodeToCommandBuffer: numInputs=%d, numOutputs=%d, totalInputBytes=%lld MB\n",
                numInputs, numOutputs, totalBytes / (1024*1024));
            fflush(stderr);
        }
#endif

        // Use encodeToCommandBuffer on the compiled executable.
        MPSCommandBuffer* commandBuffer = [MPSCommandBuffer commandBufferFromCommandQueue:wrapper.commandQueue];
        if (!commandBuffer) {
            setError(error, 118, @"Failed to create MPSCommandBuffer");
            return false;
        }

        MPSGraphExecutableExecutionDescriptor* execDesc = [[MPSGraphExecutableExecutionDescriptor alloc] init];

        // The executable returns an array of MPSGraphTensorData in its target order.
        NSArray<MPSGraphTensorData*>* resultsArray =
            [wrapper.executable encodeToCommandBuffer:commandBuffer
                                          inputsArray:execInputs
                                         resultsArray:nil
                                  executionDescriptor:execDesc];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if (commandBuffer.error) {
            NSString* msg = [NSString stringWithFormat:@"Metal command buffer error: %@", commandBuffer.error];
            setError(error, 119, msg);
            return false;
        }

#ifdef DEBUG
        if (numInputs > 10) {
            fprintf(stderr, "[EXEC] encodeToCommandBuffer completed, resultsArray.count=%lu\n",
                (unsigned long)(resultsArray ? resultsArray.count : 0));
            fflush(stderr);
        }
#endif

        if (!resultsArray || resultsArray.count == 0) {
            setError(error, 111, @"MPSGraphExecutable encodeToCommandBuffer returned nil/empty results");
            return false;
        }

        // Copy output data back. The executable's results are in its target order.
        // targetPermutation[i] = index in caller's array for executable's i-th target.
        // We need to map from executable order to caller order.
        for (int i = 0; i < numOutputs; i++) {
            // Find which executable result index corresponds to caller output index i.
            NSUInteger execIdx = (NSUInteger)i;
            if (wrapper.targetPermutation) {
                // targetPermutation maps exec index -> caller index.
                // We need the reverse: for caller index i, find exec index.
                for (NSUInteger j = 0; j < wrapper.targetPermutation.count; j++) {
                    if ([wrapper.targetPermutation[j] unsignedIntegerValue] == (NSUInteger)i) {
                        execIdx = j;
                        break;
                    }
                }
            }

            if (execIdx >= resultsArray.count) {
                NSString* msg = [NSString stringWithFormat:@"Result index %lu out of bounds (count=%lu) for output %d",
                    (unsigned long)execIdx, (unsigned long)resultsArray.count, i];
                setError(error, 117, msg);
                return false;
            }
            MPSGraphTensorData* result = resultsArray[execIdx];
            MPSNDArray* ndarray = result.mpsndarray;
            if (outputData[i] == NULL) {
                NSString* msg = [NSString stringWithFormat:@"Output %d has NULL data pointer", i];
                setError(error, 117, msg);
                return false;
            }
            [ndarray readBytes:outputData[i] strideBytes:nil];
        }

        return true;

        } @catch (NSException* exception) {
            NSString* msg = [NSString stringWithFormat:@"MPSGraph execution threw exception: %@ - %@",
                exception.name, exception.reason];
            setError(error, 200, msg);
            return false;
        }
    }
}

// ===========================================================================
// Buffer Management
// ===========================================================================

MTLBufferHandle mpsgraph_buffer_create(MPSGraphContextHandle handle, int64_t nbytes, MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        id<MTLBuffer> buffer = [ctx.device newBufferWithLength:nbytes
                                                       options:MTLResourceStorageModeShared];
        if (!buffer) {
            setError(error, 120, @"Failed to create MTLBuffer");
            return NULL;
        }
        return (__bridge_retained void*)buffer;
    }
}

void* mpsgraph_buffer_contents(MTLBufferHandle handle) {
    if (!handle) return NULL;
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)handle;
    return buffer.contents;
}

int64_t mpsgraph_buffer_length(MTLBufferHandle handle) {
    if (!handle) return 0;
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)handle;
    return (int64_t)buffer.length;
}

void mpsgraph_buffer_destroy(MTLBufferHandle handle) {
    if (handle) {
        @autoreleasepool {
            id<MTLBuffer> buffer = (__bridge_transfer id<MTLBuffer>)handle;
            (void)buffer; // ARC releases
        }
    }
}

// ===========================================================================
// Softmax
// ===========================================================================

MPSGraphTensorHandle mpsgraph_softmax(MPSGraphContextHandle handle, MPSGraphTensorHandle x,
    int axis, MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        MPSGraphTensor* input = (__bridge MPSGraphTensor*)x;
        MPSGraphTensor* result = [ctx.graph softMaxWithTensor:input axis:axis name:nil];
        if (!result) {
            setError(error, 200, @"softmax failed");
            return NULL;
        }
        return (__bridge void*)result;
    }
}

// ===========================================================================
// Random Number Generation
// ===========================================================================

MPSGraphTensorHandle mpsgraph_random_uniform(MPSGraphContextHandle handle,
    int dtype, int64_t* shape, int rank, MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        MPSDataType mpsType = toMPSDataType(dtype);
        NSArray<NSNumber*>* shapeArr = shapeArray(shape, rank);
        // Generate random bits using Philox stateless op.
        MPSGraphRandomOpDescriptor* desc = [MPSGraphRandomOpDescriptor descriptorWithDistribution:MPSGraphRandomDistributionUniform
                                                                                         dataType:mpsType];
        MPSGraphTensor* result = [ctx.graph randomTensorWithShape:shapeArr
                                                            descriptor:desc
                                                                  name:nil];
        if (!result) {
            setError(error, 210, @"random uniform failed");
            return NULL;
        }
        return (__bridge void*)result;
    }
}

// ===========================================================================
// Pooling (ReduceWindow)
// ===========================================================================

MPSGraphTensorHandle mpsgraph_pool2d(MPSGraphContextHandle handle, MPSGraphTensorHandle x,
    int mode, int64_t* windowDims, int64_t* strides, int64_t* padBefore, int64_t* padAfter,
    MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        MPSGraphTensor* input = (__bridge MPSGraphTensor*)x;

        MPSGraphPooling2DOpDescriptor* desc = [MPSGraphPooling2DOpDescriptor
            descriptorWithKernelWidth:(NSUInteger)windowDims[1]
                        kernelHeight:(NSUInteger)windowDims[0]
                           strideInX:(NSUInteger)strides[1]
                           strideInY:(NSUInteger)strides[0]
                        paddingStyle:MPSGraphPaddingStyleExplicit
                          dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
        desc.paddingLeft = (NSUInteger)padBefore[1];
        desc.paddingRight = (NSUInteger)padAfter[1];
        desc.paddingTop = (NSUInteger)padBefore[0];
        desc.paddingBottom = (NSUInteger)padAfter[0];
        if (mode == 1) {
            desc.includeZeroPadToAverage = YES;
        }

        MPSGraphTensor* result = nil;
        switch (mode) {
            case 0: // Max pooling
                result = [ctx.graph maxPooling2DWithSourceTensor:input descriptor:desc name:nil];
                break;
            case 1: // Average pooling
                result = [ctx.graph avgPooling2DWithSourceTensor:input descriptor:desc name:nil];
                break;
            default:
                setError(error, 220, @"unsupported pooling mode");
                return NULL;
        }
        if (!result) {
            setError(error, 221, @"pooling failed");
            return NULL;
        }
        return (__bridge void*)result;
    }
}

// ===========================================================================
// Max Pool 2D Gradient (SelectAndScatter)
// ===========================================================================

MPSGraphTensorHandle mpsgraph_max_pool2d_gradient(MPSGraphContextHandle handle,
    MPSGraphTensorHandle gradientH, MPSGraphTensorHandle sourceH,
    int64_t* windowDims, int64_t* strides, int64_t* padBefore, int64_t* padAfter,
    MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        MPSGraphTensor* gradient = (__bridge MPSGraphTensor*)gradientH;
        MPSGraphTensor* source = (__bridge MPSGraphTensor*)sourceH;

        MPSGraphPooling2DOpDescriptor* desc = [MPSGraphPooling2DOpDescriptor
            descriptorWithKernelWidth:(NSUInteger)windowDims[1]
                        kernelHeight:(NSUInteger)windowDims[0]
                           strideInX:(NSUInteger)strides[1]
                           strideInY:(NSUInteger)strides[0]
                        paddingStyle:MPSGraphPaddingStyleExplicit
                          dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
        desc.paddingLeft = (NSUInteger)padBefore[1];
        desc.paddingRight = (NSUInteger)padAfter[1];
        desc.paddingTop = (NSUInteger)padBefore[0];
        desc.paddingBottom = (NSUInteger)padAfter[0];

        MPSGraphTensor* result = [ctx.graph maxPooling2DGradientWithGradientTensor:gradient
                                                                     sourceTensor:source
                                                                       descriptor:desc
                                                                             name:nil];
        if (!result) {
            setError(error, 230, @"max_pool2d_gradient failed");
            return NULL;
        }
        return (__bridge void*)result;
    }
}

// ===========================================================================
// General Convolution
// ===========================================================================

MPSGraphTensorHandle mpsgraph_conv_general(MPSGraphContextHandle handle,
    MPSGraphTensorHandle inputH, MPSGraphTensorHandle kernelH,
    int numSpatialDims,
    int64_t* strides, int64_t* dilations,
    int64_t* padBefore, int64_t* padAfter,
    int groups, MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        MPSGraphTensor* input = (__bridge MPSGraphTensor*)inputH;
        MPSGraphTensor* kernel = (__bridge MPSGraphTensor*)kernelH;

        if (numSpatialDims != 2) {
            setError(error, 230, @"conv_general: only 2D convolution supported");
            return NULL;
        }

        MPSGraphConvolution2DOpDescriptor* desc = [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:(NSUInteger)strides[1]
                          strideInY:(NSUInteger)strides[0]
                    dilationRateInX:(NSUInteger)dilations[1]
                    dilationRateInY:(NSUInteger)dilations[0]
                             groups:(NSUInteger)groups
                       paddingStyle:MPSGraphPaddingStyleExplicit
                         dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                      weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        desc.paddingLeft = (NSUInteger)padBefore[1];
        desc.paddingRight = (NSUInteger)padAfter[1];
        desc.paddingTop = (NSUInteger)padBefore[0];
        desc.paddingBottom = (NSUInteger)padAfter[0];

        MPSGraphTensor* result = [ctx.graph convolution2DWithSourceTensor:input
                                                           weightsTensor:kernel
                                                              descriptor:desc
                                                                    name:nil];
        if (!result) {
            setError(error, 231, @"convolution2D failed");
            return NULL;
        }
        return (__bridge void*)result;
    }
}

// ===========================================================================
// Scatter Along Axis
// ===========================================================================

MPSGraphTensorHandle mpsgraph_scatter_along_axis(MPSGraphContextHandle handle,
    MPSGraphTensorHandle dataH, MPSGraphTensorHandle indicesH, MPSGraphTensorHandle updatesH,
    int axis, int mode, MPSGraphError* error) {
    @autoreleasepool {
        clearError(error);
        MPSGraphContext* ctx = (__bridge MPSGraphContext*)handle;
        MPSGraphTensor* data = (__bridge MPSGraphTensor*)dataH;
        MPSGraphTensor* indices = (__bridge MPSGraphTensor*)indicesH;
        MPSGraphTensor* updates = (__bridge MPSGraphTensor*)updatesH;

        MPSGraphScatterMode scatterMode;
        if (!toScatterMode(mode, &scatterMode)) {
            setError(error, 240, @"unknown scatter mode");
            return NULL;
        }

        MPSGraphTensor* result = [ctx.graph scatterAlongAxis:axis
                                               withDataTensor:data
                                              updatesTensor:updates
                                              indicesTensor:indices
                                                       mode:scatterMode
                                                       name:nil];
        if (!result) {
            setError(error, 241, @"scatter along axis failed");
            return NULL;
        }
        return (__bridge void*)result;
    }
}
