// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0
//
// MLX bridge implementation - C++ wrapping Apple's MLX framework.

#include "bridge.h"

#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <string>

static void setError(MLXError* error, int code, const std::string& msg) {
    if (error) {
        error->code = code;
        error->message = strdup(msg.c_str());
    }
}

static void clearError(MLXError* error) {
    if (error) {
        error->code = 0;
        error->message = nullptr;
    }
}

static const char* kInstallHelpMsg =
    "MLX is not installed on this system; please install it via Homebrew:\n"
    "  brew install mlx\n"
    "or see https://ml-explore.github.io/mlx/build/html/install.html for installation instructions";

#if __has_include(<mlx/mlx.h>)
#define GOMLX_HAS_MLX_HEADERS 1
#include <mlx/mlx.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>
#else
#define GOMLX_HAS_MLX_HEADERS 0
#endif

#if GOMLX_HAS_MLX_HEADERS

namespace {

std::once_flag g_init_flag;
bool g_init_ok = false;
std::string g_init_error;

void ensureLoaded() {
    std::call_once(g_init_flag, []() {
        const char* env_path = std::getenv("MLX_LIBRARY_PATH");
        const char* candidates[] = {
            env_path,
            "libmlx.dylib",
            "/opt/homebrew/lib/libmlx.dylib",
            "/usr/local/lib/libmlx.dylib",
        };
        void* handle = nullptr;
        for (const char* path : candidates) {
            if (!path || path[0] == '\0') {
                continue;
            }
            handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
            if (handle != nullptr) {
                break;
            }
        }
        if (handle == nullptr) {
            g_init_ok = false;
            g_init_error = kInstallHelpMsg;
            return;
        }
        try {
            if (!mlx::core::metal::is_available()) {
                g_init_ok = false;
                g_init_error = "MLX Metal GPU backend is not available on this device";
                return;
            }
            mlx::core::set_default_device(mlx::core::Device(mlx::core::Device::gpu, 0));
            g_init_ok = true;
        } catch (const std::exception& e) {
            g_init_ok = false;
            g_init_error = std::string("MLX initialization failed: ") + e.what();
        }
    });
}

mlx::core::Dtype toMLXDType(int dtype, bool* ok) {
    if (ok) *ok = true;
    switch (dtype) {
        case MLX_DTYPE_BOOL:      return mlx::core::bool_;
        case MLX_DTYPE_INT8:      return mlx::core::int8;
        case MLX_DTYPE_INT16:     return mlx::core::int16;
        case MLX_DTYPE_INT32:     return mlx::core::int32;
        case MLX_DTYPE_INT64:     return mlx::core::int64;
        case MLX_DTYPE_FLOAT16:   return mlx::core::float16;
        case MLX_DTYPE_BFLOAT16:  return mlx::core::bfloat16;
        case MLX_DTYPE_FLOAT32:   return mlx::core::float32;
        case MLX_DTYPE_FLOAT64:   return mlx::core::float64;
        case MLX_DTYPE_UINT8:     return mlx::core::uint8;
        case MLX_DTYPE_UINT16:    return mlx::core::uint16;
        case MLX_DTYPE_UINT32:    return mlx::core::uint32;
        case MLX_DTYPE_UINT64:    return mlx::core::uint64;
        case MLX_DTYPE_COMPLEX64: return mlx::core::complex64;
        default:
            if (ok) *ok = false;
            return mlx::core::float32;
    }
}

int fromMLXDType(mlx::core::Dtype dt) {
    switch (dt.val()) {
        case mlx::core::Dtype::Val::bool_:     return MLX_DTYPE_BOOL;
        case mlx::core::Dtype::Val::int8:      return MLX_DTYPE_INT8;
        case mlx::core::Dtype::Val::int16:     return MLX_DTYPE_INT16;
        case mlx::core::Dtype::Val::int32:     return MLX_DTYPE_INT32;
        case mlx::core::Dtype::Val::int64:     return MLX_DTYPE_INT64;
        case mlx::core::Dtype::Val::float16:   return MLX_DTYPE_FLOAT16;
        case mlx::core::Dtype::Val::bfloat16:  return MLX_DTYPE_BFLOAT16;
        case mlx::core::Dtype::Val::float32:   return MLX_DTYPE_FLOAT32;
        case mlx::core::Dtype::Val::float64:   return MLX_DTYPE_FLOAT64;
        case mlx::core::Dtype::Val::uint8:     return MLX_DTYPE_UINT8;
        case mlx::core::Dtype::Val::uint16:    return MLX_DTYPE_UINT16;
        case mlx::core::Dtype::Val::uint32:    return MLX_DTYPE_UINT32;
        case mlx::core::Dtype::Val::uint64:    return MLX_DTYPE_UINT64;
        case mlx::core::Dtype::Val::complex64: return MLX_DTYPE_COMPLEX64;
    }
    return MLX_DTYPE_FLOAT32;
}

mlx::core::Shape toMLXShape(const int32_t* shape, int rank) {
    if (rank <= 0 || shape == nullptr) {
        return mlx::core::Shape{};
    }
    return mlx::core::Shape(shape, shape + rank);
}

enum class TapeOpKind {
    Parameter,
    SubParameter,
    Constant,
    Unary,
    Binary,
    ConvertDType,
    Bitcast,
    Iota,
    Reshape,
    Transpose,
    BroadcastInDim,
    Slice,
    Concatenate,
    Reverse,
    Pad,
    Where,
    Clamp,
    Reduce,
    ArgMinMax,
    Cumulative,
    Matmul,
    Gather,
    Scatter,
    DynamicSlice,
    DynamicUpdateSlice,
    ConvGeneral,
    AsStrided,
    SelectAndScatter,
    RNGNextState,
    RNGBits,
    Sort,
    Argsort,
    TakeAlongAxis,
    Softmax,
    LayerNorm,
};

struct TapeNode {
    TapeOpKind kind;
    int index = 0; // index into params or constants, or op code / axis
    int op_code = 0;
    bool flag1 = false;
    bool flag2 = false;
    float scalar_f32 = 0.0f;
    int64_t offset = 0;
    int32_t lhs = -1;
    int32_t rhs = -1;
    int32_t third = -1;
    mlx::core::Dtype dtype{mlx::core::float32};
    mlx::core::Shape shape;
    mlx::core::Strides strides;
    std::vector<int> axes;
    std::vector<int> extra_ints1;
    std::vector<int> extra_ints2;
    std::vector<int> extra_ints3;
    std::vector<int> extra_ints4;
    std::vector<int32_t> inputs;
    mlx::core::Shape slice_start;
    mlx::core::Shape slice_stop;
    mlx::core::Shape slice_strides;
};

mlx::core::Dtype toUnsignedIntDType(mlx::core::Dtype dt, int* bits) {
    switch (dt.val()) {
        case mlx::core::Dtype::Val::int8:
        case mlx::core::Dtype::Val::uint8:
            if (bits) *bits = 8;
            return mlx::core::uint8;
        case mlx::core::Dtype::Val::int16:
        case mlx::core::Dtype::Val::uint16:
            if (bits) *bits = 16;
            return mlx::core::uint16;
        case mlx::core::Dtype::Val::int32:
        case mlx::core::Dtype::Val::uint32:
            if (bits) *bits = 32;
            return mlx::core::uint32;
        case mlx::core::Dtype::Val::int64:
        case mlx::core::Dtype::Val::uint64:
            if (bits) *bits = 64;
            return mlx::core::uint64;
        default:
            if (bits) *bits = 32;
            return mlx::core::uint32;
    }
}

bool isSignedIntDType(mlx::core::Dtype dt) {
    switch (dt.val()) {
        case mlx::core::Dtype::Val::int8:
        case mlx::core::Dtype::Val::int16:
        case mlx::core::Dtype::Val::int32:
        case mlx::core::Dtype::Val::int64:
            return true;
        default:
            return false;
    }
}

bool isFloatDType(mlx::core::Dtype dt) {
    switch (dt.val()) {
        case mlx::core::Dtype::Val::float16:
        case mlx::core::Dtype::Val::bfloat16:
        case mlx::core::Dtype::Val::float32:
        case mlx::core::Dtype::Val::float64:
            return true;
        default:
            return false;
    }
}

mlx::core::array evalBitCount(const mlx::core::array& a) {
    int bits = 32;
    mlx::core::Dtype u_dt = toUnsignedIntDType(a.dtype(), &bits);
    mlx::core::array v = (a.dtype() == u_dt) ? a : mlx::core::astype(a, u_dt);

    if (bits == 8) {
        v = mlx::core::subtract(
            v,
            mlx::core::bitwise_and(
                mlx::core::right_shift(v, mlx::core::array(uint8_t(1))),
                mlx::core::array(uint8_t(0x55))));
        v = mlx::core::add(
            mlx::core::bitwise_and(v, mlx::core::array(uint8_t(0x33))),
            mlx::core::bitwise_and(
                mlx::core::right_shift(v, mlx::core::array(uint8_t(2))),
                mlx::core::array(uint8_t(0x33))));
        v = mlx::core::bitwise_and(
            mlx::core::add(v, mlx::core::right_shift(v, mlx::core::array(uint8_t(4)))),
            mlx::core::array(uint8_t(0x0F)));
    } else if (bits == 16) {
        v = mlx::core::subtract(
            v,
            mlx::core::bitwise_and(
                mlx::core::right_shift(v, mlx::core::array(uint16_t(1))),
                mlx::core::array(uint16_t(0x5555))));
        v = mlx::core::add(
            mlx::core::bitwise_and(v, mlx::core::array(uint16_t(0x3333))),
            mlx::core::bitwise_and(
                mlx::core::right_shift(v, mlx::core::array(uint16_t(2))),
                mlx::core::array(uint16_t(0x3333))));
        v = mlx::core::bitwise_and(
            mlx::core::add(v, mlx::core::right_shift(v, mlx::core::array(uint16_t(4)))),
            mlx::core::array(uint16_t(0x0F0F)));
        v = mlx::core::bitwise_and(
            mlx::core::add(v, mlx::core::right_shift(v, mlx::core::array(uint16_t(8)))),
            mlx::core::array(uint16_t(0x001F)));
    } else if (bits == 32) {
        v = mlx::core::subtract(
            v,
            mlx::core::bitwise_and(
                mlx::core::right_shift(v, mlx::core::array(uint32_t(1))),
                mlx::core::array(uint32_t(0x55555555u))));
        v = mlx::core::add(
            mlx::core::bitwise_and(v, mlx::core::array(uint32_t(0x33333333u))),
            mlx::core::bitwise_and(
                mlx::core::right_shift(v, mlx::core::array(uint32_t(2))),
                mlx::core::array(uint32_t(0x33333333u))));
        v = mlx::core::bitwise_and(
            mlx::core::add(v, mlx::core::right_shift(v, mlx::core::array(uint32_t(4)))),
            mlx::core::array(uint32_t(0x0F0F0F0Fu)));
        v = mlx::core::add(v, mlx::core::right_shift(v, mlx::core::array(uint32_t(8))));
        v = mlx::core::bitwise_and(
            mlx::core::add(v, mlx::core::right_shift(v, mlx::core::array(uint32_t(16)))),
            mlx::core::array(uint32_t(0x0000003Fu)));
    } else {
        v = mlx::core::subtract(
            v,
            mlx::core::bitwise_and(
                mlx::core::right_shift(v, mlx::core::array(uint64_t(1))),
                mlx::core::array(uint64_t(0x5555555555555555ULL))));
        v = mlx::core::add(
            mlx::core::bitwise_and(v, mlx::core::array(uint64_t(0x3333333333333333ULL))),
            mlx::core::bitwise_and(
                mlx::core::right_shift(v, mlx::core::array(uint64_t(2))),
                mlx::core::array(uint64_t(0x3333333333333333ULL))));
        v = mlx::core::bitwise_and(
            mlx::core::add(v, mlx::core::right_shift(v, mlx::core::array(uint64_t(4)))),
            mlx::core::array(uint64_t(0x0F0F0F0F0F0F0F0FULL)));
        v = mlx::core::add(v, mlx::core::right_shift(v, mlx::core::array(uint64_t(8))));
        v = mlx::core::add(v, mlx::core::right_shift(v, mlx::core::array(uint64_t(16))));
        v = mlx::core::bitwise_and(
            mlx::core::add(v, mlx::core::right_shift(v, mlx::core::array(uint64_t(32)))),
            mlx::core::array(uint64_t(0x000000000000007FULL)));
    }
    return (a.dtype() == u_dt) ? v : mlx::core::astype(v, a.dtype());
}

mlx::core::array evalClz(const mlx::core::array& a) {
    int bits = 32;
    mlx::core::Dtype u_dt = toUnsignedIntDType(a.dtype(), &bits);
    mlx::core::array u = (a.dtype() == u_dt) ? a : mlx::core::astype(a, u_dt);
    mlx::core::array zero = mlx::core::array(0, u_dt);
    mlx::core::array n = mlx::core::broadcast_to(mlx::core::array(bits, u_dt), u.shape());

    for (int s = bits / 2; s >= 1; s /= 2) {
        mlx::core::array shift_arr = mlx::core::array(s, u_dt);
        mlx::core::array y = mlx::core::right_shift(u, shift_arr);
        mlx::core::array cond = mlx::core::not_equal(y, zero);
        n = mlx::core::where(cond, mlx::core::subtract(n, shift_arr), n);
        u = mlx::core::where(cond, y, u);
    }
    mlx::core::array res = mlx::core::subtract(n, u);
    return (a.dtype() == u_dt) ? res : mlx::core::astype(res, a.dtype());
}

mlx::core::array evalUnaryOp(int op, const mlx::core::array& a) {
    switch (static_cast<MLXUnaryOp>(op)) {
        case MLX_UNARY_ABS: {
            mlx::core::array res = mlx::core::abs(a);
            if (a.dtype() == mlx::core::complex64 && res.dtype() != mlx::core::complex64) {
                res = mlx::core::astype(res, mlx::core::complex64);
            }
            return res;
        }
        case MLX_UNARY_NEG:         return mlx::core::negative(a);
        case MLX_UNARY_SQRT:        return mlx::core::sqrt(a);
        case MLX_UNARY_RSQRT:       return mlx::core::rsqrt(a);
        case MLX_UNARY_EXP:         return mlx::core::exp(a);
        case MLX_UNARY_EXPM1:       return mlx::core::expm1(a);
        case MLX_UNARY_LOG:         return mlx::core::log(a);
        case MLX_UNARY_LOG1P:       return mlx::core::log1p(a);
        case MLX_UNARY_SIN:         return mlx::core::sin(a);
        case MLX_UNARY_COS:         return mlx::core::cos(a);
        case MLX_UNARY_TANH:        return mlx::core::tanh(a);
        case MLX_UNARY_LOGISTIC:    return mlx::core::sigmoid(a);
        case MLX_UNARY_ERF:         return mlx::core::erf(a);
        case MLX_UNARY_FLOOR:       return mlx::core::floor(a);
        case MLX_UNARY_CEIL:        return mlx::core::ceil(a);
        case MLX_UNARY_ROUND:       return mlx::core::round(a);
        case MLX_UNARY_SIGN:        return mlx::core::sign(a);
        case MLX_UNARY_LOGICAL_NOT: return mlx::core::logical_not(a);
        case MLX_UNARY_BITWISE_NOT: return mlx::core::bitwise_invert(a);
        case MLX_UNARY_IS_FINITE:
            if (a.dtype() == mlx::core::complex64) {
                return mlx::core::logical_and(
                    mlx::core::isfinite(mlx::core::real(a)),
                    mlx::core::isfinite(mlx::core::imag(a)));
            }
            return mlx::core::isfinite(a);
        case MLX_UNARY_IS_NAN:
            if (a.dtype() == mlx::core::complex64) {
                return mlx::core::logical_or(
                    mlx::core::isnan(mlx::core::real(a)),
                    mlx::core::isnan(mlx::core::imag(a)));
            }
            return mlx::core::isnan(a);
        case MLX_UNARY_CONJ:        return mlx::core::conjugate(a);
        case MLX_UNARY_REAL:        return mlx::core::real(a);
        case MLX_UNARY_IMAG:        return mlx::core::imag(a);
        case MLX_UNARY_BIT_COUNT:   return evalBitCount(a);
        case MLX_UNARY_CLZ:         return evalClz(a);
        case MLX_UNARY_IDENTITY:    return a;
    }
    return a;
}

mlx::core::array evalIntTruncDiv(const mlx::core::array& lhs, const mlx::core::array& rhs) {
    if (!isSignedIntDType(lhs.dtype())) {
        return mlx::core::floor_divide(lhs, rhs);
    }
    mlx::core::Dtype u_dt = toUnsignedIntDType(lhs.dtype(), nullptr);
    mlx::core::array u_lhs = mlx::core::astype(mlx::core::abs(lhs), u_dt);
    mlx::core::array u_rhs = mlx::core::astype(mlx::core::abs(rhs), u_dt);
    mlx::core::array q = mlx::core::astype(mlx::core::floor_divide(u_lhs, u_rhs), lhs.dtype());
    mlx::core::array zero = mlx::core::array(0, lhs.dtype());
    mlx::core::array diff_sign = mlx::core::not_equal(
        mlx::core::less(lhs, zero),
        mlx::core::less(rhs, zero));
    return mlx::core::where(diff_sign, mlx::core::negative(q), q);
}

mlx::core::array evalBinaryOp(int op, const mlx::core::array& lhs, const mlx::core::array& rhs) {
    switch (static_cast<MLXBinaryOp>(op)) {
        case MLX_BINARY_ADD: return mlx::core::add(lhs, rhs);
        case MLX_BINARY_SUB: return mlx::core::subtract(lhs, rhs);
        case MLX_BINARY_MUL: return mlx::core::multiply(lhs, rhs);
        case MLX_BINARY_DIV: {
            if (isSignedIntDType(lhs.dtype()) ||
                lhs.dtype() == mlx::core::uint8 ||
                lhs.dtype() == mlx::core::uint16 ||
                lhs.dtype() == mlx::core::uint32 ||
                lhs.dtype() == mlx::core::uint64) {
                return evalIntTruncDiv(lhs, rhs);
            }
            return mlx::core::divide(lhs, rhs);
        }
        case MLX_BINARY_REM: {
            if (isSignedIntDType(lhs.dtype())) {
                mlx::core::array q = evalIntTruncDiv(lhs, rhs);
                return mlx::core::subtract(lhs, mlx::core::multiply(q, rhs));
            }
            if (isFloatDType(lhs.dtype())) {
                mlx::core::array q = mlx::core::divide(lhs, rhs);
                mlx::core::array zero = mlx::core::array(0.0f, lhs.dtype());
                mlx::core::array trunc_q = mlx::core::where(
                    mlx::core::greater_equal(q, zero),
                    mlx::core::floor(q),
                    mlx::core::ceil(q));
                return mlx::core::subtract(lhs, mlx::core::multiply(trunc_q, rhs));
            }
            return mlx::core::remainder(lhs, rhs);
        }
        case MLX_BINARY_POW:         return mlx::core::power(lhs, rhs);
        case MLX_BINARY_MAX:         return mlx::core::maximum(lhs, rhs);
        case MLX_BINARY_MIN:         return mlx::core::minimum(lhs, rhs);
        case MLX_BINARY_ATAN2:       return mlx::core::arctan2(lhs, rhs);
        case MLX_BINARY_LOGICAL_AND: return mlx::core::logical_and(lhs, rhs);
        case MLX_BINARY_LOGICAL_OR:  return mlx::core::logical_or(lhs, rhs);
        case MLX_BINARY_LOGICAL_XOR: return mlx::core::logical_xor(lhs, rhs);
        case MLX_BINARY_BITWISE_AND: return mlx::core::bitwise_and(lhs, rhs);
        case MLX_BINARY_BITWISE_OR:  return mlx::core::bitwise_or(lhs, rhs);
        case MLX_BINARY_BITWISE_XOR: return mlx::core::bitwise_xor(lhs, rhs);
        case MLX_BINARY_SHIFT_LEFT:  return mlx::core::left_shift(lhs, rhs);
        case MLX_BINARY_SHIFT_RIGHT_ARITHMETIC:
            return mlx::core::right_shift(lhs, rhs);
        case MLX_BINARY_SHIFT_RIGHT_LOGICAL: {
            if (isSignedIntDType(lhs.dtype())) {
                mlx::core::Dtype u_dt = toUnsignedIntDType(lhs.dtype(), nullptr);
                mlx::core::array u_lhs = mlx::core::astype(lhs, u_dt);
                mlx::core::array u_rhs = mlx::core::astype(rhs, u_dt);
                return mlx::core::astype(mlx::core::right_shift(u_lhs, u_rhs), lhs.dtype());
            }
            return mlx::core::right_shift(lhs, rhs);
        }
        case MLX_BINARY_COMPLEX: {
            mlx::core::array re = mlx::core::astype(lhs, mlx::core::complex64);
            mlx::core::array im = mlx::core::multiply(
                mlx::core::array(mlx::core::complex64_t{0.0f, 1.0f}),
                mlx::core::astype(rhs, mlx::core::complex64));
            return mlx::core::add(re, im);
        }
        case MLX_BINARY_EQUAL:            return mlx::core::equal(lhs, rhs);
        case MLX_BINARY_NOT_EQUAL:        return mlx::core::not_equal(lhs, rhs);
        case MLX_BINARY_LESS_THAN:        return mlx::core::less(lhs, rhs);
        case MLX_BINARY_LESS_OR_EQUAL:    return mlx::core::less_equal(lhs, rhs);
        case MLX_BINARY_GREATER_THAN:     return mlx::core::greater(lhs, rhs);
        case MLX_BINARY_GREATER_OR_EQUAL: return mlx::core::greater_equal(lhs, rhs);
    }
    return lhs;
}

mlx::core::array evalBitwiseReduce(int op, const mlx::core::array& a, const std::vector<int>& axes) {
    if (axes.empty() || a.ndim() == 0) {
        return a;
    }
    std::vector<int> sorted_axes = axes;
    std::sort(sorted_axes.begin(), sorted_axes.end(), std::greater<int>());

    auto applyOp = [op](const mlx::core::array& x, const mlx::core::array& y) {
        switch (static_cast<MLXReduceOp>(op)) {
            case MLX_REDUCE_BITWISE_AND: return mlx::core::bitwise_and(x, y);
            case MLX_REDUCE_BITWISE_OR:  return mlx::core::bitwise_or(x, y);
            case MLX_REDUCE_BITWISE_XOR: return mlx::core::bitwise_xor(x, y);
            default:                     return mlx::core::bitwise_and(x, y);
        }
    };

    mlx::core::array cur = a;
    for (int ax : sorted_axes) {
        int rank = static_cast<int>(cur.ndim());
        int32_t n = cur.shape(ax);
        while (n > 1) {
            int32_t half = n / 2;
            mlx::core::Shape start1(rank, 0), stop1 = cur.shape();
            mlx::core::Shape start2(rank, 0), stop2 = cur.shape();
            stop1[ax] = half;
            start2[ax] = half;
            stop2[ax] = 2 * half;
            mlx::core::array left = mlx::core::slice(cur, start1, stop1);
            mlx::core::array right = mlx::core::slice(cur, start2, stop2);
            mlx::core::array combined = applyOp(left, right);
            if (n % 2 == 1) {
                mlx::core::Shape start3(rank, 0), stop3 = cur.shape();
                start3[ax] = 2 * half;
                mlx::core::array tail = mlx::core::slice(cur, start3, stop3);
                cur = mlx::core::concatenate({combined, tail}, ax);
            } else {
                cur = std::move(combined);
            }
            n = cur.shape(ax);
        }
        cur = mlx::core::squeeze(cur, ax);
    }
    return cur;
}

mlx::core::array evalReduceOp(int op, const mlx::core::array& a, const std::vector<int>& axes) {
    if (axes.empty() || a.ndim() == 0) {
        return a;
    }
    switch (static_cast<MLXReduceOp>(op)) {
        case MLX_REDUCE_SUM: {
            mlx::core::array res = mlx::core::sum(a, axes, /*keepdims=*/false);
            if (res.dtype() != a.dtype()) {
                res = mlx::core::astype(res, a.dtype());
            }
            return res;
        }
        case MLX_REDUCE_PRODUCT: {
            mlx::core::array res = mlx::core::prod(a, axes, /*keepdims=*/false);
            if (res.dtype() != a.dtype()) {
                res = mlx::core::astype(res, a.dtype());
            }
            return res;
        }
        case MLX_REDUCE_MAX:
            return mlx::core::max(a, axes, /*keepdims=*/false);
        case MLX_REDUCE_MIN:
            return mlx::core::min(a, axes, /*keepdims=*/false);
        case MLX_REDUCE_LOGICAL_AND:
            return mlx::core::all(a, axes, /*keepdims=*/false);
        case MLX_REDUCE_LOGICAL_OR:
            return mlx::core::any(a, axes, /*keepdims=*/false);
        case MLX_REDUCE_LOGICAL_XOR: {
            mlx::core::array s = mlx::core::sum(
                mlx::core::astype(a, mlx::core::uint32), axes, /*keepdims=*/false);
            return mlx::core::not_equal(
                mlx::core::bitwise_and(s, mlx::core::array(uint32_t(1))),
                mlx::core::array(uint32_t(0)));
        }
        case MLX_REDUCE_BITWISE_AND:
        case MLX_REDUCE_BITWISE_OR:
        case MLX_REDUCE_BITWISE_XOR:
            return evalBitwiseReduce(op, a, axes);
    }
    return a;
}

mlx::core::array evalMatmul(const mlx::core::array& lhs, const mlx::core::array& rhs) {
    if (lhs.dtype() == mlx::core::complex64) {
        mlx::core::array a = mlx::core::real(lhs);
        mlx::core::array b = mlx::core::imag(lhs);
        mlx::core::array c = mlx::core::real(rhs);
        mlx::core::array d = mlx::core::imag(rhs);
        mlx::core::array re = mlx::core::subtract(
            mlx::core::matmul(a, c),
            mlx::core::matmul(b, d));
        mlx::core::array im = mlx::core::add(
            mlx::core::matmul(a, d),
            mlx::core::matmul(b, c));
        return evalBinaryOp(MLX_BINARY_COMPLEX, re, im);
    }
    if (!isFloatDType(lhs.dtype())) {
        // Integer matmul: lhs[..., M, K] @ rhs[..., K, N] via exact integer broadcast multiply + sum.
        mlx::core::array l_exp = mlx::core::expand_dims(lhs, -1); // [..., M, K, 1]
        mlx::core::array r_exp = mlx::core::expand_dims(rhs, -3); // [..., 1, K, N]
        mlx::core::array prod = mlx::core::multiply(l_exp, r_exp);
        mlx::core::array res = mlx::core::sum(prod, -2, /*keepdims=*/false);
        if (res.dtype() != lhs.dtype()) {
            res = mlx::core::astype(res, lhs.dtype());
        }
        return res;
    }
    return mlx::core::matmul(lhs, rhs);
}

mlx::core::array makeSelectPadScalar(mlx::core::Dtype dt, bool is_max) {
    switch (dt.val()) {
        case mlx::core::Dtype::Val::bool_:
            return mlx::core::array(!is_max);
        case mlx::core::Dtype::Val::int8:
            return mlx::core::array(is_max ? std::numeric_limits<int8_t>::lowest() : std::numeric_limits<int8_t>::max());
        case mlx::core::Dtype::Val::int16:
            return mlx::core::array(is_max ? std::numeric_limits<int16_t>::lowest() : std::numeric_limits<int16_t>::max());
        case mlx::core::Dtype::Val::int32:
            return mlx::core::array(is_max ? std::numeric_limits<int32_t>::lowest() : std::numeric_limits<int32_t>::max());
        case mlx::core::Dtype::Val::int64:
            return mlx::core::array(is_max ? std::numeric_limits<int64_t>::lowest() : std::numeric_limits<int64_t>::max());
        case mlx::core::Dtype::Val::uint8:
            return mlx::core::array(is_max ? uint8_t(0) : std::numeric_limits<uint8_t>::max());
        case mlx::core::Dtype::Val::uint16:
            return mlx::core::array(is_max ? uint16_t(0) : std::numeric_limits<uint16_t>::max());
        case mlx::core::Dtype::Val::uint32:
            return mlx::core::array(is_max ? uint32_t(0) : std::numeric_limits<uint32_t>::max());
        case mlx::core::Dtype::Val::uint64:
            return mlx::core::array(is_max ? uint64_t(0) : std::numeric_limits<uint64_t>::max());
        default: {
            float v = is_max ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
            return mlx::core::astype(mlx::core::array(v), dt);
        }
    }
}

mlx::core::array evalSelectAndScatter(
    const mlx::core::array& operand,
    const mlx::core::array& source,
    bool is_max,
    const mlx::core::Shape& win_dims,
    const mlx::core::Shape& win_strides,
    const std::vector<int>& pad_lo,
    const std::vector<int>& pad_hi) {
    int rank = static_cast<int>(operand.ndim());
    if (rank == 0) {
        return source;
    }

    std::vector<int> pad_axes;
    mlx::core::Shape low_pads, high_pads;
    for (int i = 0; i < rank; ++i) {
        if (pad_lo[i] > 0 || pad_hi[i] > 0) {
            pad_axes.push_back(i);
            low_pads.push_back(pad_lo[i]);
            high_pads.push_back(pad_hi[i]);
        }
    }

    mlx::core::array p = operand;
    if (!pad_axes.empty()) {
        mlx::core::array pad_val = makeSelectPadScalar(operand.dtype(), is_max);
        p = mlx::core::pad(operand, pad_axes, low_pads, high_pads, pad_val);
    }
    mlx::core::Shape p_shape = p.shape();
    int32_t pn = static_cast<int32_t>(p.size());
    if (pn == 0 || source.size() == 0) {
        return mlx::core::zeros(operand.shape(), operand.dtype());
    }

    // Fast path for non-overlapping windows (e.g. standard MaxPool / MinPool backward pass
    // where win_strides[i] == win_dims[i]). Uses a zero-copy reshape + hierarchical argmax/argmin
    // + elementwise where(), avoiding as_strided copies and atomic scatter_add.
    bool non_overlapping = true;
    mlx::core::Shape covered_shape(rank);
    bool needs_tail_pad = false;
    for (int i = 0; i < rank; ++i) {
        if (win_strides[i] != win_dims[i]) {
            non_overlapping = false;
            break;
        }
        covered_shape[i] = source.shape(i) * win_dims[i];
        if (covered_shape[i] > p_shape[i]) {
            non_overlapping = false;
            break;
        }
        if (covered_shape[i] < p_shape[i]) {
            needs_tail_pad = true;
        }
    }

    mlx::core::array out_padded = p;
    if (non_overlapping) {
        mlx::core::array p_cov = p;
        if (needs_tail_pad) {
            mlx::core::Shape zero_start(rank, 0);
            p_cov = mlx::core::slice(p, zero_start, covered_shape);
        }
        if (!p_cov.flags().row_contiguous) {
            p_cov = mlx::core::contiguous(p_cov);
        }

        mlx::core::Shape split_shape;
        mlx::core::Shape src_exp_shape;
        mlx::core::Shape r_grid_shape;
        std::vector<int> win_axes;
        int32_t w_prod = 1;
        for (int i = 0; i < rank; ++i) {
            split_shape.push_back(source.shape(i));
            src_exp_shape.push_back(source.shape(i));
            r_grid_shape.push_back(1);
            if (win_dims[i] > 1) {
                win_axes.push_back(static_cast<int>(split_shape.size()));
                split_shape.push_back(win_dims[i]);
                src_exp_shape.push_back(1);
                r_grid_shape.push_back(win_dims[i]);
                w_prod *= win_dims[i];
            }
        }

        mlx::core::array out_cov = source;
        if (!win_axes.empty()) {
            mlx::core::array p_split = mlx::core::reshape(p_cov, split_shape);
            mlx::core::array win_ext = is_max
                ? mlx::core::max(p_split, win_axes, /*keepdims=*/true)
                : mlx::core::min(p_split, win_axes, /*keepdims=*/true);
            mlx::core::array r_grid = mlx::core::reshape(
                mlx::core::arange(0, w_prod, mlx::core::int32),
                r_grid_shape);
            mlx::core::array cand_idx = mlx::core::where(
                mlx::core::equal(p_split, win_ext),
                r_grid,
                mlx::core::array(w_prod, mlx::core::int32));
            mlx::core::array first_idx = mlx::core::min(cand_idx, win_axes, /*keepdims=*/true);
            mlx::core::array mask = mlx::core::equal(r_grid, first_idx);

            mlx::core::array src_exp = mlx::core::reshape(source, src_exp_shape);
            mlx::core::array zero_val = mlx::core::astype(mlx::core::array(0), operand.dtype());
            mlx::core::array out_split = mlx::core::where(mask, src_exp, zero_val);
            out_cov = mlx::core::reshape(out_split, covered_shape);
        }

        if (needs_tail_pad) {
            std::vector<int> tail_axes;
            mlx::core::Shape tail_lo, tail_hi;
            for (int i = 0; i < rank; ++i) {
                if (covered_shape[i] < p_shape[i]) {
                    tail_axes.push_back(i);
                    tail_lo.push_back(0);
                    tail_hi.push_back(p_shape[i] - covered_shape[i]);
                }
            }
            mlx::core::array zero_val = mlx::core::astype(mlx::core::array(0), operand.dtype());
            out_padded = mlx::core::pad(out_cov, tail_axes, tail_lo, tail_hi, zero_val);
        } else {
            out_padded = std::move(out_cov);
        }
    } else {
        if (!p.flags().row_contiguous) {
            p = mlx::core::contiguous(p);
        }

        mlx::core::Strides row_strides(rank, 1);
        for (int i = rank - 2; i >= 0; --i) {
            row_strides[i] = row_strides[i + 1] * static_cast<int64_t>(p_shape[i + 1]);
        }

        mlx::core::Shape view_shape;
        mlx::core::Strides view_strides;
        mlx::core::Shape r_grid_shape;
        std::vector<int> win_axes;
        int32_t o_prod = 1;
        int32_t w_prod = 1;
        for (int i = 0; i < rank; ++i) {
            view_shape.push_back(source.shape(i));
            view_strides.push_back(row_strides[i] * static_cast<int64_t>(win_strides[i]));
            r_grid_shape.push_back(1);
            o_prod *= source.shape(i);
            if (win_dims[i] > 1) {
                win_axes.push_back(static_cast<int>(view_shape.size()));
                view_shape.push_back(win_dims[i]);
                view_strides.push_back(row_strides[i]);
                r_grid_shape.push_back(win_dims[i]);
                w_prod *= win_dims[i];
            }
        }

        mlx::core::array win_pos = mlx::core::zeros(mlx::core::Shape{o_prod}, mlx::core::int32);
        if (!win_axes.empty()) {
            mlx::core::array p_view = mlx::core::as_strided(p, view_shape, view_strides, 0);
            mlx::core::array win_ext = is_max
                ? mlx::core::max(p_view, win_axes, /*keepdims=*/true)
                : mlx::core::min(p_view, win_axes, /*keepdims=*/true);
            mlx::core::array r_grid = mlx::core::reshape(
                mlx::core::arange(0, w_prod, mlx::core::int32),
                r_grid_shape);
            mlx::core::array cand_idx = mlx::core::where(
                mlx::core::equal(p_view, win_ext),
                r_grid,
                mlx::core::array(w_prod, mlx::core::int32));
            win_pos = mlx::core::reshape(
                mlx::core::min(cand_idx, win_axes, /*keepdims=*/false),
                mlx::core::Shape{o_prod});
        }

        std::vector<int32_t> win_offsets(w_prod, 0);
        for (int32_t w = 0; w < w_prod; ++w) {
            int32_t rem = w;
            int32_t off = 0;
            for (int i = rank - 1; i >= 0; --i) {
                int32_t coord = rem % win_dims[i];
                rem /= win_dims[i];
                off += coord * static_cast<int32_t>(row_strides[i]);
            }
            win_offsets[w] = off;
        }
        mlx::core::array win_offsets_arr(win_offsets.data(), mlx::core::Shape{w_prod}, mlx::core::int32);
        mlx::core::array win_flat_idx = mlx::core::take(win_offsets_arr, win_pos, 0);

        mlx::core::array base_idx = mlx::core::array(0, mlx::core::int32);
        for (int i = 0; i < rank; ++i) {
            int32_t o_dim = source.shape(i);
            int32_t step = static_cast<int32_t>(row_strides[i] * static_cast<int64_t>(win_strides[i]));
            if (o_dim > 1 && step != 0) {
                mlx::core::Shape b_shape(rank, 1);
                b_shape[i] = o_dim;
                mlx::core::array axis_off = mlx::core::reshape(
                    mlx::core::arange(0, o_dim * step, step, mlx::core::int32),
                    b_shape);
                base_idx = (base_idx.ndim() == 0) ? axis_off : mlx::core::add(base_idx, axis_off);
            }
        }
        if (base_idx.ndim() > 0) {
            base_idx = mlx::core::reshape(
                mlx::core::broadcast_to(base_idx, source.shape()),
                mlx::core::Shape{o_prod});
            win_flat_idx = mlx::core::add(base_idx, win_flat_idx);
        }

        mlx::core::array zeros_flat = mlx::core::zeros(mlx::core::Shape{pn}, operand.dtype());
        mlx::core::array src_flat = mlx::core::reshape(source, mlx::core::Shape{o_prod});
        mlx::core::array out_flat = (operand.dtype() == mlx::core::complex64)
            ? evalBinaryOp(
                  MLX_BINARY_COMPLEX,
                  mlx::core::scatter_add_axis(mlx::core::real(zeros_flat), win_flat_idx, mlx::core::real(src_flat), 0),
                  mlx::core::scatter_add_axis(mlx::core::imag(zeros_flat), win_flat_idx, mlx::core::imag(src_flat), 0))
            : mlx::core::scatter_add_axis(zeros_flat, win_flat_idx, src_flat, 0);
        out_padded = mlx::core::reshape(out_flat, p_shape);
    }

    if (!pad_axes.empty()) {
        mlx::core::Shape slice_start(rank), slice_stop(rank);
        for (int i = 0; i < rank; ++i) {
            slice_start[i] = pad_lo[i];
            slice_stop[i] = pad_lo[i] + operand.shape(i);
        }
        return mlx::core::slice(out_padded, slice_start, slice_stop);
    }
    return out_padded;
}

mlx::core::array evalRNGBits(const mlx::core::array& state, const mlx::core::Shape& shape, mlx::core::Dtype dt) {
    mlx::core::array s_contig = state.flags().row_contiguous ? state : mlx::core::contiguous(state);
    mlx::core::array s32 = mlx::core::reshape(mlx::core::view(s_contig, mlx::core::uint32), {6});
    mlx::core::array k0 = mlx::core::slice(s32, {0}, {2});
    mlx::core::array k1 = mlx::core::slice(s32, {2}, {4});
    mlx::core::array k2 = mlx::core::slice(s32, {4}, {6});
    mlx::core::array key = mlx::core::bitwise_xor(mlx::core::bitwise_xor(k0, k1), k2);
    auto [next_key, subkey] = mlx::core::random::split(key);
    (void)next_key;

    if (dt == mlx::core::bool_) {
        mlx::core::array b = mlx::core::random::bits(shape, 1, subkey);
        return mlx::core::not_equal(
            mlx::core::bitwise_and(b, mlx::core::array(uint8_t(1))),
            mlx::core::array(uint8_t(0)));
    }
    if (dt == mlx::core::complex64) {
        auto [k_re, k_im] = mlx::core::random::split(subkey);
        mlx::core::array re = mlx::core::random::uniform(shape, mlx::core::float32, k_re);
        mlx::core::array im = mlx::core::random::uniform(shape, mlx::core::float32, k_im);
        return evalBinaryOp(MLX_BINARY_COMPLEX, re, im);
    }
    if (isFloatDType(dt)) {
        return mlx::core::random::uniform(shape, dt, subkey);
    }
    int width = static_cast<int>(dt.size());
    if (width == 8) {
        mlx::core::Shape shape32 = shape;
        shape32.push_back(2);
        mlx::core::array bits = mlx::core::random::bits(shape32, 4, subkey);
        bits = mlx::core::view(bits, dt);
        if (bits.shape() != shape) {
            bits = mlx::core::reshape(bits, shape);
        }
        return bits;
    }
    mlx::core::array bits = mlx::core::random::bits(shape, width, subkey);
    if (bits.dtype() != dt) {
        if (shape.empty()) {
            bits = mlx::core::reshape(bits, {1});
            bits = mlx::core::view(bits, dt);
            bits = mlx::core::reshape(bits, {});
        } else {
            bits = mlx::core::view(bits, dt);
        }
    }
    return bits;
}

struct MLXTape {
    int num_params = 0;
    std::vector<mlx::core::Dtype> param_dtypes;
    std::vector<mlx::core::Shape> param_shapes;
    std::vector<mlx::core::array> constants;
    std::vector<TapeNode> nodes;
    std::vector<uint8_t> reachable;

    void computeReachable(const std::vector<int32_t>& output_ids) {
        reachable.assign(nodes.size(), 0);
        for (int32_t id : output_ids) {
            if (id >= 0 && static_cast<size_t>(id) < nodes.size()) {
                reachable[id] = 1;
            }
        }
        for (int32_t i = static_cast<int32_t>(nodes.size()) - 1; i >= 0; --i) {
            if (!reachable[i]) {
                continue;
            }
            const auto& n = nodes[i];
            if (n.lhs >= 0) reachable[n.lhs] = 1;
            if (n.rhs >= 0) reachable[n.rhs] = 1;
            if (n.third >= 0) reachable[n.third] = 1;
            for (int32_t child : n.inputs) {
                if (child >= 0) reachable[child] = 1;
            }
            if (n.kind == TapeOpKind::BroadcastInDim || n.kind == TapeOpKind::Reshape) {
                for (int32_t d : n.shape) {
                    if (d <= -2) {
                        int32_t ref_node = (-d - 2) / 16;
                        if (ref_node >= 0 && static_cast<size_t>(ref_node) < nodes.size()) {
                            reachable[ref_node] = 1;
                        }
                    }
                }
            }
        }
    }

    std::vector<mlx::core::array> replay(
        const std::vector<mlx::core::array>& inputs,
        const std::vector<int32_t>& output_ids,
        bool eager_contiguous) const {
        std::vector<mlx::core::array> values(nodes.size(), mlx::core::array(0.0f));
        bool use_reachable = (reachable.size() == nodes.size());

        for (size_t idx_node = 0; idx_node < nodes.size(); ++idx_node) {
            if (use_reachable && !reachable[idx_node]) {
                continue;
            }
            const auto& node = nodes[idx_node];
            switch (node.kind) {
                case TapeOpKind::Parameter:
                    values[idx_node] = inputs.at(node.index);
                    break;
                case TapeOpKind::SubParameter:
                    break;
                case TapeOpKind::Constant:
                    values[idx_node] = constants.at(node.index);
                    break;
                case TapeOpKind::Unary:
                    values[idx_node] = evalUnaryOp(node.index, values.at(node.lhs));
                    break;
                case TapeOpKind::Binary:
                    values[idx_node] = evalBinaryOp(node.index, values.at(node.lhs), values.at(node.rhs));
                    break;
                case TapeOpKind::ConvertDType:
                    values[idx_node] = mlx::core::astype(values.at(node.lhs), node.dtype);
                    break;
                case TapeOpKind::Bitcast: {
                    mlx::core::array a = values.at(node.lhs);
                    if (!a.flags().row_contiguous) {
                        a = mlx::core::contiguous(a);
                    }
                    if (a.ndim() == 0 && node.dtype.size() != a.dtype().size()) {
                        a = mlx::core::reshape(a, {1});
                    }
                    mlx::core::array v = mlx::core::view(a, node.dtype);
                    if (v.shape() != node.shape) {
                        v = mlx::core::reshape(v, node.shape);
                    }
                    values[idx_node] = std::move(v);
                    break;
                }
                case TapeOpKind::Iota: {
                    int rank = static_cast<int>(node.shape.size());
                    int axis = node.index;
                    int32_t dim = node.shape[axis];
                    mlx::core::array r = (node.dtype == mlx::core::complex64)
                        ? mlx::core::astype(mlx::core::arange(static_cast<double>(dim), mlx::core::float32), mlx::core::complex64)
                        : mlx::core::arange(static_cast<double>(dim), node.dtype);
                    if (rank > 1) {
                        mlx::core::Shape reshape_dims(rank, 1);
                        reshape_dims[axis] = dim;
                        r = mlx::core::reshape(r, std::move(reshape_dims));
                        r = mlx::core::broadcast_to(r, node.shape);
                    }
                    values[idx_node] = std::move(r);
                    break;
                }
                case TapeOpKind::Reshape: {
                    mlx::core::Shape target_shape = node.shape;
                    for (size_t i = 0; i < target_shape.size(); ++i) {
                        if (target_shape[i] <= -2) {
                            int32_t enc = -target_shape[i] - 2;
                            int32_t ref_node = enc / 16;
                            int ref_axis = enc % 16;
                            target_shape[i] = values.at(ref_node).shape(ref_axis);
                        }
                    }
                    values[idx_node] = mlx::core::reshape(values.at(node.lhs), std::move(target_shape));
                    break;
                }
                case TapeOpKind::Transpose:
                    if (node.axes.empty()) {
                        values[idx_node] = values.at(node.lhs);
                    } else {
                        values[idx_node] = mlx::core::transpose(values.at(node.lhs), node.axes);
                    }
                    break;
                case TapeOpKind::BroadcastInDim: {
                    const mlx::core::array& a = values.at(node.lhs);
                    if (node.shape.empty()) {
                        values[idx_node] = a;
                        break;
                    }
                    mlx::core::Shape inter_shape(node.shape.size(), 1);
                    for (size_t i = 0; i < node.axes.size(); ++i) {
                        inter_shape[node.axes[i]] = a.shape(static_cast<int>(i));
                    }
                    mlx::core::array r = (a.shape() == inter_shape)
                        ? a
                        : mlx::core::reshape(a, std::move(inter_shape));
                    mlx::core::Shape target_shape = node.shape;
                    for (size_t i = 0; i < target_shape.size(); ++i) {
                        if (target_shape[i] == -1) {
                            target_shape[i] = r.shape(static_cast<int>(i));
                        } else if (target_shape[i] <= -2) {
                            int32_t enc = -target_shape[i] - 2;
                            int32_t ref_node = enc / 16;
                            int ref_axis = enc % 16;
                            target_shape[i] = values.at(ref_node).shape(ref_axis);
                        }
                    }
                    if (r.shape() != target_shape) {
                        r = mlx::core::broadcast_to(r, target_shape);
                    }
                    values[idx_node] = std::move(r);
                    break;
                }
                case TapeOpKind::Slice: {
                    const mlx::core::array& a = values.at(node.lhs);
                    mlx::core::Shape stop = node.slice_stop;
                    for (size_t i = 0; i < stop.size(); ++i) {
                        if (stop[i] < 0) {
                            stop[i] = a.shape(static_cast<int>(i));
                        }
                    }
                    values[idx_node] = mlx::core::slice(a, node.slice_start, std::move(stop), node.slice_strides);
                    break;
                }
                case TapeOpKind::Concatenate: {
                    std::vector<mlx::core::array> arrs;
                    arrs.reserve(node.inputs.size());
                    for (int32_t id : node.inputs) {
                        arrs.push_back(values.at(id));
                    }
                    values[idx_node] = mlx::core::concatenate(std::move(arrs), node.index);
                    break;
                }
                case TapeOpKind::Reverse:
                    if (node.axes.empty()) {
                        values[idx_node] = values.at(node.lhs);
                    } else {
                        values[idx_node] = mlx::core::flip(values.at(node.lhs), node.axes);
                    }
                    break;
                case TapeOpKind::Pad:
                    values[idx_node] = mlx::core::pad(
                        values.at(node.lhs),
                        node.axes,
                        node.slice_start,
                        node.slice_stop,
                        values.at(node.rhs));
                    break;
                case TapeOpKind::Where:
                    values[idx_node] = mlx::core::where(
                        values.at(node.lhs),
                        values.at(node.rhs),
                        values.at(node.third));
                    break;
                case TapeOpKind::Clamp:
                    values[idx_node] = mlx::core::clip(
                        values.at(node.lhs),
                        values.at(node.rhs),
                        values.at(node.third));
                    break;
                case TapeOpKind::Reduce:
                    values[idx_node] = evalReduceOp(node.index, values.at(node.lhs), node.axes);
                    break;
                case TapeOpKind::ArgMinMax: {
                    const mlx::core::array& a = values.at(node.lhs);
                    mlx::core::array res = node.flag1
                        ? mlx::core::argmin(a, node.index, /*keepdims=*/false)
                        : mlx::core::argmax(a, node.index, /*keepdims=*/false);
                    if (res.dtype() != node.dtype) {
                        res = mlx::core::astype(res, node.dtype);
                    }
                    values[idx_node] = std::move(res);
                    break;
                }
                case TapeOpKind::Cumulative: {
                    const mlx::core::array& a = values.at(node.lhs);
                    int axis = node.index;
                    bool reverse = node.flag1;
                    bool inclusive = !node.flag2;
                    mlx::core::array res = a;
                    if (node.op_code == MLX_CUMULATIVE_SUM) {
                        if (a.dtype() == mlx::core::complex64) {
                            mlx::core::array re = mlx::core::cumsum(
                                mlx::core::real(a), axis, reverse, inclusive, mlx::core::float32);
                            mlx::core::array im = mlx::core::cumsum(
                                mlx::core::imag(a), axis, reverse, inclusive, mlx::core::float32);
                            res = evalBinaryOp(MLX_BINARY_COMPLEX, re, im);
                        } else {
                            res = mlx::core::cumsum(a, axis, reverse, inclusive, a.dtype());
                            if (res.dtype() != a.dtype()) {
                                res = mlx::core::astype(res, a.dtype());
                            }
                        }
                    } else {
                        res = mlx::core::cummax(a, axis, reverse, inclusive);
                        if (res.dtype() != a.dtype()) {
                            res = mlx::core::astype(res, a.dtype());
                        }
                    }
                    values[idx_node] = std::move(res);
                    break;
                }
                case TapeOpKind::Matmul:
                    values[idx_node] = evalMatmul(values.at(node.lhs), values.at(node.rhs));
                    break;
                case TapeOpKind::Gather: {
                    const mlx::core::array& a = values.at(node.lhs);
                    const mlx::core::array& idx = values.at(node.rhs);
                    std::vector<mlx::core::array> indices_vec;
                    if (!node.axes.empty()) {
                        if (node.index == static_cast<int>(idx.ndim())) {
                            indices_vec.push_back(idx);
                        } else {
                            indices_vec = mlx::core::unstack(idx, node.index);
                        }
                    }
                    values[idx_node] = mlx::core::gather(a, indices_vec, node.axes, node.shape);
                    break;
                }
                case TapeOpKind::Scatter: {
                    const mlx::core::array& a = values.at(node.lhs);
                    const mlx::core::array& idx = values.at(node.rhs);
                    const mlx::core::array& upd = values.at(node.third);
                    std::vector<mlx::core::array> indices_vec;
                    if (!node.axes.empty()) {
                        if (node.index == static_cast<int>(idx.ndim())) {
                            indices_vec.push_back(idx);
                        } else {
                            indices_vec = mlx::core::unstack(idx, node.index);
                        }
                    }
                    mlx::core::array res = a;
                    switch (static_cast<MLXScatterOp>(node.op_code)) {
                        case MLX_SCATTER_SUM:
                            if (a.dtype() == mlx::core::complex64) {
                                mlx::core::array re = mlx::core::scatter_add(
                                    mlx::core::real(a), indices_vec, mlx::core::real(upd), node.axes);
                                mlx::core::array im = mlx::core::scatter_add(
                                    mlx::core::imag(a), indices_vec, mlx::core::imag(upd), node.axes);
                                res = evalBinaryOp(MLX_BINARY_COMPLEX, re, im);
                            } else {
                                res = mlx::core::scatter_add(a, indices_vec, upd, node.axes);
                            }
                            break;
                        case MLX_SCATTER_MAX:
                            res = mlx::core::scatter_max(a, indices_vec, upd, node.axes);
                            break;
                        case MLX_SCATTER_MIN:
                            res = mlx::core::scatter_min(a, indices_vec, upd, node.axes);
                            break;
                    }
                    values[idx_node] = std::move(res);
                    break;
                }
                case TapeOpKind::DynamicSlice: {
                    const mlx::core::array& a = values.at(node.lhs);
                    const mlx::core::array& start = values.at(node.rhs);
                    int rank = static_cast<int>(node.shape.size());
                    if (rank == 0) {
                        values[idx_node] = a;
                        break;
                    }
                    mlx::core::Shape max_starts(rank, 0);
                    for (int i = 0; i < rank; ++i) {
                        int32_t m = a.shape(i) - node.shape[i];
                        max_starts[i] = m > 0 ? m : 0;
                    }
                    mlx::core::array max_arr = mlx::core::array(max_starts.data(), mlx::core::Shape{rank}, mlx::core::int32);
                    mlx::core::array zero_arr = mlx::core::array(0, start.dtype());
                    mlx::core::array max_cast = (start.dtype() == mlx::core::int32)
                        ? max_arr
                        : mlx::core::astype(max_arr, start.dtype());
                    mlx::core::array clamped = mlx::core::clip(start, zero_arr, max_cast);
                    if (clamped.dtype() != mlx::core::int32) {
                        clamped = mlx::core::astype(clamped, mlx::core::int32);
                    }
                    values[idx_node] = mlx::core::slice(a, clamped, node.axes, node.shape);
                    break;
                }
                case TapeOpKind::DynamicUpdateSlice: {
                    const mlx::core::array& a = values.at(node.lhs);
                    const mlx::core::array& upd = values.at(node.rhs);
                    const mlx::core::array& start = values.at(node.third);
                    int rank = static_cast<int>(node.axes.size());
                    if (rank == 0) {
                        values[idx_node] = upd;
                        break;
                    }
                    mlx::core::Shape max_starts(rank, 0);
                    for (int i = 0; i < rank; ++i) {
                        int32_t m = a.shape(i) - upd.shape(i);
                        max_starts[i] = m > 0 ? m : 0;
                    }
                    mlx::core::array max_arr = mlx::core::array(max_starts.data(), mlx::core::Shape{rank}, mlx::core::int32);
                    mlx::core::array zero_arr = mlx::core::array(0, start.dtype());
                    mlx::core::array max_cast = (start.dtype() == mlx::core::int32)
                        ? max_arr
                        : mlx::core::astype(max_arr, start.dtype());
                    mlx::core::array clamped = mlx::core::clip(start, zero_arr, max_cast);
                    if (clamped.dtype() != mlx::core::int32) {
                        clamped = mlx::core::astype(clamped, mlx::core::int32);
                    }
                    values[idx_node] = mlx::core::slice_update(a, upd, clamped, node.axes);
                    break;
                }
                case TapeOpKind::ConvGeneral:
                    values[idx_node] = mlx::core::conv_general(
                        values.at(node.lhs),
                        values.at(node.rhs),
                        node.axes,
                        node.extra_ints1,
                        node.extra_ints2,
                        node.extra_ints3,
                        node.extra_ints4,
                        node.index,
                        /*flip=*/false);
                    break;
                case TapeOpKind::AsStrided: {
                    mlx::core::array a = values.at(node.lhs);
                    if (!a.flags().row_contiguous) {
                        a = mlx::core::contiguous(a);
                    }
                    mlx::core::array s = mlx::core::as_strided(
                        a, node.shape, node.strides, static_cast<size_t>(node.offset));
                    if (!node.strides.empty() && node.strides.back() != 1) {
                        s = mlx::core::contiguous(s);
                    }
                    values[idx_node] = std::move(s);
                    break;
                }
                case TapeOpKind::SelectAndScatter:
                    values[idx_node] = evalSelectAndScatter(
                        values.at(node.lhs),
                        values.at(node.rhs),
                        node.flag1,
                        node.slice_start,
                        node.slice_strides,
                        node.extra_ints1,
                        node.extra_ints2);
                    break;
                case TapeOpKind::RNGNextState:
                    values[idx_node] = mlx::core::add(
                        values.at(node.lhs),
                        mlx::core::array(uint64_t(1)));
                    break;
                case TapeOpKind::RNGBits:
                    values[idx_node] = evalRNGBits(values.at(node.lhs), node.shape, node.dtype);
                    break;
                case TapeOpKind::Sort:
                    values[idx_node] = mlx::core::sort(values.at(node.lhs), node.index);
                    break;
                case TapeOpKind::Argsort:
                    values[idx_node] = mlx::core::argsort(values.at(node.lhs), node.index);
                    break;
                case TapeOpKind::TakeAlongAxis:
                    values[idx_node] = mlx::core::take_along_axis(
                        values.at(node.lhs),
                        values.at(node.rhs),
                        node.index);
                    break;
                case TapeOpKind::Softmax:
                    values[idx_node] = mlx::core::softmax(
                        values.at(node.lhs),
                        node.index,
                        /*precise=*/true);
                    break;
                case TapeOpKind::LayerNorm: {
                    std::optional<mlx::core::array> w = std::nullopt;
                    std::optional<mlx::core::array> b = std::nullopt;
                    if (node.rhs >= 0) {
                        w = values.at(node.rhs);
                    }
                    if (node.third >= 0) {
                        b = values.at(node.third);
                    }
                    values[idx_node] = mlx::core::fast::layer_norm(
                        values.at(node.lhs),
                        w,
                        b,
                        node.scalar_f32);
                    break;
                }
            }
        }

        std::vector<mlx::core::array> outputs;
        outputs.reserve(output_ids.size());
        for (size_t i = 0; i < output_ids.size(); ++i) {
            int32_t id = output_ids[i];
            mlx::core::array out = values.at(id);
            bool is_dup = false;
            for (size_t j = 0; j < i; ++j) {
                if (output_ids[j] == id) {
                    is_dup = true;
                    break;
                }
            }
            const auto& node = nodes.at(id);
            if (node.kind == TapeOpKind::Parameter || node.kind == TapeOpKind::Constant || is_dup) {
                out = mlx::core::copy(out);
            } else if (eager_contiguous) {
                out = mlx::core::contiguous(out);
            }
            outputs.push_back(std::move(out));
        }
        return outputs;
    }
};

struct MLXExec {
    std::shared_ptr<MLXTape> tape;
    std::vector<int32_t> output_ids;
    bool compile_graph = true;
    bool eager_contiguous = true;
    std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array>&)> compiled_fn;
};

} // namespace

extern "C" {

bool mlx_bridge_init(MLXError* error) {
    clearError(error);
    ensureLoaded();
    if (!g_init_ok) {
        setError(error, 1, g_init_error);
        return false;
    }
    return true;
}

const char* mlx_device_name(MLXError* error) {
    clearError(error);
    if (!mlx_bridge_init(error)) {
        return nullptr;
    }
    try {
        const auto& info = mlx::core::device_info(mlx::core::Device(mlx::core::Device::gpu, 0));
        auto it = info.find("device_name");
        if (it != info.end() && std::holds_alternative<std::string>(it->second)) {
            return strdup(std::get<std::string>(it->second).c_str());
        }
        return strdup("Apple Silicon GPU");
    } catch (const std::exception& e) {
        setError(error, 2, std::string("mlx_device_name: ") + e.what());
        return nullptr;
    }
}

const char* mlx_version_string(MLXError* error) {
    clearError(error);
    if (!mlx_bridge_init(error)) {
        return nullptr;
    }
    try {
        const char* v = mlx::core::version();
        return strdup(v ? v : "unknown");
    } catch (const std::exception& e) {
        setError(error, 2, std::string("mlx_version_string: ") + e.what());
        return nullptr;
    }
}

MLXArrayHandle mlx_array_create(int64_t nbytes, int dtype,
    const int32_t* shape, int rank, MLXError* error) {
    clearError(error);
    if (!mlx_bridge_init(error)) {
        return nullptr;
    }
    bool ok = false;
    mlx::core::Dtype dt = toMLXDType(dtype, &ok);
    if (!ok) {
        setError(error, 3, "mlx_array_create: unsupported dtype");
        return nullptr;
    }
    try {
        mlx::core::Shape s = toMLXShape(shape, rank);
        size_t alloc_bytes = nbytes > 0 ? static_cast<size_t>(nbytes) : 0;
        mlx::core::allocator::Data data = mlx::core::allocator::malloc(alloc_bytes);
        auto* arr = new mlx::core::array(std::move(data), std::move(s), dt);
        return static_cast<MLXArrayHandle>(arr);
    } catch (const std::exception& e) {
        setError(error, 4, std::string("mlx_array_create: ") + e.what());
        return nullptr;
    }
}

MLXArrayHandle mlx_array_from_data(const void* src, int64_t nbytes, int dtype,
    const int32_t* shape, int rank, MLXError* error) {
    clearError(error);
    if (!mlx_bridge_init(error)) {
        return nullptr;
    }
    bool ok = false;
    mlx::core::Dtype dt = toMLXDType(dtype, &ok);
    if (!ok) {
        setError(error, 3, "mlx_array_from_data: unsupported dtype");
        return nullptr;
    }
    try {
        mlx::core::Shape s = toMLXShape(shape, rank);
        size_t alloc_bytes = nbytes > 0 ? static_cast<size_t>(nbytes) : 0;
        mlx::core::allocator::Data data = mlx::core::allocator::malloc(alloc_bytes);
        if (alloc_bytes > 0 && src != nullptr) {
            std::memcpy(data.buffer().raw_ptr(), src, alloc_bytes);
        }
        auto* arr = new mlx::core::array(std::move(data), std::move(s), dt);
        return static_cast<MLXArrayHandle>(arr);
    } catch (const std::exception& e) {
        setError(error, 4, std::string("mlx_array_from_data: ") + e.what());
        return nullptr;
    }
}

void* mlx_array_data_ptr(MLXArrayHandle handle) {
    if (!handle) {
        return nullptr;
    }
    auto* arr = static_cast<mlx::core::array*>(handle);
    if (arr->nbytes() == 0) {
        return nullptr;
    }
    return arr->data<void>();
}

int64_t mlx_array_nbytes(MLXArrayHandle handle) {
    if (!handle) {
        return 0;
    }
    auto* arr = static_cast<mlx::core::array*>(handle);
    return static_cast<int64_t>(arr->nbytes());
}

int mlx_array_dtype(MLXArrayHandle handle) {
    if (!handle) {
        return MLX_DTYPE_FLOAT32;
    }
    auto* arr = static_cast<mlx::core::array*>(handle);
    return fromMLXDType(arr->dtype());
}

int mlx_array_ndim(MLXArrayHandle handle) {
    if (!handle) {
        return 0;
    }
    auto* arr = static_cast<mlx::core::array*>(handle);
    return static_cast<int>(arr->ndim());
}

void mlx_array_shape(MLXArrayHandle handle, int32_t* out_shape) {
    if (!handle || !out_shape) {
        return;
    }
    auto* arr = static_cast<mlx::core::array*>(handle);
    const auto& s = arr->shape();
    for (size_t i = 0; i < s.size(); ++i) {
        out_shape[i] = s[i];
    }
}

bool mlx_array_is_row_contiguous(MLXArrayHandle handle) {
    if (!handle) {
        return false;
    }
    auto* arr = static_cast<mlx::core::array*>(handle);
    return arr->flags().row_contiguous && (arr->data_size() == arr->size());
}

bool mlx_array_ensure_contiguous(MLXArrayHandle handle, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_array_ensure_contiguous: null array handle");
        return false;
    }
    try {
        auto* arr = static_cast<mlx::core::array*>(handle);
        if (!arr->flags().row_contiguous || arr->data_size() != arr->size()) {
            *arr = mlx::core::contiguous(*arr);
        }
        if (arr->status() == mlx::core::array::Status::unscheduled || !arr->is_available()) {
            mlx::core::eval({*arr});
        }
        return true;
    } catch (const std::exception& e) {
        setError(error, 5, std::string("mlx_array_ensure_contiguous: ") + e.what());
        return false;
    }
}

void mlx_array_destroy(MLXArrayHandle handle) {
    if (handle) {
        delete static_cast<mlx::core::array*>(handle);
    }
}

MLXTapeHandle mlx_tape_create(MLXError* error) {
    clearError(error);
    if (!mlx_bridge_init(error)) {
        return nullptr;
    }
    return static_cast<MLXTapeHandle>(new MLXTape());
}

void mlx_tape_destroy(MLXTapeHandle handle) {
    if (handle) {
        delete static_cast<MLXTape*>(handle);
    }
}

int32_t mlx_tape_add_parameter(MLXTapeHandle handle, int dtype,
    const int32_t* shape, int rank, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_parameter: null tape handle");
        return -1;
    }
    bool ok = false;
    mlx::core::Dtype dt = toMLXDType(dtype, &ok);
    if (!ok) {
        setError(error, 3, "mlx_tape_add_parameter: unsupported dtype");
        return -1;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    int param_idx = tape->num_params++;
    mlx::core::Shape s = toMLXShape(shape, rank);
    tape->param_dtypes.push_back(dt);
    tape->param_shapes.push_back(s);

    int32_t node_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode node;
    node.kind = TapeOpKind::Parameter;
    node.index = param_idx;
    node.dtype = dt;
    node.shape = std::move(s);
    tape->nodes.push_back(std::move(node));
    return node_id;
}

int32_t mlx_tape_add_constant(MLXTapeHandle handle, const void* src, int64_t nbytes,
    int dtype, const int32_t* shape, int rank, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_constant: null tape handle");
        return -1;
    }
    bool ok = false;
    mlx::core::Dtype dt = toMLXDType(dtype, &ok);
    if (!ok) {
        setError(error, 3, "mlx_tape_add_constant: unsupported dtype");
        return -1;
    }
    try {
        auto* tape = static_cast<MLXTape*>(handle);
        mlx::core::Shape s = toMLXShape(shape, rank);
        size_t alloc_bytes = nbytes > 0 ? static_cast<size_t>(nbytes) : 0;
        mlx::core::allocator::Data data = mlx::core::allocator::malloc(alloc_bytes);
        if (alloc_bytes > 0 && src != nullptr) {
            std::memcpy(data.buffer().raw_ptr(), src, alloc_bytes);
        }
        int const_idx = static_cast<int>(tape->constants.size());
        tape->constants.emplace_back(std::move(data), s, dt);

        int32_t node_id = static_cast<int32_t>(tape->nodes.size());
        TapeNode node;
        node.kind = TapeOpKind::Constant;
        node.index = const_idx;
        node.dtype = dt;
        node.shape = std::move(s);
        tape->nodes.push_back(std::move(node));
        return node_id;
    } catch (const std::exception& e) {
        setError(error, 4, std::string("mlx_tape_add_constant: ") + e.what());
        return -1;
    }
}

int32_t mlx_tape_add_unary(MLXTapeHandle handle, int op, int32_t input, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_unary: null tape handle");
        return -1;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    if (input < 0 || static_cast<size_t>(input) >= tape->nodes.size()) {
        setError(error, 2, "mlx_tape_add_unary: invalid input node ID");
        return -1;
    }
    int32_t node_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode node;
    node.kind = TapeOpKind::Unary;
    node.index = op;
    node.lhs = input;
    tape->nodes.push_back(std::move(node));
    return node_id;
}

int32_t mlx_tape_add_binary(MLXTapeHandle handle, int op, int32_t lhs, int32_t rhs, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_binary: null tape handle");
        return -1;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    if (lhs < 0 || static_cast<size_t>(lhs) >= tape->nodes.size() ||
        rhs < 0 || static_cast<size_t>(rhs) >= tape->nodes.size()) {
        setError(error, 2, "mlx_tape_add_binary: invalid operand node ID");
        return -1;
    }
    int32_t node_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode node;
    node.kind = TapeOpKind::Binary;
    node.index = op;
    node.lhs = lhs;
    node.rhs = rhs;
    tape->nodes.push_back(std::move(node));
    return node_id;
}

int32_t mlx_tape_add_convert_dtype(MLXTapeHandle handle, int32_t input, int target_dtype, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_convert_dtype: null tape handle");
        return -1;
    }
    bool ok = false;
    mlx::core::Dtype dt = toMLXDType(target_dtype, &ok);
    if (!ok) {
        setError(error, 3, "mlx_tape_add_convert_dtype: unsupported target dtype");
        return -1;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    if (input < 0 || static_cast<size_t>(input) >= tape->nodes.size()) {
        setError(error, 2, "mlx_tape_add_convert_dtype: invalid input node ID");
        return -1;
    }
    int32_t node_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode node;
    node.kind = TapeOpKind::ConvertDType;
    node.lhs = input;
    node.dtype = dt;
    tape->nodes.push_back(std::move(node));
    return node_id;
}

int32_t mlx_tape_add_bitcast(MLXTapeHandle handle, int32_t input, int target_dtype,
    const int32_t* out_shape, int out_rank, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_bitcast: null tape handle");
        return -1;
    }
    bool ok = false;
    mlx::core::Dtype dt = toMLXDType(target_dtype, &ok);
    if (!ok) {
        setError(error, 3, "mlx_tape_add_bitcast: unsupported target dtype");
        return -1;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    if (input < 0 || static_cast<size_t>(input) >= tape->nodes.size()) {
        setError(error, 2, "mlx_tape_add_bitcast: invalid input node ID");
        return -1;
    }
    mlx::core::Shape s = toMLXShape(out_shape, out_rank);
    int32_t node_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode node;
    node.kind = TapeOpKind::Bitcast;
    node.lhs = input;
    node.dtype = dt;
    node.shape = std::move(s);
    tape->nodes.push_back(std::move(node));
    return node_id;
}

int32_t mlx_tape_add_iota(MLXTapeHandle handle, int dtype,
    const int32_t* shape, int rank, int iota_axis, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_iota: null tape handle");
        return -1;
    }
    bool ok = false;
    mlx::core::Dtype dt = toMLXDType(dtype, &ok);
    if (!ok) {
        setError(error, 3, "mlx_tape_add_iota: unsupported dtype");
        return -1;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    mlx::core::Shape s = toMLXShape(shape, rank);
    int32_t node_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode node;
    node.kind = TapeOpKind::Iota;
    node.index = iota_axis;
    node.dtype = dt;
    node.shape = std::move(s);
    tape->nodes.push_back(std::move(node));
    return node_id;
}

int32_t mlx_tape_add_reshape(MLXTapeHandle handle, int32_t input,
    const int32_t* shape, int rank, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_reshape: null tape handle");
        return -1;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    if (input < 0 || static_cast<size_t>(input) >= tape->nodes.size()) {
        setError(error, 2, "mlx_tape_add_reshape: invalid input node ID");
        return -1;
    }
    int32_t node_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode node;
    node.kind = TapeOpKind::Reshape;
    node.lhs = input;
    node.shape = toMLXShape(shape, rank);
    tape->nodes.push_back(std::move(node));
    return node_id;
}

int32_t mlx_tape_add_transpose(MLXTapeHandle handle, int32_t input,
    const int32_t* axes, int rank, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_transpose: null tape handle");
        return -1;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    if (input < 0 || static_cast<size_t>(input) >= tape->nodes.size()) {
        setError(error, 2, "mlx_tape_add_transpose: invalid input node ID");
        return -1;
    }
    int32_t node_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode node;
    node.kind = TapeOpKind::Transpose;
    node.lhs = input;
    if (rank > 0 && axes != nullptr) {
        node.axes.assign(axes, axes + rank);
    }
    tape->nodes.push_back(std::move(node));
    return node_id;
}

int32_t mlx_tape_add_broadcast_in_dim(MLXTapeHandle handle, int32_t input,
    const int32_t* out_shape, int out_rank,
    const int32_t* broadcast_axes, int num_broadcast_axes, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_broadcast_in_dim: null tape handle");
        return -1;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    if (input < 0 || static_cast<size_t>(input) >= tape->nodes.size()) {
        setError(error, 2, "mlx_tape_add_broadcast_in_dim: invalid input node ID");
        return -1;
    }
    int32_t node_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode node;
    node.kind = TapeOpKind::BroadcastInDim;
    node.lhs = input;
    node.shape = toMLXShape(out_shape, out_rank);
    if (num_broadcast_axes > 0 && broadcast_axes != nullptr) {
        node.axes.assign(broadcast_axes, broadcast_axes + num_broadcast_axes);
    }
    tape->nodes.push_back(std::move(node));
    return node_id;
}

int32_t mlx_tape_add_slice(MLXTapeHandle handle, int32_t input,
    const int32_t* starts, const int32_t* limits, const int32_t* strides, int rank, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_slice: null tape handle");
        return -1;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    if (input < 0 || static_cast<size_t>(input) >= tape->nodes.size()) {
        setError(error, 2, "mlx_tape_add_slice: invalid input node ID");
        return -1;
    }
    int32_t node_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode node;
    node.kind = TapeOpKind::Slice;
    node.lhs = input;
    node.slice_start = toMLXShape(starts, rank);
    node.slice_stop = toMLXShape(limits, rank);
    node.slice_strides = toMLXShape(strides, rank);
    tape->nodes.push_back(std::move(node));
    return node_id;
}

int32_t mlx_tape_add_concatenate(MLXTapeHandle handle,
    const int32_t* inputs, int num_inputs, int axis, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_concatenate: null tape handle");
        return -1;
    }
    if (num_inputs <= 0 || inputs == nullptr) {
        setError(error, 2, "mlx_tape_add_concatenate: at least one input is required");
        return -1;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    for (int i = 0; i < num_inputs; ++i) {
        if (inputs[i] < 0 || static_cast<size_t>(inputs[i]) >= tape->nodes.size()) {
            setError(error, 2, "mlx_tape_add_concatenate: invalid input node ID");
            return -1;
        }
    }
    int32_t node_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode node;
    node.kind = TapeOpKind::Concatenate;
    node.index = axis;
    node.inputs.assign(inputs, inputs + num_inputs);
    tape->nodes.push_back(std::move(node));
    return node_id;
}

int32_t mlx_tape_add_reverse(MLXTapeHandle handle, int32_t input,
    const int32_t* axes, int num_axes, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_reverse: null tape handle");
        return -1;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    if (input < 0 || static_cast<size_t>(input) >= tape->nodes.size()) {
        setError(error, 2, "mlx_tape_add_reverse: invalid input node ID");
        return -1;
    }
    int32_t node_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode node;
    node.kind = TapeOpKind::Reverse;
    node.lhs = input;
    if (num_axes > 0 && axes != nullptr) {
        node.axes.assign(axes, axes + num_axes);
    }
    tape->nodes.push_back(std::move(node));
    return node_id;
}

int32_t mlx_tape_add_pad(MLXTapeHandle handle, int32_t input, int32_t fill_value,
    const int32_t* axes, const int32_t* low_pad, const int32_t* high_pad, int num_axes, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_pad: null tape handle");
        return -1;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    if (input < 0 || static_cast<size_t>(input) >= tape->nodes.size() ||
        fill_value < 0 || static_cast<size_t>(fill_value) >= tape->nodes.size()) {
        setError(error, 2, "mlx_tape_add_pad: invalid operand node ID");
        return -1;
    }
    int32_t node_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode node;
    node.kind = TapeOpKind::Pad;
    node.lhs = input;
    node.rhs = fill_value;
    if (num_axes > 0 && axes != nullptr) {
        node.axes.assign(axes, axes + num_axes);
    }
    node.slice_start = toMLXShape(low_pad, num_axes);
    node.slice_stop = toMLXShape(high_pad, num_axes);
    tape->nodes.push_back(std::move(node));
    return node_id;
}

int32_t mlx_tape_add_where(MLXTapeHandle handle, int32_t cond, int32_t on_true, int32_t on_false, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_where: null tape handle");
        return -1;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    if (cond < 0 || static_cast<size_t>(cond) >= tape->nodes.size() ||
        on_true < 0 || static_cast<size_t>(on_true) >= tape->nodes.size() ||
        on_false < 0 || static_cast<size_t>(on_false) >= tape->nodes.size()) {
        setError(error, 2, "mlx_tape_add_where: invalid operand node ID");
        return -1;
    }
    int32_t node_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode node;
    node.kind = TapeOpKind::Where;
    node.lhs = cond;
    node.rhs = on_true;
    node.third = on_false;
    tape->nodes.push_back(std::move(node));
    return node_id;
}

int32_t mlx_tape_add_clamp(MLXTapeHandle handle, int32_t min_val, int32_t input, int32_t max_val, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_clamp: null tape handle");
        return -1;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    if (min_val < 0 || static_cast<size_t>(min_val) >= tape->nodes.size() ||
        input < 0 || static_cast<size_t>(input) >= tape->nodes.size() ||
        max_val < 0 || static_cast<size_t>(max_val) >= tape->nodes.size()) {
        setError(error, 2, "mlx_tape_add_clamp: invalid operand node ID");
        return -1;
    }
    int32_t node_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode node;
    node.kind = TapeOpKind::Clamp;
    node.lhs = input;
    node.rhs = min_val;
    node.third = max_val;
    tape->nodes.push_back(std::move(node));
    return node_id;
}

int32_t mlx_tape_add_reduce(MLXTapeHandle handle, int op, int32_t input,
    const int32_t* axes, int num_axes, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_reduce: null tape handle");
        return -1;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    if (input < 0 || static_cast<size_t>(input) >= tape->nodes.size()) {
        setError(error, 2, "mlx_tape_add_reduce: invalid input node ID");
        return -1;
    }
    int32_t node_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode node;
    node.kind = TapeOpKind::Reduce;
    node.index = op;
    node.lhs = input;
    if (num_axes > 0 && axes != nullptr) {
        node.axes.assign(axes, axes + num_axes);
    }
    tape->nodes.push_back(std::move(node));
    return node_id;
}

int32_t mlx_tape_add_arg_min_max(MLXTapeHandle handle, int32_t input,
    int axis, bool is_min, int output_dtype, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_arg_min_max: null tape handle");
        return -1;
    }
    bool ok = false;
    mlx::core::Dtype dt = toMLXDType(output_dtype, &ok);
    if (!ok) {
        setError(error, 3, "mlx_tape_add_arg_min_max: unsupported output dtype");
        return -1;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    if (input < 0 || static_cast<size_t>(input) >= tape->nodes.size()) {
        setError(error, 2, "mlx_tape_add_arg_min_max: invalid input node ID");
        return -1;
    }
    int32_t node_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode node;
    node.kind = TapeOpKind::ArgMinMax;
    node.index = axis;
    node.lhs = input;
    node.flag1 = is_min;
    node.dtype = dt;
    tape->nodes.push_back(std::move(node));
    return node_id;
}

int32_t mlx_tape_add_cumulative(MLXTapeHandle handle, int op, int32_t input,
    int axis, bool reverse, bool exclusive, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_cumulative: null tape handle");
        return -1;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    if (input < 0 || static_cast<size_t>(input) >= tape->nodes.size()) {
        setError(error, 2, "mlx_tape_add_cumulative: invalid input node ID");
        return -1;
    }
    int32_t node_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode node;
    node.kind = TapeOpKind::Cumulative;
    node.index = axis;
    node.lhs = input;
    node.flag1 = reverse;
    node.flag2 = exclusive;
    node.op_code = op;
    tape->nodes.push_back(std::move(node));
    return node_id;
}

int32_t mlx_tape_add_matmul(MLXTapeHandle handle, int32_t lhs, int32_t rhs, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_matmul: null tape handle");
        return -1;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    if (lhs < 0 || static_cast<size_t>(lhs) >= tape->nodes.size() ||
        rhs < 0 || static_cast<size_t>(rhs) >= tape->nodes.size()) {
        setError(error, 2, "mlx_tape_add_matmul: invalid operand node ID");
        return -1;
    }
    int32_t node_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode node;
    node.kind = TapeOpKind::Matmul;
    node.lhs = lhs;
    node.rhs = rhs;
    tape->nodes.push_back(std::move(node));
    return node_id;
}

int32_t mlx_tape_add_sub_parameter(MLXTapeHandle handle, int dtype,
    const int32_t* shape, int rank, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_sub_parameter: null tape handle");
        return -1;
    }
    bool ok = false;
    mlx::core::Dtype dt = toMLXDType(dtype, &ok);
    if (!ok) {
        setError(error, 3, "mlx_tape_add_sub_parameter: unsupported dtype");
        return -1;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    int32_t node_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode node;
    node.kind = TapeOpKind::SubParameter;
    node.dtype = dt;
    node.shape = toMLXShape(shape, rank);
    tape->nodes.push_back(std::move(node));
    return node_id;
}

bool mlx_tape_inline(MLXTapeHandle handle,
    const int32_t* param_ids, const int32_t* arg_ids, int num_params,
    const int32_t* output_ids, int num_outputs, int32_t* out_cloned_ids, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_inline: null tape handle");
        return false;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    std::unordered_map<int32_t, int32_t> remap;
    for (int i = 0; i < num_params; ++i) {
        if (param_ids[i] < 0 || static_cast<size_t>(param_ids[i]) >= tape->nodes.size() ||
            arg_ids[i] < 0 || static_cast<size_t>(arg_ids[i]) >= tape->nodes.size()) {
            setError(error, 2, "mlx_tape_inline: invalid parameter or argument node ID");
            return false;
        }
        remap[param_ids[i]] = arg_ids[i];
    }

    auto cloneNode = [&](auto& self, int32_t id) -> int32_t {
        if (id < 0) {
            return -1;
        }
        auto it = remap.find(id);
        if (it != remap.end()) {
            return it->second;
        }
        TapeNode orig = tape->nodes.at(id);
        bool changed = (orig.kind == TapeOpKind::RNGBits || orig.kind == TapeOpKind::RNGNextState);
        if (orig.lhs >= 0) {
            int32_t nl = self(self, orig.lhs);
            changed = changed || (nl != orig.lhs);
            orig.lhs = nl;
        }
        if (orig.rhs >= 0) {
            int32_t nr = self(self, orig.rhs);
            changed = changed || (nr != orig.rhs);
            orig.rhs = nr;
        }
        if (orig.third >= 0) {
            int32_t nt = self(self, orig.third);
            changed = changed || (nt != orig.third);
            orig.third = nt;
        }
        for (size_t i = 0; i < orig.inputs.size(); ++i) {
            if (orig.inputs[i] >= 0) {
                int32_t ni = self(self, orig.inputs[i]);
                changed = changed || (ni != orig.inputs[i]);
                orig.inputs[i] = ni;
            }
        }
        if (orig.kind == TapeOpKind::BroadcastInDim || orig.kind == TapeOpKind::Reshape) {
            for (size_t i = 0; i < orig.shape.size(); ++i) {
                if (orig.shape[i] <= -2) {
                    int32_t enc = -orig.shape[i] - 2;
                    int32_t ref_node = enc / 16;
                    int ref_axis = enc % 16;
                    int32_t new_ref = self(self, ref_node);
                    if (new_ref != ref_node) {
                        changed = true;
                        orig.shape[i] = -(2 + new_ref * 16 + ref_axis);
                    }
                }
            }
        }
        if (!changed && orig.kind != TapeOpKind::SubParameter) {
            remap[id] = id;
            return id;
        }
        int32_t new_id = static_cast<int32_t>(tape->nodes.size());
        tape->nodes.push_back(std::move(orig));
        remap[id] = new_id;
        return new_id;
    };

    for (int i = 0; i < num_outputs; ++i) {
        if (output_ids[i] < 0 || static_cast<size_t>(output_ids[i]) >= tape->nodes.size()) {
            setError(error, 2, "mlx_tape_inline: invalid output node ID");
            return false;
        }
        out_cloned_ids[i] = cloneNode(cloneNode, output_ids[i]);
    }
    return true;
}

int32_t mlx_tape_add_gather(MLXTapeHandle handle, int32_t operand, int32_t start_indices,
    int index_vector_axis, const int32_t* start_index_map, int num_index_map,
    const int32_t* slice_sizes, int operand_rank, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_gather: null tape handle");
        return -1;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    if (operand < 0 || static_cast<size_t>(operand) >= tape->nodes.size() ||
        start_indices < 0 || static_cast<size_t>(start_indices) >= tape->nodes.size()) {
        setError(error, 2, "mlx_tape_add_gather: invalid operand node ID");
        return -1;
    }
    int32_t node_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode node;
    node.kind = TapeOpKind::Gather;
    node.lhs = operand;
    node.rhs = start_indices;
    node.index = index_vector_axis;
    if (num_index_map > 0 && start_index_map != nullptr) {
        node.axes.assign(start_index_map, start_index_map + num_index_map);
    }
    node.shape = toMLXShape(slice_sizes, operand_rank);
    tape->nodes.push_back(std::move(node));
    return node_id;
}

int32_t mlx_tape_add_scatter(MLXTapeHandle handle, int op,
    int32_t operand, int32_t indices, int32_t updates,
    int index_vector_axis, const int32_t* axes, int num_axes, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_scatter: null tape handle");
        return -1;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    if (operand < 0 || static_cast<size_t>(operand) >= tape->nodes.size() ||
        indices < 0 || static_cast<size_t>(indices) >= tape->nodes.size() ||
        updates < 0 || static_cast<size_t>(updates) >= tape->nodes.size()) {
        setError(error, 2, "mlx_tape_add_scatter: invalid operand node ID");
        return -1;
    }
    int32_t node_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode node;
    node.kind = TapeOpKind::Scatter;
    node.op_code = op;
    node.lhs = operand;
    node.rhs = indices;
    node.third = updates;
    node.index = index_vector_axis;
    if (num_axes > 0 && axes != nullptr) {
        node.axes.assign(axes, axes + num_axes);
    }
    tape->nodes.push_back(std::move(node));
    return node_id;
}

int32_t mlx_tape_add_dynamic_slice(MLXTapeHandle handle, int32_t operand, int32_t start_indices,
    const int32_t* slice_sizes, int rank, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_dynamic_slice: null tape handle");
        return -1;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    if (operand < 0 || static_cast<size_t>(operand) >= tape->nodes.size() ||
        start_indices < 0 || static_cast<size_t>(start_indices) >= tape->nodes.size()) {
        setError(error, 2, "mlx_tape_add_dynamic_slice: invalid operand node ID");
        return -1;
    }
    int32_t node_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode node;
    node.kind = TapeOpKind::DynamicSlice;
    node.lhs = operand;
    node.rhs = start_indices;
    node.shape = toMLXShape(slice_sizes, rank);
    node.axes.resize(rank > 0 ? rank : 0);
    for (int i = 0; i < rank; ++i) {
        node.axes[i] = i;
    }
    tape->nodes.push_back(std::move(node));
    return node_id;
}

int32_t mlx_tape_add_dynamic_update_slice(MLXTapeHandle handle, int32_t operand, int32_t update,
    int32_t start_indices, int rank, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_dynamic_update_slice: null tape handle");
        return -1;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    if (operand < 0 || static_cast<size_t>(operand) >= tape->nodes.size() ||
        update < 0 || static_cast<size_t>(update) >= tape->nodes.size() ||
        start_indices < 0 || static_cast<size_t>(start_indices) >= tape->nodes.size()) {
        setError(error, 2, "mlx_tape_add_dynamic_update_slice: invalid operand node ID");
        return -1;
    }
    int32_t node_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode node;
    node.kind = TapeOpKind::DynamicUpdateSlice;
    node.lhs = operand;
    node.rhs = update;
    node.third = start_indices;
    node.axes.resize(rank > 0 ? rank : 0);
    for (int i = 0; i < rank; ++i) {
        node.axes[i] = i;
    }
    tape->nodes.push_back(std::move(node));
    return node_id;
}

int32_t mlx_tape_add_conv_general(MLXTapeHandle handle, int32_t input, int32_t weight,
    const int32_t* strides, const int32_t* pad_lo, const int32_t* pad_hi,
    const int32_t* kernel_dilations, const int32_t* input_dilations,
    int num_spatial_dims, int groups, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_conv_general: null tape handle");
        return -1;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    if (input < 0 || static_cast<size_t>(input) >= tape->nodes.size() ||
        weight < 0 || static_cast<size_t>(weight) >= tape->nodes.size()) {
        setError(error, 2, "mlx_tape_add_conv_general: invalid operand node ID");
        return -1;
    }
    int32_t node_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode node;
    node.kind = TapeOpKind::ConvGeneral;
    node.lhs = input;
    node.rhs = weight;
    node.index = groups;
    if (num_spatial_dims > 0) {
        node.axes.assign(strides, strides + num_spatial_dims);
        node.extra_ints1.assign(pad_lo, pad_lo + num_spatial_dims);
        node.extra_ints2.assign(pad_hi, pad_hi + num_spatial_dims);
        node.extra_ints3.assign(kernel_dilations, kernel_dilations + num_spatial_dims);
        node.extra_ints4.assign(input_dilations, input_dilations + num_spatial_dims);
    }
    tape->nodes.push_back(std::move(node));
    return node_id;
}

int32_t mlx_tape_add_as_strided(MLXTapeHandle handle, int32_t input,
    const int32_t* shape, const int64_t* strides, int rank, int64_t offset, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_as_strided: null tape handle");
        return -1;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    if (input < 0 || static_cast<size_t>(input) >= tape->nodes.size()) {
        setError(error, 2, "mlx_tape_add_as_strided: invalid input node ID");
        return -1;
    }
    int32_t node_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode node;
    node.kind = TapeOpKind::AsStrided;
    node.lhs = input;
    node.offset = offset;
    node.shape = toMLXShape(shape, rank);
    if (rank > 0 && strides != nullptr) {
        node.strides = mlx::core::Strides(strides, strides + rank);
    }
    tape->nodes.push_back(std::move(node));
    return node_id;
}

int32_t mlx_tape_add_select_and_scatter(MLXTapeHandle handle, int32_t operand, int32_t source,
    bool is_max, const int32_t* window_dims, const int32_t* window_strides,
    const int32_t* pad_lo, const int32_t* pad_hi, int rank, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_select_and_scatter: null tape handle");
        return -1;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    if (operand < 0 || static_cast<size_t>(operand) >= tape->nodes.size() ||
        source < 0 || static_cast<size_t>(source) >= tape->nodes.size()) {
        setError(error, 2, "mlx_tape_add_select_and_scatter: invalid operand node ID");
        return -1;
    }
    int32_t node_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode node;
    node.kind = TapeOpKind::SelectAndScatter;
    node.lhs = operand;
    node.rhs = source;
    node.flag1 = is_max;
    node.slice_start = toMLXShape(window_dims, rank);
    node.slice_strides = toMLXShape(window_strides, rank);
    if (rank > 0) {
        node.extra_ints1.assign(pad_lo, pad_lo + rank);
        node.extra_ints2.assign(pad_hi, pad_hi + rank);
    }
    tape->nodes.push_back(std::move(node));
    return node_id;
}

bool mlx_tape_add_rng_bit_generator(MLXTapeHandle handle, int32_t state, int dtype,
    const int32_t* shape, int rank, int32_t* out_new_state, int32_t* out_values, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_rng_bit_generator: null tape handle");
        return false;
    }
    bool ok = false;
    mlx::core::Dtype dt = toMLXDType(dtype, &ok);
    if (!ok) {
        setError(error, 3, "mlx_tape_add_rng_bit_generator: unsupported dtype");
        return false;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    if (state < 0 || static_cast<size_t>(state) >= tape->nodes.size()) {
        setError(error, 2, "mlx_tape_add_rng_bit_generator: invalid state node ID");
        return false;
    }

    int32_t next_state_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode state_node;
    state_node.kind = TapeOpKind::RNGNextState;
    state_node.lhs = state;
    tape->nodes.push_back(std::move(state_node));

    int32_t values_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode val_node;
    val_node.kind = TapeOpKind::RNGBits;
    val_node.lhs = state;
    val_node.dtype = dt;
    val_node.shape = toMLXShape(shape, rank);
    tape->nodes.push_back(std::move(val_node));

    if (out_new_state) *out_new_state = next_state_id;
    if (out_values) *out_values = values_id;
    return true;
}

int32_t mlx_tape_add_sort(MLXTapeHandle handle, int32_t input, int axis, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_sort: null tape handle");
        return -1;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    if (input < 0 || static_cast<size_t>(input) >= tape->nodes.size()) {
        setError(error, 2, "mlx_tape_add_sort: invalid input node ID");
        return -1;
    }
    int32_t node_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode node;
    node.kind = TapeOpKind::Sort;
    node.lhs = input;
    node.index = axis;
    tape->nodes.push_back(std::move(node));
    return node_id;
}

int32_t mlx_tape_add_argsort(MLXTapeHandle handle, int32_t input, int axis, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_argsort: null tape handle");
        return -1;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    if (input < 0 || static_cast<size_t>(input) >= tape->nodes.size()) {
        setError(error, 2, "mlx_tape_add_argsort: invalid input node ID");
        return -1;
    }
    int32_t node_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode node;
    node.kind = TapeOpKind::Argsort;
    node.lhs = input;
    node.index = axis;
    tape->nodes.push_back(std::move(node));
    return node_id;
}

int32_t mlx_tape_add_take_along_axis(MLXTapeHandle handle, int32_t input, int32_t indices, int axis, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_take_along_axis: null tape handle");
        return -1;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    if (input < 0 || static_cast<size_t>(input) >= tape->nodes.size() ||
        indices < 0 || static_cast<size_t>(indices) >= tape->nodes.size()) {
        setError(error, 2, "mlx_tape_add_take_along_axis: invalid operand node ID");
        return -1;
    }
    int32_t node_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode node;
    node.kind = TapeOpKind::TakeAlongAxis;
    node.lhs = input;
    node.rhs = indices;
    node.index = axis;
    tape->nodes.push_back(std::move(node));
    return node_id;
}

int32_t mlx_tape_add_softmax(MLXTapeHandle handle, int32_t input, int axis, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_softmax: null tape handle");
        return -1;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    if (input < 0 || static_cast<size_t>(input) >= tape->nodes.size()) {
        setError(error, 2, "mlx_tape_add_softmax: invalid input node ID");
        return -1;
    }
    int32_t node_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode node;
    node.kind = TapeOpKind::Softmax;
    node.lhs = input;
    node.index = axis;
    tape->nodes.push_back(std::move(node));
    return node_id;
}

int32_t mlx_tape_add_layer_norm(MLXTapeHandle handle, int32_t input, int32_t gamma, int32_t beta,
    float eps, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_add_layer_norm: null tape handle");
        return -1;
    }
    auto* tape = static_cast<MLXTape*>(handle);
    if (input < 0 || static_cast<size_t>(input) >= tape->nodes.size() ||
        (gamma >= 0 && static_cast<size_t>(gamma) >= tape->nodes.size()) ||
        (beta >= 0 && static_cast<size_t>(beta) >= tape->nodes.size())) {
        setError(error, 2, "mlx_tape_add_layer_norm: invalid operand node ID");
        return -1;
    }
    int32_t node_id = static_cast<int32_t>(tape->nodes.size());
    TapeNode node;
    node.kind = TapeOpKind::LayerNorm;
    node.lhs = input;
    node.rhs = gamma;
    node.third = beta;
    node.scalar_f32 = eps;
    tape->nodes.push_back(std::move(node));
    return node_id;
}

MLXExecHandle mlx_tape_compile(MLXTapeHandle handle,
    const int32_t* output_ids, int num_outputs,
    bool compile_graph, bool eager_contiguous, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_tape_compile: null tape handle");
        return nullptr;
    }
    try {
        auto* orig_tape = static_cast<MLXTape*>(handle);
        auto shared_tape = std::make_shared<MLXTape>(*orig_tape);
        std::vector<int32_t> outs(output_ids, output_ids + num_outputs);
        shared_tape->computeReachable(outs);

        auto* exec = new MLXExec();
        exec->tape = shared_tape;
        exec->output_ids = outs;
        exec->compile_graph = compile_graph;
        exec->eager_contiguous = eager_contiguous;

        auto runner = [shared_tape, outs, eager_contiguous](
                          const std::vector<mlx::core::array>& inputs) {
            return shared_tape->replay(inputs, outs, eager_contiguous);
        };

        if (compile_graph) {
            exec->compiled_fn = mlx::core::compile(runner, /*shapeless=*/false);
        } else {
            exec->compiled_fn = runner;
        }
        return static_cast<MLXExecHandle>(exec);
    } catch (const std::exception& e) {
        setError(error, 6, std::string("mlx_tape_compile: ") + e.what());
        return nullptr;
    }
}

void mlx_exec_destroy(MLXExecHandle handle) {
    if (handle) {
        delete static_cast<MLXExec*>(handle);
    }
}

bool mlx_exec_execute(MLXExecHandle handle,
    MLXArrayHandle* inputs, const bool* donate, int num_inputs,
    MLXArrayHandle* outputs, int num_outputs, MLXError* error) {
    clearError(error);
    if (!handle) {
        setError(error, 1, "mlx_exec_execute: null exec handle");
        return false;
    }
    try {
        auto* exec = static_cast<MLXExec*>(handle);

        std::vector<mlx::core::array> in_arrays;
        in_arrays.reserve(num_inputs);
        for (int i = 0; i < num_inputs; ++i) {
            in_arrays.push_back(*static_cast<mlx::core::array*>(inputs[i]));
        }

        // Step 1: Build lazy output array DAG (either via compiled function or tape replay).
        std::vector<mlx::core::array> out_arrays = exec->compiled_fn(in_arrays);

        // Step 2: Release external references for donated inputs BEFORE calling mlx::core::eval()
        // so their reference counts drop to 1 inside the graph, enabling MLX's in-place buffer reuse.
        in_arrays.clear();
        if (donate != nullptr) {
            for (int i = 0; i < num_inputs; ++i) {
                if (donate[i] && inputs[i] != nullptr) {
                    delete static_cast<mlx::core::array*>(inputs[i]);
                    inputs[i] = nullptr;
                }
            }
        }

        // Step 3: Evaluate all output arrays on the Metal stream.
        if (!out_arrays.empty()) {
            mlx::core::eval(out_arrays);
        }

        // Step 4: Return heap-allocated mlx::core::array handles for Go Buffers.
        for (int i = 0; i < num_outputs; ++i) {
            outputs[i] = static_cast<MLXArrayHandle>(
                new mlx::core::array(std::move(out_arrays[i])));
        }
        return true;
    } catch (const std::exception& e) {
        setError(error, 7, std::string("mlx_exec_execute: ") + e.what());
        return false;
    }
}

} // extern "C"

#else // !GOMLX_HAS_MLX_HEADERS

extern "C" {

bool mlx_bridge_init(MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return false;
}

const char* mlx_device_name(MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return nullptr;
}

const char* mlx_version_string(MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return nullptr;
}

MLXArrayHandle mlx_array_create(int64_t, int, const int32_t*, int, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return nullptr;
}

MLXArrayHandle mlx_array_from_data(const void*, int64_t, int, const int32_t*, int, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return nullptr;
}

void* mlx_array_data_ptr(MLXArrayHandle) { return nullptr; }
int64_t mlx_array_nbytes(MLXArrayHandle) { return 0; }
int mlx_array_dtype(MLXArrayHandle) { return 0; }
int mlx_array_ndim(MLXArrayHandle) { return 0; }
void mlx_array_shape(MLXArrayHandle, int32_t*) {}
bool mlx_array_is_row_contiguous(MLXArrayHandle) { return false; }

bool mlx_array_ensure_contiguous(MLXArrayHandle, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return false;
}

void mlx_array_destroy(MLXArrayHandle) {}

MLXTapeHandle mlx_tape_create(MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return nullptr;
}

void mlx_tape_destroy(MLXTapeHandle) {}

int32_t mlx_tape_add_parameter(MLXTapeHandle, int, const int32_t*, int, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return -1;
}

int32_t mlx_tape_add_constant(MLXTapeHandle, const void*, int64_t, int, const int32_t*, int, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return -1;
}

int32_t mlx_tape_add_unary(MLXTapeHandle, int, int32_t, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return -1;
}

int32_t mlx_tape_add_binary(MLXTapeHandle, int, int32_t, int32_t, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return -1;
}

int32_t mlx_tape_add_convert_dtype(MLXTapeHandle, int32_t, int, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return -1;
}

int32_t mlx_tape_add_bitcast(MLXTapeHandle, int32_t, int, const int32_t*, int, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return -1;
}

int32_t mlx_tape_add_iota(MLXTapeHandle, int, const int32_t*, int, int, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return -1;
}

int32_t mlx_tape_add_reshape(MLXTapeHandle, int32_t, const int32_t*, int, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return -1;
}

int32_t mlx_tape_add_transpose(MLXTapeHandle, int32_t, const int32_t*, int, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return -1;
}

int32_t mlx_tape_add_broadcast_in_dim(MLXTapeHandle, int32_t, const int32_t*, int, const int32_t*, int, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return -1;
}

int32_t mlx_tape_add_slice(MLXTapeHandle, int32_t, const int32_t*, const int32_t*, const int32_t*, int, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return -1;
}

int32_t mlx_tape_add_concatenate(MLXTapeHandle, const int32_t*, int, int, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return -1;
}

int32_t mlx_tape_add_reverse(MLXTapeHandle, int32_t, const int32_t*, int, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return -1;
}

int32_t mlx_tape_add_pad(MLXTapeHandle, int32_t, int32_t, const int32_t*, const int32_t*, const int32_t*, int, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return -1;
}

int32_t mlx_tape_add_where(MLXTapeHandle, int32_t, int32_t, int32_t, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return -1;
}

int32_t mlx_tape_add_clamp(MLXTapeHandle, int32_t, int32_t, int32_t, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return -1;
}

int32_t mlx_tape_add_reduce(MLXTapeHandle, int, int32_t, const int32_t*, int, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return -1;
}

int32_t mlx_tape_add_arg_min_max(MLXTapeHandle, int32_t, int, bool, int, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return -1;
}

int32_t mlx_tape_add_cumulative(MLXTapeHandle, int, int32_t, int, bool, bool, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return -1;
}

int32_t mlx_tape_add_matmul(MLXTapeHandle, int32_t, int32_t, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return -1;
}

int32_t mlx_tape_add_sub_parameter(MLXTapeHandle, int, const int32_t*, int, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return -1;
}

bool mlx_tape_inline(MLXTapeHandle, const int32_t*, const int32_t*, int, const int32_t*, int, int32_t*, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return false;
}

int32_t mlx_tape_add_gather(MLXTapeHandle, int32_t, int32_t, int, const int32_t*, int, const int32_t*, int, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return -1;
}

int32_t mlx_tape_add_scatter(MLXTapeHandle, int, int32_t, int32_t, int32_t, int, const int32_t*, int, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return -1;
}

int32_t mlx_tape_add_dynamic_slice(MLXTapeHandle, int32_t, int32_t, const int32_t*, int, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return -1;
}

int32_t mlx_tape_add_dynamic_update_slice(MLXTapeHandle, int32_t, int32_t, int32_t, int, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return -1;
}

int32_t mlx_tape_add_conv_general(MLXTapeHandle, int32_t, int32_t, const int32_t*, const int32_t*, const int32_t*, const int32_t*, const int32_t*, int, int, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return -1;
}

int32_t mlx_tape_add_as_strided(MLXTapeHandle, int32_t, const int32_t*, const int64_t*, int, int64_t, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return -1;
}

int32_t mlx_tape_add_select_and_scatter(MLXTapeHandle, int32_t, int32_t, bool, const int32_t*, const int32_t*, const int32_t*, const int32_t*, int, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return -1;
}

bool mlx_tape_add_rng_bit_generator(MLXTapeHandle, int32_t, int, const int32_t*, int, int32_t*, int32_t*, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return false;
}

int32_t mlx_tape_add_sort(MLXTapeHandle, int32_t, int, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return -1;
}

int32_t mlx_tape_add_argsort(MLXTapeHandle, int32_t, int, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return -1;
}

int32_t mlx_tape_add_take_along_axis(MLXTapeHandle, int32_t, int32_t, int, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return -1;
}

int32_t mlx_tape_add_softmax(MLXTapeHandle, int32_t, int, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return -1;
}

int32_t mlx_tape_add_layer_norm(MLXTapeHandle, int32_t, int32_t, int32_t, float, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return -1;
}

MLXExecHandle mlx_tape_compile(MLXTapeHandle, const int32_t*, int, bool, bool, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return nullptr;
}

void mlx_exec_destroy(MLXExecHandle) {}

bool mlx_exec_execute(MLXExecHandle, MLXArrayHandle*, const bool*, int, MLXArrayHandle*, int, MLXError* error) {
    setError(error, 1, kInstallHelpMsg);
    return false;
}

} // extern "C"

#endif // GOMLX_HAS_MLX_HEADERS
