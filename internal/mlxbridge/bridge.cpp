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
#include <memory>
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
};

struct TapeNode {
    TapeOpKind kind;
    int index = 0; // index into params or constants, or op code / axis
    int32_t lhs = -1;
    int32_t rhs = -1;
    int32_t third = -1;
    mlx::core::Dtype dtype{mlx::core::float32};
    mlx::core::Shape shape;
    std::vector<int> axes;
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

struct MLXTape {
    int num_params = 0;
    std::vector<mlx::core::Dtype> param_dtypes;
    std::vector<mlx::core::Shape> param_shapes;
    std::vector<mlx::core::array> constants;
    std::vector<TapeNode> nodes;

    std::vector<mlx::core::array> replay(
        const std::vector<mlx::core::array>& inputs,
        const std::vector<int32_t>& output_ids,
        bool eager_contiguous) const {
        std::vector<mlx::core::array> values;
        values.reserve(nodes.size());

        for (const auto& node : nodes) {
            switch (node.kind) {
                case TapeOpKind::Parameter:
                    values.push_back(inputs.at(node.index));
                    break;
                case TapeOpKind::Constant:
                    values.push_back(constants.at(node.index));
                    break;
                case TapeOpKind::Unary:
                    values.push_back(evalUnaryOp(node.index, values.at(node.lhs)));
                    break;
                case TapeOpKind::Binary:
                    values.push_back(evalBinaryOp(node.index, values.at(node.lhs), values.at(node.rhs)));
                    break;
                case TapeOpKind::ConvertDType:
                    values.push_back(mlx::core::astype(values.at(node.lhs), node.dtype));
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
                    values.push_back(std::move(v));
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
                    values.push_back(std::move(r));
                    break;
                }
                case TapeOpKind::Reshape:
                    values.push_back(mlx::core::reshape(values.at(node.lhs), node.shape));
                    break;
                case TapeOpKind::Transpose:
                    if (node.axes.empty()) {
                        values.push_back(values.at(node.lhs));
                    } else {
                        values.push_back(mlx::core::transpose(values.at(node.lhs), node.axes));
                    }
                    break;
                case TapeOpKind::BroadcastInDim: {
                    const mlx::core::array& a = values.at(node.lhs);
                    if (node.shape.empty()) {
                        values.push_back(a);
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
                    values.push_back(std::move(r));
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
                    values.push_back(mlx::core::slice(a, node.slice_start, std::move(stop), node.slice_strides));
                    break;
                }
                case TapeOpKind::Concatenate: {
                    std::vector<mlx::core::array> arrs;
                    arrs.reserve(node.inputs.size());
                    for (int32_t id : node.inputs) {
                        arrs.push_back(values.at(id));
                    }
                    values.push_back(mlx::core::concatenate(std::move(arrs), node.index));
                    break;
                }
                case TapeOpKind::Reverse:
                    if (node.axes.empty()) {
                        values.push_back(values.at(node.lhs));
                    } else {
                        values.push_back(mlx::core::flip(values.at(node.lhs), node.axes));
                    }
                    break;
                case TapeOpKind::Pad:
                    values.push_back(mlx::core::pad(
                        values.at(node.lhs),
                        node.axes,
                        node.slice_start,
                        node.slice_stop,
                        values.at(node.rhs)));
                    break;
                case TapeOpKind::Where:
                    values.push_back(mlx::core::where(
                        values.at(node.lhs),
                        values.at(node.rhs),
                        values.at(node.third)));
                    break;
                case TapeOpKind::Clamp:
                    values.push_back(mlx::core::clip(
                        values.at(node.lhs),
                        values.at(node.rhs),
                        values.at(node.third)));
                    break;
                case TapeOpKind::Reduce:
                    values.push_back(evalReduceOp(node.index, values.at(node.lhs), node.axes));
                    break;
                case TapeOpKind::ArgMinMax: {
                    const mlx::core::array& a = values.at(node.lhs);
                    bool is_min = (node.rhs != 0);
                    mlx::core::array res = is_min
                        ? mlx::core::argmin(a, node.index, /*keepdims=*/false)
                        : mlx::core::argmax(a, node.index, /*keepdims=*/false);
                    if (res.dtype() != node.dtype) {
                        res = mlx::core::astype(res, node.dtype);
                    }
                    values.push_back(std::move(res));
                    break;
                }
                case TapeOpKind::Cumulative: {
                    const mlx::core::array& a = values.at(node.lhs);
                    int axis = node.index;
                    bool reverse = (node.rhs & 1) != 0;
                    bool inclusive = (node.rhs & 2) == 0;
                    mlx::core::array res = a;
                    if (node.third == MLX_CUMULATIVE_SUM) {
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
                    values.push_back(std::move(res));
                    break;
                }
                case TapeOpKind::Matmul:
                    values.push_back(evalMatmul(values.at(node.lhs), values.at(node.rhs)));
                    break;
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
    node.rhs = is_min ? 1 : 0;
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
    node.rhs = (reverse ? 1 : 0) | (exclusive ? 2 : 0);
    node.third = op;
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
