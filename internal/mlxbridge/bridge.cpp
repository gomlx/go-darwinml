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
};

struct TapeNode {
    TapeOpKind kind;
    int index; // index into params or constants
    mlx::core::Dtype dtype{mlx::core::float32};
    mlx::core::Shape shape;
};

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
            }
        }

        std::vector<mlx::core::array> outputs;
        outputs.reserve(output_ids.size());
        for (int32_t id : output_ids) {
            mlx::core::array out = values.at(id);
            if (eager_contiguous) {
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
    tape->nodes.push_back(TapeNode{
        TapeOpKind::Parameter,
        param_idx,
        dt,
        std::move(s),
    });
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
        tape->nodes.push_back(TapeNode{
            TapeOpKind::Constant,
            const_idx,
            dt,
            std::move(s),
        });
        return node_id;
    } catch (const std::exception& e) {
        setError(error, 4, std::string("mlx_tape_add_constant: ") + e.what());
        return -1;
    }
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
