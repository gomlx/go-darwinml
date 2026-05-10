#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include "gomlx_erf.h"
        
using namespace metal;

// Shared tile size with legacy dot_general; kept for bias+activation fusion kernels.
constant uint FDM_TILE = 16;

inline float fused_dense_apply_activation(float v, uint act, uint gelu_exact) {
    switch (act) {
        case 0: return v;
        case 1: {
            if (gelu_exact != 0) {
                return v * 0.5f * (1.0f + gomlx_erf(v * M_SQRTl_2_F));
            }
            float c = 0.7978845608f;
            return v * 0.5f * (1.0f + tanh(c * (v + 0.044715f * v * v * v)));
        }
        case 2: return max(0.0f, v);
        case 3: {
            float s = 1.0f / (1.0f + exp(-v));
            return v * s;
        }
        case 4: {
            float t = clamp(v * (1.0f / 6.0f) + 0.5f, 0.0f, 1.0f);
            return v * t;
        }
        case 5: return tanh(v);
        default: return v;
    }
}

kernel void fused_dense_f32(
    device const float* A     [[buffer(0)]],
    device const float* B     [[buffer(1)]],
    device const float* bias  [[buffer(2)]],
    device float* C           [[buffer(3)]],
    constant uint& m          [[buffer(4)]],
    constant uint& k          [[buffer(5)]],
    constant uint& n          [[buffer(6)]],
    constant uint& has_bias   [[buffer(7)]],
    constant uint& activation [[buffer(8)]],
    constant uint& gelu_exact [[buffer(9)]],
    uint3 gid  [[thread_position_in_grid]],
    uint3 tid  [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
    uint row = tgid.y * FDM_TILE + tid.y;
    uint col = tgid.x * FDM_TILE + tid.x;

    threadgroup float tileA[FDM_TILE][FDM_TILE];
    threadgroup float tileB[FDM_TILE][FDM_TILE];

    float acc = 0.0f;
    for (uint t = 0; t < (k + FDM_TILE - 1) / FDM_TILE; t++) {
        uint ak = t * FDM_TILE + tid.x;
        uint bk = t * FDM_TILE + tid.y;
        tileA[tid.y][tid.x] = (row < m && ak < k) ? A[row * k + ak] : 0.0f;
        tileB[tid.y][tid.x] = (bk < k && col < n) ? B[bk * n + col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < FDM_TILE; i++) {
            acc += tileA[tid.y][i] * tileB[i][tid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < m && col < n) {
        float v = acc;
        if (has_bias != 0) {
            v += bias[col];
        }
        v = fused_dense_apply_activation(v, activation, gelu_exact);
        C[row * n + col] = v;
    }
}

kernel void fused_dense_f16(
    device const half* A     [[buffer(0)]],
    device const half* B     [[buffer(1)]],
    device const half* bias  [[buffer(2)]],
    device half* C           [[buffer(3)]],
    constant uint& m         [[buffer(4)]],
    constant uint& k         [[buffer(5)]],
    constant uint& n         [[buffer(6)]],
    constant uint& has_bias   [[buffer(7)]],
    constant uint& activation [[buffer(8)]],
    constant uint& gelu_exact [[buffer(9)]],
    uint3 gid  [[thread_position_in_grid]],
    uint3 tid  [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
    uint row = tgid.y * FDM_TILE + tid.y;
    uint col = tgid.x * FDM_TILE + tid.x;

    threadgroup half tileA[FDM_TILE][FDM_TILE];
    threadgroup half tileB[FDM_TILE][FDM_TILE];

    float acc = 0.0f;
    for (uint t = 0; t < (k + FDM_TILE - 1) / FDM_TILE; t++) {
        uint ak = t * FDM_TILE + tid.x;
        uint bk = t * FDM_TILE + tid.y;
        tileA[tid.y][tid.x] = (row < m && ak < k) ? A[row * k + ak] : half(0.0);
        tileB[tid.y][tid.x] = (bk < k && col < n) ? B[bk * n + col] : half(0.0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < FDM_TILE; i++) {
            acc += float(tileA[tid.y][i]) * float(tileB[i][tid.x]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < m && col < n) {
        float v = acc;
        if (has_bias != 0) {
            v += float(bias[col]);
        }
        v = fused_dense_apply_activation(v, activation, gelu_exact);
        C[row * n + col] = half(v);
    }
}

kernel void fused_qkv_projection_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device const float* bq [[buffer(2)]],
    device const float* bk [[buffer(3)]],
    device const float* bv [[buffer(4)]],
    device float* Q [[buffer(5)]],
    device float* K [[buffer(6)]],
    device float* V [[buffer(7)]],
    constant uint& m [[buffer(8)]],
    constant uint& kk [[buffer(9)]],
    constant uint& qdim [[buffer(10)]],
    constant uint& kvm [[buffer(11)]],
    constant uint& hasq [[buffer(12)]],
    constant uint& hask [[buffer(13)]],
    constant uint& hasv [[buffer(14)]],
    uint3 gid  [[thread_position_in_grid]],
    uint3 tid  [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
    uint ntot = qdim + 2 * kvm;
    uint row = tgid.y * FDM_TILE + tid.y;
    uint col = tgid.x * FDM_TILE + tid.x;

    threadgroup float tileA[FDM_TILE][FDM_TILE];
    threadgroup float tileB[FDM_TILE][FDM_TILE];

    float acc = 0.0f;
    for (uint t = 0; t < (kk + FDM_TILE - 1) / FDM_TILE; t++) {
        uint ak = t * FDM_TILE + tid.x;
        uint bk = t * FDM_TILE + tid.y;
        tileA[tid.y][tid.x] = (row < m && ak < kk) ? A[row * kk + ak] : 0.0f;
        tileB[tid.y][tid.x] = (bk < kk && col < ntot) ? B[bk * ntot + col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < FDM_TILE; i++) {
            acc += tileA[tid.y][i] * tileB[i][tid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < m && col < ntot) {
        float v = acc;
        if (col < uint(qdim)) {
            if (hasq != 0) {
                v += bq[col];
            }
            Q[row * uint(qdim) + col] = v;
        } else if (col < uint(qdim + kvm)) {
            uint j = col - uint(qdim);
            if (hask != 0) {
                v += bk[j];
            }
            K[row * uint(kvm) + j] = v;
        } else {
            uint j = col - uint(qdim + kvm);
            if (hasv != 0) {
                v += bv[j];
            }
            V[row * uint(kvm) + j] = v;
        }
    }
}

kernel void fused_qkv_projection_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device const half* bq [[buffer(2)]],
    device const half* bk [[buffer(3)]],
    device const half* bv [[buffer(4)]],
    device half* Q [[buffer(5)]],
    device half* K [[buffer(6)]],
    device half* V [[buffer(7)]],
    constant uint& m [[buffer(8)]],
    constant uint& kk [[buffer(9)]],
    constant uint& qdim [[buffer(10)]],
    constant uint& kvm [[buffer(11)]],
    constant uint& hasq [[buffer(12)]],
    constant uint& hask [[buffer(13)]],
    constant uint& hasv [[buffer(14)]],
    uint3 gid  [[thread_position_in_grid]],
    uint3 tid  [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
    uint ntot = qdim + 2 * kvm;
    uint row = tgid.y * FDM_TILE + tid.y;
    uint col = tgid.x * FDM_TILE + tid.x;

    threadgroup half tileA[FDM_TILE][FDM_TILE];
    threadgroup half tileB[FDM_TILE][FDM_TILE];

    float acc = 0.0f;
    for (uint t = 0; t (kk + FDM_TILE - 1) / FDM_TILE; t++) {
        uint ak = t * FDM_TILE + tid.x;
        uint bk = t * FDM_TILE + tid.y;
        tileA[tid.y][tid.x] = (row < m && ak < kk) ? A[row * kk + ak] : half(0.0);
        tileB[tid.y][tid.x] = (bk < kk && col < ntot) ? B[bk * ntot + col] : half(0.0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < FDM_TILE; i++) {
            acc += float(tileA[tid.y][i]) * float(tileB[i][tid.x]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < m && col < ntot) {
        float v = acc;
        if (col < uint(qdim)) {
            if (hasq != 0) {
                v += float(bq[col]);
            }
            Q[row * uint(qdim) + col] = half(v);
        } else if (col < uint(qdim + kvm)) {
            uint j = col - uint(qdim);
            if (hask != 0) {
                v += float(bk[j]);
            }
            K[row * uint(kvm) + j] = half(v);
        } else {
            uint j = col - uint(qdim + kvm);
            if (hasv != 0) {
                v += float(bv[j]);
            }
            V[row * uint(kvm) + j] = half(v);
        }
    }
}

constant uint SG_M = 8;
constant uint SG_N = 8;
constant uint SG_K = 8;

kernel void dot_general_sg_f32(
    device const float* A     [[buffer(0)]],
    device const float* B     [[buffer(1)]],
    device float* C           [[buffer(2)]],
    constant uint& batch      [[buffer(3)]],
    constant uint& M          [[buffer(4)]],
    constant uint& K          [[buffer(5)]],
    constant uint& N          [[buffer(6)]],
    uint3 tgp   [[threadgroup_position_in_grid]],
    uint3 tpig  [[thread_position_in_threadgroup]],
    uint      simd_lane [[thread_index_in_simdgroup]])
{
    simdgroup_matrix_storage<float, SG_M, SG_N> accum;
    simdgroup_matrix_storage<float, SG_M, SG_K> mat_a;
    simdgroup_matrix_storage<float, SG_K, SG_N> mat_b;

    simdgroup_multiply_accumulate<float>(accum, mat_a, mat_b); // placeholder replaced below
}
