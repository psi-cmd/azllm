#pragma once
// Execution-layer Hygon DUMMA (v_mmac_*) wrappers for BW1000 (gfx936) and sister Hygon DCU targets,
// matching DTK 26.10's /opt/dtk/include/du_mma.hpp register/lane layout without <hip/hip_fp16.h> vs <cuda_fp16.h>
// header collisions under GPUFusion nvcc (-dcccuda12).
//
// Also supports TQLLM_BW_EMULATE on 32-lane CUDA GPUs (simulating 64-lane wavefronts across pairs of warps via shared
// memory) so every BW wave64 DUMMA kernel can be executed and numerically verified before physical BW1000 deployment.

#include "bw_primitives.cuh"
#include <cstdint>
#include <cstring>

#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__) || defined(__gfx92a__)
#define TQLLM_HAS_HW_MMAC 1
#define TQLLM_DUMMA_INLINE __forceinline__
#else
#define TQLLM_HAS_HW_MMAC 0
#define TQLLM_DUMMA_INLINE __noinline__
#endif

namespace tqllm_dumma {

struct alignas(8) FragB16x4 {
    uint16_t x[4];
};

struct alignas(8) FragI8x8 {
    int8_t x[8];
};

struct alignas(16) AccF32x4 {
    float x[4];
};

struct alignas(16) AccI32x4 {
    int32_t x[4];
};

#if TQLLM_HAS_HW_MMAC
using hw_floatx4 = __attribute__((ext_vector_type(4))) float;
using hw_intx4   = __attribute__((ext_vector_type(4))) int;
using hw_shortx4 = __attribute__((ext_vector_type(4))) short;
using hw_halfx4  = __attribute__((ext_vector_type(4))) _Float16;
#endif

// Per-CTA shared scratch used ONLY when TQLLM_BW_EMULATE simulates 64-lane wavefronts on 32-lane CUDA hardware.
// Zero bytes on real Hygon BW hardware (gfx936).
template <int MAX_WAVES64 = 4>
struct SimScratch {
#if !TQLLM_HAS_HW_MMAC
    uint64_t a[MAX_WAVES64][64];
    uint64_t b[MAX_WAVES64][64];
    float shf[MAX_WAVES64][64];
#else
    uint32_t unused;
#endif
};

// Logical 64-lane coordinates inside a CTA whose blockDim.x is a multiple of 64.
__device__ __forceinline__ int wave64_id() { return int(threadIdx.x) >> 6; }
__device__ __forceinline__ int lane64_id() {
#if TQLLM_HAS_HW_MMAC
    return static_cast<int>(__builtin_amdgcn_mbcnt_hi(-1, __builtin_amdgcn_mbcnt_lo(-1, 0)));
#else
    return int(threadIdx.x) & 63;
#endif
}

__device__ __forceinline__ AccF32x4 zero_f32x4() { return {{0.f, 0.f, 0.f, 0.f}}; }
__device__ __forceinline__ AccI32x4 zero_i32x4() { return {{0, 0, 0, 0}}; }

// du_load_matrix_sync equivalents (matching /opt/dtk/include/du_mma.hpp lines 1133-1299):
// For 16x16x16 B16 (FP16 or BF16):
//   row = lane64 & 15, k_grp = lane64 >> 4 (0..3)
//   A row_major [16][ldm]: lane loads A[row, 4*k_grp .. 4*k_grp+3]
//   B col_major [ldm][16] (i.e. B^T [16][ldm] row-contiguous along K): lane loads B_col[col=row][4*k_grp .. 4*k_grp+3]
//   B row_major [16][ldm] (K x N): lane loads B[4*k_grp + 0..3, col=row]
__device__ __forceinline__ FragB16x4 load_a_rowmajor_b16(const void * p, int ldm) {
    const int lid = lane64_id(), row = lid & 15, col = (lid >> 4) << 2;
    const auto * src = static_cast<const uint16_t *>(p) + row * ldm + col;
    FragB16x4 f;
    *reinterpret_cast<uint2 *>(f.x) = *reinterpret_cast<const uint2 *>(src);
    return f;
}

__device__ __forceinline__ FragB16x4 load_b_colmajor_b16(const void * p, int ldm) {
    const int lid = lane64_id(), col = lid & 15, row = (lid >> 4) << 2;
    const auto * src = static_cast<const uint16_t *>(p) + col * ldm + row;
    FragB16x4 f;
    *reinterpret_cast<uint2 *>(f.x) = *reinterpret_cast<const uint2 *>(src);
    return f;
}

__device__ __forceinline__ FragB16x4 load_b_rowmajor_b16(const void * p, int ldm) {
    const int lid = lane64_id(), col = lid & 15, row = (lid >> 4) << 2;
    const auto * src = static_cast<const uint16_t *>(p) + col;
    FragB16x4 f;
    f.x[0] = src[(row + 0) * ldm];
    f.x[1] = src[(row + 1) * ldm];
    f.x[2] = src[(row + 2) * ldm];
    f.x[3] = src[(row + 3) * ldm];
    return f;
}

// For 16x16x32 INT8:
//   row = lane64 & 15, k_grp = lane64 >> 4 (0..3)
//   A row_major [16][ldm]: lane loads A[row, 8*k_grp .. 8*k_grp+7]
//   B col_major [ldm][16] (K^T [16][ldm]): lane loads B_col[col=row][8*k_grp .. 8*k_grp+7]
__device__ __forceinline__ FragI8x8 load_a_rowmajor_i8(const int8_t * p, int ldm) {
    const int lid = lane64_id(), row = lid & 15, col = (lid >> 4) << 3;
    FragI8x8 f;
    *reinterpret_cast<uint2 *>(f.x) = *reinterpret_cast<const uint2 *>(p + row * ldm + col);
    return f;
}

__device__ __forceinline__ FragI8x8 load_b_colmajor_i8(const int8_t * p, int ldm) {
    const int lid = lane64_id(), col = lid & 15, row = (lid >> 4) << 3;
    FragI8x8 f;
    *reinterpret_cast<uint2 *>(f.x) = *reinterpret_cast<const uint2 *>(p + col * ldm + row);
    return f;
}

// Wave64 barrier (zero-cost wavefront fence on hardware wave64; __syncthreads on 32-lane emulation).
__device__ __forceinline__ void wave64_barrier() {
#if TQLLM_HAS_HW_MMAC
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
    __builtin_amdgcn_wave_barrier();
#else
    __syncthreads();
#endif
}

// Wave64 XOR shuffle across all 64 lanes (hardware ds_bpermute_b32: zero LDS usage, zero workgroup barriers).
template <int MAX_WAVES64 = 4>
__device__ __forceinline__ float wave64_shfl_xor(float v, int mask, SimScratch<MAX_WAVES64> & sim) {
#if TQLLM_HAS_HW_MMAC
    (void) sim;
    const int src = lane64_id() ^ mask;
    return __int_as_float(__builtin_amdgcn_ds_bpermute(src << 2, __float_as_int(v)));
#else
    const int w = wave64_id(), l = lane64_id();
    __syncthreads();
    sim.shf[w][l] = v;
    __syncthreads();
    const float out = sim.shf[w][l ^ mask];
    __syncthreads();
    return out;
#endif
}

// Full 64-lane sum reduction (for streaming wave64 GEMV).
template <int MAX_WAVES64 = 4>
__device__ __forceinline__ float wave64_sum(float v, SimScratch<MAX_WAVES64> & sim) {
#if TQLLM_HAS_HW_MMAC
    (void) sim;
    const int lid = lane64_id();
#pragma unroll
    for (int mask = 32; mask >= 1; mask >>= 1)
        v += __int_as_float(__builtin_amdgcn_ds_bpermute((lid ^ mask) << 2, __float_as_int(v)));
    return v;
#else
#pragma unroll
    for (int mask = 32; mask >= 1; mask >>= 1) v += wave64_shfl_xor(v, mask, sim);
    return v;
#endif
}

// Reduction across the 4 lanes (r + {0, 16, 32, 48}) that own the 16 columns of DUMMA row `r = lane64 & 15`.
template <int MAX_WAVES64 = 4>
__device__ __forceinline__ float row_group4_max(float v, SimScratch<MAX_WAVES64> & sim) {
    v = fmaxf(v, wave64_shfl_xor(v, 16, sim));
    v = fmaxf(v, wave64_shfl_xor(v, 32, sim));
    return v;
}

template <int MAX_WAVES64 = 4>
__device__ __forceinline__ float row_group4_sum(float v, SimScratch<MAX_WAVES64> & sim) {
    v += wave64_shfl_xor(v, 16, sim);
    v += wave64_shfl_xor(v, 32, sim);
    return v;
}

// Hardware v_mmac_f32_16x16x16_f16 (or exact 64-lane DUFragment emulation).
// Output mapping: lane (k_grp = lid >> 4, row = lid & 15) holds D[row, k_grp + 4*i] for i = 0..3.
template <int MAX_WAVES64 = 4>
__device__ TQLLM_DUMMA_INLINE AccF32x4 mmac_f32_16x16x16_f16(const FragB16x4 & a, const FragB16x4 & b,
                                                          const AccF32x4 & c, SimScratch<MAX_WAVES64> & sim) {
#if TQLLM_HAS_HW_MMAC
    (void) sim;
    hw_halfx4 ax, bx;
    hw_floatx4 cx;
    std::memcpy(&ax, a.x, sizeof(ax));
    std::memcpy(&bx, b.x, sizeof(bx));
    std::memcpy(&cx, c.x, sizeof(cx));
    hw_floatx4 dx;
#if defined(__gfx938__)
    dx = __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(ax, bx, cx, false, false);
#elif defined(__gfx92a__)
    dx = __builtin_hcu_mmac_f32_16x16x16_f16(ax, bx, cx);
#else
    dx = __builtin_amdgcn_mmac_f32_16x16x16f16(ax, bx, cx);
#endif
    AccF32x4 out;
    std::memcpy(out.x, &dx, sizeof(out.x));
    return out;
#else
    const int w = wave64_id(), lid = lane64_id(), row = lid & 15, k_grp = lid >> 4;
    uint64_t au = 0, bu = 0;
    std::memcpy(&au, a.x, sizeof(au));
    std::memcpy(&bu, b.x, sizeof(bu));
    __syncthreads();
    sim.a[w][lid] = au;
    sim.b[w][lid] = bu;
    __syncthreads();
    AccF32x4 out = c;
#pragma unroll 1
    for (int i = 0; i < 4; ++i) {
        const int col = k_grp + 4 * i;
        float sum = out.x[i];
#pragma unroll 1
        for (int g = 0; g < 4; ++g) {
            FragB16x4 fa, fb;
            std::memcpy(fa.x, &sim.a[w][g * 16 + row], sizeof(fa.x));
            std::memcpy(fb.x, &sim.b[w][g * 16 + col], sizeof(fb.x));
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float va = __half2float(__ushort_as_half(fa.x[e]));
                const float vb = __half2float(__ushort_as_half(fb.x[e]));
                sum = fmaf(va, vb, sum);
            }
        }
        out.x[i] = sum;
    }
    __syncthreads();
    return out;
#endif
}

// Hardware v_mmac_f32_16x16x16_bf16 (or exact 64-lane DUFragment emulation).
template <int MAX_WAVES64 = 4>
__device__ TQLLM_DUMMA_INLINE AccF32x4 mmac_f32_16x16x16_bf16(const FragB16x4 & a, const FragB16x4 & b,
                                                           const AccF32x4 & c, SimScratch<MAX_WAVES64> & sim) {
#if TQLLM_HAS_HW_MMAC
    (void) sim;
    hw_shortx4 ax, bx;
    hw_floatx4 cx;
    std::memcpy(&ax, a.x, sizeof(ax));
    std::memcpy(&bx, b.x, sizeof(bx));
    std::memcpy(&cx, c.x, sizeof(cx));
    hw_floatx4 dx;
#if defined(__gfx938__)
    dx = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(ax, bx, cx, false, false);
#elif defined(__gfx92a__)
    dx = __builtin_hcu_mmac_f32_16x16x16_bf16(ax, bx, cx);
#else
    dx = __builtin_amdgcn_mmac_f32_16x16x16bf16(ax, bx, cx);
#endif
    AccF32x4 out;
    std::memcpy(out.x, &dx, sizeof(out.x));
    return out;
#else
    const int w = wave64_id(), lid = lane64_id(), row = lid & 15, k_grp = lid >> 4;
    uint64_t au = 0, bu = 0;
    std::memcpy(&au, a.x, sizeof(au));
    std::memcpy(&bu, b.x, sizeof(bu));
    __syncthreads();
    sim.a[w][lid] = au;
    sim.b[w][lid] = bu;
    __syncthreads();
    AccF32x4 out = c;
#pragma unroll 1
    for (int i = 0; i < 4; ++i) {
        const int col = k_grp + 4 * i;
        float sum = out.x[i];
#pragma unroll 1
        for (int g = 0; g < 4; ++g) {
            FragB16x4 fa, fb;
            std::memcpy(fa.x, &sim.a[w][g * 16 + row], sizeof(fa.x));
            std::memcpy(fb.x, &sim.b[w][g * 16 + col], sizeof(fb.x));
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                __nv_bfloat16 ba, bb;
                std::memcpy(static_cast<void *>(&ba), &fa.x[e], 2);
                std::memcpy(static_cast<void *>(&bb), &fb.x[e], 2);
                sum = fmaf(__bfloat162float(ba), __bfloat162float(bb), sum);
            }
        }
        out.x[i] = sum;
    }
    __syncthreads();
    return out;
#endif
}

// Hardware v_mmac_i32_16x16x32_i8 (or exact 64-lane DUFragment emulation).
template <int MAX_WAVES64 = 4>
__device__ TQLLM_DUMMA_INLINE AccI32x4 mmac_i32_16x16x32_i8(const FragI8x8 & a, const FragI8x8 & b,
                                                         const AccI32x4 & c, SimScratch<MAX_WAVES64> & sim) {
#if TQLLM_HAS_HW_MMAC
    (void) sim;
    hw_intx4 cx;
    std::memcpy(&cx, c.x, sizeof(cx));
    hw_intx4 dx;
#if defined(__gfx938__)
    using i32x2 = __attribute__((ext_vector_type(2))) int;
    i32x2 ax, bx;
    std::memcpy(&ax, a.x, sizeof(ax));
    std::memcpy(&bx, b.x, sizeof(bx));
    dx = __builtin_hcu_mmac_i32_16x16x32_i8_lit_clamp_lts(ax, bx, cx, false, false, false);
#elif defined(__gfx92a__)
    using i32x2 = __attribute__((ext_vector_type(2))) int;
    i32x2 ax, bx;
    std::memcpy(&ax, a.x, sizeof(ax));
    std::memcpy(&bx, b.x, sizeof(bx));
    dx = __builtin_hcu_mmac_i32_16x16x32_i8_clamp(ax, bx, cx, 0);
#else
    long ax, bx;
    std::memcpy(&ax, a.x, sizeof(ax));
    std::memcpy(&bx, b.x, sizeof(bx));
    dx = __builtin_amdgcn_mmac_i32_16x16x32i8(ax, bx, cx);
#endif
    AccI32x4 out;
    std::memcpy(out.x, &dx, sizeof(out.x));
    return out;
#else
    const int w = wave64_id(), lid = lane64_id(), row = lid & 15, k_grp = lid >> 4;
    uint64_t au = 0, bu = 0;
    std::memcpy(&au, a.x, sizeof(au));
    std::memcpy(&bu, b.x, sizeof(bu));
    __syncthreads();
    sim.a[w][lid] = au;
    sim.b[w][lid] = bu;
    __syncthreads();
    AccI32x4 out = c;
#pragma unroll 1
    for (int i = 0; i < 4; ++i) {
        const int col = k_grp + 4 * i;
        int32_t sum = out.x[i];
#pragma unroll 1
        for (int g = 0; g < 4; ++g) {
            FragI8x8 fa, fb;
            std::memcpy(fa.x, &sim.a[w][g * 16 + row], sizeof(fa.x));
            std::memcpy(fb.x, &sim.b[w][g * 16 + col], sizeof(fb.x));
#pragma unroll
            for (int e = 0; e < 8; ++e) sum += int32_t(fa.x[e]) * int32_t(fb.x[e]);
        }
        out.x[i] = sum;
    }
    __syncthreads();
    return out;
#endif
}

}  // namespace tqllm_dumma
