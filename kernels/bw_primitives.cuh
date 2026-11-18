#pragma once
// Semantic-layer primitives for Hygon BW / GPUFusion (gfx936) and optional CUDA emulation (TQLLM_BW_EMULATE).
// Keeps NVIDIA PTX untouched in default CUDA builds while providing exact rounding, byte-permute, logical 32-lane
// subgroup operations, ldmatrix, mma, and synchronous shared-memory staging on BW wave64 devices.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstring>

#if defined(TQLLM_BW) || defined(__CUDA_MOCKER__) || defined(__HIP_PLATFORM_AMD__)
#define TQLLM_BW_BACKEND 1
#else
#define TQLLM_BW_BACKEND 0
#endif

#if TQLLM_BW_BACKEND || defined(TQLLM_BW_EMULATE)
#define TQLLM_USE_BW_PRIMITIVES 1
#else
#define TQLLM_USE_BW_PRIMITIVES 0
#endif

namespace tqllm_bw {

__host__ __device__ __forceinline__ __half2 u32_to_h2(uint32_t u) {
    __half2 h;
    std::memcpy(static_cast<void *>(&h), &u, sizeof(h));
    return h;
}

__host__ __device__ __forceinline__ uint32_t h2_to_u32(__half2 h) {
    uint32_t u;
    std::memcpy(&u, static_cast<const void *>(&h), sizeof(u));
    return u;
}

__host__ __device__ __forceinline__ __nv_bfloat162 u32_to_bf162(uint32_t u) {
    __nv_bfloat162 b;
    std::memcpy(static_cast<void *>(&b), &u, sizeof(b));
    return b;
}

__host__ __device__ __forceinline__ uint32_t bf162_to_u32(__nv_bfloat162 b) {
    uint32_t u;
    std::memcpy(&u, static_cast<const void *>(&b), sizeof(u));
    return u;
}

// Packed FP16x2 arithmetic (maps to v_pk_mul_f16 / v_pk_add_f16 / v_pk_fma_f16 on gfx936).
__device__ __forceinline__ uint32_t hmul2u(uint32_t a, uint32_t b) {
    return h2_to_u32(__hmul2(u32_to_h2(a), u32_to_h2(b)));
}

__device__ __forceinline__ uint32_t hadd2u(uint32_t a, uint32_t b) {
    return h2_to_u32(__hadd2(u32_to_h2(a), u32_to_h2(b)));
}

__device__ __forceinline__ uint32_t hfma2u(uint32_t a, uint32_t b, uint32_t c) {
    return h2_to_u32(__hfma2(u32_to_h2(a), u32_to_h2(b), u32_to_h2(c)));
}

// Exact PTX `prmt.b32 d, a, b, sel` default-mode semantics (including sign-replicate when selector bit 3 is set).
__host__ __device__ constexpr uint32_t prmt_b32(uint32_t a, uint32_t b, uint32_t sel) {
    const uint64_t ab = (uint64_t(b) << 32) | uint64_t(a);
    uint32_t out = 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t ctl = (sel >> (4 * i)) & 0xFu;
        const uint32_t byte = uint32_t((ab >> (8 * (ctl & 0x7u))) & 0xFFu);
        const uint32_t val = (ctl & 0x8u) ? ((byte & 0x80u) ? 0xFFu : 0x00u) : byte;
        out |= val << (8 * i);
    }
    return out;
}

// Hardware 1-cycle AMDGCN `v_perm_b32` override for CUDA `__byte_perm(x, y, s)`:
// GPUFusion's built-in `__byte_perm` uses a stack `union { uint32_t ui[2]; uint8_t c[8]; }` indexed dynamically,
// which forces LLVM AMDGPU to emit private-segment HBM scratch loads/stores on every nibble unpack.
__device__ __forceinline__ uint32_t byte_perm(uint32_t x, uint32_t y, uint32_t s) {
#if TQLLM_BW_BACKEND && defined(__AMDGCN__)
    const uint32_t t = (s | (s << 8)) & 0x00770077u;
    const uint32_t sel = (t | (t << 4)) & 0x07070707u;
    return __builtin_amdgcn_perm(y, x, sel);
#else
    return ::__byte_perm(x, y, s);
#endif
}

#if TQLLM_BW_BACKEND && defined(__AMDGCN__)
#undef __byte_perm
#define __byte_perm(x, y, s) ::tqllm_bw::byte_perm((x), (y), (s))
#endif

// Logical 32-lane subgroup helpers: safe on both 32-lane NVIDIA warps and 64-lane Hygon BW wavefronts.
__device__ __forceinline__ uint32_t shfl32_u32(uint32_t val, int src_lane, uint32_t mask = 0xffffffffu) {
#if TQLLM_BW_BACKEND && defined(__AMDGCN__)
    (void) mask;
    const int hw_lane = static_cast<int>(__builtin_amdgcn_mbcnt_hi(-1, __builtin_amdgcn_mbcnt_lo(-1, 0)));
    const int target = (hw_lane & 32) | (src_lane & 31);
    return static_cast<uint32_t>(__builtin_amdgcn_ds_bpermute(target << 2, static_cast<int>(val)));
#else
    return __shfl_sync(mask, val, src_lane, 32);
#endif
}

__device__ __forceinline__ uintptr_t shfl32_ptr(const void * p, int src_lane, uint32_t mask = 0xffffffffu) {
    const uint64_t raw = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p));
    const uint32_t lo = shfl32_u32(static_cast<uint32_t>(raw), src_lane, mask);
    const uint32_t hi = shfl32_u32(static_cast<uint32_t>(raw >> 32), src_lane, mask);
    return static_cast<uintptr_t>((uint64_t(hi) << 32) | uint64_t(lo));
}

__device__ __forceinline__ int any_sync32(uint32_t mask, int pred) {
#if TQLLM_BW_BACKEND && (defined(__AMDGCN__) || defined(__HIP_DEVICE_COMPILE__))
    const unsigned int hw_lane = __builtin_amdgcn_mbcnt_hi(-1, __builtin_amdgcn_mbcnt_lo(-1, 0));
    const unsigned long long b64 = __ballot(pred);
    const uint32_t b32 = static_cast<uint32_t>(hw_lane >= 32u ? (b64 >> 32) : b64);
    return (b32 & mask) != 0u;
#else
    return __any_sync(mask, pred);
#endif
}

// Exact PTX `ldmatrix.sync.aligned.m8n8.x4.shared.b16` for a 32-lane logical warp.
// Lanes 8*m + row (m = 0..3, row = 0..7) supply the row pointers of the four 8x8 b16 matrices.
// Output lane (gid = lane/4, tig = lane%4) receives M_m[gid][2*tig .. 2*tig+1] in r[m].
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void * p) {
    const int lane = int(threadIdx.x) & 31, gid = lane >> 2, tig = lane & 3;
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const uintptr_t row_ptr = shfl32_ptr(p, 8 * m + gid);
        r[m] = *reinterpret_cast<const uint32_t *>(row_ptr + uintptr_t(4 * tig));
    }
}

// Exact PTX `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16` for a 32-lane logical warp.
// Output lane (gid = lane/4, tig = lane%4) receives pack_b16(M_m[2*tig][gid], M_m[2*tig+1][gid]) in r[m].
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void * p) {
    const int lane = int(threadIdx.x) & 31, gid = lane >> 2, tig = lane & 3;
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const uintptr_t p0 = shfl32_ptr(p, 8 * m + 2 * tig);
        const uintptr_t p1 = shfl32_ptr(p, 8 * m + 2 * tig + 1);
        const uint16_t lo = reinterpret_cast<const uint16_t *>(p0)[gid];
        const uint16_t hi = reinterpret_cast<const uint16_t *>(p1)[gid];
        r[m] = uint32_t(lo) | (uint32_t(hi) << 16);
    }
}

// Exact PTX `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` on a 32-lane logical subgroup.
// Immune to half-wavefront divergence (e.g. split-K tails where only logical warp 0 is active).
__device__ inline void mma_f16(float * c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                               uint32_t b1) {
    const int lane = int(threadIdx.x) & 31, gid = lane >> 2, tig = lane & 3;
    const int col0 = 2 * tig, col1 = 2 * tig + 1;
#pragma unroll 1
    for (int u = 0; u < 4; ++u) {
        const float2 fa0 = __half22float2(u32_to_h2(shfl32_u32(a0, 4 * gid + u)));
        const float2 fa1 = __half22float2(u32_to_h2(shfl32_u32(a1, 4 * gid + u)));
        const float2 fa2 = __half22float2(u32_to_h2(shfl32_u32(a2, 4 * gid + u)));
        const float2 fa3 = __half22float2(u32_to_h2(shfl32_u32(a3, 4 * gid + u)));
        const float2 fb0_0 = __half22float2(u32_to_h2(shfl32_u32(b0, 4 * col0 + u)));
        const float2 fb1_0 = __half22float2(u32_to_h2(shfl32_u32(b1, 4 * col0 + u)));
        const float2 fb0_1 = __half22float2(u32_to_h2(shfl32_u32(b0, 4 * col1 + u)));
        const float2 fb1_1 = __half22float2(u32_to_h2(shfl32_u32(b1, 4 * col1 + u)));
        c[0] = fmaf(fa2.y, fb1_0.y, fmaf(fa2.x, fb1_0.x, fmaf(fa0.y, fb0_0.y, fmaf(fa0.x, fb0_0.x, c[0]))));
        c[1] = fmaf(fa2.y, fb1_1.y, fmaf(fa2.x, fb1_1.x, fmaf(fa0.y, fb0_1.y, fmaf(fa0.x, fb0_1.x, c[1]))));
        c[2] = fmaf(fa3.y, fb1_0.y, fmaf(fa3.x, fb1_0.x, fmaf(fa1.y, fb0_0.y, fmaf(fa1.x, fb0_0.x, c[2]))));
        c[3] = fmaf(fa3.y, fb1_1.y, fmaf(fa3.x, fb1_1.x, fmaf(fa1.y, fb0_1.y, fmaf(fa1.x, fb0_1.x, c[3]))));
    }
}

// Exact PTX `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` on a 32-lane logical subgroup.
__device__ inline void mma_bf16(float * c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                uint32_t b1) {
    const int lane = int(threadIdx.x) & 31, gid = lane >> 2, tig = lane & 3;
    const int col0 = 2 * tig, col1 = 2 * tig + 1;
#pragma unroll 1
    for (int u = 0; u < 4; ++u) {
        const float2 fa0 = __bfloat1622float2(u32_to_bf162(shfl32_u32(a0, 4 * gid + u)));
        const float2 fa1 = __bfloat1622float2(u32_to_bf162(shfl32_u32(a1, 4 * gid + u)));
        const float2 fa2 = __bfloat1622float2(u32_to_bf162(shfl32_u32(a2, 4 * gid + u)));
        const float2 fa3 = __bfloat1622float2(u32_to_bf162(shfl32_u32(a3, 4 * gid + u)));
        const float2 fb0_0 = __bfloat1622float2(u32_to_bf162(shfl32_u32(b0, 4 * col0 + u)));
        const float2 fb1_0 = __bfloat1622float2(u32_to_bf162(shfl32_u32(b1, 4 * col0 + u)));
        const float2 fb0_1 = __bfloat1622float2(u32_to_bf162(shfl32_u32(b0, 4 * col1 + u)));
        const float2 fb1_1 = __bfloat1622float2(u32_to_bf162(shfl32_u32(b1, 4 * col1 + u)));
        c[0] = fmaf(fa2.y, fb1_0.y, fmaf(fa2.x, fb1_0.x, fmaf(fa0.y, fb0_0.y, fmaf(fa0.x, fb0_0.x, c[0]))));
        c[1] = fmaf(fa2.y, fb1_1.y, fmaf(fa2.x, fb1_1.x, fmaf(fa0.y, fb0_1.y, fmaf(fa0.x, fb0_1.x, c[1]))));
        c[2] = fmaf(fa3.y, fb1_0.y, fmaf(fa3.x, fb1_0.x, fmaf(fa1.y, fb0_0.y, fmaf(fa1.x, fb0_0.x, c[2]))));
        c[3] = fmaf(fa3.y, fb1_1.y, fmaf(fa3.x, fb1_1.x, fmaf(fa1.y, fb0_1.y, fmaf(fa1.x, fb0_1.x, c[3]))));
    }
}

// Exact PTX `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` preserving FP16 accumulation semantics via v_pk_fma_f16.
__device__ inline void mma_f16acc(uint32_t * c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                  uint32_t b0, uint32_t b1) {
    const int lane = int(threadIdx.x) & 31, gid = lane >> 2, tig = lane & 3;
    const int col0 = 2 * tig, col1 = 2 * tig + 1;
    __half2 hc0 = u32_to_h2(c[0]), hc1 = u32_to_h2(c[1]);
#pragma unroll 1
    for (int u = 0; u < 4; ++u) {
        const __half2 ha0 = u32_to_h2(shfl32_u32(a0, 4 * gid + u));
        const __half2 ha1 = u32_to_h2(shfl32_u32(a1, 4 * gid + u));
        const __half2 ha2 = u32_to_h2(shfl32_u32(a2, 4 * gid + u));
        const __half2 ha3 = u32_to_h2(shfl32_u32(a3, 4 * gid + u));
        const __half2 hb0_0 = u32_to_h2(shfl32_u32(b0, 4 * col0 + u));
        const __half2 hb1_0 = u32_to_h2(shfl32_u32(b1, 4 * col0 + u));
        const __half2 hb0_1 = u32_to_h2(shfl32_u32(b0, 4 * col1 + u));
        const __half2 hb1_1 = u32_to_h2(shfl32_u32(b1, 4 * col1 + u));

        const __half2 bk0 = __halves2half2(__low2half(hb0_0), __low2half(hb0_1));
        const __half2 bk1 = __halves2half2(__high2half(hb0_0), __high2half(hb0_1));
        const __half2 bk2 = __halves2half2(__low2half(hb1_0), __low2half(hb1_1));
        const __half2 bk3 = __halves2half2(__high2half(hb1_0), __high2half(hb1_1));

        hc0 = __hfma2(__low2half2(ha0), bk0, hc0);
        hc0 = __hfma2(__high2half2(ha0), bk1, hc0);
        hc0 = __hfma2(__low2half2(ha2), bk2, hc0);
        hc0 = __hfma2(__high2half2(ha2), bk3, hc0);

        hc1 = __hfma2(__low2half2(ha1), bk0, hc1);
        hc1 = __hfma2(__high2half2(ha1), bk1, hc1);
        hc1 = __hfma2(__low2half2(ha3), bk2, hc1);
        hc1 = __hfma2(__high2half2(ha3), bk3, hc1);
    }
    c[0] = h2_to_u32(hc0);
    c[1] = h2_to_u32(hc1);
}

// Exact PTX `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` on a 32-lane logical subgroup via __dp4a.
__device__ inline void mma_s8(int (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                              uint32_t b1) {
    const int lane = int(threadIdx.x) & 31, gid = lane >> 2, tig = lane & 3;
    const int col0 = 2 * tig, col1 = 2 * tig + 1;
#pragma unroll 1
    for (int u = 0; u < 4; ++u) {
        const int qa0 = static_cast<int>(shfl32_u32(a0, 4 * gid + u));
        const int qa1 = static_cast<int>(shfl32_u32(a1, 4 * gid + u));
        const int qa2 = static_cast<int>(shfl32_u32(a2, 4 * gid + u));
        const int qa3 = static_cast<int>(shfl32_u32(a3, 4 * gid + u));
        const int qb0_0 = static_cast<int>(shfl32_u32(b0, 4 * col0 + u));
        const int qb1_0 = static_cast<int>(shfl32_u32(b1, 4 * col0 + u));
        const int qb0_1 = static_cast<int>(shfl32_u32(b0, 4 * col1 + u));
        const int qb1_1 = static_cast<int>(shfl32_u32(b1, 4 * col1 + u));
        c[0] = __dp4a(qa2, qb1_0, __dp4a(qa0, qb0_0, c[0]));
        c[1] = __dp4a(qa2, qb1_1, __dp4a(qa0, qb0_1, c[1]));
        c[2] = __dp4a(qa3, qb1_0, __dp4a(qa1, qb0_0, c[2]));
        c[3] = __dp4a(qa3, qb1_1, __dp4a(qa1, qb0_1, c[3]));
    }
}

// Exact PTX `mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32` (used in tools/tqkbench.cu).
__device__ __forceinline__ void mma_s8_k16(int (&c)[4], uint32_t a0, uint32_t a1, uint32_t b) {
    const int lane = int(threadIdx.x) & 31, gid = lane >> 2, tig = lane & 3;
    const int col0 = 2 * tig, col1 = 2 * tig + 1;
#pragma unroll 1
    for (int u = 0; u < 4; ++u) {
        const int qa0 = static_cast<int>(shfl32_u32(a0, 4 * gid + u));
        const int qa1 = static_cast<int>(shfl32_u32(a1, 4 * gid + u));
        const int qb0 = static_cast<int>(shfl32_u32(b, 4 * col0 + u));
        const int qb1 = static_cast<int>(shfl32_u32(b, 4 * col1 + u));
        c[0] = __dp4a(qa0, qb0, c[0]);
        c[1] = __dp4a(qa0, qb1, c[1]);
        c[2] = __dp4a(qa1, qb0, c[2]);
        c[3] = __dp4a(qa1, qb1, c[3]);
    }
}

// Synchronous global-to-shared copy replacements when cp.async is unavailable.
__device__ __forceinline__ void cp_async16(void * dst, const void * src) {
    *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(src);
}

__device__ __forceinline__ void cp_async4(void * dst, const void * src) {
    *reinterpret_cast<uint32_t *>(dst) = *reinterpret_cast<const uint32_t *>(src);
}

__device__ __forceinline__ void cp_commit() {}
template <int N = 0> __device__ __forceinline__ void cp_wait() {}

}  // namespace tqllm_bw
