#pragma once
// BW gfx936 prefill-only, chunkwise (WY) gated delta recurrence.
// A 64-lane MMAC wave processes 16 time steps x 16 value channels per chunk.
// Decode / speculative trees continue using gdn_multi_k unchanged.
#include "bw_dumma.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

namespace gdn_bw {

constexpr int K = 128, C = 16, V = 16;

// Four independent wave64s per CTA, one wave per (token, key head).
// Unlike a block-per-(token, value head), this avoids 144k small CTAs at P=3000.
// Normalized Q/K are FP16 for DUMMA; decay and beta stay FP32.
__global__ void __launch_bounds__(256) prep(
    const __nv_bfloat16 *qkv, const __nv_bfloat16 *a, const __nv_bfloat16 *b, int ab_stride,
    const float *neg_a, const float *dt_bias,
    __half *q, __half *k, float2 *gates, __half *kt,
    int n, int kh, int vh, int dv, bool tiled)
{
    const int item = blockIdx.x * 4 + (threadIdx.x >> 6);
    if (item >= n * kh) return;
    const int t = item / kh, h = item % kh;
    const int lane = threadIdx.x & 63;
    const int stride = 2 * kh * K + vh * dv;
    const __nv_bfloat16 *row = qkv + int64_t(t) * stride;
    const float q0 = __bfloat162float(row[h * K + lane]);
    const float q1 = __bfloat162float(row[h * K + lane + 64]);
    const float k0 = __bfloat162float(row[(kh + h) * K + lane]);
    const float k1 = __bfloat162float(row[(kh + h) * K + lane + 64]);
    tqllm_dumma::SimScratch<4> sim;
    const float nq = rsqrtf(tqllm_dumma::wave64_sum(q0 * q0 + q1 * q1, sim) + 1e-6f)
                     * 0.08838834764831845f;
    const float nk = rsqrtf(tqllm_dumma::wave64_sum(k0 * k0 + k1 * k1, sim) + 1e-6f);
    // Token-contiguous per key head, enabling direct MMAC loads.
    const int64_t off = (int64_t(h) * n + t) * K + lane;
    q[off] = __float2half_rn(q0 * nq);
    q[off + 64] = __float2half_rn(q1 * nq);
    k[off] = __float2half_rn(k0 * nk);
    k[off + 64] = __float2half_rn(k1 * nk);
    if (kt) {
        const int chunks = (n + C - 1) / C, col = t % C;
        const int64_t base = (int64_t(h) * chunks + t / C) * K * C;
        kt[base + lane * C + col] = __float2half_rn(k0 * nk);
        kt[base + (lane + 64) * C + col] = __float2half_rn(k1 * nk);
    }

    if (lane < vh / kh) {
        const int head = tiled ? lane * kh + h : h * (vh / kh) + lane;
        const float gv = __bfloat162float(b[int64_t(t) * ab_stride + head]);
        const float xv = __bfloat162float(a[int64_t(t) * ab_stride + head]) + dt_bias[head];
        const float softplus = xv > 20.f ? xv : log1pf(expf(xv));
        gates[int64_t(t) * vh + head] =
            make_float2(1.f / (1.f + expf(-gv)), expf(neg_a[head] * softplus));
    }
}

struct alignas(16) Scratch {
    // Stride 20 halves keeps the K-transposed state-update operand 8-byte
    // aligned while avoiding the 16-way LDS bank conflict of stride 16.
    __half k[C][K], q[C][K], kt[K][C + 4], state_h[K][V], delta_h[C][V];
    float state[K][V];
    float gram[C][C], gram_q[C][C];
    float ks[C][V], qs[C][V], delta[C][V];
    float beta[C], decay[C], prefix[C], suffix[C];
};

// One CTA (one 64-lane matrix wave) per value head and 16 value channels.
// The only recurrence loop is over n / 16 chunks. Within a chunk, both
// projections against the previous state and the state update are MMACs.
__global__ void __launch_bounds__(64) recurrence(
    const __nv_bfloat16 *qkv, const __half *q, const __half *k, const float2 *gates,
    const float *initial, float *next, __nv_bfloat16 *output,
    int n, int kh, int vh, int dv, bool tiled, unsigned long long *trace)
{
    __shared__ Scratch sh;
    tqllm_dumma::SimScratch<1> sim;
    const int h = blockIdx.x, v0 = blockIdx.y * V, tid = threadIdx.x;
    const int row = tid & 15, cg = tid >> 4;
    const int key_head = tiled ? h % kh : h / (vh / kh);
    const int stride = 2 * kh * K + vh * dv;
    unsigned long long phase[4] = {0, 0, 0, 0};
    unsigned long long stamp = clock64();

    for (int i = tid; i < K * V; i += 64) {
        const int r = i / V, v = i % V;
        sh.state[r][v] = initial[(int64_t(h) * K + r) * dv + v0 + v];
    }
    __syncthreads();

    for (int t0 = 0; t0 < n; t0 += C) {
        const int cn = min(C, n - t0);
        for (int i = tid; i < C * K; i += 64) {
            const int t = i / K, d = i % K;
            const int64_t ix = (int64_t(key_head) * n + t0 + t) * K + d;
            const __half kv = t < cn ? k[ix] : __float2half_rn(0.f);
            sh.k[t][d] = kv;
            sh.q[t][d] = t < cn ? q[ix] : __float2half_rn(0.f);
            sh.kt[d][t] = kv;
        }
        for (int i = tid; i < K * V; i += 64) {
            const int r = i / V, v = i % V;
            sh.state_h[r][v] = __float2half_rn(sh.state[r][v]);
        }
        if (tid < C) {
            const float2 gb = tid < cn ? gates[int64_t(t0 + tid) * vh + h] : make_float2(0.f, 1.f);
            sh.beta[tid] = gb.x;
            sh.decay[tid] = gb.y;
        }
        __syncthreads();
        if (tid == 0) {
            float prod = 1.f;
            for (int t = 0; t < cn; ++t) {
                prod *= sh.decay[t];
                sh.prefix[t] = prod;
            }
            prod = 1.f;
            for (int t = cn - 1; t >= 0; --t) {
                sh.suffix[t] = prod;
                prod *= sh.decay[t];
            }
        }
        __syncthreads();
        if (trace && tid == 0) { const auto now = clock64(); phase[0] += now - stamp; stamp = now; }

        // K*K^T and Q*K^T, 16x16 each, with normalized FP16 input.
        auto gram = tqllm_dumma::zero_f32x4();
        auto gram_q = tqllm_dumma::zero_f32x4();
        auto proj_k = tqllm_dumma::zero_f32x4();
        auto proj_q = tqllm_dumma::zero_f32x4();
        for (int kd = 0; kd < K; kd += 16) {
            const auto ak = tqllm_dumma::load_a_rowmajor_b16(&sh.k[0][kd], K);
            const auto aq = tqllm_dumma::load_a_rowmajor_b16(&sh.q[0][kd], K);
            const auto bk = tqllm_dumma::load_b_colmajor_b16(&sh.k[0][kd], K);
            const auto bs = tqllm_dumma::load_b_rowmajor_b16(&sh.state_h[kd][0], V);
            gram = tqllm_dumma::mmac_f32_16x16x16_f16(ak, bk, gram, sim);
            gram_q = tqllm_dumma::mmac_f32_16x16x16_f16(aq, bk, gram_q, sim);
            proj_k = tqllm_dumma::mmac_f32_16x16x16_f16(ak, bs, proj_k, sim);
            proj_q = tqllm_dumma::mmac_f32_16x16x16_f16(aq, bs, proj_q, sim);
        }
        for (int i = 0; i < 4; ++i) {
            const int col = cg + 4 * i;
            sh.gram[row][col] = gram.x[i];
            sh.gram_q[row][col] = gram_q.x[i];
            sh.ks[row][col] = proj_k.x[i];
            sh.qs[row][col] = proj_q.x[i];
        }
        __syncthreads();
        if (trace && tid == 0) { const auto now = clock64(); phase[1] += now - stamp; stamp = now; }
        // Cache the gated triangular coefficients once per (t,j), rather than
        // rebuilding decay products in every value-channel recurrence step.
        if (tid < C) {
            const int t = tid;
            float ratio = 1.f;
            for (int j = t; j >= 0; --j) {
                sh.gram[t][j] *= sh.beta[t] * ratio;
                sh.gram_q[t][j] *= ratio;
                ratio *= sh.decay[j];
            }
        }
        __syncthreads();

        // Solve (I + A) E = beta * (V - g K S0), per value channel.
        // A[t,j] = beta[t] * (product_{u=j+1..t} decay[u]) * K[t].K[j].
        if (tid < V) {
            const int v = tid;
            for (int t = 0; t < cn; ++t) {
                const float value = __bfloat162float(
                    qkv[int64_t(t0 + t) * stride + 2 * kh * K + h * dv + v0 + v]);
                float e = sh.beta[t] * (value - sh.prefix[t] * sh.ks[t][v]);
                for (int j = 0; j < t; ++j)
                    e -= sh.gram[t][j] * sh.delta[j][v];
                sh.delta[t][v] = e;
                float out = sh.prefix[t] * sh.qs[t][v];
                for (int j = 0; j <= t; ++j)
                    out += sh.gram_q[t][j] * sh.delta[j][v];
                output[(int64_t(t0 + t) * vh + h) * dv + v0 + v] = __float2bfloat16_rn(out);
            }
        }
        __syncthreads();
        if (trace && tid == 0) { const auto now = clock64(); phase[2] += now - stamp; stamp = now; }

        // S_end = g_C S0 + K^T diag(g_C / g_j) E. For the MMAC
        // accumulator we load g_C S0 in FP32, avoiding state truncation.
        for (int i = tid; i < C * V; i += 64) {
            const int t = i / V, v = i % V;
            const float d = t < cn ? sh.delta[t][v] * sh.suffix[t] : 0.f;
            sh.delta_h[t][v] = __float2half_rn(d);
        }
        __syncthreads();
        for (int r0 = 0; r0 < K; r0 += 16) {
            tqllm_dumma::AccF32x4 acc;
            for (int i = 0; i < 4; ++i)
                acc.x[i] = sh.prefix[cn - 1] * sh.state[r0 + row][cg + 4 * i];
            const auto ak = tqllm_dumma::load_a_rowmajor_b16(&sh.kt[r0][0], C + 4);
            const auto bd = tqllm_dumma::load_b_rowmajor_b16(&sh.delta_h[0][0], V);
            acc = tqllm_dumma::mmac_f32_16x16x16_f16(ak, bd, acc, sim);
            for (int i = 0; i < 4; ++i)
                sh.state[r0 + row][cg + 4 * i] = acc.x[i];
        }
        __syncthreads();
        if (trace && tid == 0) { const auto now = clock64(); phase[3] += now - stamp; stamp = now; }
    }

    for (int i = tid; i < K * V; i += 64) {
        const int r = i / V, v = i % V;
        next[(int64_t(h) * K + r) * dv + v0 + v] = sh.state[r][v];
    }
    if (trace && h == 0 && v0 == 0 && tid == 0)
        for (int i = 0; i < 4; ++i) trace[i] = phase[i];
}

#include "gdn_bw_precomp.cuh"

// The scratch is shared by calls inside a single model process, just as the
// existing prefill GEMM workspaces are. Never allocate during graph capture.
inline bool launch(
    const __nv_bfloat16 *qkv, int n, const __nv_bfloat16 *a, const __nv_bfloat16 *b, int ab_stride,
    const float *neg_a, const float *dt_bias, const float *initial, float *next,
    __nv_bfloat16 *output, int kh, int vh, int dv, bool tiled, cudaStream_t stream)
{
    const bool precomp = std::getenv("TQLLM_GDN_PRECOMP") != nullptr;
    const bool compensated = std::getenv("TQLLM_GDN_COMPENSATED") != nullptr;
    const size_t qsize = size_t(n) * kh * K, gsize = size_t(n) * vh;
    const int chunks = (n + C - 1) / C;
    const size_t coeff = size_t(vh) * chunks * C * C;
    const size_t transposed = size_t(kh) * chunks * K * C;
    const size_t bytes = 2 * qsize * sizeof(__half) + gsize * sizeof(float2) +
                         (precomp ? (2 * coeff + transposed) * sizeof(__half) : 0);
    static void *work = nullptr;
    static size_t capacity = 0;
    if (bytes > capacity) {
        cudaStreamCaptureStatus cs;
        cudaStreamIsCapturing(stream, &cs);
        if (cs != cudaStreamCaptureStatusNone) return false;
        if (work) cudaFree(work);
        if (cudaMalloc(&work, bytes) != cudaSuccess) return false;
        capacity = bytes;
    }
    auto *q = static_cast<__half *>(work);
    auto *k = q + qsize;
    auto *gates = reinterpret_cast<float2 *>(k + qsize);
    auto *winv = reinterpret_cast<__half *>(gates + gsize);
    auto *matm = winv + coeff;
    auto *kt = matm + coeff;
    const bool profile = std::getenv("TQLLM_GDN_TRACE") != nullptr;
    const bool phases = std::getenv("TQLLM_GDN_PHASES") != nullptr;
    const int phase_count = precomp ? 6 : 4;
    unsigned long long * detail = nullptr;
    if (phases) {
        cudaMalloc(&detail, sizeof(unsigned long long) * phase_count);
        cudaMemset(detail, 0, sizeof(unsigned long long) * phase_count);
    }
    cudaEvent_t e0{}, e1{}, e2{}, e3{};
    if (profile) {
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventCreate(&e2); cudaEventCreate(&e3);
        cudaEventRecord(e0, stream);
    }
    if (precomp) cudaMemsetAsync(kt, 0, transposed * sizeof(__half), stream);
    prep<<<(n * kh + 3) / 4, 256, 0, stream>>>(
        qkv, a, b, ab_stride, neg_a, dt_bias, q, k, gates,
        precomp ? kt : nullptr, n, kh, vh, dv, tiled);
    if (profile) cudaEventRecord(e1, stream);
    if (precomp) {
        precompute<<<dim3(vh, chunks), 64, 0, stream>>>(
            q, k, gates, winv, matm, n, kh, vh, tiled);
    }
    if (profile) cudaEventRecord(e2, stream);
    if (precomp) {
        if (compensated)
            recurrence_precomp<true><<<dim3(vh, dv / PV), 64, 0, stream>>>(
                qkv, q, k, kt, gates, winv, matm, initial, next, output,
                n, kh, vh, dv, tiled, detail);
        else
            recurrence_precomp<false><<<dim3(vh, dv / PV), 64, 0, stream>>>(
                qkv, q, k, kt, gates, winv, matm, initial, next, output,
                n, kh, vh, dv, tiled, detail);
    } else {
        recurrence<<<dim3(vh, dv / V), 64, 0, stream>>>(
            qkv, q, k, gates, initial, next, output, n, kh, vh, dv, tiled, detail);
    }
    if (profile) {
        cudaEventRecord(e3, stream);
        cudaEventSynchronize(e3);
        float prep_ms = 0.f, coeff_ms = 0.f, recur_ms = 0.f;
        cudaEventElapsedTime(&prep_ms, e0, e1);
        cudaEventElapsedTime(&coeff_ms, e1, e2);
        cudaEventElapsedTime(&recur_ms, e2, e3);
        std::fprintf(stderr, "gdn_bw n=%d: prep %.3f ms precompute %.3f ms recurrence %.3f ms\n",
                     n, prep_ms, coeff_ms, recur_ms);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        cudaEventDestroy(e2); cudaEventDestroy(e3);
    }
    if (phases) {
        unsigned long long ticks[6]{};
        cudaMemcpy(ticks, detail, phase_count * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, 0);
        std::fprintf(stderr, "gdn_bw phases n=%d: ", n);
        for (int i = 0; i < phase_count; ++i)
            std::fprintf(stderr, "%.2f ", ticks[i] / double(prop.clockRate));
        std::fprintf(stderr, "ms/CTA\n");
        cudaFree(detail);
    }
    return true;
}

} // namespace gdn_bw
