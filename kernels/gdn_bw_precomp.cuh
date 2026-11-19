// Optional BW1000 GDN prefill with precomputed chunk-local triangular inverse.
// Included from gdn_chunk_bw.cuh inside namespace gdn_bw, after Scratch is defined.
// The per-(head,chunk) coefficient work is independent of recurrence state and is
// executed in parallel rather than repeated in each value-column CTA.

__global__ void __launch_bounds__(64) precompute(
    const __half *q, const __half *k, const float2 *gates,
    __half *winv, __half *mout, int n, int kh, int vh, bool tiled)
{
    __shared__ __half sq[C][K], sk[C][K];
    __shared__ float gg[C][C], mg[C][C];
    __shared__ float a[C], beta[C];
    const int h = blockIdx.x, chunk = blockIdx.y, nc = (n + C - 1) / C;
    const int t0 = chunk * C, cn = min(C, n - t0);
    const int khd = tiled ? h % kh : h / (vh / kh);
    const int tid = threadIdx.x, row = tid & 15, g = tid >> 4;
    for (int i = tid; i < C * K; i += 64) {
        const int t = i / K, d = i % K;
        const int64_t ix = (int64_t(khd) * n + t0 + t) * K + d;
        sq[t][d] = t < cn ? q[ix] : __float2half_rn(0.f);
        sk[t][d] = t < cn ? k[ix] : __float2half_rn(0.f);
    }
    if (tid < C) {
        const float2 gb = tid < cn ? gates[int64_t(t0 + tid) * vh + h] : make_float2(0.f, 1.f);
        beta[tid] = gb.x;
        a[tid] = gb.y;
    }
    __syncthreads();
    auto gkk = tqllm_dumma::zero_f32x4();
    auto gqk = tqllm_dumma::zero_f32x4();
    tqllm_dumma::SimScratch<1> sim;
    for (int kd = 0; kd < K; kd += 16) {
        const auto ak = tqllm_dumma::load_a_rowmajor_b16(&sk[0][kd], K);
        const auto aq = tqllm_dumma::load_a_rowmajor_b16(&sq[0][kd], K);
        const auto bk = tqllm_dumma::load_b_colmajor_b16(&sk[0][kd], K);
        gkk = tqllm_dumma::mmac_f32_16x16x16_f16(ak, bk, gkk, sim);
        gqk = tqllm_dumma::mmac_f32_16x16x16_f16(aq, bk, gqk, sim);
    }
    for (int i = 0; i < 4; ++i) {
        gg[row][g + 4 * i] = gkk.x[i];
        mg[row][g + 4 * i] = gqk.x[i];
    }
    __syncthreads();

    // Each lane j owns a whole column of (I+A)^-1, so solving the
    // lower-triangular system requires no inter-lane synchronization.
    // W[t,j] = delta(t,j) - sum_{l=j}^{t-1} A[t,l] W[l,j].
    if (tid < C) {
        const int j = tid;
        float wcol[C]{};
        for (int t = j; t < C; ++t) {
            float val = t == j ? 1.f : 0.f;
            float ratio = 1.f;
            for (int l = t - 1; l >= j; --l) {
                ratio *= a[l + 1];
                val -= beta[t] * ratio * gg[t][l] * wcol[l];
            }
            wcol[t] = val;
        }
        __half *wdst = winv + (int64_t(h) * nc + chunk) * C * C;
        for (int t = 0; t < C; ++t)
            wdst[t * C + j] = __float2half_rn(wcol[t]);
    }
    // M[t,j] = prod_{u=j+1}^t a[u] * Q[t].K[j] (j<=t)
    const int64_t offset = (int64_t(h) * nc + chunk) * C * C;
    for (int i = tid; i < C * C; i += 64) {
        const int t = i / C, j = i % C;
        float v = 0.f;
        if (t < cn && j <= t) {
            float ratio = 1.f;
            for (int u = j + 1; u <= t; ++u) ratio *= a[u];
            v = ratio * mg[t][j];
        }
        mout[offset + i] = __float2half_rn(v);
    }
}

constexpr int PV = 16;
struct alignas(16) ScratchPC {
    __half k[C][K], q[C][K], state_h[K][PV];
    __half rhs_h[C][PV], solved_h[C][PV], delta_h[C][PV];
    float state[K][PV], ks[C][PV], qs[C][PV], solved[C][PV];
    float beta[C], decay[C], prefix[C], suffix[C];
};

// One wave64 solves 16 value columns. The two-wave variant lost throughput
// on gfx936 because of larger LDS footprint and reduced block residency.
template <bool Accurate>
__global__ void __launch_bounds__(64) recurrence_precomp(
    const __nv_bfloat16 *qkv, const __half *q, const __half *k, const __half *kt,
    const float2 *gates, const __half *winv, const __half *matm,
    const float *initial, float *next, __nv_bfloat16 *output,
    int n, int kh, int vh, int dv, bool tiled, unsigned long long *trace)
{
    __shared__ ScratchPC sh;
    // The compensated variant approximates FP32 matrix products by accumulating
    // the FP16 high part and the FP16 rounding residual with a second DUMMA.
    __shared__ __half state_lo_h[K][PV], rhs_lo_h[C][PV];
    __shared__ __half solved_lo_h[C][PV], delta_lo_h[C][PV];
    tqllm_dumma::SimScratch<1> sim;
    const int h = blockIdx.x, v0 = blockIdx.y * PV, tid = threadIdx.x;
    const int row = tid & 15, cg = (tid & 63) >> 4, wv = (tid >> 6) * V;
    const int nc = (n + C - 1) / C;
    const int khd = tiled ? h % kh : h / (vh / kh);
    const int stride = 2 * kh * K + vh * dv;
    unsigned long long phase[6]{};
    unsigned long long stamp = clock64();

    for (int i = tid; i < K * PV; i += 64) {
        const int r = i / PV, v = i % PV;
        const float value = initial[(int64_t(h) * K + r) * dv + v0 + v];
        sh.state[r][v] = value;
        const __half high = __float2half_rn(value);
        sh.state_h[r][v] = high;
        if constexpr (Accurate)
            state_lo_h[r][v] = __float2half_rn(value - __half2float(high));
    }
    __syncthreads();

    for (int t0 = 0; t0 < n; t0 += C) {
        const int cn = min(C, n - t0);
        const int64_t off = (int64_t(h) * nc + t0 / C) * C * C;
        if (cn < C) {
            // Only the final partial chunk needs padded local K/Q matrices.
            for (int i = tid; i < C * K; i += 64) {
                const int t = i / K, d = i % K;
                const int64_t ix = (int64_t(khd) * n + t0 + t) * K + d;
                sh.k[t][d] = t < cn ? k[ix] : __float2half_rn(0.f);
                sh.q[t][d] = t < cn ? q[ix] : __float2half_rn(0.f);
            }
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

        auto pk = tqllm_dumma::zero_f32x4();
        auto pq = tqllm_dumma::zero_f32x4();
        const __half *ksrc = cn < C ? &sh.k[0][0] : k + (int64_t(khd) * n + t0) * K;
        const __half *qsrc = cn < C ? &sh.q[0][0] : q + (int64_t(khd) * n + t0) * K;
        for (int kd = 0; kd < K; kd += 16) {
            const auto ak = tqllm_dumma::load_a_rowmajor_b16(ksrc + kd, K);
            const auto aq = tqllm_dumma::load_a_rowmajor_b16(qsrc + kd, K);
            const auto bs = tqllm_dumma::load_b_rowmajor_b16(&sh.state_h[kd][wv], PV);
            pk = tqllm_dumma::mmac_f32_16x16x16_f16(ak, bs, pk, sim);
            pq = tqllm_dumma::mmac_f32_16x16x16_f16(aq, bs, pq, sim);
            if constexpr (Accurate) {
                const auto lo = tqllm_dumma::load_b_rowmajor_b16(&state_lo_h[kd][wv], PV);
                pk = tqllm_dumma::mmac_f32_16x16x16_f16(ak, lo, pk, sim);
                pq = tqllm_dumma::mmac_f32_16x16x16_f16(aq, lo, pq, sim);
            }
        }
        for (int i = 0; i < 4; ++i) {
            sh.ks[row][wv + cg + 4 * i] = pk.x[i];
            sh.qs[row][wv + cg + 4 * i] = pq.x[i];
        }
        __syncthreads();
        if (trace && tid == 0) { const auto now = clock64(); phase[1] += now - stamp; stamp = now; }

        for (int i = tid; i < C * PV; i += 64) {
            const int t = i / PV, v = i % PV;
            float rhs = 0.f;
            if (t < cn) {
                const float val = __bfloat162float(
                    qkv[int64_t(t0 + t) * stride + 2 * kh * K + h * dv + v0 + v]);
                rhs = sh.beta[t] * (val - sh.prefix[t] * sh.ks[t][v]);
            }
            const __half hi = __float2half_rn(rhs);
            sh.rhs_h[t][v] = hi;
            if constexpr (Accurate)
                rhs_lo_h[t][v] = __float2half_rn(rhs - __half2float(hi));
        }
        __syncthreads();
        if (trace && tid == 0) { const auto now = clock64(); phase[2] += now - stamp; stamp = now; }

        // E = (I+A)^-1 R, where the triangular inverse is reused for all V tiles.
        auto e = tqllm_dumma::zero_f32x4();
        const auto aw = tqllm_dumma::load_a_rowmajor_b16(winv + off, C);
        const auto br = tqllm_dumma::load_b_rowmajor_b16(&sh.rhs_h[0][wv], PV);
        e = tqllm_dumma::mmac_f32_16x16x16_f16(aw, br, e, sim);
        if constexpr (Accurate) {
            const auto brlo = tqllm_dumma::load_b_rowmajor_b16(&rhs_lo_h[0][wv], PV);
            e = tqllm_dumma::mmac_f32_16x16x16_f16(aw, brlo, e, sim);
        }
        for (int i = 0; i < 4; ++i) {
            const int v = wv + cg + 4 * i;
            sh.solved[row][v] = e.x[i];
            const __half hi = __float2half_rn(e.x[i]);
            sh.solved_h[row][v] = hi;
            if constexpr (Accurate)
                solved_lo_h[row][v] = __float2half_rn(e.x[i] - __half2float(hi));
        }
        __syncthreads();
        if (trace && tid == 0) { const auto now = clock64(); phase[3] += now - stamp; stamp = now; }

        auto out = tqllm_dumma::zero_f32x4();
        for (int i = 0; i < 4; ++i)
            out.x[i] = sh.prefix[row < cn ? row : 0] * sh.qs[row][wv + cg + 4 * i];
        const auto am = tqllm_dumma::load_a_rowmajor_b16(matm + off, C);
        const auto be = tqllm_dumma::load_b_rowmajor_b16(&sh.solved_h[0][wv], PV);
        out = tqllm_dumma::mmac_f32_16x16x16_f16(am, be, out, sim);
        if constexpr (Accurate) {
            const auto bel = tqllm_dumma::load_b_rowmajor_b16(&solved_lo_h[0][wv], PV);
            out = tqllm_dumma::mmac_f32_16x16x16_f16(am, bel, out, sim);
        }
        if (row < cn)
            for (int i = 0; i < 4; ++i)
                output[(int64_t(t0 + row) * vh + h) * dv + v0 + wv + cg + 4 * i] =
                    __float2bfloat16_rn(out.x[i]);
        __syncthreads();
        if (trace && tid == 0) { const auto now = clock64(); phase[4] += now - stamp; stamp = now; }

        for (int i = tid; i < C * PV; i += 64) {
            const int t = i / PV, v = i % PV;
            const float delta = t < cn ? sh.solved[t][v] * sh.suffix[t] : 0.f;
            const __half hi = __float2half_rn(delta);
            sh.delta_h[t][v] = hi;
            if constexpr (Accurate)
                delta_lo_h[t][v] = __float2half_rn(delta - __half2float(hi));
        }
        __syncthreads();
        for (int r0 = 0; r0 < K; r0 += 16) {
            tqllm_dumma::AccF32x4 acc;
            for (int i = 0; i < 4; ++i)
                acc.x[i] = sh.prefix[cn - 1] * sh.state[r0 + row][wv + cg + 4 * i];
            const auto ak = tqllm_dumma::load_a_rowmajor_b16(
                kt + (int64_t(khd) * nc + t0 / C) * K * C + r0 * C, C);
            const auto bd = tqllm_dumma::load_b_rowmajor_b16(&sh.delta_h[0][wv], PV);
            acc = tqllm_dumma::mmac_f32_16x16x16_f16(ak, bd, acc, sim);
            if constexpr (Accurate) {
                const auto bdlo = tqllm_dumma::load_b_rowmajor_b16(&delta_lo_h[0][wv], PV);
                acc = tqllm_dumma::mmac_f32_16x16x16_f16(ak, bdlo, acc, sim);
            }
            for (int i = 0; i < 4; ++i) {
                const int v = wv + cg + 4 * i;
                sh.state[r0 + row][v] = acc.x[i];
                const __half hi = __float2half_rn(acc.x[i]);
                sh.state_h[r0 + row][v] = hi;
                if constexpr (Accurate)
                    state_lo_h[r0 + row][v] = __float2half_rn(acc.x[i] - __half2float(hi));
            }
        }
        __syncthreads();
        if (trace && tid == 0) { const auto now = clock64(); phase[5] += now - stamp; stamp = now; }
    }
    for (int i = tid; i < K * PV; i += 64) {
        const int r = i / PV, v = i % PV;
        next[(int64_t(h) * K + r) * dv + v0 + v] = sh.state[r][v];
    }
    if (trace && h == 0 && v0 == 0 && tid == 0)
        for (int i = 0; i < 6; ++i) trace[i] = phase[i];
}
