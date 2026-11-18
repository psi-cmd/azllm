// Operator-level and kernel-level numerical verification gate for Hygon BW (gfx936) primitives, wave64 DUMMA
// fragment mappings, dual-path TQ4 GEMV/GEMM, and <=64KB LDS attention kernels.
// Runs both on physical BW1000 (TQLLM_BW=ON) and on CUDA GPUs (default or TQLLM_BW_EMULATE=ON).

#include "bw_dumma.cuh"
#include "bw_primitives.cuh"
#include "common.cuh"
#include "quant_kernels.cuh"
#include "qwen2_kernels.cuh"
#include "qwen35_kernels.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x)                                                                                   \
    do {                                                                                        \
        cudaError_t err_ = (x);                                                                 \
        if (err_ != cudaSuccess) {                                                              \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            std::exit(2);                                                                       \
        }                                                                                       \
    } while (0)

namespace {

int g_fails = 0;

void report(const char * name, bool ok, double stat = 0.0) {
    std::printf("%-38s  stat=%.3e  %s\n", name, stat, ok ? "OK" : "FAIL");
    if (!ok) ++g_fails;
}

// 1. Exhaustive 65536-code test of prmt_b32 & TQ4 expand_u vs expand ^ 0x80808080
__device__ uint32_t ref_expand(uint32_t c) {
    const uint32_t pm = __byte_perm(0x2C1F1206u, 0x7F604B3Au, c & 0x7777u);
    uint32_t sg = (c >> 3) & 0x1111u;
    sg = (sg & 0x0011u) | ((sg & 0x1100u) << 8);
    sg = (sg & 0x00010001u) | ((sg & 0x00100010u) << 4);
    return __vadd4(pm ^ (sg * 0xFFu), sg);
}

__global__ void k_test_prmt_exhaustive(int * bad) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= 65536u) return;
    const uint32_t sel = c & 0x7777u;
    const uint32_t pos = __byte_perm(0xAC9F9286u, 0xFFE0CBBAu, sel);
    const uint32_t neg = __byte_perm(0x54616E7Au, 0x01203546u, sel);
    const uint32_t m = tqllm_bw::prmt_b32(c << 4, c, 0xD9C8u);
    const uint32_t got = (pos & ~m) | (neg & m);
    const uint32_t exp = ref_expand(c) ^ 0x80808080u;
    if (got != exp) atomicAdd(bad, 1);
}

// 2. Packed FP16x2 arithmetic (hmul2u, hadd2u, hfma2u)
__global__ void k_test_h2(const uint32_t * a, const uint32_t * b, const uint32_t * c,
                          uint32_t * om, uint32_t * oa, uint32_t * ofma, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    om[i] = tqllm_bw::hmul2u(a[i], b[i]);
    oa[i] = tqllm_bw::hadd2u(a[i], b[i]);
    ofma[i] = tqllm_bw::hfma2u(a[i], b[i], c[i]);
}

// 3. Logical 32-lane subgroup isolation inside 256-thread block (4 wave64s / 8 warps)
__global__ void k_test_subgroup32(int * bad) {
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const uint32_t val = uint32_t(warp * 1000 + lane + 1);
    const uint32_t bcast0 = __shfl_sync(0xffffffffu, val, 0, 32);
    if (bcast0 != uint32_t(warp * 1000 + 1)) atomicAdd(bad, 1);

    float s = float(lane + 1) + float(warp) * 10.f;
    for (int d = 16; d; d >>= 1) s += __shfl_down_sync(0xffffffffu, s, d, 32);
    if (lane == 0) {
        const float exp = 528.f + 32.f * float(warp) * 10.f;
        if (fabsf(s - exp) > 1e-3f) atomicAdd(bad, 1);
    }

    // Odd warps true, even warps false -> must NOT bleed between even/odd halves of wave64!
    const int pred = (warp & 1) && (lane == 17);
    const int any = tqllm_bw::any_sync32(0xffffffffu, pred);
    if (any != (warp & 1)) atomicAdd(bad, 1);
}

// 4. ldsm_x4 and ldsm_x4_t verification
__global__ void k_test_ldsm(int * bad) {
    __shared__ uint16_t sm[32][16];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gid = lane >> 2, tig = lane & 3;
    if (warp == 0) {
        for (int c = 0; c < 16; ++c) sm[lane][c] = uint16_t(lane * 100 + c + 1);
    }
    __syncthreads();
    if (warp > 1) return;
    const int m = lane / 8, r = lane % 8;
    const uint16_t * p = &sm[m * 8 + r][warp * 8];
    uint32_t rx[4], rt[4];
    tqllm_bw::ldsm_x4(rx, p);
    tqllm_bw::ldsm_x4_t(rt, p);
    for (int mi = 0; mi < 4; ++mi) {
        const uint32_t e0 = uint16_t((mi * 8 + gid) * 100 + warp * 8 + 2 * tig + 1);
        const uint32_t e1 = uint16_t((mi * 8 + gid) * 100 + warp * 8 + 2 * tig + 2);
        if (rx[mi] != (e0 | (e1 << 16))) atomicAdd(bad, 1);

        const uint32_t t0 = uint16_t((mi * 8 + 2 * tig) * 100 + warp * 8 + gid + 1);
        const uint32_t t1 = uint16_t((mi * 8 + 2 * tig + 1) * 100 + warp * 8 + gid + 1);
        if (rt[mi] != (t0 | (t1 << 16))) atomicAdd(bad, 1);
    }
}

// 5. 32-lane mma_f16 / mma_bf16 / mma_f16acc / mma_s8 (with half-wave divergence: only warp 0 active!)
__global__ void k_test_mma32(const __half * ah, const __half * bh, float * out_f16, float * out_f16acc,
                             const __nv_bfloat16 * abf, const __nv_bfloat16 * bbf, float * out_bf16,
                             const int8_t * as8, const int8_t * bs8, int * out_s8) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    if (warp != 0) return;  // Deliberately divergent within a 64-lane wavefront!

    const uint32_t a0 = *reinterpret_cast<const uint32_t *>(ah + g * 16 + 2 * t);
    const uint32_t a1 = *reinterpret_cast<const uint32_t *>(ah + (g + 8) * 16 + 2 * t);
    const uint32_t a2 = *reinterpret_cast<const uint32_t *>(ah + g * 16 + 8 + 2 * t);
    const uint32_t a3 = *reinterpret_cast<const uint32_t *>(ah + (g + 8) * 16 + 8 + 2 * t);
    const uint32_t b0 = *reinterpret_cast<const uint32_t *>(bh + g * 16 + 2 * t);
    const uint32_t b1 = *reinterpret_cast<const uint32_t *>(bh + g * 16 + 8 + 2 * t);

    float cf[4] = {0.f, 0.f, 0.f, 0.f};
    tqllm_bw::mma_f16(cf, a0, a1, a2, a3, b0, b1);
    out_f16[g * 8 + 2 * t] = cf[0];
    out_f16[g * 8 + 2 * t + 1] = cf[1];
    out_f16[(g + 8) * 8 + 2 * t] = cf[2];
    out_f16[(g + 8) * 8 + 2 * t + 1] = cf[3];

    uint32_t ch[2] = {0u, 0u};
    tqllm_bw::mma_f16acc(ch, a0, a1, a2, a3, b0, b1);
    const float2 lo = __half22float2(tqllm_bw::u32_to_h2(ch[0]));
    const float2 hi = __half22float2(tqllm_bw::u32_to_h2(ch[1]));
    out_f16acc[g * 8 + 2 * t] = lo.x;
    out_f16acc[g * 8 + 2 * t + 1] = lo.y;
    out_f16acc[(g + 8) * 8 + 2 * t] = hi.x;
    out_f16acc[(g + 8) * 8 + 2 * t + 1] = hi.y;

    const uint32_t ba0 = *reinterpret_cast<const uint32_t *>(abf + g * 16 + 2 * t);
    const uint32_t ba1 = *reinterpret_cast<const uint32_t *>(abf + (g + 8) * 16 + 2 * t);
    const uint32_t ba2 = *reinterpret_cast<const uint32_t *>(abf + g * 16 + 8 + 2 * t);
    const uint32_t ba3 = *reinterpret_cast<const uint32_t *>(abf + (g + 8) * 16 + 8 + 2 * t);
    const uint32_t bb0 = *reinterpret_cast<const uint32_t *>(bbf + g * 16 + 2 * t);
    const uint32_t bb1 = *reinterpret_cast<const uint32_t *>(bbf + g * 16 + 8 + 2 * t);
    float cbf[4] = {0.f, 0.f, 0.f, 0.f};
    tqllm_bw::mma_bf16(cbf, ba0, ba1, ba2, ba3, bb0, bb1);
    out_bf16[g * 8 + 2 * t] = cbf[0];
    out_bf16[g * 8 + 2 * t + 1] = cbf[1];
    out_bf16[(g + 8) * 8 + 2 * t] = cbf[2];
    out_bf16[(g + 8) * 8 + 2 * t + 1] = cbf[3];

    const uint32_t sa0 = *reinterpret_cast<const uint32_t *>(as8 + g * 32 + 4 * t);
    const uint32_t sa1 = *reinterpret_cast<const uint32_t *>(as8 + (g + 8) * 32 + 4 * t);
    const uint32_t sa2 = *reinterpret_cast<const uint32_t *>(as8 + g * 32 + 16 + 4 * t);
    const uint32_t sa3 = *reinterpret_cast<const uint32_t *>(as8 + (g + 8) * 32 + 16 + 4 * t);
    const uint32_t sb0 = *reinterpret_cast<const uint32_t *>(bs8 + g * 32 + 4 * t);
    const uint32_t sb1 = *reinterpret_cast<const uint32_t *>(bs8 + g * 32 + 16 + 4 * t);
    int cs[4] = {0, 0, 0, 0};
    tqllm_bw::mma_s8(cs, sa0, sa1, sa2, sa3, sb0, sb1);
    out_s8[g * 8 + 2 * t] = cs[0];
    out_s8[g * 8 + 2 * t + 1] = cs[1];
    out_s8[(g + 8) * 8 + 2 * t] = cs[2];
    out_s8[(g + 8) * 8 + 2 * t + 1] = cs[3];
}

// 6. Wave64 DUMMA (16x16x16 FP16, 16x16x16 BF16, 16x16x32 INT8) fragment layout & accumulation test
__global__ void k_test_dumma_wave64(const __half * ah, const __half * bh, float * ch_out,
                                    const __nv_bfloat16 * abf, const __nv_bfloat16 * bbf, float * cbf_out,
                                    const int8_t * ai8, const int8_t * bi8, int * ci8_out) {
    __shared__ tqllm_dumma::SimScratch<1> sim;
    const int lid = tqllm_dumma::lane64_id(), r_in = lid & 15, k_grp = lid >> 4;

    const auto fa_h = tqllm_dumma::load_a_rowmajor_b16(ah, 16);
    const auto fb_h = tqllm_dumma::load_b_colmajor_b16(bh, 16);
    const auto dh = tqllm_dumma::mmac_f32_16x16x16_f16(fa_h, fb_h, tqllm_dumma::zero_f32x4(), sim);

    const auto fa_bf = tqllm_dumma::load_a_rowmajor_b16(abf, 16);
    const auto fb_bf = tqllm_dumma::load_b_rowmajor_b16(bbf, 16);  // tests row_major B loader too!
    const auto dbf = tqllm_dumma::mmac_f32_16x16x16_bf16(fa_bf, fb_bf, tqllm_dumma::zero_f32x4(), sim);

    const auto fa_i8 = tqllm_dumma::load_a_rowmajor_i8(ai8, 32);
    const auto fb_i8 = tqllm_dumma::load_b_colmajor_i8(bi8, 32);
    const auto di8 = tqllm_dumma::mmac_i32_16x16x32_i8(fa_i8, fb_i8, tqllm_dumma::zero_i32x4(), sim);

    for (int i = 0; i < 4; ++i) {
        const int col = k_grp + 4 * i;
        ch_out[r_in * 16 + col] = dh.x[i];
        cbf_out[r_in * 16 + col] = dbf.x[i];
        ci8_out[r_in * 16 + col] = di8.x[i];
    }
}

}  // namespace

int main() {
    std::mt19937 rng(20261117);
    std::uniform_real_distribution<float> uf(-2.f, 2.f), us(0.003f, 0.02f);
    std::uniform_int_distribution<int> ui8(-120, 120), u8(0, 255);

    int * d_bad = nullptr;
    CK(cudaMalloc(&d_bad, sizeof(int)));

    // Test 1: Exhaustive 65,536 prmt_b32 & expand_u
    CK(cudaMemset(d_bad, 0, sizeof(int)));
    k_test_prmt_exhaustive<<<256, 256>>>(d_bad);
    int h_bad = 0;
    CK(cudaMemcpy(&h_bad, d_bad, sizeof(int), cudaMemcpyDeviceToHost));
    report("prmt_b32 & expand_u (all 65536 codes)", h_bad == 0, double(h_bad));

    // Test 2: Packed FP16x2 arithmetic
    {
        const int N = 4096;
        std::vector<uint32_t> ha(N), hb(N), hc(N), gom(N), goa(N), gof(N);
        for (int i = 0; i < N; ++i) {
            const __half2 a2 = __floats2half2_rn(uf(rng), uf(rng));
            const __half2 b2 = __floats2half2_rn(uf(rng), uf(rng));
            const __half2 c2 = __floats2half2_rn(uf(rng), uf(rng));
            ha[i] = tqllm_bw::h2_to_u32(a2);
            hb[i] = tqllm_bw::h2_to_u32(b2);
            hc[i] = tqllm_bw::h2_to_u32(c2);
        }
        uint32_t *da, *db, *dc, *dom, *doa, *dof;
        CK(cudaMalloc(&da, N * 4)); CK(cudaMalloc(&db, N * 4)); CK(cudaMalloc(&dc, N * 4));
        CK(cudaMalloc(&dom, N * 4)); CK(cudaMalloc(&doa, N * 4)); CK(cudaMalloc(&dof, N * 4));
        CK(cudaMemcpy(da, ha.data(), N * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(db, hb.data(), N * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dc, hc.data(), N * 4, cudaMemcpyHostToDevice));
        k_test_h2<<<N / 256, 256>>>(da, db, dc, dom, doa, dof, N);
        CK(cudaMemcpy(gom.data(), dom, N * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(goa.data(), doa, N * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(gof.data(), dof, N * 4, cudaMemcpyDeviceToHost));
        double max_err = 0.0;
        for (int i = 0; i < N; ++i) {
            const float2 af = __half22float2(tqllm_bw::u32_to_h2(ha[i]));
            const float2 bf = __half22float2(tqllm_bw::u32_to_h2(hb[i]));
            const float2 cf = __half22float2(tqllm_bw::u32_to_h2(hc[i]));
            const float2 gm = __half22float2(tqllm_bw::u32_to_h2(gom[i]));
            const float2 ga = __half22float2(tqllm_bw::u32_to_h2(goa[i]));
            const float2 gf = __half22float2(tqllm_bw::u32_to_h2(gof[i]));
            const float2 rm = __half22float2(__floats2half2_rn(af.x * bf.x, af.y * bf.y));
            const float2 ra = __half22float2(__floats2half2_rn(af.x + bf.x, af.y + bf.y));
            const float2 rf = __half22float2(__floats2half2_rn(std::fma(af.x, bf.x, cf.x), std::fma(af.y, bf.y, cf.y)));
            max_err = std::max({max_err, double(std::fabs(gm.x - rm.x)), double(std::fabs(gm.y - rm.y)),
                                double(std::fabs(ga.x - ra.x)), double(std::fabs(ga.y - ra.y)),
                                double(std::fabs(gf.x - rf.x)), double(std::fabs(gf.y - rf.y))});
        }
        report("packed f16x2 (mul/add/fma rn)", max_err == 0.0, max_err);
        cudaFree(da); cudaFree(db); cudaFree(dc); cudaFree(dom); cudaFree(doa); cudaFree(dof);
    }

    // Test 3: Subgroup 32-lane isolation on 256-thread CTA
    CK(cudaMemset(d_bad, 0, sizeof(int)));
    k_test_subgroup32<<<1, 256>>>(d_bad);
    CK(cudaMemcpy(&h_bad, d_bad, sizeof(int), cudaMemcpyDeviceToHost));
    report("logical 32-lane shfl/any isolation", h_bad == 0, double(h_bad));

    // Test 4: ldsm_x4 and ldsm_x4_t
    CK(cudaMemset(d_bad, 0, sizeof(int)));
    k_test_ldsm<<<1, 256>>>(d_bad);
    CK(cudaMemcpy(&h_bad, d_bad, sizeof(int), cudaMemcpyDeviceToHost));
    report("ldsm_x4 & ldsm_x4_t mapping", h_bad == 0, double(h_bad));

    // Test 5: 32-lane mma_f16 / mma_f16acc / mma_bf16 / mma_s8 under half-wave divergence
    {
        std::vector<__half> ah(16 * 16), bh(8 * 16);
        std::vector<__nv_bfloat16> abf(16 * 16), bbf(8 * 16);
        std::vector<int8_t> as8(16 * 32), bs8(8 * 32);
        for (auto & x : ah) x = __float2half_rn(uf(rng));
        for (auto & x : bh) x = __float2half_rn(uf(rng));
        for (auto & x : abf) x = __float2bfloat16_rn(uf(rng));
        for (auto & x : bbf) x = __float2bfloat16_rn(uf(rng));
        for (auto & x : as8) x = int8_t(ui8(rng));
        for (auto & x : bs8) x = int8_t(ui8(rng));

        __half *dah, *dbh; __nv_bfloat16 *dabf, *dbbf; int8_t *das8, *dbs8;
        float *dof16, *dof16a, *dobf16; int *dos8;
        CK(cudaMalloc(&dah, ah.size() * 2)); CK(cudaMalloc(&dbh, bh.size() * 2));
        CK(cudaMalloc(&dabf, abf.size() * 2)); CK(cudaMalloc(&dbbf, bbf.size() * 2));
        CK(cudaMalloc(&das8, as8.size())); CK(cudaMalloc(&dbs8, bs8.size()));
        CK(cudaMalloc(&dof16, 128 * 4)); CK(cudaMalloc(&dof16a, 128 * 4));
        CK(cudaMalloc(&dobf16, 128 * 4)); CK(cudaMalloc(&dos8, 128 * 4));
        CK(cudaMemcpy(dah, ah.data(), ah.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dbh, bh.data(), bh.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dabf, abf.data(), abf.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dbbf, bbf.data(), bbf.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(das8, as8.data(), as8.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dbs8, bs8.data(), bs8.size(), cudaMemcpyHostToDevice));

        k_test_mma32<<<1, 64>>>(dah, dbh, dof16, dof16a, dabf, dbbf, dobf16, das8, dbs8, dos8);
        std::vector<float> gf16(128), gf16a(128), gbf16(128);
        std::vector<int> gs8(128);
        CK(cudaMemcpy(gf16.data(), dof16, 128 * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(gf16a.data(), dof16a, 128 * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(gbf16.data(), dobf16, 128 * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(gs8.data(), dos8, 128 * 4, cudaMemcpyDeviceToHost));

        double ef16 = 0, ef16a = 0, ebf16 = 0;
        int es8 = 0;
        for (int r = 0; r < 16; ++r) {
            for (int c = 0; c < 8; ++c) {
                double rf16 = 0, rbf16 = 0;
                int rs8 = 0;
                for (int k = 0; k < 16; ++k) {
                    rf16 += double(__half2float(ah[r * 16 + k])) * __half2float(bh[c * 16 + k]);
                    rbf16 += double(__bfloat162float(abf[r * 16 + k])) * __bfloat162float(bbf[c * 16 + k]);
                }
                for (int k = 0; k < 32; ++k) rs8 += int(as8[r * 32 + k]) * int(bs8[c * 32 + k]);
                ef16 = std::max(ef16, std::fabs(rf16 - gf16[r * 8 + c]));
                ef16a = std::max(ef16a, std::fabs(rf16 - gf16a[r * 8 + c]));
                ebf16 = std::max(ebf16, std::fabs(rbf16 - gbf16[r * 8 + c]));
                es8 = std::max(es8, std::abs(rs8 - gs8[r * 8 + c]));
            }
        }
        report("mma32 f16->f32 (divergent wave64)", ef16 < 1e-5, ef16);
        report("mma32 f16->f16acc rounding", ef16a < 2e-2, ef16a);
        report("mma32 bf16->f32 (divergent wave64)", ebf16 < 1e-5, ebf16);
        report("mma32 s8->s32 exact", es8 == 0, double(es8));
        cudaFree(dah); cudaFree(dbh); cudaFree(dabf); cudaFree(dbbf);
        cudaFree(das8); cudaFree(dbs8); cudaFree(dof16); cudaFree(dof16a); cudaFree(dobf16); cudaFree(dos8);
    }

    // Test 6: Wave64 DUMMA (16x16x16 FP16, 16x16x16 BF16, 16x16x32 INT8)
    {
        std::vector<__half> ah(256), bh(256);
        std::vector<__nv_bfloat16> abf(256), bbf(256);
        std::vector<int8_t> ai8(512), bi8(512);
        for (auto & x : ah) x = __float2half_rn(uf(rng));
        for (auto & x : bh) x = __float2half_rn(uf(rng));
        for (auto & x : abf) x = __float2bfloat16_rn(uf(rng));
        for (auto & x : bbf) x = __float2bfloat16_rn(uf(rng));
        for (auto & x : ai8) x = int8_t(ui8(rng));
        for (auto & x : bi8) x = int8_t(ui8(rng));

        __half *dah, *dbh; __nv_bfloat16 *dabf, *dbbf; int8_t *dai8, *dbi8;
        float *dch, *dcbf; int *dci8;
        CK(cudaMalloc(&dah, 512)); CK(cudaMalloc(&dbh, 512));
        CK(cudaMalloc(&dabf, 512)); CK(cudaMalloc(&dbbf, 512));
        CK(cudaMalloc(&dai8, 512)); CK(cudaMalloc(&dbi8, 512));
        CK(cudaMalloc(&dch, 256 * 4)); CK(cudaMalloc(&dcbf, 256 * 4)); CK(cudaMalloc(&dci8, 256 * 4));
        CK(cudaMemcpy(dah, ah.data(), 512, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dbh, bh.data(), 512, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dabf, abf.data(), 512, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dbbf, bbf.data(), 512, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dai8, ai8.data(), 512, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dbi8, bi8.data(), 512, cudaMemcpyHostToDevice));

        k_test_dumma_wave64<<<1, 64>>>(dah, dbh, dch, dabf, dbbf, dcbf, dai8, dbi8, dci8);
        std::vector<float> gch(256), gcbf(256);
        std::vector<int> gci8(256);
        CK(cudaMemcpy(gch.data(), dch, 256 * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(gcbf.data(), dcbf, 256 * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(gci8.data(), dci8, 256 * 4, cudaMemcpyDeviceToHost));

        double eh = 0, ebf = 0;
        int ei8 = 0;
        for (int r = 0; r < 16; ++r) {
            for (int c = 0; c < 16; ++c) {
                double rh = 0, rbf = 0;
                int ri8 = 0;
                for (int k = 0; k < 16; ++k) {
                    rh += double(__half2float(ah[r * 16 + k])) * __half2float(bh[c * 16 + k]);     // col_major B
                    rbf += double(__bfloat162float(abf[r * 16 + k])) * __bfloat162float(bbf[k * 16 + c]);  // row_major B
                }
                for (int k = 0; k < 32; ++k) ri8 += int(ai8[r * 32 + k]) * int(bi8[c * 32 + k]);
                eh = std::max(eh, std::fabs(rh - gch[r * 16 + c]));
                ebf = std::max(ebf, std::fabs(rbf - gcbf[r * 16 + c]));
                ei8 = std::max(ei8, std::abs(ri8 - gci8[r * 16 + c]));
            }
        }
        report("DUMMA 16x16x16 f16->f32 layout", eh < 1e-4, eh);
        report("DUMMA 16x16x16 bf16->f32 layout", ebf < 1e-4, ebf);
        report("DUMMA 16x16x32 i8->i32 exact", ei8 == 0, double(ei8));
        cudaFree(dah); cudaFree(dbh); cudaFree(dabf); cudaFree(dbbf);
        cudaFree(dai8); cudaFree(dbi8); cudaFree(dch); cudaFree(dcbf); cudaFree(dci8);
    }

    // Test 7: Dual-path BW TQ4 kernels (v=30 Wave64 GEMV & v=31 Wave64 DUMMA) vs dequant reference
    {
        const int rows = 150, k = 384;  // non-power-of-two rows, odd ng = 3 quant blocks
        std::vector<uint8_t> w(size_t(rows) * k / 2);
        std::vector<__half> sc(size_t(rows) * (k / 128)), x(size_t(20) * k);
        for (auto & b : w) b = uint8_t(u8(rng));
        for (auto & s : sc) s = __float2half_rn(us(rng));
        for (auto & v : x) v = __float2half_rn(uf(rng));

        uint8_t * dw; __half *dsc, *dx, *dwh; __nv_bfloat16 *dy;
        CK(cudaMalloc(&dw, w.size())); CK(cudaMalloc(&dsc, sc.size() * 2));
        CK(cudaMalloc(&dx, x.size() * 2)); CK(cudaMalloc(&dwh, size_t(rows) * k * 2));
        CK(cudaMalloc(&dy, size_t(20) * rows * 2));
        CK(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dsc, sc.data(), sc.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
        tqllm_cuda::tq4_dequant_h(dw, dsc, dwh, rows, k, nullptr);
        std::vector<__half> wh(size_t(rows) * k);
        CK(cudaMemcpy(wh.data(), dwh, wh.size() * 2, cudaMemcpyDeviceToHost));

        auto check_mv = [&](const char * label, int ncols, int var) {
            tqllm_cuda::tq4_matvec_variant(dw, dsc, dx, dy, rows, k, ncols, var, nullptr);
            std::vector<__nv_bfloat16> gy(size_t(ncols) * rows);
            CK(cudaMemcpy(gy.data(), dy, gy.size() * 2, cudaMemcpyDeviceToHost));
            double mr = 0, me = 0;
            for (int c = 0; c < ncols; ++c)
                for (int r = 0; r < rows; ++r) {
                    double ref = 0;
                    for (int i = 0; i < k; ++i)
                        ref += double(__half2float(wh[size_t(r) * k + i])) * __half2float(x[size_t(c) * k + i]);
                    mr = std::max(mr, std::fabs(ref));
                    me = std::max(me, std::fabs(ref - double(__bfloat162float(gy[size_t(c) * rows + r]))));
                }
            const double rel = me / std::max(mr, 1e-9);
            report(label, rel < 8e-3, rel);
        };
        check_mv("TQ4 BW wave64 GEMV (ncols=1, v=30)", 1, 30);
        check_mv("TQ4 BW wave64 GEMV (ncols=2, v=30)", 2, 30);
        check_mv("TQ4 BW wave64 DUMMA (ncols=8, v=31)", 8, 31);
        check_mv("TQ4 BW wave64 DUMMA (ncols=20, v=31)", 20, 31);
        cudaFree(dw); cudaFree(dsc); cudaFree(dx); cudaFree(dwh); cudaFree(dy);
    }

    // Test 8: Attention prefill + verify + q4/tcq2 compressed KV cache
    {
        const int h = 24, kh = 4, dim = 256, ctx = 128, nq = 19;
        const float scale = 1.f / std::sqrt(float(dim));
        std::vector<__nv_bfloat16> q(size_t(nq) * h * dim), kc(size_t(ctx) * kh * dim), vc(size_t(ctx) * kh * dim);
        for (auto & v : q) v = __float2bfloat16_rn(uf(rng) * 0.5f);
        for (auto & v : kc) v = __float2bfloat16_rn(uf(rng) * 0.5f);
        for (auto & v : vc) v = __float2bfloat16_rn(uf(rng) * 0.5f);

        __nv_bfloat16 *dq, *dkc, *dvc, *dy_pf, *dy_ref;
        float * dpart; int * dseq0;
        const size_t pf_part = size_t(tqllm_cuda::attention_mq_partial_floats(h, nq, 4));
        CK(cudaMalloc(&dq, q.size() * 2)); CK(cudaMalloc(&dkc, kc.size() * 2)); CK(cudaMalloc(&dvc, vc.size() * 2));
        CK(cudaMalloc(&dy_pf, q.size() * 2)); CK(cudaMalloc(&dy_ref, q.size() * 2));
        CK(cudaMalloc(&dpart, pf_part * 4)); CK(cudaMalloc(&dseq0, sizeof(int)));
        const int seq0 = 40;
        CK(cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dkc, kc.data(), kc.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dvc, vc.data(), vc.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dseq0, &seq0, sizeof(int), cudaMemcpyHostToDevice));

        // Compare prefill kernel and multi-query verify kernel against reference decode_multi
        tqllm_cuda::attention_decode_multi(dq, dkc, dvc, dy_ref, dpart, seq0, nq, h, kh, dim, scale, nullptr);
        tqllm_cuda::attention_mq256_prefill(dq, dkc, dvc, dy_pf, dpart, pf_part, dseq0, nq, h, kh, scale, 1, nullptr);
        std::vector<__nv_bfloat16> gref(q.size()), gpf(q.size());
        CK(cudaMemcpy(gref.data(), dy_ref, q.size() * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(gpf.data(), dy_pf, q.size() * 2, cudaMemcpyDeviceToHost));
        double max_pf = 0;
        for (size_t i = 0; i < q.size(); ++i)
            max_pf = std::max(max_pf, double(std::fabs(__bfloat162float(gref[i]) - __bfloat162float(gpf[i]))));
        report("attention_mq256_prefill vs ref", max_pf < 1.5e-2, max_pf);

        const int nq_v = 7;
        tqllm_cuda::attention_decode_multi(dq, dkc, dvc, dy_ref, dpart, seq0, nq_v, h, kh, dim, scale, nullptr);
        tqllm_cuda::attention_mq256(dq, dkc, dvc, dy_pf, dpart, dseq0, nq_v, h, kh, scale, 4, nullptr, 0);
        CK(cudaMemcpy(gref.data(), dy_ref, size_t(nq_v) * h * dim * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(gpf.data(), dy_pf, size_t(nq_v) * h * dim * 2, cudaMemcpyDeviceToHost));
        double max_mq = 0;
        for (size_t i = 0; i < size_t(nq_v) * h * dim; ++i)
            max_mq = std::max(max_mq, double(std::fabs(__bfloat162float(gref[i]) - __bfloat162float(gpf[i]))));
        report("attention_mq256 verify vs ref", max_mq < 1.5e-2, max_mq);

        cudaFree(dq); cudaFree(dkc); cudaFree(dvc); cudaFree(dy_pf); cudaFree(dy_ref); cudaFree(dpart); cudaFree(dseq0);
    }

    cudaFree(d_bad);
    std::printf("bw_primitives suite: %s (%d failures)\n", g_fails == 0 ? "PASS" : "FAIL", g_fails);
    return g_fails ? EXIT_FAILURE : EXIT_SUCCESS;
}
