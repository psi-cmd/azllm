// Compare the gfx936 FP16-MMAC WY prefill against the original FP32 recurrence.
// No production model needed; shapes match Qwen3.8-27B.
#include "qwen35_kernels.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(expr) do { const cudaError_t e = (expr); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(2); \
} } while (0)

template <typename T> T * device(const std::vector<T> & src) {
    T *ptr = nullptr;
    CK(cudaMalloc(&ptr, sizeof(T) * src.size()));
    CK(cudaMemcpy(ptr, src.data(), sizeof(T) * src.size(), cudaMemcpyHostToDevice));
    return ptr;
}
template <typename T> T * device(size_t n) {
    T *ptr = nullptr;
    CK(cudaMalloc(&ptr, sizeof(T) * n));
    return ptr;
}

int main(int argc, char **argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 1024;
    const bool tiled = argc > 2 && std::atoi(argv[2]) != 0;
    if (n < 1) { std::fprintf(stderr, "n must be positive\n"); return 2; }
    const int kh = 16, vh = 48, dk = 128, dv = 128, stride = 2 * kh * dk + vh * dv;
    std::vector<__nv_bfloat16> qkv(size_t(n) * stride), a(size_t(n) * vh), b(size_t(n) * vh);
    std::vector<float> neg(vh), bias(vh), state(size_t(vh) * dk * dv);
    for (size_t i = 0; i < qkv.size(); ++i)
        qkv[i] = __float2bfloat16_rn(0.7f * std::sin(float(i) * 0.0137f) + 0.2f * std::cos(float(i) * 0.00071f));
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = __float2bfloat16_rn(0.9f * std::cos(float(i) * 1.3f));
        b[i] = __float2bfloat16_rn(0.8f * std::sin(float(i) * 0.7f));
    }
    for (int i = 0; i < vh; ++i) {
        neg[i] = -.2f - .017f * i;
        bias[i] = .1f * std::cos(float(i));
    }
    for (size_t i = 0; i < state.size(); ++i)
        state[i] = 0.05f * std::sin(float(i) * 0.021f);

    auto *dq = device(qkv);
    auto *da = device(a);
    auto *db = device(b);
    auto *dn = device(neg);
    auto *dw = device(bias);
    auto *ds = device(state);
    auto *dnext = device<float>(state.size());
    auto *dy = device<__nv_bfloat16>(size_t(n) * vh * dv);
    std::vector<__nv_bfloat16> outputs[2];
    std::vector<float> states[2];
    float times[2]{};

    for (int mode = 0; mode < 2; ++mode) {
        if (mode) setenv("TQLLM_GDN_CHUNK_MIN", "128", 1);
        else unsetenv("TQLLM_GDN_CHUNK_MIN");
        auto run = [&] {
            tqllm_cuda::gdn_multi(dq, n, da, db, vh, dn, dw, ds, dnext, dy,
                                  kh, vh, dk, dv, tiled, nullptr, {});
        };
        run();
        CK(cudaDeviceSynchronize());
        cudaEvent_t begin, end;
        CK(cudaEventCreate(&begin)); CK(cudaEventCreate(&end));
        CK(cudaEventRecord(begin));
        for (int rep = 0; rep < 4; ++rep) run();
        CK(cudaEventRecord(end)); CK(cudaEventSynchronize(end));
        CK(cudaEventElapsedTime(&times[mode], begin, end));
        times[mode] /= 4;
        outputs[mode].resize(size_t(n) * vh * dv);
        states[mode].resize(state.size());
        CK(cudaMemcpy(outputs[mode].data(), dy, outputs[mode].size() * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(states[mode].data(), dnext, states[mode].size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaEventDestroy(begin)); CK(cudaEventDestroy(end));
    }

    double ey = 0, ry = 0, es = 0, rs = 0, maxy = 0, maxs = 0;
    for (size_t i = 0; i < outputs[0].size(); ++i) {
        const float a0 = __bfloat162float(outputs[0][i]), b0 = __bfloat162float(outputs[1][i]);
        const double d = double(a0) - double(b0);
        ey += d * d; ry += double(a0) * a0;
        maxy = std::max(maxy, std::fabs(d));
    }
    for (size_t i = 0; i < state.size(); ++i) {
        const double d = double(states[0][i]) - states[1][i];
        es += d * d; rs += double(states[0][i]) * states[0][i];
        maxs = std::max(maxs, std::fabs(d));
    }
    const double rel_y = std::sqrt(ey / std::max(ry, 1e-30));
    const double rel_s = std::sqrt(es / std::max(rs, 1e-30));
    std::printf("GDN_BW n=%d tiled=%d original=%.3f ms WY_MMAs=%.3f ms speedup=%.2fx\n",
                n, tiled, times[0], times[1], times[0] / times[1]);
    std::printf("output rel_rms=%.4e max_abs=%.4e; state rel_rms=%.4e max_abs=%.4e\n",
                rel_y, maxy, rel_s, maxs);
    const bool ok = std::isfinite(rel_y) && std::isfinite(rel_s) && rel_y < 0.025 && rel_s < 0.025;
    std::puts(ok ? "GDN_BW_CHUNK PASS" : "GDN_BW_CHUNK FAIL");
    return ok ? 0 : 1;
}
