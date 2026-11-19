// Bench only: every TQ3 (GGUF 301) / TQ4 (300) matvec variant on the Qwen3.8-27B TQ_L verify shapes, at 8 and 16
// columns, so the rule table (kernels/formats/matvec_rules.hpp) can be checked per GPU. v0 = what the engine runs.
// usage: matvec-sweep [iters=200] [shapes="bits:rows:k,..."] [ncols="8,16"]
#include "formats/weight_format.hpp"
#include "quant_kernels.cuh"
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 200;
    const std::string shapes = argc > 2 ? argv[2]
        : "3:34816:5120,3:5120:17408,3:10240:5120,3:6144:5120,3:5120:4096,3:5120:25600,"
          "4:34816:5120,4:16384:5120,4:14336:5120,4:5120:17408,4:5120:6144";
    const std::string ncs = argc > 3 ? argv[3] : "8,16";
    std::vector<int> ncols;
    for (std::stringstream ss(ncs); ss.good();) { std::string t; std::getline(ss, t, ','); ncols.push_back(std::stoi(t)); }
    std::mt19937 g(7);
    std::stringstream ss(shapes);
    for (std::string item; std::getline(ss, item, ',');) {
        int bits = 0, rows = 0, k = 0;
        if (std::sscanf(item.c_str(), "%d:%d:%d", &bits, &rows, &k) != 3) continue;
        const unsigned ty = bits == 3 ? 301u : 300u;
        const int bb = bits == 3 ? 100 : 132;
        std::vector<char> raw(size_t(rows) * k / 256 * bb);
        for (auto & c : raw) c = char(g() & 255);
        for (size_t b = 0; b < raw.size() / bb; ++b) {
            const __half d = __float2half(0.001f);
            std::memcpy(raw.data() + b * bb, &d, 2), std::memcpy(raw.data() + b * bb + (bits == 3 ? 50 : 66), &d, 2);
        }
        const tqllm::Upload up = [](const void * h, size_t n) { void * p = nullptr; cudaMalloc(&p, n); cudaMemcpy(p, h, n, cudaMemcpyHostToDevice); return static_cast<const void *>(p); };
        // copies cycled per call so the weights stream from DRAM like in the model (the 4090 L2 holds 72 MB)
        const int ncopy = int(std::max<size_t>(2, (size_t(512) << 20) / raw.size() + 1));
        std::vector<tqllm::Mat> ms;
        for (int i = 0; i < ncopy; ++i) ms.push_back(tqllm::weight_format(ty)->load(raw, rows, k, up, nullptr));
        const auto & m = ms[0];
        const int ncmax = 16;
        std::vector<__half> x(size_t(ncmax) * k);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto & v : x) v = __float2half(nd(g) * 0.1f);
        __half * dx = nullptr; cudaMalloc(&dx, x.size() * 2); cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
        __nv_bfloat16 * dy = nullptr; cudaMalloc(&dy, size_t(ncmax) * rows * 2);
        for (int nc : ncols) {
            std::printf("TQ%d %5d x %5d nc=%2d:", bits, rows, k, nc);
            double best = 1e30; int bestv = -1;
            for (int v : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 21, 22}) {
                int it = 0;
                auto call = [&] {
                    const auto & mm = ms[size_t(it++ % ncopy)];
                    const auto * w = static_cast<const uint8_t *>(mm.p);
                    const auto * sc = static_cast<const __half *>(mm.aux);
                    if (bits == 3) tqllm_cuda::tq4b3_matvec_variant(w, sc, dx, dy, rows, k, nc, v, nullptr);
                    else tqllm_cuda::tq4_matvec_variant(w, sc, dx, dy, rows, k, nc, v, nullptr);
                };
                for (int i = 0; i < 20; ++i) call();
                cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
                cudaEventRecord(e0);
                for (int i = 0; i < iters; ++i) call();
                cudaEventRecord(e1); cudaEventSynchronize(e1);
                float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
                const double us = ms * 1000.0 / iters;
                if (cudaGetLastError() != cudaSuccess) { std::printf(" v%d:err", v); continue; }
                std::printf(" v%d:%.1f", v, us);
                if (us < best) best = us, bestv = v;
                cudaEventDestroy(e0); cudaEventDestroy(e1);
            }
            std::printf("  | best v%d %.1f us %.0f GB/s\n", bestv, best, double(raw.size()) / (best * 1e-6) / 1e9);
        }
        cudaFree(dx); cudaFree(dy);
    }
    return 0;
}
