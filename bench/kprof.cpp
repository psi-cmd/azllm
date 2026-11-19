// kprof: LD_PRELOAD CUPTI kernel tracer (works inside CUDA graphs, unlike stageprof which synchronises per stage).
// SIGUSR2 clears the record, SIGUSR1 writes $KPROF_OUT (default /tmp/kprof.txt): per kernel name total GPU time, calls
// and mean; plus the GPU busy time vs the span of the recorded window (idle gaps between kernels).
// Build: see bench/kprof.sh.
#include <cupti.h>
#include <cxxabi.h>
#include <algorithm>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace {
struct Rec { uint64_t s, e; std::string name; };
std::mutex mu;
std::vector<Rec> recs;
volatile sig_atomic_t want_dump = 0, want_reset = 0;

std::string shortname(const char * m) {
    int st = 0;
    if (std::strncmp(m, "__nv_static_", 12) == 0)  // file-static kernels: "__nv_static_<n>__<hash>_<file>__Z..."
        if (const char * z = std::strstr(m + 12, "__Z")) m = z + 1;
    char * d = abi::__cxa_demangle(m, nullptr, nullptr, &st);
    std::string s = st == 0 && d ? d : m;
    std::free(d);
    // keep "ns::func<first template args>" without the parameter list
    for (size_t q; (q = s.find("(anonymous namespace)::")) != std::string::npos;) s.erase(q, 23);
    const size_t p = s.find('(');
    if (p != std::string::npos) s = s.substr(0, p);
    for (const char * a : {"(anonymous namespace)::", "tqllm_cuda::", "__nv_bfloat16", "void "}) {
        const std::string rep = std::strcmp(a, "__nv_bfloat16") == 0 ? "bf16" : "";
        for (size_t q; (q = s.find(a)) != std::string::npos;) s.replace(q, std::strlen(a), rep);
    }
    if (s.size() > 90) s = s.substr(0, 90);
    return s;
}
void CUPTIAPI buf_req(uint8_t ** b, size_t * sz, size_t * maxrec) {
    *sz = 8 << 20;
    *b = static_cast<uint8_t *>(std::aligned_alloc(8, *sz));
    *maxrec = 0;
}
void CUPTIAPI buf_done(CUcontext, uint32_t, uint8_t * b, size_t, size_t valid) {
    CUpti_Activity * r = nullptr;
    std::vector<Rec> got;
    while (cuptiActivityGetNextRecord(b, valid, &r) == CUPTI_SUCCESS) {
        if (r->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || r->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
            auto * k = reinterpret_cast<CUpti_ActivityKernel9 *>(r);
            got.push_back({k->start, k->end, shortname(k->name)});
        }
    }
    std::free(b);
    std::lock_guard<std::mutex> l(mu);
    recs.insert(recs.end(), got.begin(), got.end());
}
void dump() {
    cuptiActivityFlushAll(1);
    std::lock_guard<std::mutex> l(mu);
    const char * out = std::getenv("KPROF_OUT");
    FILE * f = std::fopen(out ? out : "/tmp/kprof.txt", "w");
    if (!f) return;
    std::sort(recs.begin(), recs.end(), [](const Rec & a, const Rec & b) { return a.s < b.s; });
    uint64_t busy = 0, cur_s = 0, cur_e = 0;
    std::map<std::string, std::pair<double, long>> by;
    for (const auto & r : recs) {
        auto & v = by[r.name];
        v.first += double(r.e - r.s), ++v.second;
        if (r.s > cur_e) busy += cur_e - cur_s, cur_s = r.s, cur_e = r.e;
        else cur_e = std::max(cur_e, r.e);
    }
    busy += cur_e - cur_s;
    const double span = recs.empty() ? 0 : double(recs.back().e - recs.front().s);
    double tot = 0;
    for (auto & kv : by) tot += kv.second.first;
    std::fprintf(f, "kernels %zu  span %.3f ms  busy %.3f ms (%.1f%%)  kernel sum %.3f ms\n", recs.size(), span / 1e6,
                 busy / 1e6, span ? 100.0 * busy / span : 0, tot / 1e6);
    std::vector<std::pair<std::string, std::pair<double, long>>> v(by.begin(), by.end());
    std::sort(v.begin(), v.end(), [](auto & a, auto & b) { return a.second.first > b.second.first; });
    for (auto & kv : v)
        std::fprintf(f, "%10.3f ms %5.1f%% %8ld calls %8.2f us  %s\n", kv.second.first / 1e6, 100 * kv.second.first / tot,
                     kv.second.second, kv.second.first / 1e3 / kv.second.second, kv.first.c_str());
    std::fclose(f);
}
void * watcher(void *) {
    for (;;) {
        struct timespec ts{0, 50 * 1000 * 1000};
        nanosleep(&ts, nullptr);
        if (want_reset) { want_reset = 0; cuptiActivityFlushAll(1); std::lock_guard<std::mutex> l(mu); recs.clear(); }
        if (want_dump) { want_dump = 0; dump(); }
    }
    return nullptr;
}
__attribute__((constructor)) void init() {
    cuptiActivityRegisterCallbacks(buf_req, buf_done);
    if (cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) != CUPTI_SUCCESS)
        std::fprintf(stderr, "kprof: CUPTI concurrent-kernel activity unavailable\n");
    std::signal(SIGUSR1, [](int) { want_dump = 1; });
    std::signal(SIGUSR2, [](int) { want_reset = 1; });
    pthread_t t;
    pthread_create(&t, nullptr, watcher, nullptr);
    pthread_detach(t);
    std::fprintf(stderr, "kprof: tracing kernels (SIGUSR2 reset, SIGUSR1 dump)\n");
}
}  // namespace
