// Sampled tree verification is exact: per node, the emitted token (an accepted child, else a draw from the residual)
// has the target distribution p, and a child is accepted with probability p's mass on the children. Monte Carlo over
// random distributions and child lists (children outside the support, repeated tokens, all of the support).
#include "../src/tree_sample.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

int main() {
    std::mt19937_64 g(17);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    int failures = 0;
    const int N = 400000;
    for (int trial = 0; trial < 24; ++trial) {
        const int np = 1 + int(g() % 16);
        std::vector<uint32_t> ids(np);
        std::vector<double> p(np);
        double z = 0;
        for (int i = 0; i < np; ++i) ids[i] = 100 + uint32_t(i), z += p[i] = std::pow(u01(g), 3.0);
        for (auto & v : p) v /= z;
        std::vector<uint32_t> kids;
        const int nc = trial % 6 == 5 ? np : 1 + int(g() % 4);
        for (int c = 0; c < nc; ++c)  // trial % 6 == 5: every token of the support; otherwise any, some outside it
            kids.push_back(trial % 6 == 5 ? ids[c] : (g() % 5 == 0 ? 999u : ids[g() % np]));
        std::vector<double> freq(np + 1, 0.0);
        double accepted = 0;
        for (int it = 0; it < N; ++it) {
            std::vector<double> r = p;
            const int a = tqllm::tree_sample_step(ids.data(), r.data(), np, kids.data(), int(kids.size()), [&] { return u01(g); });
            int tok = -1;
            if (a >= 0) {
                accepted += 1;
                tok = int(std::find(ids.begin(), ids.end(), kids[size_t(a)]) - ids.begin());
            } else {
                double tot = 0;
                for (double v : r) tot += v;
                double x = u01(g) * tot;
                for (int i = 0; i < np; ++i)
                    if ((x -= r[i]) <= 0 && r[i] > 0) { tok = i; break; }
                if (tok < 0) tok = np;  // empty residual: impossible unless the children cover all mass
            }
            freq[size_t(tok)] += 1;
        }
        double worst = 0, mass = 0;
        std::vector<bool> seen(np, false);
        for (uint32_t k : kids)
            for (int i = 0; i < np; ++i)
                if (ids[i] == k && !seen[size_t(i)]) seen[size_t(i)] = true, mass += p[i];
        for (int i = 0; i < np; ++i) {  // |freq - p| in standard errors
            const double se = std::sqrt(std::max(p[i] * (1 - p[i]), 1e-12) / N);
            worst = std::max(worst, std::fabs(freq[size_t(i)] / N - p[i]) / se);
        }
        const double acc_se = std::sqrt(std::max(mass * (1 - mass), 1e-12) / N);
        const double acc_dev = std::fabs(accepted / N - mass) / acc_se;
        const bool ok = worst < 5.0 && acc_dev < 5.0 && freq[size_t(np)] == 0;
        if (!ok) {
            ++failures;
            std::printf("trial %d: np %d children %zu  worst token deviation %.1f se, accept %.4f vs mass %.4f (%.1f se)\n",
                        trial, np, kids.size(), worst, accepted / N, mass, acc_dev);
        }
    }
    std::printf("tree_sample: %s (%d of 24 trials failed)\n", failures ? "FAIL" : "ok", failures);
    return failures ? 1 : 0;
}
