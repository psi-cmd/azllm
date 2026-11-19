// Drafts + prompt-lookup columns must fit the GDN replay limit for every option combination.
#include "../src/tree_columns.hpp"

#include <cstdio>

int main() {
    int failures = 0, worst = 0;
    const int cap = tqllm::kGdnReplayCols;
    for (int D = 1; D <= 15; ++D)              // --draft-max 1..15
        for (int L = 0; L <= 15; ++L)          // --lookup 0..15
            for (int found = 0; found <= 15; ++found)
                for (int lf = 0; lf <= 1; ++lf) {
                    const int nl = lf ? std::min(found, tqllm::lookup_first_limit(cap)) : std::min(found, tqllm::lookup_limit(L, cap));
                    const int n = tqllm::tree_cols(D, nl, lf, cap) + nl;
                    worst = std::max(worst, n);
                    if (n > cap || n < 1) ++failures;
                }
    // the deferred-commit GDN kernel holds the previous accepted path (<= 8) plus the tree in one 24-token group
    std::printf("widest verify %d columns over all options (limit %d)\n", worst, cap);
    if (failures || cap > 16 || worst > 16) { std::printf("tree_columns: %d failure(s)\n", failures); return 1; }
    std::puts("tree_columns: ok");
    return 0;
}
