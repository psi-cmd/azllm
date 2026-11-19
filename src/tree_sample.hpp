#pragma once
// One node of sampled tree verification (exact): the target distribution p over ids[0..np) (unnormalised, >= 0) is
// sampled without replacement over the node's child tokens in order. Child c is accepted with probability
// r(child[c]) / sum(r), where r is p with the earlier rejected children removed; returns its index, or -1 when every
// child is rejected (p then holds the residual r, which the caller draws the token from). The token this produces is
// an exact sample of p either way, and a child is accepted with probability p's mass on the children.
#include <cstdint>

namespace tqllm {

template <class Uniform>  // Uniform(): double in [0, 1)
inline int tree_sample_step(const uint32_t * ids, double * p, int np, const uint32_t * child, int nc, Uniform && uni) {
    for (int c = 0; c < nc; ++c) {
        double tot = 0, pc = 0;
        for (int i = 0; i < np; ++i) {
            tot += p[i];
            if (ids[i] == child[c]) pc += p[i];
        }
        if (pc > 0 && uni() * tot < pc) return c;
        for (int i = 0; i < np; ++i)
            if (ids[i] == child[c]) p[i] = 0;
    }
    return -1;
}

}  // namespace tqllm
