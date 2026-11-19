#pragma once
#include <algorithm>

namespace tqllm {
// Verify columns of one token-tree step: nt tree columns (root + drafts) and nl prompt-lookup columns. The hybrid
// models' GDN layers replay at most kGdnReplayCols columns per verify, so drafts and lookup share that budget
// (the deferred-commit GDN kernel holds the previous path plus a tree of up to 16 columns in one token group).
inline constexpr int kGdnReplayCols = 16;
// longest lookup chain to ask for: leaves the root and at least one draft
inline int lookup_limit(int requested, int cap) { return std::max(0, std::min(requested, cap - 2)); }
// lookup-first takes the whole block except the root
inline int lookup_first_limit(int cap) { return cap - 1; }
// tree columns (root included) for D drafts next to nl lookup columns (lf: lookup-first took the block)
inline int tree_cols(int D, int nl, bool lf, int cap) { return lf ? 1 : 1 + std::max(0, std::min(D, cap - 1 - nl)); }
}  // namespace tqllm
