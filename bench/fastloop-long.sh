#!/usr/bin/env bash
# /tmp/fastloop-long.txt: a ~30K-token prompt (100 KB of src/qwen35.cpp at a fixed revision, so it never changes)
cd "$(dirname "$0")/.." || exit 1
[ -s /tmp/fastloop-long.txt ] && exit 0
{ echo "Here is a C++ source file from an inference engine:"; echo '```cpp'; git show 9944dd5:src/qwen35.cpp | head -c 100000
  echo; echo '```'; echo "Explain in detail what verify_body does, step by step."; } > /tmp/fastloop-long.txt
