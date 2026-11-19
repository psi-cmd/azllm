#!/usr/bin/env bash
# Unsynchronised per-kernel GPU time of one fast-loop request (bench/kprof.cpp via LD_PRELOAD).
#   bench/kprof.sh [--draft-max N] [--top N] [--long]      -> /tmp/kprof.txt
# --long: ~30K-token prompt; the window then starts at the first generated token (decode only, bench/kprof_client.py)
# Stops any running tqserve (8086 included); restore with opslatch start azllm-qwen38-uc.
set -u
cd "$(dirname "$0")/.." || exit 1
DM=7 TOP=40 B=${BUILD:-build-conda131-fixed} PORT=8087 CTX=32768 PROMPT=()
while [ $# -gt 0 ]; do case $1 in --draft-max) DM=$2; shift ;; --top) TOP=$2; shift ;; --long) bash bench/fastloop-long.sh; CTX=65536; PROMPT=(--prompt /tmp/fastloop-long.txt) ;; esac; shift; done
CP=${CUPTI:-/home/psi-cmd/miniconda3/pkgs/cuda-cupti-12.4.127-0}
SO=/tmp/kprof.so
if [ ! -f $SO ] || [ bench/kprof.cpp -nt $SO ]; then
  g++ -O2 -shared -fPIC -std=c++17 -I$CP/include -I/home/psi-cmd/miniconda3/envs/cdgpt/targets/x86_64-linux/include \
    -I/home/psi-cmd/miniconda3/envs/cdgpt/include bench/kprof.cpp -o $SO -L$CP/lib -lcupti -lpthread -Wl,-rpath,$CP/lib || exit 1
fi
pkill -x tqserve; for _ in $(seq 30); do pgrep -x tqserve >/dev/null || break; sleep 1; done
LD_PRELOAD=$SO $B/tqserve --model models/Qwen3.8-27B-Uncensored-TQ_L-GPTQ.gguf --name x --host 127.0.0.1 --port $PORT \
  --context $CTX --np 1 --kv q4/tcq2 --kv-host 0 --prompt-cache off --reasoning on --max-tokens 8192 \
  -md models/drafters/Qwen3.8-27B-DFlash2-TQ3.gguf --draft-vocab models/mtp-vocab/atx_65536.txt \
  --draft-max $DM > /tmp/kprof-server.log 2>&1 < /dev/null &
SP=$!
for _ in $(seq 150); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; sleep 1; done
REF=$HOME/.cache/azllm-fastloop/ref.txt
python3 bench/fastloop.py http://127.0.0.1:$PORT "$REF" --reps 1 "${PROMPT[@]}" >/dev/null  # warm-up (graphs captured)
if [ ${#PROMPT[@]} -gt 0 ]; then python3 bench/kprof_client.py http://127.0.0.1:$PORT $SP ${PROMPT[1]} 1024; else
kill -USR2 $SP; sleep 0.5
python3 bench/fastloop.py http://127.0.0.1:$PORT "$REF" --reps 1 "${PROMPT[@]}"
sleep 0.5; kill -USR1 $SP; fi; sleep 2
kill $SP; wait $SP 2>/dev/null
grep -i kprof /tmp/kprof-server.log | head -3
grep -E "per step" /tmp/kprof-server.log | tail -1
head -$TOP /tmp/kprof.txt
