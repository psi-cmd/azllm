#!/usr/bin/env bash
# One optimisation round in about a minute: rebuild, start a throwaway tqserve, check one greedy request against the
# saved reference, print tok/s; with --prof also the per-stage verify table (synchronised timing, slower).
#
#   bench/fastloop.sh [--draft-max N] [--save] [--prof] [--no-build] [--reps N] [--max-tokens N] [--temp T]
#                     [--prompt FILE] [--long (~30K-token prompt, timing only)]
# env: FL_ENV="VAR=value ..." for the timed server, TQSERVE=binary (A/B against an older build)
#
# --save stores the reference from the current build (do this once on a known-good build). Stops any running
# tqserve (port 8086 included): restore the service afterwards (opslatch start azllm-qwen38-uc).
set -u
cd "$(dirname "$0")/.." || exit 1
M=$PWD B=${BUILD:-build-conda131-fixed} BIN=${TQSERVE:-} PORT=${PORT:-8087} DM=15 SAVE= PROF= BUILD_IT=1 PASS=()
REF=${REF:-$HOME/.cache/azllm-fastloop/ref.txt}
while [ $# -gt 0 ]; do
  case $1 in
    --draft-max) DM=$2; shift ;;
    --save) SAVE=--save ;;
    --prof) PROF=1 ;;
    --no-build) BUILD_IT= ;;
    --reps|--max-tokens|--temp|--prompt) PASS+=("$1" "$2"); shift ;;
    --long) bash bench/fastloop-long.sh; PASS+=(--prompt /tmp/fastloop-long.txt); CTX=65536 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac; shift
done
mkdir -p "$(dirname "$REF")"
export PATH=/home/psi-cmd/miniconda3/envs/cdgpt/bin:$PATH
if [ -n "$BUILD_IT" ]; then
  t=$(date +%s)
  cmake --build $B -j 24 --target tqserve > /tmp/fastloop-build.log 2>&1 || { grep -E "error|FAILED" /tmp/fastloop-build.log | head; exit 1; }
  echo "build $(( $(date +%s) - t ))s"
fi
serve() {  # serve ENV...: start tqserve with the production model / drafter / KV settings, short context
  pkill -x tqserve; for _ in $(seq 30); do pgrep -x tqserve >/dev/null || break; sleep 1; done
  env "$@" ${BIN:-$B/tqserve} --model models/Qwen3.8-27B-Uncensored-TQ_L-GPTQ.gguf --name x --host 127.0.0.1 --port $PORT \
    --context ${CTX:-32768} --np 1 --kv q4/tcq2 --kv-host 0 --prompt-cache off --reasoning on --max-tokens 8192 \
    -md models/drafters/Qwen3.8-27B-DFlash2-TQ3.gguf --draft-vocab models/mtp-vocab/atx_65536.txt \
    --draft-max $DM > /tmp/fastloop-server.log 2>&1 < /dev/null &
  SP=$!
  for _ in $(seq 150); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && return 0; kill -0 $SP 2>/dev/null || break; sleep 1; done
  echo "server did not start:"; tail -5 /tmp/fastloop-server.log; return 1
}
t=$(date +%s)
serve ${FL_ENV:-} || exit 1  # FL_ENV="TQLLM_MATVEC_RULE=4:split16 ...": extra environment for the timed server
echo "server up $(( $(date +%s) - t ))s (draft-max $DM)"
python3 bench/fastloop.py http://127.0.0.1:$PORT "$REF" $SAVE "${PASS[@]}"; rc=$?
grep -i -E "trap|illegal|error" /tmp/fastloop-server.log | head -3
if [ -n "$PROF" ]; then
  kill $SP; wait $SP 2>/dev/null
  serve ${FL_ENV:-} TQLLM_DEBUG=profile,stageprof || exit 1
  python3 bench/fastloop.py http://127.0.0.1:$PORT "$REF" --reps 1 >/dev/null
  python3 - /tmp/fastloop-server.log <<'P'
import re, sys, collections
s, n = collections.defaultdict(float), 0
for l in open(sys.argv[1], errors="ignore"):
    if "stage profile" in l: n += 1; continue
    m = re.match(r"\s{2}(\S.*?)\s+([\d.]+) us/block", l)
    if m and n: s[m.group(1)] += float(m.group(2))
tot = sum(s.values())
print(f"stage profile, mean of {n} tables, {tot / max(1, n) / 1000:.2f} ms per verify block (synchronised)")
for k, v in sorted(s.items(), key=lambda kv: -kv[1])[:12]: print(f"  {k:34s} {v / max(1, n):8.1f} us")
P
fi
kill $SP; wait $SP 2>/dev/null
exit $rc
