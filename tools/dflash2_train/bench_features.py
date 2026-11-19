#!/usr/bin/env python3
"""Measure tqfeatures extraction throughput: N servers in parallel, a mix of sequence lengths.

Reports tokens/s overall, a per-request fit  t = a + b*P  (fixed overhead a, per-token cost b), and
optionally the cost of writing each frame to one or more cache directories.

  python tools/dflash2_train/bench_features.py --features-bin build/tqfeatures \
      --target-gguf TARGET.gguf --drafter-gguf DRAFTER.gguf --data train.jsonl \
      --servers 2 --lengths 512,1024,2048,3000 --per-length 4 --write-dirs /tmp/fc,/data/fc
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FeatureCache, FeatureServer  # noqa: E402


def token_pool(path: str) -> np.ndarray:
    ids: list[int] = []
    for line in open(path):
        r = json.loads(line)
        ids += r.get("input_ids") or (r["prompt_ids"] + r["response_ids"])
    return np.asarray(ids, dtype=np.int64)


def make_requests(pool: np.ndarray, lengths: list[int], per_length: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    reps = -(-max(lengths) * 2 // len(pool))
    pool = np.tile(pool, max(1, reps))
    out = []
    for L in lengths:
        for _ in range(per_length):
            o = int(rng.integers(0, len(pool) - L))
            out.append(pool[o:o + L].copy())
    rng.shuffle(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-bin", required=True)
    ap.add_argument("--target-gguf", required=True)
    ap.add_argument("--drafter-gguf", required=True)
    ap.add_argument("--data", required=True, help="JSONL; its tokens are sliced into requests")
    ap.add_argument("--servers", type=int, default=1)
    ap.add_argument("--feature-context", type=int, default=4096)
    ap.add_argument("--feature-args", default="")
    ap.add_argument("--lengths", default="512,1024,2048,3000")
    ap.add_argument("--per-length", type=int, default=4)
    ap.add_argument("--pipeline-depth", type=int, choices=(1, 2), default=1,
                    help="number of in-flight frames per server (2 overlaps compute and pipe writes)")
    ap.add_argument("--write-dirs", default="", help="comma-separated dirs to time cache writes into (wiped)")
    ap.add_argument("--log-dir", default=".")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    lengths = [int(x) for x in args.lengths.split(",")]
    reqs = make_requests(token_pool(args.data), lengths, args.per_length, args.seed)
    cmd = [args.features_bin, "--model", args.target_gguf, "-md", args.drafter_gguf,
           "--context", str(args.feature_context)] + shlex.split(args.feature_args)

    t0 = time.time()
    servers = [FeatureServer(cmd, log_path=f"{args.log_dir}/bench_tqfeatures.{i}.log") for i in range(args.servers)]
    warm = [threading.Thread(target=s.features, args=(reqs[0][:64],)) for s in servers]
    for t in warm:
        t.start()
    for t in warm:
        t.join()
    load_s = time.time() - t0

    lock = threading.Lock()
    todo = list(reqs)
    timings: list[tuple[int, float]] = []
    frames: list[tuple[np.ndarray, np.ndarray]] = []
    errors: list[Exception] = []

    def worker(srv: FeatureServer):
        pending = deque()
        try:
            while True:
                while len(pending) < args.pipeline_depth:
                    with lock:
                        ids = todo.pop() if todo else None
                    if ids is None:
                        break
                    start = time.perf_counter()
                    srv.send(ids)
                    pending.append((ids, start))
                if not pending:
                    break
                ids, start = pending.popleft()
                hid = srv.receive()
                dt = time.perf_counter() - start
                if hid is None:
                    raise RuntimeError(f"server rejected a {len(ids)}-token request")
                with lock:
                    timings.append((len(ids), dt))
                    if len(frames) < 8:
                        frames.append((ids, hid.copy()))
        except Exception as exc:
            with lock:
                errors.append(exc)

    t1 = time.time()
    threads = [threading.Thread(target=worker, args=(s,)) for s in servers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t1
    for s in servers:
        s.close()
    if errors:
        raise errors[0]
    if not timings:
        raise RuntimeError("feature benchmark returned no frames")

    P = np.array([p for p, _ in timings], dtype=np.float64)
    T = np.array([t for _, t in timings])
    b, a = np.polyfit(P, T, 1) if len(set(P)) > 1 else (T.mean() / P.mean(), 0.0)
    tok = int(P.sum())
    print(f"config: servers={args.servers} feature_args={args.feature_args!r} "
          f"context={args.feature_context} pipeline_depth={args.pipeline_depth}")
    print(f"load+warmup {load_s:.1f}s | {len(timings)} requests, {tok} tokens in {wall:.1f}s "
          f"-> {tok / wall:.0f} tok/s total, {tok / wall / args.servers:.0f} tok/s per server")
    print(f"per-request fit: t = {a * 1e3:.0f} ms + {b * 1e3:.3f} ms/token")
    for L in lengths:
        m = P == L
        if m.any():
            print(f"  P={L:5d}: {T[m].mean() * 1e3:8.0f} ms  ({L / T[m].mean():.0f} tok/s)")

    for d in filter(None, args.write_dirs.split(",")):
        shutil.rmtree(d, ignore_errors=True)
        cache = FeatureCache(d)
        s = time.perf_counter()
        nbytes = 0
        for ids, hid in frames:
            cache.put(ids, hid)
            nbytes += hid.nbytes
        dt = time.perf_counter() - s
        ntok = sum(len(i) for i, _ in frames)
        print(f"cache write {d}: {nbytes / 2**20 / dt:.0f} MiB/s ({ntok / dt:.0f} tok/s)")
        shutil.rmtree(d, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
