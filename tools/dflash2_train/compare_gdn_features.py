#!/usr/bin/env python3
"""Compare baseline and experimental GDN prefill across actual target-model features.

Use the same GGUFs, token stream and server binary for both runs. The
comparison is BF16 feature-data based, not a top-token-only check.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from bench_features import make_requests, token_pool
from features import FeatureServer


def bf16_to_float(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        raise ValueError(f"frame shape differs: {a.shape} vs {b.shape}")
    numerator = denominator = 0.0
    max_abs = 0.0
    differing = 0
    for start in range(0, len(a), 32):
        aa, bb = a[start:start + 32], b[start:start + 32]
        differing += int(np.count_nonzero(aa != bb))
        left, right = bf16_to_float(aa), bf16_to_float(bb)
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("non-finite feature output")
        err = left.astype(np.float64) - right.astype(np.float64)
        numerator += float(np.sum(err ** 2))
        denominator += float(np.sum(left.astype(np.float64) ** 2))
        max_abs = max(max_abs, float(np.max(np.abs(err))))
    return {
        "relative_rms": float(np.sqrt(numerator / max(denominator, 1e-30))),
        "max_absolute": max_abs,
        "different_fraction": differing / a.size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-bin", required=True)
    parser.add_argument("--target-gguf", required=True)
    parser.add_argument("--drafter-gguf", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--lengths", default="55,111,512,1024,2048,3000")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    dest = Path(args.out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    reqs = make_requests(token_pool(args.data),
                         [int(x) for x in args.lengths.split(",")], 1, 0)
    cmd = [args.features_bin, "--model", args.target_gguf,
           "-md", args.drafter_gguf, "--context", "4096", "--prefill-int8", "off"]
    results = []
    baseline: list[np.ndarray] = []

    for mode in ("baseline", "precomp"):
        env = {"TQLLM_GDN_CHUNK_MIN": "16", "TQLLM_GDN_PRECOMP": "1"} if mode == "precomp" else {
            "TQLLM_GDN_CHUNK_MIN": "", "TQLLM_GDN_PRECOMP": ""
        }
        # Blank GDN_CHUNK_MIN is not a valid baseline: FeatureServer inherits
        # the parent env. Launch baseline with the flag explicitly set to an
        # unreachable sequence length while preserving the same executable.
        if mode == "baseline":
            env = {"TQLLM_GDN_CHUNK_MIN": "9999999"}
        server = FeatureServer(cmd, env=env, log_path=str(dest / f"{mode}.stderr.log"))
        try:
            for index, ids in enumerate(reqs):
                start = time.perf_counter()
                frame = server.features(ids)
                elapsed_ms = (time.perf_counter() - start) * 1000
                if frame is None:
                    raise RuntimeError(f"{mode} rejected {len(ids)} tokens")
                if mode == "baseline":
                    baseline.append(frame)
                    result = {"tokens": len(ids), "baseline_ms": round(elapsed_ms, 1)}
                    results.append(result)
                else:
                    result = results[index]
                    result["precomp_ms"] = round(elapsed_ms, 1)
                    result.update(compare(baseline[index], frame))
                    print(json.dumps(result), flush=True)
        finally:
            server.close()

    (dest / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    ok = all(r["relative_rms"] < 0.005 for r in results)
    print("MODEL_FEATURE_GDN " + ("PASS" if ok else "FAIL"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
