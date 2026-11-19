#!/usr/bin/env python3
"""Compare azllm's DFlash2 drafter against the upstream reference, tensor by tensor, on one draft block.

The engine side is a dump written by an azllm server started with ``TQLLM_DFLASH_DUMP=DIR`` (the first DFlash2
draft after a request's prefill: per-layer block states, the drafter's K/V rows, top-16 candidates and selector
scores, plus ``meta.txt`` with the block position and anchor). The reference side is the unmodified NeMo
Automodel ``Qwen3DFlash2DraftModel`` run in FP32 on the same target features (a ``.npy`` from tqfeatures for
the prompt), the same anchor and the same positions.

  python tools/dflash2_train/compare_drafter.py --automodel PATH/Automodel --drafter HF_DIR \\
      --target-gguf TARGET.gguf --dump DIR --features prompt_feats.npy

Each row prints relative RMS error and cosine of engine vs reference; the first tensor whose error jumps well
above the BF16 / 4-bit weight noise floor of its predecessors is where the implementations diverge.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def bf16(path: Path, shape) -> np.ndarray:
    a = np.fromfile(path, dtype=np.uint16)
    return (a.astype(np.uint32) << 16).view(np.float32).reshape(shape)


def stat(name: str, eng: np.ndarray, ref: np.ndarray, rows: slice | None = None) -> None:
    e, r = eng.astype(np.float64), ref.astype(np.float64)
    if rows is not None:
        e, r = e[rows], r[rows]
    e, r = e.reshape(len(e), -1), r.reshape(len(r), -1)
    rel = np.sqrt(((e - r) ** 2).sum() / max((r ** 2).sum(), 1e-30))
    cos = (e * r).sum(1) / (np.linalg.norm(e, axis=1) * np.linalg.norm(r, axis=1) + 1e-30)
    worst = int(np.argmin(cos))
    print(f"  {name:28s} rel-RMS {rel:8.4f}   cos mean {cos.mean():.5f}  min {cos.min():.5f} (row {worst})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--automodel", required=True)
    ap.add_argument("--drafter", required=True, help="HF drafter dir (model.safetensors + config.json)")
    ap.add_argument("--target-gguf", required=True)
    ap.add_argument("--dump", required=True, type=Path)
    ap.add_argument("--features", required=True, type=Path, help="[P, layers*hidden] target features of the context")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    meta = {}
    for line in (args.dump / "meta.txt").read_text().splitlines():
        k, *v = line.split()
        meta[k] = [int(x) for x in v] if k == "draft" else int(v[0])
    pos, anchor, B, H, qd, kd = (meta[k] for k in ("pos", "anchor", "block", "hidden", "qd", "kd"))
    print(f"engine dump: pos {pos} anchor {anchor} block {B} ring {meta['ring']} window {meta['window']} "
          f"draft-vocab subset {meta['subset']}/{meta['dsub_rows']} -> engine draft {meta['draft']}")

    sys.path.insert(0, args.automodel)
    import train as T  # noqa: E402  -- reuse the trainer's import shims and draft construction
    T._stub_hf_datasets()
    T._skip_package_inits(args.automodel, ["nemo_automodel.components.datasets",
                                           "nemo_automodel.components.datasets.llm",
                                           "nemo_automodel.components.speculative.dflash"])
    import torch
    from gguf_target import build_frozen_modules

    dev = torch.device(args.device)
    embed, head, tcfg = build_frozen_modules(args.target_gguf, torch.float32, dev)
    ns = argparse.Namespace(init=args.drafter, draft_layers=5, attention_backend="eager")
    draft, cfg = T.build_draft(ns, tcfg, torch)
    draft = draft.to(device=dev, dtype=torch.float32).eval()
    mask_id = int(cfg.dflash_config["mask_token_id"])

    feats = torch.from_numpy(np.load(args.features).astype(np.float32)).to(dev)
    if feats.shape[0] != pos:
        print(f"WARNING: features cover {feats.shape[0]} positions, the engine block sits at {pos}")
        feats = feats[:pos]
    block = torch.tensor([[anchor] + [mask_id] * (B - 1)], device=dev)
    noise = embed(block)
    position_ids = torch.arange(pos + B, device=dev)[None]

    cap: dict[str, torch.Tensor] = {}
    hooks = []
    for l, layer in enumerate(draft.layers):
        def attn_pre(mod, a, kw, l=l):
            hs, th = kw["hidden_states"], kw["target_hidden"]
            cos_, sin_ = kw["position_embeddings"]
            from nemo_automodel.components.speculative.dflash.draft_qwen3 import apply_rotary_pos_emb
            q = mod.q_norm(mod.q_proj(hs).view(1, B, -1, mod.head_dim)).transpose(1, 2)
            k = torch.cat([mod.k_proj(th), mod.k_proj(hs)], 1).view(1, pos + B, -1, mod.head_dim)
            v = torch.cat([mod.v_proj(th), mod.v_proj(hs)], 1)
            k = mod.k_norm(k).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos_, sin_)
            cap[f"l{l}.q"] = q.transpose(1, 2).reshape(B, -1)
            cap[f"l{l}.kc"] = k.transpose(1, 2).reshape(pos + B, -1)
            cap[f"l{l}.vc"] = v.reshape(pos + B, -1)
        hooks.append(layer.self_attn.register_forward_pre_hook(attn_pre, with_kwargs=True))
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(
            lambda m, a, l=l: cap.__setitem__(f"l{l}.attn", a[0].reshape(B, -1))))
        hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(
            lambda m, a, l=l: cap.__setitem__(f"l{l}.mid", a[0].reshape(B, -1))))
        hooks.append(layer.register_forward_hook(
            lambda m, a, o, l=l: cap.__setitem__(f"l{l}.out", (o[0] if isinstance(o, tuple) else o).reshape(B, -1))))
    with torch.no_grad():
        final = draft(position_ids=position_ids, noise_embedding=noise, target_hidden=feats[None])[0]
        logits = head(final)
        unary, cand = torch.topk(logits, 16, dim=-1)
    for h in hooks:
        h.remove()
    cpu = {k: v.float().cpu().numpy() for k, v in cap.items()}

    d = args.dump
    print("\n== block inputs")
    stat("embed", bf16(d / "embed.bf16", (B, H)), noise[0].cpu().numpy())
    for l in range(len(draft.layers)):
        print(f"== layer {l}")
        if (d / f"l{l}.kc.bf16").exists():
            kc, vc = bf16(d / f"l{l}.kc.bf16", (pos + B, kd)), bf16(d / f"l{l}.vc.bf16", (pos + B, kd))
            stat("K context rows", kc, cpu[f"l{l}.kc"], slice(0, pos))
            stat("V context rows", vc, cpu[f"l{l}.vc"], slice(0, pos))
            stat("K block rows", kc, cpu[f"l{l}.kc"], slice(pos, pos + B))
            stat("V block rows", vc, cpu[f"l{l}.vc"], slice(pos, pos + B))
        stat("Q (normed, rotated)", bf16(d / f"l{l}.q.bf16", (B, qd)), cpu[f"l{l}.q"])
        stat("attention output", bf16(d / f"l{l}.attn.bf16", (B, qd)), cpu[f"l{l}.attn"])
        stat("after attn residual", bf16(d / f"l{l}.mid.bf16", (B, H)), cpu[f"l{l}.mid"])
        stat("layer output", bf16(d / f"l{l}.out.bf16", (B, H)), cpu[f"l{l}.out"])
    print("== head")
    stat("final normed hidden", bf16(d / "final.bf16", (B, H)), final.cpu().numpy())

    e_idx = np.fromfile(d / "cand_idx.u32", dtype=np.uint32).reshape(B, 16)
    r_idx = cand.cpu().numpy()
    for i in range(1, B):
        e, r = list(e_idx[i]), list(r_idx[i])
        print(f"  pos {i}: top-1 engine {e[0]:6d} ref {r[0]:6d} {'==' if e[0] == r[0] else '!='}   "
              f"top-16 overlap {len(set(e) & set(r)):2d}/16")

    # reference selector walk (greedy) over the reference candidates
    sel = draft.candidate_selector if hasattr(draft, "candidate_selector") else None
    if sel is not None:
        with torch.no_grad():
            path, _, _ = sel.walk(final[None, 1:], logits[None, 1:], torch.tensor([anchor], device=dev), 0.0)
        print(f"\nreference draft (selector walk) {path[0].tolist()}")
    print(f"engine draft                    {meta['draft']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
