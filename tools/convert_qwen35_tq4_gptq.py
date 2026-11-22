#!/usr/bin/env python3
"""Qwen3.5 / Qwen3.8 dense HF checkpoint -> tqLLM GGUF in the layout of the released Qwen3.8-27B TQ_L file, with the
TQ4 body quantised by GPTQ on calibration text (see tools/tq4_gptq.py).

Recipe (read from the released TQ_L header, tools/tq_l_recipe.json): every body matrix TQ4; the Gated DeltaNet
ssm_alpha / ssm_beta BF16 in the rotated basis; norms and GDN vectors F32; token_embd Q4_K and output Q6_K (copied
byte for byte from a Q4_K_M donor GGUF of the same weights when it has them, as the release does, else quantised
here); the MTP block (blk.<layers>) in K-quants (Q6_K, attn_k / attn_v Q8_0).

Layer by layer on one stream of calibration hidden states: collect the input Hessians of the layer's matrices on the
outputs of the already quantised layers, quantise, run the quantised layer to get the next inputs. A few held-out
evaluation chunks travel along three times (original weights, GPTQ, plain round to nearest) for a KL divergence and
top-token agreement against the original model at the end. Resumable: finished layers are staged on disk.

  python tools/convert_qwen35_tq4_gptq.py --fp-check            # forward sanity check on the original weights
  python tools/convert_qwen35_tq4_gptq.py --layers 1            # first layer only (timing, proxies)
  python tools/convert_qwen35_tq4_gptq.py                       # everything, then the GGUF
  python tools/convert_qwen35_tq4_gptq.py --calib-text DIR --stage NEW   # recalibrate on your own text

How to run and retune: docs/TQ4_GPTQ_RECIPE.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tq4_gptq as tq  # noqa: E402
from qwen35_tq4_reference_copy import Checkpoint  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = "model.language_model."
GG = {"F32": 0, "Q8_0": 8, "Q4_K": 12, "Q6_K": 14, "BF16": 30, "TQ4": tq.TQ4}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------------------------ calibration text --

def _read(p):
    return Path(p).read_text(encoding="utf-8", errors="ignore")


def build_text_sets(args, tok):
    """-> calibration ids [n][L], evaluation ids [m][L], evaluation kinds. Prose: WikiText-2 (train for calibration,
    test for evaluation); code: Python standard library and this repository's C++ (disjoint files); chat: prose
    passages wrapped in the model's chat template. --calib-text adds your own text: it takes --calib-text-share of the
    calibration chunks (the built-in mix shrinks to the rest) and a quarter of the evaluation chunks, kind "custom",
    from --eval-text or else from the end of the calibration text, which is then kept out of calibration."""
    L = args.seq_len
    data = ROOT / "bench/data"
    train, test = data / "wiki.train.raw", data / "wiki.test.raw"
    if not train.exists() or not test.exists():
        raise SystemExit(f"need {train} and {test} (WikiText-2 raw)")

    def enc(text):
        return tok(text, add_special_tokens=False)["input_ids"]

    def chunks(ids, n, stride=None):
        stride = stride or L
        out = [ids[i:i + L] for i in range(0, len(ids) - L + 1, stride)]
        return out[:n]

    def paragraphs(text):
        return [p.strip() for p in text.split("\n") if len(p.strip()) > 400 and not p.strip().startswith("=")]

    def chat(paras, n):
        asks = ["Summarize the following passage, then continue it in the same style.",
                "Read this text and explain its main points.",
                "Rewrite the passage below more clearly and add what happens next.",
                "这段文字讲了什么？请说明要点并续写。"]
        out, i = [], 0
        while len(out) < n and i + 8 < len(paras):
            msgs = [{"role": "system", "content": "You are a helpful assistant."}] if len(out) % 2 == 0 else []
            ids = []
            while len(ids) < L and i + 2 < len(paras):
                msgs.append({"role": "user", "content": asks[(i // 2) % len(asks)] + "\n\n" + paras[i]})
                msgs.append({"role": "assistant", "content": paras[i + 1]})
                i += 2
                ids = enc(tok.apply_chat_template(msgs, tokenize=False))
            if len(ids) >= L:
                out.append(ids[:L])
        return out

    def code_files():
        import sysconfig
        std = Path(sysconfig.get_paths()["stdlib"])
        py = sorted(p for p in std.glob("*.py") if p.stat().st_size > 30000)
        cc = sorted(p for p in list((ROOT / "src").glob("*.cpp")) + list((ROOT / "tools").glob("*.cpp")) if p.stat().st_size > 20000)
        return py + cc

    def custom_ids(paths):
        """text files (or folders of .txt / .md) as they are; .jsonl lines {"messages": [...]} through the chat template"""
        files = []
        for q in map(Path, paths):
            if not q.exists():
                raise SystemExit(f"no such calibration text: {q}")
            files += sorted(f for f in q.rglob("*") if f.suffix in (".txt", ".md", ".jsonl")) if q.is_dir() else [q]
        ids = []
        for f in files:
            if f.suffix == ".jsonl":
                for line in _read(f).splitlines():
                    if line.strip():
                        ids += enc(tok.apply_chat_template(json.loads(line)["messages"], tokenize=False))
            else:
                ids += enc(_read(f))
        return ids

    def spread(pool, n, what):
        if len(pool) < n:
            raise SystemExit(f"{what}: {len(pool)} chunks of {L} tokens, need {n}: give more text, or lower "
                             f"--calib-text-share / --calib-seqs / --seq-len")
        return [pool[i] for i in np.linspace(0, len(pool) - 1, n).round().astype(int)] if n else []

    n_cal, n_eval = args.calib_seqs, args.eval_chunks
    n_cust = round(n_cal * args.calib_text_share) if args.calib_text else 0
    e_cust = round(n_eval * 0.25) if args.calib_text or args.eval_text else 0
    cust, ev_cust = [], []
    if n_cust or e_cust:
        pool = chunks(custom_ids(args.calib_text), 10 ** 9) if args.calib_text else []
        if args.eval_text:
            ev_cust = spread(chunks(custom_ids(args.eval_text), 10 ** 9), e_cust, "--eval-text")
        else:
            pool, ev_cust = pool[:len(pool) - e_cust], pool[len(pool) - e_cust:]
        cust = spread(pool, n_cust, "--calib-text")
    n_base, e_base = n_cal - n_cust, n_eval - e_cust
    n_chat, n_code = round(n_base * 0.15), round(n_base * 0.25)
    n_prose = n_base - n_chat - n_code
    e_prose, e_code = round(e_base * 6 / 16), round(e_base * 5 / 16)
    e_chat = e_base - e_prose - e_code

    train_text, test_text = _read(train), _read(test)
    rng = np.random.default_rng(0)
    prose_ids = enc(train_text[:n_prose * L * 12])  # ~4 chars per token: plenty
    prose = chunks(prose_ids, 10 ** 9)
    prose = [prose[i] for i in sorted(rng.choice(len(prose), n_prose, replace=False))]
    cal_chat = chat(paragraphs(train_text[len(train_text) // 2:]), n_chat)
    files = code_files()
    ev_files, cal_files = files[::5], [f for i, f in enumerate(files) if i % 5]
    per = max(1, -(-n_code // max(1, len(cal_files))))
    code = []
    for f in cal_files:
        code += chunks(enc(_read(f)), per)
    code = code[:n_code]
    ev_code = []
    for f in ev_files:
        ev_code += chunks(enc(_read(f)), 1)
    ev = ([(c, "prose") for c in chunks(enc(test_text[:e_prose * L * 40]), e_prose, stride=L * 3)]
          + [(c, "code") for c in ev_code[:e_code]]
          + [(c, "chat") for c in chat(paragraphs(test_text), e_chat)]
          + [(c, "custom") for c in ev_cust])
    cal = prose + code + cal_chat + cust
    assert len(cal) == n_cal and len(ev) == n_eval, (len(prose), len(code), len(cal_chat), len(ev))
    log(f"calibration: {len(prose)} prose + {len(code)} code + {len(cal_chat)} chat" + (f" + {len(cust)} custom" if cust else "")
        + f" chunks of {L} tokens; evaluation: {e_prose} prose + {e_code} code + {e_chat} chat"
        + (f" + {len(ev_cust)} custom" if ev_cust else ""))
    return (torch.tensor(cal, dtype=torch.long), torch.tensor([c for c, _ in ev], dtype=torch.long), [k for _, k in ev])


# ---------------------------------------------------------------------------------------- HF -> GGUF tensor rules --

class Rules:
    """llama.cpp's Qwen3.5 conversion rules (as src/core/hf/arch_qwen35.cpp of tqLLM reads a HF folder)"""
    def __init__(self, t):
        self.kh, self.vh = int(t["linear_num_key_heads"]), int(t["linear_num_value_heads"])
        self.vd, self.kd = int(t["linear_value_head_dim"]), int(t["linear_key_head_dim"])
        self.qk = 2 * self.kh * self.kd
        # GGUF (tiled) value head j is HF (grouped) head order[j]
        self.order = np.asarray([(j % self.kh) * (self.vh // self.kh) + j // self.kh for j in range(self.vh)])

    def rows(self, a, block, offset=0):
        """value-head row blocks of `block` rows, after `offset` plain rows, to tiled order"""
        head = a[offset:].reshape(self.vh, block, *a.shape[1:])[self.order].reshape(-1, *a.shape[1:])
        return np.concatenate((a[:offset], head), 0) if offset else head

    def cols(self, a, block):
        """value-head column blocks to tiled order: a [rows][vh * block (...)]"""
        return a.reshape(a.shape[0], self.vh, block, *a.shape[2:])[:, self.order].reshape(a.shape)


def bf16_bits(a: np.ndarray):
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def save(stage: Path, name: str, a: np.ndarray):
    tmp = stage / (name + ".tmp.npy")
    np.save(tmp, a)
    os.replace(tmp, stage / (name + ".npy"))


# ----------------------------------------------------------------------------------------------------- the model --

class Runner:
    def __init__(self, args):
        from transformers import AutoConfig
        from transformers.models.qwen3_5 import modeling_qwen3_5 as M
        self.M = M
        self.args = args
        self.cfg = AutoConfig.from_pretrained(args.source).text_config
        self.cfg._attn_implementation = "sdpa"
        self.tconf = json.loads((args.source / "config.json").read_text())["text_config"]
        self.rules = Rules(self.tconf)
        self.st = Checkpoint(args.source)
        self.rot = M.Qwen3_5TextRotaryEmbedding(self.cfg)
        self.L = int(self.cfg.num_hidden_layers)
        self.dev = args.device

    def w(self, name):
        return torch.from_numpy(self.st.f32(name).copy())

    def layer(self, l):
        layer = self.M.Qwen3_5DecoderLayer(self.cfg, l)
        pre = f"{P}layers.{l}."
        sd = {k: self.w(pre + k) for k in layer.state_dict()}
        layer.load_state_dict(sd, strict=True)
        return layer.float().eval().to(self.dev)

    @torch.no_grad()
    def forward(self, layer, h, bs):
        """h [n][T][d] (CPU) -> layer output, same shape"""
        n, T, _ = h.shape
        out = torch.empty_like(h)
        mask = None
        if layer.block_type == "full_attention":
            mask = torch.full((T, T), torch.finfo(torch.float32).min, device=self.dev).triu(1)[None, None]
        for i in range(0, n, bs):
            x = h[i:i + bs].to(self.dev)
            pos = torch.arange(T, device=self.dev).view(1, 1, -1).expand(3, x.shape[0], -1)
            pe = self.rot.to(self.dev)(x, pos)
            out[i:i + bs] = layer(x, position_embeddings=pe, attention_mask=mask).cpu()
        return out

    def matrices(self, layer):
        """(hf suffix, module, gguf name, Hessian group, row/col rule) of the TQ4 matrices; and the group heads"""
        if layer.block_type == "full_attention":
            a = layer.self_attn
            mats = [("self_attn.q_proj", a.q_proj, "attn_q", "in", None), ("self_attn.k_proj", a.k_proj, "attn_k", "in", None),
                    ("self_attn.v_proj", a.v_proj, "attn_v", "in", None), ("self_attn.o_proj", a.o_proj, "attn_output", "out", None)]
            heads = {"in": a.q_proj, "out": a.o_proj}
        else:
            a = layer.linear_attn
            mats = [("linear_attn.in_proj_qkv", a.in_proj_qkv, "attn_qkv", "in", "qkv"),
                    ("linear_attn.in_proj_z", a.in_proj_z, "attn_gate", "in", "val"),
                    ("linear_attn.out_proj", a.out_proj, "ssm_out", "out", "out")]
            heads = {"in": a.in_proj_qkv, "out": a.out_proj}
        m = layer.mlp
        mats += [("mlp.gate_proj", m.gate_proj, "ffn_gate", "mlp", None), ("mlp.up_proj", m.up_proj, "ffn_up", "mlp", None),
                 ("mlp.down_proj", m.down_proj, "ffn_down", "down", None)]
        heads.update({"mlp": m.gate_proj, "down": m.down_proj})
        return mats, heads

    def to_gguf_layout(self, packed, rule):
        """TQ4 bytes [rows][groups][66] in HF layout -> GGUF layout [rows][groups * 66]"""
        r = self.rules
        if rule == "qkv":
            packed = r.rows(packed, r.vd, r.qk)
        elif rule == "val":
            packed = r.rows(packed, r.vd)
        elif rule == "out":
            assert r.vd == 128
            packed = packed[:, r.order]  # one value head = one 128-column TQ4 group
        return np.ascontiguousarray(packed.reshape(packed.shape[0], -1))

    def stage_small(self, l, stage):
        """the F32 / BF16 tensors of layer l"""
        r, s, d = self.rules, f"{P}layers.{l}.", f"blk.{l}."
        f32 = lambda n: self.st.f32(n).astype(np.float32)
        save(stage, d + "attn_norm.weight", 1 + f32(s + "input_layernorm.weight"))
        save(stage, d + "post_attention_norm.weight", 1 + f32(s + "post_attention_layernorm.weight"))
        if self.tconf["layer_types"][l] == "full_attention":
            save(stage, d + "attn_q_norm.weight", 1 + f32(s + "self_attn.q_norm.weight"))
            save(stage, d + "attn_k_norm.weight", 1 + f32(s + "self_attn.k_norm.weight"))
            return
        a = s + "linear_attn."
        for dst, src in (("ssm_alpha.weight", "in_proj_a.weight"), ("ssm_beta.weight", "in_proj_b.weight")):
            w = torch.from_numpy(r.rows(f32(a + src), 1))
            save(stage, d + dst, bf16_bits(tq.rotate_cols(w.double()).float().numpy()))
        save(stage, d + "ssm_a", r.rows(-np.exp(f32(a + "A_log")), 1))
        save(stage, d + "ssm_dt.bias", r.rows(f32(a + "dt_bias"), 1))
        conv = f32(a + "conv1d.weight")
        save(stage, d + "ssm_conv1d.weight", r.rows(conv.reshape(conv.shape[0], conv.shape[-1]), r.vd, r.qk))
        save(stage, d + "ssm_norm.weight", f32(a + "norm.weight"))


# ---------------------------------------------------------------------------------------------------- main loop --

def hessians(run, layer, heads, h, bs):
    """one forward of h through the original-weight layer, accumulating sum x x^T at the group heads -> {group: H}"""
    acc, hooks = {}, []
    for g, mod in heads.items():
        n = mod.in_features
        acc[g] = torch.zeros((n, n), dtype=torch.float64, device=run.dev)

        def hook(_m, inp, g=g):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            acc[g].add_((x.T @ x).double())
        hooks.append(mod.register_forward_pre_hook(hook))
    run.forward(layer, h, bs)
    for k in hooks:
        k.remove()
    return acc


def rel(a, b):
    return float((a - b).pow(2).sum().sqrt() / b.pow(2).sum().sqrt())


def convert(args, run, stage):
    tok = load_tokenizer(args)
    state_f = stage / "state.json"
    state = json.loads(state_f.read_text()) if state_f.exists() else None
    cal_ids, ev_ids, kinds = build_text_sets(args, tok)
    sig = {"calib_seqs": args.calib_seqs, "seq_len": args.seq_len, "eval_chunks": args.eval_chunks, "percdamp": args.percdamp,
           "scale_weight": args.scale_weight, "grid": args.grid,
           "calib_sha1": hashlib.sha1(cal_ids.numpy().tobytes()).hexdigest(), "eval_sha1": hashlib.sha1(ev_ids.numpy().tobytes()).hexdigest()}
    if state:
        old = {k: state["sig"].get(k, v) for k, v in sig.items()}  # stages from before the text hashes were recorded
        assert old == sig, f"staged run used {state['sig']}, now {sig}: use another --stage"
        nxt = state["next"]
        hs = {k: torch.from_numpy(np.load(stage / f"h_{k}.{nxt}.npy")) for k in ("cal", "fp", "q", "rtn")}
        log(f"resuming at layer {nxt}")
    else:
        nxt = 0
        emb = run.w(P + "embed_tokens.weight")
        emb_q = donor_embedding(args, emb, torch.unique(torch.cat((cal_ids.flatten(), ev_ids.flatten()))))
        hs = {"cal": emb_q[cal_ids], "fp": emb[ev_ids], "q": emb_q[ev_ids], "rtn": emb_q[ev_ids].clone()}
        del emb, emb_q
        np.save(stage / "ev_ids.npy", ev_ids.numpy())
        (stage / "ev_kinds.json").write_text(json.dumps(kinds))
    last = min(run.L, args.layers) if args.layers else run.L
    for l in range(nxt, last):
        t0 = time.time()
        layer = run.layer(l)
        mats, heads = run.matrices(layer)
        hs["fp"] = run.forward(layer, hs["fp"], args.batch)
        t1 = time.time()
        H = hessians(run, layer, heads, hs["cal"], args.batch)
        t2 = time.time()
        orig = {name: mod.weight.data.clone() for name, mod, *_ in mats}
        line = []
        for name, mod, gname, grp, rule in mats:
            codes, scales, wq, stt = tq.tq4_quantize(orig[name], H[grp], args.percdamp, args.scale_weight, args.grid, run.dev)
            save(stage, f"blk.{l}.{gname}.weight", run.to_gguf_layout(tq.tq4_pack(codes, scales), rule))
            mod.weight.data.copy_(wq)
            line.append(f"{gname} {stt.get('proxy', float('nan')):.2e}/{stt['wrel']:.3f}" + ("!" if stt.get("redamped") else ""))
            if args.compare_rtn and "proxy" in stt:
                line[-1] += f" (rtn {tq.rtn_proxy(orig[name], H[grp]):.2e})"
        del H
        run.stage_small(l, stage)
        t3 = time.time()
        hs["cal"] = run.forward(layer, hs["cal"], args.batch)
        hs["q"] = run.forward(layer, hs["q"], args.batch)
        for name, mod, *_ in mats:  # the round-to-nearest baseline stream
            mod.weight.data.copy_(tq.tq4_quantize(orig[name], None, grid=args.grid)[2])
        hs["rtn"] = run.forward(layer, hs["rtn"], args.batch)
        del layer, orig
        for k, v in hs.items():
            np.save(stage / f"h_{k}.{l + 1}.npy", v.numpy())
        state_f.write_text(json.dumps({"next": l + 1, "sig": sig}))
        for k in hs:
            (stage / f"h_{k}.{l}.npy").unlink(missing_ok=True)
        log(f"layer {l}: hidden rel err gptq {rel(hs['q'], hs['fp']):.4f} rtn {rel(hs['rtn'], hs['fp']):.4f} | "
            f"fwd {t1 - t0:.0f}s hess {t2 - t1:.0f}s quant {t3 - t2:.0f}s rest {time.time() - t3:.0f}s | proxy/wrel: " + ", ".join(line))
    if last < run.L:
        return False
    evaluate(args, run, stage, hs, ev_ids, kinds)
    return True


def load_tokenizer(args):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(args.source)


def donor_tensor(args, name):
    """(type id, raw bytes [rows][bytes per row] or array) of a donor tensor, or None"""
    if not args.donor or not Path(args.donor).exists():
        return None
    import gguf
    rd = donor_tensor.cache.get(args.donor) or gguf.GGUFReader(args.donor)
    donor_tensor.cache[args.donor] = rd
    for t in rd.tensors:
        if t.name == name:
            return int(t.tensor_type), t.data
    return None


donor_tensor.cache = {}


def donor_matches(args, name, w: torch.Tensor, type_id, tol):
    """does the donor hold `name` in type `type_id`, and is it a quantisation of w (first rows compared)?"""
    import gguf
    d = donor_tensor(args, name)
    if d is None or d[0] != type_id:
        return None
    raw = np.asarray(d[1])
    n = min(64, w.shape[0])
    blk, bts = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(type_id)]
    raw = raw.reshape(w.shape[0], w.shape[1] // blk * bts)
    y = gguf.quants.dequantize(raw[:n], gguf.GGMLQuantizationType(type_id))
    e = float(np.linalg.norm(y - w[:n].numpy()) / np.linalg.norm(w[:n].numpy()))
    log(f"donor {name}: relative error against the checkpoint {e:.4f}")
    return raw if e < tol else None


def donor_embedding(args, emb, ids):
    """the embedding rows the quantised model will see (Q4_K): from the donor when it matches, else quantised here"""
    import gguf
    out = emb.clone()
    raw = donor_matches(args, "token_embd.weight", emb, GG["Q4_K"], 0.2)
    idn = ids.numpy()
    if raw is None:
        raw_rows = tq.q4k_quantize(emb[ids])
    else:
        raw_rows = np.ascontiguousarray(raw[idn])
    out[ids] = torch.from_numpy(gguf.quants.dequantize(raw_rows, gguf.GGMLQuantizationType.Q4_K).astype(np.float32))
    return out


def output_head(args, run, stage):
    """Q6_K bytes of output.weight (donor copy or quantised here), staged; -> dequantised weight"""
    import gguf
    w = run.w("lm_head.weight")
    f = stage / "output.weight.npy"
    if not f.exists():
        raw = donor_matches(args, "output.weight", w, GG["Q6_K"], 0.1)
        if raw is None:
            log("quantising output.weight to Q6_K")
            raw = np.concatenate([tq.q6k_quantize(w[i:i + 8192]) for i in range(0, w.shape[0], 8192)])
        save(stage, "output.weight", np.ascontiguousarray(raw))
    raw = np.load(f, mmap_mode="r")
    wq = np.concatenate([gguf.quants.dequantize(np.ascontiguousarray(raw[i:i + 16384]), gguf.GGMLQuantizationType.Q6_K)
                         for i in range(0, raw.shape[0], 16384)])
    return w, torch.from_numpy(wq.astype(np.float32))


@torch.no_grad()
def evaluate(args, run, stage, hs, ev_ids, kinds):
    norm_w = 1 + run.w(P + "norm.weight")
    eps = float(run.tconf["rms_norm_eps"])
    w_fp, w_q = output_head(args, run, stage)

    def logp(h, w):
        x = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * norm_w
        return F.log_softmax(x @ w.T, -1)

    res = {k: {"kld": [], "top1": [], "nll": []} for k in ("q", "rtn")}
    nll_fp = []
    skip = 16
    for c in range(ev_ids.shape[0]):
        ref = logp(hs["fp"][c], w_fp)[skip:-1]
        tgt = ev_ids[c, skip + 1:]
        nll_fp.append(float(-ref.gather(1, tgt[:, None]).mean()))
        for k in res:
            lp = logp(hs[k][c], w_q)[skip:-1]
            res[k]["kld"].append(float((ref.exp() * (ref - lp)).sum(-1).mean()))
            res[k]["top1"].append(float((ref.argmax(-1) == lp.argmax(-1)).float().mean()))
            res[k]["nll"].append(float(-lp.gather(1, tgt[:, None]).mean()))
    out = {"kinds": kinds, "nll_original": nll_fp, **res}
    (stage / "eval.json").write_text(json.dumps(out, indent=1))
    kd = np.asarray(kinds)
    log(f"original model: mean NLL {np.mean(nll_fp):.3f} (perplexity {np.exp(np.mean(nll_fp)):.2f}); by kind "
        + ", ".join(f"{k} {np.mean(np.asarray(nll_fp)[kd == k]):.3f}" for k in sorted(set(kinds))))
    for k, name in (("q", "GPTQ"), ("rtn", "RTN ")):
        a = {m: np.asarray(v) for m, v in res[k].items()}
        log(f"{name}: KLD {a['kld'].mean():.4f}, same top token {100 * a['top1'].mean():.1f}%, NLL {a['nll'].mean():.3f} | "
            + " | ".join(f"{kk}: KLD {a['kld'][kd == kk].mean():.4f} top {100 * a['top1'][kd == kk].mean():.1f}%" for kk in sorted(set(kinds))))
    return out


@torch.no_grad()
def fp_check(args, run):
    """the original weights through this script's layer loop: next-token NLL on a little evaluation text"""
    tok = load_tokenizer(args)
    text = _read(ROOT / "bench/data/wiki.test.raw")[:20000]
    ids = torch.tensor(tok(text, add_special_tokens=False)["input_ids"][:args.seq_len * 2]).reshape(2, -1)
    h = run.w(P + "embed_tokens.weight")[ids]
    for l in range(run.L):
        h = run.forward(run.layer(l), h, args.batch)
        if l % 8 == 7:
            log(f"layer {l} rms {float(h.pow(2).mean().sqrt()):.3f}")
    norm_w = 1 + run.w(P + "norm.weight")
    x = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + float(run.tconf["rms_norm_eps"])) * norm_w
    w = run.w("lm_head.weight")
    nll = [float(-F.log_softmax(x[c] @ w.T, -1)[16:-1].gather(1, ids[c, 17:, None]).mean()) for c in range(2)]
    log(f"original weights, WikiText-2 test: NLL {np.mean(nll):.3f}, perplexity {np.exp(np.mean(nll)):.2f}")


# ------------------------------------------------------------------------------------------------------ assembly --

def stage_mtp(args, run, stage):
    """the MTP block and model-level tensors (everything the recipe lists outside the 64 body layers)"""
    import gguf
    L = run.L
    f32 = lambda n: run.st.f32(n).astype(np.float32)
    save(stage, "output_norm.weight", 1 + f32(P + "norm.weight"))
    if not (stage / "token_embd.weight.npy").exists():
        emb = run.w(P + "embed_tokens.weight")
        raw = donor_matches(args, "token_embd.weight", emb, GG["Q4_K"], 0.2)
        if raw is None:
            log("quantising token_embd.weight to Q4_K")
            raw = np.concatenate([tq.q4k_quantize(emb[i:i + 8192]) for i in range(0, emb.shape[0], 8192)])
        save(stage, "token_embd.weight", np.ascontiguousarray(raw))
    if not run.st.has("mtp.fc.weight"):
        return
    s, d = "mtp.layers.0.", f"blk.{L}."
    for dst, src in (("attn_norm", s + "input_layernorm"), ("post_attention_norm", s + "post_attention_layernorm"),
                     ("attn_q_norm", s + "self_attn.q_norm"), ("attn_k_norm", s + "self_attn.k_norm"),
                     ("nextn.enorm", "mtp.pre_fc_norm_embedding"), ("nextn.hnorm", "mtp.pre_fc_norm_hidden"),
                     ("nextn.shared_head_norm", "mtp.norm")):
        save(stage, d + dst + ".weight", 1 + f32(src + ".weight"))
    for dst, src, ty in (("attn_q", s + "self_attn.q_proj", "Q6_K"), ("attn_k", s + "self_attn.k_proj", "Q8_0"),
                         ("attn_v", s + "self_attn.v_proj", "Q8_0"), ("attn_output", s + "self_attn.o_proj", "Q6_K"),
                         ("ffn_gate", s + "mlp.gate_proj", "Q6_K"), ("ffn_up", s + "mlp.up_proj", "Q6_K"),
                         ("ffn_down", s + "mlp.down_proj", "Q6_K"), ("nextn.eh_proj", "mtp.fc", "Q6_K")):
        w = run.w(src + ".weight")
        raw = tq.q6k_quantize(w) if ty == "Q6_K" else gguf.quants.quantize(w.numpy(), gguf.GGMLQuantizationType.Q8_0)
        save(stage, d + dst + ".weight", np.ascontiguousarray(raw))


def assemble(args, run, stage):
    import gguf
    gguf.constants.GGML_QUANT_SIZES[tq.TQ4] = (128, 66)
    recipe = json.loads(Path(args.recipe).read_text())
    stage_mtp(args, run, stage)
    if not (stage / "output.weight.npy").exists():
        output_head(args, run, stage)
    tmp = args.output.with_name(args.output.name + ".building")
    tmp.unlink(missing_ok=True)
    w = gguf.GGUFWriter(str(tmp), "qwen35", use_temp_file=False)
    write_metadata(args, run, w)
    np_dt = {GG["F32"]: np.float32, GG["BF16"]: np.uint16}
    for name, dims, ty in recipe:
        a = np.load(stage / (name + ".npy"))  # in memory: one mapping per tensor runs out of file descriptors
        shape = list(reversed(dims))
        if ty in np_dt:
            assert a.dtype == np_dt[ty] and list(a.shape) == shape, (name, a.dtype, a.shape, shape)
            w.add_tensor(name, a, raw_dtype=gguf.GGMLQuantizationType(ty))
        else:
            blk, bts = gguf.GGML_QUANT_SIZES[ty]
            assert a.dtype == np.uint8 and list(a.shape) == [shape[0], shape[1] // blk * bts], (name, a.shape, shape)
            w.add_tensor(name, a, raw_dtype=ty if ty == tq.TQ4 else gguf.GGMLQuantizationType(ty))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    os.replace(tmp, args.output)
    log(f"wrote {args.output} ({args.output.stat().st_size / 1e9:.2f} GB, {len(recipe)} tensors)")


def write_metadata(args, run, w):
    """the donor's key-values when there is one (it is the same model converted by llama.cpp), else built here"""
    import gguf
    t = run.tconf
    over = {"general.name": args.name, "general.quantized_by": "tools/convert_qwen35_tq4_gptq.py",
            "general.description": "TQ4 body (GPTQ on calibration text, rotated basis), Q6_K output head, Q4_K embeddings; "
                                   "layout of the tqLLM Qwen3.8-27B TQ_L release"}
    if args.donor and Path(args.donor).exists():
        rd = gguf.GGUFReader(args.donor)
        for f in rd.fields.values():
            if f.name.startswith(("GGUF.", "quantize.")) or f.name == "general.architecture" or f.name in over:
                continue
            vt = f.types[0]
            sub = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
            w.add_key_value(f.name, f.contents(), vt, sub_type=sub)
        for k, v in over.items():
            w.add_string(k, v)
        return
    from qwen35_tq4_reference_copy import tokenizer_metadata
    k = "qwen35."
    for key, v in over.items():
        w.add_string(key, v)
    mtp = int(t.get("mtp_num_hidden_layers", 0)) if run.st.has("mtp.fc.weight") else 0
    for name, v in (("block_count", t["num_hidden_layers"] + mtp), ("nextn_predict_layers", mtp),
                    ("context_length", t["max_position_embeddings"]), ("embedding_length", t["hidden_size"]),
                    ("feed_forward_length", t["intermediate_size"]), ("attention.head_count", t["num_attention_heads"]),
                    ("attention.head_count_kv", t["num_key_value_heads"]), ("attention.key_length", t["head_dim"]),
                    ("attention.value_length", t["head_dim"]), ("ssm.conv_kernel", t["linear_conv_kernel_dim"]),
                    ("ssm.state_size", t["linear_value_head_dim"]), ("ssm.group_count", t["linear_num_key_heads"]),
                    ("ssm.time_step_rank", t["linear_num_value_heads"]),
                    ("ssm.inner_size", t["linear_num_value_heads"] * t["linear_value_head_dim"]),
                    ("full_attention_interval", t["full_attention_interval"])):
        w.add_uint32(k + name, int(v))
    w.add_float32(k + "attention.layer_norm_rms_epsilon", float(t["rms_norm_eps"]))
    rope = t["rope_parameters"]
    w.add_float32(k + "rope.freq_base", float(rope["rope_theta"]))
    w.add_uint32(k + "rope.dimension_count", int(t["head_dim"] * rope["partial_rotary_factor"]))
    w.add_array(k + "rope.dimension_sections", list(rope["mrope_section"]) + [0] * (4 - len(rope["mrope_section"])))
    tokens, types, merges, special, _ = tokenizer_metadata(args.source, int(t["vocab_size"]))
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen35")
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(merges)
    if "eos_token" in special:
        w.add_eos_token_id(special["eos_token"])
    if "pad_token" in special:
        w.add_pad_token_id(special["pad_token"])
    chat = args.source / "chat_template.jinja"
    if chat.exists():
        w.add_chat_template(chat.read_text())
    w.add_uint32("general.file_type", 15)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=ROOT / "models/bf16-uncensored")
    ap.add_argument("--donor", default=str(Path.home() / "Qwen3.8-27B-UC-Q4_K_M.gguf"), help="Q4_K_M GGUF of the same weights")
    ap.add_argument("--recipe", default=str(Path(__file__).with_name("tq_l_recipe.json")))
    ap.add_argument("--stage", type=Path, default=ROOT / "models/tq4-gptq-stage")
    ap.add_argument("--output", type=Path, default=ROOT / "models/Qwen3.8-27B-Uncensored-TQ_L-GPTQ.gguf")
    ap.add_argument("--name", default="Qwen3.8-27B Uncensored TQ_L (GPTQ)")
    ap.add_argument("--calib-seqs", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--eval-chunks", type=int, default=16)
    ap.add_argument("--calib-text", action="append", default=[], metavar="PATH",
                    help="your own calibration text: a text file, a folder of .txt / .md, or .jsonl chat lines (repeatable)")
    ap.add_argument("--calib-text-share", type=float, default=0.5, help="share of the calibration chunks taken from --calib-text")
    ap.add_argument("--eval-text", action="append", default=[], metavar="PATH",
                    help="held-out text of the same kind for the evaluation (default: the end of --calib-text)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--scale-weight", choices=("hinv", "none"), default="hinv")
    ap.add_argument("--grid", type=int, default=21)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--layers", type=int, default=0, help="stop after this many layers (testing)")
    ap.add_argument("--compare-rtn", action="store_true", help="also log the round-to-nearest proxy of every matrix")
    ap.add_argument("--fp-check", action="store_true")
    ap.add_argument("--assemble-only", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    run = Runner(args)
    if args.fp_check:
        return fp_check(args, run)
    args.stage.mkdir(parents=True, exist_ok=True)
    if args.assemble_only or convert(args, run, args.stage):
        assemble(args, run, args.stage)


if __name__ == "__main__":
    main()
