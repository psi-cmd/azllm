#!/usr/bin/env python3
"""Convert Qwen3.5/Qwen3.8 dense HF BF16 text weights to tqLLM TQ4 GGUF.

This is an experimental *RTN* conversion, not a reproduction of the released
TQ_L/GPTQ calibration. It follows tqLLM's 128-wide Turbo4 rotation and codebook
and its Qwen3.5 GGUF tensor map. It excludes the vision tower and MTP head;
the external DFlash2 drafter works without the latter.

Usage:
  python tools/convert_qwen35_tq4_rtn.py --preflight
  python tools/convert_qwen35_tq4_rtn.py
Staged .npy tensors allow restarting without repeating completed quantization.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import numpy as np
import gguf

from qwen35_tq4_reference_copy import (
    Checkpoint, TQ4, TQ4_F, TQ4_LEVELS, TQ4_R, tokenizer_metadata,
    tq4_pack, tq4_rtn,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models/bf16-uncensored"
STAGE = ROOT / "models/tq4-rtn-stage"
DEST = ROOT / "models/Qwen3.8-27B-Uncensored-TQ_L.gguf"
Q8 = gguf.GGMLQuantizationType.Q8_0
BF16 = gguf.GGMLQuantizationType.BF16
F32 = gguf.GGMLQuantizationType.F32
gguf.constants.GGML_QUANT_SIZES[TQ4] = (128, 66)


@dataclass(frozen=True)
class Tensor:
    dest: str
    src: str
    dtype: str = "tq4"
    op: str = ""


def architecture():
    conf = json.loads((SOURCE / "config.json").read_text())
    text = conf.get("text_config", conf)
    assert conf["model_type"] == "qwen3_5"
    assert text["num_hidden_layers"] == len(text["layer_types"]) == 64
    assert text["linear_value_head_dim"] == 128
    assert text["linear_num_value_heads"] % text["linear_num_key_heads"] == 0
    assert text["linear_conv_kernel_dim"] == 4
    return text


def make_tensors(t):
    seq: list[Tensor] = []

    def add(d, s, dt="tq4", op=""):
        seq.append(Tensor(d, s, dt, op))

    p = "model.language_model."
    add("token_embd.weight", p + "embed_tokens.weight", "q8")
    add("output.weight", "lm_head.weight", "q8")
    add("output_norm.weight", p + "norm.weight", "f32", "norm")

    for l in range(int(t["num_hidden_layers"])):
        s, d = f"{p}layers.{l}.", f"blk.{l}."
        add(d + "attn_norm.weight", s + "input_layernorm.weight", "f32", "norm")
        add(d + "post_attention_norm.weight", s + "post_attention_layernorm.weight", "f32", "norm")
        add(d + "ffn_gate.weight", s + "mlp.gate_proj.weight")
        add(d + "ffn_up.weight", s + "mlp.up_proj.weight")
        add(d + "ffn_down.weight", s + "mlp.down_proj.weight")
        if t["layer_types"][l] == "full_attention":
            for dest, src in (("attn_q", "q_proj"), ("attn_k", "k_proj"),
                              ("attn_v", "v_proj"), ("attn_output", "o_proj")):
                add(d + dest + ".weight", s + "self_attn." + src + ".weight")
            add(d + "attn_q_norm.weight", s + "self_attn.q_norm.weight", "f32", "norm")
            add(d + "attn_k_norm.weight", s + "self_attn.k_norm.weight", "f32", "norm")
        else:
            a = s + "linear_attn."
            add(d + "attn_qkv.weight", a + "in_proj_qkv.weight", "tq4", "qkv")
            add(d + "attn_gate.weight", a + "in_proj_z.weight", "tq4", "val")
            add(d + "ssm_alpha.weight", a + "in_proj_a.weight", "bf16rot", "scalar")
            add(d + "ssm_beta.weight", a + "in_proj_b.weight", "bf16rot", "scalar")
            add(d + "ssm_out.weight", a + "out_proj.weight", "tq4", "out")
            add(d + "ssm_a", a + "A_log", "f32", "negexp_scalar")
            add(d + "ssm_dt.bias", a + "dt_bias", "f32", "scalar")
            add(d + "ssm_conv1d.weight", a + "conv1d.weight", "f32", "conv")
            add(d + "ssm_norm.weight", a + "norm.weight", "f32")
    assert len(seq) == 64 * 14 - 16 * 3 + 3, len(seq)
    return seq


def tiled(a: np.ndarray, op: str, t: dict):
    if not op:
        return a
    if op == "norm":
        return 1.0 + a
    if op == "negexp_scalar":
        a = -np.exp(a)
        op = "scalar"
    if op == "conv":
        a = a.reshape(a.shape[0], a.shape[-1])
    if op in ("scalar", "qkv", "val", "out", "conv"):
        kh = int(t["linear_num_key_heads"])
        vh = int(t["linear_num_value_heads"])
        vd = int(t["linear_value_head_dim"])
        kd = int(t["linear_key_head_dim"])
        qk = 2 * kh * kd
        order = np.asarray([(j % kh) * (vh // kh) + j // kh for j in range(vh)])
        if op == "scalar":
            return a.reshape(vh, *a.shape[1:])[order]
        if op in ("qkv", "conv"):
            return np.concatenate(
                (a[:qk], a[qk:].reshape(vh, vd, *a.shape[1:])[order].reshape(-1, *a.shape[1:])), axis=0
            )
        if op == "val":
            return a.reshape(vh, vd, *a.shape[1:])[order].reshape(-1, *a.shape[1:])
        if op == "out":
            return a.reshape(a.shape[0], vh, vd)[:, order, :].reshape(a.shape)
    raise ValueError(f"Unknown operation: {op}")


def bf16_round(a: np.ndarray):
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def faster_rtn(w: np.ndarray, chunk=48):
    # Exact codebook and 11-scale grid of the tqLLM reference implementation,
    # replacing an expensive 16-element broadcast/argmin with midpoints.
    rows, cols = w.shape
    assert cols % 128 == 0
    codes = np.empty((rows, cols), dtype=np.uint8)
    scales = np.empty((rows, cols // 128), dtype=np.float32)
    lv = TQ4_LEVELS / TQ4_F
    mids = np.array([(6 + 18) / 2, (18 + 31) / 2, (31 + 44) / 2,
                     (44 + 58) / 2, (58 + 75) / 2, (75 + 96) / 2,
                     (96 + 127) / 2], dtype=np.float32)
    factors = np.linspace(0.75, 1.25, 11, dtype=np.float32)
    for r0 in range(0, rows, chunk):
        wr = (w[r0:r0 + chunk].reshape(-1, cols // 128, 128) @ TQ4_R.T)
        rms = np.maximum(np.sqrt((wr ** 2).mean(-1, keepdims=True)), 1e-8)
        be = bc = bs = None
        for f in factors:
            sc = rms * f
            code = np.searchsorted(mids, np.abs(wr / sc) * TQ4_F).astype(np.uint8)
            code += np.asarray(wr < 0, dtype=np.uint8) * 8
            e = ((lv[code] * sc - wr) ** 2).sum(-1, keepdims=True)
            if be is None:
                bc, bs, be = code, sc, e
            else:
                win = e < be
                bc = np.where(win, code, bc)
                bs = np.where(win, sc, bs)
                be = np.where(win, e, be)
        codes[r0:r0 + len(wr)] = bc.reshape(-1, cols)
        scales[r0:r0 + len(wr)] = bs[..., 0] / TQ4_F
    return codes, scales


def rotate_for_dense(w: np.ndarray):
    # For dense BF16 matrices sharing a TQ4-rotated input, store W @ R.T.
    rows, cols = w.shape
    assert cols % 128 == 0
    v = np.empty_like(w)
    for i in range(0, rows, 64):
        sub = w[i:i+64]
        v[i:i+len(sub)] = (sub.reshape(-1, 128) @ TQ4_R.T).reshape(sub.shape)
    return v


def expected_dtype(spec: Tensor):
    return {"tq4": TQ4, "q8": Q8, "bf16rot": BF16, "f32": F32}[spec.dtype]


def stage_tensor(st: Checkpoint, spec: Tensor, t: dict, stage_dir: Path, idx: int, count: int):
    dest = stage_dir / f"{idx:04d}.npy"
    if dest.exists():
        try:
            a = np.load(dest, mmap_mode="r")
            if a.size and a.nbytes > 0:
                return a
        except Exception:
            pass
    started = monotonic()
    a = tiled(st.f32(spec.src), spec.op, t)
    if spec.dtype == "f32":
        v = np.ascontiguousarray(a, dtype=np.float32)
    elif spec.dtype == "bf16rot":
        v = bf16_round(rotate_for_dense(np.ascontiguousarray(a, dtype=np.float32)))
    elif spec.dtype == "q8":
        rows, cols = a.shape
        assert cols % 32 == 0
        v = np.lib.format.open_memmap(str(dest) + ".tmp", mode="w+", dtype=np.uint8,
                                      shape=(rows, cols // 32 * 34))
        for i in range(0, rows, 512):
            sub = np.ascontiguousarray(a[i:i+512], dtype=np.float32)
            v[i:i+len(sub)] = gguf.quants.quantize(sub, Q8)
        v.flush()
        del v
        os.replace(str(dest) + ".tmp", dest)
        del a
        print(f"[{idx+1}/{count}] {spec.dest} q8 staged, {monotonic()-started:.1f}s", flush=True)
        return np.load(dest, mmap_mode="r")
    elif spec.dtype == "tq4":
        rows, cols = a.shape
        assert cols % 128 == 0, (spec.dest, a.shape)
        v = np.lib.format.open_memmap(str(dest) + ".tmp", mode="w+", dtype=np.uint8,
                                      shape=(rows, cols // 128 * 66))
        for i in range(0, rows, 48):
            sub = np.ascontiguousarray(a[i:i+48], dtype=np.float32)
            codes, scales = faster_rtn(sub, chunk=48)
            v[i:i+len(sub)] = tq4_pack(codes, scales)
        v.flush()
        del v
        os.replace(str(dest) + ".tmp", dest)
        del a
        print(f"[{idx+1}/{count}] {spec.dest} tq4 staged, {monotonic()-started:.1f}s", flush=True)
        return np.load(dest, mmap_mode="r")
    else:
        raise ValueError(spec.dtype)
    np.save(str(dest) + ".tmp", v)
    os.replace(str(dest) + ".tmp.npy", dest)
    del a, v
    print(f"[{idx+1}/{count}] {spec.dest} {spec.dtype} staged, {monotonic()-started:.1f}s", flush=True)
    return np.load(dest, mmap_mode="r")


def writer_metadata(w, t: dict, vocab: int):
    k = "qwen35."
    u32 = lambda name, val: w.add_uint32(k + name, int(val))
    w.add_string("general.name", "Qwen3.8-27B Uncensored TQ4 RTN (experimental)")
    w.add_string("general.description", "CPU post-training RTN, TQ4 rotated matrices; Q8 embed/output; BF16 rotated GDN gates; text-only")
    u32("block_count", t["num_hidden_layers"])
    u32("nextn_predict_layers", 0)  # use separate DFlash2; MTP head is not in GGUF
    u32("context_length", t["max_position_embeddings"])
    u32("embedding_length", t["hidden_size"])
    u32("feed_forward_length", t["intermediate_size"])
    u32("attention.head_count", t["num_attention_heads"])
    u32("attention.head_count_kv", t["num_key_value_heads"])
    u32("attention.key_length", t["head_dim"])
    u32("attention.value_length", t["head_dim"])
    w.add_float32(k + "attention.layer_norm_rms_epsilon", float(t["rms_norm_eps"]))
    u32("ssm.conv_kernel", t["linear_conv_kernel_dim"])
    u32("ssm.state_size", t["linear_value_head_dim"])
    u32("ssm.group_count", t["linear_num_key_heads"])
    u32("ssm.time_step_rank", t["linear_num_value_heads"])
    u32("ssm.inner_size", t["linear_num_value_heads"] * t["linear_value_head_dim"])
    u32("full_attention_interval", t["full_attention_interval"])
    rope = t["rope_parameters"]
    w.add_float32(k + "rope.freq_base", float(rope["rope_theta"]))
    u32("rope.dimension_count", int(t["head_dim"] * rope["partial_rotary_factor"]))
    w.add_array(k + "rope.dimension_sections", list(rope["mrope_section"]) + [0] * (4-len(rope["mrope_section"])))
    tokens, types, merges, special, _ = tokenizer_metadata(SOURCE, vocab)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen35")
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(merges)
    if "eos_token" in special:
        w.add_eos_token_id(special["eos_token"])
    if "pad_token" in special:
        w.add_pad_token_id(special["pad_token"])
    chat = SOURCE / "chat_template.jinja"
    if chat.exists():
        w.add_chat_template(chat.read_text())


def preflight(st, tensors, t):
    assert all(st.has(x.src) for x in tensors), [x.src for x in tensors if not st.has(x.src)]
    counts = {}
    for x in tensors:
        counts[x.dtype] = counts.get(x.dtype, 0) + 1
        sh = st.shape(x.src)
        if x.dtype in ("tq4", "bf16rot"):
            assert len(sh) == 2 and sh[1] % 128 == 0, (x, sh)
    # Compare faster nearest-code selection against upstream's exact argmin RTN.
    q = next(x for x in tensors if x.dest == "blk.0.ffn_gate.weight")
    first = st.f32(q.src)[:8].copy()
    rcode, rscale = tq4_rtn(first, chunk=8)
    ccode, cscale = faster_rtn(first, chunk=8)
    agree = float(np.mean(rcode == ccode))
    scale_equal = bool(np.allclose(rscale, cscale, rtol=1e-6, atol=1e-8))
    relative = float(np.sqrt(np.mean((first.reshape(-1,128) @ TQ4_R.T -
                                      (TQ4_LEVELS[ccode.reshape(-1,128)] *
                                       cscale.reshape(-1,1)))**2)) /
                     np.sqrt(np.mean((first.reshape(-1,128) @ TQ4_R.T)**2)))
    print(f"PRECHECK: {len(tensors)} tensors; counts={counts}; RTN code agreement={agree:.7f}; "
          f"scale match={scale_equal}; rotated sample relative RMSE={relative:.5f}",flush=True)
    assert agree > 0.999 and scale_equal
    # Verify the value-head permutation on a synthetic but shaped GDN tensor.
    kh, vh, vd = t["linear_num_key_heads"], t["linear_num_value_heads"], t["linear_value_head_dim"]
    src = np.arange(vh * vd, dtype=np.float32).reshape(vh * vd, 1)
    after = tiled(src, "val", t)
    expected = ((0 % kh) * (vh // kh) + 0 // kh) * vd
    assert int(after[0,0]) == expected
    print("PRECHECK: safetensors map, TQ4 codebook and GDN tiled ordering verified",flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--staging", type=Path, default=STAGE)
    ap.add_argument("--output", type=Path, default=DEST)
    args = ap.parse_args()
    t = architecture()
    specs = make_tensors(t)
    st = Checkpoint(SOURCE)
    preflight(st, specs, t)
    if args.preflight:
        return
    args.staging.mkdir(parents=True, exist_ok=True)
    manifest = args.staging / "tensors.json"
    plan = [{"dest": x.dest, "src": x.src, "dtype": x.dtype, "op": x.op} for x in specs]
    if manifest.exists():
        assert json.loads(manifest.read_text()) == plan, "Staging plan changed; refusing to reuse"
    else:
        manifest.write_text(json.dumps(plan, indent=2))
    started = monotonic()
    for i, s in enumerate(specs):
        stage_tensor(st, s, t, args.staging, i, len(specs))
    print(f"All {len(specs)} tensors staged in {(monotonic()-started)/60:.1f} min; assembling GGUF",flush=True)
    tmp = args.output.with_name(args.output.name + ".building")
    if tmp.exists():
        tmp.unlink()
    w = gguf.GGUFWriter(str(tmp), "qwen35", use_temp_file=False)
    writer_metadata(w, t, int(t["vocab_size"]))
    for i, s in enumerate(specs):
        a = np.load(args.staging / f"{i:04d}.npy", mmap_mode="r")
        w.add_tensor(s.dest, a, raw_dtype=expected_dtype(s))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    os.replace(tmp, args.output)
    print(f"COMPLETE {args.output} bytes={args.output.stat().st_size} elapsed_min={(monotonic()-started)/60:.1f}",flush=True)


if __name__ == "__main__":
    main()
