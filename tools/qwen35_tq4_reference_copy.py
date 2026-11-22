#!/usr/bin/env python3
"""Convert perplexity-ai/pplx-embed-v2-context-* (PplxContextualModel over Qwen3_5Model, MIT) to a qwen35 GGUF that
tqserve serves on /v1/embeddings.

The text tower is written as the qwen35 engine reads Qwen3.5 (the rules of src/core/hf/arch_qwen35.cpp: every RMSNorm
stored as 1 + w except the GDN output norm, A_log -> -exp(A_log), conv1d [ch][1][k] -> [ch][k]; GDN value heads stay in
the Hugging Face order: value head j reads key head j // (value heads / key heads), as repeat_interleave does),
plus the contextual projection (FP32) and the pooling metadata (pplx.*). The vision tower is not written.
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np

DTYPES = {"BF16": (np.uint16, 2), "F32": (np.float32, 4), "F16": (np.float16, 2)}


class Safetensors:
    def __init__(self, path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.base = 8 + n
        self.header.pop("__metadata__", None)

    def has(self, name):
        return name in self.header

    def raw(self, name):
        """(array as stored, dtype tag): BF16 as uint16 bit patterns"""
        e = self.header[name]
        np_type, size = DTYPES[e["dtype"]]
        a, b = e["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self.base + a)
            data = np.frombuffer(f.read(b - a), dtype=np_type)
        return data.reshape(e["shape"]), e["dtype"]

    def f32(self, name):
        a, tag = self.raw(name)
        if tag == "BF16":
            return (a.astype(np.uint32) << 16).view(np.float32)
        return a.astype(np.float32)


class Checkpoint:
    """model.safetensors, or the shards of model.safetensors.index.json, plus contextual_head.safetensors (the 9B)"""
    def __init__(self, snap):
        snap = Path(snap)
        files = [snap / "model.safetensors"] if (snap / "model.safetensors").exists() else []
        index = snap / "model.safetensors.index.json"
        if index.exists():
            files = sorted({snap / f for f in json.loads(index.read_text())["weight_map"].values()})
        if (snap / "contextual_head.safetensors").exists():
            files.append(snap / "contextual_head.safetensors")
        if not files:
            raise ValueError("no safetensors in " + str(snap))
        self.where = {}
        for f in files:
            st = Safetensors(f)
            for name in st.header:
                self.where[name] = st

    def has(self, name):
        return name in self.where

    def raw(self, name):
        return self.where[name].raw(name)

    def shape(self, name):
        return self.where[name].header[name]["shape"]

    def f32(self, name):
        return self.where[name].f32(name)


def tokenizer_metadata(snap, vocab):
    t = json.loads((snap / "tokenizer.json").read_text())
    c = json.loads((snap / "tokenizer_config.json").read_text())
    tokens, types = [""] * vocab, [5] * vocab                       # 5 unused
    for piece, i in t["model"]["vocab"].items():
        if i < vocab:
            tokens[i], types[i] = piece, 1                          # normal
    for a in t["added_tokens"]:
        i = a["id"]
        if i < vocab:
            tokens[i], types[i] = a["content"], 3 if a.get("special") else 4  # control : user-defined
    for i in range(vocab):
        if not tokens[i]:
            tokens[i] = f"[PAD{i}]"
    merges = [m if isinstance(m, str) else m[0] + " " + m[1] for m in t["model"]["merges"]]
    ids = {s: i for i, s in enumerate(tokens)}
    special = {}
    for key in ("eos_token", "pad_token"):
        v = c.get(key)
        v = v.get("content") if isinstance(v, dict) else v
        if v in ids:
            special[key] = ids[v]
    return tokens, types, merges, special, ids


def full_attention_interval(layer_types):
    """the period of the full-attention layers (layer l is full when (l + 1) % n == 0, as the qwen35 engine reads it).
    pplx-embed-v2-context-0.6b says full_attention_interval 4 in its config but its layer_types alternate (n = 2):
    layer_types is what the model runs."""
    full = [t == "full_attention" for t in layer_types]
    for n in range(1, len(full) + 1):
        if full == [(l + 1) % n == 0 for l in range(len(full))]:
            return n
    raise ValueError("layer_types is not periodic: " + str(layer_types))


# TQ4 (GGUF 300, tqLLM): per 128 weights of a row, an FP16 scale in engine units then 64 bytes of 4-bit codes (low nibble =
# even column); w_rot = L[code] * scale with L the int8 Turbo4 levels, w = w_rot R per 128, R = H diag(s1) / sqrt(128).
TQ4 = 300
TQ4_LEVELS = np.array([6, 18, 31, 44, 58, 75, 96, 127, -6, -18, -31, -44, -58, -75, -96, -127], dtype=np.float32)
TQ4_F = 127.0 / 2.7326  # engine level = F x Lloyd-Max level
_S1 = np.array([1 - 2 * ((0xE46A0359, 0x42F85949, 0xA4C63BBB, 0x49F250A9)[i // 32] >> (i % 32) & 1) for i in range(128)], dtype=np.float32)
_H = np.ones((1, 1), dtype=np.float32)
while _H.shape[0] < 128:
    _H = np.block([[_H, _H], [_H, -_H]])
TQ4_R = _H * _S1[None, :] / np.sqrt(128.0)


def tq4_pack(codes, scale_eng):
    """codes uint8 [rows][cols] (0..15), engine scales [rows][cols / 128] -> bytes [rows][cols / 128 * 66]"""
    rows, cols = codes.shape
    c = codes.reshape(rows, cols // 2, 2)
    nib = (c[..., 0] | (c[..., 1] << 4)).astype(np.uint8).reshape(rows, cols // 128, 64)
    sc = scale_eng.astype(np.float16).view(np.uint8).reshape(rows, cols // 128, 2)
    return np.concatenate([sc, nib], axis=-1).reshape(rows, -1)


def tq4_rtn(w, chunk=1024):
    """round to nearest on the rotated rows, per-group scale searched over 0.75 .. 1.25 x RMS -> codes, engine scales"""
    rows, cols = w.shape
    lv = TQ4_LEVELS / TQ4_F
    codes = np.empty((rows, cols), dtype=np.uint8)
    scales = np.empty((rows, cols // 128), dtype=np.float32)
    for r0 in range(0, rows, chunk):
        wr = (w[r0:r0 + chunk].reshape(-1, cols // 128, 128) @ TQ4_R.T)
        rms = np.maximum(np.sqrt((wr ** 2).mean(-1, keepdims=True)), 1e-8)
        best_c = best_s = best_e = None
        for fct in np.linspace(0.75, 1.25, 11, dtype=np.float32):
            sc = rms * fct
            cc = np.abs((wr / sc)[..., None] - lv).argmin(-1)
            e = ((lv[cc] * sc - wr) ** 2).sum(-1, keepdims=True)
            if best_e is None:
                best_c, best_s, best_e = cc, sc, e
            else:
                m = e < best_e
                best_c, best_s, best_e = np.where(m, cc, best_c), np.where(m, sc, best_s), np.where(m, e, best_e)
        codes[r0:r0 + chunk] = best_c.reshape(-1, cols).astype(np.uint8)
        scales[r0:r0 + chunk] = best_s[..., 0] / TQ4_F
    return codes, scales

