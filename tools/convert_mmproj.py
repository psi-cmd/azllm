#!/usr/bin/env python3
"""Qwen3.5/3.6/3.8 HF checkpoint (model.visual.*) -> mmproj GGUF for tqserve --mmproj (projector qwen3vl_merger).

Needs only numpy: safetensors is read by its header, GGUF is written directly. Tensor names and metadata follow
llama.cpp's mmproj layout, which src/vision.cpp loads.

  python3 tools/convert_mmproj.py models/bf16-uncensored models/mmproj-BF16.gguf [--type bf16|f16]

--type sets the matrices' storage (norms, biases and embeddings stay F32): bf16 keeps the checkpoint's bits.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

GGML_F32, GGML_F16, GGML_BF16 = 0, 1, 30
ALIGN = 32


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


class Checkpoint:
    def __init__(self, d):
        self.d = d
        idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.where = {k: v for k, v in idx.items() if k.startswith("model.visual.")}
        self.heads = {}

    def _head(self, fn):
        if fn not in self.heads:
            with open(os.path.join(self.d, fn), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self.heads[fn] = (json.loads(f.read(n)), 8 + n)
        return self.heads[fn]

    def get(self, name):
        """(raw uint16 BF16 bits or float32 array, shape); raw bits keep BF16 exact."""
        fn = self.where[name]
        h, base = self._head(fn)
        t = h[name]
        a, b = t["data_offsets"]
        dt = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}[t["dtype"]]
        arr = np.memmap(os.path.join(self.d, fn), dtype=dt, mode="r", offset=base + a, shape=tuple(t["shape"]))
        assert arr.nbytes == b - a
        return np.array(arr), t["dtype"]


def to_f32(arr, dtype):
    return bf16_to_f32(arr) if dtype == "BF16" else arr.astype(np.float32)


class Writer:
    def __init__(self):
        self.kv, self.tensors = [], []

    def add(self, key, vtype, value):
        self.kv.append((key, vtype, value))

    def tensor(self, name, data, gtype):
        self.tensors.append((name, np.ascontiguousarray(data), gtype))

    @staticmethod
    def _str(s):
        b = s.encode()
        return struct.pack("<Q", len(b)) + b

    def _val(self, vtype, v):
        if vtype == "u32": return struct.pack("<I", 4) + struct.pack("<I", v)
        if vtype == "f32": return struct.pack("<I", 6) + struct.pack("<f", v)
        if vtype == "bool": return struct.pack("<I", 7) + struct.pack("<?", v)
        if vtype == "str": return struct.pack("<I", 8) + self._str(v)
        if vtype == "f32[]": return struct.pack("<IIQ", 9, 6, len(v)) + struct.pack("<%df" % len(v), *v)
        if vtype == "bool[]": return struct.pack("<IIQ", 9, 7, len(v)) + struct.pack("<%d?" % len(v), *v)
        raise ValueError(vtype)

    def write(self, path):
        out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(self.tensors), len(self.kv)))
        for k, t, v in self.kv:
            out += self._str(k) + self._val(t, v)
        off = 0
        for name, data, gtype in self.tensors:
            ne = list(reversed(data.shape))
            out += self._str(name) + struct.pack("<I", len(ne)) + struct.pack("<%dQ" % len(ne), *ne)
            out += struct.pack("<IQ", gtype, off)
            off += (data.nbytes + ALIGN - 1) // ALIGN * ALIGN
        out += b"\0" * (-len(out) % ALIGN)
        with open(path + ".tmp", "wb") as f:
            f.write(out)
            for _, data, _ in self.tensors:
                f.write(data.tobytes())
                f.write(b"\0" * (-data.nbytes % ALIGN))
        os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--type", choices=["bf16", "f16"], default="bf16")
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.model_dir, "config.json")))
    vc = cfg["vision_config"]
    if vc.get("deepstack_visual_indexes"):
        sys.exit("deepstack layers are not supported by src/vision.cpp")
    pp = json.load(open(os.path.join(a.model_dir, "preprocessor_config.json")))
    ck = Checkpoint(a.model_dir)

    w = Writer()
    w.add("general.architecture", "str", "clip")
    w.add("general.name", "str", os.path.basename(os.path.normpath(a.model_dir)) + " vision")
    w.add("clip.has_vision_encoder", "bool", True)
    w.add("clip.projector_type", "str", "qwen3vl_merger")
    w.add("clip.vision.embedding_length", "u32", vc["hidden_size"])
    w.add("clip.vision.feed_forward_length", "u32", vc["intermediate_size"])
    w.add("clip.vision.block_count", "u32", vc["depth"])
    w.add("clip.vision.attention.head_count", "u32", vc["num_heads"])
    w.add("clip.vision.patch_size", "u32", vc["patch_size"])
    w.add("clip.vision.projection_dim", "u32", vc["out_hidden_size"])
    w.add("clip.vision.spatial_merge_size", "u32", vc["spatial_merge_size"])
    w.add("clip.vision.attention.layer_norm_epsilon", "f32", 1e-6)
    w.add("clip.vision.image_mean", "f32[]", [float(x) for x in pp["image_mean"]])
    w.add("clip.vision.image_std", "f32[]", [float(x) for x in pp["image_std"]])
    w.add("clip.use_gelu", "bool", True)

    def mat(name, src):
        arr, dt = ck.get(src)
        if a.type == "bf16" and dt == "BF16": w.tensor(name, arr, GGML_BF16)
        else: w.tensor(name, to_f32(arr, dt).astype(np.float16), GGML_F16)

    def f32(name, src):
        arr, dt = ck.get(src)
        w.tensor(name, to_f32(arr, dt), GGML_F32)

    V = "model.visual."
    pw, dt = ck.get(V + "patch_embed.proj.weight")  # [out][3][t=2][p][p]
    assert pw.shape[2] == 2, pw.shape
    w.tensor("v.patch_embd.weight", to_f32(pw[:, :, 0], dt), GGML_F32)
    w.tensor("v.patch_embd.weight.1", to_f32(pw[:, :, 1], dt), GGML_F32)
    f32("v.patch_embd.bias", V + "patch_embed.proj.bias")
    f32("v.position_embd.weight", V + "pos_embed.weight")
    for l in range(vc["depth"]):
        s, d = V + "blocks.%d." % l, "v.blk.%d." % l
        for hf, gg in (("norm1", "ln1"), ("norm2", "ln2")):
            f32(d + gg + ".weight", s + hf + ".weight")
            f32(d + gg + ".bias", s + hf + ".bias")
        for hf, gg in (("attn.qkv", "attn_qkv"), ("attn.proj", "attn_out"), ("mlp.linear_fc1", "ffn_up"),
                       ("mlp.linear_fc2", "ffn_down")):
            mat(d + gg + ".weight", s + hf + ".weight")
            f32(d + gg + ".bias", s + hf + ".bias")
    f32("v.post_ln.weight", V + "merger.norm.weight")
    f32("v.post_ln.bias", V + "merger.norm.bias")
    for hf, gg in (("linear_fc1", "mm.0"), ("linear_fc2", "mm.2")):
        mat(gg + ".weight", V + "merger." + hf + ".weight")
        f32(gg + ".bias", V + "merger." + hf + ".bias")

    used = {n for n, _, _ in w.tensors}
    print("%d tensors from %d checkpoint tensors" % (len(used), len(ck.where)))
    w.write(a.out)
    print("wrote", a.out, "%.2f GB" % (os.path.getsize(a.out) / 1e9))


if __name__ == "__main__":
    main()
