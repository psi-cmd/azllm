#!/usr/bin/env python3
"""Export a trained DFlash2 drafter (train.py checkpoint) to a GGUF azllm loads with ``-md``.

Metadata (architecture keys, tokenizer, target layer list, conv/selector geometry) is copied from a
reference DFlash2 GGUF for the same target -- e.g. models/drafters/Qwen3.8-27B-DFlash2-TQ3.gguf --
then the keys the checkpoint actually determines are overwritten. Tensors are written as BF16
(matrices, codebooks) and F32 (norms, convolution base kernels); azllm quantizes on load.

Tensor naming follows the reference HF->GGUF converter (convert-dflash2-ref.py). numpy only.

  python tools/dflash2_train/export_gguf.py runs/dflash2/final -o drafter-bf16.gguf \
      --like models/drafters/Qwen3.8-27B-DFlash2-TQ3.gguf
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_target import GGML_BF16, GGML_F32, GGUFFile  # noqa: E402

DIRECT = {
    "fc.weight": "fc.weight",
    "hidden_norm.weight": "enc.output_norm.weight",
    "norm.weight": "output_norm.weight",
    "candidate_selector.hidden_projection.weight": "selector_hidden.weight",
    "candidate_selector.predecessor_codebook": "selector_predecessor.weight",
    "candidate_selector.successor_codebook": "selector_successor.weight",
}
PER_LAYER = {
    "input_layernorm.weight": "blk.{i}.attn_norm.weight",
    "post_attention_layernorm.weight": "blk.{i}.ffn_norm.weight",
    "self_attn.q_proj.weight": "blk.{i}.attn_q.weight",
    "self_attn.k_proj.weight": "blk.{i}.attn_k.weight",
    "self_attn.v_proj.weight": "blk.{i}.attn_v.weight",
    "self_attn.o_proj.weight": "blk.{i}.attn_output.weight",
    "self_attn.q_norm.weight": "blk.{i}.attn_q_norm.weight",
    "self_attn.k_norm.weight": "blk.{i}.attn_k_norm.weight",
    "mlp.gate_proj.weight": "blk.{i}.ffn_gate.weight",
    "mlp.up_proj.weight": "blk.{i}.ffn_up.weight",
    "mlp.down_proj.weight": "blk.{i}.ffn_down.weight",
    "attention_conv.base_kernel": "blk.{i}.attn_conv_base",
    "attention_conv.kernel_projection.weight": "blk.{i}.attn_conv_proj.weight",
    "mlp_conv.base_kernel": "blk.{i}.ffn_conv_base",
    "mlp_conv.kernel_projection.weight": "blk.{i}.ffn_conv_proj.weight",
}
F32_SUFFIXES = ("attention_conv.base_kernel", "mlp_conv.base_kernel")
ALIGN = 32
_FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def read_safetensors(path: Path):
    """Yield (name, float32 array, safetensors shape)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            s, e = meta["data_offsets"]
            f.seek(base + s)
            raw = f.read(e - s)
            if meta["dtype"] == "BF16":
                arr = (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32)
            elif meta["dtype"] == "F16":
                arr = np.frombuffer(raw, "<f2").astype(np.float32)
            elif meta["dtype"] == "F32":
                arr = np.frombuffer(raw, "<f4")
            else:
                raise ValueError(f"{name}: dtype {meta['dtype']}")
            yield name, arr.reshape(meta["shape"])


def gguf_name(hf: str) -> str:
    if hf in DIRECT:
        return DIRECT[hf]
    if hf.startswith("layers."):
        _, idx, rest = hf.split(".", 2)
        if rest in PER_LAYER:
            return PER_LAYER[rest].format(i=int(idx))
    raise KeyError(f"unmapped tensor {hf}")


class Writer:
    def __init__(self):
        self.kv: list[bytes] = []
        self.n_kv = 0

    @staticmethod
    def _s(x: str) -> bytes:
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    def _val(self, t: int, v) -> bytes:
        if t == 8:
            return self._s(v)
        return struct.pack("<" + _FMT[t], v)

    def add(self, key: str, t: int, v, sub: int | None = None):
        out = self._s(key) + struct.pack("<I", t)
        if t == 9:
            out += struct.pack("<IQ", sub, len(v)) + b"".join(self._val(sub, x) for x in v)
        else:
            out += self._val(t, v)
        self.kv.append(out)
        self.n_kv += 1

    def write(self, path: Path, tensors: list[tuple[str, np.ndarray, int]]):
        infos, offs, off = [], [], 0
        for name, arr, gtype in tensors:
            dims = list(reversed(arr.shape))  # ggml: ne0 = innermost
            infos.append(self._s(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
                         + struct.pack("<IQ", gtype, off))
            offs.append(off)
            nbytes = arr.size * (4 if gtype == GGML_F32 else 2)
            off += (nbytes + ALIGN - 1) // ALIGN * ALIGN
        with open(path, "wb") as f:
            f.write(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), self.n_kv))
            for kv in self.kv:
                f.write(kv)
            for info in infos:
                f.write(info)
            f.write(b"\0" * ((-f.tell()) % ALIGN))
            base = f.tell()
            for (name, arr, gtype), o in zip(tensors, offs):
                f.write(b"\0" * (base + o - f.tell()))
                if gtype == GGML_F32:
                    f.write(np.ascontiguousarray(arr, np.float32).tobytes())
                else:  # BF16, round to nearest even
                    u = np.ascontiguousarray(arr, np.float32).view(np.uint32)
                    f.write(((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2").tobytes())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path, help="checkpoint dir (model.safetensors + config.json)")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--like", type=Path, required=True, help="reference DFlash2 GGUF for the same target")
    args = ap.parse_args()

    cfg = json.loads((args.ckpt / "config.json").read_text())
    dcfg = cfg["dflash_config"]
    ref = GGUFFile(str(args.like))
    if ref.kv.get("general.architecture") != "dflash":
        raise SystemExit(f"{args.like}: not a dflash GGUF")

    L = int(cfg["num_hidden_layers"])
    overrides = {  # key -> (type, value, array subtype)
        "general.name": (8, f"{ref.kv.get('general.name', 'DFlash2')}-azllm-trained", None),
        "general.file_type": (4, 32, None),  # MOSTLY_BF16
        "dflash.block_count": (4, L, None),
        "dflash.feed_forward_length": (4, int(cfg["intermediate_size"]), None),
        "dflash.attention.head_count": (4, int(cfg["num_attention_heads"]), None),
        "dflash.attention.head_count_kv": (4, int(cfg["num_key_value_heads"]), None),
        "dflash.attention.key_length": (4, int(cfg["head_dim"]), None),
        "dflash.attention.value_length": (4, int(cfg["head_dim"]), None),
        "dflash.block_size": (4, int(dcfg["block_size"]), None),
        "dflash.conv_kernel_size": (4, int(dcfg["conv_kernel_size"]), None),
        "dflash.conv_group_size": (4, int(dcfg["conv_group_size"]), None),
        "dflash.selector_rank": (4, int(dcfg["selector_rank"]), None),
        "dflash.selector_top_k": (4, int(dcfg["selector_top_k"]), None),
        # GGUF stores layer-input hook indices; HF target_layer_ids are layer outputs.
        "dflash.target_layers": (9, [int(i) + 1 for i in dcfg["target_layer_ids"]], 4),
    }
    if cfg.get("use_sliding_window") and cfg.get("sliding_window"):
        overrides["dflash.attention.sliding_window"] = (4, int(cfg["sliding_window"]), None)
        overrides["dflash.attention.sliding_window_pattern"] = (9, [True] * L, 7)
    drop = {"general.quantization.tqllm"}

    w = Writer()
    for k, v in ref.kv.items():
        if k in drop or k in overrides:
            continue
        t, sub = ref.kv_types[k]
        w.add(k, t, v, sub)
    for k, (t, v, sub) in overrides.items():
        if k in ref.kv_types and t != 9 and ref.kv_types[k][0] != t:
            t = ref.kv_types[k][0]  # keep the reference's integer width
        w.add(k, t, v, sub)

    tensors = []
    for shard in sorted(args.ckpt.glob("*.safetensors")):
        for hf, arr in read_safetensors(shard):
            gtype = GGML_F32 if arr.ndim == 1 or hf.endswith(F32_SUFFIXES) else GGML_BF16
            tensors.append((gguf_name(hf), arr, gtype))
    tensors.sort(key=lambda x: x[0])
    have = {n for n, _, _ in tensors}
    missing = sorted(set(ref.tensors) - have) if L == ref.kv.get("dflash.block_count") else []
    if missing:
        raise SystemExit(f"checkpoint lacks tensors the reference has: {missing[:8]}")
    for name, arr, _ in tensors:
        r = ref.tensors.get(name)
        if r is not None and tuple(reversed(arr.shape)) != tuple(r.shape):
            raise SystemExit(f"{name}: shape {tuple(reversed(arr.shape))} != reference {r.shape}")
    w.write(args.out, tensors)
    print(f"wrote {len(tensors)} tensors, {w.n_kv} KVs -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
