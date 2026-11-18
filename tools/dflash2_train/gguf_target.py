"""Frozen target pieces for DFlash2 training, read straight from an azllm target GGUF.

The DFlash2 trainer only needs the target's ``embed_tokens`` and ``lm_head``. Taking them
from the same quantized GGUF that azllm serves keeps training and inference on identical
weights, and avoids shipping the 54 GB BF16 checkpoint to the training box.

Pure numpy: header parsing, memmapped tensor access, and dequantization of the ggml types
the azllm target files use for these two tensors (F32/F16/BF16/Q4_K/Q6_K). Torch is only
imported by ``build_frozen_modules``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

GGML_F32, GGML_F16, GGML_Q4_K, GGML_Q6_K, GGML_BF16 = 0, 1, 12, 14, 30
QK_K = 256
_BLOCK_BYTES = {GGML_Q4_K: 144, GGML_Q6_K: 210}

_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


@dataclass
class TensorInfo:
    name: str
    shape: tuple[int, ...]  # ggml order: ne0 (row length) first
    ggml_type: int
    offset: int  # absolute file offset


class GGUFFile:
    def __init__(self, path: str):
        self.path = path
        self.kv: dict[str, object] = {}
        self.kv_types: dict[str, tuple[int, int | None]] = {}  # key -> (gguf value type, array element type)
        self.tensors: dict[str, TensorInfo] = {}
        with open(path, "rb") as f:
            self._parse(f)
        self._mm = np.memmap(path, dtype=np.uint8, mode="r")

    def _parse(self, f) -> None:
        def rd(fmt: str):
            n = struct.calcsize("<" + fmt)
            return struct.unpack("<" + fmt, f.read(n))[0]

        def rs() -> str:
            return f.read(rd("Q")).decode("utf-8", "replace")

        def val(t: int):
            if t == 8:
                return rs()
            if t == 9:
                at, n = rd("I"), rd("Q")
                return [val(at) for _ in range(n)]
            return rd(_SCALAR[t])

        if f.read(4) != b"GGUF":
            raise ValueError(f"{self.path}: not a GGUF file")
        version, n_tensors, n_kv = rd("I"), rd("Q"), rd("Q")
        if version < 2:
            raise ValueError(f"{self.path}: GGUF v{version} unsupported")
        for _ in range(n_kv):
            k = rs()
            t = rd("I")
            if t == 9:
                at, n = rd("I"), rd("Q")
                self.kv[k], self.kv_types[k] = [val(at) for _ in range(n)], (9, at)
            else:
                self.kv[k], self.kv_types[k] = val(t), (t, None)
        infos = []
        for _ in range(n_tensors):
            name = rs()
            nd = rd("I")
            shape = tuple(rd("Q") for _ in range(nd))
            infos.append((name, shape, rd("I"), rd("Q")))
        align = int(self.kv.get("general.alignment", 32))
        base = (f.tell() + align - 1) // align * align
        for name, shape, t, off in infos:
            self.tensors[name] = TensorInfo(name, shape, t, base + off)

    def raw_rows(self, name: str) -> tuple[TensorInfo, np.ndarray]:
        """Return (info, uint8 view [rows, row_bytes]) for a 2-D tensor."""
        ti = self.tensors[name]
        ne0 = ti.shape[0]
        rows = int(np.prod(ti.shape[1:])) if len(ti.shape) > 1 else 1
        if ti.ggml_type in _BLOCK_BYTES:
            if ne0 % QK_K:
                raise ValueError(f"{name}: row length {ne0} not a multiple of {QK_K}")
            row_bytes = ne0 // QK_K * _BLOCK_BYTES[ti.ggml_type]
        elif ti.ggml_type == GGML_F32:
            row_bytes = ne0 * 4
        elif ti.ggml_type in (GGML_F16, GGML_BF16):
            row_bytes = ne0 * 2
        else:
            raise NotImplementedError(f"{name}: ggml type {ti.ggml_type} not supported here")
        buf = self._mm[ti.offset : ti.offset + rows * row_bytes]
        return ti, buf.reshape(rows, row_bytes)

    def dequant(self, name: str, row_start: int = 0, row_end: int | None = None) -> np.ndarray:
        """Dequantize rows [row_start, row_end) of a tensor to float32 [rows, ne0]."""
        ti, raw = self.raw_rows(name)
        raw = raw[row_start:row_end]
        ne0 = ti.shape[0]
        t = ti.ggml_type
        if t == GGML_F32:
            return np.ascontiguousarray(raw).view(np.float32).reshape(-1, ne0)
        if t == GGML_F16:
            return np.ascontiguousarray(raw).view(np.float16).astype(np.float32).reshape(-1, ne0)
        if t == GGML_BF16:
            u = np.ascontiguousarray(raw).view(np.uint16).astype(np.uint32) << 16
            return u.view(np.float32).reshape(-1, ne0)
        blocks = np.ascontiguousarray(raw).reshape(-1, _BLOCK_BYTES[t])
        out = dequant_q4_k(blocks) if t == GGML_Q4_K else dequant_q6_k(blocks)
        return out.reshape(-1, ne0)


def _f16(b: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(b).view(np.float16).astype(np.float32).reshape(-1)


def dequant_q4_k(blocks: np.ndarray) -> np.ndarray:
    """ggml Q4_K: [nb, 144] -> float32 [nb, 256]."""
    nb = blocks.shape[0]
    d = _f16(blocks[:, 0:2])[:, None]
    dmin = _f16(blocks[:, 2:4])[:, None]
    s = blocks[:, 4:16].astype(np.uint8)
    qs = blocks[:, 16:144]
    # 8 sub-blocks of 32 values, 6-bit scale + 6-bit min each (get_scale_min_k4).
    sc = np.empty((nb, 8), np.uint8)
    mn = np.empty((nb, 8), np.uint8)
    sc[:, :4] = s[:, 0:4] & 63
    mn[:, :4] = s[:, 4:8] & 63
    sc[:, 4:] = (s[:, 8:12] & 0xF) | ((s[:, 0:4] >> 6) << 4)
    mn[:, 4:] = (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)
    q = qs.reshape(nb, 4, 32)
    lo, hi = q & 0xF, q >> 4  # sub-blocks 2j and 2j+1 of each 64-value chunk
    vals = np.stack([lo, hi], axis=2).reshape(nb, 8, 32).astype(np.float32)
    out = (d * sc.astype(np.float32))[:, :, None] * vals - (dmin * mn.astype(np.float32))[:, :, None]
    return out.reshape(nb, QK_K)


def dequant_q6_k(blocks: np.ndarray) -> np.ndarray:
    """ggml Q6_K: [nb, 210] -> float32 [nb, 256]."""
    nb = blocks.shape[0]
    ql = blocks[:, 0:128].reshape(nb, 2, 64)
    qh = blocks[:, 128:192].reshape(nb, 2, 32)
    sc = np.ascontiguousarray(blocks[:, 192:208]).view(np.int8).reshape(nb, 2, 8).astype(np.float32)
    d = _f16(blocks[:, 208:210])[:, None, None]
    a, b = ql[:, :, :32], ql[:, :, 32:]
    q1 = (a & 0xF) | (((qh >> 0) & 3) << 4)
    q2 = (b & 0xF) | (((qh >> 2) & 3) << 4)
    q3 = (a >> 4) | (((qh >> 4) & 3) << 4)
    q4 = (b >> 4) | (((qh >> 6) & 3) << 4)
    q = np.stack([q1, q2, q3, q4], axis=2).astype(np.float32) - 32.0  # [nb, 2, 4, 32]
    # scale index: is = l // 16, value group g uses sc[is + 2g]
    scale = sc.reshape(nb, 2, 4, 2)  # [half, group g, l//16]
    q = q.reshape(nb, 2, 4, 2, 16) * scale[..., None]
    return (d[..., None, None] * q).reshape(nb, QK_K)


def target_config(g: GGUFFile) -> dict:
    arch = g.kv["general.architecture"]
    return {
        "arch": arch,
        "hidden_size": int(g.kv[f"{arch}.embedding_length"]),
        # block_count includes the MTP (nextn) blocks; the decoder proper is what the drafter hooks.
        "num_layers": int(g.kv[f"{arch}.block_count"]) - int(g.kv.get(f"{arch}.nextn_predict_layers", 0)),
        "vocab_size": g.tensors["token_embd.weight"].shape[1],
        "rms_norm_eps": float(g.kv[f"{arch}.attention.layer_norm_rms_epsilon"]),
    }


def dequant_to_torch(g: GGUFFile, name: str, dtype, device, chunk_rows: int = 8192):
    """Dequantize a [rows, ne0] tensor in row chunks directly into a torch tensor."""
    import torch

    ti = g.tensors[name]
    rows = int(np.prod(ti.shape[1:]))
    out = torch.empty((rows, ti.shape[0]), dtype=dtype, device=device)
    for r0 in range(0, rows, chunk_rows):
        r1 = min(rows, r0 + chunk_rows)
        out[r0:r1] = torch.from_numpy(g.dequant(name, r0, r1)).to(device=device, dtype=dtype)
    return out


def build_frozen_modules(gguf_path: str, dtype=None, device="cpu"):
    """Return (embed_tokens: nn.Embedding, lm_head: nn.Linear, cfg) frozen, from the target GGUF.

    Falls back to the tied embedding when the file has no separate ``output.weight``.
    """
    import torch
    from torch import nn

    dtype = dtype or torch.bfloat16
    g = GGUFFile(gguf_path)
    cfg = target_config(g)
    V, H = cfg["vocab_size"], cfg["hidden_size"]
    embed = nn.Embedding(V, H, device="meta")
    embed.weight = nn.Parameter(dequant_to_torch(g, "token_embd.weight", dtype, device), requires_grad=False)
    head_name = "output.weight" if "output.weight" in g.tensors else "token_embd.weight"
    head = nn.Linear(H, V, bias=False, device="meta")
    head.weight = nn.Parameter(dequant_to_torch(g, head_name, dtype, device), requires_grad=False)
    return embed.requires_grad_(False), head.requires_grad_(False), cfg
