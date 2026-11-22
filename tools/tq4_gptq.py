#!/usr/bin/env python3
"""TQ4 quantisation with GPTQ error feedback in the rotated basis, plus the Q4_K / Q6_K packers the TQ_L recipe needs.

TQ4 (GGUF type 300): per 128 weights of a row, an FP16 scale then 64 bytes of 4-bit codes (low nibble = even column).
w_rot = LEVEL[code] * scale, w = w_rot R per 128 columns, R = H diag(s1) / sqrt(128) (orthogonal).

GPTQ here works on W' = W Rb^T (Rb = blockdiag(R)) against the rotated Hessian H' = Rb H Rb^T, since
W x = W' (Rb x). Columns are quantised in natural order, 128 at a time (one TQ4 group = one lazy-update block); the
scale of a (row, group) is fixed when its group starts, searched on a grid around the RMS of the error-compensated
weights. This matches what the released Qwen3.8-27B TQ_L file shows (codes off the nearest level by exactly one step,
more often towards the last columns; scales within 0.75 .. 1.25 x the block RMS).
"""
from __future__ import annotations

import numpy as np
import torch

TQ4 = 300
TQ4_F = 127.0 / 2.7326  # engine level = F x Lloyd-Max level
_MAG = [6.0, 18.0, 31.0, 44.0, 58.0, 75.0, 96.0, 127.0]  # code c: magnitude _MAG[c & 7], negative when c & 8
_S1_WORDS = (0xE46A0359, 0x42F85949, 0xA4C63BBB, 0x49F250A9)


def _rotation() -> torch.Tensor:
    s1 = torch.tensor([1 - 2 * (_S1_WORDS[i // 32] >> (i % 32) & 1) for i in range(128)], dtype=torch.float64)
    h = torch.ones((1, 1), dtype=torch.float64)
    while h.shape[0] < 128:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h * s1[None, :] / 128.0 ** 0.5


R64 = _rotation()


def rotate_cols(w: torch.Tensor) -> torch.Tensor:
    """W -> W Rb^T (the basis TQ4 codes live in; also how dense matrices sharing a TQ input are stored)"""
    rows, cols = w.shape
    return (w.reshape(rows, cols // 128, 128) @ R64.to(w).T).reshape(rows, cols)


def unrotate_cols(w: torch.Tensor) -> torch.Tensor:
    rows, cols = w.shape
    return (w.reshape(rows, cols // 128, 128) @ R64.to(w)).reshape(rows, cols)


def rotate_hessian(h: torch.Tensor) -> torch.Tensor:
    """H -> Rb H Rb^T"""
    n = h.shape[0]
    r = R64.to(h)
    h = (r @ h.reshape(n // 128, 128, n)).reshape(n, n)
    return (h.reshape(n, n // 128, 128) @ r.T).reshape(n, n)


def _quant(t: torch.Tensor, s: torch.Tensor, mag: torch.Tensor, mids: torch.Tensor):
    """nearest level of t at engine scale s (broadcastable) -> (quantised value, code uint8)"""
    idx = torch.bucketize(t.abs() / s, mids)
    q = mag[idx] * s
    neg = t < 0
    return torch.where(neg, -q, q), (idx + 8 * neg).to(torch.uint8)


def _fp16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float16).to(x.dtype)


def choose_scale(w1: torch.Tensor, weight: torch.Tensor | None, mag, mids, grid: int = 21) -> torch.Tensor:
    """engine scale per row for one 128-group: the grid point f x RMS (f in 0.75 .. 1.25) with the least (weighted)
    round-to-nearest error. Returned already rounded to FP16."""
    rms = w1.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-12)
    best_e = best_s = None
    for f in torch.linspace(0.75, 1.25, grid).tolist():
        s = _fp16(rms * (f / TQ4_F)).clamp_min(1e-12)
        q, _ = _quant(w1, s, mag, mids)
        e = (q - w1).pow(2)
        e = (e * weight).sum(1, keepdim=True) if weight is not None else e.sum(1, keepdim=True)
        if best_e is None:
            best_e, best_s = e, s
        else:
            m = e < best_e
            best_e, best_s = torch.where(m, e, best_e), torch.where(m, s, best_s)
    return best_s


@torch.no_grad()
def tq4_quantize(w: torch.Tensor, h: torch.Tensor | None = None, percdamp: float = 0.01, scale_weight: str = "hinv",
                 grid: int = 21, device: str = "cpu"):
    """w [rows][cols] in the original basis; h [cols][cols] = sum x x^T in the original basis (None: plain RTN).
    -> codes uint8 [rows][cols], engine scales float16 [rows][cols / 128], dequantised w (original basis, float32),
       stats dict (proxy = tr(D H D^T) / tr(W H W^T), the relative output error on the calibration inputs)."""
    rows, cols = w.shape
    assert cols % 128 == 0
    g = cols // 128
    wr = rotate_cols(w.to(device, torch.float32))
    w0 = wr.clone()
    mag = torch.tensor(_MAG, dtype=torch.float32, device=device)
    mids = (mag[1:] + mag[:-1]) / 2
    codes = torch.empty((rows, cols), dtype=torch.uint8, device=device)
    scales = torch.empty((rows, g), dtype=torch.float32, device=device)
    q_all = torch.empty_like(wr)
    stats = {}
    if h is None:
        for b in range(g):
            i1, i2 = b * 128, b * 128 + 128
            s = choose_scale(wr[:, i1:i2], None, mag, mids, grid)
            q_all[:, i1:i2], codes[:, i1:i2] = _quant(wr[:, i1:i2], s, mag, mids)
            scales[:, b] = s[:, 0]
    else:
        hr = rotate_hessian(h.to(device, torch.float64))
        diag = torch.diagonal(hr)
        dead = diag <= 0
        if dead.any():
            diag[dead] = 1.0
            wr[:, dead] = 0
        h_keep = hr.clone() if cols <= 8192 else None  # for the proxy; skipped on the widest matrices to bound memory
        damp = percdamp * diag.mean()
        diag += damp  # in place: diag is a view
        try:
            u = torch.linalg.cholesky(hr)
        except Exception:  # not positive definite at this damping: raise it
            torch.diagonal(hr).add_(9 * damp)
            u = torch.linalg.cholesky(hr)
            stats["redamped"] = True
        del hr
        u = torch.cholesky_inverse(u)
        u = torch.linalg.cholesky(u, upper=True).to(torch.float32)
        ud = torch.diagonal(u).clone()
        for b in range(g):
            i1, i2 = b * 128, b * 128 + 128
            w1 = wr[:, i1:i2].clone()
            u1 = u[i1:i2, i1:i2]
            d = ud[i1:i2]
            wt = None
            if scale_weight == "hinv":
                wt = 1.0 / d.pow(2)
                wt = (wt / wt.mean())[None, :]
            s = choose_scale(w1, wt, mag, mids, grid)
            err = torch.empty_like(w1)
            for j in range(128):
                q, c = _quant(w1[:, j], s[:, 0], mag, mids)
                q_all[:, i1 + j] = q
                codes[:, i1 + j] = c
                e = (w1[:, j] - q) / d[j]
                if j < 127:
                    w1[:, j + 1:].addr_(e, u1[j, j + 1:], alpha=-1.0)
                err[:, j] = e
            scales[:, b] = s[:, 0]
            if i2 < cols:
                wr[:, i2:].addmm_(err, u[i1:i2, i2:], alpha=-1.0)
        del u
        if h_keep is not None:
            dlt = (q_all - w0).to(torch.float64)
            stats["proxy"] = float(((dlt @ h_keep) * dlt).sum() / ((w0.to(torch.float64) @ h_keep) * w0).sum().clamp_min(1e-30))
    stats["wrel"] = float((q_all - w0).pow(2).sum().sqrt() / w0.pow(2).sum().sqrt().clamp_min(1e-30))
    return codes.cpu(), scales.to(torch.float16).cpu(), unrotate_cols(q_all).cpu(), stats


def rtn_proxy(w: torch.Tensor, h: torch.Tensor) -> float:
    """the same proxy for plain round to nearest (to compare against GPTQ on one matrix)"""
    _, _, wq, _ = tq4_quantize(w, None)
    w = w.to(h.device)
    dlt = (wq.to(h.device) - w.float()).to(torch.float64)
    h = h.to(torch.float64)
    return float(((dlt @ h) * dlt).sum() / ((w.to(torch.float64) @ h) * w).sum())


def tq4_pack(codes: torch.Tensor, scales: torch.Tensor) -> np.ndarray:
    """codes uint8 [rows][cols], scales float16 [rows][cols / 128] -> bytes [rows][cols / 128][66]"""
    rows, cols = codes.shape
    c = codes.numpy().reshape(rows, cols // 128, 64, 2)
    nib = (c[..., 0] | (c[..., 1] << 4)).astype(np.uint8)
    sc = scales.numpy().astype(np.float16).view(np.uint8).reshape(rows, cols // 128, 2)
    return np.concatenate([sc, nib], axis=-1)


def tq4_unpack(packed: np.ndarray) -> torch.Tensor:
    """bytes [rows][groups][66] -> dequantised weights in the original basis, float32 [rows][groups * 128]"""
    rows, g, _ = packed.shape
    sc = torch.from_numpy(packed[..., :2].copy().view(np.float16).astype(np.float32))
    nib = packed[..., 2:]
    codes = torch.from_numpy(np.stack([nib & 15, nib >> 4], -1).reshape(rows, g, 128).astype(np.int64))
    lv = torch.tensor(_MAG + [-m for m in _MAG], dtype=torch.float32)
    return unrotate_cols((lv[codes] * sc).reshape(rows, g * 128))


# ---------------------------------------------------------------- K-quants (ggml block layouts, own scale search) ---

def _nearest_int(x):
    return torch.round(x)


@torch.no_grad()
def q6k_quantize(x: torch.Tensor) -> np.ndarray:
    """x float32 [rows][cols], cols % 256 == 0 -> Q6_K bytes [rows][cols / 256 * 210]"""
    rows, cols = x.shape
    assert cols % 256 == 0
    xb = x.reshape(-1, 16, 16).float()  # [blocks][16 sub-blocks][16]
    amax_i = xb.abs().argmax(-1, keepdim=True)
    mx = xb.gather(-1, amax_i)  # signed value of the largest magnitude
    wgt = xb.pow(2)
    best_s = torch.zeros_like(mx)
    best_e = torch.full_like(mx, float("inf"))
    safe = mx.abs() > 1e-30
    mxs = torch.where(safe, mx, torch.ones_like(mx))
    for k in range(-9, 10):
        isc = -(32 + 0.1 * k) / mxs
        lq = torch.clamp(_nearest_int(isc * xb), -32, 31)
        den = (wgt * lq * lq).sum(-1, keepdim=True)
        sc = (wgt * xb * lq).sum(-1, keepdim=True) / den.clamp_min(1e-30)
        e = (wgt * (sc * lq - xb).pow(2)).sum(-1, keepdim=True)
        m = e < best_e
        best_e, best_s = torch.where(m, e, best_e), torch.where(m, sc, best_s)
    best_s = torch.where(safe, best_s, torch.zeros_like(best_s))[..., 0]  # [blocks][16]
    mi = best_s.abs().argmax(-1, keepdim=True)
    max_scale = best_s.gather(-1, mi)
    ok = max_scale.abs() > 1e-30
    iscale = torch.where(ok, -128.0 / torch.where(ok, max_scale, torch.ones_like(max_scale)), torch.zeros_like(max_scale))
    sq = torch.clamp(_nearest_int(iscale * best_s), -128, 127)
    d = torch.where(ok, 1.0 / torch.where(ok, iscale, torch.ones_like(iscale)), torch.zeros_like(iscale))
    d = d.to(torch.float16).float()
    dd = (d * sq)[..., None]
    lq = torch.where(dd != 0, torch.clamp(_nearest_int(xb / torch.where(dd != 0, dd, torch.ones_like(dd))), -32, 31),
                     torch.zeros_like(xb)) + 32
    lq = lq.to(torch.uint8).reshape(-1, 2, 4, 32).numpy()  # [blocks][half][q1..q4][32]
    nb = lq.shape[0]
    ql = np.empty((nb, 2, 2, 32), dtype=np.uint8)
    ql[:, :, 0] = (lq[:, :, 0] & 0xF) | ((lq[:, :, 2] & 0xF) << 4)
    ql[:, :, 1] = (lq[:, :, 1] & 0xF) | ((lq[:, :, 3] & 0xF) << 4)
    qh = (lq[:, :, 0] >> 4) | ((lq[:, :, 1] >> 4) << 2) | ((lq[:, :, 2] >> 4) << 4) | ((lq[:, :, 3] >> 4) << 6)
    out = np.concatenate([ql.reshape(nb, 128), qh.reshape(nb, 64), sq.to(torch.int8).numpy().view(np.uint8),
                          d.numpy().astype(np.float16).view(np.uint8).reshape(nb, 2)], axis=1)
    return out.reshape(rows, -1)


@torch.no_grad()
def q4k_quantize(x: torch.Tensor) -> np.ndarray:
    """x float32 [rows][cols], cols % 256 == 0 -> Q4_K bytes [rows][cols / 256 * 144]"""
    rows, cols = x.shape
    assert cols % 256 == 0
    xb = x.reshape(-1, 8, 32).float()  # [blocks][8 sub-blocks][32]
    av = xb.pow(2).mean(-1, keepdim=True).sqrt()
    wgt = av + xb.abs()
    mn = xb.min(-1, keepdim=True).values.clamp_max(0)
    mxv = xb.max(-1, keepdim=True).values
    rng = (mxv - mn).clamp_min(1e-30)
    sw = wgt.sum(-1, keepdim=True)
    swx = (wgt * xb).sum(-1, keepdim=True)
    best_e = torch.full_like(mn, float("inf"))
    best_s = torch.zeros_like(mn)
    best_m = mn.clone()
    for k in range(0, 21):
        isc = (-1.0 + 0.1 * k + 15) / rng
        lq = torch.clamp(_nearest_int(isc * (xb - mn)), 0, 15)
        swl = (wgt * lq).sum(-1, keepdim=True)
        swl2 = (wgt * lq * lq).sum(-1, keepdim=True)
        swxl = (wgt * xb * lq).sum(-1, keepdim=True)
        det = sw * swl2 - swl * swl
        okd = det > 0
        dets = torch.where(okd, det, torch.ones_like(det))
        sc = (sw * swxl - swx * swl) / dets
        m0 = (swl2 * swx - swl * swxl) / dets
        pos = m0 > 0  # the offset must be <= 0: refit the scale alone
        sc = torch.where(pos, swxl / swl2.clamp_min(1e-30), sc)
        m0 = torch.where(pos, torch.zeros_like(m0), m0)
        e = (wgt * (sc * lq + m0 - xb).pow(2)).sum(-1, keepdim=True)
        take = okd & (e < best_e)
        best_e, best_s, best_m = torch.where(take, e, best_e), torch.where(take, sc, best_s), torch.where(take, m0, best_m)
    scale = best_s[..., 0].clamp_min(0)  # [blocks][8]
    mins = (-best_m[..., 0]).clamp_min(0)
    max_scale = scale.max(-1, keepdim=True).values
    max_min = mins.max(-1, keepdim=True).values
    inv_s = torch.where(max_scale > 0, 63.0 / max_scale.clamp_min(1e-30), torch.zeros_like(max_scale))
    inv_m = torch.where(max_min > 0, 63.0 / max_min.clamp_min(1e-30), torch.zeros_like(max_min))
    ls = torch.clamp(_nearest_int(inv_s * scale), 0, 63)
    lm = torch.clamp(_nearest_int(inv_m * mins), 0, 63)
    d = (max_scale / 63.0).to(torch.float16).float()
    dmin = (max_min / 63.0).to(torch.float16).float()
    ds = (d * ls)[..., None]
    dm = (dmin * lm)[..., None]
    lq = torch.where(ds != 0, torch.clamp(_nearest_int((xb + dm) / torch.where(ds != 0, ds, torch.ones_like(ds))), 0, 15),
                     torch.zeros_like(xb)).to(torch.uint8).numpy()
    nb = lq.shape[0]
    ls8, lm8 = ls.to(torch.uint8).numpy(), lm.to(torch.uint8).numpy()
    sc12 = np.zeros((nb, 12), dtype=np.uint8)
    sc12[:, 0:4] = ls8[:, 0:4] | ((ls8[:, 4:8] >> 4) << 6)
    sc12[:, 4:8] = lm8[:, 0:4] | ((lm8[:, 4:8] >> 4) << 6)
    sc12[:, 8:12] = (ls8[:, 4:8] & 0xF) | ((lm8[:, 4:8] & 0xF) << 4)
    l4 = lq.reshape(nb, 4, 2, 32)
    qs = (l4[:, :, 0] | (l4[:, :, 1] << 4)).reshape(nb, 128)
    out = np.concatenate([d.numpy().astype(np.float16).view(np.uint8).reshape(nb, 2),
                          dmin.numpy().astype(np.float16).view(np.uint8).reshape(nb, 2), sc12, qs], axis=1)
    return out.reshape(rows, -1)
