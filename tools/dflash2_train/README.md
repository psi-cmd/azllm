# DFlash2 drafter training on azllm

Trains a DFlash2 drafter for an azllm target, on the target's **quantized** weights, using the
unmodified upstream NeMo Automodel modules (`Qwen3DFlash2DraftModel` + `DFlash2TrainerModule`).

```
JSONL ids ──► tqfeatures (azllm: target GGUF + drafter layer list) ──► [P, 5·5120] BF16 features
                                                                         │
target GGUF ─► gguf_target.py: Q4_K embed / Q6_K head → frozen torch ────┤
                                                                         ▼
                                                  train.py (upstream model + loss) ─► model.safetensors
                                                                         │
                                         export_gguf.py (--like reference drafter GGUF) ─► drafter.gguf ─► azllm -md
```

| file | role |
|---|---|
| `../tqfeatures.cpp` | feature server: token-id lines on stdin → `TQF1` frames on stdout (one model load, many requests) |
| `features.py` | `FeatureServer` (one tqfeatures process), `FeaturePool` (N servers, prefetch), `FeatureCache` (on-disk replay) |
| `gguf_target.py` | numpy GGUF reader + Q4_K/Q6_K/F16/BF16 dequant; frozen `embed_tokens` / `lm_head` |
| `train.py` | training loop (AdamW FP32 master, BF16 autocast, cosine LR, grad accumulation, metrics.jsonl) |
| `export_gguf.py` | checkpoint → BF16 DFlash2 GGUF, metadata copied from a reference drafter for the same target |

## Verified (RTX 4090, 2026-11-17)
- `tqfeatures` frames are bit-identical to the golden `tqgolden` features (`case_short`, `case_prefill`); repeated requests are deterministic; bad ids / over-long requests get an empty frame.
- Dequantized embed/head vs. BF16 HF weights: cos ≈ 0.996 (Q4_K embed), ≈ 0.9998 (Q6_K head).
- Smoke run (1-layer draft, cache replay): loss falls, 13.9 GB peak; exported GGUF loads with `-md`, speculative greedy == plain greedy.

## Usage
```bash
# live features (target + trainer on one card)
python tools/dflash2_train/train.py --automodel PATH/Automodel \
  --target-gguf models/Qwen3.8-27B-Uncensored-TQ_L-GPTQ.gguf \
  --features-bin build/tqfeatures --drafter-gguf models/drafters/Qwen3.8-27B-DFlash2-TQ3.gguf \
  --data train.jsonl --out runs/dflash2 --steps 2000 [--init HF_DRAFTER_DIR]

# or two phases: fill a feature cache, then train without the target loaded
python tools/dflash2_train/train.py ... --feature-cache cache/ --dump-only
python tools/dflash2_train/train.py ... --feature-cache cache/            # no --features-bin

python tools/dflash2_train/export_gguf.py runs/dflash2/final -o drafter-bf16.gguf \
  --like models/drafters/Qwen3.8-27B-DFlash2-TQ3.gguf
```

Data lines: `{"input_ids": [...], "loss_mask": [...]}` or `{"prompt_ids": [...], "response_ids": [...]}`.
Responses should come from the target itself (teacher forcing on its own outputs).

`--automodel` must point at the full checkout; `train.py` skips the package `__init__`s that import
unrelated recipes and stubs HF `datasets` when it is not installed.

## Verified (Hygon BW1000 gfx936, DTK 26.04, 2026-11-17)
- Full 5-layer draft (1.924B params), sequences up to 3000 tokens: trainer peak **41.3 GB** (`max_memory_allocated`).
- Live mode (`--features-bin`, target + trainer on one 64 GB card) **OOMs** on a 3000-token sample in the
  248k-vocab logits. Use the two-phase split below on a single card.
- `--dump-only`: target load ≈ 50 s, then features at 5·5120·2 B ≈ 51 KB/token (≈ 51 GB per 1M tokens).
- Train from cache: frozen embed/head dequant 47 s; 1–7k tok/s on tiny smoke samples (noisy).
- Exported GGUF → `tqgolden -md` on BW: greedy ids/logits byte-identical to the stock-drafter golden run.

BW env: `source /opt/dtk-26.04/cuda/env.sh` plus `sga8-dtk2604/lib{,/gpufusion-bridge}` on `LD_LIBRARY_PATH`;
pass `--feature-args '--prefill-int8 off'` to match the golden runs. `expandable_segments` is unsupported on HIP here.

## Memory (full 5-layer draft, ~1.9B params)
FP32 weights + grads + AdamW ≈ 30 GB, frozen embed/head ≈ 5 GB, activations/logits push the peak to ≈ 41 GB at
3k tokens. Together with a resident tqfeatures target this exceeds 64 GB, so on one BW1000 run
`--dump-only` first and train from the cache; live features need a second card (or a smaller `--num-anchors`).
