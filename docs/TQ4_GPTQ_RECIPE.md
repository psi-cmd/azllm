# TQ_L 配方：用 GPTQ 把 Qwen3.5 / Qwen3.8 稠密权重转成 TQ4 GGUF

这份文档说明 `tools/convert_qwen35_tq4_gptq.py` 做了什么、怎么跑、以及怎么换一套校准文本重调。

上游只发布了成品 `Qwen3.8-27B-TQ_L.gguf`，没有发布 TQ4 的 GPTQ 量化器和校准文本。这里的实现是按成品反推的：
张量类型和元数据照成品的文件头对齐，量化算法按成品码字的统计特征对齐（见“和官方成品的对照”）。

## 配方

张量类型逐条记录在 `tools/tq_l_recipe.json`（866 条，取自官方 TQ_L 的文件头），组装时逐条断言形状和类型。

| 张量 | 类型 | 来源 |
|---|---|---|
| 64 层主体矩阵：`attn_qkv` `attn_gate` `ssm_out` `attn_q/k/v/output` `ffn_gate/up/down` | TQ4（4.125 bpw） | GPTQ |
| `ssm_alpha` `ssm_beta` | BF16，按旋转后的基存储 | 直接转换 |
| 各种 norm、`ssm_a` `ssm_dt` `ssm_conv1d` | F32 | 直接转换 |
| `token_embd` | Q4_K | 从同权重的 Q4_K_M GGUF 逐字节复制，没有就现算 |
| `output` | Q6_K | 同上 |
| MTP 块 `blk.64` 和 `nextn.*` | Q6_K（`attn_k` `attn_v` 为 Q8_0），norm 为 F32 | 现算 |

TQ4 的格式：每行每 128 个权重一组，先乘固定的 Hadamard 旋转，再存一个 FP16 的 scale 和 128 个 4 bit 码字
（8 个幅度级 `6 18 31 44 58 75 96 127`，加符号位）。

## 量化算法

逐层进行，每层：

1. 用**已量化的前面各层**的输出作为这一层的输入，累计每个矩阵输入的 Hessian `H = XᵀX`。
2. 把权重和 Hessian 都转到旋转后的基：`W' = W Rᵀ`，`H' = R H Rᵀ`。
3. 对每个矩阵按列顺序做 GPTQ，128 列为一个更新块，正好对应一个 TQ4 分组。
4. 进入一个分组时，在“已经吸收了前面误差补偿”的权重上为每一行选 scale：
   在该组 RMS 的 0.75–1.25 倍之间取 21 个候选，选加权误差最小的。
5. 用量化后的这一层前向一次，得到下一层的输入。

同时有 16 段不参与校准的评测文本跟着走三遍（原始权重、GPTQ、纯 RTN），最后给出相对原模型的 KL 散度和首选 token 一致率。

## 准备

- **源权重**：HF 格式的检查点目录（`--source`，默认 `models/bf16-uncensored`）。
- **donor**：同一份权重的 Q4_K_M GGUF（`--donor`），用来复制 `token_embd`、`output` 和全部元数据。
  脚本会先核对 donor 的张量确实和源权重一致，不一致就改为现算。
- **Python 环境**：`torch`（带 CUDA）、`transformers` ≥ 5.20、`gguf`、`safetensors`。本机用的是 `~/.venvs/azllm-gptq`。
- **内置校准文本**：WikiText-2 原始文本，放到 `bench/data/wiki.train.raw` 和 `bench/data/wiki.test.raw`。
  来源是 `https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip`。
- **硬件**：一张 24 GB 的显卡即可，同一时刻只有一层在显存里。
  27B 模型在 4090 上每层约 85 秒，全程约 1.5 小时。纯 CPU 也能跑，但要 20 小时以上。

## 运行

```bash
python tools/convert_qwen35_tq4_gptq.py --fp-check --device cuda
```

先做这一步：用原始权重走一遍本脚本的前向，在 WikiText-2 测试集上应得到 NLL ≈ 1.94（困惑度 ≈ 6.9）。
数值离谱说明张量映射或前向有问题，不要继续。

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python tools/convert_qwen35_tq4_gptq.py --device cuda --batch 16 --threads 16
```

这是完整转换，结束时自动评测并组装 GGUF。可以随时中断，用同一条命令重跑会从下一层接着做。

- 中间结果在 `--stage`（默认 `models/tq4-gptq-stage`），成品在 `--output`。
- `--layers 1` 只跑第一层，用来看耗时和误差；`--compare-rtn` 额外打印每个矩阵的 RTN 误差作对照。
- `--assemble-only` 跳过量化，只用已有的中间结果重新组装 GGUF。

## 用自己的校准文本重调

内置校准集是 128 段 × 1024 token：60% WikiText-2 散文、25% 代码（Python 标准库和本仓库的 C++）、
15% 套了对话模板的段落。里面几乎没有中文，也不含你自己的使用场景。要补上，用 `--calib-text`：

```bash
python tools/convert_qwen35_tq4_gptq.py --device cuda --batch 16 --threads 16 --calib-text data/my_text --stage models/tq4-gptq-stage-zh --output models/my-TQ_L-zh.gguf
```

- `--calib-text PATH` 可以重复给多次。接受三种输入：
  - 纯文本文件，原样使用；
  - 目录，递归读取其中的 `.txt` `.md` `.jsonl`；
  - `.jsonl`，每行一个 `{"messages": [{"role": ..., "content": ...}, ...]}`，会套上模型自己的对话模板。
- `--calib-text-share`（默认 0.5）是自定义文本占校准段数的比例，内置的三类按原比例缩到剩下的份额。
  设成 1 就完全只用你的文本。
- 评测集里有四分之一会换成你的文本，日志里单列为 `custom`。默认取 `--calib-text` 的末尾几段并从校准里剔除；
  想用另一份留出文本就给 `--eval-text PATH`。
- 文本量不够时脚本会直接报出现有段数和需要的段数。默认配置下需要约 7 万 token（64 段校准 + 4 段评测）。
- **换校准文本必须换 `--stage`**。中间结果里记了校准集和评测集的哈希，对不上会拒绝续跑。

选文本的原则：

- 覆盖面比数量重要。GPTQ 把量化误差从校准文本里常见的输入方向挪到少见的方向，没覆盖到的用途得不到好处，
  最坏情况下比 RTN 略差（纯权重误差约高一成）。
- 所以应当放进真实使用时的输入类型：实际用的语言、带系统提示的对话、代码、工具调用格式。
- 数量上，13 万 token 相对 5120 维的输入已有 25 倍，继续加收益很小。

## 可调参数

| 参数 | 默认 | 作用 |
|---|---|---|
| `--calib-seqs` | 128 | 校准段数。翻倍会让 Hessian 更平滑，耗时里只有前向和 Hessian 部分随之增加 |
| `--seq-len` | 1024 | 每段的 token 数。超长上下文场景可以加到 2048，显存不够就同时调小 `--batch` |
| `--percdamp` | 0.01 | Hessian 对角阻尼，占平均对角元的比例。调大更接近 RTN、更稳；调小更贴合校准集 |
| `--scale-weight` | `hinv` | 选 scale 时按 `1/U_jj²` 加权；`none` 为不加权 |
| `--grid` | 21 | scale 的候选个数，覆盖 RMS 的 0.75–1.25 倍 |
| `--eval-chunks` | 16 | 评测段数。只影响最后 KLD 的统计噪声 |

一次只改一个参数，各用各的 `--stage`，用最后的 KLD 比较。`--layers 4` 加 `--compare-rtn` 可以在几分钟内先看趋势。

## 怎么读日志

每层一行：

```
layer 3: hidden rel err gptq 0.0220 rtn 0.0465 | fwd 4s hess 6s quant 30s rest 46s | proxy/wrel: attn_q 1.2e-03/0.104, ...
```

- `hidden rel err`：这一层输出的隐状态相对原模型的误差，是逐层累积的。GPTQ 应明显低于 RTN。
- `proxy`：单个矩阵在校准输入上的相对输出误差，`tr(ΔW H ΔWᵀ) / tr(W H Wᵀ)`。`ffn_down` 列数太多不算，显示 `nan`。
- `wrel`：纯权重相对误差，正常在 0.10–0.13。GPTQ 的这个值会略高于 RTN，这是预期的。
- 名字后面带 `!` 表示 Cholesky 分解失败后加大了阻尼，偶尔出现无妨，大面积出现说明校准文本太少或太单一。

最后几行是相对原模型的 KLD、首选 token 一致率和 NLL，GPTQ 和 RTN 各一行，并按文本类别分列，
同时写入 `<stage>/eval.json`。官方 TQ_L 公布的数字是 KLD 0.022、首选 token 一致率 93.7%，
但那是在它自己的评测文本上测的，只能作量级参考。

## 和官方成品的对照

`blk.3.attn_k.weight` 在去审查版和原版里权重完全相同，可以拿官方成品逐码字比较（内置校准集，默认参数）：

| 指标 | 官方 TQ_L | 本脚本 |
|---|---|---|
| 权重相对误差 | 0.0984 | 0.1006 |
| 非最近邻码字占比 | 10.3% | 11.4% |
| 其中差一级 / 符号翻转 | 87.6% / 12.4% | 87.7% / 12.3% |
| 非最近邻占比，前 128 列 → 后 1024 列 | 2.6% → 16.6% | 2.4% → 18.5% |
| scale / 分组 RMS 的均值、标准差 | 0.967、0.118 | 0.967、0.118 |

码字逐个一致的比例是 82.9%：第一个分组 97.2%，逐组降到最后一组的 70.2%。第一组不受误差补偿影响，
那里 scale 有 66% 与官方逐位相同，说明 scale 的选择准则和官方接近但不完全一样；后面各组的差别主要来自校准文本不同。

## 实测结果

Qwen3.8-27B Uncensored，内置校准集，默认参数，RTX 4090，全程约 1 小时 40 分。
评测为 16 段 × 1024 token（6 散文、5 代码、5 对话），相对原始权重：

| | KLD | 首选 token 一致率 | 散文 KLD | 代码 KLD | 对话 KLD |
|---|---|---|---|---|---|
| GPTQ | 0.0531 | 93.0% | 0.0162 | 0.0755 | 0.0750 |
| RTN | 0.1091 | 89.7% | 0.0369 | 0.1188 | 0.1861 |

成品 14.72 GB、866 个张量。用 `tqserve` 加载后中文、英文、代码三个提示都生成正常，约 61 token/s；
加 `--mtp` 后 180 个草稿 token 接受 166 个，约 200 token/s，说明 MTP 块可用。

## 已知的不确定之处

- 官方的校准文本、阻尼和 scale 选择准则都没有公开，以上是按成品反推的近似，做不到逐位相同。
- KLD 是在 PyTorch 里按层前向算的，不经过引擎。引擎侧只做了加载和生成的冒烟测试（见“实测结果”），
  没有在引擎里测 KLD。
- 只支持稠密的 Qwen3.5 / Qwen3.8 文本结构；视觉部分的张量不写入成品。

## RTX 4090 上下文 / 并行槽验收（2026-11-17）

Windows WDDM + WSL，GPTQ TQ_L 成品 14.72 GB；azllm 引擎 65430de，DFlash2 草稿模型 7 tokens。
tqserve 的 --context 是所有 --np 槽的总 token 数；/health.max_context 则是每槽窗口。
测试统一使用 --kv q4/tcq2 --kv-host 0（4-bit K、2-bit trellis V，KV 保留在 GPU）。
-fitl off 禁止影响输出的自动适配，但不禁止 embedding / LM head 自动转到主存。

| 槽数 | 总 context | 每槽 context | 检验结果 | GPU 内存 |
|---:|---:|---:|---|---|
| 1 | 262144 | 262144 | HTTP 200：261852 输入 token、1 输出，约 196 秒 | 占用约 21643 MiB，无卸载 |
| 2 | 262144 | 131072 | 两条并发均 HTTP 200，各 130852 输入、1 输出，约 102 秒 | 占用约 23039 MiB，无卸载 |
| 2 | 303104 | 151552 | 服务加载成功，未跑满长输入 | 剩余约 1146 MiB，无卸载 |
| 2 | 327680 | 163840 | 服务加载成功，未跑满长输入 | 剩余约 852 MiB，无卸载 |
| 2 | 335872 | 167936 | 服务加载成功，未跑满长输入 | 剩余约 692 MiB，无卸载 |
| 2 | 344064 | 172032 | 自动适配触发 embedding 转主存 | 已越过纯 GPU 的余量线 |
| 2 | 524288 | 262144 | 迁移 embedding / LM head、收缩 prefill 后可加载；压力测试未完成 | 不适合作为默认值 |

稳定默认选 2 × 131072。若想提高窗口可先尝试 2 × 163840（OpsLatch Total context 填 327680），
但上述较大配置仅测了加载，不能视为完整长上下文推理或性能验收。

同机 NInfer 默认配置：--max-context 131072（每请求）、--kv-capacity 262144（共享总量）、
--max-concurrency 2、--kv-dtype rk4v4-e8，另有 4 GiB pinned host KV。
启动成功的日志给出 runtime_reservation_bytes=6204130560，
available_after_weights_bytes=6429868032，剩余预算约 225.7 MB。
双槽各 147456（共享 KV 294912）启动时被拒绝：需要 6810211584 字节，
超出可用预算约 380 MB，故不建议在现有硬件和其他参数不变时扩张 NInfer 的双槽容量。

本机 OpsLatch 清单 C:/Users/longbang-cmd/projects/OpsLatch/opslatch.yaml 已暴露
NInfer 的并行数、每请求窗口和共享 KV 容量，以及 azllm 的并行槽数、总上下文。
编辑后通过 OpsLatch Reapply 应用。azllm 默认 2 槽，总上下文 262144，
KV 固定 q4/tcq2 且 --kv-host 0。Pi 配置 ~/.pi/agent/models.json 已与当前默认值
对齐（azllm contextWindow=131072）；通过 OpsLatch 改变窗口后，Pi 的静态模型描述
需要另行同步为实际每槽窗口。

Pi 客户端链路也已实测：从 WSL 运行 pi --print --no-session --no-tools
--provider azllm --model qwen3.8-27b-uc-tq --thinking off，
向 8086 端口的 OpsLatch 服务发起实际请求，返回 READY，退出码 0。
该请求之后 WDDM 报告显存只剩约 488 MiB，说明双槽 131K 日常配置已经接近
GPU 内存预算；较大窗口仅供专门的长上下文场景使用，并需在实际负载上复测。
