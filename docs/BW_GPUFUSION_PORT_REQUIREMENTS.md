# BW / GPUFusion 适配需求：将编译探针恢复为真实 GPU 实现

状态：**代码与 gfx936 设备指令生成已实现；待 BW1000 真机前向验收**  
目标：海光 BW（优先 BW1000）上运行 azllm 的量化模型前向，为 DFlash2 提供可信的目标模型特征。  
基线：azllm commit ff03db8；FlagOS Hygon DTK 26.10 镜像 harbor.baai.ac.cn/flagos-base/flagos-base-hygon-dtk26.10:2.3.0。

## 1. 已验证事实和未验证事项

2026-11-17，在没有 BW 真机的 WSL/Docker 环境中完成了以下交叉编译实验：

1. DTK 容器包含海光 DCC nvcc（CUDA 12.9 兼容接口）、hipcc、定制 Clang。原始 CUDA AXPY、cuBLAS 调用和 CMake CUDA 最小工程均编译成功。
2. 原始 azllm 的 tqserve CMake 配置成功，但源码编译失败，首个错误位于 kernels/common.cuh:15：invalid output constraint '+f' in asm；指向 NVIDIA Tensor Core mma.sync 内联 PTX。
3. 仅在独立 Git 导出副本中，将四个文件中的 **38 处内联 asm 语句**替换为 __builtin_trap()，用于探测后续编译问题。所有 C++ / CUDA 编译单元随后通过。
4. 诊断副本的最终链接遇到 __hip_fatbin 符号与位置无关代码问题。只在副本内启用 CMAKE_POSITION_INDEPENDENT_CODE ON，并显式链接 DTK librccl.so.1，随后 **100% 编译和链接成功**。
5. 诊断 ELF 能显示 tqserve --help，但容器没有 BW 驱动，未执行 GPU 算子，未验证模型数值。**含 __builtin_trap() 的诊断产物不能用于推理或生成特征。**

因此，已证实 GPUFusion 可以构建 azllm 的常规 CUDA/C++ 工程骨架，**没有证实量化前向、hidden states、DFlash2 训练特征或推理性能已经兼容**。

实验资料存储在独立目录 /home/psi-cmd/projects/azllm-bw-crosscompile（不属于本仓库）：

- README.md：实验条件、步骤与结果。
- out/azllm.log：原版首次失败日志。
- out/probe-pic.log：PTX 被探针替换、PIC + RCCL 链接成功的日志。
- azllm-probe-src/：只用于编译诊断的独立源码副本。
- out/azllm-compile-probe/tqserve：**不得部署**的诊断二进制。

## 2. 目标和实施边界

### 必须实现

- 为需要执行的 NVIDIA 专用 PTX 操作提供 BW 上**有真实计算语义**的实现。可以更换整个 kernel 计算路径，不要求逐语句翻译；不能留下 __builtin_trap()、空操作或任意常量输出作为“兼容实现”。
- 通过 GPUFusion/DTK 编译、设备代码生成与链接，实际指定目标 BW 架构；不能仅依赖编译器在无卡环境中选择的默认 gfx 架构。
- 在 BW 真机对必要 kernel、TQ_L-GPTQ teacher 前向、logits、DFlash2 所用 hidden states 做对照测试。
- 现有 NVIDIA/CUDA 路径保持有效，不损害其量化格式和推测解码一致性。

### 暂不强制

- 在 BW 上达到 NVIDIA 的相同吞吐量，或一次性移植全部手工优化的低比特矩阵算子。
- 支持本阶段无关的全部模型、量化格式和多模态路径。
- 为跨平台兼容引入额外的大型框架或无关的安全/管理脚手架。

若 BW 无某个专用指令的直接替代，可先采用正确但较慢的 HIP / C++ 计算实现或分块反量化 + GEMM。只要该路径的数值满足目标 teacher 契约，就不必为形式上的“原生指令对等”付费。

## 3. 38 处汇编探针对应的代码范围

以下处数来自编译探针替换计数，是 **asm 语句出现次数**，不是行数，也不是跨架构问题的全部数量。

| 源文件 | 探针处数 | 被屏蔽的主要语义 | 对应任务 |
|---|---:|---|---|
| kernels/common.cuh | 8 | FP16、INT8 MMA；FP16x2 算术；ldmatrix / 转置加载 | 恢复共享矩阵片段与打包运算 helper |
| kernels/quant.cu | 5 | FP16 MMA；FP16x2 mul/add/fma | 恢复量化矩阵运算和累加语义 |
| kernels/attention.cu | 23 | BF16/FP16/INT8 MMA；ldmatrix；cp.async；FP16x2 | 恢复 attention 计算和异步搬运次序 |
| kernels/formats/tq4.cu | 2 | prmt.b32 字节排列；TQ4 的 mma.sync | 恢复量化展开及 TQ4 GEMM |
| **合计** | **38** | | |

**A. 矩阵计算与加载（mma.sync / ldmatrix）**

- 不得假定 NVIDIA m16n8k16、m16n8k32 与 BW 的矩阵指令拥有相同的寄存器片段布局；需要恢复每线程输入、输出、累加器与矩阵维度的实际对应。
- 分开验证 FP16、BF16、INT8 输入及 FP16 / FP32 / INT32 累加变体。
- ldmatrix 与 .trans 应重建 shared-memory 布局、lane-to-element 映射和转置语义；普通 load 只有在寄存器结果一致时才能直接替代。
- 可采用 DTK 支持的矩阵原语，或先用正确的块 GEMM 实现回退。

**B. 打包低精度算术与字节排列（f16x2 / prmt.b32）**

- mul.rn.f16x2、add.rn.f16x2、fma.rn.f16x2 必须体现两个 half lane 的计算和舍入规则，不能未经测试便以单次 FP32 运算替代。
- prmt.b32 出现在 kernels/formats/tq4.cu 的 tq4k::expand_u；要先确认当前 GPTQ teacher 调用路径是否会执行该分支，再确定移植优先级。
- HIPIFY 源码层面还留下 __byte_perm、__vadd4、__vsub4、__dp4a、BF16 转换以及 __ldg 等 CUDA 内建函数。即便 GPUFusion 能编译，也需验证它们在实际目标设备上的行为。编译探针的 38 处**不包含**这些潜在语义问题。

**C. 异步拷贝与同步（cp.async）**

- kernels/attention.cu 的 cp.async.ca / cp.async.cg、commit_group、wait_group 组成 shared-memory 流水线；必须保持缓冲生命周期及读写同步关系。
- 若 BW 不支持等价异步搬运，允许换为显式 load/store 加正确的同步；先消除未就绪读取、竞态和死锁，再考虑性能优化。

**D. 32-lane 逻辑 warp 与真实 wavefront**

- 仓库还使用 threadIdx.x % 32、/ 32、__shfl_xor_sync(0xffffffffu, ...)、32-lane 归约与固定 tile 排列。
- BW 的实际 wavefront 可能不同；不能把代码中的 32 全部替换成 64。需验证能否在实际 wavefront 中保持原来的**逻辑 32-lane 子组**语义，包括 mask、lane 映射、warp 同步和归约。
- 相关代码不仅存在于以上四个文件，也存在于 kernels/qwen35.cu 等文件。消除全部内联 PTX 不等于波前语义已移植。

## 4. 编译、链接要求

原 CMakeLists.txt 已通过 enable_language(CUDA)、find_package(CUDAToolkit) 和 CUDA::cudart/cublas/cublasLt/cusolver 接入 GPUFusion。优先在既有构建入口中提供明确可选择的 BW 构建配置，保留 NVIDIA 默认路径。

本次实验还证明：

- GPUFusion 能直接编译 .cu，无需把源码全部 HIPIFY。
- 原始 azllm 配置成功，首个编译障碍是 NVIDIA 专用 PTX。
- DTK 诊断对象使用了 __hip_fatbin，实验中这个符号由 librccl.so.1 导出；添加该库跨过第一处链接障碍，但进一步暴露 PIE/绝对重定位问题。
- 在副本中设置 CMAKE_POSITION_INDEPENDENT_CODE ON 并显式链接 RCCL 后，最终链接成功。**这不证明正式实现应以 RCCL 作为 fatbin 注册库依赖。** 应调查 GPUFusion/DTK 的正确运行时链接规则，避免把诊断参数未经解释地固化为生产要求。

编译验收需记录实际 DTK/DCC 版本、CMake 参数、目标架构、最终链接库及设备代码检查结果。初步关注 gfx936，但必须以实际租用 BW 型号为准；编译器若默认生成多个其他 gfx 目标不能算作 BW 目标代码已验证。

## 5. 验收顺序：先 teacher，再性能

目标 workload 是 **OrcaRouter Qwen3.8-27B-Uncensored 的 TQ_L-GPTQ 量化部署模型**，优先获得与 RTX 4090 上的 azllm teacher 可比较的 DFlash2 特征。

1. **静态编译**：有效实现不含探针空洞，完成对应 BW 架构设备代码生成及 tqserve 链接；NVIDIA 版本继续可用。
2. **操作级数值对照**：分别检查 packed half 算术、码本展开与排列、矩阵乘法及累加、ldmatrix 转置、subgroup 操作、cp.async 同步。可使用 CPU 参考计算及 RTX 4090 原始 kernel 输出。
3. **量化 teacher 前向**：在相同模型文件和 token 序列下，对比 BW 与 RTX 的 TQ4 展开、矩阵结果、关键层状态及 logits。特别检查 TQ4 128 维旋转、Gated DeltaNet 和 attention 路径。
4. **DFlash2 hidden states**：使用现有 drafter 特征接口逐层对齐层 ID、token 位置和形状，比较数值偏差，并验证数据能被下游 trainer 消费。
5. **完整推理一致性（若迁移 tqserve 全功能）**：核查同一 teacher 的单步 decode 与 verify 路径，并沿用架构文档要求的有/无 drafter greedy 结果一致性；复用 test-dflash-exact、test-tree 等既有测试。
6. **性能分析**：正确性达标后，再测 BW 显存峰值、prefill/feature throughput、tokens/s，对比分块 GEMM 回退与原生优化路径，按瓶颈决定要不要进一步实现 BW 原生 MMA。

仓库中已有 tests/kv_tcq.cu、tests/dflash_exact.cpp、tests/dflash.cu、tests/gemm_pad.cu、tests/gdn_fast.cu、tests/gdn_chunk.cu、tests/kernels.cu 等测试可用于组织验收；它们在 BW 上是否运行成功要分别核实。

验收数据至少记录：双方模型版本、输入 token、DTK/显卡环境、目标层与位置、误差统计和判定依据。允许因 BF16/FP16 累加顺序不同存在合理误差，但应在执行测试前确定容差，不能用过宽容差掩盖算法错误。

## 6. 交付物与最终完成标准

按依赖推进，不预先锁定具体实现 API：

- **阶段一：正确性优先**。为关键 teacher 前向建立可执行的 BW 算术路径（允许慢速回退），避免继续使用陷阱探针。
- **阶段二：真实设备语义验证**。覆盖 attention 同步、逻辑 warp、量化模型特征和 DFlash2 数据接口。
- **阶段三：针对瓶颈优化**。依据真实 BW 数据选择是否实现矩阵融合、原生 MMA 与异步流水线。

交付应包含：BW 可选构建方案与源代码、复现命令、实际 BW 数值对照结果，以及具体功能覆盖范围。不得把“探针版本 100% 编译成功”写成“模型可运行”。

**最终完成标准：BW 真机能够从我们的 TQ_L-GPTQ 权重执行语义正确的 Qwen3.8 前向，导出与 RTX 4090 azllm teacher 路径可比较的 hidden states；原 NVIDIA 路径保持正常。** 进一步追求高性能完整推理可独立后续进行。

## 7. 已落地的分层实现与离线验证进展（2026-11-17）

1. **根因查明与构建修复（`CMakeLists.txt`）**：
   - 新增 `-DTQLLM_BW=ON -DTQLLM_BW_ARCH=gfx936`（及供 CUDA 侧对账的 `-DTQLLM_BW_EMULATE=ON`）。
   - 查明诊断版 `U __hip_fatbin` 链接失败的根因：`tqllm` 无跨编译单元 `__device__` 调用，而 GPUFusion 的 `dcc -dlink` 仅生成模拟桩函数并留空 `__hip_fatbin`（此前链接 `librccl.so.1` 会导致运行时错误绑定 RCCL fatbin）。在 `TQLLM_BW=ON` 下关闭 `CUDA_SEPARABLE_COMPILATION`，启用 C++20 与 PIC，并链接真实运行库 `libgalaxyhip.so`，生成的 `tqserve` 自带 `gfx936:sramecc+` `.hip_fatbin` 段且不再依赖 `librccl.so.1`。
   - 查明 GPUFusion `nvcc`（`-dcccuda12`）下直接包含 `<du_mma.h>` 会触发 `__HIPCC__` 宏缺失与 FP8/FP6 类型冲突；同时查明 `GPUFUSION_ENABLE_WARPSIZE32` 会将 `__lane_id()` 截断为 `0..31` 从而破坏 64-lane DUMMA 片段映射。最终保持硬件原生 wave64 `__lane_id()`，在 32-lane 逻辑子组处显式约束 `width=32` 及半波 ballot 掩码。
   - 查明 `gfx936` 单 workgroup LDS 硬上限为 `64 KiB`（65536 B），将原先 `85.5–99 KiB` 的 4 类 Attention / 压缩 KV 动态共享内存路径全部收敛至 `38.9–60.25 KiB`。

2. **语义层 + 执行层代码结构**：
   - `kernels/bw_primitives.cuh` + `kernels/common.cuh`：实现零陷阱的 `f16x2` 打包算术（生成 `v_pk_mul_f16` / `v_pk_add_f16` / `v_pk_fma_f16`）、完整符号扩展语义 `prmt_b32`、32-lane 逻辑子组 `ldsm_x4` / `ldsm_x4_t` / `mma_f16` / `mma_bf16` / `mma_f16acc`（保持 FP16 累加舍入）/ `mma_s8` / `any_sync32` 及同步 staging。
   - `kernels/bw_dumma.cuh`：封装 `gfx936` 原生 `__builtin_amdgcn_mmac_f32_16x16x16f16`、`__builtin_amdgcn_mmac_f32_16x16x16bf16`、`__builtin_amdgcn_mmac_i32_16x16x32i8` 及 `du_mma.hpp` 对齐的 `16x16` 矩阵片段加载器，并提供 `TQLLM_BW_EMULATE` 64-lane 波前仿真器。
   - `kernels/formats/tq4.cu`：按形状分离双路径——Decode（`ncols <= 2`）走流式 Wave64 向量 GEMV（`tq4_gemv_wave64_bw_k`，LDS 复用激活 + 64-lane 归约，避免浪费 15/16 MMAC 列）；Verify/小批量（`ncols >= 3`）走零 shuffle K 重排双射 Wave64 Split-K DUMMA（`tq4_matvec_dummabw_k`）；Prefill/特征提取走 4-wave64 `128x64` 原生 DUMMA GEMM（`tq_gemm_bw_k`，FP32 累加避免异常值通道溢出 65504）。
   - `kernels/attention.cu` & `kernels/formats/kv_tq3.cu`：提供 `<= 49 KiB` LDS 的原生 Wave64 BF16 DUMMA Prefill/Verify Attention（`attn_bf16_prefill_bw_k`、`attn_mq_bw_k`），以及 `q4/tcq2` 等分裂压缩 KV Cache 的原生 INT8 QK（`v_mmac_i32_16x16x32_i8`）+ FP16 PV（`v_mmac_f32_16x16x16_f16`）路径。

3. **已完成验证与待真机项**：
   - **DTK 26.10 `gfx936` 编译与 ISA 检查**：`tqserve`、`tqrun`、`tqkld`、`tqkbench`、`test-bw-primitives`、`test-tq-gemm`、`test-kv-tcq`、`test-gdn-fast`、`test-gdn-chunk`、`test-dflash-exact`、`test-tree` 全部编译链接通过；反汇编确认生成 `v_mmac_f32_16x16x16_f16`、`v_mmac_f32_16x16x16_bf16`、`v_mmac_i32_16x16x32_i8`、`v_pk_*_f16`，`0` 处 `s_trap`、`0` 处 VGPR spill。
   - **本地 RTX 4090 `TQLLM_BW_EMULATE=ON` 与默认 NVIDIA 双模实跑**：`test-bw-primitives`（含 65536 穷举 `prmt_b32`/`expand_u`、半波分歧 MMA、Wave64 DUMMA 布局、TQ4 双路径、Prefill/Verify Attention）、`test-tq-gemm`（含 `|x|<=300` 离群通道）、`test-tree`（含 `q4/tcq2` tree verify）、`test-kv-tcq`、`test-gdn-chunk`、`test-topk-warp` 全部 PASS，默认 NVIDIA PTX 路径零回归。
   - **待办（需租用 BW1000 真机）**：在物理 `gfx936` 设备上运行上述单测，并对 `Qwen3.8-27B-Uncensored-TQ_L-GPTQ.gguf` 执行 teacher 前向与 DFlash2 hidden states 数值对账及吞吐剖析。
