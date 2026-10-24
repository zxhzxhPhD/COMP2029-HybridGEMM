# SM120 原生低精度浮点与输出通道混合精度：研究动机和证据边界

日期：2026-10-24（Asia/Shanghai）。本文用于明确研究问题、硬件机制、资源权衡和下一步验证方法，不作为已经超过 MixLLM 的性能或精度结论。

当前实现依据：RTX 5090 / SM120，CUDA 13.0.48（PTX 9.0 代际），CUTLASS `e6233cbac5d7c7a865c19c91cd684ceece19513c`。MixLLM 源码依据为本地 upstream `00263510f9760d3a1d958a388e0e7c25961a7d59`。本轮另查阅了更新的 PTX 9.4；其中新增指令单独标注，未纳入现有 CUDA 13.0 实测能力。

本研究的主要假设是：**沿输出通道划分精度，以 MXFP4 weight × MXFP8 activation 为主体、MXFP8 weight × MXFP8 activation 为补充，可以保留较高的浮点 MMA 吞吐，同时避免 MixLLM 整数路径中的部分软件解包、零点处理、group 部分和转换与缩放开销。最值得验证的资源收益是消除 group accumulator 和缩短供数依赖链；shared memory 容量则可能成为代价。**

这里的机制由“数据格式 + MMA 指令族 + block scaling + 数据布局”共同决定，不能只凭 FP4 / FP8 的格式名称判断。

## 1. 从沿 K 提升精度转向输出通道分解

采用以下记号：

- `T`：token / batch 数；`O`：输出通道数；`I`：输入通道数。
- `X ∈ R^(T×I)`，`W ∈ R^(O×I)`，`Y = X W^T ∈ R^(T×O)`。
- 现有 benchmark 的底层 GEMM 使用 `M_out=O, N_tokens=T, K_in=I`，计算 column-major 的 `Y^T`；其物理输出仍为 token-major `Y`。

沿输入通道划分时，需要对同一输出求和：

\[
Y = X_{[:,K_L]}W_{[:,K_L]}^T + X_{[:,K_H]}W_{[:,K_H]}^T.
\]

两种精度路径共享同一个归约问题。旧方案因此必须协调不同 MMA K 粒度、部分和累加、K 重排以及 activation 在线打包。相关旧实现可以继续作为技术参考，但不再约束当前设计；[旧 execution/layout contract](execution_policy_and_layout_contract.md)描述的是这一历史方案。

沿输出通道划分则是两个独立输出子问题。令输出索引集合 `O_L`、`O_H` 不相交且覆盖全部输出通道：

\[
Y_{[:,O_L]} = X W_{[O_L,:]}^T,\qquad
Y_{[:,O_H]} = X W_{[O_H,:]}^T.
\]

两部分可以独立设置 tile / schedule，使用同一份 FP8 activation，写回目标输出的对应通道。它们不需要相互累加。主要新增协调成本变成输出索引映射、scatter、少量高精度通道的 tile 利用率和调度。

这一分解来自 MixLLM 的核心系统思路；不能把“按输出通道拆分”本身作为新贡献。值得探索的是：在 SM120 上将其与原生 block-scaled 浮点指令结合，能否得到更好的精度—存储—执行效率折中。

## 2. MixLLM 主路径实际做了什么

MixLLM 的 W4A8 指的是存储位宽；其 Tensor Core 实际执行 INT8×INT8。以单个输出元素和一个量化 group `g` 为例：

\[
P_g=\sum_{k\in g}(q^W_k-z^W_g)q^X_k,
\qquad Y\mathrel{+}=s^W_gs^X_g\,\mathrm{float}(P_g).
\]

权重采用 UINT4 asymmetric quantization，激活采用 INT8 symmetric quantization；源码的 group size 为 128。`q^W-z^W` 的范围可以超过 signed INT4，因此先展开到 INT8 再计算。不同 group 的 scale 不同，使 INT32 部分和不能无条件跨全部 K 累加后只做一次 epilogue scaling。

实际数据流为：

```text
global packed UINT4 ──cp.async──> shared packed UINT4
    ──普通矩阵加载──> packed register fragment
    ──掩码/移位──> INT8 fragment
    ──zero-point 广播和减法──> IMMA
    ──group INT32 部分和──> fast I2F、乘两边 scale
    ──合并──> 跨 group 的最终 FP32 accumulator
```

源码直接支持以下判断：

- `mq_numeric_conversion.h` 的 `NumericArrayConverter<int8_t,uint4b_t,8>` 用两个 `and` 和一个 `shr` 把 8 个 nibble 展开成两个 32-bit 字节容器。这里计数是 PTX 源级操作，不是最终 SASS 条数或周期数。
- `mq_mma_tensor_op_dequantizer.h` 的 `apply_zero` 包含 zero-point 广播及 `vsub4`，或者按元素相减的分支。
- `mq_mma_multistage.h` 同时存在 `tmp_accum_` 与最终 `accum`，并维护 loaded/transformed operand fragments。
- fast I2F 已通过偏置技巧优化：group accumulator 以整数偏置初始化，结束时重解释并减浮点偏置。不能把 MixLLM 描述成逐 group 执行昂贵的普通 INT32→FP32 转换。
- 原版代码使用异步 global→shared 拷贝，不应声称换成 TMA 才首次消除了 global 数据经过通用寄存器的需要。

原始代码与关键定位见文末来源表中的 MixLLM converter、dequantizer、mainloop 和 shared storage。

## 3. 原生浮点路径改变了哪些机制

当前低精度实验统一使用 MXFP4 E2M1 / MXFP8 E4M3，每 K32 一个 UE8M0 scale，FP32 累加、FP16 输出。MXFP4 的格式和 scale 限制与 MixLLM 的 asymmetric INT4 不同，数值精度需要重新评估。

原生 block-scaled MMA 可以把 scale 随操作数一起消费，对同一份 FP32 accumulator 持续更新。数学上仍然是在求不同 block 的缩放乘积之和，但软件不再显式维护“当前 group INT32 部分和 → 转浮点 → 缩放 → 加入最终和”的循环。

| 相对原版 MixLLM 的变化 | 机制 | 结论强度 |
|---|---|---|
| 减少软件 UINT4 unpack | 当前路径把展开放入 shared→register 的 LDSM | 源码及 SASS 确认；不是 FP4 独占能力 |
| 去掉 zero-point 处理 | 当前 MXFP4 不使用 additive zero-point | 机制明确；伴随量化格式变化 |
| 去掉 fast I2F 偏置处理 | MMA 直接生成 FP32 累加结果 | 机制明确 |
| 去掉 group 输出缩放和软件合并 | block-scale 由 MMA 消费 | 机制明确，限硬件支持的 scale 语义 |
| 去掉独立 group accumulator | 持续更新最终 FP32 accumulator | 有结构性寄存器节省空间 |
| 更高 occupancy / 更好的 pipeline | 取决于实际寄存器、shared、线程分工及 tile | 待同 GPU、同输出工作量比较 |
| shared 更小 | 当前混合路径反而有 FP4 padding | 不能作为默认优势 |
| 已经超过 MixLLM | 尚无同卡整数主路径对照 | 未证明 |

如果未来的精度方案要求额外的、无法由 UE8M0 block scale 或一次可分离 epilogue 表达的修正，这些处理可能重新出现。因此“只需一份 accumulator”是当前量化契约下的设计性质，不是任意 FP4 算法的保证。

## 4. FP4×FP4 与 FP4×FP8 的物理布局并不相同

本轮的 FP4×FP4 主对照走专门的 **K64 OMMA**。FP4×FP8 走通用混合格式的 **K32 QMMA**。两者在 shared、register 中对 FP4 的要求不同。

| 项目 | 本轮 MXFP4×MXFP4：K64 OMMA | 本轮 MXFP4×MXFP8：K32 QMMA |
|---|---|---|
| global weight | packed 4-bit | packed 4-bit |
| global activation | packed 4-bit | 8-bit |
| shared weight value 存储 | packed 4-bit | 每 16 个 FP4 加 64 bit padding，平均 8 bit 地址空间/值 |
| shared activation value 存储 | packed 4-bit | 8-bit |
| shared→register load | 普通 `LDSM.16.M88.*`，搬运 packed bits | FP4 侧 `LDSM.U4x16P64TO8.M816.*`，加载时展开 |
| register weight 编码 | 每个 32-bit word 包含 8 个 FP4 | 每个 32-bit word 包含 4 个带 padding 的 FP4 |
| register activation 编码 | 每个 32-bit word 包含 8 个 FP4 | 每个 32-bit word 包含 4 个 FP8 |
| FP4 编码左移两位 | 不需要 | 当前 CUTLASS 需要 |
| FP32 accumulator | 一份持续累加 | 一份持续累加 |

这里“packed”描述 value 的位密度，不表示完全没有 tile 边界对齐、swizzle 或 scale layout 的额外空间。**K64 FP4×FP4 不存在混合 K32 路径那种每 16 个值附加 64 bit 的 `_p64` 格式开销。**

固定版本 CUTLASS 中，K64 MMA traits 的 `ValTypeA/ValTypeB` 为 `uint4_t`；K32 混合 MMA traits 为 `uint8_t`。mainloop 的 shared 分配也遵循这一选择。K64 走普通 LDSM，并非先展开到 8-bit 再偷偷用浮点转换压回 4-bit。现有 SASS 中对应关系是：

```text
K64 W4A4: LDSM.16.M88.4
           OMMA.SF.16864.F32.E2M1.E2M1.E8

K32 W4A8: LDSM.U4x16P64TO8.M816.4
           FP4 编码位置调整
           QMMA.SF.16832.F32.E2M1.E4M3.E8
```

这一差异必须按**相同逻辑元素数**比较。单条 K64 和 K32 MMA 的 A/B value operand 都可以分别用 4/2 个 32-bit 寄存器；但 K64 在相同寄存器 word 数中携带了两倍数量的 FP4 元素。不能由“每条 MMA 的寄存器参数个数相同”推断它们具有相同的每元素占用，也不能由“FP4 每元素少一半”推断完整 kernel 的寄存器总数必然少一半。

还要避免将所有 FP4×FP4 混为一谈：若显式选择通用 `f8f6f4/mxf8f6f4` 的 **K32 E2M1×E2M1**，两边 FP4 都会走 8-bit operand 容器路径，通常需要对应的 padded shared 布局和对齐处理。该路径不等同于本轮最快的 K64 FP4×FP4 baseline。

## 5. LDSM 的加载便利、硬件展开与 shared 容量

“使用 LDSM”不能直接等同于“使用硬件 unpack”。普通 LDSM 可以按矩阵分发规则搬运 packed bits，而不扩展数值。MixLLM 的已有 SASS 中就能看到普通 `LDSM.16.M88.4`，其后仍执行软件解包；K64 FP4×FP4 也使用普通 LDSM，但下游 MMA 直接接受 packed FP4，因此无须解包。

### 5.1 当前 `_p64` 形式的明确要求

当前 CUDA 13.0 / PTX 9.0 中使用的是：

```ptx
ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64
```

其输入格式规定，每 16 个 unsigned 4-bit 数据配有 64 bit padding；输出是放入 16 个 8-bit 容器的位编码。对于当前标准 TMA/CUTLASS 布局：

```text
global:  [16 个 nibble = 8 B][16 个 nibble = 8 B]...
shared:  [16 个 nibble = 8 B][padding 8 B]...
register: 每个 byte 的低 4 bit 装一个 nibble
```

TMA 添加 padding 不等于在 shared 中逐值做数值转换。padding 内容也不需要初始化为零。真正按字节展开发生在 LDSM，随后当前浮点路径再左移两位满足 E2M1 MMA 的位位置要求。权重依然以 FP4 而非 FP8 数值被解释。

如果把原版 MixLLM 改成这一 `_p64` 加载路径，就不能保持原先 dense packed INT4 shared layout 而直接替换 load 指令；需要匹配输入格式、warp fragment 分发和地址映射。对原始 unsigned INT4，加载得到的字节可以表示 `0..15`，之后减 zero-point，**不需要浮点 E2M1 的左移两位**。这一适配在数值上有依据，但尚未编译和测量完整整数 kernel。

所以，对当前 `_p64` 标准路径，确实存在以下权衡：

- 保持 dense packed shared，再用软件展开到 INT8；
- 使用 padded shared，让矩阵加载同时完成展开。

这不是“LDSM 加载便利与 packed shared 必须二选一”：普通 LDSM 已能与 packed shared 共存。也不是关于所有指令版本的不可突破定律。是否能在不违反对齐、地址映射、TMA 写入及并发访问契约的情况下复用 padding 空间，需要单独证明；本文不把这类未验证布局技巧当作现成优化。

### 5.2 更新的 PTX 9.4 提供了需要另行验证的整数路径

截至本次文档查阅，NVIDIA PTX 9.4 新增了：

```ptx
ldmatrix.sync.aligned.m8n16.x1.shared::cta.s8.s4
```

文档描述其从 signed 4-bit 序列加载并扩展为 signed 8-bit，并明确将 `sm_120f` family 列入支持目标。这一接口不使用 `.b4x16_p64` 形式，为“保留紧凑 INT4，同时在加载时扩展”提供了值得验证的候选。

当前实验用 CUDA 13.0.48；对应的 PTX 9.0 文档没有 `.s8.s4`。**本轮没有升级工具链，没有验证这一形式的编译、物理 shared 地址/对齐要求、生成 SASS、正确性或性能。** 不能把新版文档支持直接记为当前环境已经可用，也不能只凭缺少 `_p64` 后缀就宣称完整 kernel 的 shared 用量已经减半。

这也不是原版 MixLLM UINT4 编码的直接替换。signed S4 对原始 nibble 的解释与 UINT4 不同，但可以考虑如下离线重编码，保持量化数值不变：

\[
q\in[0,15],\quad q'=q-8\in[-8,7],\quad z'=z-8,
\]

\[
q'-z'=q-z.
\]

存储 signed S4 的 `q'` 等价于将原 nibble 的最高位翻转，即 `q XOR 0x8`，但零点必须同步改为 `z'`。直接 sign-extend 原始 UINT4 nibble 不满足这个等价关系。加载后的 zero-point 修正仍需执行，原版 MixLLM 的 group scale 和双 accumulator 问题也不会因 signed unpack 消失。

这是可行的代数变换和文档支持的候选接口，尚不是已实现的优化。若后续要主张硬件 unpack 相对整数方案的优势，应将这样的加强版整数加载路径纳入对照，或明确受工具链限制未能测试。

## 6. 寄存器与 shared 的资源模型

### 6.1 最有依据的寄存器收益来自 group accumulator

MixLLM 在软件中同时保留 group 部分和与最终累加器。对每线程持有 `Q` 个输出元素的实现，这两部分在逻辑上约为 `2Q` 个 32-bit 槽位；当前浮点路径不因量化 group 需要第二份完整 accumulator，约为 `Q` 个槽位。

例如，一个 warp 的输出 tile 为 `32×64`，平均每线程持有 64 个结果。消除 group accumulator 对应最多一份 64-word 的逻辑状态。这是资源预算推导，而不是承诺某个 ptxas 结果恰好减少 64 个寄存器。实际分配受线程分工、operand pipeline、生命周期重叠、编译器复用与 schedule 影响。

此外还可能减少 packed 原始 fragment 的中间状态、zero-point fragment/广播临时值，以及 software group scaling 的临时状态。scale 自身仍需要被加载并提供给 MMA。MixLLM 源码声明的两个数组也不能直接相加当成物理寄存器数量，必须检查最终生命周期及 SASS。

FP4×FP8 最终权重 operand 仍占 8-bit 容器，因此其寄存器优势不应表述为“4-bit 权重让 MMA operand 自动减半”。先前与纯 MXFP8 相同的 ptxas 寄存器数，也不能用来否定相对 MixLLM 双 accumulator 的潜在收益：它们是不同的比较问题。

### 6.2 shared 容量可能变差

设一个 pipeline stage 覆盖 `O_b` 个输出通道、`T_b` 个 tokens、`I_b` 个归约元素，仅计算 A/B value 地址空间：

\[
S_{\mathrm{INT4\times INT8}}\approx \tfrac12O_bI_b+T_bI_b,
\]

\[
S_{\mathrm{FP4\times FP8,K32}}\approx O_bI_b+T_bI_b,
\]

\[
S_{\mathrm{FP4\times FP4,K64}}\approx \tfrac12O_bI_b+\tfrac12T_bI_b.
\]

单位为 byte。对 `O_b=128, T_b=64, I_b=128`：

| 路径 | weight | activation | 合计/stage |
|---|---:|---:|---:|
| 原版 MixLLM INT4×INT8 | 8 KiB | 8 KiB | 16 KiB |
| 当前 FP4×FP8 K32 | 16 KiB | 8 KiB | 24 KiB |
| FP4×FP4 K64 | 8 KiB | 4 KiB | 12 KiB |

这不是最终 kernel shared storage：还要考虑 scale、zero-point、TMA/barrier、epilogue、对齐、scale tensor 的 interleave/padding，以及实际 stage 数。

例如本轮固定 `128×64×128 cooperative` 时，W4A8 是 3 stages / 91,136 bytes；W4A4 是 6 stages / 94,208 bytes。W4A4 总 shared 略高，是自动 stage selection 利用了更小的每-stage footprint，不能据此否定其更紧凑的 value 存储。

还需区分三件事：shared 地址空间、真实读写事务字节数、bank/端口利用率。padding 扩大容量占用，不足以单独证明每次硬件访问的事务流量也按同一比例增加。后两者需要 profiler 或专门 microbenchmark。

### 6.3 metadata 与 global 流量没有自动优势

原版 INT4×INT8 和 FP4×FP8 的 global 主体数据都为 W4A8。浮点路径相比纯 FP8 权重更省空间，但相比 MixLLM INT4 权重并没有这一项额外减半。

每条向量沿 K 的 scale 密度也不同：MX 每 K32 一个 1-byte scale，相当于每 K128 四个 byte；MixLLM 每 K128 一个 2-byte scale，权重再有 zero-point。当前 upstream 的 zero-point 元素为 `uint8_t`。去掉 zero-point 不等于 scale/metadata 总量必然更少，更不等于 shared 中的 scale layout 更紧凑。

## 7. 初步测量支持了什么

原始数据、完整报告及图仍保存在实验 tmp，不复制 build / profiling 产物进源码仓库：

- 本地：[实验报告](D:/User/document/repos/tmp/20261024_190557_sm120_format_probe/report.md)。
- 本地：[最终结果 CSV](D:/User/document/repos/tmp/20261024_190557_sm120_format_probe/summary_tuned.csv)。
- 本地：[SASS 摘要](D:/User/document/repos/tmp/20261024_190557_sm120_format_probe/sass_summary.json)。
- 远程结果目录：`/root/autodl-tmp/20261024_190557_sm120_format_probe`。
- 实验代码：[GEMM probe](../src/kernel/sm120_format_probe.cu)、[MMA probe](../src/kernel/sm120_format_mma_probe.cu)、[runner](../scripts/run_format_probe.py)。

首轮寄存器驻留 MMA 测试测得：FP16 约 248 TFLOP/s，W4A8 与 MX W8A8 均约 980 TFLOP/s，W4A4 约 1961 TFLOP/s。它说明当前配置中混合浮点类型没有降低到明显弱于 MXFP8 的计算吞吐，但 W4A8 没有 W4A4 的两倍吞吐优势。

七种 GEMM shape 中，W4A8 相对 cuBLASLt FP16 为 0.96–3.43×，与匹配 MX W8A8 大体持平。小 batch 的 token tile 从 64 缩到 16，使 W4A8 由约 15.1 μs 降至 8.8 μs；较大 batch 需要不同 tile。W4A4 在这些纯路径中更快，但其 FP4 activation 的模型精度代价尚未测量。

这些是预打包、同一 buffer 重复执行的 hot-cache 短测：每配置 3 次 CUDA Graph 测量，每次 100 个 GEMM。输入数值被选择为各格式可精确表示，以检验计算路径；它们不是模型量化误差实验。小矩阵全检查和正式 shape 的输出抽查均通过。

尚未测得同卡 INT8 MMA 上限、整数 MixLLM GEMM、两路输出通道混合 kernel、在线 activation quantization/pack 或全模型端到端结果。不能把跨架构额定吞吐或本轮相对 FP16 的 speedup 当成已经超越 MixLLM 的证据。

## 8. 有价值的研究问题及其可证伪条件

**问题 A：浮点主路径是否能更接近自己的 MMA 上限？**

需要比较相同 SM120 上原版整数主路径、适配后的整数主路径、原生浮点主路径。分别报告各自寄存器驻留 MMA 吞吐和完整 GEMM 吞吐，并观察两者的比例。这样才能区分原始算力差异与供数/反量化效率。

若整数路径本来已经把 unpack 和 group 后处理完全隐藏，或者浮点路径的 padded shared 使其成为主要瓶颈，软件操作减少未必兑现为更低延迟。

**问题 B：单 accumulator 能否形成实际 kernel 设计优势？**

需要在可比输出 tile 和线程工作分配下检查 ptxas registers、spill、shared/stage、可驻留 CTA，以及吞吐。关注它是否允许更大的输出 tile、更充分的 operand 预取或更多有效并发，而不是只比较某两个任意配置的总寄存器数。

若 kernel 已被 shared、访存或 grid 并行度限制，释放 accumulator 寄存器也可能没有速度收益。不能用“寄存器减少”代替“性能提升”的验证。

**问题 C：少量高精度输出通道能否以低成本改善精度？**

验证 W4A8 / W8A8 子路径共享 FP8 activation 时的精度恢复、tile 对齐、scatter、并行或融合调度。应同时报告目标比例与由于 tile 对齐产生的实际高精度比例。

如果高精度通道占比很小，其子矩阵可能无法充分利用 GPU；两个 kernel 的调度成本也可能抵消主体路径的收益。输出通道分解使问题更简单，但不保证高精度补充是免费的。

**问题 D：面对更强的整数加载基线，核心优势是否仍存在？**

至少区分三个整数对照：原版 packed shared + 软件 unpack；适配 `_p64` 硬件展开的路径；在工具链允许后验证 `.s8.s4` 的 signed 重编码路径。后两者不是本文已经实现的 baseline。

即使整数路径也拿到硬件 unpack，zero-point 和 software group scaling/双 accumulator 仍然是不同点。因此动机应围绕原生浮点 block scaling 的完整数据流，而不是仅围绕几条掩码/移位指令。

## 9. 后续验证的最小顺序

1. **补足同 GPU 整数参照。** 测少数代表 shape 的 INT8 MMA 和 MixLLM W4A8，固定输出类型、计时协议与输入成本边界；区分移植收益和格式收益。
2. **做机制消融。** 在数值语义可控的前提下区分 unpack、zero-point、group 后处理开销；对重写版本明确是否仍保持原量化数学等价。
3. **实现最小输出通道分解。** 先测试 0%、10%、100% 高精度输出比例，用两个独立子 GEMM 和真实写回建立参照，再决定是否融合。它不是旧 K-slice 方案，不要求两路共享 accumulator。
4. **补上真实输入链。** 分别报告预打包 GEMM、activation quantize/pack + GEMM，以及混合调度/scatter 总成本。在线 FP8 量化与 scale 生成不会自动消失。
5. **验证精度—效率折中。** 在相同模型/校准范围内比较 INT4 asymmetric 与 MXFP4 的误差和高精度通道恢复效果。MixLLM 的 INT4 精度结论不能直接迁移到 E2M1。

本次文档整理没有新增 GPU benchmark、修改现有 kernel 或升级工具链。上述步骤是待验证计划，不是已完成工作。

## 10. 论文式动机应如何表述

可以据现有证据提出：

> 输出通道级混合精度允许不同权重格式共享 activation，并独立计算输出子矩阵。SM120 的原生 block-scaled 浮点 MMA 进一步改变了传统 W4A8 整数 kernel 的反量化路径：它将尺度处理纳入 MMA，使软件无需保留每个量化 group 的整数部分和及其后处理。由此产生的寄存器预算和供数链变化，可能为混合精度 GEMM 提供新的优化空间，但需要同时处理 mixed-FP4 operand 的 shared padding 代价。

应避免未经验证的表述：

- “FP4×FP8 等于 FP4×FP4 的计算吞吐”。
- “原生 FP4 完全没有 unpack / 格式调整”。
- “相对 MixLLM，shared 和 register 必然同时减半”。
- “LDSM 一旦做 unpack，就永远不能维持紧凑 shared”。
- “相对纯 FP8 的权重压缩，也是相对 MixLLM 的额外带宽优势”。
- “现有速度结果已经证明同精度预算下优于 MixLLM”。

性能收益、模型精度和方法新颖性均需分别验证；本文的作用是将最有依据的机制与尚未证明的贡献分开。

## 11. 可追溯来源

| 来源 | 关键内容 |
|---|---|
| [MixLLM 论文抽取文本](../../../MixLLM/paper/MixLLM_MLSys2026.txt) | 当前工作区内的论文全文；重点为 §3.3、§3.4 |
| [MixLLM numeric conversion](https://github.com/microsoft/MixLLM/blob/00263510f9760d3a1d958a388e0e7c25961a7d59/mixllm/kernels/cutlass_extension/mq_numeric_conversion.h#L55) | UINT4→INT8 掩码/移位，fast I2F |
| [MixLLM dequantizer](https://github.com/microsoft/MixLLM/blob/00263510f9760d3a1d958a388e0e7c25961a7d59/mixllm/kernels/cutlass_extension/mq_mma_tensor_op_dequantizer.h#L225) | zero-point、scale 处理 |
| [MixLLM mainloop](https://github.com/microsoft/MixLLM/blob/00263510f9760d3a1d958a388e0e7c25961a7d59/mixllm/kernels/cutlass_extension/mq_mma_multistage.h#L864) | group 初始化、双 accumulator、缩放合并 |
| [MixLLM shared storage](https://github.com/microsoft/MixLLM/blob/00263510f9760d3a1d958a388e0e7c25961a7d59/mixllm/kernels/cutlass_extension/mq_mma_base.h#L152) | packed operand 与 scale/zero 缓冲 |
| [CUTLASS MMA traits](https://github.com/NVIDIA/cutlass/blob/e6233cbac5d7c7a865c19c91cd684ceece19513c/include/cute/atom/mma_traits_sm120.hpp#L136) | K64 的 uint4、K32 的 uint8、FP4 shift |
| [CUTLASS load selector](https://github.com/NVIDIA/cutlass/blob/e6233cbac5d7c7a865c19c91cd684ceece19513c/include/cutlass/gemm/collective/builders/sm120_common.inl#L55) | 普通 LDSM 与展开 LDSM 的选择 |
| [CUTLASS mainloop](https://github.com/NVIDIA/cutlass/blob/e6233cbac5d7c7a865c19c91cd684ceece19513c/include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp#L230) | TMA 输入类型、shared value 类型及 scale 消费 |
| [CUDA 13.0 / PTX 9.0 ldmatrix](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix) | 当前工具链代际的 `.b4x16_p64` 格式 |
| [PTX 9.4 ldmatrix](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix) | 新增 `.s8.s4`、SM120 family 支持；当前未验证 |
| [PTX sub-byte padding](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-dimension-size-and-format) | `.b4x16` 与 `.b4x16_p64` 的区别 |

网上最新 PTX 页面会继续更新；本文对 `.s8.s4` 的说明对应查阅时显示的 PTX ISA 9.4。固定版本 CUTLASS/MixLLM 链接和本地实测 SASS 的证据优先于对更新接口的推测。
