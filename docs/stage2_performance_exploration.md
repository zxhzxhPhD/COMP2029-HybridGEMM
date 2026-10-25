# 阶段二：冻结质量方案后的性能探索

本阶段只改变 HyGEMM 的计算、在线 pack、load-time 布局和调度。四模型的阶段一主 mask、FP4 clipping bank、FP8 RTN 数值规则及 payload 保持冻结；MixLLM 原 device kernel 不修改。实验根目录为三阶段总目录下 `02_performance/runs/20261026_015237_stage2_performance`。完整结果、原始失败、构建文件、计划和交接清单均在该目录，不提交模型或结果数据。

## 可复现入口

`integrations/vllm/performance/prepare.py` 根据阶段一 handoff 校验并建立只读模型 view，使用 development validation prompts；旧 profiling 工具读取的 legacy `test` 字段也明确装入同一 validation 数据。最终 test/C4 不用于本阶段选择。

`build.py` 编译新 factory、light、GEMV 和 serving ABI，复用冻结 archive 中已有 CUTE 对象并记录 hash；常规 CMake 也能从源码构建全部目标。`build_fused.py` 为旧 TMA precision-switch 实现生成独立诊断库。它需要 PIC 对象，不能直接把过去非 PIC 的 archive 链接成 shared library。

`tune.py` 在精确 O/K/H 和8个 decode token bucket 上搜索。先对 O/K/purity 分组做单路 resident screen，再对不同 kernel 家族的候选进行 cold refinement，之后比较两路执行策略和每个精确 H。256 MiB eviction 位于计时区间外，当前 activation 在计时前恢复到 cache。每次计算检查所有输出行及 token，FP32 dequant reference 的容差预登记为 atol=.01、rtol=.003。微基准只提出配置，不决定部署。

`benchmark.py` 使用固定 FULL Graph、禁用 Inductor、相同 attention/KV/scheduler 配置，5样本、CV≤5%、最多3组。`bind_plan.py` 支持逐层选择旧/新配置，防止0.5B新H合并后丢失原逐层配置；绑定 binary、mask、bank、shape 和原plans。loader 对最终绑定执行校验，不能只依据文件存在复用结果。

## 计算与调度候选

- 原 IDs0–27 保留。IDs28–34直接读取packed权重，每warp计算一行，比较token复用、warp数、固定split-K和有限grid-stride；不保留展开的FP8权重副本。split-K使用FP32临时部分和及固定顺序归约。
- IDs35–44沿用cp.async轻量MMA，比较32-bit scale-word复用及M/N/K、warps、stage组合。与对应旧候选对照，资源和SASS必须由实际编译结果报告。
- ABI policy 0/1为原serial/concurrent，2/3为high-first/high-main，4/5/6为一次launch按CTA分派两个精度的GEMV。所有策略保持logical输出scatter。
- 旧TMA单kernel按原设计共用tile/pipeline，在输出tile边界切换精度；probe记录两种expected transaction bytes。初始固定指针probe之外，另加同时更新两个problem的serving rebind ABI，并以`HYGEMM_FUSED_GATEUP=8`在整模型验证。该候选未获益，最终关闭。

## Pack、producer和离线布局

`HYGEMM_PACK_MODE` 0是旧版，1把单128-row scale block的padding融合扩展至128行，2是每lane4值/每K32组8 lanes的vector pack，3在rows≥16时使用2，其余使用旧版。FP4权重pack保持原算法。完整scale padding buffer必须逐字节相等，不能仅比较反量化结果。

SwiGLU producer是独立opt-in库：`HYGEMM_FUSE_SILU=1`、`HYGEMM_PRODUCER_LIBRARY` 和 `HYGEMM_SILU_MATH`。只适用于本阶段单GPU Qwen2、OfflineLinearMethod、T≤128且down projection无bias。gate-up输出直接生成MXFP8，保留原SiLU和乘法的两次FP16舍入；该FP16中间张量没有其他consumer。原vLLM的`up + beta`即使beta=0也必须保留signed-zero行为。采用前与实际安装的vLLM算子做bitwise检查；近似exp模式不能因数学公式相同而自动通过。

`HYGEMM_ROW_ORDER=reverse/block_reverse` 是gate-up分区内load-time重排实验；同步置换packed权重、logical scale与channel IDs，再恢复原physical scale layout。precision membership不变，不留shadow weight。默认none；只有整模型通过后才可进入最终配置。

## 验证与测量边界

`check.py`、既有`check_light.py`以及sanitizer覆盖full output、短K、tail、空分区、scale padding、pointer rebind和changed-input Graph。`decode_quality.py`通过独立worker hook执行teacher forcing：替换采样token，保留原logits再让vLLM提取目标logprob。使用129-token续写，在两次运行实际FULL Graph decode的共同位置比较，排除prefill和mixed步骤；不把该诊断的同步开销当作速度数据。PPL prefill不能替代这个门禁。

质量诊断另设enqueue barrier：先暂停scheduler、保留cache、提交完整请求组，再恢复执行，并核对两次prefill schedule hash。异步请求入队可改变prefill分块及KV数值，不能将这种差异误判为decode kernel误差。速度实验保持原调度协议，不加此barrier。最终每模型比较16,320个共同FULL Graph目标位置，四模型均为零差异；SIMT候选仍超过预登记最大logprob差门槛，保留为未采用实验，而非断言其PPL已劣化。

Nsight只作分解，真实时延来自无profiler控制组。GEMM与scatter在同kernel，不能伪造独立scatter耗时；并发两路按区间并集/关键路径计。带宽实测仅给乐观参照；无counter权限时不声称测得occupancy或stall。

所有自定义Graph probe都在launch时读取current stream，并检查capture节点数。曾有旧TMA probe错误复用创建stream而捕获空Graph，该版计时明确作废；修正版和原失败数据分别保留。模型计时与DualLinear调优在调用时读取stream，不受该probe错误影响。

最终采用项、四模型计划、消融、数值验收和负结果以本阶段`report.md`、`handoff.json`为准；中间最佳微基准不等于最终方案。

最终主路径保持双kernel，使用实际shape的逐层计划、adaptive vector pack（mode3）及精确SwiGLU producer（math0）。原ID0–27与新增light MMA候选共同筛选；SIMT、row-order和TMA融合均不进入冻结主配置。`freeze_model.py`与`check_binding.py`负责绑定和拒绝过期配置，`validate_results.py`核对完整矩阵、质量、预算与Mix二进制。`analyze.py`用有matplotlib的基础Python生成图，`analyze_profiles.py`统一重算初末profile，`report.py`生成报告和阶段交接；结果归档使用`archive_results.py`，不跟随模型符号链接。
