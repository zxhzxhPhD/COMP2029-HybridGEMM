"""Render the completed G0/Q0/Q1 evidence without changing method selection."""
import argparse
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
root=args.out
read=lambda p:json.loads((root/p).read_text(encoding='utf-8'))
s=read('summary.json');assert s['passed']
names=['0.5B','1.5B','3B','7B']
families=['weight_gain','output_gain','loss_abs_gain','loss_signed_gain']
short=s['shortlist']['families']
native_example=read('0.5B/native/original.json')
chance_text='；'.join(f"{n}: {100*r['selected_channel_retention']:.2f}% / {100*r['selected_weight_retention']:.2f}%" for n,r in s['mask_retention_random_expectation'].items())
lookup={n:{r['method']:r for r in s['models'][n]['validation']} for n in names}
fig,axes=plt.subplots(1,2,figsize=(12,4.8))
x=np.arange(4);width=.18
for i,method in enumerate(families):
    axes[0].bar(x+(i-1.5)*width,[-100*lookup[n][method]['ppl_relative_change'] for n in names],width,label=method)
axes[0].axhline(0,color='#777',linewidth=.8);axes[0].set_xticks(x,names)
axes[0].set_ylabel('Validation PPL reduction vs original (%)')
axes[0].set_title('Fixed high count in every original projection');axes[0].legend(fontsize=8)
for i,method in enumerate(short):
    values=[];lo=[];hi=[]
    for n in ['3B','7B']:
        r=s['robustness'][method][n];values.append(r['mean_delta_nll'])
        lo.append(r['mean_delta_nll']-r['delta_nll_range'][0]);hi.append(r['delta_nll_range'][1]-r['mean_delta_nll'])
    axes[1].errorbar(np.arange(2)+(i-.5)*.12,values,yerr=[lo,hi],fmt='o',capsize=4,label=method)
axes[1].axhline(0,color='#777',linewidth=.8);axes[1].set_xticks([0,1],['3B','7B'])
axes[1].set_ylabel('Mean NLL change vs original (lower is better)')
axes[1].set_title('Calibration seeds: mean and min/max');axes[1].legend(fontsize=8)
for ax in axes:ax.grid(axis='y',alpha=.2)
fig.tight_layout()
for ext in ['png','svg']:fig.savefig(root/f'selection_comparison.{ext}',dpi=180)
plt.close(fig)

budget='| 模型 | Payload平均bits/weight | packed权重 bytes | padded weight scale bytes | 通道索引 bytes | 合计 bytes | 其他模型参数/注册buffer bytes |\n|---|---:|---:|---:|---:|---:|---:|\n'
quality='| 模型 | FP16 | MixLLM原RTN | 原Hy选择 | 随机3seed均值 | Weight gain | Output gain | Loss abs gain | Loss signed gain |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
ci='| 模型 | 方法 | ΔNLL vs原Hy | 配对块bootstrap 95%区间 | PPL相对变化 |\n|---|---|---:|---|---:|\n'
diagnosis='| 模型 | dense参考 | 仅A8 ΔNLL | 仅混合W ΔNLL | 混合W+A8 ΔNLL | 软件/native ΔNLL绝对差 |\n|---|---|---:|---:|---:|---:|\n'
robust='| 模型 | 入围方法 | 3seed平均ΔNLL | seed范围 | 是否所有seed优于原选择 |\n|---|---|---:|---|---|\n'
stability='| 模型 | 方法 | seed间高精度通道保留比例 | 按权重元素加权的保留比例 |\n|---|---|---:|---:|\n'
for n in names:
    d=s['models'][n];b=d['budget'];m=read(f'{n}/original_calibration_manifest.json')
    budget+=f"| {n} | {b['payload_bits']/m['weights']:.6f} | {b['weight_bytes']:,} | {b['weight_scale_bytes']:,} | {b['channel_index_bytes']:,} | {b['persistent_quantized_bytes']:,} | {d['other_model_parameter_buffer_bytes']:,} |\n"
    v=lookup[n]
    numbers=[v[k]['ppl'] for k in ['reference_fp16','reference_mixllm','original']]
    numbers+=[statistics.mean(v[f'random_{seed}']['ppl'] for seed in [2029,2030,2031])]
    numbers+=[v[k]['ppl'] for k in families]
    quality+='| '+n+' | '+' | '.join(f'{value:.6f}' for value in numbers)+' |\n'
    for method in families:
        row=v[method];interval=row['paired_block_bootstrap_95ci']
        ci+=f"| {n} | {method} | {row['delta_nll']:+.6f} | [{interval[0]:+.6f}, {interval[1]:+.6f}] | {100*row['ppl_relative_change']:+.3f}% |\n"
    for file,r in d['diagnosis'].items():
        mm=r['modes'];label='FP32 dense' if 'fp32' in file else 'FP16 dense'
        diagnosis+=f"| {n} | {label} | {mm['w16_a8']['delta_nll_vs_reference']:+.6f} | {mm['mixed_a16']['delta_nll_vs_reference']:+.6f} | {mm['mixed_a8']['delta_nll_vs_reference']:+.6f} | {r['software_native_nll_discrepancy']:.6f} |\n"
for family in short:
    for n in names:
        r=s['robustness'][family][n]
        robust+=f"| {n} | {family} | {r['mean_delta_nll']:+.6f} | [{r['delta_nll_range'][0]:+.6f}, {r['delta_nll_range'][1]:+.6f}] | {'是' if r['all_seeds_improve'] else '否'} |\n"
        pairs=s['mask_stability'][family][n]
        stability+=f"| {n} | {family} | {100*statistics.mean(x['selected_channel_retention'] for x in pairs):.2f}% | {100*statistics.mean(x['selected_weight_retention'] for x in pairs):.2f}% |\n"
speed='| 方法 | Batch | ms/step | 相对原选择速度 | CV |\n|---|---:|---:|---:|---:|\n'
for method,rows in s['decode_7B'].items():
    for r in rows:speed+=f"| {method} | {r['batch']} | {r['ms']:.5f} | {r['speedup_vs_original']:.4f}× | {100*r['cv']:.3f}% |\n"
rank='| 模型 | 投影 | 方法 | Top−Bottom ΔNLL | Top−原选择 ΔNLL |\n|---|---|---|---:|---:|\n'
for n in ['3B','7B']:
    probe=read(f'{n}/rank_probe.json')
    for layer in probe['projection_selection']:
        rs={r['method']:r for r in probe['records'] if r['layer']==layer}
        for f in probe['families']:
            rank+=f"| {n} | {layer} | {f} | {rs[f+'_top']['mean_nll']-rs[f+'_bottom']['mean_nll']:+.6f} | {rs[f+'_top']['mean_nll']-rs['original']['mean_nll']:+.6f} |\n"
text=f'''# 第一阶段 G0/Q0/Q1：预算审计、误差诊断与固定数量浮点选择

日期：2026-10-25。源码起点 `3a26570`，最终提交见 `completion.json`。本轮只完成预算审计、误差来源诊断和固定每层高精度数量比较；没有进行跨层预算重分配，没有更改kernel、量化器、MixLLM或tiling plans。

## 预算审计

四模型真实vLLM加载后的量化存储均与解析公式逐字节一致。每个候选保持**每个原始unfused projection**的高精度通道数量；不是只保持全模型数量。QKV/gate-up合并后shape、payload、padding与索引字节均相同，因此本轮不消耗额外bit预算。

{budget}

这里的packed合计包括w4/w8/sf4/sf8/lo/hi的唯一tensor storage，检查原FP16 projection参数已删除；没有持久展开权重副本。其他参数/注册buffer单列并按storage去重。KV、Graph pool、allocator缓存及kernel workspace不属于权重表示预算，也没有被算成节省的权重内存。预算相等指各Hy候选对原Hy；MixLLM沿用原mask/RTN，但其scale/zero开销不同，不因此声称与Hy的物理表示字节相等。

## 误差来源：先看量级，保留参考不确定性

相同16个512-token validation窗口、同一HF SDPA模型，对比只量化activation、只量化混合weight、两者同时量化，以及全W4A8/全W8A8端点。所有量化值来自冻结native pack。下表各ΔNLL相对对应dense参考的FP16条件；不同参考的绝对PPL不跨表比较。

{diagnosis}

两种参考都支持：当前混合权重量化的影响明显大于单独activation量化，继续研究weight选择有依据。两类误差并非严格可加；小的interaction或activation差异可能接近软件/native偏差，不应精细归因。

初始0.5B两窗口的软件/native平均NLL≤0.001门禁失败；FP16和FP32参考的失败记录保存在`0.5B/alignment_pilot/`。逐层核查没有发现packed值或FP16表示变化，native全输出仍通过原有atol=0.002、rtol=0.002检查。扩展至全部16窗口后仍记录软件/native差异，未把失败改写成“数值完全等价”。两种dense参考及完整native对照都保留；诊断只解释超过这种差异的效应量级。所有选择优劣均由以下真正的vLLM/native结果决定。

## 固定数量选择：真实native validation PPL

校准：原WT2 train，32×512，主seed2029的token逐项等于原正式校准。开发评测：同版本WT2 validation的全部完整2048-token块，无BOS，每块2047个target；每方法{len(native_example['validation']['blocks'])}块、{native_example['validation']['tokens']}个目标，{native_example['discarded_tail_tokens']}个尾部token未计入。保存逐块NLL，group8与前4块group1控制差≤0.001/token。本轮没有对WT2 test重评分，也没有按test选择方案。

所有native比较固定vLLM0.26.0、Torch2.11.0+cu130、CUDA13.0、RTX5090、FULL配置、Inductor禁用、10GiB KV与原capture sizes。PPL使用prefill路径，不称其为decode PPL。

{quality}

Weight gain为Q4→Q8的weight重构改善；Output gain用量化activation估计输出重构改善；两种loss proxy使用浮点Q4/Q8扰动分别估计可恢复损失。后者是启发式，不是精确Hessian。所有排名在每个原始projection内部进行，采用stable sort。算法定义和代码见`docs/floating_selection_fixed_counts.md`。

{ci}

区间由5000次配对validation块bootstrap获得，是探索性开发集描述，未作多重比较校正。不能把候选筛选后的区间当独立final-test显著性。

![候选比较和校准seed稳定性](selection_comparison.png)

## 校准seed稳定性

预登记规则按3B/7B seed2029的平均ΔNLL选两种score family，入围为 `{', '.join(short)}`。观察到主seed结果的模型差异后，在评测额外小模型seed前，将这两种family的seed2030/2031核查扩展到全部四模型，记录于`robustness_extension.json`；入围规则和评分公式未改。主seed仍是2029，不采用“最好seed”。上图展示3B/7B，下表覆盖全部模型。

{robust}

下表是三个seed两两对照的平均保留比例（交集/固定高精度数量），不是Jaccard；权重元素一列按每通道K_j加权。随机同数量选择的理论期望（通道/权重保留比例）为：{chance_text}。mask稳定性与PPL波动一起解释，不能仅凭重合率判算法优劣。

{stability}

## 通道排名的小规模实际loss验证

在3B/7B中，各选3个确定的非纯q/down投影，单独用冻结native kernel量化，其他projection保持FP16；在8×512 validation窗口上比较score顶部/底部、原选择和随机同数量选择。它用于检查排名的预测性，不是预算内部署方案，也不用于模型速度统计。

{rank}

## 固定kernel/plan的7B速度抽测

原选择和两种入围family的seed2029，B1/8/64、prompt128，使用相同validation prompt、固定生成长度、1 warmup+5 samples、CV≤5%，逐步验证实际FULL。所有原capture sizes/max_num_seqs=128保持不变；没有重新调优。速度比大于1表示更快。这是scatter地址分布变化后的抽测，不是最终16点性能结论。

{speed}

## 产物和边界

- `summary.json`：机器可读汇总；`protocol.json`、`diagnostic_protocol_amendment.json`、`reference_protocol.json`：原协议与明确披露的诊断修订。
- 每模型`budget/`：实际tensor/storage审计；`views/*/selection.json`、`selection_audit.json`：候选及每层固定数量核查；`plans.json`为冻结旧计划。
- `diagnosis.json`、`diagnosis_fp32.json`：全部受控误差条件、native层级检查和软件参考偏差。
- `scores/<seed>/scores.pt`及manifest：完整通道分数、output误差、calibration损失、activation首样本量化统计。
- `native/`：完整validation块NLL、预算/配置hash、实际分派audit及decode原始样本。
- `rank_probe.json`：通道排名的局部loss对照；`logs/`：全部日志与恢复身份信息。
- `provenance.json`、`source_snapshot.tar`、`completion.json`、`delivery_sha256.json`：冻结库、数据、源码、提交和交付校验。

下一步属于阶段1的Q2：可以把本轮有支持的score带到真实bit成本约束的跨层分配，但要重新计算H_j、scale padding和实际shape，再做有界性能调优。本轮没有执行Q2，也没有把某个候选自动部署成项目默认mask。
'''
(root/'report.md').write_text(text,encoding='utf-8')
print('Wrote report and plots')
