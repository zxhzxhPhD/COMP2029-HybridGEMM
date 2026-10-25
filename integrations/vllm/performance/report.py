"""Generate the stage-two research report and stage-three handoff from evidence."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def gm(values):
    return math.exp(sum(math.log(x) for x in values) / len(values))


def cells(path):
    return {(r['batch'], r['context']): r for r in read(path)['cells']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', type=Path, required=True)
    ap.add_argument('--commit', default='pending stage-two commit')
    args = ap.parse_args()
    stage = args.stage
    root = stage.parents[2]
    summary = read(stage / 'summary.json')['models']
    acceptance = read(stage / 'acceptance.json')
    assert acceptance['passed']
    profiles = read(stage / 'profile_comparison.json')['records']
    assert len(profiles) == 12
    mix = read(stage / 'mix_decisions.json')
    inventories = []
    manifests = {}
    for item in summary:
        model = item['model']
        folder = stage / model
        path = folder / 'final/manifest.json'
        manifest = read(path)
        manifest['source_commit'] = args.commit
        write(path, manifest)
        manifests[model] = manifest
        for kind in ['tune_old', 'tune_new', 'tune_mix']:
            directory = folder / kind
            if not directory.exists():
                continue
            counts = Counter()
            measurements = 0
            for log in directory.glob('*.jsonl'):
                with log.open() as f:
                    for line in f:
                        value = json.loads(line)
                        measurements += 1
                        counts[value.get('phase', value.get('stage', 'record'))] += 1
            plan = directory / 'plans.json'
            plans = read(plan) if plan.exists() else {}
            inventories.append({'model': model, 'search': kind,
                                'measurement_records': measurements,
                                'phases': dict(counts),
                                'plans': len(plans.get('hygemm', plans.get('mixllm', []))),
                                'path': str(directory.relative_to(stage))})
    write(stage / 'search_inventory.json', inventories)
    plot_data = read(stage / 'plot_data.json')
    for model, manifest in manifests.items():
        plot_data[model]['manifest'] = manifest
    write(stage / 'plot_data.json', plot_data)
    base = cells(stage / '7B/baseline_full/result.json')
    ablations = []
    for label in ['ablation_no_silu', 'ablation_no_pack', manifests['7B']['candidate']]:
        values = cells(stage / '7B' / label / 'result.json')
        assert set(values) == set(base)
        ablations.append({'label': label, 'cells': len(values),
            'speedup': gm([base[k]['median_ms_per_step'] / values[k]['median_ms_per_step'] for k in base]),
            'small_speedup': gm([base[k]['median_ms_per_step'] / values[k]['median_ms_per_step'] for k in base if k[0] <= 8]),
            'large_speedup': gm([base[k]['median_ms_per_step'] / values[k]['median_ms_per_step'] for k in base if k[0] >= 64]),
            'max_cv': max(v['cv'] for v in values.values())})
    write(stage / 'ablation_summary.json', ablations)
    lines = [
        '# 阶段二：固定质量方案后的性能探索报告', '',
        f'日期：2026-10-26。源码提交：`{args.commit}`。阶段一输入冻结，本阶段完成四模型性能探索、正确性与数值门禁、两组完整模型测试及初末profiling。阶段三尚未启动。', '',
        '## 结论与采用方案', '',
        '主方案仍是输出通道拆分的双kernel。采用按真实O/K/H和token bucket生成的逐层计划、adaptive vector MXFP8 pack，以及保持vLLM数值规则的SwiGLU→MXFP8 producer融合。冻结配置没有采用SIMT GEMV、分区内重排或单kernel TMA切换。MixLLM原device实现保持不变。', '',
        '以下为batch 1/2/4/8/16/32/64/128 × prompt 128/1024的16点几何平均。相对“阶段一交接”的收益同时含实际shape重调优；相对“重调优旧Hy”的收益更能隔离新增实现。首组作为主数字，独立复测只作确认，不择优取两组。', '',
        (stage / 'tables.md').read_text().strip(), '',
        '1.5B与3B相对Mix约为持平，不能把万分位差距解读成胜负。0.5B仍落后于Mix和FP16；7B相对Mix的总体领先不足1%，不宣称明确胜出。此处Hy使用浮点选择与clipping，Mix保持原整数选择/RTN，属于不同量化方法的系统比较，不是同mask或等质量结论。', '',
        '| 模型 | B≤8 vs重调优旧Hy | B≥64 | 最差点延迟比 | payload bits/weight | 持久量化字节 |',
        '|---|---:|---:|---:|---:|---:|']
    for s, a in zip(summary, acceptance['models']):
        lines.append(f"| {s['model']} | {s['small_batch_speedup']:.4f}× | {s['large_batch_speedup']:.4f}× | {s['max_latency_ratio']:.4f} | {a['payload_bits_per_weight']:.6f} | {a['persistent_quantized_bytes']:,} |")
    lines += ['', '所有主组测试点均快于对应重调优旧Hy；收益更多集中在B64/128，小batch仍有明显未解决空间。', '',
        '![相对旧Hy](figures/speedup_hy_pre_f.png)', '',
        '![相对Mix](figures/speedup_mix.png)', '',
        '![相对FP16](figures/speedup_fp16.png)', '',
        '## 固定输入、预算与测量协议', '',
        '平台为RTX5090 SM120、driver595.71.05、CUDA13.0.48、CUTLASS e6233cba、Torch2.11+cu130与vLLM0.26.0 wheel。环境详情和版本hash见`environment.json`、`build/build.json`。FULL Graph、Inductor禁用；FP16 KV、FLASH_ATTN、10GiB KV预算、max_model_len=2049、max_num_seqs=128、max_num_batched_tokens=2048、prefix cache关闭。', '',
        '0.5B冻结Q3新量化器下absolute自由分配；1.5B/3B/7B冻结Q1 signed的配额/身份和输入加权FP4 clipping。E2M1/E4M3、K32/E8M0、FP32累加/FP16输出不变。payload低于各模型阶段一预算且均<5.1bit；持久weights/scales/IDs/padding逐字节大小也不变，没有保存展开FP8权重副本。这里<5.1只指量化projection的payload，不能称为含metadata及FP16模块的模型总位宽。', '',
        '主计时含在线量化、scatter和完整decode框架间隙，排除prefill/mixed/tail；1 warmup+5 samples、CV≤5%、最多3组，GPU实验串行，不锁频或治理额外外部波动。', '',
        f"最终主组及复测共{acceptance['primary_repeat_cells']}个计时点、{acceptance['primary_repeat_samples']}个样本，核对{acceptance['audited_full_graph_steps']:,}个实际FULL Graph步骤；最大CV={acceptance['max_primary_repeat_cv']:.2%}。见`acceptance.json`。", '',
        '开发性能prompt、teacher forcing与PPL都来自validation；最终test/C4本轮未用来选型。历史阶段已测过WT2 test，下一阶段不得称其为从未见过的盲测。', '',
        '## 搜索范围与实现结果', '',
        '| 路线 | 实际尝试与证据 | 结论 |',
        '|---|---|---|',
        '| 精确shape调优 | 保留ID0–27，resident单路筛选→cold精化→两路策略→精确H→整模型；四模型和Mix原候选均调优 | 采用通过模型门禁的逐层/bucket配置；非默认整份替换 |',
        '| 极小N SIMT | ID28–34：packed读取、warp/row、token复用1/2/4、4/8warps、split-K4/8、有限grid-stride | 有极小batch潜力；未过decode数值等价门禁，不进入主配置 |',
        '| 轻量MMA与scale复用 | ID35–44：32-bit scale-word缓存、M32/64/128、N8/16、K128/256及stage/warp组合；保留原TMA/cp.async | 由实际shape和模型门禁选择；资源减少不自动代表模型更快 |',
        '| 两路调度 | serial/concurrent/high-first/high-main；GEMV按CTA分派两精度的单launch策略 | 进入调优候选；最终实际分派见下表 |',
        '| 在线pack | padding融合扩至128行、每lane4值vector pack、mode3按rows≥16切换 | 采用mode3；q和完整sf padding字节一致 |',
        '| Producer融合 | SwiGLU→MXFP8，保留两次FP16舍入、up+beta的signed zero及精确exp；另测近似exp | 采用math0；近似模式拒绝 |',
        '| 分区内布局 | reverse/block_reverse，同时置换packed weights/scales/IDs | 微基准获益没有延续到模型；gate-up reverse约1.0004×，拒绝 |',
        '| 单kernel TMA | 原输出tile边界切换，共用tile/pipeline；新增serving pointer rebind；gate-up T≤8 | 7B 14点约0.9977×，无收益，拒绝；数值通过 |',
        '| 更小TMA tile | M64构建探针 | 当前CUTLASS CTA_Tile/SLayout等价性断言失败；不是证明硬件不能支持M64 |',
        '| Persistent/grouped | 有限grid-stride及两分区同launch的单层/模型候选 | 不进入最终方案；未扩展为跨层持久调度系统 |', '',
        'RMSNorm/residual融合、任意新swizzle、跨层grouped persistent没有全部实现。P1已覆盖，P2选取单consumer的SwiGLU完成实证；其余路线未有足够证据支持扩大工程。阶段目标是有边界的探索及可复现增益，不是穷举或全局最优证明。', '',
        '各模型/搜索记录量见`search_inventory.json`，原始candidate JSONL保留。计数是计时/正确性记录数，单条含多次replay，不等于独立统计样本。两种权重cache proxy只用于提案，最终取舍由整模型决定。', '',
        '| 模型 | 搜索 | 记录数 | 输出shape/bucket计划数 |',
        '|---|---|---:|---:|']
    for r in inventories:
        lines.append(f"| {r['model']} | {r['search']} | {r['measurement_records']} | {r['plans']} |")
    lines += ['', 'Mix原候选调优也执行相同完整矩阵和整模型门禁，没有改device kernel：', '',
        '| 模型 | retune/原配置速度比 | 最差点延迟比 | 最终选择 |', '|---|---:|---:|---|']
    for model, r in mix.items():
        lines.append(f"| {model} | {r['first_gain']:.4f}× | {r['first_max_latency_ratio']:.4f} | {r['selected']} |")
    lines += ['', '## 数值验证与资源', '',
        '新增variant的1,292条全输出/changed-input Graph记录、pack/layout与执行策略检查、极值/短K/tail/空分区/rebind，以及memcheck/synccheck/racecheck均通过。完整检查入口与状态见`acceptance.json`和`checks/`。所有finite FP16值作为group max的pack检查及constant/ramp/negative-zero样本与原版q/sf逐字节相等；SiLU producer另外与已安装vLLM算子比较FP16 bit pattern及最终q/sf。常规CMake全源码构建也通过smoke与producer parity，主计时使用的增量库另有hash。', '',
        '每模型16,320个共同FULL Graph teacher-forced目标位置，四模型共65,280个，最终配置max/mean目标logprob差均为0。各模型validation PPL使用128×2048块（262,016目标），与阶段一值保持一致。数值门槛事前固定为meanabs≤0.002、maxabs≤0.05；不能以greedy token相同代替此门禁。', '',
        '| 模型 | SIMT提案maxabs logprob | meanabs | mean NLL变化 | 主方案maxabs |', '|---|---:|---:|---:|---:|']
    for s in summary:
        q = read(stage / s['model'] / 'quality_new/quality.json')['comparison']
        lines.append(f"| {s['model']} | {q.get('max_abs_logprob', float('nan')):.6f} | {q.get('mean_abs_logprob', float('nan')):.6f} | {q.get('mean_nll_delta', float('nan')):+.8f} | 0 |")
    lines += ['', 'SIMT提案在7B B1约有5.7%速度潜力，0.5B B1也有更大收益，但改变累加顺序。均值NLL变化很小，当前结果只证明未过阶段二数值等价门禁，并没有证明其最终质量劣化；若再采用应作为独立Pareto候选回到质量验证。', '',
        '编译资源记录在`checks/variants.json`，新候选约39–104 registers/thread，无reported local storage；SIMT shared=0，light MMA为10,560–71,808 bytes。SASS证据见`build/sass_counts.json`与构建日志。旧TMA switch ID0 shared=83,968 bytes、168 registers；low/high expected bytes分别11,264/19,456。资源数是编译事实，不等于实测occupancy。正确过滤kernel后ncu仍报ERR_NVGPUCTRPERM，因此不报告stall、achieved occupancy或带宽饱和。', '',
        '最终每token bucket的活跃低/高分区kernel族计数（空分区不计）：', '',
        '| 模型 | Tokens | 活跃分区分派 |', '|---|---:|---|']
    for s in summary:
        for t, counts in s['active_kernel_families_by_tokens'].items():
            lines.append(f"| {s['model']} | {t} | " + ', '.join(f'{k}: {v}' for k, v in counts.items()) + ' |')
    lines += ['', '## 消融与初末profiling', '',
        '7B保持同一GEMM逐层计划，完整16点单因素消融如下。此处单组诊断用于归因，主组合已独立复测；两个增益不能简单相加。', '',
        '| 方案 | 总体vs旧Hy | B≤8 | B≥64 | 最大CV |', '|---|---:|---:|---:|---:|']
    for r in ablations:
        label = {'ablation_no_silu': '仅adaptive pack', 'ablation_no_pack': '仅精确SwiGLU producer'}.get(r['label'], '最终组合')
        lines.append(f"| {label} | {r['speedup']:.4f}× | {r['small_speedup']:.4f}× | {r['large_speedup']:.4f}× | {r['max_cv']:.2%} |")
    lines += ['', '同一classifier重算初末7B B1/8/64、prompt128的12条trace。无profiler控制组为速度来源；Graph meter单独测量，trace内预算按GPU区间并集拆分。GEMM已经含scatter，无法独立报告其store代价；fused producer项同时含SiLU与pack。', '',
        '| 阶段/方法 | B | 无profiler总ms | meter Graph ms | meter Graph外ms | trace GEMM+scatter ms | quant+fused producer ms | trace Graph idle ms | trace/meter偏差 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in profiles:
        b = r['exclusive_graph_budget_ms']
        lines.append(f"| {r['phase']}/{r['kind']} | {r['batch']} | {r['control_ms']:.4f} | {r['unprofiled_graph_ms']:.4f} | {r['unprofiled_outside_graph_ms']:.4f} | {b.get('gemm',0):.4f} | {b.get('activation_quant',0)+b.get('fused_producer_quant',0):.4f} | {b.get('gpu_idle',0):.4f} | {r['trace_vs_meter_relative_delta']:+.2%} |")
    lines += ['', '7B B64的GEMM+scatter仍约3.77ms，主要变化是quant/fused producer从0.337降至0.172ms、Graph idle从0.406降至0.274ms；独立meter的Graph时间从5.736降至5.353ms。B1的GEMM仍约3.33ms，Graph仅减少约0.059ms。因此7B本轮主要改善配套量化/producer及节点开销，没有解决低batch GEMM主体差距。注意旧SiLU位于other_model_gpu，融合后移入fused_producer_quant，不能只看后者上升便声称量化变慢。', '',
        '其余attention/norm/activation/FP16 GEMM等类别保留在`profile_comparison.json`和逐trace accounting文件。Graph外部分含框架、调度及其他操作，不能全部称为CPU纯开销；profile存在扰动时以独立meter/control为准。', '',
        '实测copy读+写带宽约1.527TB/s，7B量化projection weights/scales/IDs一次流式读取约2.825ms；只是乐观传输参照。`probes/traffic_bounds.json`另计FP16 lm-head与prompt+64处的KV读写，并给出GEMM提速1.1/1.25/2倍的Amdahl情景。实际GEMM/attention有不同访问方式、复用和额外流量，不能据此断言已带宽饱和。', '',
        '## 发现并修正的实验问题', '',
        '1. 旧fused probe复用create时stream，导致空Graph和虚假巨大加速。该次结果作废（`probes/invalid_capture_v1.json`），改为launch时current stream并断言capture≥2节点后整组重跑。模型计时和DualLinear调优不受影响。',
        '2. SiLU v1遗漏up+beta=0的signed-zero行为，FP16数值比较会漏掉2个FP8 byte差异。v2保留显式加零，并按int16检查FP16 bit pattern；近似exp模式保留失败记录。',
        '3. 33-token续写不足以覆盖大batch共同FULL区间，改为129 tokens。异步入队又可能改变prefill chunking/KV，质量诊断新增enqueue barrier并核对prefill schedule hash；原unpaired记录保留，7B canonical结果以paired重测替换。性能协议不加barrier。',
        '4. 0.5B空high分区的索引tensor默认浮点dtype，修为显式long；partial调优另存后重跑。这是调优脚本错误，不能当作kernel负结果。',
        '5. 绘图环境与推理环境分开：matplotlib使用基础Python。绘图工具失败不触发GPU实验重跑。', '',
        '## 冻结产物与阶段三交接', '',
        '每模型`final/manifest.json`记录model view、环境选项、plans/hash、mask/hash、bank manifest/hash、kernel/producer binary hash、runtime source hash和比较协议。`final/plans.json`绑定shape/bucket并由loader校验；`final/binding_checks.json`包含换mask/binary/pack mode的拒绝检查。默认环境仍保持旧路径，复现实验必须按manifest显式设置环境，不能只拿源码默认值运行。', '',
        '源代码、构建脚本和复现入口提交Git；模型权重、二进制和大build产物留在记录的远程路径。结果归档不跟随checkpoint符号链接，保留JSON/JSONL、原始失败、logs、trace/sqlite和PNG，逐文件SHA256验证本地副本。', '',
        '下一阶段沿`03_final_evaluation/PLAN.md`补齐共同mask与格式各自选择的交叉消融、任务质量、预算曲线、泛化和适用基线，再冻结最终协议评测。本阶段仅冻结性能主候选，不将validation结果冒充最终test；不自动继续无边界调kernel。RMSNorm融合、跨层persistent和更改累加顺序的SIMT留为明确未采用路线。', '',
        '参考：[CUDA13.0 PTX](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html)；[vLLM0.26 activation源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/csrc/libtorch_stable/activation_kernels.cu)。指令与资源结论以本轮实际编译/SASS为准。', '']
    (stage / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    handoff = {'status': 'complete', 'source_commit': args.commit,
        'run': str(stage.relative_to(root)), 'report': str((stage / 'report.md').relative_to(root)),
        'stage1': '01_quality/handoff.json', 'next_stage': '03_final_evaluation/PLAN.md',
        'acceptance_sha256': digest(stage / 'acceptance.json'),
        'mix_device_sha256': acceptance['mix_device_sha256'],
        'models': {model: {'manifest': str((stage / model / 'final/manifest.json').relative_to(root)),
            'manifest_sha256': digest(stage / model / 'final/manifest.json'),
            'model_path': m['model_path'], 'environment': m['environment'],
            'plans_sha256': m['plans_sha256'], 'selection_sha256': m['selection_sha256'],
            'budget': m['budget'], 'ppl': m['ppl']} for model, m in manifests.items()},
        'final_test_used_this_stage': False,
        'excluded_from_main': ['SIMT GEMV/split-K', 'partition row reversal', 'single-kernel TMA gate-up'],
        'limitations': ['single RTX5090', 'development validation only', 'no hardware performance counters',
                        'not an equal-quality comparison to Mix RTN', 'not a global optimum proof']}
    write(stage / 'handoff.json', handoff)
    write(stage.parents[1] / 'handoff.json', handoff)
    (stage.parents[1] / 'report.md').write_text(
        '# 阶段二结果\n\n性能探索已完成。详见[完整报告](runs/' + stage.name + '/report.md)、'
        '[冻结交接](handoff.json)及[验收](runs/' + stage.name + '/acceptance.json)。\n\n'
        f'源码提交：`{args.commit}`。阶段三尚未启动。\n', encoding='utf-8')
    print('Report and handoff generated:', stage / 'report.md')


if __name__ == '__main__':
    main()
