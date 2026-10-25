"""Render complete, non-cherry-picked result tables from native measurements."""
import argparse
import json
import math
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def table(headers,rows):
    return ['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',*['| '+' | '.join(map(str,r))+' |' for r in rows],'']
def f(x):return f'{x:.6f}'

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--q1',type=Path)
args=ap.parse_args();summary=read(args.out/'summary.json');q1=args.q1 or Path(summary['protocol']['q1'])
lines=['# 完整结果表','', '所有 PPL 均为同一 WT2 validation 的 128×2048 完整块、262016 个目标 token；越低越好。主 seed 为 2029，校准为 32×512。','']
methods=['integer','random_2029','random_2030','random_2031','weight_gain','output_gain','loss_abs_gain','loss_signed_gain','uniform_abs']
for model,result in summary['models'].items():
    lines += [f'## {model}：Q1 → Q2','']
    old={p.stem:read(p) for p in (q1/model/'native').glob('*.json')};base=old['original']['validation']['ppl'];rows=[]
    for method in methods:
        prior=old.get('original' if method=='integer' else method);new=result['q1_to_q2'].get(method)
        rows.append([method,f(prior['validation']['ppl']) if prior else '—',
            f(prior['validation']['ppl']-base) if prior else '—',f(new['ppl']) if new else '—',
            f(new['vs_q1_same_method']['delta_ppl']) if new and new['vs_q1_same_method'] else '—'])
    lines+=table(['方法','Q1 PPL','Q1−原 Hy','Q2 PPL','Q2−Q1'],rows)
    lines+=[f"FP16={f(old['reference_fp16']['validation']['ppl'])}；Mix-I={f(old['reference_mixllm']['validation']['ppl'])}。",'']
    rows=[]
    for family,stages in result['seed_robustness'].items():
        for stage,values in stages.items():
            rows.append([family,stage,*map(f,values['ppl']),f(values['geomean_ppl']),str(values['all_seeds_better_than_original'])])
    lines+=table(['方法','阶段','seed2029','seed2030','seed2031','几何均值','三 seed 均优于原 Hy'],rows)
    design=args.out/model/'q3_design.json'
    if design.exists():
        lines += [f'## {model}：Q3 固定 mask 消融','']
        d=read(design);rows=[]
        for mask in d['fixed_mask_keys']+d['extra_masks']:
            original=old['original'] if mask=='q1_integer' else old[mask[3:]] if mask.startswith('q1_') else read(args.out/model/'native'/f'{mask}.json')
            row=[mask,f(original['validation']['ppl'])]
            for mode in ['mse_clip','act_mse_clip']:
                key=f'q3_{mode}__{mask}';row.append(f(result['rows'][key]['ppl']) if key in result['rows'] else '—')
            rows.append(row)
        lines+=table(['固定 mask','RTN','MSE clipping','输入二阶矩加权 clipping'],rows)
    joint=[(name,row) for name,row in result['rows'].items() if name.startswith(('q3_joint_','q3_fixed_reselect_','q2_cal128'))]
    if joint:
        lines += [f'## {model}：重新评分和校准量敏感性','']
        lines+=table(['方法','PPL','相对原 Hy ΔPPL'],[[n,f(r['ppl']),f(r['vs_q1_original']['delta_ppl'])] for n,r in joint])
lines+=['## 主候选与预算','']
rows=[]
for model,result in summary['models'].items():
    name=result['primary_winner'];row=result['rows'][name]
    spec=read(args.out/model/'views'/name/'view.json') if (args.out/model/'views'/name/'view.json').exists() else None
    rows.append([model,name,f(row['ppl']),f(result['primary_winner_vs_mix']['delta_ppl']),
        f(spec['payload_bits_per_weight']) if spec else f(row['payload_bits_per_weight']) if 'payload_bits_per_weight' in row else 'Q1同预算',
        spec['budget']['persistent_quantized_bytes'] if spec else row.get('physical_bytes','Q1同字节'),
        spec['physical_byte_delta_vs_q1'] if spec else row.get('physical_byte_delta_vs_q1',0)])
lines+=table(['模型','主候选','PPL','ΔPPL vs Mix','payload bit/w','物理 bytes','Δbytes vs Q1'],rows)
perf=args.out/'performance_complete.json'
if perf.exists():
    lines+=['## 7B decode 初步交接','']
    def measurement(path):return read(args.out/'7B/performance'/Path(path).parent.name/Path(path).name)
    roles=read(perf)['roles'];old=measurement(roles['original']['inherited']);rows=[]
    for role,spec in roles.items():
        result=measurement(spec.get('accepted_measurement',spec['inherited']))
        for a,b in zip(old['cells'],result['cells']):
            rows.append([role,b['batch'],b['context'],f(b['median_ms_per_step']),
                f(a['median_ms_per_step']/b['median_ms_per_step']),f(b['cv']*100)])
    lines+=table(['候选','batch','prompt','ms/step','相对原 Hy 速度比','CV %'],rows)
(args.out/'tables.md').write_text('\n'.join(lines),encoding='utf-8')
print('Wrote',args.out/'tables.md')
