"""Bounded native-MMA tiling proposals, accepted only by validation decode gates."""
import argparse
from copy import deepcopy
import math
import os
from pathlib import Path
import subprocess
from final_common import *


def cells(path):
    rows=[r for r in read(path)['operations']['speed'] if r.get('feasible',True)]
    assert all(r['stable'] and r['cv']<=.05 for r in rows)
    return {(r['batch'],r['context']):r for r in rows}
def gain(a,b,t):
    ratios=[a[k]['median_ms_per_step']/b[k]['median_ms_per_step'] for k in a if k[0]==t and k in b]
    assert len(ratios)==2
    return {'speedup':math.prod(ratios)**.5,'max_latency_ratio':1/min(ratios)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--model',required=True)
    ap.add_argument('--method',required=True);ap.add_argument('--tokens',default='1,2,4,8,16,32,64,128');args=ap.parse_args()
    stage=args.stage;folder=stage/args.model;out=stage/'tuning'/args.model/args.method;out.mkdir(parents=True,exist_ok=True)
    methods=read(stage/'methods.json');spec=methods[args.model][args.method]
    cfg=read(Path(spec['model_path'])/'config.json');planfile=Path(cfg['quantization_config']['decode_plans'])
    if (out/'decision.json').exists():
        decision=read(out/'decision.json');assert decision['final_plan_sha256']==sha(planfile);return
    source=Path(__file__).resolve().parents[1];env={**os.environ,**spec['environment']};tokens=list(map(int,args.tokens.split(',')))
    def run(label,command):
        write(out/'status.json',{'phase':label,'command':command})
        with (out/(label+'.log')).open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    baseline=read(planfile);shape=read(spec['shape_manifest']);compat=out/'inputs';compat.mkdir(exist_ok=True)
    write(compat/'reference_cases.json',{'layers':shape['layers']});write(compat/'reference_plans.json',baseline)
    write(compat/'inputs.json',{args.model:{'source_root':str(compat),'shape_manifest':spec['shape_manifest'],
        'selection_sha256':canonical_sha(cfg['quantization_config']['selection'])}})
    run('micro',[PYTHON,str(source/'performance/tune.py'),'--stage',str(compat),'--model',args.model,
        '--out',str(out/'micro'),'--family','old','--tokens',args.tokens])
    # Only established native MMA IDs 0-27 are searched. Already accepted
    # inherited IDs can remain; the rejected SIMT family is never introduced.
    proposal=read(out/'micro/plans.json');mapping={(r['out'],r['k'],r['high'],r['tokens']):r for r in proposal['hygemm']}
    original_layers=baseline.get('layer_plans')
    if original_layers is None:
        original_layers=[]
        for layer in shape['layers']:
            for t in BATCHES:
                row=next(r for r in sorted(baseline['hygemm'],key=lambda r:r['tokens']) if
                    (r['out'],r['k'],r['high'])==(layer['out'],layer['k'],layer['high']) and r['tokens']>=t)
                original_layers.append({**row,'tokens':t,'layer':layer['layer']})
    def make_plan(adopt):
        plan=deepcopy(baseline);plan['layer_plans']=deepcopy(original_layers);unique={}
        for row in plan['layer_plans']:
            if row['tokens'] in adopt:
                new=mapping[(row['out'],row['k'],row['high'],row['tokens'])]
                row.update({k:new[k] for k in ['low_id','high_id','policy']})
            unique.setdefault((row['out'],row['k'],row['high'],row['tokens']),{k:v for k,v in row.items() if k!='layer'})
        plan['hygemm']=list(unique.values());plan['binding']['stage3_native_mma_search_sha256']=sha(out/'micro/plans.json')
        return plan
    methods=read(stage/'methods.json')
    aliases={}
    for name,plan in [('baseline',baseline),('candidate',make_plan(tokens))]:
        path=out/(name+'_plans.json');write(path,plan);config=deepcopy(cfg);config['quantization_config']['decode_plans']=str(path)
        model=out/(name+'_model');view(spec['model_path'],model,config)
        alias='tune-'+args.method+'-'+name;aliases[name]=alias
        methods[args.model][alias]={**deepcopy(spec),'model_path':str(model),'config_sha256':sha(model/'config.json'),
            'development_only':True,'status':'tiling gate alias'}
    write(stage/'methods.json',methods)
    def timing(which,label,buckets):
        alias=aliases[which]
        run(label,[PYTHON,str(source/'final_eval/native_run.py'),'--stage',str(stage),'--model',args.model,
            '--method',alias,'--mode','development','--operations','speed','--batches',','.join(map(str,buckets)),
            '--contexts','128,1024','--label',label])
        return cells(stage/'development'/args.model/alias/label/'result.json')
    a=timing('baseline','baseline',tokens);b=timing('candidate','candidate',tokens)
    first={str(t):gain(a,b,t) for t in tokens if sum(k[0]==t for k in a)==2}
    adopt=[int(t) for t,r in first.items() if r['speedup']>=1.01 and r['max_latency_ratio']<=1.02]
    second={}
    if adopt:
        c=timing('candidate','repeat',adopt)
        second={str(t):gain(a,c,t) for t in adopt}
        adopt=[t for t in adopt if second[str(t)]['speedup']>1 and second[str(t)]['max_latency_ratio']<=1.02]
    final=make_plan(adopt);write(planfile,final)
    methods=read(stage/'methods.json');methods[args.model][args.method]['tiling_decision']=str(out/'decision.json')
    methods[args.model][args.method]['status']='actual-shape tiling gated on development validation; final freeze pending'
    if args.method=='Hy-F-Q' and 'Hy-F' in methods[args.model]:
        # Quantizer-only comparisons keep the identical accepted compute plan.
        peer_cfg=read(Path(methods[args.model]['Hy-F']['model_path'])/'config.json')
        assert peer_cfg['quantization_config']['selection']==cfg['quantization_config']['selection']
        peer=deepcopy(final);peer['binding']['bank_manifest_sha256']=sha(peer_cfg['quantization_config']['q4_bank'])
        write(peer_cfg['quantization_config']['decode_plans'],peer)
        methods[args.model]['Hy-F']['tiling_shared_with']='Hy-F-Q'
    write(stage/'methods.json',methods)
    write(out/'decision.json',{'completed':True,'model':args.model,'method':args.method,'adopted_buckets':adopt,
        'first':first,'repeat':second,'final_plan_sha256':sha(planfile),'micro_reference':'all output rows/tokens, atol .01 rtol .003',
        'candidate_scope':'established native MMA 0-27 plus inherited accepted configuration; no SIMT',
        'timing_input':'development validation only','quality_selection':False})
    write(out/'status.json',{'phase':'complete'});print('Tiling gated',args.model,args.method,adopt,flush=True)


if __name__=='__main__':main()
