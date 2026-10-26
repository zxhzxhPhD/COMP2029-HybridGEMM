"""Prepare explicit mask/quantizer/implementation ablations before final freeze."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
from final_common import *
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'quality'))
import importlib.util
spec=importlib.util.spec_from_file_location('quality_common',Path(__file__).resolve().parents[1]/'quality/common.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);args=ap.parse_args()
    stage=args.stage;stage.mkdir(parents=True,exist_ok=True)
    handoff=read(ROOT/'02_performance/handoff.json')
    assert handoff['source_commit']=='47f2ec37dfc4ea4a1959093af2e77565487a0fea'
    assert sha(MIX)==handoff['mix_device_sha256']
    protocol={'status':'3A preparation; not frozen','models':MODELS,'core_methods':METHODS,
        'main_speed':{'batches':BATCHES,'prompts':[128,1024],'new_tokens':129,'samples':5,'warmup':1,'cv_limit':.05,'max_groups':3},
        'core_qwen_speed_cells':320,'second_family_speed_cells':80,
        'ppl':{'wt2':'all nonoverlapping 2048 blocks, no BOS, 2047 targets/block',
               'c4':'256 nonoverlapping 2048-token windows from preregistered validation documents',
               'batch':8,'batch1_control_blocks':4,'pair_bootstrap_samples':1000,'pair_bootstrap_seed':3071},
        'tasks':{'names':['hellaswag','arc_challenge','piqa'],'shots':0,'lm_eval':'0.4.13','full_public_split':True,'chat_template':False,'seeds':[2029,2029,2029]},
        'decode_quality':{'models':['3B','7B'],'batches':[1,8,32,64,128],'prompts':[128,1024],
             'teacher_tokens':129,'common_full_targets_per_request':32,'distribution_request':0,
             'distribution_target_indices':[97,105,113,121],'KL':'FP16 || candidate',
             'data':'C4 holdout, never used for selection','paired_prefill':True},
        'ablation':{'models':['3B','7B'],'masks':['I','F'],'implementations':['K0','K1'],
             'quantizer':'RTN fixed for 2x2; separately F RTN vs F clipped','batches':[1,8,32,64,128],'prompts':[128,1024]},
        'budget':{'models':['3B','7B'],'alpha':[0,.25,.5,.75,1],
             'rules':['integer_cost','frozen_float_fixed_quota','uniform_float'],
             'random_diagnostic_alpha':1,'quantizer':'same frozen act_mse_clip bank for every curve',
             'float_rule':'scale original per-layer counts by alpha; signed within layer, cost-feasible residual fill',
             'speed_batches':[1,8,64],'speed_prompts':[128,1024],'same_model_original_payload_cap':True},
        'robustness':{'models':['3B','7B'],'seeds':[2029,2030,2031],'baseline':[32,512],
             'one_factor_settings':[[128,512],[32,2048]],'distribution':'C4 train, distinct from holdout',
             'primary_metric':'No selecting best seed; report every calibration setting',
             'bank_policy':'Recompute signed RTN scores and input moments/clipping bank for every setting; freeze the per-layer high counts and algorithm.',
             'paired_windows':'Share 128 sampled start positions; repair only positions too close to corpus end for the 2048-token extension.'},
        'extensions':{'long_context':{'model':'7B','prompts':[2048,4096],'batches':[1,8,32],'kv_bytes':10*1024**3},
             'non_bucket':[3,12,24,48],'prefill':{'model':'7B','batches':[1,8],'prompts':[128,1024,4096]},
             'endpoints':['W4A8 same clipped bank','W8A8 native MX endpoint, over budget'],
             'optional_baselines':'Audit pinned vLLM native FP8/NVFP4/INT4 and Mix paper PTQ feasibility before freeze; publish unsupported/skipped reasons'},
        'freeze_rule':'No final WT2/C4/task scoring until all adapters, dataset manifests, plans, scripts and dry runs are frozen; any post-freeze repair creates a new manifest and reruns affected cells.',
        'main_impl':'Stage-two native binaries; Mix original binary unchanged','parent_handoff_sha256':sha(ROOT/'02_performance/handoff.json')}
    write(stage/'protocol_draft.json',protocol)
    records={}
    for model in MODELS[:4]:
        folder=stage/model;records[model]={}
        main=read(S2/model/'final/manifest.json');final_cfg=read(Path(main['model_path'])/'config.json')
        metadata=read(Q1/model/'original_calibration_manifest.json')
        selections={'I':read(Q1/model/'original_selection.json'),'F':final_cfg['quantization_config']['selection']}
        for method in METHODS+(['K0-I','K0-F','K0-F-Q'] if model in ['3B','7B'] else []):
            dest=folder/'models'/method;old=method.startswith('K0-')
            if method in ['FP16','Mix-I']:
                source=FORMAL/model/'models'/('fp16' if method=='FP16' else 'mixllm')
                cfg=read(source/'config.json');budget=None;layers=None
            else:
                source=Path(main['model_path']);cfg=deepcopy(final_cfg)
                mask='I' if method.endswith('-I') else 'F';clip=method.endswith('-Q')
                selection=selections[mask]
                bank=(Path(final_cfg['quantization_config']['q4_bank']) if clip else S1/model/'banks/native_rtn/manifest.json')
                layers=legacy.fused_cases(selection,metadata['shapes'],cfg['num_hidden_layers']);budget=legacy.analytic_budget(layers)
                cfg['quantization_config']={'quant_method':'hygemm_offline','checkpoint_format':'fp16_rtn','selection':selection,
                    'reference_plans':str(S1/model/'reference_plans.json'),'reference_cases':str(S1/model/'reference_cases.json'),'q4_bank':str(bank)}
                # Materialize known correct layer plans for the exact mask shape;
                # new-mask tuning proposals remain development-only until gated.
                if mask=='F' and not old:
                    plan=deepcopy(read(S2/model/'final/plans.json'))
                else:
                    references={r['layer']:r for r in read(S1/model/'reference_cases.json')['layers']}
                    base=read(S1/model/'reference_plans.json')['hygemm'];layer_plans=[]
                    for layer in layers:
                        for t in BATCHES:
                            r=next(r for r in sorted(base,key=lambda r:r['tokens']) if (r['out'],r['k'],r['high'])==(layer['out'],layer['k'],references[layer['layer']]['high']) and r['tokens']>=t)
                            layer_plans.append({**r,'layer':layer['layer'],'high':layer['high'],'tokens':t})
                    unique={}
                    for r in layer_plans:unique.setdefault((r['out'],r['k'],r['high'],r['tokens']),{k:v for k,v in r.items() if k!='layer'})
                    plan={'hygemm':list(unique.values()),'layer_plans':layer_plans}
                plan['binding']={'library_sha256':sha(OLD if old else NEW),'layout':'MX K32 E8M0 native physical scales v1',
                    'selection_canonical_sha256':canonical_sha(selection),'bank_manifest_sha256':sha(bank),
                    'stage3_method':method,'shape_sha256':canonical_sha(layers)}
                planfile=folder/'plans'/(method+'.json');write(planfile,plan);cfg['quantization_config']['decode_plans']=str(planfile)
                write(folder/'selections'/(method+'.json'),selection)
                write(folder/'shapes'/(method+'.json'),{'layers':layers,'budget':budget,'weights':metadata['weights']})
            view(source,dest,cfg)
            records[model][method]={'model_path':str(dest),'environment':environment(old),'budget':budget,
                'config_sha256':sha(dest/'config.json'),'implementation':'K0' if old else 'K1',
                'source_checkpoint':str(FORMAL/model/'models/fp16'),'shape_manifest':str(folder/'shapes'/(method+'.json')) if layers else None,
                'status':'prepared; dry-run and freeze required'}
    write(stage/'methods.json',records)
    print('Prepared',sum(map(len,records.values())),'Qwen method views; no final scores computed',flush=True)


if __name__=='__main__':main()
