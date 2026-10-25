"""Final completeness gate over the frozen plans, measured paths and payloads."""
import argparse
from pathlib import Path
from prepare import read,write,sha
ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);args=ap.parse_args();stage=args.stage
handoff=read(stage/'stage1_handoff.json');records=[];sample_count=0;graph_steps=0;max_cv=0
policies={'serial':0,'concurrent':1,'high_first':2,'high_main':3,'gemv_pair1':4,'gemv_pair2':5,'gemv_pair4':6}
for model in ['0.5B','1.5B','3B','7B']:
    folder=stage/model;manifest=read(folder/'final/manifest.json');original=handoff['models'][model]
    assert manifest['selection_sha256']==original['selection_sha256']
    assert manifest['budget']==original['budget'] and original['payload_bits_per_weight']<5.1
    assert sha(folder/'final/plans.json')==manifest['plans_sha256']
    assert sha(folder/'final/model/config.json')==manifest['config_sha256']
    assert read(folder/'final/binding_checks.json')['passed']
    plan=read(folder/'final/plans.json');by_layer={(r['layer'],r['tokens']):r for r in plan['layer_plans']}
    primary=read(folder/manifest['candidate']/'result.json');repeat=read(folder/manifest['repeat']/'result.json')
    expected={(t,p) for t in [1,2,4,8,16,32,64,128] for p in [128,1024]}
    for label,result in [('primary',primary),('repeat',repeat)]:
        assert result['completed'] and result['graph']=='FULL' and result['inductor'] is False
        assert {(r['batch'],r['context']) for r in result['cells']}==expected
        assert result['budget']['persistent_quantized_storage_bytes']==original['budget']['persistent_quantized_bytes']
        for row in result['cells']:
            assert row['stable'] and row['cv']<=.05
            max_cv=max(max_cv,row['cv']);samples=row['attempts'][-1]['samples'];assert len(samples)==5
            for sample in samples:
                assert sample['steps']==sample['graph_steps']>=32
                sample_count+=1;graph_steps+=sample['graph_steps']
        for layer in result['audit']:
            for t,used in layer['plans_used'].items():
                if int(t)>128:continue
                candidates=[p for (name,_),p in by_layer.items() if name==layer['name'] and p['tokens']>=int(t)]
                p=min(candidates,key=lambda p:p['tokens'])
                assert used==[p['tokens'],p['low_id'],p['high_id'],policies[p['policy']]],(model,layer['name'],t,used,p)
    quality=read(folder/manifest['quality']/'quality.json');assert quality['comparison']['passed']
    assert all('prefill_schedule_sha256' in r for r in quality['records'])
    assert read(folder/'validation_ppl.json')['passed']
    records.append({'model':model,'payload_bits_per_weight':original['payload_bits_per_weight'],
        'persistent_quantized_bytes':original['budget']['persistent_quantized_bytes'],
        'quality':quality['comparison'],'ppl':manifest['ppl'],'passed':True})
checks={}
for name in ['variants','pack_policy','variants_memcheck','variants_synccheck','variants_racecheck','pack_policy_memcheck',
             'pack_maxima','row_order','extremes','fused_rebind','fused_memcheck','fused_synccheck','fused_racecheck',
             'cmake_variants_smoke']:
    value=read(stage/'checks'/(name+'.json'));assert value['passed'];checks[name]=True
for name in ['producer','producer_memcheck','producer_synccheck','producer_racecheck','cmake_producer']:
    value=read(stage/'checks'/(name+'.json'));assert 0 in value['passing_math_modes'];checks[name]=True
for name in ['variants_memcheck','variants_synccheck','variants_racecheck','pack_policy_memcheck',
             'producer_memcheck','producer_synccheck','producer_racecheck','fused_memcheck','fused_synccheck','fused_racecheck']:
    text=(stage/(name+'.log')).read_text()
    assert '0 errors' in text or '0 hazards displayed' in text,name
mix=Path('/root/autodl-tmp/20261024_235427_formal_decode/mixllm_build/extension/mixllm_sm120.so')
assert sha(mix)=='1d6190c2bb2effb24d1c12a5438ce051dc0967ec8cdf71e3120e207a38d426da'
write(stage/'acceptance.json',{'passed':True,'models':records,'checks':checks,'primary_repeat_cells':128,
    'primary_repeat_samples':sample_count,'audited_full_graph_steps':graph_steps,'max_primary_repeat_cv':max_cv,
    'mix_device_sha256':sha(mix),'mix_device_unchanged':True,'final_test_not_used':True,
    'scope':'Only accepted primary/repeat timing groups are counted here; all exploratory/rejected runs remain separately available.'})
