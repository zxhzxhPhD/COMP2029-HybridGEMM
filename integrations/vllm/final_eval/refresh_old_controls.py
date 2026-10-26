"""Reuse the accepted stage-two old implementation's tuning in 2x2 controls."""
import argparse
from copy import deepcopy
from pathlib import Path
from final_common import *

ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);args=ap.parse_args();s=args.stage
assert not (s/'freeze/current.json').exists()
methods=read(s/'methods.json');records=[]
for model in ['3B','7B']:
    accepted=read(S2/model/'final/manifest.json');label=accepted['baseline']
    config=read(S2/model/label/'model/config.json')['quantization_config']
    if 'decode_plans' not in config:
        records.append({'model':model,'accepted_old_label':label,'action':'already uses the same inherited reference plan'});continue
    source=Path(config['decode_plans']);proposal=read(source)
    for name in ['K0-I','K0-F','K0-F-Q']:
        spec=methods[model][name];cfg=read(Path(spec['model_path'])/'config.json')['quantization_config']
        path=Path(cfg['decode_plans']);existing=read(path);layers=read(spec['shape_manifest'])['layers'];rows=[]
        for layer in layers:
            for tokens in BATCHES:
                chosen=next(r for r in sorted(proposal['hygemm'],key=lambda r:r['tokens']) if
                    (r['out'],r['k'],r['high'])==(layer['out'],layer['k'],layer['high']) and r['tokens']>=tokens)
                rows.append({**chosen,'layer':layer['layer'],'tokens':tokens})
        unique={}
        for row in rows:unique.setdefault((row['out'],row['k'],row['high'],row['tokens']),{k:v for k,v in row.items() if k!='layer'})
        plan={'binding':deepcopy(existing['binding']),'hygemm':list(unique.values()),'layer_plans':rows}
        plan['binding']['accepted_stage2_old_plan_sha256']=sha(source)
        write(path,plan)
        records.append({'model':model,'method':name,'accepted_old_label':label,'source_plan':str(source),
            'source_sha256':sha(source),'plan_sha256':sha(path),'action':'same exact shapes; reuse accepted retuned old control instead of untuned inherited plan'})
write(s/'old_control_provenance.json',records)
print('Old implementation controls aligned with accepted stage-two baseline',records,flush=True)
