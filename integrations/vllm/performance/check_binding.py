"""Positive and negative checks for the final, frozen plan bindings (CPU only)."""
import argparse
import copy
import json
import os
from pathlib import Path
from prepare import read,write
ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--model',required=True);args=ap.parse_args()
folder=args.stage/args.model/'final';manifest=read(folder/'manifest.json');os.environ.update(manifest['environment'])
from hygemm_vllm.offline import OfflineConfig
config=read(folder/'model/config.json')['quantization_config']
OfflineConfig(copy.deepcopy(config));checks=['matching configuration accepted']
changed=copy.deepcopy(config);name=next(n for n,ids in changed['selection'].items() if ids)
changed['selection'][name]=changed['selection'][name][1:]
try:OfflineConfig(changed)
except ValueError as e:checks.append(str(e))
else:raise AssertionError('Changed mask was accepted')
original=os.environ['HYGEMM_LIBRARY'];os.environ['HYGEMM_LIBRARY']='/root/autodl-tmp/20261025_101734_small_batch/build/libhygemm_serving.so'
try:OfflineConfig(copy.deepcopy(config))
except ValueError as e:checks.append(str(e))
else:raise AssertionError('Changed binary was accepted')
os.environ['HYGEMM_LIBRARY']=original
old=os.environ['HYGEMM_PACK_MODE'];os.environ['HYGEMM_PACK_MODE']='0' if old!='0' else '2'
try:OfflineConfig(copy.deepcopy(config))
except ValueError as e:checks.append(str(e))
else:raise AssertionError('Changed execution mode was accepted')
os.environ['HYGEMM_PACK_MODE']=old
write(folder/'binding_checks.json',{'passed':True,'checks':checks})
