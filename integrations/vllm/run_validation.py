"""Sequential small-model integration suite; all artifacts go to --out."""
import argparse
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--selection',type=Path,required=True)
    ap.add_argument('--hygemm-library',type=Path,required=True)
    ap.add_argument('--mixllm-library',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--skip-graphs',action='store_true')
    ap.add_argument('--sanitizers',action='store_true')
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).resolve().parent
    env=dict(os.environ, HYGEMM_LIBRARY=str(args.hygemm_library.resolve()),
             MIXLLM_LIBRARY=str(args.mixllm_library.resolve()), VLLM_PLUGINS='hygemm_quantization',
             VLLM_NO_USAGE_STATS='1', VLLM_WORKER_MULTIPROC_METHOD='spawn')
    import torch
    assert md.version('vllm')=='0.26.0' and torch.version.cuda=='13.0'
    meta={'vllm':md.version('vllm'),'torch':torch.__version__,'cuda':torch.version.cuda,
          'gpu':torch.cuda.get_device_name(),'python':sys.version,
          'hygemm_sha256':hashlib.sha256(args.hygemm_library.read_bytes()).hexdigest(),
          'mixllm_sha256':hashlib.sha256(args.mixllm_library.read_bytes()).hexdigest(),
          'source_sha256':{str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in source.rglob('*.py')}}
    (args.out/'environment.json').write_text(json.dumps(meta,indent=2))
    (args.out/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))

    def run(name,script,*flags,sanitizer=False):
        cmd=[sys.executable,str(source/script),*map(str,flags)]
        if sanitizer:cmd=['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','99',*cmd]
        with (args.out/f'{name}.log').open('w') as f:
            f.write('$ '+' '.join(cmd)+'\n');f.flush()
            subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=900)
        print(name,'passed',flush=True)

    run('prepare','prepare_smoke.py','--model',args.model,'--selection',args.selection,'--out',args.out/'models')
    for kind in ['hygemm','mixllm']:
        run('components_'+kind,'test_components.py','--kind',kind,'--out',args.out/f'components_{kind}.json')
        if args.sanitizers:
            run('memcheck_'+kind,'test_components.py','--kind',kind,'--out',args.out/f'memcheck_{kind}.json',sanitizer=True)
    run('export_mixllm','export_mixllm_checkpoint.py','--model',args.model,'--selection',args.selection,'--out',args.out/'models/mixllm_v1')
    for kind,path in [('fp16',args.model),('hygemm',args.out/'models/hygemm'),('mixllm',args.out/'models/mixllm_v1')]:
        for graph in ([False] if args.skip_graphs else [False,True]):
            name='smoke_'+kind+('_graph_no_compile' if graph else '')
            run(name,'smoke.py','--kind',kind,'--model',path,'--out',args.out/f'{name}.json',
                *(['--graph','--no-compile'] if graph else []))
    # Validate the original five-tensor loader against the same RTN weights
    # prepared after vLLM's standard fused projection loading.
    run('smoke_mixllm_online','smoke.py','--kind','mixllm','--model',args.out/'models/mixllm',
        '--out',args.out/'smoke_mixllm_online.json')
    run('summary','summarize.py','--out',args.out)


if __name__=='__main__':main()
