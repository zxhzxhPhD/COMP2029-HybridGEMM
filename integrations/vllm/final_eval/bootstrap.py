"""Prepare an isolated CPU evaluation tool environment and pinned public assets."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True)
    ap.add_argument('--mode',choices=['tools','model'],required=True);args=ap.parse_args()
    stage=args.stage;stage.mkdir(parents=True,exist_ok=True)
    if args.mode=='tools':
        env=Path('/root/autodl-tmp/hygemm-eval-tools')
        if not (env/'bin/python').exists():
            subprocess.run(['/root/miniconda3/bin/python','-m','venv','--system-site-packages',str(env)],check=True)
        subprocess.run([str(env/'bin/python'),'-m','pip','install','lm_eval==0.4.13'],check=True)
        frozen=subprocess.check_output([str(env/'bin/python'),'-m','pip','freeze'],text=True)
        (stage/'evaluation_tools_freeze.txt').write_text(frozen)
        write(stage/'tools_environment.json',{'python':str(env/'bin/python'),'lm_eval':'0.4.13',
            'gpu_inference_environment_unchanged':True,'freeze_sha256':hashlib.sha256(frozen.encode()).hexdigest()})
    else:
        from huggingface_hub import HfApi,snapshot_download
        repo='mistralai/Mistral-7B-v0.3';api=HfApi();info=api.model_info(repo)
        dest=Path('/root/autodl-tmp/models/Mistral-7B-v0.3')
        names=[s.rfilename for s in info.siblings if '/' not in s.rfilename]
        allowed=[n for n in names if n.endswith(('.json','.txt','.model')) or n in ['README.md','LICENSE','tokenizer.model.v3'] or n.startswith('model-') and n.endswith('.safetensors') or n=='model.safetensors']
        write(stage/'mistral_download.json',{'repo':repo,'revision':info.sha,'path':str(dest),'files':allowed,'status':'downloading'})
        snapshot_download(repo,revision=info.sha,local_dir=dest,allow_patterns=allowed,max_workers=4)
        records={}
        for name in allowed:
            p=dest/name
            with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
            records[name]={'bytes':p.stat().st_size,'sha256':digest}
        value={'repo':repo,'revision':info.sha,'path':str(dest),'files':records,'status':'complete'}
        write(dest/'asset_manifest.json',value);write(stage/'mistral_download.json',value)


if __name__=='__main__':main()
