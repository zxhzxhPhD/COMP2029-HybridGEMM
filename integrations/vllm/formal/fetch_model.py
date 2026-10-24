"""Download one explicitly pinned public model snapshot; retain a file manifest."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download

ap=argparse.ArgumentParser()
ap.add_argument('--repo',required=True)
ap.add_argument('--revision',required=True)
ap.add_argument('--out',type=Path,required=True)
args=ap.parse_args()
snapshot_download(args.repo,revision=args.revision,local_dir=args.out,
                  allow_patterns=['*.json','*.safetensors','*.txt','*.model','merges.txt','vocab.json','LICENSE','README.md'],
                  max_workers=4)
files={}
for path in sorted(args.out.iterdir()):
    if not path.is_file():continue
    h=hashlib.sha256()
    with path.open('rb') as f:
        while data:=f.read(8*1024*1024):h.update(data)
    files[path.name]={'bytes':path.stat().st_size,'sha256':h.hexdigest()}
(args.out/'asset_manifest.json').write_text(json.dumps({'repo':args.repo,'revision':args.revision,'files':files},indent=2))
print('Pinned snapshot ready:',args.repo,args.revision,flush=True)
