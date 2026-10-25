"""Archive results without following model/checkpoint symlinks or copying weights."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
allowed={'.json','.jsonl','.md','.log','.txt','.gz','.png','.pdf','.sqlite','.nsys-rep','.ncu-rep'}
files=[]
for p in sorted(args.stage.rglob('*')):
    if p.is_symlink() or not p.is_file() or p.resolve()==args.out.resolve():continue
    if p.suffix not in allowed and p.name!='tokens.pt':continue
    files.append(p)
manifest=[]
with tarfile.open(args.out,'w:gz') as tar:
    for p in files:
        with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        rel=str(p.relative_to(args.stage));manifest.append({'path':rel,'bytes':p.stat().st_size,'sha256':digest})
        tar.add(p,arcname=rel,recursive=False)
(args.out.with_suffix('.manifest.json')).write_text(json.dumps({'files':manifest,'excluded':'checkpoint symlinks, safetensors, object/static/shared libraries; binaries remain at their recorded remote paths'},indent=2))
print('Archived',len(files),'files',args.out.stat().st_size,'bytes',flush=True)
