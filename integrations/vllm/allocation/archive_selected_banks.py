"""Preserve the adopted FP4 banks locally without copying all rejected banks."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--archive',type=Path,required=True)
args=ap.parse_args();selected=json.loads((args.root/'selected.json').read_text());records=[]
with tarfile.open(args.archive,'w:gz',compresslevel=1) as tar:
    for model,record in selected['models'].items():
        if not record['bank']:continue
        manifest_path=Path(record['bank']);manifest=json.loads(manifest_path.read_text())
        for spec in manifest['layers'].values():
            path=manifest_path.parent/spec['file'];actual=hashlib.sha256(path.read_bytes()).hexdigest()
            assert actual==spec['sha256'];relative=str(path.relative_to(args.root))
            tar.add(path,arcname=relative,recursive=False);records.append({'path':relative,'sha256':actual,'bytes':path.stat().st_size})
ledger={'files':records,'role':'Only primary selected full-row FP4 banks; rejected bank files remain remote with per-file manifests',
        'archive_bytes':args.archive.stat().st_size}
(args.archive.parent/(args.archive.name+'.json')).write_text(json.dumps(ledger,indent=2))
print('Archived selected banks',len(records),ledger['archive_bytes'])
