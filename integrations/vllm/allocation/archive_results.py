"""Archive compact artifacts, leaving large FP4 banks at their audited origin."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--archive',type=Path,required=True)
ap.add_argument('--previous-ledger',type=Path)
args=ap.parse_args();records=[];excluded=[];included=[]
previous={r['path']:r['sha256'] for r in json.loads(args.previous_ledger.read_text())['files']} if args.previous_ledger else {}
with tarfile.open(args.archive,'w:xz' if args.archive.suffix=='.xz' else 'w:gz') as tar:
    for path in sorted(args.root.rglob('*')):
        if path.is_symlink() or not path.is_file() or path.suffix=='.tmp':continue
        relative=path.relative_to(args.root)
        if 'banks' in relative.parts and path.suffix=='.pt':excluded.append(str(relative));continue
        if path.resolve()==args.archive.resolve():continue
        data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
        records.append({'path':str(relative),'bytes':len(data),'sha256':digest})
        if previous.get(str(relative))==digest:continue
        info=tar.gettarinfo(str(path),arcname=str(relative));info.size=len(data)
        tar.addfile(info,io.BytesIO(data));included.append(str(relative))
(args.archive.parent/(args.archive.name+'.json')).write_text(json.dumps({'files':records,'remote_only_bank_files':excluded,
    'included_files':included,'base_ledger':str(args.previous_ledger) if args.previous_ledger else None,
    'bank_hash_source':'each banks/<mode>/manifest.json','symlinks':'not followed; original models/tokens remain at audited Q1/formal sources'},indent=2))
print('Archived',len(included),'/',len(records),'files;',len(excluded),'large bank files remain remote;',args.archive.stat().st_size,'bytes')
