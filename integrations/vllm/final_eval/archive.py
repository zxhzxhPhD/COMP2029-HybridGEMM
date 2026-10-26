"""Export reviewable results without following model/bank checkpoint symlinks."""
import argparse
from pathlib import Path
import shutil
import tarfile
from final_common import read,write,sha

ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
args=ap.parse_args();stage=args.stage.resolve();records={};excluded=[]
assert not args.out.resolve().is_relative_to(stage),'Archive destination must be outside the exported tree'
analysis_source=stage/'final_analysis_source';analysis_source.mkdir(exist_ok=True)
for name in ['report.py','accept.py','archive.py']:
    shutil.copy2(Path(__file__).resolve().parent/name,analysis_source/name)
for file in sorted(stage.rglob('*')):
    if not file.is_file() or file.is_symlink():continue
    rel=file.relative_to(stage)
    blocked=any(x in rel.parts for x in ['format_build','dataset_cache','download_cache','__pycache__','.cache','.locks'])
    blocked=blocked or file.suffix in ['.safetensors','.so','.o','.a','.bin','.lock'] or file.name in ['gpu.lock']
    blocked=blocked or file.suffix=='.pt' and ('bank' in rel.parts or 'banks' in rel.parts)
    blocked=blocked or file.name in ['archive_manifest.json','archive_delivery.json']
    if blocked:excluded.append(str(rel));continue
    records[str(rel)]={'bytes':file.stat().st_size,'sha256':sha(file)}
write(stage/'archive_manifest.json',{'files':records,'bytes':sum(v['bytes'] for v in records.values()),
    'excluded_files':excluded,'symlinks_followed':False,
    'scope':'Weights, quantized bank tensors, executable binaries and caches stay on the remote host. Their full identity is preserved in freeze/v1/file_ledger.json. Calibration scores/moments/tokens and decode distribution samples are included.'})
args.out.parent.mkdir(parents=True,exist_ok=True)
with tarfile.open(args.out,'w:gz',compresslevel=3) as archive:
    for name in [*records,'archive_manifest.json']:archive.add(stage/name,arcname=name,recursive=False)
write(stage/'archive_delivery.json',{'archive':str(args.out),'sha256':sha(args.out),'bytes':args.out.stat().st_size,'files':len(records)+1})
print(read(stage/'archive_delivery.json'),flush=True)
