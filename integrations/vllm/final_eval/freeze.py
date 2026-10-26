"""Immutable stage-three input ledger and guards for every final model process."""
import argparse
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import time
from final_common import *

REPO=Path(__file__).resolve().parents[3]
MIX_SHA='1d6190c2bb2effb24d1c12a5438ce051dc0967ec8cdf71e3120e207a38d426da'


def file_record(path):
    p=Path(path).resolve();s=p.stat()
    return {'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':sha(p)}


def source_files():
    return sorted(p for base in ['integrations/vllm','src','scripts'] for p in (REPO/base).rglob('*')
        if p.is_file() and (p.suffix in ['.py','.cu','.cuh','.h','.cpp','.sh','.cmake'] or p.name=='CMakeLists.txt') and '__pycache__' not in p.parts
        and not (p.parent.name=='final_eval' and p.name in ['report.py','accept.py','archive.py']))


def selected_methods(stage):
    return {m:{k:v for k,v in rows.items() if not v.get('development_only') and not k.startswith('tune-')}
            for m,rows in read(stage/'methods.json').items()}


def create(stage):
    destination=stage/'freeze/v1';assert not (destination/'manifest.json').exists(),'Never overwrite a freeze'
    assert not list((stage/'results').rglob('result.json')),'Final scoring has already begun'
    gates=read(stage/'preflight.json');assert gates['passed'],gates
    methods=selected_methods(stage);files={};references={}
    def add(path):
        p=Path(path).resolve();key=str(p)
        if key not in files:files[key]=file_record(p)
        return key
    for p in source_files():add(p)
    for p in [stage/'protocol.json',stage/'final_jobs.json',stage/'preflight.json',stage/'additional_baselines.json',
              stage/'baseline_compatibility.json',stage/'old_control_provenance.json',stage/'tools_environment.json',stage/'evaluation_tools_freeze.txt']:
        add(p)
    for p in (stage/'data').rglob('*'):
        if p.is_file() and not any(x in p.parts for x in ['.cache','.locks','__pycache__']) and not p.name.endswith('.lock'):add(p)
    for p in (stage/'environment').rglob('*'):
        if p.is_file():add(p)
    for name in ['torch','vllm','transformers','tokenizers','safetensors']:
        dist=importlib.metadata.distribution(name)
        for entry in dist.files or []:
            if str(entry).endswith(('.dist-info/RECORD','.dist-info/METADATA')):add(dist.locate_file(entry))
    for model,rows in methods.items():
        references[model]={}
        for name,spec in rows.items():
            refs=[];folder=Path(spec['model_path']);cfg=read(folder/'config.json')
            for p in folder.iterdir():
                if p.is_file():refs.append(add(p))
            for value in spec['environment'].values():
                if isinstance(value,str) and value.endswith('.so'):refs.append(add(value))
            for key,value in cfg.get('quantization_config',{}).items():
                if isinstance(value,str) and Path(value).is_file():
                    refs.append(add(value))
                    if key=='q4_bank':
                        for layer in read(value)['layers'].values():
                            bank=Path(value).parent/layer['file'];ref=add(bank)
                            assert files[ref]['sha256']==layer['sha256'],bank
                            refs.append(ref)
            if spec.get('shape_manifest'):refs.append(add(spec['shape_manifest']))
            references[model][name]=sorted(set(refs))
    for p in (stage/'format_build').glob('hygemm_sm120_format_probe*'):
        if p.is_file():add(p)
    for p in [stage/'format_build/CMakeCache.txt',REPO/'CMakeLists.txt']:
        if p.exists():add(p)
    assert sha(MIX)==MIX_SHA
    versions={name:importlib.metadata.version(name) for name in ['torch','vllm','transformers','numpy','safetensors','tokenizers']}
    snapshot=destination/'source';snapshot.mkdir(parents=True,exist_ok=True)
    import shutil
    for p in source_files():
        target=snapshot/p.relative_to(REPO);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    write(destination/'methods.json',methods)
    for name in ['protocol.json','final_jobs.json','preflight.json','baseline_compatibility.json']:
        shutil.copy2(stage/name,destination/name)
    write(destination/'file_ledger.json',files)
    manifest={'version':1,'frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'parent_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'working_tree_status':subprocess.check_output(['git','status','--short'],cwd=REPO,text=True),
        'source_tree_canonical_sha256':canonical_sha({str(p.relative_to(REPO)):sha(p) for p in source_files()}),
        'source_snapshot':str(snapshot),'methods_sha256':sha(destination/'methods.json'),
        'file_ledger_sha256':sha(destination/'file_ledger.json'),'method_files':references,'versions':versions,
        'python':sys.version,'mix_device_sha256':MIX_SHA,'preflight_sha256':sha(stage/'preflight.json'),
        'protocol_sha256':sha(stage/'protocol.json'),'jobs_sha256':sha(stage/'final_jobs.json'),
        'large_file_verification':'All files fully hashed at freeze; each process checks all source/config/data hashes and large-file stat identity. Full hashes rechecked at final acceptance.',
        'quality_holdout':'No final model scoring before this manifest; historical WT2 test was seen in earlier stages, C4/tasks were not used for selection.'}
    write(destination/'manifest.json',manifest)
    write(stage/'freeze/current.json',{'manifest':str(destination/'manifest.json'),'sha256':sha(destination/'manifest.json')})
    print('FROZEN',manifest['frozen_utc'],len(files),'files',sum(map(len,methods.values())),'methods',flush=True)


def verify_frozen(stage,model=None,method=None,full=False):
    stage=Path(stage);pointer=read(stage/'freeze/current.json');path=Path(pointer['manifest'])
    assert sha(path)==pointer['sha256'];manifest=read(path);folder=path.parent
    assert sha(folder/'file_ledger.json')==manifest['file_ledger_sha256']
    assert sha(folder/'methods.json')==manifest['methods_sha256']
    methods=read(folder/'methods.json');ledger=read(folder/'file_ledger.json')
    if model is not None:
        assert read(stage/'methods.json')[model][method]==methods[model][method],'Changed method after freeze'
    for name,version in manifest['versions'].items():assert importlib.metadata.version(name)==version,(name,version)
    assert sha(stage/'protocol.json')==manifest['protocol_sha256']
    assert sha(stage/'final_jobs.json')==manifest['jobs_sha256']
    selected=set(manifest['method_files'][model][method]) if model is not None else set(ledger)
    selected.update(str(p.resolve()) for p in source_files())
    selected.update(k for k in ledger if k.startswith(str(stage/'data')) and (model is None or '/'+model+'/' in k or Path(k).suffix=='.json'))
    assert canonical_sha({str(p.relative_to(REPO)):sha(p) for p in source_files()})==manifest['source_tree_canonical_sha256'],'Changed source tree after freeze'
    for key in sorted(selected):
        expected=ledger[key];p=Path(key);s=p.stat()
        assert (s.st_size,s.st_mtime_ns)==(expected['bytes'],expected['mtime_ns']),('Changed frozen input',key)
        if full or s.st_size<32*1024**2 or p.suffix in ['.py','.cu','.cuh','.h','.cpp','.sh','.json','.so']:
            assert sha(p)==expected['sha256'],key
    assert sha(MIX)==MIX_SHA
    return manifest


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True)
    ap.add_argument('--verify',action='store_true');ap.add_argument('--full',action='store_true');args=ap.parse_args()
    if args.verify:verify_frozen(args.stage,full=args.full);print('Freeze verified',flush=True)
    else:create(args.stage)
