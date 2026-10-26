"""Pin a small C4 document sample before any held-out quality scoring."""
import argparse
import gzip
import json
from pathlib import Path
import random
from huggingface_hub import HfApi,hf_hub_download
from final_common import read,write,sha,canonical_sha

ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);args=ap.parse_args()
out=args.stage/'data/c4';out.mkdir(parents=True,exist_ok=True)
specfile=out/'source.json'
if specfile.exists():spec=read(specfile)
else:
    info=HfApi().dataset_info('allenai/c4')
    spec={'dataset':'allenai/c4','revision':info.sha,'configuration':'en',
          'validation_shard':'en/c4-validation.00000-of-00008.json.gz',
          'train_shard':'en/c4-train.00000-of-01024.json.gz','read_document_prefix':4096,
          'selection_seed':3071,'validation_documents':2048,'train_documents':1024,
          'boundary':'Join selected documents with two newlines, no extra BOS/EOS; first 256 complete 2048-token windows per tokenizer.',
          'frozen_before_scoring':True}
    write(specfile,spec)
seen=set();files={}
for split in ['validation','train']:
    path=Path(hf_hub_download(spec['dataset'],repo_type='dataset',revision=spec['revision'],
        filename=spec[split+'_shard'],cache_dir=out/'download_cache'))
    documents=[]
    with gzip.open(path,'rt') as f:
        for index,line in enumerate(f):
            if index>=spec['read_document_prefix']:break
            row=json.loads(line);text=row['text'];key=canonical_sha(text)
            if key in seen:continue
            documents.append({'index':index,'url':row.get('url'),'text_sha256':key,'text':text})
    random.Random(spec['selection_seed']).shuffle(documents)
    selected=documents[:spec[split+'_documents']];assert len(selected)==spec[split+'_documents']
    seen.update(r['text_sha256'] for r in selected)
    target=out/(split+'_documents.jsonl')
    with target.open('w') as f:
        for row in selected:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    textfile=out/(split+'.txt');textfile.write_text('\n\n'.join(r['text'] for r in selected))
    files[split]={'source_file':str(path),'source_sha256':sha(path),'document_list_sha256':sha(target),
                  'text_sha256':sha(textfile),'documents':len(selected),'characters':textfile.stat().st_size,
                  'sampled_indices':[r['index'] for r in selected]}
    print('Prepared C4',split,len(selected),flush=True)
write(out/'manifest.json',{'source':spec,'files':files,'train_holdout_content_disjoint':True,'scored':False})
