"""Fetch only pinned WT2 validation; usable on the local CPU/pyarrow host."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
revision='b08601e04326c79dfdd32d625aee71d232d685c3'
file=Path(hf_hub_download('Salesforce/wikitext',repo_type='dataset',revision=revision,
    filename='wikitext-2-raw-v1/validation-00000-of-00001.parquet',cache_dir=args.out/'cache'))
texts=pq.read_table(file)['text'].to_pylist()
assert len(texts)==3760
target=args.out/'wikitext2_validation.txt'
target.write_text('\n\n'.join(texts),encoding='utf-8',newline='\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(args.out/'validation_source.json').write_text(json.dumps({'dataset':'Salesforce/wikitext','revision':revision,
 'configuration':'wikitext-2-raw-v1','split':'validation','rows':len(texts),'parquet_sha256':sha(file),
 'text_sha256':sha(target),'join':'two newline characters'},indent=2))
print('Validation ready',len(texts),sha(target))
