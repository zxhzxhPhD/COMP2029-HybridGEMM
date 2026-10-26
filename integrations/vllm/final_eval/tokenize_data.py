"""Tokenize all fixed evaluation inputs with the inference environment tokenizer."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from final_common import *


def pair(tokenizer,context,continuation):
    spaces=len(context)-len(context.rstrip())
    if spaces:continuation=context[-spaces:]+continuation;context=context[:-spaces]
    if context:
        whole=tokenizer.encode(context+continuation,add_special_tokens=False)
        prefix=tokenizer.encode(context,add_special_tokens=False);suffix=whole[len(prefix):]
    else:
        prefix=[tokenizer.eos_token_id];suffix=tokenizer.encode(continuation,add_special_tokens=False)
    assert prefix and suffix and len(suffix)<2048
    joined=prefix+suffix;cut=max(0,len(joined)-2048)
    return joined[cut:],len(prefix)-cut,cut


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--models',default=','.join(MODELS));args=ap.parse_args()
    stage=args.stage;tasks=read(stage/'data/tasks/manifest.json')
    for model in args.models.split(','):
        source=Path('/root/autodl-tmp/models/Mistral-7B-v0.3') if model=='Mistral-7B' else FORMAL/model/'models/fp16'
        out=stage/'data'/model;out.mkdir(parents=True,exist_ok=True)
        tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True)
        paths={'train':FORMAL/'data/wikitext2_train.txt','test':FORMAL/'data/wikitext2_test.txt',
               'validation':Q1/'data/wikitext2_validation.txt','c4':stage/'data/c4/validation.txt','c4_train':stage/'data/c4/train.txt'}
        data={name:torch.tensor(tokenizer(path.read_text(),add_special_tokens=False)['input_ids'],dtype=torch.int64) for name,path in paths.items()}
        assert len(data['c4'])>=256*2048
        data['speed']=data['c4']
        torch.save(data,out/'tokens.pt')
        records={'tokenizer':str(source),'add_special_tokens':False,'tokens_file_sha256':sha(out/'tokens.pt'),
            'corpora':{name:{'text_sha256':sha(paths[name]),'tokens':len(value),
                'token_sha256':hashlib.sha256(value.numpy().tobytes()).hexdigest()} for name,value in data.items() if name!='speed'},'tasks':{}}
        for name in tasks['tasks']:
            for suffix in ['', '_development']:
                input_path=stage/'data/tasks'/(name+suffix+'.jsonl');target=out/(name+suffix+'.jsonl.gz');count=0;clipped=0;maxlen=0
                with input_path.open() as f,gzip.open(target,'wt') as g:
                    for line in f:
                        row=json.loads(line);tokens,begin,cut=pair(tokenizer,row['context'],row['continuation'])
                        assert begin>=1
                        g.write(json.dumps({'request_id':row['request_id'],'doc_id':row['doc_id'],'choice':row['choice'],
                            'tokens':tokens,'target_begin':begin,'discarded_context_tokens':cut})+'\n')
                        count+=1;clipped+=bool(cut);maxlen=max(maxlen,len(tokens))
                records['tasks'][name+suffix]={'requests':count,'clipped_contexts':clipped,'max_length':maxlen,
                    'input_sha256':sha(input_path),'token_file_sha256':sha(target)}
        write(out/'manifest.json',records);print('Tokenized',model,flush=True)


if __name__=='__main__':main()
