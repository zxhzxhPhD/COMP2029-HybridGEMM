"""Immutable offline tuning plans; exact partition widths, ceiling token bucket."""
import json
from pathlib import Path


class PlanTable:
    def __init__(self,path,kind):
        document=json.loads(Path(path).read_text())
        self.binding=document.get('binding')
        if self.binding:
            import hashlib
            import os
            variable='HYGEMM_LIBRARY' if kind=='hygemm' else 'MIXLLM_LIBRARY'
            with open(os.environ[variable],'rb') as f:
                actual=hashlib.file_digest(f,'sha256').hexdigest()
            if actual!=self.binding['library_sha256']:
                raise ValueError('Tuning plan is bound to a different binary')
            if self.binding['layout']!='MX K32 E8M0 native physical scales v1' and kind=='hygemm':
                raise ValueError('Unknown tuning-plan quantization layout')
        self.rows={}
        for row in document[kind]:
            self.rows.setdefault((row['out'],row['k'],row['high']),[]).append(row)
        for rows in self.rows.values():
            rows.sort(key=lambda r:r['tokens'])
            if len({r['tokens'] for r in rows})!=len(rows):raise ValueError('Duplicate tuning bucket')
        self.layer_rows={}
        for row in document.get('layer_plans',[]):
            self.layer_rows.setdefault(row['layer'],[]).append(row)
        for rows in self.layer_rows.values():
            rows.sort(key=lambda r:r['tokens'])
            if len({r['tokens'] for r in rows})!=len(rows):raise ValueError('Duplicate layer tuning bucket')

    def get_layer(self,name,outputs,k,high,tokens):
        if not self.layer_rows:return self.get(outputs,k,high,tokens)
        if name not in self.layer_rows:raise ValueError(f'Missing tuned layer {name}')
        for row in self.layer_rows[name]:
            if (row['out'],row['k'],row['high'])!=(outputs,k,high):raise ValueError(f'Changed tuned shape for {name}')
            if row['tokens']>=tokens:return row
        raise ValueError(f'Token count exceeds layer buckets for {name}')

    def get(self,outputs,k,high,tokens):
        key=(outputs,k,high)
        if key not in self.rows:raise ValueError(f'Missing tuned shape {key}')
        for row in self.rows[key]:
            if row['tokens']>=tokens:return row
        raise ValueError(f'Token count {tokens} exceeds tuned buckets for {key}')
