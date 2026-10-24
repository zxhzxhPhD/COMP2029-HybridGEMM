"""Immutable offline tuning plans; exact partition widths, ceiling token bucket."""
import json
from pathlib import Path


class PlanTable:
    def __init__(self,path,kind):
        self.rows={}
        for row in json.loads(Path(path).read_text())[kind]:
            self.rows.setdefault((row['out'],row['k'],row['high']),[]).append(row)
        for rows in self.rows.values():rows.sort(key=lambda r:r['tokens'])

    def get(self,outputs,k,high,tokens):
        key=(outputs,k,high)
        if key not in self.rows:raise ValueError(f'Missing tuned shape {key}')
        for row in self.rows[key]:
            if row['tokens']>=tokens:return row
        raise ValueError(f'Token count {tokens} exceeds tuned buckets for {key}')
