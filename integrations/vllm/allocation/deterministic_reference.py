"""Run the unmodified Q1 scorer with deterministic teacher backward enabled."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import runpy
import sys
import torch

torch.use_deterministic_algorithms(True)
source=Path(__file__).resolve().parents[1]/'quality/score.py'
sys.path.insert(0,str(source.parent))
sys.argv[0]=str(source)
runpy.run_path(str(source),run_name='__main__')
