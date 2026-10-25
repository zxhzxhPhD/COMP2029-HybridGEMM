"""Freeze stage-two inputs without modifying any stage-one artifact."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import torch


def read(p): return json.loads(Path(p).read_text())
def write(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2))
def sha(p): return hashlib.file_digest(open(p, 'rb'), 'sha256').hexdigest()
def runtime_sha():
    folder=Path(__file__).resolve().parents[1]/'hygemm_vllm'
    return hashlib.sha256(json.dumps({p.name:sha(p) for p in sorted(folder.glob('*.py'))},sort_keys=True).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    handoff = read(args.root / '01_quality/handoff.json')
    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out / 'stage1_handoff.json', handoff)
    records = {}
    for name, spec in handoff['models'].items():
        source = args.root / '01_quality' / spec['config_file']
        assert sha(source) == spec['config_sha256']
        view = source.parent
        assert sha(view / 'selection.json') == spec['selection_sha256']
        dest = args.out / name
        for kind, origin in [('hygemm', view), ('mixllm', Path('/root/autodl-tmp/20261024_235427_formal_decode') / name / 'models/mixllm'), ('fp16', Path('/root/autodl-tmp/20261024_235427_formal_decode') / name / 'models/fp16'), ('hy_pre_i', Path('/root/autodl-tmp/20261024_235427_formal_decode') / name / 'models/hygemm')]:
            model = dest / 'models' / kind
            model.mkdir(parents=True, exist_ok=True)
            for file in origin.iterdir():
                target = model / file.name
                if target.exists(): continue
                if file.name == 'config.json': shutil.copy2(file, target)
                elif file.is_file(): target.symlink_to(file.resolve())
        source_root = view.parents[1]
        shutil.copy2(source_root / 'reference_plans.json', dest / 'plans.json')
        tokens = torch.load(source_root / 'tokens.pt', map_location='cpu', weights_only=True)['validation']
        (dest / 'calibration').mkdir(exist_ok=True)
        # Legacy profiling uses the field "test". Contents are explicitly development validation.
        torch.save({'test': tokens, 'validation': tokens}, dest / 'calibration/tokens.pt')
        records[name] = {'source_root': str(source_root), 'view': spec['view'],
                         'selection_sha256': spec['selection_sha256'],
                         'tokens_split': 'validation', 'tokens_sha256': sha(dest / 'calibration/tokens.pt'),
                         'budget': spec['budget'], 'shape_manifest': str(view / 'view.json')}
    write(args.out / 'inputs.json', records)
    write(args.out / 'protocol.json', {'baseline_commit': handoff['source_commit'],
        'graph': 'FULL', 'inductor': False, 'batches': [1,2,4,8,16,32,64,128],
        'contexts': [128,1024], 'samples': 5, 'max_groups': 3, 'cv_limit': .05,
        'regression_latency_limit': 1.02, 'model_adoption_gain': 1.01,
        'micro_reference_atol': .01, 'micro_reference_rtol': .003,
        'decode_mean_target_logprob_abs_limit': .002, 'decode_max_target_logprob_abs_limit': .05,
        'quantization': 'frozen stage1 payload/mask/clipped FP4 bank and native FP8 RTN',
        'final_test_untouched': True})


if __name__ == '__main__': main()
