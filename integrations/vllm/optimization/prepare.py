"""Create independent model views while reusing frozen weights/calibration."""
import argparse
import json
from pathlib import Path
import shutil

ap = argparse.ArgumentParser()
ap.add_argument('--formal', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
for label in ['baseline', 'optimized', 'mixllm', 'fp16']:
    folder = args.out / label
    folder.mkdir(exist_ok=True)
    if not (folder / 'calibration').exists():
        (folder / 'calibration').symlink_to(args.formal / 'calibration', target_is_directory=True)
    for name in ['cases.json', 'plans.json']:
        if not (folder / name).exists():
            shutil.copy2(args.formal / name, folder / name)
    kind = label if label in ['mixllm', 'fp16'] else 'hygemm'
    model = folder / 'models' / kind
    model.mkdir(parents=True, exist_ok=True)
    source = args.formal / 'models' / kind
    for file in source.iterdir():
        if file.name != 'config.json' and file.is_file() and not (model / file.name).exists():
            (model / file.name).symlink_to(file.resolve())
    cfg = json.loads((source / 'config.json').read_text())
    if kind != 'fp16':
        cfg['quantization_config']['tuning_plan'] = str(folder / 'plans.json')
    (model / 'config.json').write_text(json.dumps(cfg))
print('Prepared four model views with frozen selection and weights')
