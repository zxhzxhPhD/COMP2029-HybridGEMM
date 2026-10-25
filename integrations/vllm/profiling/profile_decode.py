"""Reuse the frozen formal model/plans and capture a bounded pure-decode ROI."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import torch

os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
os.environ.setdefault('VLLM_PLUGINS', 'hygemm_quantization')
os.environ.setdefault('VLLM_NO_USAGE_STATS', '1')
CAPTURE_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--formal', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--kind', choices=['mixllm', 'hygemm'], required=True)
    ap.add_argument('--batches', default='1,8,64')
    ap.add_argument('--context', type=int, default=128)
    ap.add_argument('--profile', action='store_true')
    ap.add_argument('--meter', action='store_true', help='Unprofiled Graph boundary event control')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    batches = list(map(int, args.batches.split(',')))
    if args.profile:
        assert len(batches) == 1
    assert not (args.profile and args.meter)
    from vllm import LLM, SamplingParams
    torch.set_num_threads(4)
    llm = LLM(model=str(args.formal / 'models' / args.kind), dtype='float16', tensor_parallel_size=1,
              max_model_len=2049, max_num_seqs=128, max_num_batched_tokens=2048,
              enable_prefix_caching=False, gpu_memory_utilization=.93,
              kv_cache_memory_bytes=10 * 1024**3, seed=2029, enforce_eager=False,
              attention_config={'backend': 'FLASH_ATTN'},
              worker_extension_cls='hygemm_vllm.profile_worker.ProfileWorkerExtension',
              compilation_config={'mode': 0, 'cudagraph_mode': 'FULL',
                                  'cudagraph_capture_sizes': CAPTURE_SIZES, 'max_cudagraph_capture_size': 128})
    tokens = torch.load(args.formal / 'calibration/tokens.pt', map_location='cpu', weights_only=True)['test']
    generator = torch.Generator().manual_seed(2029 + args.context)
    offsets = torch.randint(0, tokens.numel() - args.context, (128,), generator=generator)
    params = SamplingParams(temperature=0, max_tokens=129, min_tokens=129, ignore_eos=True, detokenize=False)
    report = {'kind': args.kind, 'profiled': args.profile, 'metered': args.meter, 'context': args.context,
              'formal': str(args.formal), 'plans_sha256': hashlib.sha256((args.formal / 'plans.json').read_bytes()).hexdigest(),
              'configuration': {'inductor': False, 'graph': 'FULL', 'capture_sizes': CAPTURE_SIZES,
                                'max_num_seqs': 128, 'max_num_batched_tokens': 2048, 'kv_cache_bytes': 10 * 1024**3},
              'cells': []}
    report['model_config_sha256']=hashlib.sha256((args.formal/'models'/args.kind/'config.json').read_bytes()).hexdigest()
    report['library_sha256']=hashlib.sha256(Path(os.environ['HYGEMM_LIBRARY' if args.kind=='hygemm' else 'MIXLLM_LIBRARY']).read_bytes()).hexdigest()
    report['runtime_options']={name:os.environ.get(name,default) for name,default in
        [('HYGEMM_PACK_MODE','0'),('HYGEMM_FUSE_SILU','0'),('HYGEMM_SILU_MATH','0'),('HYGEMM_ROW_ORDER','none'),('HYGEMM_FUSED_GATEUP','0')]}
    for batch in batches:
        prompts = [{'prompt_token_ids': tokens[i:i + args.context].tolist()} for i in offsets[:batch]]

        def generate(profile=False):
            mode = 'profile' if profile else 'meter' if args.meter else 'formal'
            llm.collective_rpc(mode + '_begin', args=(batch,))
            start = time.perf_counter()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            wall = time.perf_counter() - start
            result = llm.collective_rpc(mode + '_end')[0]
            assert all(len(r.outputs[0].token_ids) == 129 for r in outputs)
            result['wall_s'] = wall
            result['output_sha256'] = hashlib.sha256(json.dumps([r.outputs[0].token_ids for r in outputs]).encode()).hexdigest()
            return result

        generate()
        if args.profile:
            cell = {'batch': batch, 'sample': generate(True)}
        else:
            attempts = []
            for attempt in range(3):
                samples = [generate() for _ in range(5)]
                times = [s['decode_ms_per_step'] for s in samples]
                cv = statistics.pstdev(times) / statistics.mean(times)
                roi_cv = (statistics.pstdev(s['roi_period_ms'] for s in samples) / statistics.mean(s['roi_period_ms'] for s in samples)) if args.meter else cv
                attempts.append({'samples': samples, 'cv': cv, 'roi_cv': roi_cv, 'median_ms_per_step': statistics.median(times)})
                if max(cv, roi_cv) <= .05:
                    break
                generate()
            cell = {'batch': batch, 'attempts': attempts, 'stable': max(cv, roi_cv) <= .05,
                    'median_ms_per_step': statistics.median(times), 'cv': cv}
        report['cells'].append(cell)
        print('Finished', args.kind, batch, 'profile' if args.profile else cell['median_ms_per_step'], flush=True)
    report['audit'] = llm.collective_rpc('hygemm_model_audit')[0]
    assert len(report['audit']) == 112
    name = f'profile_{args.kind}_b{batches[0]}.json' if args.profile else f'{"meter" if args.meter else "control"}_{args.kind}.json'
    (args.out / name).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
