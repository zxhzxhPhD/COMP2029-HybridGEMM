"""Check actual small-batch decode values, separately from speed/PPL runs."""
import argparse
import json
import os
from pathlib import Path
import torch

os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
os.environ.setdefault('VLLM_PLUGINS', 'hygemm_quantization')
os.environ.setdefault('VLLM_NO_USAGE_STATS', '1')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    from vllm import LLM, SamplingParams
    torch.set_num_threads(4)
    batches = [1, 2, 4, 8, 16, 32]
    llm = LLM(model=str(args.model_root / 'models/hygemm'), dtype='float16', tensor_parallel_size=1,
              max_model_len=2049, max_num_seqs=128, max_num_batched_tokens=2048,
              enable_prefix_caching=False, gpu_memory_utilization=.93,
              kv_cache_memory_bytes=10 * 1024**3, seed=2029, enforce_eager=False,
              attention_config={'backend': 'FLASH_ATTN'},
              worker_extension_cls='hygemm_vllm.worker.AuditWorkerExtension',
              compilation_config={'mode': 0, 'cudagraph_mode': 'FULL',
                                  'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 32, 64, 128], 'max_cudagraph_capture_size': 128})
    test = torch.load(args.model_root / 'calibration/tokens.pt', map_location='cpu', weights_only=True)['test']
    generator = torch.Generator().manual_seed(2029 + 128)
    offsets = torch.randint(0, test.numel() - 128, (128,), generator=generator)
    params = SamplingParams(temperature=0, max_tokens=17, min_tokens=17, ignore_eos=True, detokenize=False, logprobs=0)
    records = []
    for batch in batches:
        prompts = [{'prompt_token_ids': test[i:i + 128].tolist()} for i in offsets[:batch]]
        llm.collective_rpc('formal_begin', args=(batch,))
        outputs = llm.generate(prompts, params, use_tqdm=False)
        timing = llm.collective_rpc('formal_end')[0]
        assert timing['steps'] == timing['graph_steps'] >= 8
        values = []
        for result in outputs:
            sample = result.outputs[0]
            assert len(sample.token_ids) == 17 and len(sample.logprobs) == 17
            values.append({'tokens': list(sample.token_ids),
                           'logprobs': [row[token].logprob for token, row in zip(sample.token_ids, sample.logprobs)]})
        records.append({'batch': batch, 'outputs': values, 'measured_graph_steps': timing['steps']})
        print('Quality decode', batch, flush=True)
    args.out.write_text(json.dumps({'records': records, 'audit': llm.collective_rpc('hygemm_model_audit')[0]}, indent=2))


if __name__ == '__main__':
    main()
