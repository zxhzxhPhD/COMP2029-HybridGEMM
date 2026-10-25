"""Optimistic traffic and Amdahl references; not a hardware-attainable promise."""
import argparse
import json
from pathlib import Path
from prepare import read,write
ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);args=ap.parse_args()
stage=args.stage;bw=read(stage/'probes/bandwidth.json')['measured_bytes_per_second'];inputs=read(stage/'inputs.json');rows=[]
for model,meta in inputs.items():
    cfg=read(stage/model/'models/hygemm/config.json');layers=cfg['num_hidden_layers'];hidden=cfg['hidden_size']
    kv_heads=cfg.get('num_key_value_heads',cfg['num_attention_heads']);head=cfg.get('head_dim',hidden//cfg['num_attention_heads'])
    lm_bytes=cfg['vocab_size']*hidden*2
    for batch in [1,2,4,8,16,32,64,128]:
        for context in [128,1024]:
            average_context=context+64
            kv_read=2*layers*kv_heads*head*2*batch*average_context
            kv_write=2*layers*kv_heads*head*2*batch
            projections=meta['budget']['persistent_quantized_bytes']
            bytes_=projections+lm_bytes+kv_read+kv_write
            rows.append({'model':model,'batch':batch,'context':context,'assumed_mean_attended_context':average_context,
                'projection_weights_scales_ids_bytes':projections,'fp16_lm_head_bytes':lm_bytes,
                'kv_read_bytes':kv_read,'kv_write_bytes':kv_write,'optimistic_bytes':bytes_,
                'optimistic_stream_ms':bytes_/bw*1000})
amdahl=[]
for batch in [1,8,64]:
    data=read(stage/'profile_initial'/f'analysis_hygemm_b{batch}.json');summary=data.get('summary',data)
    control=next(r for r in read(stage/'profile_initial/control_hygemm.json')['cells'] if r['batch']==batch)
    gemm=summary['graph_budget_ms_per_step']['gemm'];quant=summary['graph_budget_ms_per_step']['activation_quant']
    total=control['median_ms_per_step']
    for speed in [1.1,1.25,2.0]:
        amdahl.append({'batch':batch,'gemm_speed_factor':speed,'profile_gemm_ms':gemm,
            'control_total_ms':total,'predicted_total_speedup':total/(total-gemm+gemm/speed)})
write(stage/'probes/traffic_bounds.json',{'measured_copy_bandwidth_bytes_s':bw,'traffic':rows,'amdahl':amdahl,
    'assumptions':['each projection payload/scale/index and FP16 lm-head streamed once per step',
        'one K/V read at mean context prompt+64 and one new K/V write, FP16 cache',
        'ignores activation/residual/output intermediates, repeated reads, kernels and graph/host gaps',
        'memcpy read+write bandwidth is a reference, not equivalent to GEMM gather or attention access',
        'Amdahl mixes diagnostic fractions with unprofiled total; an explanatory estimate only']})
