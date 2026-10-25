"""Exact deterministic port control and disclosed historical backward variation."""
import argparse
from pathlib import Path
from alloc_utils import *

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
q1=Path(read(args.out/'protocol.json')['q1'])
paths={'legacy':q1/'0.5B/scores/2029/scores.pt',
    'current':args.out/'0.5B/scores/rtn_32/scores.pt',
    'deterministic_original':args.out/'controls/original_deterministic/scores/2029/scores.pt',
    'original_repeat':args.out/'controls/original_repeat/scores/2029/scores.pt',
    'initial_port':args.out/'controls/initial_port/scores.pt',
    'port_repeat':args.out/'controls/ported_repeat/scores/scores.pt'}
data={k:torch.load(p,map_location='cpu',weights_only=True)['scores'] for k,p in paths.items() if p.exists()}
def compare(a,b):
    result={}
    for family in FAMILIES:
        xs=data[a][family];ys=data[b][family]
        delta=torch.cat([xs[n]-ys[n] for n in xs]);reference=torch.cat([ys[n] for n in xs])
        result[family]={'exact_projection_vectors':sum(torch.equal(xs[n],ys[n]) for n in xs),'total_projections':len(xs),
            'max_abs_error':float(delta.abs().max()),'global_relative_l2':float(delta.norm()/reference.norm()),
            'max_projection_relative_l2':max(float((xs[n]-ys[n]).norm()/ys[n].norm().clamp_min(1e-30)) for n in xs)}
    return result
exact=compare('current','deterministic_original')
assert all(v['exact_projection_vectors']==v['total_projections'] for v in exact.values())
for family in ['weight_gain','output_gain']:
    assert all(torch.equal(v,data['legacy'][family][n]) for n,v in data['current'][family].items())
historical={key:compare(key,'legacy') for key in ['current','original_repeat','initial_port','port_repeat'] if key in data}
write(args.out/'score_control.json',{'passed':True,'deterministic_comparison':exact,'historical_comparisons':historical,
    'source_sha256':{str(p):sha(p) for p in paths.values() if p.exists()},
    'original_scorer_sha256':sha(Path(__file__).resolve().parents[1]/'quality/score.py'),
    'resolution':'Original Q1 scorer and port are exactly equal with deterministic teacher backward. Q1/Q2 historical vectors remain unchanged. Q3 also carries same-backward RTN scores.',
    'initial_gate':'Historical nondeterministic gradients failed rtol1e-4/atol1e-9; preserved under controls, not hidden or relabeled as a pass.'})
print('All four deterministic score families match exactly')
