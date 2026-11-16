"""CPU-only benchmark: genuine Hugging Face GPT-2 weights + Ray remote actors.

This is a NEW reconstruction of the original NIR's architecture, not a copy
of its unavailable source. Full-context greedy decoding, no KV cache.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import statistics
import time

from research.partition import optimal_partition, brute_force_partition, synthetic_case
from research.workloads import build_workloads, write_workloads

MODES=('B0','D0','D1','D2','D3')


def boundaries_for_layers(n_layers:int, k:int=3):
    if n_layers<=0 or k<=0: raise ValueError('n_layers and k must be positive')
    per=[n_layers//k+int(i<n_layers%k) for i in range(k)]
    bounds=[0]
    for c in per: bounds.append(bounds[-1]+c)
    return bounds


def make_shard_class(ray):
    @ray.remote(num_cpus=1)
    class GPT2ShardActor:
        def __init__(self, name, shard_id, begin, end, first=False, final=False):
            import gc
            import torch
            from transformers import AutoModelForCausalLM
            torch.set_num_threads(1)
            full=AutoModelForCausalLM.from_pretrained(name).eval().cpu()
            tr=full.transformer
            self.shard_id,self.begin,self.end=shard_id,begin,end
            self.first,self.final=first,final
            self.wte=tr.wte if first else None
            self.wpe=tr.wpe if first else None
            self.drop=tr.drop if first else None
            self.blocks=list(tr.h[begin:end])
            self.ln_f=tr.ln_f if final else None
            self.head=full.lm_head if final else None
            del full, tr
            gc.collect()

        def metadata(self):
            import os,ray
            return {'shard_id':self.shard_id,'layer_range':[self.begin,self.end],
                    'node_id':ray.get_runtime_context().get_node_id(), 'pid':os.getpid()}

        def forward(self, value, mode):
            import numpy as np
            import torch
            with torch.inference_mode():
                if self.first:
                    ids=torch.from_numpy(np.asarray(value).copy()).long()
                    b,seq=ids.shape
                    position_ids=torch.arange(seq, dtype=torch.long).unsqueeze(0).expand(b,-1)
                    hidden=self.drop(self.wte(ids)+self.wpe(position_ids))
                else:
                    hidden=torch.from_numpy(np.asarray(value).copy())
                for block in self.blocks:
                    hidden=block(hidden,use_cache=False)[0]
                if not self.final:
                    return hidden.detach().cpu().numpy()
                if mode not in ('D0','D1','D2','D3'):
                    raise ValueError('Unknown mode '+str(mode))
                if mode in ('D2','D3'):
                    hidden=hidden[:,-1:,:]
                logits=self.head(self.ln_f(hidden))
                if mode == 'D0':
                    return logits.detach().cpu().numpy()
                last=logits[:,-1,:]
                if mode == 'D3':
                    return last.argmax(dim=-1).to(torch.int64).cpu().numpy()
                return last.detach().cpu().numpy()
    return GPT2ShardActor


def create_pipeline(ray,model_name,n_layers,k=3,resources=None,bounds=None):
    Actor=make_shard_class(ray)
    bounds=list(bounds) if bounds is not None else boundaries_for_layers(n_layers,k)
    if len(bounds)!=k+1 or bounds[0]!=0 or bounds[-1]!=n_layers or any(b<a for a,b in zip(bounds,bounds[1:])):
        raise ValueError(f'Invalid layer boundaries: {bounds}')
    actors=[]
    for i in range(k):
        options={}
        if resources:
            options['resources']={resources[i]:0.01}
        actors.append(Actor.options(**options).remote(
            model_name,i,bounds[i],bounds[i+1],i==0,i==k-1))
    metadata=ray.get([a.metadata.remote() for a in actors])
    if len({(m['pid'],m['node_id']) for m in metadata})!=k:
        raise RuntimeError('Actors are not in independent worker processes')
    return actors,metadata


def distributed_greedy(ray,actors,input_ids,steps,mode):
    import numpy as np
    ids=list(input_ids)
    payload=0
    for _ in range(steps):
        work=np.asarray([ids],dtype=np.int64)
        for actor in actors:
            work=ray.get(actor.forward.remote(work,mode))
        payload+=int(np.asarray(work).nbytes)
        if mode=='D3': token=int(np.asarray(work).reshape(-1)[0])
        elif mode=='D0': token=int(np.asarray(work)[0,-1,:].argmax())
        else: token=int(np.asarray(work)[0,:].argmax())
        ids.append(token)
    return ids[len(input_ids):],payload


def monolithic_greedy(model,input_ids,steps):
    import torch
    ids=list(input_ids)
    with torch.inference_mode():
        for _ in range(steps):
            output=model(input_ids=torch.tensor([ids]),use_cache=False)
            ids.append(int(output.logits[0,-1,:].argmax()))
    return ids[len(input_ids):]


def predicted_payload(mode,initial_length,steps,vocab_size,batch=1,element_bytes=4,token_bytes=8):
    if mode=='B0': return 0
    if mode=='D0': return batch*vocab_size*element_bytes*(steps*initial_length+steps*(steps-1)//2)
    if mode in ('D1','D2'): return batch*vocab_size*element_bytes*steps
    if mode=='D3': return batch*token_bytes*steps
    raise ValueError(mode)


def benchmark(args):
    import numpy as np
    import torch
    import ray
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(args.threads)
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForCausalLM.from_pretrained(args.model).eval().cpu()
    n_layers=len(model.transformer.h)
    vocab_size=model.config.vocab_size
    cases=build_workloads(tokenizer,max_position_embeddings=model.config.n_positions)
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    write_workloads(output/'workloads.jsonl',cases,args.model)
    selected=cases[:args.limit] if args.limit else cases
    stage_modes=tuple(x.strip().upper() for x in args.modes.split(','))
    if any(m not in MODES for m in stage_modes): raise ValueError('Unknown mode selected')
    if 'B0' not in stage_modes: stage_modes=('B0',)+stage_modes
    ray.init(address=args.ray_address if args.ray_address else None,
             include_dashboard=False, ignore_reinit_error=True,
             **({} if args.ray_address else {'num_cpus':max(4,args.shards+1)}))
    bounds=None
    partition_metadata={'method':'equal-layer-count'}
    if args.bounds:
        bounds=[int(x) for x in args.bounds.split(',')]
        partition_metadata={'method':'user-provided','boundaries':bounds}
    if args.profile_json:
        if bounds is not None: raise ValueError('Use either --bounds or --profile-json')
        from research.partition import optimal_partition
        profile=json.loads(Path(args.profile_json).read_text(encoding='utf-8'))
        plan=optimal_partition(profile['compute_ms'], profile['memory_mib'],profile['limits_mib'],
            profile.get('link_ms'),profile.get('fixed_mem_mib'))
        if plan is None: raise RuntimeError('No feasible partition for cost profile')
        bounds=list(plan.boundaries)
        partition_metadata={'method':'cost-model-dynamic-programming','boundaries':bounds,
            'predicted_ms':plan.predicted_ms,'profile_provenance':profile.get('provenance','unspecified')}
    actors,metadata=create_pipeline(ray,args.model,n_layers,args.shards,bounds=bounds)
    runtime={'host':platform.platform(),'python':platform.python_version(),
             'torch':torch.__version__,'ray':ray.__version__,
             'model':args.model,'transformer_layers':n_layers,
             'seed':args.seed,'threads':args.threads,
             'logical_ray_nodes':len(ray.nodes()),'actors':metadata,
             'cases_evaluated':len(selected),'total_cases_generated':len(cases),
             'warmup':args.warmup,'repetitions':args.repeats,
             'modes':stage_modes,'partition':partition_metadata,
             'warning':'Times from THIS host only; local Ray nodes are not physical servers; payload bytes exclude transport/RPC overhead.'}
    (output/'environment.json').write_text(json.dumps(runtime,ensure_ascii=False,indent=2),encoding='utf-8')
    fields=('case_id','category','mode','repeat','prompt_tokens','new_tokens',
            'elapsed_ms','tokens_per_second','payload_bytes','predicted_payload_bytes','tokens_match','model')
    mismatches=[]
    csv_path=output/'measurements.csv'
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for ix,case in enumerate(selected,1):
            ids=case['input_ids']; steps=case['max_new_tokens']
            for _ in range(args.warmup):
                monolithic_greedy(model,ids,min(2,steps))
                distributed_greedy(ray,actors,ids,min(2,steps),'D3')
            expected=monolithic_greedy(model,ids,steps)
            for mode in stage_modes:
                for repeat in range(args.repeats):
                    t0=time.perf_counter()
                    if mode=='B0': result=monolithic_greedy(model,ids,steps); payload=0
                    else: result,payload=distributed_greedy(ray,actors,ids,steps,mode)
                    elapsed=time.perf_counter()-t0
                    match=(result==expected)
                    pred=predicted_payload(mode,len(ids),steps,vocab_size)
                    if payload!=pred: raise AssertionError(f'Unexpected payload {mode}: {payload} != {pred}')
                    writer.writerow({'case_id':case['case_id'],'category':case['category'],'mode':mode,
                        'repeat':repeat,'prompt_tokens':len(ids),'new_tokens':steps,
                        'elapsed_ms':round(elapsed*1000,4),
                        'tokens_per_second':round(steps/elapsed,4),'payload_bytes':payload,
                        'predicted_payload_bytes':pred,'tokens_match':match,'model':args.model})
                    if not match: mismatches.append({'case_id':case['case_id'],'mode':mode,'repeat':repeat,
                        'expected':expected,'observed':result})
            print(f'[{ix}/{len(selected)}] {case["case_id"]} complete',flush=True)
    (output/'mismatches.json').write_text(json.dumps(mismatches,ensure_ascii=False,indent=2),encoding='utf-8')
    # Avoid silently hiding errors or treating calculated payload as measured networking.
    print(f'Finished {len(selected)} / 144 cases; mismatches={len(mismatches)}; {csv_path}')
    ray.shutdown()
    if mismatches: raise AssertionError(f'{len(mismatches)} output token mismatches; see mismatches.json')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='sshleifer/tiny-gpt2')
    ap.add_argument('--output',default='research/results')
    ap.add_argument('--limit',type=int,default=4,help='0 runs all 144')
    ap.add_argument('--modes',default='B0,D0,D1,D2,D3')
    ap.add_argument('--repeats',type=int,default=2)
    ap.add_argument('--warmup',type=int,default=1)
    ap.add_argument('--threads',type=int,default=1)
    ap.add_argument('--shards',type=int,default=3)
    ap.add_argument('--ray-address',default='')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--bounds',default='',help='Optional comma-separated boundaries e.g. 0,1,3,6')
    ap.add_argument('--profile-json',default='',help='JSON cost profile for automatic DP partition')
    a=ap.parse_args()
    if a.limit<0 or a.limit>144 or a.repeats<1 or a.warmup<0: ap.error('Invalid limit/repeats/warmup')
    benchmark(a)

if __name__=='__main__': main()

