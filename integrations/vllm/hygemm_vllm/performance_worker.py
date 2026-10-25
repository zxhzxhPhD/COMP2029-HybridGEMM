"""Development-only teacher forcing after model Graph, before logprob gather."""
from .quality_worker import QualityWorkerExtension


class PerformanceWorkerExtension(QualityWorkerExtension):
    def teacher_begin(self,prompts,continuations,batch):
        import torch
        self.formal_begin(batch)
        lookup={tuple(p):(c,i) for i,(p,c) in enumerate(zip(prompts,continuations))}
        assert len(lookup)==len(prompts)
        self._teacher_requests={};self._teacher_positions=[];self._teacher_lookup=lookup;self._teacher_prefill=[]
        self._teacher_active=True
        if not hasattr(self,'_teacher_execute'):
            self._teacher_execute=self.execute_model
            def execute(scheduled,*args,**kwargs):
                if self._teacher_active:
                    for req in scheduled.scheduled_new_reqs:
                        continuation,ordinal=self._teacher_lookup[tuple(req.prompt_token_ids)]
                        self._teacher_requests[req.req_id]=(len(req.prompt_token_ids),continuation,ordinal)
                    positions={r.req_id:r.num_computed_tokens for r in scheduled.scheduled_new_reqs}
                    positions.update(zip(scheduled.scheduled_cached_reqs.req_ids,scheduled.scheduled_cached_reqs.num_computed_tokens))
                    ns=scheduled.num_scheduled_tokens
                    if any(positions[r]<self._teacher_requests[r][0] for r in ns):
                        self._teacher_prefill.append(sorted((self._teacher_requests[r][2],positions[r],n) for r,n in ns.items()))
                return self._teacher_execute(scheduled,*args,**kwargs)
            self.execute_model=execute
            runner_sample=self.model_runner.sample
            def model_sample(hidden_states,input_batch,grammar_output):
                self._teacher_input_batch=input_batch
                return runner_sample(hidden_states,input_batch,grammar_output)
            self.model_runner.sample=model_sample
            sampler=self.model_runner.sampler
            self._teacher_sample=sampler.sample
            def sample(logits,expanded_idx_mapping,idx_mapping_np,pos,input_ids,expanded_local_pos,**kwargs):
                if not self._teacher_active:
                    return self._teacher_sample(logits,expanded_idx_mapping,idx_mapping_np,pos,input_ids,expanded_local_pos,**kwargs)
                ib=self._teacher_input_batch
                positions=pos.tolist();assert len(positions)==len(ib.req_ids)==len(logits)
                targets=[]
                for req,p in zip(ib.req_ids,positions):
                    length,continuation,ordinal=self._teacher_requests[req]
                    index=p+1-length
                    # Incomplete chunked prefill produces a discarded token.
                    targets.append(continuation[max(0,min(index,len(continuation)-1))])
                    self._teacher_positions.append({'request':req,'prompt_index':ordinal,'position':p,'target_index':index,
                        'full_batch_decode':self._formal_pending is not None})
                # The original logits are returned untouched. vLLM subsequently
                # gathers raw target logprobs and updates KV with these target IDs.
                return torch.tensor(targets,device=logits.device,dtype=torch.int64),logits
            sampler.sample=sample
        return True

    def teacher_end(self):
        self._teacher_active=False
        result=self.formal_end()
        result['positions']=self._teacher_positions
        result['prefill_schedule']=self._teacher_prefill
        return result
