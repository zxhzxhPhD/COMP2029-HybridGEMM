class AuditWorkerExtension:
    def hygemm_model_audit(self):
        from .quantization import model_audit
        return model_audit(self.model_runner.model)

    def formal_begin(self,batch):
        """Bracket pure fixed-batch decode from execute_model through sampling.

        No per-step device synchronization. Event span includes engine idle
        gaps between decode steps; prefill/mixed-batch steps are excluded.
        """
        import torch
        if not hasattr(self,'_formal_original_execute'):
            self._formal_original_execute=self.execute_model
            self._formal_original_sample=self.sample_tokens
            full=self.model_runner.cudagraph_manager.run_fullgraph
            def fullgraph(*args,**kwargs):
                if self._formal_enabled and self._formal_pending is not None:self._formal_graph_steps+=1
                return full(*args,**kwargs)
            self.model_runner.cudagraph_manager.run_fullgraph=fullgraph
            self._formal_pool=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(1024)]
            def execute(scheduled,*args,**kwargs):
                self._formal_pending=None
                if self._formal_enabled:
                    for req in scheduled.scheduled_new_reqs:
                        self._formal_prompt[req.req_id]=len(req.prompt_token_ids)
                        self._formal_position[req.req_id]=req.num_computed_tokens
                    cached=scheduled.scheduled_cached_reqs
                    for req,n in zip(cached.req_ids,cached.num_computed_tokens):
                        self._formal_position[req]=n
                    ns=scheduled.num_scheduled_tokens
                    decode=(len(ns)==self._formal_batch and all(n==1 and self._formal_position.get(req,-1)>=self._formal_prompt.get(req,10**9)
                                                               for req,n in ns.items()))
                    if decode:
                        slot=len(self._formal_records)
                        if slot>=len(self._formal_pool):raise RuntimeError('Timing event pool exhausted')
                        start,stop=self._formal_pool[slot]
                        positions=[self._formal_position[req] for req in ns]
                        self._formal_pending=(start,stop,positions)
                        start.record()
                    self._formal_total_steps+=1
                return self._formal_original_execute(scheduled,*args,**kwargs)
            def sample(*args,**kwargs):
                output=self._formal_original_sample(*args,**kwargs)
                if self._formal_enabled and self._formal_pending is not None:
                    start,stop,positions=self._formal_pending
                    stop.record()
                    self._formal_records.append((start,stop,positions))
                    self._formal_pending=None
                return output
            self.execute_model=execute
            self.sample_tokens=sample
        torch.cuda.synchronize()
        self._formal_enabled=True;self._formal_batch=int(batch)
        self._formal_records=[];self._formal_position={};self._formal_prompt={}
        self._formal_pending=None;self._formal_total_steps=0
        self._formal_graph_steps=0
        return True

    def formal_end(self):
        import torch
        self._formal_enabled=False
        torch.cuda.synchronize()
        rows=self._formal_records
        if not rows:raise RuntimeError('No fixed-batch pure decode steps were recorded')
        if self._formal_graph_steps!=len(rows):raise RuntimeError('A measured decode step did not use FULL Graph')
        times=[a.elapsed_time(b) for a,b,_ in rows]
        # The inclusive event span retains host/scheduler gaps between steps.
        span=rows[0][0].elapsed_time(rows[-1][1])
        return {'batch':self._formal_batch,'steps':len(rows),'graph_steps':self._formal_graph_steps,'total_engine_steps':self._formal_total_steps,
                'decode_span_ms':span,'decode_ms_per_step':span/len(rows),
                'sum_step_gpu_ms':sum(times),'step_gpu_ms':times,
                'context_min':min(min(pos) for _,_,pos in rows),'context_max':max(max(pos) for _,_,pos in rows)}
