"""Read-only final-evaluation diagnostics; no quantization arithmetic changes."""
from .performance_worker import PerformanceWorkerExtension


class FinalWorkerExtension(PerformanceWorkerExtension):
    def task_scoring_begin(self):
        self._task_score_active=True;self._task_score_calls=0;self._task_score_max_requests=0
        if not hasattr(self,'_task_score_execute'):
            self._task_score_execute=self.execute_model
            def execute(scheduled,*args,**kwargs):
                if self._task_score_active:
                    count=len(scheduled.num_scheduled_tokens)
                    assert count<=1,'Task scoring must execute one sequence per device batch'
                    self._task_score_max_requests=max(self._task_score_max_requests,count)
                    self._task_score_calls+=1
                return self._task_score_execute(scheduled,*args,**kwargs)
            self.execute_model=execute
        return True

    def task_scoring_end(self):
        self._task_score_active=False
        return {'execute_calls':self._task_score_calls,'maximum_scheduled_requests':self._task_score_max_requests,
                'single_sequence_device_batches':self._task_score_max_requests<=1}

    def formal_begin(self,batch):
        result=super().formal_begin(batch)
        self._final_first_descriptor=None
        if not hasattr(self,'_final_graph_original'):
            self._final_graph_original=self.model_runner.cudagraph_manager.run_fullgraph
            def run(desc):
                if self._formal_enabled and self._formal_pending is not None and self._final_first_descriptor is None:
                    self._final_first_descriptor={'num_tokens':int(desc.num_tokens),'num_reqs':int(desc.num_reqs),'mode':str(desc.cg_mode)}
                return self._final_graph_original(desc)
            self.model_runner.cudagraph_manager.run_fullgraph=run
        return result

    def formal_end(self):
        value=super().formal_end()
        value['first_pure_decode_graph_descriptor']=self._final_first_descriptor
        assert self._final_first_descriptor is not None
        return value

    def teacher_begin(self,prompts,continuations,batch,distribution_path=None):
        result=super().teacher_begin(prompts,continuations,batch)
        self._distribution_path=distribution_path;self._distribution_records=[]
        if not hasattr(self,'_distribution_original_sample'):
            self._distribution_original_sample=self.model_runner.sampler.sample
            def sample(logits,expanded_idx_mapping,idx_mapping_np,pos,input_ids,expanded_local_pos,**kwargs):
                if self._teacher_active and self._distribution_path and self._formal_pending is not None:
                    for row,(req,p) in enumerate(zip(self._teacher_input_batch.req_ids,pos.tolist())):
                        length,continuation,ordinal=self._teacher_requests[req]
                        index=p+1-length
                        if ordinal==0 and index in [97,105,113,121]:
                            self._distribution_records.append({'target_index':index,'logprob':logits[row].detach().float().log_softmax(-1).cpu()})
                return self._distribution_original_sample(logits,expanded_idx_mapping,idx_mapping_np,pos,input_ids,expanded_local_pos,**kwargs)
            self.model_runner.sampler.sample=sample
        return result

    def teacher_end(self):
        value=super().teacher_end()
        if self._distribution_path:
            import torch
            from pathlib import Path
            path=Path(self._distribution_path);path.parent.mkdir(parents=True,exist_ok=True)
            assert sorted(r['target_index'] for r in self._distribution_records)==[97,105,113,121]
            torch.save(self._distribution_records,path)
            value['distribution_file']=str(path);value['distribution_targets']=[r['target_index'] for r in self._distribution_records]
        self._distribution_path=None;self._distribution_records=[]
        return value

    def final_runtime_audit(self):
        import os
        import torch
        from .quantization import MixedLinearMethod
        from .offline import OfflineLinearMethod
        from .load_timing import records as load_records
        model=self.model_runner.model;quantized={};registered={};modules=[];producers=[]
        for name,module in model.named_modules():
            method=getattr(module,'quant_method',None)
            if method is not None:
                item={'name':name,'method':type(method).__module__+'.'+type(method).__name__}
                for key in ['use_marlin','act_q_static','fp8_linear']:
                    if hasattr(method,key):
                        v=getattr(method,key);item[key]=v if isinstance(v,(str,int,float,bool)) else type(v).__module__+'.'+type(v).__name__
                modules.append(item)
            if isinstance(method,MixedLinearMethod):
                for key,value in vars(method.runtime).items():
                    if isinstance(value,torch.Tensor):
                        storage=value.untyped_storage()
                        if storage.nbytes():quantized[storage.data_ptr()]=storage.nbytes()
            down=getattr(module,'down_proj',None)
            if down is not None and hasattr(module,'gate_up_proj'):
                enabled=isinstance(getattr(down,'quant_method',None),OfflineLinearMethod) and bool(getattr(type(module),'_hygemm_producer_installed',False)) and os.environ.get('HYGEMM_FUSE_SILU')=='1'
                producers.append({'name':name,'class':type(module).__name__,'fused_decode_eligible':enabled})
        for _,value in list(model.named_parameters())+list(model.named_buffers()):
            storage=value.untyped_storage()
            if storage.nbytes() and storage.data_ptr() not in quantized:registered[storage.data_ptr()]=storage.nbytes()
        torch.cuda.synchronize();free,total=torch.cuda.mem_get_info();snapshot=torch.cuda.memory_snapshot()
        assert self.model_runner.kv_cache_dtype==torch.float16,'Final comparison requires FP16 KV for every method'
        pool_available=all('segment_pool_id' in r for r in snapshot)
        graph_segments=[r for r in snapshot if any(r['segment_pool_id'])] if pool_available else []
        return {'quantization_modules':modules,'producer_adapters':producers,
            'actual_kv_cache_dtype':str(self.model_runner.kv_cache_dtype),
            'kv_cache_blocks':self.model_runner.kv_cache_config.num_blocks,
            'startup_quantization_layout_timing':list(load_records),
            'quantized_runtime_tensor_storage_bytes':sum(quantized.values()),
            'registered_parameter_buffer_storage_bytes':sum(registered.values()),
            'model_representation_bytes':sum(quantized.values())+sum(registered.values()),
            'torch_allocated_bytes':torch.cuda.memory_allocated(),'torch_reserved_bytes':torch.cuda.memory_reserved(),
            'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'torch_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'device_used_bytes_including_context_and_external_allocations':total-free,
            'allocator_graph_pool_available':pool_available,
            'graph_pool_reserved_bytes':sum(r['total_size'] for r in graph_segments) if pool_available else None,
            'graph_pool_allocated_bytes':sum(r['allocated_size'] for r in graph_segments) if pool_available else None,
            'memory_scope':'Tensor storage deduplicated by address; allocator pools/peaks include runtime/KV/scratch. External CUDA workspace is not separately attributed.'}
