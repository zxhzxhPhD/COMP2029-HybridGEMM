"""Opt-in startup-only quantization/layout timing, outside inference runs."""
records=[]
_depth=0
_installed=False


def install():
    global _installed
    if _installed:return
    from .quantization import MixedLinearMethod
    from .offline import OfflineLinearMethod
    import time
    import torch
    def decorate(original):
        def process(self,layer):
            global _depth
            outer=_depth==0;_depth+=1
            if outer:
                torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                begin=time.perf_counter();a.record()
            try:value=original(self,layer)
            finally:_depth-=1
            if outer:
                b.record();b.synchronize()
                records.append({'module_id':id(layer),'name':getattr(self,'prefix',type(self).__name__),
                    'method':type(self).__name__,'host_elapsed_ms':(time.perf_counter()-begin)*1000,
                    'gpu_event_span_ms':a.elapsed_time(b)})
            return value
        return process
    # Offline's outer call includes its bank import; the nested RTN preparation
    # is not double counted. No forward method or device kernel is modified.
    MixedLinearMethod.process_weights_after_loading=decorate(MixedLinearMethod.process_weights_after_loading)
    OfflineLinearMethod.process_weights_after_loading=decorate(OfflineLinearMethod.process_weights_after_loading)
    _installed=True
