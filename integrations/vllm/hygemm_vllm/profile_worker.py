"""Opt-in tracing annotations; no arithmetic, kernel, or dispatch changes."""
import torch
import time
from .worker import AuditWorkerExtension
from .native import DualLinear
from .mixllm import MixLinear
from .quantization import MixedLinearMethod


def _install_capture_labels():
    original_load = MixedLinearMethod.process_weights_after_loading

    def load(self, layer):
        result = original_load(self, layer)
        self.runtime.profile_name = self.prefix
        return result

    MixedLinearMethod.process_weights_after_loading = load
    for cls in [DualLinear, MixLinear]:
        original = cls.__call__

        def call(self, x, original=original):
            if not torch.cuda.is_current_stream_capturing():
                return original(self, x)
            label = (f"projection|{getattr(self, 'profile_name', 'unknown')}|"
                     f"O={self.outputs}|K={self.k}|H={self.hi.numel()}|T={x.shape[0]}")
            with torch.cuda.nvtx.range(label):
                return original(self, x)

        cls.__call__ = call


_install_capture_labels()


class ProfileWorkerExtension(AuditWorkerExtension):
    def meter_begin(self, batch):
        """Two events around existing Graph replay; no nodes inserted in Graph."""
        self.formal_begin(batch)
        if not hasattr(self, '_meter_pool'):
            self._meter_pool = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(1024)]
            full = self.model_runner.cudagraph_manager.run_fullgraph

            def fullgraph(*args, **kwargs):
                if self._formal_pending is None:
                    return full(*args, **kwargs)
                index = len(self._formal_records)
                a, b = self._meter_pool[index]
                a.record()
                start = time.perf_counter()
                result = full(*args, **kwargs)
                cpu_ms = (time.perf_counter() - start) * 1000
                b.record()
                self._meter_rows.append((index, a, b, cpu_ms))
                return result

            self.model_runner.cudagraph_manager.run_fullgraph = fullgraph
        self._meter_rows = []
        return True

    def meter_end(self):
        result = self.formal_end()
        assert len(self._meter_rows) == result['steps']
        interior = self._meter_rows[10:38]
        assert len(interior) == 28
        graph_ms = [a.elapsed_time(b) for _, a, b, _ in interior]
        period = interior[0][1].elapsed_time(self._meter_rows[38][1]) / 28
        result['roi_period_ms'] = period
        result['roi_graph_ms'] = sum(graph_ms) / 28
        result['roi_outside_graph_ms'] = period - result['roi_graph_ms']
        result['roi_graph_cpu_ms'] = sum(r[3] for r in interior) / 28
        result['roi_graph_samples_ms'] = graph_ms
        positions = [pos for _, _, pos in self._formal_records[10:38]]
        result['roi_context'] = [min(map(min, positions)), max(map(max, positions))]
        return result

    def profile_begin(self, batch, skip=8, steps=32):
        self.formal_begin(batch)
        if not hasattr(self, '_profile_execute'):
            self._profile_execute = self._formal_original_execute
            self._profile_sample = self._formal_original_sample
            full = self.model_runner.cudagraph_manager.run_fullgraph

            def fullgraph(*args, **kwargs):
                if self._profile_active:
                    with torch.cuda.nvtx.range('decode_full_graph'):
                        return full(*args, **kwargs)
                return full(*args, **kwargs)

            self.model_runner.cudagraph_manager.run_fullgraph = fullgraph

            def execute(*args, **kwargs):
                if self._profile_enabled and self._formal_pending is not None:
                    ordinal = self._profile_eligible
                    self._profile_eligible += 1
                    if self._profile_skip <= ordinal < self._profile_skip + self._profile_steps:
                        if not self._profile_started:
                            # Boundary-only synchronization; discard edge steps in analysis.
                            torch.cuda.synchronize()
                            torch.cuda.cudart().cudaProfilerStart()
                            self._profile_started = True
                        self._profile_active = True
                        torch.cuda.nvtx.range_push(f'decode_step|{ordinal - self._profile_skip}')
                        self._profile_ordinals.append(len(self._formal_records))
                return self._profile_execute(*args, **kwargs)

            def sample(*args, **kwargs):
                output = self._profile_sample(*args, **kwargs)
                if self._profile_active:
                    torch.cuda.nvtx.range_pop()
                    self._profile_active = False
                    if len(self._profile_ordinals) == self._profile_steps:
                        torch.cuda.synchronize()
                        torch.cuda.cudart().cudaProfilerStop()
                        self._profile_enabled = False
                return output

            self._formal_original_execute = execute
            self._formal_original_sample = sample
        self._profile_enabled = True
        self._profile_active = False
        self._profile_started = False
        self._profile_eligible = 0
        self._profile_ordinals = []
        self._profile_skip, self._profile_steps = int(skip), int(steps)
        return True

    def profile_end(self):
        result = self.formal_end()
        assert len(self._profile_ordinals) == self._profile_steps
        result['profile_ordinals'] = self._profile_ordinals
        rows = [self._formal_records[i] for i in self._profile_ordinals]
        # Drop the start/stop instrumentation boundaries from latency diagnostics.
        interior = rows[2:-2]
        result['profile_interior_steps'] = len(interior)
        result['profile_interior_event_ms_per_step'] = interior[0][0].elapsed_time(interior[-1][1]) / len(interior)
        result['profile_interior_context'] = [min(min(r[2]) for r in interior), max(max(r[2]) for r in interior)]
        return result
