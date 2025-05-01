import torch
from torch.profiler import profile, ProfilerActivity


class Profiler:
    def __init__(self, enabled=True, wait=1, warmup=1, active=3, repeat=1, output_json=None):
        self.enabled = enabled
        self.output_json = output_json
        self.finalized = False
        self.total_profile_steps = (wait + warmup + active) * repeat
        self.step_counter = 0

        if self.enabled:
            self.profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
        else:
            self.profiler = None

    def __enter__(self):
        if self.enabled and self.profiler:
            self.profiler.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # No-op — cleanup handled in step()
        pass

    def step(self):
        if not self.enabled or self.finalized:
            return

        self.profiler.step()
        self.step_counter += 1

        if self.step_counter >= self.total_profile_steps:
            self._finalize()

    def _finalize(self):
        if self.profiler:
            self.profiler.__exit__(None, None, None)
            if self.output_json:
                self.profiler.export_chrome_trace(self.output_json)
        self.finalized = True