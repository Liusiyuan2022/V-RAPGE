import torch
import conf
def log_memory(step_name):
    if not conf.DEBUG:
        return
    print(f"[Memory at {step_name}] Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB, "
          f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
