import gc
import torch

def clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def reset_cuda():
    clear_cuda()

def print_system_usage(tag=""):
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"[{tag}] RAM used: {(mem.total-mem.available)/1e9:.2f} GB / {mem.total/1e9:.2f} GB")
    except Exception:
        print(f"[{tag}] System usage not available.")
