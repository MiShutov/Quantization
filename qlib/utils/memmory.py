import torch
import gc
import ctypes

def free_unused_memory():
    torch.cuda.empty_cache()
    gc.collect()
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    libc.malloc_trim(ctypes.c_int(0))