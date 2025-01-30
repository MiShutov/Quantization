import torch
import gc
import ctypes
from ctypes.util import find_library

def free_unused_memory():
    torch.cuda.empty_cache()
    gc.collect()
    libc = ctypes.CDLL(find_library("c"))
    libc.malloc_trim(ctypes.c_int(0))