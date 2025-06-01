import math
from functools import cache
import random 
from tqdm import tqdm
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
import os
torch.manual_seed(0)

from qlib.utils.incoherence_preprocessing.incoherence_process_functions import incoherence_process, incoherence_preprocess
from qlib.qlayers.trellis_quantizer import trellis_quantizer


class TrellisLinear(torch.nn.Module):
    def __init__(self,
                 T=256,
                 L=16,
                 V=2,
                 K=2,
                 tlut_bits=10,
                 decode_mode='LowBitSym',
                 incoh_proc_mode='qtip',
                 viterby_bs='auto',
                 init_device='cuda:0'):
        super().__init__()
        self.incoh_proc_mode = incoh_proc_mode
        self.init_device = init_device

        self.weight_quantizer = trellis_quantizer(
            L=L, 
            K=K, 
            V=V,
            T=T,
            decode_mode=decode_mode,
            tlut_bits=tlut_bits,
            viterby_bs=viterby_bs
        )
        self.input_quantizer = torch.nn.Identity()


    @torch.no_grad()
    def wrap_module(self, module, verbose=False, *args, **kwargs):
        orig_device = module.weight.device
        self.w_shape = module.weight.shape
        self.weight_quantizer = self.weight_quantizer.to(self.init_device)
        w = module.weight.to(self.init_device)
        
        # Incoherence processing
        if self.incoh_proc_mode == 'qtip':
            w, SU, SV = incoherence_preprocess(w)
            self.SU = torch.nn.Parameter(SU).to(self.init_device)
            self.SV = torch.nn.Parameter(SV).to(self.init_device)
        elif self.incoh_proc_mode == 'skip':
            self.SU = torch.nn.Parameter(torch.tensor(1.0)).to(self.init_device)
            self.SV = torch.nn.Parameter(torch.tensor(1.0)).to(self.init_device)
        else:
            raise RuntimeError
        

        # Scale to Normal(0,1)
        w_std = w.std()
        w = w / w_std
        self.SU *= torch.sqrt(w_std)
        self.SV *= torch.sqrt(w_std)

        # Quantize
        reco, states = self.weight_quantizer.quantize(w)
        if verbose:
            err = torch.mean((reco - w)**2)
            print(f"error (w-wq)^2: {err.mean():.3f}")

        self.trellis = self.weight_quantizer.pack_trellis(states)
        
        # self.trellis = torch.empty(
        #     self.w_shape[0] * self.w_shape[1] // self.weight_quantizer.T, 
        #     self.weight_quantizer.T // self.weight_quantizer.L * self.weight_quantizer.K,
        #     dtype = torch.uint16
        # )
        #print("self.trellis.shape", self.trellis.shape)
        
        return deepcopy(self).to(orig_device)


    @property
    def weight(self):
        if self.weight_quantizer.L == 16:
            w = self.weight_quantizer.reconstruct_weight_fast(self.trellis, self.w_shape)
        else:
            w = self.weight_quantizer.reconstruct_weight(self.trellis, self.w_shape)

        if self.incoh_proc_mode == 'qtip':
            w = incoherence_process(w.float(), self.SU, self.SV)
        elif self.incoh_proc_mode == 'skip':
            w = w * self.SU * self.SV
        else:
            raise RuntimeError
        return w


    def forward(self, x):
        x = self.input_quantizer(x)
        w = self.weight
        
        return F.linear(weight=w, input=x)

