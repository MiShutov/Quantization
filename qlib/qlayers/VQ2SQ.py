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
                 weight_shape=None,
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
        self.weight_shape = weight_shape
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
        self.SU = torch.nn.Parameter(torch.empty(self.weight_shape[1], dtype=torch.float16), requires_grad=True)
        self.SV = torch.nn.Parameter(torch.empty(self.weight_shape[0], dtype=torch.float16), requires_grad=True)
        
        self.scales = torch.nn.Parameter(
            torch.ones(
                (self.weight_shape[0] * self.weight_shape[1] // 256, 1), 
                dtype=torch.float16
            ),
            requires_grad=True
        )
        
        trellis_shape = (
            self.weight_shape[0] * self.weight_shape[1] // self.weight_quantizer.T, 
            self.weight_quantizer.T // 16 * self.weight_quantizer.K
        )
        self.trellis = torch.nn.Parameter(torch.empty(trellis_shape, dtype=torch.uint16), requires_grad=False)
        # TODO bias

    @torch.no_grad()
    def wrap_module(self, module, verbose=False, *args, **kwargs):
        # TODO bias
        orig_device = module.weight.device
        self.weight_quantizer = self.weight_quantizer.to(self.init_device)
        w = module.weight.to(self.init_device)
        
        # Incoherence processing
        if self.incoh_proc_mode == 'qtip':
            w, SU, SV = incoherence_preprocess(w)
            self.SU.data = SU.to(self.init_device)
            self.SV.data = SV.to(self.init_device)
        elif self.incoh_proc_mode == 'skip':
            self.SU = self.SU.to(self.init_device)
            self.SV = self.SU.to(self.init_device)
        else:
            raise RuntimeError
        

        # Scale to Normal(0,1)
        w_std = w.std()
        w = w / w_std
        self.SU *= torch.sqrt(w_std)
        self.SV *= torch.sqrt(w_std)

        # Quantize
        if verbose and kwargs.get('module_name', False):
            print(kwargs['module_name'])
        reco, states = self.weight_quantizer.quantize(w)
        if verbose:
            err = torch.mean((reco - w)**2)
            print(f"error (w-wq)^2: {err.mean():.3f}")

        self.trellis.data = self.weight_quantizer.pack_trellis(states)
        
        return deepcopy(self).to(orig_device)


    @property
    def weight(self):
        if self.weight_quantizer.L == 16:
            w = self.weight_quantizer.reconstruct_weight_fast(self.trellis, self.weight_shape)
        else:
            w = self.weight_quantizer.reconstruct_weight(self.trellis, self.weight_shape)
        
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
        
        w = w.reshape(-1, 256)
        w = w * self.scales
        w = w.reshape(self.weight_shape)

        return F.linear(weight=w, input=x)

