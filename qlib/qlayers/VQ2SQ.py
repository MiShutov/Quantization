from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F

from qlib.utils.incoherence_preprocessing.incoherence_process_functions import (
    incoherence_process, incoherence_preprocess, matmul_hadUt_cuda, matmul_hadU_cuda)
from qlib.qlayers.trellis_quantizer import TrellisQuantizer, TrellisQuantizerParams
from qlib.qlayers.input_quantizer import InputQuantizer, InputQuantizerParams


class TrellisLinear(torch.nn.Module):
    def __init__(
            self,
            weight_shape,
            weight_scales_group_size=256,
            incoh_proc_mode='qtip',
            init_device='cuda:0',
            input_quantizer_params=InputQuantizerParams(
                bit_width=8,
                group_mode="PER_CHANNEL",
                use_offset=True
            ),
            weight_quantizer_params=TrellisQuantizerParams(
                T=256,
                L=16,
                V=2,
                K=2,
                decode_mode="LowBitSym",
                viterbi_bs="auto"
            )
        ):
        super().__init__()
        self.weight_shape = weight_shape
        self.incoh_proc_mode = incoh_proc_mode
        self.init_device = init_device
        self.weight_quantizer = TrellisQuantizer(weight_quantizer_params)
        if input_quantizer_params is not None:
            self.input_quantizer = InputQuantizer(in_channels=weight_shape[1],
                                                  params=input_quantizer_params)
        else:
            self.input_quantizer = torch.nn.Identity()
        
        self.SU = torch.nn.Parameter(torch.empty(self.weight_shape[1], dtype=torch.float16), requires_grad=True)
        self.SV = torch.nn.Parameter(torch.empty(self.weight_shape[0], dtype=torch.float16), requires_grad=True)
        self.weight_scales_group_size = weight_scales_group_size
        
        # Init weight scales
        if isinstance(self.weight_scales_group_size, int):
            self.weight_scales = torch.nn.Parameter(
                torch.ones(
                    (self.weight_shape[0] * self.weight_shape[1] // weight_scales_group_size, 1), 
                    dtype=torch.float16
                ),
                requires_grad=True
            )
        elif self.weight_scales_group_size=='PER_CHANNEL':
            self.weight_scales = torch.nn.Parameter(
                torch.ones(
                    (self.weight_shape[0], 1), 
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


    @torch.no_grad()
    def update_indices(self, w_shifted):
        reassine_frac = self.reassine_params["reassine_frac"]
        vector_dim = self.weight_quantizer.T
        
        # Which vectors to reassign
        n_vectors_to_reassign = int(torch.numel(w_shifted) // vector_dim * reassine_frac)
        vector_ids_to_reassign = torch.topk(
            torch.max((
                    torch.abs(self.latent_weight) / (torch.abs(w_shifted) + 1e-10)
            ).reshape(-1, vector_dim), dim=1).values,
            n_vectors_to_reassign
        ).indices.to(torch.int)

        # # Get vectors to reassign
        vectors = w_shifted.reshape(-1, vector_dim)[vector_ids_to_reassign]

        # # Compute reassigns
        _, states = self.weight_quantizer.quantize(vectors)
        trellis = self.weight_quantizer.pack_trellis(states)
        
        # # Set new indices
        self_trellis_dtype = self.trellis.data.dtype
        self.trellis.data = self.trellis.data.to(torch.int)
        self.trellis.data[vector_ids_to_reassign] = trellis.to(torch.int)
        self.trellis.data = self.trellis.data.to(self_trellis_dtype)
        

    @property
    def weight(self):
        if self.weight_quantizer.L == 16:
            w = self.weight_quantizer.reconstruct_weight_fast(self.trellis, self.weight_shape)
        else:
            w = self.weight_quantizer.reconstruct_weight(self.trellis, self.weight_shape)
        

        if hasattr(self, 'latent_weight') and (self.trainable==True):
            self.step_counter += 1
            w_shifted = (w + self.latent_weight).detach()
            if self.step_counter % self.reassine_params.get("reassine_step", 1) == 0:
                self.update_indices(
                    w_shifted
                )
            w = w_shifted - self.latent_weight

        if self.incoh_proc_mode == 'qtip':
            #w = incoherence_process(w.float(), self.SU, self.SV).to(w.dtype)
            #w = (matmul_hadU_cuda(w.T.float()) * self.SV).to(w.dtype).T
            w = (matmul_hadU_cuda(w.T) * self.SV).T
        elif self.incoh_proc_mode == 'skip':
            w = w * self.SU * self.SV
        else:
            raise RuntimeError
        
        w = w.reshape(-1, self.weight_scales_group_size)
        w = w * self.weight_scales
        w = w.reshape(self.weight_shape)
        return w


    def forward(self, x):
        if self.incoh_proc_mode == 'qtip':
            #x = matmul_hadUt_cuda((x * self.SU).float()).to(x.dtype)
            x = matmul_hadUt_cuda(x * self.SU)
        x = self.input_quantizer(x)
        w = self.weight

        return F.linear(weight=w, input=x)


##### INCOHERENCE PROCESSING FOR ACTIVATIONS #####

# from qlib.utils.incoherence_preprocessing.incoherence_process_functions import (incoherence_process, 
#                                                                                 icoherence_process_left,
#                                                                                 icoherence_process_right,
# 																				incoherence_process_, 
# 																				incoherence_preprocess,
# 																				matmul_hadU_cuda,
# 																				matmul_hadUt_cuda)
# import torch.nn.functional as F

# w = torch.randn(11008, 4096).cuda()
# x = torch.randn(4, 4096, 4096).cuda()
# res = F.linear(weight=w, input=x)


# Wr, SU, SV = incoherence_preprocess(w)
# SU = SU.to(Wr.device)
# SV = SV.to(Wr.device)

# res1 = x @ incoherence_process(Wr, SU, SV).T
# assert torch.allclose(res, res1, atol=1e-3)

# res2 = x @ incoherence_process_(Wr, SU, SV).T
# assert torch.allclose(res, res2, atol=1e-3)

# res3 = x @ icoherence_process_left(icoherence_process_right(Wr, SU), SV).T
# assert torch.allclose(res, res3, atol=1e-3)

# res4 = x @ icoherence_process_left(matmul_hadU_cuda(Wr) * SU, SV).T
# assert torch.allclose(res, res4, atol=1e-3)

# res5 = x @ (matmul_hadU_cuda( (matmul_hadU_cuda(Wr) * SU).T) * SV.unsqueeze(0))
# assert torch.allclose(res, res5, atol=1e-3)

# res6 = (matmul_hadUt_cuda(x * SU)) @ (matmul_hadU_cuda(Wr.T) * SV)
# assert torch.allclose(res, res6, atol=1e-3)

# res7 = F.linear(weight=(matmul_hadU_cuda(Wr.T) * SV).T, input=matmul_hadUt_cuda(x * SU))
# assert torch.allclose(res, res7, atol=1e-3)