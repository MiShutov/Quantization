from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F

from qlib.utils.incoherence_preprocessing.incoherence_process_functions import (
    incoherence_process, 
    incoherence_preprocess, 
    incoherence_process_lukashevich, 
    incoherence_preprocess_lukashevich, 
    matmul_hadUt_cuda, 
    matmul_hadU_cuda)
from qlib.qlayers.trellis_quantizer import TrellisQuantizer, TrellisQuantizerParams
from qlib.qlayers.input_quantizer import InputQuantizer, InputQuantizerParams

from microxcaling.mx import finalize_mx_specs
from microxcaling.mx.elemwise_ops import quantize_elemwise_op
from microxcaling.mx.mx_ops import (
    _get_format_params, _reshape_to_blocks, 
    _shared_exponents, _undo_reshape_to_blocks,
)


def get_shared_exps(A, mx_specs, axes):
    block_size = mx_specs['block_size']
    if mx_specs["scale_bits"] == 0:
        scale_bits = 8
    else:
        scale_bits = mx_specs["scale_bits"]
    
    axes = [axes] if type(axes) == int else axes
    axes = [x + A.ndim if x < 0 else x for x in axes]    
    shared_exp_axes = [x + 1 for x in axes] if block_size > 0 else axes
            
    A, axes, orig_shape, padded_shape = _reshape_to_blocks(
        A, axes, block_size
    )

    shared_exp = _shared_exponents(
        A, method=mx_specs['shared_exp_method'], axes=shared_exp_axes, ebits=0,
    )

    emax = _get_format_params(mx_specs['w_elem_format'])[2]
    shared_exp = shared_exp - emax
    
    scale_emax = 2**(scale_bits-1) - 1
    shared_exp[shared_exp > scale_emax] = float("NaN")
    shared_exp[shared_exp < -scale_emax] = -scale_emax
    
    return shared_exp


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
            self.input_quantizer = InputQuantizer(in_channels=weight_shape[1],
                                                  use_as_Identity=True,
                                                  params=None
                                                  )
        
        if self.incoh_proc_mode == 'qtip':
            self.SU = torch.nn.Parameter(torch.empty(self.weight_shape[1], dtype=torch.float16), requires_grad=True)
            self.SV = torch.nn.Parameter(torch.empty(self.weight_shape[0], dtype=torch.float16), requires_grad=True)
        elif self.incoh_proc_mode == 'lukashevich':
            self.SU = torch.nn.Parameter(torch.empty(self.weight_shape[1], dtype=torch.float16), requires_grad=True)
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
        assert list(self.weight_shape) == list(module.weight.data.shape)
        orig_device = module.weight.device
        self.weight_quantizer = self.weight_quantizer.to(self.init_device)
        w = module.weight.to(self.init_device)
        
        # Incoherence preprocessing
        if self.incoh_proc_mode in ['qtip', 'qtip_act', 'lukashevich']:
            w_std_scales = w.std(dim=-1, keepdim=True) / w.std()
            w_std_scales = torch.ones_like(w_std_scales)
            w_std_scaled = w / w_std_scales
            if self.incoh_proc_mode in ['qtip', 'qtip_act']:
                w_inc, SU, SV = incoherence_preprocess(w_std_scaled)
                self.SU.data = SU.to(self.init_device)
                self.SV.data = SV.to(self.init_device)
            elif self.incoh_proc_mode == 'lukashevich':
                w_inc, SU = incoherence_preprocess_lukashevich(w_std_scaled)
                self.SU.data = SU.to(self.init_device)
            #self.SU.data *= w_std_scales[:, 0]
        elif self.incoh_proc_mode == 'skip':
            w_inc = w
        else:
            raise RuntimeError
        

        if self.weight_quantizer.mx_mode:
            mx_specs = {
                'w_elem_format': self.weight_quantizer.w_elem_format,
                'block_size': self.weight_scales_group_size,
                'bfloat': 16,
                'custom_cuda': True,
                'quantize_backprop': False,
            }
            mx_specs = finalize_mx_specs(mx_specs)

            axes = [-1]
            block_size = mx_specs['block_size']
            shared_exp = get_shared_exps(w_inc, mx_specs, axes=axes)

            _w_inc, axes, orig_shape, padded_shape = _reshape_to_blocks(w_inc, axes, block_size)
            _w_inc_scaled = _w_inc / 2**shared_exp
            w_inc_scaled = _undo_reshape_to_blocks(_w_inc_scaled, padded_shape, orig_shape, axes)

            self.weight_quantizer.init_lut_mx(w_inc, shared_exp, mx_specs)
            self.weight_scales.data = (2**shared_exp).view(-1, 1)

            # Quantize
            if verbose and kwargs.get('module_name', False):
                print(kwargs['module_name'])
            w_inc_scaled_q, states = self.weight_quantizer.quantize(w_inc_scaled)
            if verbose:
                _w_inc_scaled_q, axes, orig_shape, padded_shape = _reshape_to_blocks(w_inc_scaled_q, axes, block_size)
                _w_inc_q = _w_inc_scaled_q * 2**shared_exp
                w_inc_q = _undo_reshape_to_blocks(_w_inc_q, padded_shape, orig_shape, axes)

                w_q = incoherence_process_lukashevich(w_inc_q, self.SU)
                w_q_grouped = w_q.reshape(-1, 256)
                w_grouped = w.reshape(-1, 256)

                err = torch.mean((w_q_grouped / w.std() - w_grouped / w.std()) ** 2, dim=-1)
                #err  = torch.mean(((w_q - w)**2).reshape(-1, 256), dim=-1)
                print(f"error: {err.mean():.3f} ± {err.std():.3f}")


            self.trellis.data = self.weight_quantizer.pack_trellis(states)
            return deepcopy(self).to(orig_device)

        else:
            # Scaling weights to N(0,1) -> scale sign vectors and scales
            w_std = w.std(dim=-1, keepdim=True)
            w_scaled = w / w_std


            # Incoherence preprocessing
            if self.incoh_proc_mode in ['qtip', 'qtip_act']:
                w_scaled_inc, SU, SV = incoherence_preprocess(w_scaled)
                self.SU.data = SU.to(self.init_device)
                self.SV.data = SV.to(self.init_device)
            elif self.incoh_proc_mode == 'lukashevich':
                w_scaled_inc, SU = incoherence_preprocess_lukashevich(w_scaled)
                self.SU.data = SU.to(self.init_device)
            else:
                raise RuntimeError

            # Scaling sign vectors and weight_scales according to weights scaling
            w_std_norm = w_std.mean()        
            self.SU.data *= torch.sqrt(w_std_norm)
            groups_per_row = w.shape[1] // self.weight_scales_group_size
            weight_scales_data = torch.ones_like(self.weight_scales.data)
            weight_scales_data = weight_scales_data.view(w.shape[0], groups_per_row)    
            if hasattr(self.weight_quantizer, "codebook_scale"):
                weight_scales_data *= self.weight_quantizer.codebook_scale
            weight_scales_data *= (w_std / torch.sqrt(w_std_norm)).expand(-1, groups_per_row).to(weight_scales_data.device)
            self.weight_scales.data = weight_scales_data.view(-1, 1)

            # Quantize
            if verbose and kwargs.get('module_name', False):
                print(kwargs['module_name'])
            reco, states = self.weight_quantizer.quantize(w_scaled_inc)
            if verbose:
                err = torch.mean((reco * self.weight_quantizer.codebook_scale - w_scaled_inc)**2)
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
        

    def apply_weight_scales(self, w):
        w = w.reshape(-1, self.weight_scales_group_size)
        w = w * self.weight_scales
        w = w.reshape(self.weight_shape)
        #TODO: self.weight_scales_group_size=='PER_CHANNEL'
        return w

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
            w = incoherence_process(w, self.SU, self.SV)

        w = self.apply_weight_scales(w)
        return w


    def forward(self, x):
        x = self.input_quantizer(x, incoh_proc_mode=self.incoh_proc_mode, SU=self.SU)
        w = self.weight

        out = F.linear(weight=w, input=x)
        
        if (self.incoh_proc_mode == 'qtip_act'):
            out = matmul_hadU_cuda(out) * self.SV
        
        return out


##### INCOHERENCE PROCESSING FOR ACTIVATIONS #####

# from qlib.utils.incoherence_preprocessing.incoherence_process_functions import (incoherence_process, 
#                                                                                 incoherence_preprocess,
# 																				  matmul_hadU_cuda,
# 																				  matmul_hadUt_cuda)
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


### SPEED TEST

# seq_len = 25
# bs = 100

# w = torch.randn(11008, 4096).cuda().half()
# x = torch.randn(bs, seq_len, 4096).cuda().half()
# res = F.linear(weight=w, input=x)

# Wr, SU, SV = incoherence_preprocess(w)
# SU = SU.to(Wr.device).half()
# SV = SV.to(Wr.device).half()
# Wr = Wr.half()

#res1 = F.linear(input=x, weight=incoherence_process(Wr, SU, SV))
#assert torch.allclose(res, res1, atol=1e-3)

#res2 = matmul_hadU_cuda(F.linear(input=matmul_hadUt_cuda(x * SU), weight=Wr)) * SV
#assert torch.allclose(res, res2, atol=1e-3)

#All in ms

### seqlen 25
#  bs  t1   t2
# 100  24   15  # 2500 

### seqlen 50
#  bs  t1   t2 
# 100  32   29  # 5000

### seqlen 512
#  bs  t1   t2
# 100  186  291 # 5120

### seqlen 4096
# bs  t1   t2
#  1  29   24   #4096
#  5  84   117  #20000
# 10  151  230  #40000
# 20  287  OOM

### seqlen 16384
# bs  t1   t2   
#  1  70   93   #16000
#  4 235  373   #64000

