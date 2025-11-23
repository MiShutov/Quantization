import torch
import torch.nn as nn
from enum import Enum
from dataclasses import dataclass
from qlib.utils.incoherence_preprocessing.incoherence_process_functions import matmul_hadUt_cuda
# from microxcaling.mx import finalize_mx_specs
# from microxcaling.mx.mx_ops import quantize_mx_op

CALIB_SCALE = 1.1

class InputQuantGroupMode(Enum):
    PER_CHANNEL = 1
    PER_TENSOR = 2


# class IndentityQuantizer(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super(IndentityQuantizer, self).__init__()
    
#     def forward(self, x, *args, **kwargs):
#         return x

@dataclass
class InputQuantizerParams:
    bit_width : int = 8,
    group_mode : str = "PER_CHANNEL",
    use_offset : bool = True,
    calib_mode: bool = False,
    relative_scale: bool = False,
    mx_format: str = None


class InputQuantizer(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 params: InputQuantizerParams=None,
                 use_as_Identity=False):
        super().__init__()
        self.use_as_Identity = use_as_Identity
        if self.use_as_Identity:
            return

        if params.mx_format is not None:
            self.mx_specs = finalize_mx_specs({
                'w_elem_format': params.mx_format,
                'a_elem_format': params.mx_format,
                'block_size': 32,
                'custom_cuda': True,
                'quantize_backprop': False,
            })
            return


        self.in_channels = in_channels
        self.group_mode = InputQuantGroupMode[params.group_mode]
        self.calib_mode = params.calib_mode
        self.use_offset = params.use_offset
        if self.group_mode==InputQuantGroupMode.PER_TENSOR:
            self.act_scale = nn.Parameter(torch.tensor(0.0))
        elif self.group_mode==InputQuantGroupMode.PER_CHANNEL:
            self.act_scale = nn.Parameter(torch.zeros(in_channels))
        else:
            raise

        self.act_offset = None
        if self.use_offset:
            self.act_offset = nn.Parameter(torch.zeros_like(self.act_scale.data))

        self.bit_width = params.bit_width
        self.N = - 2 ** (self.bit_width - 1)
        self.P = 2 ** (self.bit_width - 1) - 1


    def forward(self, x, incoh_proc_mode='qtip', SU=None):
        if incoh_proc_mode in ['qtip_act', 'lukashevich']:
            #x = matmul_hadUt_cuda((x * self.SU).float()).to(x.dtype)
            x = matmul_hadUt_cuda(x * SU)

        if self.use_as_Identity:
            return x

        if hasattr(self, 'mx_specs'):
            x = quantize_mx_op(
                x,
                self.mx_specs,
                elem_format=self.mx_specs['a_elem_format'],
                axes=[-1],
                round=self.mx_specs["round_mx_output"],
            ).to(x.dtype)
            return x


        if self.calib_mode:
            if self.group_mode==InputQuantGroupMode.PER_TENSOR:
                x_scale = torch.max(x.min() / self.N, x.max() / self.P)
                if x_scale > self.act_scale.data:
                    self.act_scale.data = x_scale * CALIB_SCALE
            elif self.group_mode==InputQuantGroupMode.PER_CHANNEL:
                reduce_dims = tuple(range(x.dim() - 1))
                x_scale = torch.maximum(
                    x.amin(dim=reduce_dims) / self.N, 
                    x.amax(dim=reduce_dims) / self.P, 
                )
                replace_indices = self.act_scale.data < x_scale
                self.act_scale.data[replace_indices] = x_scale[replace_indices] * CALIB_SCALE
            else:
                raise
            return x
        else:
            x_scaled = (x - self.act_offset) / self.act_scale
            x_clamped = torch.clamp(x_scaled, self.N, self.P)
            x_q = (x_clamped.round_() - x_clamped).detach() + x_clamped
            x_q = x_q * self.act_scale + self.act_offset
            return x_q
