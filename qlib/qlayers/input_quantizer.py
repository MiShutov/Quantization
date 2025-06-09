import torch
import torch.nn as nn
from enum import Enum
from dataclasses import dataclass


class InputQuantGroupMode(Enum):
    PER_CHANNEL = 1
    PER_TENSOR = 2


@dataclass
class InputQuantizerParams:
    bit_width : int = 8,
    group_mode : str = "PER_CHANNEL",
    use_offset : bool = True,
    calib_mode: bool = False


class InputQuantizer(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 params: InputQuantizerParams):
        super().__init__()
        self.in_channels = in_channels
        self.group_mode = InputQuantGroupMode[params.group_mode]
        self.calib_mode = params.calib_mode
        if self.group_mode==InputQuantGroupMode.PER_TENSOR:
            self.act_scale = nn.Parameter(torch.tensor(0.0))
        elif self.group_mode==InputQuantGroupMode.PER_CHANNEL:
            self.act_scale = nn.Parameter(torch.zeros(in_channels))
        else:
            raise
        self.bit_width = params.bit_width
        self.N = - 2 ** (self.bit_width - 1)
        self.P = 2 ** (self.bit_width - 1) - 1

    def forward(self, x):
        if self.calib_mode:
            if self.group_mode==InputQuantGroupMode.PER_TENSOR:
                x_scale = torch.max(x.min() / self.N, x.max() / self.P)
                if x_scale > self.act_scale.data:
                    self.act_scale.data = x_scale * 1.1
            elif self.group_mode==InputQuantGroupMode.PER_CHANNEL:
                reduce_dims = tuple(range(x.dim() - 1))
                x_scale = torch.maximum(
                    x.amin(dim=reduce_dims) / self.N, 
                    x.amax(dim=reduce_dims) / self.P, 
                )
                replace_indices = self.act_scale.data < x_scale
                self.act_scale.data[replace_indices] = 1.1 * x_scale[replace_indices]
            else:
                raise
            return x
        else:
            x_scaled = x / self.act_scale
            x_clamped = torch.clamp(x_scaled, self.N, self.P)
            x_q = (x_clamped.round_() - x_clamped).detach() + x_clamped
            x_q = x_q * self.act_scale
            return x_q
