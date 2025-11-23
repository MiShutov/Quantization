import torch
from enum import Enum


FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


class QuantizationMode(Enum):
    TOKEN_INT8_127 = 1
    TOKEN_INT16_32767 = 2
    TOKEN_FP8 = 3


class DynamicActivationQuantizer:
    def __init__(
            self,
            quantization_mode,
            scale_dtype=None,
            **kwargs
        ):
        self.quantization_mode = QuantizationMode[quantization_mode]
        self.scale_dtype = scale_dtype


    def configure(self, *args, **kwargs):
        pass


    def __call__(self, x, simulation=True, ste=False):
        if self.quantization_mode == QuantizationMode.TOKEN_INT8_127:
            assert self.scale_dtype is not None
            scale_dtype = getattr(torch, self.scale_dtype)

            x_dtype = x.dtype
            per_token_max = torch.max(torch.abs(x), dim=-1, keepdim=True).values
            scale = (per_token_max / 127).to(scale_dtype)
            x_scaled = x / scale
            x_rounded = torch.round(x_scaled)

            if simulation:
                if ste:
                    x_rounded = x_rounded + (x - x.detach())
                return (x_rounded * scale).to(x_dtype)
            else:
                return x_rounded.to(torch.int8), scale.squeeze(-1)
        if self.quantization_mode == QuantizationMode.TOKEN_INT16_32767:
            assert self.scale_dtype is not None
            scale_dtype = getattr(torch, self.scale_dtype)

            x_dtype = x.dtype
            per_token_max = torch.max(torch.abs(x), dim=-1, keepdim=True).values
            scale = (per_token_max / 32767).to(scale_dtype)
            x_scaled = x / scale
            x_rounded = torch.round(x_scaled)

            if simulation:
                if ste:
                    x_rounded = x_rounded + (x - x.detach())
                return (x_rounded * scale).to(x_dtype)
            else:
                return x_rounded.to(torch.int8), scale
        if self.quantization_mode == QuantizationMode.TOKEN_FP8:
            assert self.scale_dtype is not None
            scale_dtype = getattr(torch, self.scale_dtype)

            x_dtype = x.dtype
            
            per_token_max = torch.max(torch.abs(x), dim=-1, keepdim=True).values
            scale = torch.max(per_token_max.to(torch.float32) / FP8_MAX, torch.tensor(1.0))
            x_scaled = x / scale
            x_rounded = x_scaled.to(torch.float8_e4m3fn)

            if simulation:
                if ste:
                    x_rounded = x_rounded.to(torch.float32) + (x - x.detach())
                return (x_rounded.to(torch.float32) * scale).to(x_dtype)
            else:
                return x_rounded, scale
        else:
            raise RuntimeError
