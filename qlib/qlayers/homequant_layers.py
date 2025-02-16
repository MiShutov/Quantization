import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class VQLayer(nn.Module):
    def __init__(self, path_to_init_data):
        super().__init__()
        self.path_to_init_data = path_to_init_data
        self.indices = None
        self.codebook = None
        self.scales = None
        self.weight_shape = None

    @torch.no_grad()
    def wrap_module(self, module, module_name):
        self.weight_shape = module.weight.shape # [out_channels, in_channels]

        quant_data = torch.load(os.path.join(self.path_to_init_data, f'{module_name}.pth'), weights_only=True)
        self.indices = nn.Parameter(quant_data['indices'].to(torch.uint16), requires_grad=False) # shape: [n_indices,] = [out_channels * in_channels // vector_size]
        self.codebook = nn.Parameter(quant_data['codebook'].to(module.weight.dtype)) # shape: [codebook_size, vector_size]
        self.scales = nn.Parameter(quant_data['scales'].to(module.weight.dtype)) # shape: [out_channels, 1]
        return deepcopy(self).to(module.weight.device)

    @property
    def weight(self):
        return self.codebook[self.indices.to(torch.int)].reshape(self.weight_shape) * self.scales

    def forward(self, x):
        pass


class VQLinear(VQLayer):
    def __init__(self, path_to_init_data):
        super().__init__(path_to_init_data)

    def forward(self, x):
        w = self.weight
        return F.linear(x.to(w.dtype), self.weight)
