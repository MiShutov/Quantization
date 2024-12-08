import torch
import torch.nn as nn
from qlib.scalers.scaler import Scaler


class RowScaler(Scaler):
    def __init__(self, n_blocks=1, trainable=False):
        super().__init__(trainable=trainable)
        self.n_blocks = n_blocks

    @torch.no_grad()
    def configure(self, module_weight):
        rows, cols = module_weight.shape
        shift = cols // self.n_blocks
        blocks = module_weight.view(rows, self.n_blocks, shift)
        block_scales = blocks.std(dim=2, unbiased=False)
        self.scales = nn.Parameter(block_scales, requires_grad=self.trainable)


    def scale(self, x):
        rows, cols = x.shape
        shift = cols // self.n_blocks
        
        blocks = x.view(rows, self.n_blocks, shift)
        x_scaled = blocks / self.scales.unsqueeze(-1)
        return x_scaled.view(x.shape)


    def unscale(self, x_scaled):
        rows, cols = x_scaled.shape
        shift = cols // self.n_blocks
        
        blocks = x_scaled.view(rows, self.n_blocks, shift)
        x = blocks * self.scales.unsqueeze(-1)
        return x.view(x_scaled.shape)
