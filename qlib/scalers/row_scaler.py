import torch
import torch.nn as nn
from qlib.scalers.scaler import Scaler


class RowScaler(Scaler):
    def __init__(self, n_blocks=1, trainable=False):
        super().__init__()
        self.n_blocks = n_blocks
        self.scales = nn.Parameter(torch.tensor(1.0), requires_grad=trainable)
        self._initialized = False
    
    @torch.no_grad()
    def _initialize(self, x):
        rows, cols = x.shape
        shift = cols // self.n_blocks
        
        blocks = x.view(rows, self.n_blocks, shift)
        block_scales = blocks.std(dim=2, unbiased=False)
        self.scales.data = block_scales
        self._initialized = True
        #print('scaler initialized!')


    def scale(self, x):
        if not self._initialized:
            self._initialize(x)
        rows, cols = x.shape
        shift = cols // self.n_blocks
        
        blocks = x.view(rows, self.n_blocks, shift)
        x_scaled = blocks / self.scales.unsqueeze(-1)
        return x_scaled.view(x.shape)

    def unscale(self, x_scaled):
        if not self._initialized:
            raise RuntimeError("StdScaler::unscale error: scaler was not initialized!")
        rows, cols = x_scaled.shape
        shift = cols // self.n_blocks
        
        blocks = x_scaled.view(rows, self.n_blocks, shift)
        x = blocks * self.scales.unsqueeze(-1)
        return x.view(x_scaled.shape)
