import torch
import torch.nn as nn
from qlib.scalers.scaler import Scaler


class LoraScaler(Scaler):
    def __init__(self, n_blocks=1, rank=1):
        super().__init__()
        self.rank = rank
        self.n_blocks = n_blocks
        self.lora_row = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.lora_col = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    
    @torch.no_grad()
    def _initialize(self, x):
        self.lora_row.data = torch.ones(x.shape[0], self.rank).to(x.device) / (self.rank)**0.5
        self.lora_col.data = torch.ones(self.rank, self.n_blocks).to(x.device) / (self.rank)**0.5
        self._initialized.data = torch.tensor(True)

    def scale(self, x):
        if not self._initialized:
            self._initialize(x)
        rows, cols = x.shape
        shift = cols // self.n_blocks
        
        blocks = x.view(rows, self.n_blocks, shift)
        scales = self.lora_row @ self.lora_col
        x_scaled = blocks / scales.unsqueeze(-1)
        return x_scaled.view(x.shape)

    def unscale(self, x_scaled):
        if not self._initialized:
            raise RuntimeError("StdScaler::unscale error: scaler was not initialized!")
        rows, cols = x_scaled.shape
        shift = cols // self.n_blocks
        
        blocks = x_scaled.view(rows, self.n_blocks, shift)
        scales = self.lora_row @ self.lora_col
        x = blocks * scales.unsqueeze(-1)
        return x.view(x_scaled.shape)
