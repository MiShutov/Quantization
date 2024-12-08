import torch
import torch.nn as nn


class Scaler(nn.Module):
    def __init__(self, trainable=False):
        super().__init__()
        self.trainable = trainable
        self._initialized = nn.Parameter(torch.tensor(False), requires_grad=False)
    
    def configure(self, module_weight):
        pass

    def scale(self, x):
        pass

    def unscale(self, x):
        pass

