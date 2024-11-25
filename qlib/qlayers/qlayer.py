import torch
import torch.nn as nn
from copy import deepcopy

class QLayer(nn.Module):
	def __init__(self, **kwargs):
		super().__init__()
	
	def wrap_module(self, module):
		self.module = module
		return deepcopy(self).to(module.weight.device)

	def forward(self, x):
		pass