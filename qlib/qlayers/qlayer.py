import torch
import torch.nn as nn
from copy import deepcopy

class QLayer(nn.Module):
	def __init__(self, **kwargs):
		super().__init__()
	
	def configure(self, module):
		pass

	def wrap_module(self, module):
		self.module = module
		self.configure(module)
		return deepcopy(self).to(module.weight.device)

	def forward(self, x):
		pass