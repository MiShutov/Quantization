import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR


class Quantizer(nn.Module):
	def __init__(self,
			     group_size, 
				 bit_width):
		super().__init__()
		self.group_size = group_size
		self.bit_width = bit_width
		self._quantize = True
		self._initialized = False

	def regroup(self, x):
		if self.group_size == 'tensor':
			return x.flatten()
	
		if self.group_size == 'channel':
			return x.flatten(start_dim=1)

		return x.reshape(-1, self.group_size)

	def __repr__(self):
		return f"{self.__class__.__name__}(group_size={self.group_size}, bit_width={self.bit_width})"


		
