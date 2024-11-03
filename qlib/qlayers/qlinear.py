import torch.nn as nn
import torch.nn.functional as F

class QLinear(nn.Module):
	def __init__(self, module, weight_quantizer=None, input_quantizer=None):
		super().__init__()
		self.module = module
		self.weight_quantizer = weight_quantizer
		self.input_quantizer = input_quantizer
		
	
	def forward(self, x):
		if self.input_quantizer:
			x = self.input_quantizer(x)
			
		if self.weight_quantizer:
			w = self.module.weight
			w_q = self.weight_quantizer(w)
		
		bias = self.module.bias
		return F.linear(x, w_q, self.bias)	
