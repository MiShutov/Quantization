import torch
import torch.nn as nn
import torch.nn.functional as F
from qlib.qlayers.qlayer import QLayer

class QLinear(QLayer):
	def __init__(self, weight_quantizer=None, input_quantizer=None):
		super().__init__()
		self.weight_quantizer = weight_quantizer
		self.input_quantizer = input_quantizer
	
	
	def forward(self, x):
		bias = self.module.bias

		if self.input_quantizer:
			x = self.input_quantizer(x)
			
		if self.weight_quantizer:
			w = self.module.weight
			w_q = self.weight_quantizer(w)
			return F.linear(x, w_q, bias)	

		return F.linear(x, w, bias)	
