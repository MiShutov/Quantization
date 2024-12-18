import torch
import torch.nn as nn
import torch.nn.functional as F
from qlib.qlayers.qlayer import QLayer
from qlib.quantizers.quantizer import Quantizer
from typing import List, Optional


class QLinear(QLayer):
	def __init__(
		self, 
		weight_quantizer: Optional[Quantizer]=None, 
		input_quantizer: Optional[Quantizer]=None, 
		#bias: Optional[nn.Parameter]=None
		):
		super().__init__()
		self.weight_quantizer = weight_quantizer
		self.input_quantizer = input_quantizer
		#self.bias = bias
	
	def configure(self, module):
		if self.input_quantizer:
			self.input_quantizer.configure(module)
		if self.weight_quantizer:
			self.weight_quantizer.configure(module)
	
	def forward(self, x):
		bias = self.module.bias

		if self.input_quantizer:
			x = self.input_quantizer(x)
			
		if self.weight_quantizer:
			w = self.module.weight
			w_q = self.weight_quantizer(w)
			return F.linear(x, w_q, bias)	

		return F.linear(x, w, bias)	
