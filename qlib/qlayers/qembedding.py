import torch.nn as nn
import torch.nn.functional as F

class QEmbedding(nn.Module):
	def __init__(self, module, weight_quantizer):
		super().__init__()
		self.module = module
		self.weight_quantizer = weight_quantizer
		
	
	def forward(self, x):
		w = self.module.weight
		w_q = self.weight_quantizer(w)
		return F.embedding(x, w_q)	