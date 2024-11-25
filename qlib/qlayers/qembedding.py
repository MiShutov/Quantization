import torch
import torch.nn as nn
import torch.nn.functional as F
from qlib.qlayers.qlayer import QLayer


class QEmbedding(QLayer):
	def __init__(self, weight_quantizer=None):
		super().__init__()
		self.weight_quantizer = weight_quantizer
		
	
	def forward(self, x):
		w = self.module.weight
		w_q = self.weight_quantizer(w)
		return F.embedding(x, w_q.to(x.device))
