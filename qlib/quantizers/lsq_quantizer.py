import torch
import torch.nn as nn
from qlib.utils.ste import ste_round_pass
from qlib.quantizers.quantizer import Quantizer


class QuantizerLSQ(Quantizer):
	def __init__(
			self, 
			group_size, 
			bit_width, 
			initializer, 
			use_offset=True, 
			with_additions=False
			):
		super().__init__(group_size, bit_width)
		self.negative_clip = -2**(bit_width-1)
		self.positive_clip = 2**(bit_width-1) - 1
		self.use_offset = use_offset
		self.with_additions = with_additions
		self.initializer = initializer
	

	@torch.no_grad()
	def configure(self, module):
		module_weight_shape = self.regroup(module.weight).shape
		self.step = nn.Parameter(torch.empty(module_weight_shape[0], 1), requires_grad=True)
		if self.use_offset:
			self.offset = nn.Parameter(torch.empty(module_weight_shape[0], 1), requires_grad=True)
		else:
			self.offset = None
		if self.with_additions:
			self.additions = nn.Parameter(torch.zeros(module_weight_shape), requires_grad=True)
	

	def _initialize(self, x):
		x_shape = x.shape
		self._initialized.data = torch.tensor(True)
		x_grouped = self.regroup(x)

		self.step.data = torch.ones(x_grouped.shape[0], 1).to(x.device)
		if self.use_offset:
			self.offset.data = torch.zeros(x_grouped.shape[0], 1).to(x.device)

		self.initializer(x_grouped, self)
		x = x_grouped.reshape(x_shape)
		

	def lsq_forward(self, x):
		x_scaled = x / self.step
		x_clamped = torch.clamp(x_scaled, self.negative_clip, self.positive_clip)
		x_q = ste_round_pass(x_clamped)
		x_q = self.step * x_q
		return x_q


	def quantize(self, x):
		x_shape = x.shape
		if not self._initialized:
			with torch.no_grad():
				self._initialize(x)

		x_q = self.regroup(x)
		
		if self.with_additions:
			x_q = x_q + self.additions

		if self.use_offset:
			x_q = x_q + self.offset 

		x_q = self.lsq_forward(x_q)
		
		if self.use_offset:
			x_q = x_q - self.offset
		
		return x_q.reshape(x_shape)
