from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from qlib import MinMaxInitializer


def ste_round_pass(x):
	return x.round().detach() - x.detach() + x

@dataclass
class LSQQuantizerParams:
	bit_width : int = None
	group_size : int = None
	use_offset : bool = True
	use_scaled : bool = False
	initialization_method : str = "minmax"


class LSQQuantizer:
	def __init__(
			self, 
			params : LSQQuantizerParams
			):
		self.bit_width = params.bit_width
		self.negative_clip = -2**(params.bit_width-1)
		self.positive_clip = 2**(params.bit_width-1) - 1

		self.group_size = params.group_size
		self.use_offset = params.use_offset
		self.use_scaled = params.use_scaled
		self.initialization_method = params.initialization_method


	def initialize(self, x):
		x_shape = x.shape

		if self.initialization_method =='minmax':
			scale, offset = MinMaxInitializer()(x, self)
		else:
			raise

		x_q_reshaped = x.reshape(x_shape[0], x_shape[1] // self.group_size, self.group_size)

		if self.use_offset:
			x_q_reshaped = x_q_reshaped + offset.unsqueeze(-1)
		x_q_reshaped = (x_q_reshaped / scale.unsqueeze(-1)).round_().clamp_(self.negative_clip, self.positive_clip)
		x_q = x_q_reshaped.reshape(x_shape[0], x_shape[1])
		return x_q, scale, offset


	# def lsq_forward(self, x):
	# 	x_scaled = x / self.step
	# 	x_clamped = torch.clamp(x_scaled, self.negative_clip, self.positive_clip)
	# 	x_q = ste_round_pass(x_clamped)
	# 	x_q = self.step * x_q
	# 	return x_q


	# def quantize(self, x):
	# 	w_shape = x.shape

	# 	if self.with_additions:
	# 		x_q = x_q + self.additions

	# 	if self.use_offset:
	# 		x_q = x_q + self.offset 

	# 	x_q = self.lsq_forward(x_q)
		
	# 	if self.use_offset:
	# 		x_q = x_q - self.offset
		
	# 	return x_q.reshape(w_shape)


class LSQLinear(torch.nn.Module):
	def __init__(
			self,
			weight_shape,
			bias,
			params,
		):
		super().__init__()
		self.weight_shape = weight_shape
		self.bias = bias
		self.processing_type = params.get("processing", None)
		self.weight_quantizer = LSQQuantizer(
			LSQQuantizerParams(**params["weight_quantizer_params"])
		)		
		self.act_quantizer_params = params.get("act_quantizer_params", None)

		self.register_buffer(
			"quant_weight", 
			torch.empty(
				weight_shape[0], 
				weight_shape[1], 
				dtype=torch.int8, 
				requires_grad=False
			)
		)

		self.scale = torch.nn.Parameter(
			torch.empty(
			   weight_shape[0], 
			   weight_shape[1] // self.weight_quantizer.group_size, 
			   dtype=torch.float32
			)
		)
		if self.weight_quantizer.use_offset:
			self.offset = torch.nn.Parameter(
				torch.empty(
				weight_shape[0], 
				weight_shape[1] // self.weight_quantizer.group_size, 
				dtype=torch.float32
				)
			)

	@torch.no_grad()
	def configure(self, fp_weight, verbose=False, *args, **kwargs):
		# TODO bias
		assert list(self.weight_shape) == list(fp_weight.shape)
		quant_weight, scale, offset = self.weight_quantizer.initialize(fp_weight)
		self.quant_weight = quant_weight.to(torch.int8)
		self.scale.copy_(scale)
		if self.weight_quantizer.use_offset:
			self.offset.copy_(offset)
		
	@property
	def weight(self):
		w = self.quant_weight.reshape(
			self.weight_shape[0], 
			self.weight_shape[1] // self.weight_quantizer.group_size, 
			self.weight_quantizer.group_size
		)
		w = w * self.scale.unsqueeze(-1)
		
		if self.weight_quantizer.use_offset:
			w = w - self.offset.unsqueeze(-1)

		return w.reshape(self.weight_shape)

	@torch.compile
	def forward(self, x):
		w = self.weight
		out = F.linear(weight=w, input=x)
		return out
		
