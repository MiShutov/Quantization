import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from qlib.utils.ste import ste_round_pass
from qlib.quantizers.lut_quantizer import QuantizerLUT


class QuantizerLUT_reparametrized(QuantizerLUT):
	def __init__(self,
			     group_size, 
				 bit_width, 
				 initialization_params,
				 with_additions=False):
		super().__init__(group_size, 
						 bit_width, 
						 initialization_params,
						 with_additions)


	def quantize(self, x):
		x_shape = x.shape
		if not self._initialized:
			self._initialize(x.detach())

		if self.additions is not None:
			x = x + self.additions

		x_q = LUTAutograd.apply(self.regroup(x), self.levels, self.borders_pos)[0]

		x_q = x_q.reshape(x_shape)
		return x_q - x.detach() + x


class LUTAutograd(torch.autograd.Function):
	@staticmethod
	def forward(x, levels, borders_pos):
		borders = borders_pos * levels[:, :-1] + (1-borders_pos) * levels[:, 1:]
		lut_mask = x.unsqueeze(2) > borders.unsqueeze(1)
		lut_indices = lut_mask.sum(dim=2)
		x_q = torch.take_along_dim(
			input=levels,
			indices=lut_indices,
			dim=1
			)
		return x_q, lut_indices


	@staticmethod
	def setup_context(ctx, inputs, output):
		ctx.x = inputs[0]
		ctx.levels = inputs[1]
		ctx.borders_pos = inputs[2]
		ctx.x_q = output[0]
		ctx.lut_indices = output[1]
		ctx.set_materialize_grads(False)


	@staticmethod
	def backward(ctx, grad_output, *ignore_args): 
		if grad_output is None:
			return None, None, None

		x_grad = levels_grad = borders_pos_grad = None

		if ctx.needs_input_grad[0]:
			x_grad = torch.ones_like(grad_output)
		if ctx.needs_input_grad[1]:
			lut_indices = ctx.lut_indices
			#borders = ctx.borders
			levels = ctx.levels
			lut_mask_binary = one_hot(lut_indices, num_classes=levels.shape[-1]).float()
			levels_grad = (grad_output.unsqueeze(2)*lut_mask_binary).sum(dim=1)
		if ctx.needs_input_grad[2]:
			return None
			# borders_pos = ctx.borders_pos
			# borders = borders_pos * levels[:, :-1] + (1-borders_pos) * levels[:, 1:]

			# borders_mask = (ctx.x.unsqueeze(2) > levels[:, 1:].unsqueeze(1))
			# borders_indices = borders_mask.sum(dim=2)
			# borders_indices[borders_indices>borders.shape[-1]-1] = borders.shape[-1] - 1
			# borders_binary = one_hot(borders_indices, num_classes=borders.shape[-1]).float()
			# borders_pos_grad = (grad_output.unsqueeze(2)*borders_binary).sum(dim=1)
			# #borders_pos_grad /= borders

		return x_grad, levels_grad, borders_pos_grad

