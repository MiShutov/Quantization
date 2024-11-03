import torch
import torch.nn as nn


class LSQAutograd(torch.autograd.Function):
	@staticmethod
	def forward(x, step, negative_clip, positive_clip):
		not_perform_clamp = negative_clip is None and positive_clip is None
		if not (x.requires_grad or step.requires_grad):
			if not_perform_clamp:
				return x.div(step).round().mul_(step),
			return x.div(step).clamp(negative_clip, positive_clip).round().mul_(step),

		x_scaled = x / step

		if not_perform_clamp:
			x_clamped = x_scaled.clone()
		else:
			x_clamped = torch.clamp(x_scaled, negative_clip, positive_clip)

		mask_pass_grad = x_clamped==x_scaled


		x_rounded = x_clamped.round_()
		
		step_grad_mult = None
		if step.requires_grad:
			step_grad_mult = x_rounded - x_scaled * mask_pass_grad

		return x_rounded * step, mask_pass_grad, step_grad_mult

	@staticmethod
	def setup_context(ctx, inputs, output):
		step = inputs[1]
		ctx.step_shape = step.shape
		ctx.save_for_backward(*output[1:])
		ctx.set_materialize_grads(False)

	@staticmethod
	def backward(ctx, grad_output, *ignore_args): 
		if grad_output is None:
			return None, None, None, None

		x_grad = step_grad = None
		mask_pass_grad, step_grad_mult = ctx.saved_tensors
		
		if ctx.needs_input_grad[0]:
			x_grad = mask_pass_grad
		if ctx.needs_input_grad[1]:
			reduction_dims = tuple(i for i, dim_size in enumerate(ctx.step_shape) if dim_size == 1)
			if len(reduction_dims)==0:
				reduction_dims - None
			step_grad = torch.sum(step_grad_mult * grad_output, keepdim=True, dim=reduction_dims)
		return x_grad, step_grad, None, None
	

class QuantizerLSQ(nn.Module):
	def __init__(self, group_size, bit_width):
		super().__init__()
		self.group_size = group_size
		self.bit_width = bit_width
		self.negative_clip = -2**(bit_width-1)
		self.positive_clip = 2**(bit_width-1) - 1
		self.step = nn.Parameter(torch.tensor(1.0), requires_grad=True)
		self.offset = nn.Parameter(torch.tensor(0.0), requires_grad=True)


	def regroup(self, x):
		return x.reshape(-1, self.group_size)
	
	
	def _initialize(self, x):
		x_grouped = self.regroup(x)
		x_min = x.min(axis=1)[0].unsqueeze(-1)
		x_max = x.max(axis=1)[0].unsqueeze(-1)
		
		self.step.data = (x_max - x_min) / (2**self.bit_width - 1)
		self.offset.data = -self.step / 2


	def quantize(self, x):
		x_q = x - self.offset
		x_q = LSQAutograd.apply(
			x, self.step, self.negative_clip, self.positive_clip
			)[0]
		return x_q + self.offset
	
	def forward(self, x):
		return self.quantize(x)
