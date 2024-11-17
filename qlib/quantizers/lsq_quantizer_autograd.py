import torch
import torch.nn as nn


class LSQAutograd(torch.autograd.Function):
	@staticmethod
	def forward(x, step, negative_clip, positive_clip):
		x_scaled = x / step
		x_clamped = torch.clamp(x_scaled, negative_clip, positive_clip)
		mask_pass_grad = x_clamped==x_scaled
		x_rounded = x_clamped.round_()

		#step_grad_mult = x_rounded - x_scaled * mask_pass_grad
		step_grad_mult = x_rounded #- x_scaled * mask_pass_grad
		
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
			x_grad = grad_output * mask_pass_grad
		if ctx.needs_input_grad[1]:
			reduction_dims = tuple(i for i, dim_size in enumerate(ctx.step_shape) if dim_size == 1)
			if len(reduction_dims)==0:
				reduction_dims = None
			step_grad = torch.sum(step_grad_mult * grad_output, keepdim=True, dim=reduction_dims)
		return x_grad, step_grad, None, None
	

class QuantizerLSQwithAutograd(nn.Module):
	def __init__(self, group_size, bit_width, use_offset=True, initializer=None):
		super().__init__()
		self.group_size = group_size
		self.bit_width = bit_width
		self.negative_clip = -2**(bit_width-1)
		self.positive_clip = 2**(bit_width-1) - 1
		self.use_offset = use_offset
		self.initializer = initializer
		self._initialized = False
		self.step = nn.Parameter(torch.ones(1, 1), requires_grad=True)
		if self.use_offset:
			self.offset = nn.Parameter(torch.zeros(1, 1), requires_grad=True)
		else:
			self.offset = None



	def regroup(self, x):
		if self.group_size == 'tensor':
			return x.flatten()
	
		if self.group_size == 'channel':
			return x.flatten(start_dim=1)

		return x.reshape(-1, self.group_size)
	
	
	def _initialize(self, x):
		self._initialized = True
		x_grouped = self.regroup(x)

		self.step.data = torch.ones(x_grouped.shape[0], 1).to(x.device)
		if self.use_offset:
			self.offset.data = torch.zeros(x_grouped.shape[0], 1).to(x.device)
		else:
			self.offset = None
		
		if self.initializer is not None:
			self.initializer(x_grouped, self)
		else:
			with torch.no_grad():
				x_min = x_grouped.float().min(axis=-1)[0].unsqueeze(-1)
				x_max = x_grouped.float().max(axis=-1)[0].unsqueeze(-1)
				step = (x_max - x_min) / (2**self.bit_width - 1)
				self.step.copy_(step)
				if self.use_offset:
					self.offset.copy_(torch.zeros_like(self.step.data))

	def quantize(self, x):
		x_shape = x.shape
		if not self._initialized:
			with torch.no_grad():
				self._initialize(x)

		x_q = self.regroup(x)
			
		if self.use_offset:
			x_q = x_q + self.offset 

		x_q = LSQAutograd.apply(
			x_q, self.step, self.negative_clip, self.positive_clip
			)[0]
		
		if self.use_offset:
			x_q = x_q - self.offset
	
		return x_q.reshape(x_shape)
	

	def forward(self, x):
		x_q = self.quantize(x)
		return x_q
