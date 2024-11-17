import torch
import torch.nn as nn
from qlib.quantizers.lsq_quantizer import QuantizerLSQ

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
	

class QuantizerLSQwithAutograd(QuantizerLSQ):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		
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
