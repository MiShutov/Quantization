import torch
from torch.nn.functional import one_hot
from qlib.utils.ste import ste_round_pass
from qlib.quantizers.lut_quantizer import QuantizerLUT


class LUTAutograd(torch.autograd.Function):
	@staticmethod
	def forward(x, levels):
		borders = (levels[:, 1:] + levels[:, :-1])/2
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
		ctx.lut_indices = output[1]
		ctx.set_materialize_grads(False)


	@staticmethod
	def backward(ctx, grad_output, *ignore_args): 
		if grad_output is None:
			return None, None, None

		x_grad = levels_grad = None

		if ctx.needs_input_grad[0]:
			x_grad = grad_output
		if ctx.needs_input_grad[1]:
			lut_indices = ctx.lut_indices
			lut_mask_binary = one_hot(lut_indices)
			levels_grad = (grad_output.unsqueeze(2)*lut_mask_binary).sum(dim=1)

		return x_grad, levels_grad


class LUTAutograd_reparametrized(torch.autograd.Function):
	@staticmethod
	def forward(x, levels):
		borders = (levels[:, 1:] + levels[:, :-1])/2
		lut_mask = x.unsqueeze(2) > borders.unsqueeze(1)
		lut_indices = lut_mask.sum(dim=2)
		x_q = torch.take_along_dim(
			input=levels,
			indices=lut_indices,
			dim=1
			)
		return x_q, None


	@staticmethod
	def setup_context(ctx, inputs, output):
		ctx.x = inputs[0]
		ctx.levels = inputs[1]
		ctx.set_materialize_grads(False)


	@staticmethod
	def backward(ctx, grad_output, *ignore_args): 
		if grad_output is None:
			return None, None, None

		x_grad = levels_grad = None

		x = ctx.x
		levels = ctx.levels
		diap_mask = x.unsqueeze(2) > levels.unsqueeze(1)
		diap_indices = diap_mask.sum(dim=2)
		
		edges_bot = diap_indices==0
		edges_top = diap_indices==levels.shape[1]

		if ctx.needs_input_grad[0]:
			x_grad = grad_output * ~(edges_bot + edges_top)
		if ctx.needs_input_grad[1]:

			levels_expand = torch.cat([levels[:, :1] + 1, levels], axis=1)

			l_i = torch.take_along_dim(
				input=levels_expand[:, :-1],
				indices=diap_mask[:, :, :-1].sum(dim=2),
				dim=1
				)

			l_i_plus_1 = torch.take_along_dim(
				input=levels_expand[:, 1:],
				indices=diap_mask[:, :, :-1].sum(dim=2),
				dim=1
				)

			l_delta = l_i_plus_1 - l_i
			x_q_scaled = (x - l_i)/l_delta
			
			l_i_grads = -x_q_scaled -x_q_scaled.round_()
			l_i_plus_1_grads = -l_i_grads - 1

			diap_indices_binary = one_hot(diap_indices)

			l_i_grads = l_i_grads.unsqueeze(2) * diap_indices_binary[:, :, 1:-1]
			l_i_grads = torch.cat([edges_bot.unsqueeze(2), l_i_grads], axis=2)
			l_i_plus_1_grads = l_i_plus_1_grads.unsqueeze(2) * diap_indices_binary[:, :, 1:-1]
			l_i_plus_1_grads = torch.cat([l_i_plus_1_grads, edges_top.unsqueeze(2)], axis=2)

			l_grads = l_i_grads + l_i_plus_1_grads
			levels_grad = (grad_output.unsqueeze(2)*l_grads).sum(dim=1)

		return x_grad, levels_grad


class QuantizerLUT_autograd_reparametrized(QuantizerLUT):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)


	def quantize(self, x):
		x_shape = x.shape
		if not self._initialized:
			self._initialize(x.detach())

		if self.additions is not None:
			x = x + self.additions

		x_q = LUTAutograd_reparametrized.apply(self.regroup(x), self.levels)[0]
		x_q = x_q.reshape(x_shape)
		return x_q


class QuantizerLUT_reparametrized(QuantizerLUT):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)


	def quantize(self, x):
		x_shape = x.shape
		if not self._initialized:
			self._initialize(x.detach())

		if self.additions is not None:
			x = x + self.additions

		x_ = self.regroup(x)

		diap_mask = x_.unsqueeze(2) > self.levels.unsqueeze(1)
		diap_indices = diap_mask.sum(dim=2)
		
		edges_bot = diap_indices==0
		edges_top = diap_indices==self.levels.shape[1]

		levels_expand = torch.cat([self.levels[:, :1] + 1, self.levels], axis=1)

		l_i = torch.take_along_dim(
			input=levels_expand[:, :-1],
			indices=diap_mask[:, :, :-1].sum(dim=2),
			dim=1
			)

		l_i_plus_1 = torch.take_along_dim(
			input=levels_expand[:, 1:],
			indices=diap_mask[:, :, :-1].sum(dim=2),
			dim=1
			)

		l_delta = l_i_plus_1 - l_i

		x_q_scaled = (x_ - l_i)/l_delta

		x_q = ste_round_pass(x_q_scaled)

		x_q = x_q * l_delta + l_i
		x_q = x_q * ~edges_bot + self.levels[:, :1] * edges_bot
		x_q = x_q * ~edges_top + self.levels[:, -1:] * edges_top

		x_q = x_q.reshape(x_shape)
		return x_q
