import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.functional import one_hot

class LUTAutograd(torch.autograd.Function):
	@staticmethod
	def forward(x, levels, borders):
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
		ctx.borders = inputs[2]
		ctx.x_q = output[0]
		ctx.lut_indices = output[1]
		ctx.set_materialize_grads(False)


	@staticmethod
	def backward(ctx, grad_output, *ignore_args): 
		if grad_output is None:
			return None, None, None

		x_grad = levels_grad = borders_grad = None

		if ctx.needs_input_grad[0]:
			x_grad = torch.ones_like(grad_output)
		if ctx.needs_input_grad[1]:
			lut_indices = ctx.lut_indices
			#borders = ctx.borders
			levels = ctx.levels
			lut_mask_binary = one_hot(lut_indices, num_classes=levels.shape[-1]).float()
			levels_grad = (grad_output.unsqueeze(2)*lut_mask_binary).sum(dim=1)
		if ctx.needs_input_grad[2]:
			#borders_grad = (levels_grad[:, 1:] + levels_grad[:, :-1])/2
			borders = ctx.borders
			borders_mask = (ctx.x.unsqueeze(2) > levels[:, 1:].unsqueeze(1))
			#print("borders_mask.shape", borders_mask.shape)
			borders_indices = borders_mask.sum(dim=2)
			borders_indices[borders_indices>borders.shape[-1]-1] = borders.shape[-1] - 1
			#print("borders_indices.shape", borders_indices.shape, borders_indices.max())
			borders_binary = one_hot(borders_indices, num_classes=borders.shape[-1]).float()
			borders_grad = -(grad_output.unsqueeze(2)*borders_binary).sum(dim=1)

		return x_grad, levels_grad, borders_grad


class QuantizerGigaLUT(nn.Module):
	def __init__(self,
			     group_size, 
				 bit_width, 
				 initialization_params,
				 with_additions=False):
		super().__init__()
		self.group_size = group_size
		self.bit_width = bit_width
		self.negative_clip = -2**(bit_width-1)
		self.positive_clip = 2**(bit_width-1) - 1
		
		self.levels = nn.Parameter(torch.tensor(1.0), requires_grad=True)
		self.borders = nn.Parameter(torch.tensor(1.0), requires_grad=True)

		self.initialization_params = initialization_params
		if self.initialization_params is None:
			self.initialization_params = {
				'optim' : 'Adam',
				'lr' : 1e-2,
				'steps' : 100,
				'loss_p' : 4
			}
		self._initialized = False
		self.with_additions = with_additions
		if self.with_additions:
			self.weight_additions=nn.Parameter(torch.zeros(1), requires_grad=True)

	# @property
	# @torch.no_grad()
	# def borders(self):
	# 	return (self.levels[:, 1:] + self.levels[:, :-1])/2

	@property
	def additions(self):
		if self.with_additions:
			return self.weight_additions
		else:
			return None


	def regroup(self, x):
		if self.group_size == 'tensor':
			return x.flatten()
	
		if self.group_size == 'channel':
			return x.flatten(start_dim=1)

		return x.reshape(-1, self.group_size)
	

	def _initialize(self, x):
		self._initialized = True

		if self.with_additions:
			self.weight_additions.data=torch.zeros(x.shape).to(x.device)

		with torch.no_grad():
			x_grouped = self.regroup(x)
			x_min = x_grouped.min(axis=-1)[0].unsqueeze(-1)
			x_max = x_grouped.max(axis=-1)[0].unsqueeze(-1)
			indices = torch.linspace(0, 1, self.bit_width**2, device=x.device).view(1, -1)
			self.levels.data = x_min.view(-1, 1) + (x_max.view(-1, 1) - x_min.view(-1, 1)) * indices
			self.borders.data = (self.levels[:, 1:] + self.levels[:, :-1])/2


		steps = self.initialization_params['steps']
		lr = self.initialization_params['lr']
		loss_fn = self.initialization_params['criteria']
		if self.initialization_params['optim']=='Adam':
			optim = Adam([self.levels, self.borders], lr=lr)
		elif self.initialization_params['optim']=='SGD':
			optim = SGD([self.levels, self.borders], lr=lr)
		else:
			raise RuntimeError(f'QuantizerLUT::_initialize: Can\'t use {self.initialization_params['optim']} optimizer')

		#print(self.borders)
		with torch.enable_grad():
			# optimize in fp32:
			x_ = x.float()
			for i in range(steps):
				x_q = self(x_)
				loss = loss_fn(x_, x_q)
				loss.backward()
				#print("self.levels.grad:", self.levels.grad)
				#torch.nn.utils.clip_grad_norm_(self.levels, max_norm=1.0)
				optim.step()
				optim.zero_grad()
				#print(self.borders)


	def quantize(self, x):
		x_shape = x.shape
		if not self._initialized:
			self._initialize(x.detach())

		if self.additions is not None:
			x = x + self.additions

		x_q = LUTAutograd.apply(self.regroup(x), self.levels, self.borders)[0]

		x_q = x_q.reshape(x_shape)
		return x_q - x.detach() + x
	

	def forward(self, x):
		return self.quantize(x)
