import torch
import torch.nn as nn
from torch.optim import Adam

	
class QuantizerLUT(nn.Module):
	def __init__(self, group_size, bit_width, initialization_params):
		super().__init__()
		self.group_size = group_size
		self.bit_width = bit_width
		self.negative_clip = -2**(bit_width-1)
		self.positive_clip = 2**(bit_width-1) - 1
		
		self.levels = nn.Parameter(torch.tensor(1.0), requires_grad=True)

		self.initialization_params = initialization_params
		if self.initialization_params is None:
			self.initialization_params = {
				'optim' : 'Adam',
				'lr' : 3e-3,
				'steps' : 250,
				'loss_p' : 4
			}
		self._initialized = False

	@property
	@torch.no_grad()
	def borders(self):
		return (self.levels[:, 1:] + self.levels[:, :-1])/2


	def regroup(self, x):
		if self.group_size == 'tensor':
			return x.flatten()
	
		if self.group_size == 'channel':
			return x.flatten(start_dim=1)

		return x.reshape(-1, self.group_size)
	

	def _initialize(self, x):
		self._initialized = True
		x_grouped = self.regroup(x)
		x_min = x_grouped.min(axis=-1)[0].unsqueeze(-1)
		x_max = x_grouped.max(axis=-1)[0].unsqueeze(-1)
		indices = torch.linspace(0, 1, self.bit_width**2, device=x.device, dtype=x.dtype).view(1, -1)
		self.levels.data = x_min.view(-1, 1) + (x_max.view(-1, 1) - x_min.view(-1, 1)) * indices

		steps = self.initialization_params['steps']
		lr = self.initialization_params['lr']
		loss_fn = self.initialization_params['criteria']
		if self.initialization_params['optim']=='Adam':
			optim = Adam(self.parameters(), lr=lr)
		else:
			raise RuntimeError(f'QuantizerLUT::_initialize: Can\'t use {self.initialization_params['optim']} optimizer')

		with torch.enable_grad():
			for i in range(steps):
				x_q = self(x)
				loss = loss_fn(x, x_q)
				loss.backward()
				optim.step()
				optim.zero_grad()


	def quantize(self, x):
		x_shape = x.shape
		if not self._initialized:
			self._initialize(x.detach())

		x_ = self.regroup(x)

		mask = x_.unsqueeze(2) > self.borders.unsqueeze(1)
		indices = mask.sum(dim=2)
		x_q = torch.take_along_dim(
			input=self.levels,
			indices=indices,
			dim=1
			)
	
		return x_q.reshape(x_shape)
	

	def forward(self, x):
		return self.quantize(x)
