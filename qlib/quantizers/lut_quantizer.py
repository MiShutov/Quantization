import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from qlib.quantizers.quantizer import Quantizer


class QuantizerLUT(Quantizer):
	def __init__(self,
			     group_size, 
				 bit_width, 
				 initialization_params,
				 with_additions=False):
		super().__init__(group_size, bit_width)
		self.initialization_params = initialization_params
		self.with_additions = with_additions

		self.levels = nn.Parameter(torch.tensor(1.0), requires_grad=True)
		if self.with_additions:
			self.weight_additions=nn.Parameter(torch.zeros(1), requires_grad=True)
		
		if self.initialization_params is None:
			self.initialization_params = {
				'optim' : 'Adam',
				'lr' : 3e-3,
				'steps' : 250,
				'loss_p' : 4
			}

	
	@property
	@torch.no_grad()
	def borders(self):
		return (self.levels[:, 1:] + self.levels[:, :-1])/2

	@property
	def additions(self):
		if self.with_additions:
			return self.weight_additions
		else:
			return None
	

	def _initialize(self, x):
		self._initialized.data = torch.tensor(True)

		if self.with_additions:
			self.weight_additions.data=torch.zeros(x.shape).to(x.device)

		with torch.no_grad():
			x_grouped = self.regroup(x)
			x_min = x_grouped.min(axis=-1)[0].unsqueeze(-1)
			x_max = x_grouped.max(axis=-1)[0].unsqueeze(-1)
			#indices = torch.linspace(0, 1, self.bit_width**2, device=x.device, dtype=torch.float).view(1, -1)
			indices = torch.linspace(0, 1, self.bit_width**2, device=x.device).view(1, -1)
			self.levels.data = x_min.view(-1, 1) + (x_max.view(-1, 1) - x_min.view(-1, 1)) * indices
	
		optim_steps = self.initialization_params['steps']
		lr = self.initialization_params['lr']
		loss_fn = self.initialization_params['criteria']
		if self.initialization_params['optim']=='Adam':
			optim = Adam([self.levels], lr=lr)
		elif self.initialization_params['optim']=='SGD':
			optim = SGD([self.levels], lr=lr)
		else:
			raise RuntimeError(f"QuantizerLUT::_initialize: Can\'t use {self.initialization_params['optim']} optimizer")
		scheduler = CosineAnnealingLR(optim, T_max=optim_steps//2)

		with torch.enable_grad():
			# optimize in fp32:
			x_ = x.float()
			for i in range(optim_steps):
				x_q = self(x_)
				loss = loss_fn(x_, x_q)
				loss.backward()
				if self.initialization_params.get('grad_norm', False):
					torch.nn.utils.clip_grad_norm_(self.levels, max_norm=1.0)
				optim.step()
				scheduler.step()
				optim.zero_grad()
				

	def quantize(self, x):
		x_shape = x.shape
		if not self._initialized:
			self._initialize(x.detach())

		if self.additions is not None:
			x = x + self.additions

		x_ = self.regroup(x)

		mask = x_.unsqueeze(2) > self.borders.unsqueeze(1)
		indices = mask.sum(dim=2)
		x_q = torch.take_along_dim(
			input=self.levels,
			indices=indices,
			dim=1
			)
		#reshape back
		x_q = x_q.reshape(x_shape)
		return x_q - x.detach() + x



