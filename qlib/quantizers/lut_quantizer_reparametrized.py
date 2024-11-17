import torch
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

		x_ = self.regroup(x)

		diap_mask = x_.unsqueeze(2) > self.levels.unsqueeze(1)
		diap_indices = diap_mask.sum(dim=2)
		
		edges_bot = diap_indices==0
		edges_top = diap_indices==4
		
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



# class QuantizerLUT_reparametrized(nn.Module):
# 	def __init__(self,
# 			     group_size, 
# 				 bit_width, 
# 				 initialization_params,
# 				 with_additions=False):
# 		super().__init__()
# 		self.group_size = group_size
# 		self.bit_width = bit_width
# 		self.negative_clip = -2**(bit_width-1)
# 		self.positive_clip = 2**(bit_width-1) - 1
		
# 		self.levels = nn.Parameter(torch.tensor(1.0), requires_grad=True)

# 		self.initialization_params = initialization_params
# 		if self.initialization_params is None:
# 			self.initialization_params = {
# 				'optim' : 'Adam',
# 				'lr' : 1e-2,
# 				'steps' : 100,
# 				'loss_p' : 4
# 			}
# 		self._initialized = False
# 		self.with_additions = with_additions
# 		if self.with_additions:
# 			self.weight_additions=nn.Parameter(torch.zeros(1), requires_grad=True)

# 	@property
# 	@torch.no_grad()
# 	def borders(self):
# 		return (self.levels[:, 1:] + self.levels[:, :-1])/2

# 	@property
# 	def additions(self):
# 		if self.with_additions:
# 			return self.weight_additions
# 		else:
# 			return None


# 	def regroup(self, x):
# 		if self.group_size == 'tensor':
# 			return x.flatten()
	
# 		if self.group_size == 'channel':
# 			return x.flatten(start_dim=1)

# 		return x.reshape(-1, self.group_size)
	

# 	def _initialize(self, x):
# 		self._initialized = True

# 		if self.with_additions:
# 			self.weight_additions.data=torch.zeros(x.shape).to(x.device)

# 		with torch.no_grad():
# 			x_grouped = self.regroup(x)
# 			x_min = x_grouped.min(axis=-1)[0].unsqueeze(-1)
# 			x_max = x_grouped.max(axis=-1)[0].unsqueeze(-1)
# 			indices = torch.linspace(0, 1, self.bit_width**2, device=x.device).view(1, -1)
# 			self.levels.data = x_min.view(-1, 1) + (x_max.view(-1, 1) - x_min.view(-1, 1)) * indices
	
# 		optim_steps = self.initialization_params['steps']
# 		lr = self.initialization_params['lr']
# 		loss_fn = self.initialization_params['criteria']
# 		if self.initialization_params['optim']=='Adam':
# 			optim = Adam([self.levels], lr=lr)
# 		elif self.initialization_params['optim']=='SGD':
# 			optim = SGD([self.levels], lr=lr)
# 		else:
# 			raise RuntimeError(f'QuantizerLUT::_initialize: Can\'t use {self.initialization_params['optim']} optimizer')
# 		scheduler = CosineAnnealingLR(optim, T_max=optim_steps)

# 		with torch.enable_grad():
# 			# optimize in fp32:
# 			x_ = x.float()
# 			for i in range(optim_steps):
# 				x_q = self(x_)
# 				loss = loss_fn(x_, x_q)
# 				loss.backward()
# 				#torch.nn.utils.clip_grad_norm_(self.levels, max_norm=1.0)
# 				optim.step()
# 				scheduler.step()
# 				optim.zero_grad()
				

# 	def quantize(self, x):
# 		x_shape = x.shape
# 		if not self._initialized:
# 			self._initialize(x.detach())

# 		if self.additions is not None:
# 			x = x + self.additions

# 		x_ = self.regroup(x)

# 		diap_mask = x_.unsqueeze(2) > self.levels.unsqueeze(1)
# 		diap_indices = diap_mask.sum(dim=2)
		
# 		edges_bot = diap_indices==0
# 		edges_top = diap_indices==4
		
# 		levels_expand = torch.cat([self.levels[:, :1] + 1, self.levels], axis=1)

# 		l_i = torch.take_along_dim(
# 			input=levels_expand[:, :-1],
# 			indices=diap_mask[:, :, :-1].sum(dim=2),
# 			dim=1
# 			)

# 		l_i_plus_1 = torch.take_along_dim(
# 			input=levels_expand[:, 1:],
# 			indices=diap_mask[:, :, :-1].sum(dim=2),
# 			dim=1
# 			)

# 		l_delta = l_i_plus_1 - l_i

# 		x_q_scaled = (x_ - l_i)/l_delta

# 		#STE
# 		x_q = ste_round_pass(x_q_scaled)

# 		x_q = x_q * l_delta + l_i

# 		x_q = x_q * ~edges_bot + self.levels[:, :1] * edges_bot
# 		x_q = x_q * ~edges_top + self.levels[:, -1:] * edges_top

# 		x_q = x_q.reshape(x_shape)
# 		return x_q
	

# 	def forward(self, x):
# 		return self.quantize(x)



