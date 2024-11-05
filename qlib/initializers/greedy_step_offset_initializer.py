import torch

class GreedyInitializer:
	def __init__(self, criteria, n_grid_steps, n_grid_zooms):
		self.criteria = criteria
		self.n_grid_steps = n_grid_steps
		self.n_grid_zooms = n_grid_zooms
	
	def __call__(self, x_grouped, quantizer):
		x_min = x_grouped.min(axis=-1)[0].unsqueeze(-1)
		x_max = x_grouped.max(axis=-1)[0].unsqueeze(-1)		
		diaps = x_max - x_min
		
		self.init_scale = 0.5 * diaps / (2**quantizer.bit_width - 1)
		self.best_scale = self.init_scale
		quantizer.step.data = self.init_scale

		self.grid_step = 2 * self.init_scale / (self.n_grid_steps-1)
		
		if quantizer.use_offset == False:
			with torch.no_grad():
				x_q = quantizer(x_grouped)
			self.best_loss = self.criteria(x_grouped, x_q)
			
			# for _ in range(self.n_grid_zooms):
			# 	self.run_grid_search(x_grouped, quantizer)
			# quantizer.step.data = self.best_scale

			iter_params = {
				'scale' : {
					'init' : self.init_scale,
					'step' : self.grid_step,
					'best' : self.best_scale,
					'module' : quantizer.step,
				}
			}
			for _ in range(self.n_grid_zooms):
				self.run_multy_dim_search(
					x_grouped,
					quantizer, 
					iter_params)
			
			quantizer.step.data = iter_params['scale']['best']

		else:
			self.init_offset = -self.init_scale.clone() #torch.zeros_like(self.init_scale)
			self.best_offset = self.init_offset
			quantizer.offset.data = self.init_offset
			
			with torch.no_grad():
				x_q = quantizer(x_grouped)
			self.best_loss = self.criteria(x_grouped, x_q)
			
			iter_params = {
						'offset': {
							'init' : self.init_offset,
							'step' : self.grid_step.clone(),
							'best' : self.best_offset,
							'module' : quantizer.offset,
						},
						'scale' : {
							'init' : self.init_scale,
							'step' : self.grid_step.clone(),
							'best' : self.best_scale,
							'module' : quantizer.step,
						}
					}

			for _ in range(self.n_grid_zooms):
				self.run_multy_dim_search(
					x_grouped,
					quantizer, 
					iter_params)
			
			quantizer.step.data = iter_params['scale']['best'].clone()
			quantizer.offset.data = iter_params['offset']['best'].clone()
			
	def run_grid_search(self, 
						x_grouped, 
						quantizer):
		for i in range(self.n_grid_steps):
			scale = self.init_scale + self.grid_step * i
			quantizer.step.data = scale
			loss = self.criteria(x_grouped, quantizer(x_grouped))
			best_loss_mask_upd = loss<self.best_loss
			self.best_scale = torch.where(best_loss_mask_upd, scale, self.best_scale)
			self.best_loss = torch.where(best_loss_mask_upd, loss, self.best_loss)
		
		self.init_scale = self.best_scale - self.grid_step
		self.grid_step = (2 * self.grid_step) / (self.n_grid_steps-1)
		
	def run_multy_dim_search(self, x_grouped, quantizer, iter_params):
		dim = len(iter_params.keys())
		grid_iter_idxs = dim*[torch.arange(0, self.n_grid_steps)]
		for grid_idxs in torch.cartesian_prod(*grid_iter_idxs):
			if not grid_idxs.shape:
				grid_idxs = grid_idxs.unsqueeze(-1)
			
			# update params
			for param_idx, param_name in enumerate(iter_params.keys()):
				param_dict = iter_params[param_name]
				param = param_dict['init'] + grid_idxs[param_idx] * param_dict['step']
				param_dict['module'].data = param

			# calc loss
			loss = self.criteria(x_grouped, quantizer(x_grouped))
			best_loss_mask_upd = loss<self.best_loss
			
			self.best_loss = torch.where(best_loss_mask_upd, loss, self.best_loss)
			
			# reset best params
			for param_idx, param_name in enumerate(iter_params.keys()):
				param_dict = iter_params[param_name]
				param = param_dict['init'] + grid_idxs[param_idx] * param_dict['step']
				param_dict['best'] = torch.where(best_loss_mask_upd, param, param_dict['best'])

		# update inits and steps
		for param_idx, param_name in enumerate(iter_params.keys()):
			param_dict = iter_params[param_name]
			param_dict['init'] = param_dict['best'] - param_dict['step']
			param_dict['step'] = 2 * param_dict['step'] / (self.n_grid_steps-1)


		