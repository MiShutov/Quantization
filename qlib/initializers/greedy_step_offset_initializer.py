import torch

class GreedyInitializer:
	def __init__(self, criteria, n_grid_steps, n_grid_zooms):
		self.criteria = criteria
		self.n_grid_steps = n_grid_steps
		self.n_grid_zooms = n_grid_zooms
	
	@torch.no_grad()
	def __call__(self, x_grouped, quantizer):
		x_min = x_grouped.min(axis=-1)[0].unsqueeze(-1).float()
		x_max = x_grouped.max(axis=-1)[0].unsqueeze(-1).float()		
		diaps = (x_max - x_min)
		
		init_scale = 0.1 * diaps / (2**quantizer.bit_width - 1)
		quantizer.step.data = init_scale

		init_scale_grid_step = 9 * init_scale / (self.n_grid_steps-1)
		
		if quantizer.use_offset == False:
			x_q = quantizer(x_grouped)
			self.best_loss = self.criteria(x_grouped, x_q)

			iter_params = {
				'scale' : {
					'init' : init_scale,
					'step' : init_scale_grid_step,
					'best' : init_scale,
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
			init_offset = x_min
			quantizer.offset.data = init_offset
			init_offset_grid_step = diaps / (self.n_grid_steps-1)

			x_q = quantizer(x_grouped)
			self.best_loss = self.criteria(x_grouped, x_q)
			
			iter_params = {
						'scale' : {
							'init' : init_scale,
							'step' : init_scale_grid_step,
							'best' : init_scale,
							'module' : quantizer.step,
						},
						'offset': {
							'init' : init_offset,
							'step' : init_offset_grid_step,
							'best' : init_offset,
							'module' : quantizer.offset,
						},
					}

			for _ in range(self.n_grid_zooms):
				self.run_multy_dim_search(
					x_grouped,
					quantizer, 
					iter_params)

			quantizer.step.data = iter_params['scale']['best'].clone()
			quantizer.offset.data = iter_params['offset']['best'].clone()
			
		
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


		