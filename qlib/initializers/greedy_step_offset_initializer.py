import torch


class MomentCriteria:
	def __init__(self, p=2):
		self.p = p
	
	def __call__(self, x, x_q):
		errors = torch.pow(torch.abs(x - x_q), self.p)
		return torch.sum(errors, axis=-1, keepdim=True)


class GreedyInitializer:
	def __init__(self, criteria, grid_steps, grid_zooms):
		self.criteria = criteria
		self.grid_steps = grid_steps
		self.grid_zooms = grid_zooms
	
	def __call__(self, x_grouped, quantizer):
		x_min = x_grouped.min(axis=-1)[0].unsqueeze(-1)
		x_max = x_grouped.max(axis=-1)[0].unsqueeze(-1)		

		diaps = x_max - x_min
		self.init_steps = 0.5 * diaps / (2**quantizer.bit_width - 1)
		self.grid_step_steps = (3 * self.init_steps - self.init_steps) / self.grid_steps
		
		quantizer.step.data = 2*self.init_steps

		if quantizer.use_offset == False:
			with torch.no_grad():
				x_q = quantizer(x_grouped)
			self.best_loss = self.criteria(x_grouped, x_q)
			self.best_steps = 2*self.init_steps
			
			for _ in range(self.grid_zooms):
				self.run_grid_search(x_grouped, quantizer)
				self.init_steps = self.best_steps - self.grid_step_steps
				self.grid_step_steps = (2 * self.grid_step_steps) / self.grid_steps
			
			quantizer.step.data = self.best_steps

		else:
			self.init_steps = torch.zeros_like(quantizer.step.data)
			quantizer.offset.data = self.init_steps
			with torch.no_grad():
				x_q = quantizer(x_grouped)
			self.best_loss = self.criteria(x_grouped, x_q)
			self.best_steps = 2*self.init_steps
			self.best_offsets = 2*self.best_offsets
			
			for _ in range(self.grid_zooms):
				self.run_grid_search(x_grouped, quantizer)
				self.init_steps = self.best_steps - self.grid_step_steps
				self.grid_step_steps = (2 * self.grid_step_steps) / self.grid_steps
			
			quantizer.step.data = self.best_steps


	def run_grid_search(self, 
						x_grouped, 
						quantizer):
		for i in range(self.grid_steps):
			steps = self.init_steps + self.grid_step_steps * i
			quantizer.step.data = steps
			loss = self.criteria(x_grouped, quantizer(x_grouped))
			self.best_steps = torch.where(loss<self.best_loss, steps, self.best_steps)
			self.best_loss = torch.where(loss<self.best_loss, loss, self.best_loss)
			
	def run_multydimentional_search(self, x_grouped, quantizer):
		#for step,  torch.cartesian_prod(steps, offsets)
		pass