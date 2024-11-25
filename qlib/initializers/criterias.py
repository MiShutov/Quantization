import torch

class MomentCriteria:
	def __init__(self, p, sum_along_axis=None):
		self.p = p
		self.sum_along_axis = sum_along_axis
	
	def __call__(self, x, x_q):
		errors = torch.pow(torch.abs(x - x_q), self.p)
		loss = torch.sum(errors, axis=self.sum_along_axis, keepdim=True)
		return loss