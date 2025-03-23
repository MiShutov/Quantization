import torch

class MomentCriteria:
	def __init__(self, p, along_axis=None):
		self.p = p
		self.along_axis = along_axis
	
	def __call__(self, x, x_q):
		errors = torch.pow(torch.abs(x - x_q), self.p)
		loss = torch.mean(errors, axis=self.along_axis, keepdim=True)
		return loss

	
class LpLoss:
	def __init__(self, p):
		self.p = p
	
	def __call__(self, x, x_q):
		diff = torch.abs(x - x_q)
		errors = torch.pow(diff, self.p)
		return torch.mean(errors)
