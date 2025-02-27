import torch

class MomentCriteria:
	def __init__(self, p, along_axis=None):
		self.p = p
		self.along_axis = along_axis
	
	def __call__(self, x, x_q):
		errors = torch.pow(torch.abs(x - x_q), self.p)
		loss = torch.mean(errors, axis=self.along_axis, keepdim=True)
		return loss


class MAE:
	def __init__(self):
		self.p = 2
	
	def __call__(self, x, x_q):
		errors = torch.abs(x - x_q)
		return torch.mean(errors)

class L1:
	def __init__(self):
		self.p = 2
	
	def __call__(self, x, x_q):
		errors = torch.abs(x - x_q)
		return torch.mean(errors)

class MSE:
	def __init__(self):
		self.p = 2
	
	def __call__(self, x, x_q):
		errors = (x - x_q)**self.p
		return torch.mean(errors)
	
class L2:
	def __init__(self):
		self.p = 2
	
	def __call__(self, x, x_q):
		errors = (x - x_q)**self.p
		return torch.mean(errors)