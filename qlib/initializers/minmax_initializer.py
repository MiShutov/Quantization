import torch

class MinMaxInitializer:
	def __init__(self):
		pass

	@torch.no_grad()
	def __call__(self, x_grouped, quantizer):
		x_min = x_grouped.min(axis=-1)[0].unsqueeze(-1).float()
		x_max = x_grouped.max(axis=-1)[0].unsqueeze(-1).float()		
		
		if quantizer.use_offset==False:				
			negative_clip = -2**(quantizer.bit_width-1)
			positive_clip = 2**(quantizer.bit_width-1) - 1
			step = torch.where(torch.abs(x_min) > torch.abs(x_max), x_min/negative_clip, x_max/positive_clip)
			quantizer.step.data = torch.abs(step)
		else:
			pos_clip = 2**(quantizer.bit_width-1) - 1
			neg_clip = -2**(quantizer.bit_width-1)
			offset = (x_max * neg_clip - x_min * pos_clip) / (pos_clip - neg_clip)
			step = (x_max + offset) / pos_clip
			
			quantizer.step.data = torch.abs(step)
			quantizer.offset.data = offset
			

		