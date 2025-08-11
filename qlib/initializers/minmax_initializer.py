import torch

class MinMaxInitializer:
	def __init__(self):
		pass

	@torch.no_grad()
	def __call__(self, x, quantizer):
		x_grouped = x.reshape(-1, quantizer.group_size)

		x_min = x_grouped.min(axis=-1)[0].unsqueeze(-1).float()
		x_max = x_grouped.max(axis=-1)[0].unsqueeze(-1).float()		
		
		if quantizer.use_offset == False:				
			scale = torch.where(
				torch.abs(x_min) > torch.abs(x_max), 
				x_min / quantizer.negative_clip, 
				x_max / quantizer.positive_clip
			)
			scale = torch.abs(scale)
			scale = scale.reshape(x.shape[0], x.shape[1] // quantizer.group_size)
			return scale, None
		else:
			offset = (x_max * quantizer.negative_clip - x_min * quantizer.positive_clip) / (quantizer.positive_clip - quantizer.negative_clip)
			scale = (x_max + offset) / quantizer.positive_clip
			scale = torch.abs(scale)
			
			scale = scale.reshape(x.shape[0], x.shape[1] // quantizer.group_size)
			offset = offset.reshape(x.shape[0], x.shape[1] // quantizer.group_size)

			return scale.contiguous(), offset.contiguous()
			

		