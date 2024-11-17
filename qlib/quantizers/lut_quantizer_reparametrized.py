import torch
from qlib.utils.ste import ste_round_pass
from qlib.quantizers.lut_quantizer import QuantizerLUT


class QuantizerLUT_reparametrized(QuantizerLUT):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)


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
