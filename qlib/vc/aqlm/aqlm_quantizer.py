import torch
import torch.nn as nn
from tqdm.auto import trange
from qlib.quantizers.quantizer import Quantizer
from typing import List, Optional, Tuple, Union
from qlib.aqlm.kmeans import find_nearest_cluster, fit_faiss_kmeans, fit_kmeans, fit_kmeans_1d


class QuantizerAQLM(Quantizer):
	def __init__(
		self,
        group_size: int,
        num_codebooks: int,
        **init_kwargs,
    ):
		super().__init__(group_size, bit_width=None)
		self.group_size = group_size
		self.num_codebooks = num_codebooks
	
	@torch.no_grad()
	def configure(self, module):
		reference_weight = module.weiht
		self.out_features, self.in_features = reference_weight.shape
		module_weight_shape = self.regroup(module.weight).shape
		self.step = nn.Parameter(torch.empty(module_weight_shape[0], 1), requires_grad=True)
		if self.use_offset:
			self.offset = nn.Parameter(torch.empty(module_weight_shape[0], 1), requires_grad=True)
		else:
			self.offset = None
		if self.with_additions:
			self.additions = nn.Parameter(torch.zeros(module.weight.shape), requires_grad=True)

	def _initialize(self, x):
		pass
	

	def quantize(self, x):
		x_shape = x.shape
		if not self._initialized:
			self._initialize(x.detach())

		if self.additions is not None:
			x = x + self.additions

		x_ = self.regroup(x)

		mask = x_.unsqueeze(2) > self.borders.unsqueeze(1)
		indices = mask.sum(dim=2)
		x_q = torch.take_along_dim(
			input=self.levels,
			indices=indices,
			dim=1
			)
		#reshape back
		x_q = x_q.reshape(x_shape)
		return x_q - x.detach() + x
	

	def forward(self, x):
		if not self._quantize:
			return x
		return self.quantize(x)



@torch.no_grad()
def init_aq_kmeans(
    reference_weight: torch.Tensor,
    *,
    num_codebooks: int,
    codebook_size: int,
	in_group_size: int,
    out_group_size: int = 1,
    verbose: bool = False,
    use_faiss: bool = False,
    max_points_per_centroid: Optional[int] = None,
    max_iter: int = 1000,
    devices: Optional[List[torch.device]] = None,
    **kwargs,
):
    """
    Create initial codes and codebooks using residual K-means clustering of weights
    :params reference_weight, num_codebooks, out_group_size, in_group_size, nbits, verbose: same as in QuantizedWeight
    :params use_faiss  whether to use faiss implementation of kmeans or pure torch
    :params max_point_per_centorid maximum data point per cluster
    :param kwargs: any additional params are forwarded to fit_kmeans
    """
    out_features, in_features = reference_weight.shape
    num_out_groups = out_features // out_group_size
    num_in_groups = in_features // in_group_size
    weight_residue = (
        reference_weight.reshape(num_out_groups, out_group_size, num_in_groups, in_group_size)
        .clone()
        .swapaxes(-3, -2)
        .reshape(num_out_groups * num_in_groups, out_group_size * in_group_size)
    )
    codebooks = []
    codes = []

    if max_points_per_centroid is not None:
        print("Clustering:", max_points_per_centroid * codebook_size, "points from", weight_residue.shape[0])

    for _ in trange(num_codebooks, desc="initializing with kmeans") if verbose else range(num_codebooks):
        if use_faiss:
            codebook_i, codes_i, reconstructed_weight_i = fit_faiss_kmeans(
                weight_residue,
                k=codebook_size,
                max_iter=max_iter,
                gpu=(weight_residue.device.type == "cuda"),
                max_points_per_centroid=max_points_per_centroid,
            )
        else:
            chosen_ids = None
            if max_points_per_centroid is not None:
                chosen_ids = torch.randperm(weight_residue.shape[0], device=weight_residue.device)[
                    : max_points_per_centroid * codebook_size
                ]
            codebook_i, _, _ = fit_kmeans(
                weight_residue if chosen_ids is None else weight_residue[chosen_ids, :],
                k=codebook_size,
                max_iter=max_iter,
                devices=devices,
                **kwargs,
            )
            codes_i, reconstructed_weight_i = find_nearest_cluster(weight_residue, codebook_i, devices=devices)

        codes_i = codes_i.reshape(num_out_groups, num_in_groups, 1)
        codebook_i = codebook_i.reshape(1, codebook_size, out_group_size, in_group_size)
        weight_residue -= reconstructed_weight_i
        codes.append(codes_i)
        codebooks.append(codebook_i)
        del reconstructed_weight_i
    codebooks = torch.cat(codebooks, dim=0)
    codes = torch.cat(codes, dim=-1)
    return codes, codebooks
