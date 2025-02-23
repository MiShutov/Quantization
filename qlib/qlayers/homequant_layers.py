import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from qlib.vector_quantization.nn_search.faiss_nn_search import reassign


class HQLayer(nn.Module):
    def __init__(self, path_to_vc_data):
        super().__init__()
        self.path_to_vc_data = path_to_vc_data
        self.indices = None
        self.codebook = None
        self.scales = None
        self.weight_shape = None
        self.trainable = False


    @torch.no_grad()
    def wrap_module(self, module, module_name):
        self.weight_shape = module.weight.shape # [out_channels, in_channels]

        quant_data = torch.load(os.path.join(self.path_to_vc_data, f'{module_name}.pth'), weights_only=True)
        self.indices = nn.Parameter(quant_data['indices'].to(torch.int16), requires_grad=False) # shape: [n_indices,] = [out_channels * in_channels // vector_size]
        self.codebook = nn.Parameter(quant_data['codebook'].to(module.weight.dtype)) # shape: [codebook_size, vector_size]
        
        if quant_data['scales'] is not None:
            self.scales = nn.Parameter(quant_data['scales'].to(module.weight.dtype)) # shape: [out_channels, 1]
        else: 
            self.scales = None
        return deepcopy(self).to(module.weight.device)

    @property
    def weight(self):
        w = self.codebook[self.indices.to(torch.int)].reshape(self.weight_shape)
        if self.scales is not None:
            w *= self.scales
        return w

    def forward(self, x):
        pass

@torch.no_grad()
def get_new_indices_number(prev_indices, new_indices):
    return (prev_indices != new_indices).sum().item()

@torch.no_grad()
def get_reassign_ratio(prev_indices, new_indices):
    return ((prev_indices != new_indices).sum() / torch.numel(prev_indices)).item()

@torch.no_grad()
def get_weight_change_absolute(latent_weight, weight):
    abs_diff = torch.abs(latent_weight.detach() - weight.detach()).cpu()
    return torch.mean(abs_diff).item()

@torch.no_grad()
def get_weight_change_relative(latent_weight, weight):
    abs_diff = torch.abs(latent_weight.detach() - weight.detach()).cpu()
    norm = torch.abs(weight) + 1e-10
    return torch.mean(abs_diff / norm).item()

class HQLinear(HQLayer):
    def __init__(self, path_to_vc_data):
        super().__init__(path_to_vc_data)

    def _inference_forward(self, x):
        w = self.weight
        return F.linear(x, w)


    def _train_forward(self, x):
        assert hasattr(self, 'latent_weight')

        self.metadata['weight_change_relative'].append(
            get_weight_change_relative(self.latent_weight, self.weight)
        )
        self.metadata['weight_change_absolute'].append(
            get_weight_change_absolute(self.latent_weight, self.weight)
        )

        if self.scales is not None:
            vectors = (self.latent_weight / self.scales).reshape(-1, self.codebook.shape[1])
        else:
            vectors = self.latent_weight.reshape(-1, self.codebook.shape[1])
            
        indices = torch.tensor(
            reassign(vectors.cpu(), self.codebook.cpu()),
            dtype=self.indices.dtype, 
            device=self.indices.device
        )

        self.metadata['new_indices_ratio'].append(
            get_reassign_ratio(self.indices.data, indices)
        )
        self.metadata['new_indices_number'].append(
            get_new_indices_number(self.indices.data, indices)
        )
        with torch.no_grad():
            self.indices.data.copy_(indices)

        return F.linear(
            x, 
            self.weight + self.latent_weight - self.latent_weight.detach()
        )

    def forward(self, x):
        if self.trainable:
            return self._train_forward(x)
        else:
            return self._inference_forward(x)



# class VQLinear(nn.Module):
#     def __init__(self, weight_shape, indices, codebook, scales):
#         super().__init__()
#         self.weight_shape = weight_shape
#         self.register_buffer('indices', indices.reshape(weight_shape[0], -1))  # [O, K]
#         self.register_buffer('codebook', codebook)
#         self.register_buffer('scales', scales.unsqueeze(-1))  # [O, 1]
#         self.vector_size = codebook.size(-1)

#     def forward(self, x):
#         B, S, H = x.shape
#         O, _ = self.weight_shape
#         K = H // self.vector_size
        
#         x_reshaped = x.view(B, S, K, self.vector_size)  # [B, S, K, V]
#         codebook_vectors = self.codebook[self.indices.to(torch.int)]  # [O, K, V]
#         scaled_codebook = codebook_vectors * self.scales  # [O, K, V]
        
#         return torch.einsum('bskv,okv->bso', x_reshaped, scaled_codebook)