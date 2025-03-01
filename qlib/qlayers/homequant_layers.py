import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from qlib.ptq.homequant_ptq_utils import get_reassign_ratio
from qlib.vector_quantization.nn_search.faiss_nn_search import reassign
from qlib.utils.pack_effective import unpack_bool_tensor


class HQLayer(nn.Module):
    def __init__(self, path_to_vc_data):
        super().__init__()
        self.path_to_vc_data = path_to_vc_data
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
            self.register_buffer('scales', None)
        return deepcopy(self).to(module.weight.device)

    @property
    def weight(self):
        w = self.codebook[self.indices.to(torch.int)].reshape(self.weight_shape)
        if self.scales is not None:
            w *= self.scales
        return w

    def forward(self, x):
        pass


class HQLinear(HQLayer):
    def __init__(self, path_to_vc_data):
        super().__init__(path_to_vc_data)


    def _inference_forward(self, x):
        w = self.weight
        return F.linear(x, w)


    def _train_forward(self, x):
        vector_dim = self.codebook.shape[1]
        n_vectors_to_reassign = int(torch.numel(self.weight) // vector_dim * self.reassine_params.get("reassine_frac", 1.0))
        if n_vectors_to_reassign > 0:
            with torch.no_grad():
                assert hasattr(self, 'latent_weight')
                vector_ids_to_reassign = torch.topk(
                    torch.max((torch.abs(self.latent_weight - self.weight) / (torch.abs(self.weight) + 1e-10)).reshape(-1, vector_dim), dim=1).values,
                    #torch.linalg.norm((torch.abs(self.latent_weight - self.weight) / (torch.abs(self.weight) + 1e-10)).reshape(-1, vector_dim), axis=1), 
                    n_vectors_to_reassign
                ).indices
                
                if self.scales is not None:
                    vectors_to_reassign = (self.latent_weight / self.scales).reshape(-1, vector_dim)[vector_ids_to_reassign]
                else:
                    vectors_to_reassign = self.latent_weight.reshape(-1, self.codebook.shape[1])[vector_ids_to_reassign]
                    
                new_indices = torch.tensor(
                    reassign(vectors_to_reassign.cpu(), self.codebook.cpu(), self.reassine_params),
                    dtype=self.indices.dtype, 
                    device=self.indices.device
                )

                self.metadata['new_indices_ratio'].append(
                    get_reassign_ratio(self.indices.data[vector_ids_to_reassign], new_indices) * self.reassine_params.get("reassine_frac", 1.0)
                )
                self.indices.data[vector_ids_to_reassign] = new_indices

            return F.linear(
                x, 
                self.weight + self.latent_weight - self.latent_weight.detach()
            )
        return F.linear(x, self.weight)

    def forward(self, x):
        if self.trainable:
            return self._train_forward(x)
        else:
            return self._inference_forward(x)

    @torch.no_grad()
    def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            prepared_dict = {}
            for k in state_dict:
                prepared_dict.update({k.split(".")[-1]: state_dict[k]})
            
            self.indices.data.copy_(prepared_dict["indices"])
            self.codebook.data.copy_(prepared_dict["codebook"])

            if self.scales is None and prepared_dict["scales"] is not None:
                self.scales = nn.Parameter(prepared_dict["scales"])
            elif self.scales is not None and prepared_dict["scales"] is not None:
                self.scales.data.copy_(prepared_dict["scales"])
            elif self.scales is not None and prepared_dict["scales"] is None:
                self.scales = None                
                

class SymHQLinear(nn.Module):
    def __init__(self, path_to_vc_data):
        super().__init__()
        self.path_to_vc_data = path_to_vc_data
        self.weight_shape = None
        self.trainable = False

    @torch.no_grad()
    def wrap_module(self, module, module_name):
        self.weight_shape = module.weight.shape

        quant_data = torch.load(os.path.join(self.path_to_vc_data, f'{module_name}.pth'), weights_only=True)
        self.register_buffer('indices', quant_data['indices'])
        self.codebook = nn.Parameter(quant_data['codebook'].to(module.weight.dtype)) # shape: [codebook_size, vector_size]
        self.register_buffer('signs', quant_data['signs']['packed'])

        if quant_data['scales'] is not None:
            self.scales = nn.Parameter(quant_data['scales'].to(module.weight.dtype)) # shape: [out_channels, 1]
        else: 
            self.register_buffer('scales', None)
        
        return deepcopy(self).to(module.weight.device)
    
    @property
    def weight(self):
        if self.codebook.ndim == 2:
            w = self.codebook[self.indices.to(torch.int)].reshape(self.weight_shape)
        
        elif self.codebook.ndim == 3:
            n_blocks = self.indices.shape[0]
            all_vectors = []
            for i_block in range(n_blocks):
                indices_part = self.indices[i_block]
                codebook_part = self.codebook[i_block]
                all_vectors.append(codebook_part[indices_part.to(torch.int)])
            w = torch.stack(all_vectors).reshape(self.weight_shape)
        
        if self.scales is not None:
            w *= self.scales
        return w

    def _inference_forward(self, x):
        w = self.weight * (2 * unpack_bool_tensor(self.signs, self.weight_shape) - 1)
        return F.linear(x, w)


    def _train_forward(self, x):
        assert hasattr(self, 'latent_weight')

        vector_dim = self.codebook.shape[-1]
        n_vectors_to_reassign = int(torch.numel(self.weight) // vector_dim * self.reassine_params.get("reassine_frac", 1.0))
        if n_vectors_to_reassign > 0:
            with torch.no_grad():
                vector_ids_to_reassign = torch.topk(
                    torch.max((torch.abs(self.latent_weight - self.weight) / (torch.abs(self.weight) + 1e-10)).reshape(-1, vector_dim), dim=1).values,
                    #torch.linalg.norm((torch.abs(self.latent_weight - self.weight) / (torch.abs(self.weight) + 1e-10)).reshape(-1, vector_dim), axis=1), 
                    n_vectors_to_reassign
                ).indices
                
                if self.scales is not None:
                    vectors_to_reassign = (self.latent_weight / self.scales).reshape(-1, vector_dim)[vector_ids_to_reassign]
                else:
                    vectors_to_reassign = self.latent_weight.reshape(-1, self.codebook.shape[1])[vector_ids_to_reassign]
                    
                new_indices = torch.tensor(
                    reassign(vectors_to_reassign.cpu(), self.codebook.cpu(), self.reassine_params),
                    dtype=self.indices.dtype, 
                    device=self.indices.device
                )

                self.metadata['new_indices_ratio'].append(
                    get_reassign_ratio(self.indices.data[vector_ids_to_reassign], new_indices) * self.reassine_params.get("reassine_frac", 1.0)
                )
                self.indices.data[vector_ids_to_reassign] = new_indices

            return F.linear(
                x, 
                self.weight + self.latent_weight - self.latent_weight.detach()
            )
        return F.linear(x, self.weight)

    def forward(self, x):
        if self.trainable:
            return self._train_forward(x)
        else:
            return self._inference_forward(x)