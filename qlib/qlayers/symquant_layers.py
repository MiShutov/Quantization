import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from qlib.ptq.homequant_ptq_utils import get_reassign_ratio
from qlib.vector_quantization.nn_search.faiss_nn_search import reassign
from qlib.utils.pack_effective import unpack_bool_tensor
from qlib.utils.incoherence_preprocessing.incoherence_process_functions import incoherence_process
from qlib.utils.incoherence_preprocessing.haar_wavelet import HaarWavelet
from qlib.utils.fast_functions import batch_gathering

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
        self.register_buffer('vector_dim', torch.tensor(quant_data['codebook'].shape[1]))        
        self.codebook = nn.Embedding(
            num_embeddings=quant_data['codebook'].shape[0], 
            embedding_dim=self.vector_dim, 
            scale_grad_by_freq=True
        )
        self.codebook.weight.data = quant_data['codebook'].clone()        
        self.register_buffer('signs', quant_data['signs']['packed'])
        # if quant_data.get('SU') is not None:
        #     self.register_buffer('SU', quant_data['SU'])
        #     self.register_buffer('SV', quant_data['SV'])
        if quant_data['scales'] is not None:
            self.scales = nn.Parameter(quant_data['scales'].to(module.weight.dtype))
        else: 
            self.register_buffer('scales', None)
        
        return deepcopy(self).to(module.weight.device)
    

    @property
    def weight(self):
        w = self.codebook(self.indices.to(torch.int)).reshape(self.weight_shape)
        if self.scales is not None:
            w *= self.scales
        return w


    @torch.no_grad()
    def update_indices_qat(self):
        if self.reassine_params.get('reassign_step', None) is not None:
            if self.reassine_params.get('reassign_counter', None) is None:
                self.reassine_params.update({'reassign_counter' : -1})
            self.reassine_params['reassign_counter'] += 1
            
            if self.reassine_params['reassign_counter'] % self.reassine_params['reassign_step']!=0:
                #print('skip reassigns')
                return None
            else:
                #print('with reassigns')
                pass

        # Get vectors to reassign
        if self.scales is not None:
            vectors = (self.fp_weight / self.scales).reshape(-1, self.vector_dim)
        else:
            vectors = self.fp_weight.reshape(-1, self.vector_dim)
        
        # Compute reassigns
        new_indices = torch.tensor(
            reassign(torch.abs(vectors), self.codebook.weight.data, self.reassine_params),
            dtype=self.indices.dtype, 
            device=self.indices.device
        ).to(torch.int)
        self.indices = self.indices.to(torch.int)
        
        # Set new indices
        self.indices.data = new_indices


    def _train_qat(self, x):
        assert hasattr(self, 'fp_weight')
        self.update_indices_qat()
        w = self.weight * torch.sign(self.fp_weight)
        # w = self.weight * (2 * unpack_bool_tensor(self.signs, self.weight_shape) - 1)
        return F.linear(x, w)


    @torch.no_grad()
    def update_indices(self):
        if self.indices.dtype == torch.uint16:
            self.indices = self.indices.int()

        reassine_frac = self.reassine_params["reassine_frac"]

        # Which vectors to reassign        
        n_vectors_to_reassign = int(torch.numel(self.weight) // self.vector_dim * reassine_frac)
        vector_ids_to_reassign = torch.topk(
            torch.max((
                    torch.abs(torch.abs(self.latent_weight) - self.weight) / (torch.abs(self.weight) + 1e-10)
            ).reshape(-1, self.vector_dim), dim=1).values,
            n_vectors_to_reassign
        ).indices.to(torch.int)

        # Get vectors to reassign
        if self.scales is not None:
            vectors = (self.latent_weight / self.scales).reshape(-1, self.vector_dim)[vector_ids_to_reassign]
        else:
            vectors = self.latent_weight.reshape(-1, self.vector_dim)[vector_ids_to_reassign]
        
        # Compute reassigns
        new_indices = reassign(torch.abs(vectors), self.codebook.weight.data, self.reassine_params)
        
        # Log reassigns
        self.metadata['new_indices_ratio'].append(
            get_reassign_ratio(self.indices.data[vector_ids_to_reassign.int()], new_indices.int()) * reassine_frac
        )
        # Set new indices
        self.indices.data[vector_ids_to_reassign] = new_indices.to(self.indices.dtype)


    def _reassign_forward(self, x):
        assert hasattr(self, 'latent_weight')
        self.update_indices()
        w = self.weight * torch.sign(self.latent_weight).detach()
        return F.linear(
            x, w + self.latent_weight - self.latent_weight.detach()
        )

    def _inference_forward(self, x):
        if hasattr(self, 'signs'):
            w = self.weight * (2 * unpack_bool_tensor(self.signs, self.weight_shape) - 1)
        elif hasattr(self, 'latent_weight'):
            w = self.weight * torch.sign(self.latent_weight)
        else:
            raise

        # if hasattr(self, 'SU'):
        #     w = incoherence_process(w, self.SU, self.SV)
        return F.linear(x, w)


    def forward(self, x):
        if self.trainable == False:
            return self._inference_forward(x)
        
        else: # (self.trainable == True)
            training_features = self.train_mode.split(':')
            if 'reassines' in training_features:
                return self._reassign_forward(x)
            else:
                return self._inference_forward(x)



class HaarSymHQLinear(nn.Module):
    def __init__(self, path_to_vc_data):
        super().__init__()
        self.path_to_vc_data = path_to_vc_data
        self.weight_shape = None
        self.trainable = False


    @torch.no_grad()
    def wrap_module(self, module, module_name):
        self.weight_shape = module.weight.shape

        quant_data = torch.load(os.path.join(self.path_to_vc_data, f'{module_name}.pth'), weights_only=True)

        self.codebook = nn.Parameter(quant_data['codebook'], requires_grad=True)
        self.register_buffer('vector_dim', torch.tensor(quant_data['codebook'].shape[-1]))  
        self.register_buffer('indices', quant_data['indices'])
        self.register_buffer('signs', quant_data['signs']['packed'])
        self.signs_shape = quant_data['signs']['shape']
        
        self.haar_transform = HaarWavelet(
            level=quant_data['metadata']['haar_decomposition_level'],
            device='cuda',
            dtype=torch.float32,
        )
        
        if quant_data['scales'] is not None:
            self.scales = nn.Parameter(quant_data['scales'].to(module.weight.dtype))
        else: 
            self.register_buffer('scales', None)
        
        return deepcopy(self).to(module.weight.device)


    def _inference_forward(self, x):
        haar_freq = batch_gathering(self.codebook, self.indices.to(torch.int32)).reshape(self.signs_shape)
        haar_signs = 2 * unpack_bool_tensor(self.signs, self.signs_shape) - 1
        haar_freq *= haar_signs
        w = self.haar_transform.inverse(haar_freq)
        if self.scales is not None:
            w *= self.scales
        return F.linear(x, w)


    def forward(self, x):
        return self._inference_forward(x)    
    