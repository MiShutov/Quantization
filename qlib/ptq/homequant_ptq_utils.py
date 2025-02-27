import torch
from qlib.qlayers.homequant_layers import HQLinear


@torch.no_grad()
def prepare_block_for_training(block: torch.nn.Module, method_params={}):
    for module_name, module in block.named_modules():
        if isinstance(module, HQLinear):
            module.trainable = True
            if method_params.get('with_additions', False):
                module.latent_weight = torch.nn.Parameter(module.weight.clone())
            if method_params.get('reassine_params', False):
                module.reassine_params = method_params['reassine_params']
            module.metadata = {
                'new_indices_ratio': [],
                # 'weight_change_relative': [],
                # 'weight_change_absolute': [],
            }

@torch.no_grad()
def switch_trainable(block: torch.nn.Module, mode):
    for module_name, module in block.named_modules():
        if isinstance(module, HQLinear):
            module.trainable = mode


@torch.no_grad()
def prepare_block_for_inference(block: torch.nn.Module):
    for module_name, module in block.named_modules():
        if isinstance(module, HQLinear) and hasattr(module, 'latent_weight'):
            del module.latent_weight
        if isinstance(module, HQLinear) and hasattr(module, 'metadata'):
            del module.metadata
        if isinstance(module, HQLinear) and hasattr(module, 'reassine_params'):
            del module.reassine_params
        module.trainable = False
