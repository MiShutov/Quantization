import torch
from qlib.qlayers.homequant_layers import HQLinear


@torch.no_grad()
def prepare_block_for_training(block: torch.nn.Module):
    for module_name, module in block.named_modules():
        if isinstance(module, HQLinear):
            module.trainable = True
            module.latent_weight = torch.nn.Parameter(module.weight.clone())
            module.metadata = {
                'new_indices_number': [],
                'new_indices_ratio': [],
                'weight_change_relative': [],
                'weight_change_absolute': [],
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
            del module.metadata
        module.trainable = False
