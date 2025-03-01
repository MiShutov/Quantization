import torch


@torch.no_grad()
def prepare_block_for_training(
    block: torch.nn.Module,
    wrap_classes: list, 
    method_params={},
    path_to_fp_block=None):

    if method_params.get('use_latent_weight'):
        assert path_to_fp_block is not None
        fp_block = torch.load(path_to_fp_block, map_location='cuda')

    for module_name, module in block.named_modules():
        if (module.__class__ in wrap_classes):
            module.trainable = True
            if method_params.get('use_latent_weight', False):
                #module.latent_weight = torch.nn.Parameter(module.weight.clone())
                module.latent_weight = torch.nn.Parameter(fp_block.get_submodule(module_name).weight.clone().to(torch.float32))


            if method_params.get('reassine_params', False):
                module.reassine_params = method_params['reassine_params']
            module.metadata = {
                'new_indices_ratio': [],
                # 'weight_change_relative': [],
                # 'weight_change_absolute': [],
            }

    if method_params.get('with_additions'):
        del fp_block


@torch.no_grad()
def switch_trainable(block: torch.nn.Module, mode, wrap_classes: list):
    for module_name, module in block.named_modules():
        if (module.__class__ in wrap_classes):
            module.trainable = mode


@torch.no_grad()
def prepare_block_for_inference(block: torch.nn.Module, wrap_classes: list):
    for module_name, module in block.named_modules():
        if (module.__class__ in wrap_classes) and hasattr(module, 'latent_weight'):
            del module.latent_weight
        if (module.__class__ in wrap_classes) and hasattr(module, 'metadata'):
            del module.metadata
        if (module.__class__ in wrap_classes) and hasattr(module, 'reassine_params'):
            del module.reassine_params
        module.trainable = False

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
    norm = torch.abs(weight.detach().cpu()) + 1e-10
    return torch.mean(abs_diff / norm).item()