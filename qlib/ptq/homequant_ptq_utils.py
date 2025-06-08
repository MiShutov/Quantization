import torch
from qlib.utils.pack_effective import pack_bool_tensor


@torch.no_grad()
def prepare_block_for_training(
    block: torch.nn.Module,
    quant_classes: list, 
    method_params={},
    path_to_fp_block=None):

    # reassigns
    reassine_params = method_params.get('reassine_params', None)
    if reassine_params is not None:
        reassine_frac = reassine_params.get('reassine_frac', 0.0)
        if reassine_frac > 0.0:
            assert path_to_fp_block is not None
            fp_block = torch.load(path_to_fp_block, map_location='cuda')
        else:
            reassine_params = None

    # configure each module
    for module_name, module in block.named_modules():
        if (module.__class__ in quant_classes):
            module.trainable = True
            module.train_mode = 'train'
            module.metadata = {}

            # reassigns
            if reassine_params is not None:
                module.reassine_params = reassine_params
                #module.latent_weight = torch.nn.Parameter(fp_block.get_submodule(module_name).weight.clone().to(torch.float32))
                module.latent_weight = torch.nn.Parameter(torch.zeros(module.weight_shape).to(fp_block.get_submodule(module_name).weight.device))
                module.step_counter = 0
                if hasattr(module, 'signs'): # for SymHQLinear
                    del module.signs

                module.metadata.update({'new_indices_ratio': [],})
                module.train_mode += ':reassines'


@torch.no_grad()
def switch_trainable(block: torch.nn.Module, trainable_flag: bool, quant_classes: list):
    for module_name, module in block.named_modules():
        if (module.__class__ in quant_classes):
            module.trainable = trainable_flag


@torch.no_grad()
def prepare_block_for_inference(block: torch.nn.Module, quant_classes: list):
    for module_name, module in block.named_modules():
        if (module.__class__ in quant_classes):
            #del module.train_mode
            if hasattr(module, 'metadata'):
                del module.metadata
            if hasattr(module, 'reassine_params'):
                packed_signs, _ = pack_bool_tensor((1+torch.sign(module.latent_weight)).bool())
                module.register_buffer('signs', packed_signs)
                del module.latent_weight
                del module.reassine_params    
                del module.step_counter                
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
