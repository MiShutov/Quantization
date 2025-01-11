import os
from torch.utils.tensorboard import SummaryWriter

class LoggerPTQ:
    def __init__(self, logdir):
        self.logdir = logdir
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        self.set_version_dir()
        self.writer = SummaryWriter(log_dir=self.version_dir)
        self.block_label = None
        
       
    def set_version_dir(self):
        logdir_files = os.listdir(self.logdir)
        last_v = -1
        for file in logdir_files:
            if 'verison_' in file:
                v = int(file.split('verison_')[-1])
                if v > last_v:
                    last_v = v
        new_version = last_v+1
        self.version_dir = f"{self.logdir}/verison_{new_version}"
        os.makedirs(self.version_dir, exist_ok=False)


    def log_scalar(self, mode, scalar_name, scalar, step):
        assert self.block_label
        self.writer.add_scalar(
            f"{self.block_label}/{mode}/{scalar_name}", scalar, step
        )
        self.writer.flush()


def get_layers_data(block):
    layers_data = {}
    for module_name, module in block.named_modules():
        if isinstance(module, qlib.Quantizer):
            layer_name = module_name.split(".weight_quantizer")[0]
            matrix = block.get_submodule(f"{layer_name}.module").weight.data
            vectors = module.regroup(matrix)
            layers_data.update({
                layer_name : {
                    "quantizer" : module,
                    "vectors" : vectors,
                    'prev_idxs': module.idxs.clone().detach(),
                    'new_idxs': None,
                    'reassign_mask': None
                }
            })
    return layers_data

def log_reassing(layers_data, logger, step):
    for layer_name in layers_data:      
        l_data = layers_data[layer_name]
        l_data["new_idxs"] = l_data['quantizer'].idxs.clone().detach()
        l_data['reassign_mask'] = l_data["new_idxs"]!=l_data["prev_idxs"]
        l_data["prev_idxs"] = deepcopy(l_data["new_idxs"])

        reassign_ration = (l_data['reassign_mask'].sum() / l_data['reassign_mask'].shape[0]).item()

        additions_grad = l_data['quantizer'].additions.grad
        
        vectors_grad = l_data['quantizer'].regroup(additions_grad)
        vectors_grad_scaled = vectors_grad/l_data['vectors']

        vectors_grad_norms = torch.linalg.norm(vectors_grad, axis=1, keepdim=True)
        vectors_grad_scaled_norms = torch.linalg.norm(vectors_grad_scaled, axis=1, keepdim=True)

        logger.log_scalar(mode=layer_name, scalar_name='reassign_ration', scalar=reassign_ration, step=step)
        #logger.log_scalar(mode=layer_name, scalar_name='vectors_grad_norms', scalar=vectors_grad_norms.mean(), step=step)
        #logger.log_scalar(mode=layer_name, scalar_name='reassigned_vectors_grad_norms', scalar=vectors_grad_norms[l_data['reassign_mask']].mean(), step=step)
        #logger.log_scalar(mode=layer_name, scalar_name='vectors_grad_scaled_norms', scalar=vectors_grad_scaled_norms.mean(), step=step)
        #logger.log_scalar(mode=layer_name, scalar_name='reassigned_vectors_grad_scaled_norms', scalar=vectors_grad_scaled_norms[l_data['reassign_mask']].mean(), step=step)
        