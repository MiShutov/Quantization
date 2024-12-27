import torch
import tqdm

import qlib
from qlib.ptq.ptq_utils import *
from qlib.ptq.ptq_logger import LoggerPTQ

from prettytable import PrettyTable
from copy import deepcopy


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

def process_after_backward(layers_data):
    table = PrettyTable()
    table.field_names = ["layer", 
                         "reassign_ration", 
                         "vectors_grad_norms", 
                         "reassigned_vectors_grad_norms", 
                         "vectors_grad_scaled_norms",
                         "reassigned_vectors_grad_scaled_norms",
                         ]

    for layer_name in layers_data:      
        l_data = layers_data[layer_name]
        l_data["new_idxs"] = l_data['quantizer'].idxs.clone().detach()
        l_data['reassign_mask'] = l_data["new_idxs"]!=l_data["prev_idxs"]

        reassign_ration = (l_data['reassign_mask'].sum() / l_data['reassign_mask'].shape[0]).item()

        additions_grad = l_data['quantizer'].additions.grad
        
        vectors_grad = l_data['quantizer'].regroup(additions_grad)
        vectors_grad_scaled = vectors_grad/l_data['vectors']

        vectors_grad_norms = torch.linalg.norm(vectors_grad, axis=1, keepdim=True)
        vectors_grad_scaled_norms = torch.linalg.norm(vectors_grad_scaled, axis=1, keepdim=True)

        table.add_row([
            layer_name,
            f'{reassign_ration:.2e}',
            f'{vectors_grad_norms.mean().item():.2e}',
            f'{vectors_grad_norms[l_data['reassign_mask']].mean().item():.2e}',
            f'{vectors_grad_scaled_norms.mean().item():.2e}',
            f'{vectors_grad_scaled_norms[l_data['reassign_mask']].mean().item():.2e}'
            ])

        l_data["prev_idxs"] = deepcopy(l_data["new_idxs"])
    return table

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
        


class TrainerPTQ():
    def __init__(
            self, 
            logdir,
            optimization_config,
            rotary_emb,
            store_dtype=torch.float16,
            device_map='cuda',
            verbose=False,
            validation_settings={
                'val_before_trainig' : False,
                'n_intermediate_val' : 0,
                }
            ):
        self.logger = LoggerPTQ(logdir=logdir)
        self.store_dtype = store_dtype
        self.device_map = device_map
        self.optimization_config = optimization_config
        self.rotary_emb = rotary_emb
        self.verbose = verbose
        self.validation_settings = validation_settings


    @torch.no_grad()
    def prepare_input(self, hidden_states):
        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        return {
            'hidden_states' : hidden_states,
            "position_embeddings" : position_embeddings,
        }


    @torch.no_grad()
    def collect_block_activations(self, block, activations, with_input_preparation):
        print("collecting block activations...")
        switch_reassings(block, 'off')
        block.eval()
        for act_idx, batch in enumerate(activations):
            batch = batch.to(self.device_map)
            if with_input_preparation:
                input = self.prepare_input(batch)
                block_activations = block(**input)[0]
            else:
                block_activations = block(batch)
            activations[act_idx] = block_activations.detach().cpu().to(self.store_dtype)
        free_unused_memory()


    @torch.enable_grad()
    def train_block(
            self, 
            block, 
            activation_storage,
            with_reassings=True
        ):
        print("train block...")
        block.train()
        switch_quantizers(block, 'q')
        
        loss_fn = self.optimization_config['loss_fn']
        n_epochs = self.optimization_config['n_epochs']
        optimizers = prepare_optimizers(
            self.optimization_config['optimizers'], block
        )

        # init validations
        if self.validation_settings.get('val_before_trainig', False):
            switch_reassings(block, 'off')
            val_loss = self.validate(block, activation_storage)
            self.logger.log_scalar(mode='val', scalar_name='loss', scalar=val_loss, step=0)
            
        # setup intermediate validation
        if self.validation_settings.get('n_intermediate_val', 0):
            val_step = n_epochs * activation_storage.n_train_batches//(self.validation_settings['n_intermediate_val'])
            val_step = max(1, val_step)

        # training cycle
        if with_reassings:
            switch_reassings(block, 'on')
        for epoch in tqdm.trange(n_epochs):
            for act_idx, q_batch in enumerate(activation_storage.train_q):
                step = epoch * activation_storage.n_train_batches + act_idx

                #layers_data = get_layers_data(block)
                
                q_batch = q_batch.to(self.device_map)
                q_input = self.prepare_input(q_batch)
                q_act = block(**q_input)[0]
                fp_act = activation_storage.train_fp[act_idx].to(self.device_map)
                
                loss = loss_fn(q_act, fp_act)
                loss.backward()

                additions = block.self_attn.q_proj.weight_quantizer.additions
                #print("additions:", additions.dtype, additions.grad.dtype)

                #table = process_after_backward(layers_data)
                #print(table)
                #log_reassing(layers_data=layers_data, logger=self.logger, step=step)


                for optimizer_name in optimizers:
                    optim = optimizers[optimizer_name]['optimizer']
                    scheduler = optimizers[optimizer_name]['scheduler']
                    optim.step()
                    if scheduler is not None:
                        scheduler.step()
                    optim.zero_grad()

                # intermediate validation
                if self.validation_settings.get('n_intermediate_val', 0):
                    if (step) and ((step+1)%val_step==0):
                        switch_reassings(block, 'off')
                        val_loss = self.validate(block, activation_storage)
                        if with_reassings:
                            switch_reassings(block, 'on')
                        self.logger.log_scalar(mode='val', scalar_name='loss', scalar=val_loss, step=step)
                        
        free_unused_memory()
        switch_reassings(block, 'off')


    @torch.no_grad()
    def validate(self, block, activation_storage):
        switch_quantizers(block, 'q')
        losses = []
        for act_idx, q_batch in enumerate(activation_storage.val_q):
            q_batch = q_batch.to(self.device_map)
            q_input = self.prepare_input(q_batch)
            q_act = block(**q_input)[0]
            fp_act = activation_storage.val_fp[act_idx].to(self.device_map)
            losses.append(torch.mean((q_act-fp_act)**2))

        free_unused_memory()
        return sum(losses)/len(losses)


    def finetune_block_ptq(
            self,
            block,
            activation_storage,
            collect_fp=True,
            collect_q=True,
            train=False,
            with_input_preparation=False,
            with_reassings=True,
            ):

        block = block.to(self.device_map)

        if collect_fp:
            switch_quantizers(block, 'fp')
            self.collect_block_activations(block, activation_storage.train_fp, with_input_preparation)
            self.collect_block_activations(block, activation_storage.val_fp, with_input_preparation)
            switch_quantizers(block, 'q')
            
            print_mem(name="after collect_fp", verbose=self.verbose)

        if train:
            self.train_block(block, activation_storage, with_reassings=with_reassings)
            print_mem(name="after train_block", verbose=self.verbose)

        if collect_q:
            switch_quantizers(block, 'q')
            self.collect_block_activations(block, activation_storage.train_q, with_input_preparation)
            self.collect_block_activations(block, activation_storage.val_q, with_input_preparation)
            
            print_mem(name="after collect_q", verbose=self.verbose)
        
        block.cpu()
        print_mem(name="after block.cpu()", verbose=self.verbose)
        
        free_unused_memory()
