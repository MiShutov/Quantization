import torch
from qlib.ptq.ptq_utils import switch_quantizers, switch_reassings, prepare_optimizers
import psutil
import gc
import ctypes
import qlib

# @torch.no_grad()
# def reassing(block):
#     for _, module in block.named_modules():
#         if isinstance(module, qlib.QLinear) and hasattr(module, 'weight_quantizer'):
#             if isinstance(module.weight_quantizer, qlib.VectorQuantizer):
#                 weight = module.module.weight
#                 module.weight_quantizer.reassign(weight)

def free_unused_memory():
    torch.cuda.empty_cache()
    gc.collect()
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    libc.malloc_trim(ctypes.c_int(0))


def print_mem(name='mem', verbose=True):
    if verbose:
        process = psutil.Process()
        memory_info = process.memory_info()
        print(name, memory_info.rss / (1024 ** 3), 'Gb')


class TrainerPTQ():
    def __init__(
        self, 
        optimization_config,
        position_embeddings,
        store_dtype=torch.float16,
        device_map='cuda',
        verbose=False,
        validation_settings={
            'before_trainig' : False,
            'intermediate' : 0,
            }
        ):
        self.store_dtype = store_dtype
        self.device_map = device_map
        self.optimization_config = optimization_config
        self.position_embeddings = position_embeddings
        self.verbose = verbose
        self.validation_settings = validation_settings


    @torch.no_grad()
    def prepare_input(self, hidden_states):
        return {
            'hidden_states' : hidden_states,
            "position_embeddings" : self.position_embeddings,
        }


    @torch.no_grad()
    def collect_block_activations(self, block, activations, with_input_preparation):
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
    def train_block(self, block, activation_storage, with_reassings=True):
        block.train()

        switch_quantizers(block, 'q')
        if with_reassings:
            switch_reassings(block, 'on')
        
        loss_fn = self.optimization_config['loss_fn']
        n_epochs = self.optimization_config['n_epochs']
        optimizers = prepare_optimizers(
            self.optimization_config['optimizers'], block
        )
        # init validations
        if self.validation_settings.get('before_trainig', False):
            switch_reassings(block, 'off')
            val_loss = self.validate(block, activation_storage)
            if with_reassings:
                switch_reassings(block, 'on')
            print(f"Init val loss: {val_loss.item():.3e}")

        # setup intermediate validation
        if self.validation_settings.get('intermediate', 0):
            val_step = n_epochs * activation_storage.n_train_batches//(self.validation_settings['intermediate'])
            val_step = max(1, val_step)

        for epoch in range(n_epochs):
            for act_idx, q_batch in enumerate(activation_storage.train_q):
                q_batch = q_batch.to(self.device_map)
                q_input = self.prepare_input(q_batch)
                q_act = block(**q_input)[0]
                fp_act = activation_storage.train_fp[act_idx].to(self.device_map)
                
                loss = loss_fn(q_act, fp_act)
                loss.backward()
                for optimizer_name in optimizers:
                    optim = optimizers[optimizer_name]['optimizer']
                    scheduler = optimizers[optimizer_name]['scheduler']
                    optim.step()
                    if scheduler is not None:
                        scheduler.step()
                    optim.zero_grad()

                # intermediate validations
                if self.validation_settings.get('intermediate', 0):
                    step = epoch * activation_storage.n_train_batches + act_idx
                    if (step) and ((step+1)%val_step==0):
                        switch_reassings(block, 'off')
                        val_loss = self.validate(block, activation_storage)
                        if with_reassings:
                            switch_reassings(block, 'on')
                        print(f"Step {step}: {val_loss.item():.3e}")

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
