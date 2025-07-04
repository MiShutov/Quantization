import torch
from tqdm import tqdm

from qlib.ptq.ptq_utils import (
	optimization_step, 
    prepare_optimizers, 
    prepare_trainig_dataset
)

from qlib.ptq.homequant_ptq_utils import (
	prepare_block_for_training, 
    switch_trainable, 
    prepare_block_for_inference
)

from qlib.qlayers.homequant_layers import HQLayer
from qlib.qlayers.symquant_layers import SymHQLinear
from qlib.ptq.ptq_logger import LoggerPTQ
from qlib.utils.memmory import free_unused_memory


class HomequantTrainerPTQ():
    def __init__(
            self, 
            logdir,
            training_config,
            tokenizer,
            rotary_emb,
            store_dtype=torch.float16,
            autocast_dtype=torch.float16,
            device_map='cuda',
            quant_classes=SymHQLinear):
        self.logdir = logdir
        self.logger = LoggerPTQ(logdir=logdir)
    
        self.activation_storage = prepare_trainig_dataset(training_config['train_data'], tokenizer)
        self.optimization_config = training_config['optimization_settings']
        self.validation_settings = training_config.get('validation_settings', None)
        self.quant_classes = quant_classes
        
        self.store_dtype = store_dtype
        self.autocast_dtype = autocast_dtype
        self.device_map = device_map
        self.rotary_emb = rotary_emb.to(device_map)


    @torch.no_grad()
    def prepare_input(self, hidden_states):
        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        return {
            'hidden_states' : hidden_states,
            "position_embeddings" : position_embeddings,
        }

    def log_reassigns(self, block, step):
        names = [
            'self_attn.q_proj',
            'self_attn.k_proj',
            'self_attn.v_proj',
            'self_attn.o_proj',
            'mlp.down_proj',
            'mlp.gate_proj',
            'mlp.up_proj',
        ]

        def check_metadata(metadata, key):
            data = metadata.get(key, None)
            if data is None:
                return None
            else:
                if len(data) > 0:
                    return data[-1]
                else:
                    print('check_metadata::reassigns not logged!')
                    return 0.0

        for name in names:
            submodule_meta = block.get_submodule(name).metadata
            reassigns_data = check_metadata(submodule_meta, 'new_indices_ratio')
            if reassigns_data is not None:
                self.logger.log_scalar(
                    dir='train', 
                    scalar_name=f'new_indices_ratio ({name})', 
                    scalar=reassigns_data,
                    step=step
                )


    @torch.enable_grad()
    def train_block(
            self, 
            block,
            path_to_fp_block=None,
        ):
        # Add latent weigh and other features
        prepare_block_for_training(
            block,
            self.quant_classes,
            self.optimization_config.get('method_params', {}), 
            path_to_fp_block)
        block.train()
        
        loss_fn = self.optimization_config['loss_fn']
        n_epochs = self.optimization_config['n_epochs']
        optimizers = prepare_optimizers(
            self.optimization_config['optimizers'], block
        )

        # Init validations
        if self.validation_settings.get('val_before_trainig', False):
            self.validation(block=block, step=0)

        # Setup intermediate validation
        if self.validation_settings.get('n_intermediate_val', 0):
            val_step = n_epochs * self.activation_storage.n_train_batches//(self.validation_settings['n_intermediate_val'])
            val_step = max(1, val_step)

        
        for epoch in range(n_epochs):
            for act_idx, q_batch in enumerate(tqdm(
                    self.activation_storage.train_q,
                    desc=f"training block (epoch {epoch+1})",
                    leave=True
                )):
                step = epoch * self.activation_storage.n_train_batches + act_idx
                
                q_batch = q_batch.to(self.device_map)
                q_input = self.prepare_input(q_batch)
                fp_act = self.activation_storage.train_fp[act_idx].to(self.device_map)

                with torch.amp.autocast('cuda', dtype=self.autocast_dtype):
                    q_act = block(**q_input)[0]
                    loss = loss_fn(q_act, fp_act)

                grad_scaler = optimizers['grad_scaler']
                grad_scaler.scale(loss).backward()

                # try:
                #     self.log_reassigns(block, step)
                # except:
                #     print('reassigns not logged!')
                # codebook = block.self_attn.q_proj.codebook
                # print("codebook.grad:", codebook.grad)
                #latent_weight = block.self_attn.q_proj.latent_weight
                #print("latent_weight.grad:", latent_weight.grad)

                # Optimization step
                optimization_step(
                    optimizers=optimizers, 
                    step=step, 
                    optimization_config=self.optimization_config
                )

                # Intermediate validation
                if self.validation_settings.get('n_intermediate_val', 0):
                    if (step) and ((step+1)%val_step==0):
                        self.validation(block=block, step=step)
        
        # Remove latent weigh and other features
        prepare_block_for_inference(block, self.quant_classes)             
        free_unused_memory()


    @torch.no_grad()
    def validation(self, block, step):
        switch_trainable(block, False, self.quant_classes)
        val_loss = self.validate(block)
        self.logger.log_scalar(dir='val', scalar_name='loss', scalar=val_loss, step=step)
        switch_trainable(block, True, self.quant_classes)

    @torch.no_grad()
    def validate(self, block):
        losses = []
        for act_idx, q_batch in enumerate(self.activation_storage.val_q):
            q_batch = q_batch.to(self.device_map)
            q_input = self.prepare_input(q_batch)
            with torch.amp.autocast('cuda', dtype=self.autocast_dtype):
                q_act = block(**q_input)[0]
            fp_act = self.activation_storage.val_fp[act_idx].to(self.device_map)
            losses.append(torch.mean((q_act.to(torch.float32)-fp_act.to(torch.float32))**2))

        free_unused_memory()
        val_loss = sum(losses)/len(losses) if losses else 0
        #print('val_loss:', val_loss)
        return val_loss


    @torch.no_grad()
    def collect_block_activations(self, block, activations, with_input_preparation):
        block.eval()
        for act_idx, batch in enumerate(tqdm(
                activations, 
                desc="collecting activations",
                leave=True)
            ):
            batch = batch.to(self.device_map)
            
            with torch.amp.autocast('cuda', dtype=self.autocast_dtype):
                if with_input_preparation:
                    input = self.prepare_input(batch)
                    block_activations = block(**input)[0]
                else:
                    block_activations = block(batch)

            activations[act_idx] = block_activations.detach().cpu().to(self.store_dtype)
        free_unused_memory()


    def finetune_block_ptq(
            self,
            path_to_fp_block,
            path_to_q_block,
            path_to_q_block_trained,
            train=False,
            with_input_preparation=False,
            use_fp_act_only=False,
            ):

        #fp
        block = torch.load(path_to_fp_block, map_location=self.device_map)
        self.collect_block_activations(block, self.activation_storage.train_fp, with_input_preparation)
        self.collect_block_activations(block, self.activation_storage.val_fp, with_input_preparation)

        #q
        block = torch.load(path_to_q_block, map_location=self.device_map)
        if train:
            self.train_block(block, path_to_fp_block)
        torch.save(block, path_to_q_block_trained)

        if use_fp_act_only:
            block = torch.load(path_to_fp_block, map_location=self.device_map)
        self.collect_block_activations(block, self.activation_storage.train_q, with_input_preparation)
        self.collect_block_activations(block, self.activation_storage.val_q, with_input_preparation)
