import torch
import tqdm
import qlib
from qlib.ptq.ptq_utils import *
from qlib.ptq.ptq_logger import LoggerPTQ
from qlib.utils.memmory import free_unused_memory

class TrainerPTQ():
    def __init__(
            self, 
            logdir,
            training_config,
            tokenizer,
            rotary_emb,
            store_dtype=torch.float16,
            device_map='cuda',
            verbose=False):
        self.logdir = logdir
        self.logger = LoggerPTQ(logdir=logdir)
    
        self.activation_storage = prepare_trainig_dataset(training_config['train_data'], tokenizer)
        self.optimization_config = training_config['optimization_settings']
        self.validation_settings = training_config.get('validation_settings', None)
        self.training_settings = training_config.get('training_settings', None)
        
        self.store_dtype = store_dtype
        self.device_map = device_map
        self.rotary_emb = rotary_emb
        self.verbose = verbose


    def save_model(self, model):
        print('saving model...')
        torch.save(model.state_dict(), f"{self.logdir}/qmodel.pth")


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
            self.validation(block=block, step=0)

        # setup intermediate validation
        if self.validation_settings.get('n_intermediate_val', 0):
            val_step = n_epochs * self.activation_storage.n_train_batches//(self.validation_settings['n_intermediate_val'])
            val_step = max(1, val_step)

        # training cycle
        reassign_ratio = self.training_settings.get('reassign_ratio', 1)
        
        for epoch in range(n_epochs):
            for act_idx, q_batch in enumerate(tqdm.tqdm(self.activation_storage.train_q)):
                step = epoch * self.activation_storage.n_train_batches + act_idx

                # apply reassign_ratio setting
                if act_idx % reassign_ratio == 0:
                    switch_reassings(block, 'on')
                else:
                    switch_reassings(block, 'off')
                
                #layers_data = get_layers_data(block)
                
                q_batch = q_batch.to(self.device_map)
                q_input = self.prepare_input(q_batch)
                q_act = block(**q_input)[0]
                fp_act = self.activation_storage.train_fp[act_idx].to(self.device_map)
                
                loss = loss_fn(q_act, fp_act)
                loss.backward()

                #additions = block.self_attn.q_proj.weight_quantizer.additions
                #print("additions:", additions.sum(), additions.grad)
                #log_reassing(layers_data=layers_data, logger=self.logger, step=step)

                # optimization step
                optimization_step(
                    optimizers=optimizers, 
                    step=step, 
                    training_settings=self.training_settings
                )

                # intermediate validation
                if self.validation_settings.get('n_intermediate_val', 0):
                    if (step) and ((step+1)%val_step==0):
                        self.validation(block=block, step=step)
                        
        free_unused_memory()
        switch_reassings(block, 'off')


    @torch.no_grad()
    def validation(self, block, step):
        switch_reassings(block, 'off')
        val_loss = self.validate(block)
        self.logger.log_scalar(mode='val', scalar_name='loss', scalar=val_loss, step=step)
        switch_reassings(block, 'on')


    @torch.no_grad()
    def validate(self, block):
        switch_quantizers(block, 'q')
        losses = []
        for act_idx, q_batch in enumerate(self.activation_storage.val_q):
            q_batch = q_batch.to(self.device_map)
            q_input = self.prepare_input(q_batch)
            q_act = block(**q_input)[0]
            fp_act = self.activation_storage.val_fp[act_idx].to(self.device_map)
            losses.append(torch.mean((q_act-fp_act)**2))

        free_unused_memory()
        return sum(losses)/len(losses)


    def finetune_block_ptq(
            self,
            block,
            collect_fp=True,
            collect_q=True,
            train=False,
            with_input_preparation=False,
            save_block=False,
            ):

        block = block.to(self.device_map)

        if collect_fp:
            switch_quantizers(block, 'fp')
            self.collect_block_activations(block, self.activation_storage.train_fp, with_input_preparation)
            self.collect_block_activations(block, self.activation_storage.val_fp, with_input_preparation)
            switch_quantizers(block, 'q')

        if train:
            self.train_block(block)
        
        if collect_q:
            switch_quantizers(block, 'q')
            self.collect_block_activations(block, self.activation_storage.train_q, with_input_preparation)
            self.collect_block_activations(block, self.activation_storage.val_q, with_input_preparation)
        
        block.cpu()
        
        if save_block:
            print('block saving...')
            torch.save(block.state_dict(), f"{self.logdir}/{self.logger.block_label}.pth")

        free_unused_memory()
