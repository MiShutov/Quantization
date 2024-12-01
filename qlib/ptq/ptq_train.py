import torch
from qlib.ptq.ptq_utils import switch_quantizers, prepare_optimizers
import psutil
import gc

class TrainerPTQ():
    def __init__(
            self, 
            optimization_config,
            prepare_layer_inputs_fn,
            act_store_dtype=torch.float16,
            device_map='cuda'):
        self.prepare_layer_inputs_fn = prepare_layer_inputs_fn
        self.act_store_dtype = act_store_dtype
        self.device_map = device_map
        self.optimization_config = optimization_config

    @torch.no_grad()
    def collect_block_activations(self, block, activations, with_input_preparation):
        for act_idx, batch in enumerate(activations):
            batch = batch.to(self.device_map)
            if with_input_preparation:
                input = self.prepare_layer_inputs_fn(activations=batch)
                block_activations = block(**input)[0]
            else:
                block_activations = block(batch)
            activations[act_idx] = block_activations.detach().cpu().to(self.act_store_dtype)
        torch.cuda.empty_cache()
        gc.collect()

    @torch.no_grad()
    def validate(self, block, activations):
        switch_quantizers(block, 'q')
        val_activations_fp = activations['val_fp']
        val_activations_q = activations['val_q']
        losses = []
        for act_idx, q_batch in enumerate(val_activations_q):
            q_batch = q_batch.to(self.device_map)
            q_input = self.prepare_layer_inputs_fn(activations=q_batch)
            q_act = block(**q_input)[0]
            fp_act = val_activations_fp[act_idx].to(self.device_map)
            losses.append(torch.mean((q_act-fp_act)**2))
        return sum(losses)/len(losses)


    def finetune_block_ptq(
            self,
            block,
            activations,
            collect_fp=True,
            collect_q=True,
            train_block=False,
            with_input_preparation=False,
            ):

        block = block.to(self.device_map)

        process = psutil.Process()
        memory_info = process.memory_info()
        print("before collect_fp", memory_info.rss / (1024 ** 3))

        if collect_fp:
            switch_quantizers(block, 'fp')
            self.collect_block_activations(block, activations['train_fp'], with_input_preparation)
            self.collect_block_activations(block, activations['val_fp'], with_input_preparation)
            switch_quantizers(block, 'q')

        process = psutil.Process()
        memory_info = process.memory_info()
        print("after collect_fp", memory_info.rss / (1024 ** 3))


        if train_block:
            val_loss = self.validate(block, activations)
            print(f"Init val loss: {val_loss.item():.3e}")

            switch_quantizers(block, 'q')
            with torch.enable_grad():
                loss_fn = self.optimization_config['loss_fn']
                n_epochs = self.optimization_config['n_epochs']
                optimizers = prepare_optimizers(
                    self.optimization_config['optimizers'], block
                )
                
                for epoch in range(n_epochs):
                    for act_idx, q_batch in enumerate(activations['train_q']):
                        q_batch = q_batch.to(self.device_map)
                        q_input = self.prepare_layer_inputs_fn(activations=q_batch)
                        q_act = block(**q_input)[0]
                        fp_act = activations['train_fp'][act_idx].to(self.device_map)
                        
                        loss = loss_fn(q_act, fp_act)
                        loss.backward()
                        for optimizer_name in optimizers:
                            optim = optimizers[optimizer_name]['optimizer']
                            scheduler = optimizers[optimizer_name]['scheduler']
                            optim.step()
                            if scheduler is not None:
                                scheduler.step()
                            optim.zero_grad()
                        
            val_loss = self.validate(block, activations)
            print(f"Result val loss: {val_loss.item():.3e}")
        
        process = psutil.Process()
        memory_info = process.memory_info()
        print("after train_block", memory_info.rss / (1024 ** 3))

        if collect_q:
            switch_quantizers(block, 'q')
            self.collect_block_activations(block, activations['train_q'], with_input_preparation)
            self.collect_block_activations(block, activations['val_q'], with_input_preparation)

        process = psutil.Process()
        memory_info = process.memory_info()
        print("after collect_q", memory_info.rss / (1024 ** 3))

        block = block.cpu()

        process = psutil.Process()
        memory_info = process.memory_info()
        print("after block.cpu()", memory_info.rss / (1024 ** 3))

        torch.cuda.empty_cache()
        gc.collect()
        return activations


print('Hello!')
