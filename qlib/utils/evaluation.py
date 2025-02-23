import torch
from tqdm import tqdm
import json
import math

from qlib.utils.memmory import free_unused_memory
from qlib.utils.loading import get_data, load_llama, QATDataset
from qlib.ptq.ptq_utils import switch_quantizers, switch_reassings


class Tester:
    def __init__(
            self, 
            logdir,
            model_name,
            tokenizer,
            test_config,
            device_map
        ):
        self.logdir = logdir
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.test_config = test_config['test_data']
        self.device_map = device_map

    @torch.no_grad()
    def run_test(self, model):
        dataloader = QATDataset(
            config=self.test_config,
            tokenizer=self.tokenizer
        ).get_dataloader()

        switch_quantizers(model, 'q')
        switch_reassings(model, 'off')
        quantized_model = get_quantized_model(
            fp_model=load_llama(model_name=self.model_name)[1],
            wrapped_model=model
        )
        del model
        free_unused_memory()

        quantized_model = quantized_model.half().to(self.device_map)

        ppl = evaluate(quantized_model, dataloader, print_times=25)
        json.dump({'test' : ppl}, open(f'{self.logdir}/test.json', 'w'))

@torch.no_grad()
def get_quantized_model(wrapped_model, fp_model):
    for module_name, _ in tqdm(wrapped_model.named_modules()):
        if "weight_quantizer" in module_name:
            module_prefix = module_name.split('.weight_quantizer')[0]
            fp_module = fp_model.get_submodule(module_prefix).cpu()
            q_module = wrapped_model.get_submodule(module_prefix).cuda()
            fp_module.weight.data = q_module.weight_quantizer(q_module.module.weight).detach().cpu()
            q_module = q_module.cpu()
    return fp_model


@torch.no_grad()
def evaluate(model, dataloader, print_times=10):
    n_steps = len(dataloader)
    model.eval()
    
    loss = 0
    n_processed_samples = 0
    for step, batch in enumerate(tqdm(dataloader)):
        batch = batch.to(model.device)
        n_samples = batch.shape[0]
        neg_log_likelihood = model(batch, labels=batch).loss.detach()
        
        loss *= n_processed_samples / (n_processed_samples + n_samples)
        n_processed_samples = n_processed_samples + n_samples
        loss += neg_log_likelihood * n_samples / n_processed_samples

        if step!=0 and step%(n_steps//print_times)==0:
            print(math.exp(loss.item()))

    ppl = math.exp(loss.item())
    return ppl
