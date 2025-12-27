import torch
from tqdm import tqdm
import math


@torch.inference_mode()
def evaluate(model, dataloader, print_times=10):
    n_steps = len(dataloader)
    model.eval()
    
    model.config.output_attentions = False
    model.config.output_hidden_states = False

    loss = 0
    n_processed_samples = 0
    for step, batch in enumerate(tqdm(dataloader)):
        batch = batch.to(model.device)
        n_samples = batch.shape[0]
        neg_log_likelihood = model(batch, labels=batch).loss.detach()
        
        loss *= n_processed_samples / (n_processed_samples + n_samples)
        n_processed_samples = n_processed_samples + n_samples
        loss += neg_log_likelihood * n_samples / n_processed_samples

        if step!=0 and step%(max(1, n_steps//print_times))==0:
            print(math.exp(loss.item()))

    ppl = math.exp(loss.item())
    return ppl
