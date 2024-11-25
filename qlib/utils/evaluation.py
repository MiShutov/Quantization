import torch
from tqdm import tqdm


def eval(model, eval_seq, seq_length, batch_size=1, print_times=10, last_batch=True):
	'''
	model
	eval_seq
	'''
	model.eval()
	if seq_length is not None:
		stride = seq_length
	else:
		stride = model.config.max_position_embeddings


	input_len = torch.prod(torch.tensor(eval_seq.shape))
	losses = []

	max_idx = input_len//(stride * batch_size)
	for step in tqdm(range(max_idx+1)):
		
		begin_loc = batch_size * stride * step
		if step==max_idx:
			batch = eval_seq[:, begin_loc:]
			if not last_batch:
				continue
		else:		
			batch = eval_seq[:, begin_loc:begin_loc+batch_size*stride]
			batch = batch.reshape(-1, stride)

		batch = batch.to(model.device)

		with torch.no_grad():
			with torch.autocast(device_type="cuda"):
				outputs = model(batch, labels=batch)
				neg_log_likelihood = outputs.loss.detach()
		
		losses.append(neg_log_likelihood)
		if step!=0 and step%(max_idx//print_times)==0:
			print(torch.exp(torch.stack(losses).mean()).item())

		
	ppl = torch.exp(torch.stack(losses).mean())
	return ppl.item()