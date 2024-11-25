import torch
from qlib.quantizers.quantizer import Quantizer



@torch.no_grad()
def switch_quantizers(model, mode):
	'''
	'quntize' : turn on quantizers
	'fp' : turn off quantizers
	'''
	if mode=='quantize':
		quantize = True
	elif mode=='fp':
		quantize = False
	else:
		raise RuntimeError(f"mode {mode} not in ('quantize'|'fp')")
		
	#net.apply(init_weights)

	for module_name, module in model.named_modules():
		if isinstance(module, Quantizer):
			module._quantize = quantize


@torch.no_grad()
def configure_train_data(train_data, seq_length, n_seq, batch_size):
	train_data_len = train_data.shape[1]
	batch_tokens = seq_length*batch_size
	n_batches = n_seq//batch_size

	train_data_n_seq = train_data_len//batch_tokens
	train_seq_ids = torch.randperm(train_data_n_seq)[:n_batches]

	train_ids = []
	for train_seq_id in train_seq_ids:
		batch = train_data[:, train_seq_id*batch_tokens:(train_seq_id+1)*batch_tokens]
		
		train_ids.append(batch.reshape(batch_size, seq_length))

	return train_ids


@torch.no_grad()
def prepare_llama_layer_inputs(activations, config, position_embeddings_func):
	causal_mask=None
	past_key_values=None
	past_seen_tokens=0
	output_attentions=config.output_attentions
	use_cache=False
	cache_position = torch.arange(
					past_seen_tokens, past_seen_tokens + activations.shape[1],
					device=activations.device
				)
	position_ids = cache_position.unsqueeze(0)
	position_embeddings = position_embeddings_func(activations, position_ids)
	return {
		"hidden_states" : activations,
		"attention_mask" : causal_mask,
		"position_ids" : position_ids,
		"past_key_value" : past_key_values,
		"output_attentions" : output_attentions,
		"use_cache" : use_cache,
		"cache_position" : cache_position,
		"position_embeddings": position_embeddings
	}