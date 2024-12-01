import torch
from nip import load
from qlib.ptq.ptq_train import TrainerPTQ
from qlib.ptq.ptq_utils import (
	prepare_llama_layer_inputs, 
	prepare_trainig_dataset
)

def parse_config(path_to_config, model, tokenizer, device):
	config = load(path_to_config)
	training_config = config['training_params']

	#wrap model
	config['wrapper'].wrap_model(model)

	#configure dataset
	init_activations = prepare_trainig_dataset(training_config['dataset'], tokenizer)

	#configure training pipline
	trainer = TrainerPTQ(
		optimization_config=training_config['optimization'],
		prepare_layer_inputs_fn=lambda activations: prepare_llama_layer_inputs(
			activations, position_embeddings_func=model.get_decoder().rotary_emb.to(device)
		)
	)

	del config
	torch.cuda.empty_cache()
	
	return init_activations, trainer