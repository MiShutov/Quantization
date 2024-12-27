import torch
from nip import load
from qlib.utils.loading import load_llama
from qlib.ptq.ptq_train import TrainerPTQ, free_unused_memory
from qlib.ptq.ptq_utils import prepare_trainig_dataset

def parse_config(path_to_config, device):
    config = load(path_to_config)
    training_config = config['training_params']

    #load model
    tokenizer, model = load_llama(model_name=config['model_name'])

    #wrap model
    config['wrapper'].wrap_model(model)

    if config.get('path_to_checkpoint', False):
        print(f'loading checkpoint {config['path_to_checkpoint']}')
        model.load_state_dict(torch.load(config['path_to_checkpoint'], weights_only=True))
        print(f'checkpoint loaded!')

    #configure dataset
    activation_storage = prepare_trainig_dataset(training_config['dataset'], tokenizer)

    #configure training pipeline
    rotary_emb = model.get_decoder().rotary_emb.to(device)
    trainer = TrainerPTQ(
        logdir = training_config['logdir'],
        optimization_config=training_config['optimization'],
        validation_settings=training_config['validation_settings'],
        rotary_emb=rotary_emb,
        store_dtype=torch.float16,
        #verbose = True,
    )

    del config
    free_unused_memory()
    return {
        "trainer" : trainer,
        "activation_storage": activation_storage,
        "tokenizer" : tokenizer,
        "model" : model
    }
