import torch
import os
import shutil
from nip import load
from qlib.utils.loading import load_llama
from qlib.ptq.ptq_train import TrainerPTQ, free_unused_memory
from qlib.utils.evaluation import Tester


@torch.no_grad()
def parse_config(path_to_config, device):
    config = load(path_to_config)
    
    logdir = config['logdir']
    os.makedirs(logdir, exist_ok=True)
    try:
      shutil.copy(path_to_config, logdir)
    except shutil.SameFileError:
        pass

    #load model and tokenizer
    tokenizer, model = load_llama(model_name=config['model_name'])

    #wrap model
    config['wrapper'].wrap_model(model)

    if config.get('path_to_checkpoint', False):
        print(f"loading checkpoint {config['path_to_checkpoint']}")
        model.load_state_dict(torch.load(config['path_to_checkpoint'], weights_only=True))
        print(f"checkpoint loaded!")

    #configure trainer
    trainer = TrainerPTQ(
        logdir = logdir,
        training_config=config['training_params'],
        tokenizer=tokenizer,
        rotary_emb=model.get_decoder().rotary_emb.to(device),
        store_dtype=torch.float16,
        device_map=device
    )

    #configure trainer
    tester = Tester(
        logdir = logdir,
        model_name=config['model_name'],
        tokenizer=tokenizer,
        test_config=config['test_params'],
        device_map=device
    )

    del config
    free_unused_memory()

    return {
        "trainer" : trainer,
        "tester" : tester,
        "model" : model
    }
