import torch
from qlib.quantizers.quantizer import Quantizer
from qlib.utils.loading import get_data
from qlib.ptq.activations import ActivationStorage

def set_quantize(module, quantize):
    if isinstance(module, Quantizer):
        module._quantize = quantize

@torch.no_grad()
def switch_quantizers(model, mode):
    '''
    'q' : turn on quantizers
    'fp' : turn off quantizers
    '''
    if mode=='q':
        quantize = True
    elif mode=='fp':
        quantize = False
    else:
        raise RuntimeError(f"mode {mode} not in ('q'|'fp')")
        
    #model.apply(lambda x: set_quantize(x, quantize))

    for module_name, module in model.named_modules():
        if isinstance(module, Quantizer):
            module._quantize = quantize


@torch.no_grad()
def switch_reassings(model, mode):
    '''
    'on' : turn on quantizers
    'off' : turn off quantizers
    '''
    if mode=='on':
        reassine = True
    elif mode=='off':
        reassine = False
    else:
        raise RuntimeError(f"mode {mode} not in ('on'|'off')")
        
    for module_name, module in model.named_modules():
        if isinstance(module, Quantizer) and hasattr(module, 'with_reassings'):
            module.with_reassings = reassine


@torch.no_grad()
def configure_train_data(train_data, seq_length, n_train_seq, n_val_seq, batch_size):
    train_data_len = train_data.shape[1]
    tokens_per_batch = seq_length*batch_size

    n_train_batches = n_train_seq//batch_size
    n_val_batches = n_val_seq//batch_size
    
    n_batches = n_train_batches+n_val_batches
    loaded_n_seq = train_data_len//tokens_per_batch

    assert n_batches<loaded_n_seq, 'You need to load more data! Please, increase train_data volume'
    seq_ids = torch.randperm(loaded_n_seq)[:n_batches]
    

    batches = []
    for train_seq_id in seq_ids:
        batch = train_data[:, train_seq_id*tokens_per_batch:(train_seq_id+1)*tokens_per_batch]
        batches.append(batch.reshape(batch_size, seq_length))

    return {
        'train_batches' : batches[:n_train_batches],
        'val_batches' : batches[n_train_batches:]
    }


def configure_optimizer(config, module):
    trainable_params = []
    for param_name, param in module.named_parameters(recurse=True):
        if (config['param_label'] in param_name) and (param.requires_grad==True):
            #param.requires_grad = True
            trainable_params.append(param)

    if trainable_params:
        optimizer_class = getattr(torch.optim, config['class'])
        optimizer = optimizer_class(params=trainable_params, **config['kwargs'])
        scheduler = None
        if config.get('scheduler', False):
            scheduler_class = getattr(torch.optim.lr_scheduler, config['scheduler']['class'])
            scheduler = scheduler_class(optimizer, **config['scheduler']['kwargs'])

        return {
            "optimizer" : optimizer,
            "scheduler" : scheduler
        }
    else:
        return None

def prepare_optimizers(config, module):
    optimizers = {}
    for optimizer_name in config:
        configured_optimizer = configure_optimizer(config[optimizer_name], module)
        if configured_optimizer is not None:
            optimizers.update({optimizer_name: configure_optimizer(config[optimizer_name], module)})
    return optimizers


def prepare_trainig_dataset(dataset_config, tokenizer):
    train_data = get_data(
        dataset_name=dataset_config['dataset_name'],
        split=dataset_config['split'],
        tokenizer=tokenizer
    )
    batches = configure_train_data(
        train_data=train_data,
        seq_length=dataset_config['seq_length'],
        n_train_seq=dataset_config['n_train_seq'],
        n_val_seq=dataset_config['n_val_seq'],
        batch_size=dataset_config['batch_size']
    )
    train_batches = batches['train_batches']
    val_batches = batches['val_batches']
    
    return ActivationStorage(
        train_fp = train_batches,
        train_q = [batch.clone() for batch in train_batches],
        val_fp = val_batches,
        val_q = [batch.clone() for batch in val_batches]
    )
