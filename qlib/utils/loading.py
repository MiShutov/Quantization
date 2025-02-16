import os
import nip
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

path_to_datasets = '/mnt/ssd_storage/ml/llm/datasets'
cache_dir = "/mnt/ssd_storage/ml/llm/datasets/huggingface_cache"
path_to_pretrained_models = '/mnt/ssd_storage/ml/llm/pretrained_models'

def load_llama(path_to_pretrained=None, model_name=None, **model_kwargs):
    if (path_to_pretrained is not None) and (model_name is not None):
        raise RuntimeError(f'Specify path_to_pretrained or model_name')
    if model_name is not None:
        path_to_pretrained = os.path.join(path_to_pretrained_models, model_name)
    tokenizer = AutoTokenizer.from_pretrained(path_to_pretrained)
    model = AutoModelForCausalLM.from_pretrained(path_to_pretrained, **model_kwargs)
    return tokenizer, model


def get_data(dataset_name, split, tokenizer):
    path_to_dataset = os.path.join(path_to_datasets, dataset_name)
    test = load_dataset(path_to_dataset, split=split, cache_dir=cache_dir)
    ids = tokenizer("\n\n".join(test["text"]), return_tensors="pt").input_ids
    return ids


class QATDataset(Dataset):
    def __init__(self, config, tokenizer):
        '''
        dataloader.yaml example:
        
        test_data:
            dataset_name: slim_pajama
            split: train[:2500]
            seq_length: 2048
            batch_size: 2
        '''
        self.tokenizer = tokenizer
        self.config = config

        self.batch_size = self.config['batch_size']
        self.seq_length = self.config['seq_length']
        self.token_seq = get_data(
            dataset_name=self.config['dataset_name'],
            split=self.config['split'],
            tokenizer=self.tokenizer
        )
        self.len = self.token_seq.shape[1] // (self.seq_length)
        
        random_seed = self.config.get('random_seed', None)
        if random_seed == 'no_rand':
            self.indices = torch.arange(self.len)
        elif random_seed is not None:
            torch.manual_seed(self.config.get('random_seed', torch))
            self.indices = torch.randperm(self.len)
        else:
            self.indices = torch.randperm(self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        assert index < self.len
        rand_index = self.indices[index]

        begin_loc = self.seq_length * rand_index
        end_loc = self.seq_length * (rand_index+1)
        batch = self.token_seq[0, begin_loc:end_loc]
        return batch

    def get_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size)
