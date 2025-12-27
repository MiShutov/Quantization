import os
# import nip
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

PATH_TO_DATASETS = '/media/msst/ssd_storage1/ml/llm/datasets'
CACHE_DIR = "/media/msst/ssd_storage1/ml/llm/datasets/huggingface_cache"
PATH_TO_PRETRAINED_MODELS = '/media/msst/ssd_storage1/ml/llm/pretrained_models'


def get_data(dataset_name, split, tokenizer):
    path_to_dataset = os.path.join(PATH_TO_DATASETS, dataset_name)
    test = load_dataset(path_to_dataset, split=split, cache_dir=CACHE_DIR)
    ids = tokenizer("\n\n".join(test["text"]), return_tensors="pt").input_ids
    return ids


class QATDataset(Dataset):
    def __init__(self, config, tokenizer=None, return_dict=False, return_dict_with_labels=False):
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
        self.return_dict = return_dict
        self.return_dict_with_labels = return_dict_with_labels
        self.batch_size = self.config['batch_size']
        self.seq_length = self.config['seq_length']
        
        self.n_seq = self.config.get('n_seq', -1)
        
        self.token_seq = get_data(
            dataset_name=self.config['dataset_name'],
            split=self.config['split'],
            tokenizer=self.tokenizer
        )[0]

        random_seed = self.config.get('random_seed', None)
        
        self.len = self.token_seq.shape[0] // (self.seq_length)
        if self.n_seq != -1:
            self.len = min(self.len, self.n_seq)
        self.token_seq = self.token_seq[:self.len * self.seq_length]

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
        batch = self.token_seq[begin_loc:end_loc]      
        if self.return_dict:
            return {'input_ids' : batch}
        return batch

    def get_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size)


class KnowledgeDistillationDataset(Dataset):
    def __init__(self, data):
        self.data = data
        #self.lm_head = lm_head.cpu() #.to(DEVICE)
        self.len = len(self.data['decoder_output'])
        self.rand_indices = torch.randperm(self.len)
        

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        assert index < self.len
        input_ids = self.data['input'][self.rand_indices[index]]
        decoder_output = self.data['decoder_output'][self.rand_indices[index]]
        return {
            'input_ids' : input_ids[0],
            'decoder_output' : decoder_output[0]
        }
    
