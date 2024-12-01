import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

path_to_datasets = '/home/msst/repo/datasets'
path_to_pretrained_models = '/home/msst/repo/pretrained_models'


def load_llama(path_to_pretrained=None, model_name=None):
    if (path_to_pretrained is not None) and (model_name is not None):
        raise RuntimeError(f'Specify path_to_pretrained or model_name')
    if model_name is not None:
        path_to_pretrained = os.path.join(path_to_pretrained_models, model_name)
    tokenizer = AutoTokenizer.from_pretrained(path_to_pretrained)
    model = AutoModelForCausalLM.from_pretrained(path_to_pretrained)
    return tokenizer, model


def get_data(dataset_name, split, tokenizer):
    path_to_dataset = os.path.join(path_to_datasets, dataset_name)
    test = load_dataset(path_to_dataset, split=split)
    ids = tokenizer("\n\n".join(test["text"]), return_tensors="pt").input_ids
    return ids