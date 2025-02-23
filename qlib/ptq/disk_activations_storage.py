import torch
import os
import h5py
import re
from torch.utils.data import Dataset
from typing import Literal
import time


class HomequantActivationDataset(Dataset):
    def __init__(self, subset, mode: Literal["static", "dynamic"]):
        self.subset = subset
        self.mode = mode

        keys = list(self.subset.keys())
        self.indices = sorted([int(re.search(r'\d+', key).group()) for key in keys])


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("Index out of range")
        item = self.subset[f'tensor_{index}']
        if self.mode == "static":
            return torch.from_numpy(item[:])
        elif self.mode == "dynamic":
            return item
        else:
            raise


class HomequantActivationStorage:
    def __init__(self, storage_path):
        self.storage_path = storage_path
        if os.path.exists(self.storage_path):
            self.h5 = h5py.File(self.storage_path, 'r+')
        else:
            self.h5 = h5py.File(self.storage_path, 'w')


    def __del__(self):
        try:
            if hasattr(self, 'h5') and self.h5:
                self.h5.close()
        except Exception:
            pass


    def add_activation(self, subset_name, tensor):
        index = len(self.h5[subset_name]) if subset_name in self.h5 else 0
        self.h5.create_dataset(f'{subset_name}/tensor_{index}', data=tensor)


    def get_subset_dataset(self, subset_name, mode='static'):
        if self.h5.get(subset_name) is not None:
            return HomequantActivationDataset(self.h5[subset_name], mode)
        else:
            raise