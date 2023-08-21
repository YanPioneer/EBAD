import numpy as np
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def my_collate_fn(x):
    label = torch.LongTensor(np.array([x_['label'] for x_ in x]))
    kg_matrix = torch.LongTensor([x_['kg_matrix'].cpu().numpy() for x_ in x])
    # mask = torch.LongTensor([x_['mask'].numpy() for x_ in x])
    mask = torch.Tensor([x_['mask'].cpu().numpy() for x_ in x])

    return label, kg_matrix, mask