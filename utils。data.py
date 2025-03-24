import torch
from torch.utils import data
import numpy as np

class TestDataset(data.Dataset):
    def __init__(self):
        self.Data = np.asarray([[1, 2], [3, 4], [2, 1], [3, 4], [4, 5]])
        self.Label = np.asarray([0, 1, 0, 1, 2])

    def __getitem__(self, index):
        txt = torch.from_numpy(self.Data[index])
        label = torch.tensor(self.Label[index])
        return txt, label

    def __len__(self):
        return len(self.Data)

Test = TestDataset()
print(Test[2])
print(Test.__len__())