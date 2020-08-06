
from torch.utils.data import DataLoader, Dataset
import torch
from . import param_default


class MyDataSet(Dataset):
    def __init__(self, segments, labels):
        self.data = torch.Tensor(segments)
        self.labels = torch.Tensor(labels)

        self.data = self.data.transpose(1, 2)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data.shape[0]


def preprocess_data(data, labels=None, param=None, is_train=True):
    import numpy as np
    if labels is None:
        labels = np.zeros([data.shape[0], 1])

    dataset = MyDataSet(data, labels)
    batch_size = param_default(param, "batch_size", 64)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_train,
        drop_last=is_train
    )

    return data_loader
