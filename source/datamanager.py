import os, inspect, glob, torch

import numpy as np
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader

def sorted_list(path):
    tmplist = glob.glob(path)
    tmplist.sort()

    return tmplist

class load_data(Dataset):

    def __init__(self, train=True):

        data_type = 'train' if train else 'test'
        self.data_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."
        self.list_lr = sorted_list(os.path.join(self.data_path, data_type + "_lr", "*.npy"))
        self.list_hr = sorted_list(os.path.join(self.data_path, data_type + "_hr", "*.npy"))

    def __getitem__(self, idx):
        data = np.expand_dims(np.load(self.list_lr[idx]), axis=0)
        label = np.expand_dims(np.load(self.list_hr[idx]), axis=0)

        return torch.from_numpy(np.transpose(data, (0, 3, 1, 2))), torch.from_numpy(np.transpose(label, (0, 3, 1, 2)))

    def __len__(self):
        return len(self.list_lr)