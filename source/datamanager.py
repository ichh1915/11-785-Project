import os, inspect, glob, torch

import numpy as np
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
import PIL
from torchvision import transforms

def sorted_list(path):
    tmplist = glob.glob(path)
    tmplist.sort()

    return tmplist

class load_data(Dataset):

    def __init__(self, train=True, bicubic=True, dataset='DIV2K'):
        self.bicubic = bicubic
        self.dataset = dataset

        if dataset == 'DIV2K':
            data_type = 'train' if train else 'valid'
            self.data_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/../DIV2K"
            self.list_hr = sorted_list(self.data_path + "/DIV2K_" + data_type + "_HR/*.png")
            self.list_lr = sorted_list(self.data_path + "/DIV2K_" + data_type + "_LR_unknown/X2/*.png")
        else:
          if bicubic:
            data_type = 'train' if train else 'test'
            self.data_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."
            self.list_lr = sorted_list(os.path.join(self.data_path, data_type + "_lr", "*.npy"))
            self.list_hr = sorted_list(os.path.join(self.data_path, data_type + "_hr", "*.npy"))
          else:
            data_type = '[img_001-img_091]' if train else '[img_092-img_100]'
            self.data_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/../dataset/Urban100/image_SRF_2"
            self.list_lr = sorted_list(self.data_path + "/" + data_type + "*_LR.png")
            self.list_hr = sorted_list(self.data_path + "/" + data_type + "*_HR.png")
    
    def __getitem__(self, idx):

        if self.dataset == 'DIV2K':
          data = transforms.ToTensor()(PIL.Image.open(self.list_lr[idx]).convert('RGB'))
          data = data / 255

          label = transforms.ToTensor()(PIL.Image.open(self.list_hr[idx]).convert('RGB'))
          label = label / 255
          
          data = torch.unsqueeze(data, 0)
          label = torch.unsqueeze(label, 0)
        else:
          if self.bicubic:
            data = np.expand_dims(np.load(self.list_lr[idx]), axis=0)
            label = np.expand_dims(np.load(self.list_hr[idx]), axis=0)

            return torch.from_numpy(np.transpose(data, (0, 3, 1, 2))), torch.from_numpy(np.transpose(label, (0, 3, 1, 2)))
          else:
            data = transforms.ToTensor()(PIL.Image.open(self.list_lr[idx]))
            label = transforms.ToTensor()(PIL.Image.open(self.list_hr[idx]))
            data = torch.unsqueeze(data, 0)
            label = torch.unsqueeze(label, 0)
        
        return data, label

    def __len__(self):
        return len(self.list_lr)