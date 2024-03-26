from __future__ import absolute_import
import os
import os.path as osp
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image
import torch.utils.data as data
import random


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        #fname, pid, camid = self.dataset[index]
        input_data = self.dataset[index]
        fname = input_data[0]
        pid = input_data[1]
        camid = input_data[2]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, camid, index



class TrainPreprocessor(object):
    def __init__(self, dataset, root=None, transform=None, aug_transform=None):
        super(TrainPreprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.aug_transform = aug_transform
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        #print('batch image indices= ', indices)
        if isinstance(indices, (tuple, list)):    
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        data = self.dataset[index]
        fpath = data[0] # fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img1 = self.transform(img)

        if self.aug_transform is not None:
            img2 = self.aug_transform(img)
            return (img1, img2) + data[1:]  # img1, img2, pid, camid, img_index

        return (img1,) + data  # img, fname, pid, camid, img_index




