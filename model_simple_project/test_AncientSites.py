from __future__ import print_function, division
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage import io
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")


class test_SitesDataset(Dataset):

    def __init__(self, root_dir='', transform=None):
        self.root_dir = os.path.join(root_dir, 'test')
        self.img_dir = [os.path.join(self.root_dir, i) for i in sorted(os.listdir(self.root_dir))]
        self.transform = transform

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img_dir = self.img_dir[idx]
        # print(img_dir)
        image = Image.open(img_dir)
        if self.transform is not None:
            image = self.transform(image)
        return {'dir': img_dir, 'image': image}
