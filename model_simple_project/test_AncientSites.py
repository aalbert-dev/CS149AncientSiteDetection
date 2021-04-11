from __future__ import division, print_function

import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from skimage import io
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from tqdm import tqdm

warnings.filterwarnings("ignore")


class test_SitesDataset(Dataset):
    def __init__(self, root_dir="", transform=None):
        self.root_dir = os.path.join(root_dir, "test")
        self.img_dir = [
            os.path.join(self.root_dir, i) for i in sorted(os.listdir(self.root_dir))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img_dir = self.img_dir[idx]
        image = Image.open(img_dir).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {"dir": img_dir, "image": image}
