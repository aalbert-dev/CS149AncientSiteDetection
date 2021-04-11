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


class SitesDataset(Dataset):
    def __init__(self, coordinate_txt, root_dir="", transform=None):
        self.info = pd.read_csv(coordinate_txt, sep="\t")
        # root where you put the train folder
        self.root_dir = [os.path.join(root_dir, i) for i in self.info.iloc[:, 5]]

        self.labels = torch.tensor(self.info.iloc[:, 4] > 0)
        self.transform = transform
        self.image_dict = {}

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        if idx in self.image_dict:
            return self.image_dict.get(idx)
        img_dir = self.root_dir[idx]
        image = Image.open(img_dir).convert("RGB")
        # for detection: 1/0
        # for detection on confidence level: 3/2/0
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        sample = {"Image": image, "label": label}
        self.image_dict[idx] = sample
        return sample
