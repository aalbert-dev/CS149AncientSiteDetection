from __future__ import division, print_function

import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from tqdm import tqdm

from AncientModel_simple import SimpleModel
from test_AncientSites import test_SitesDataset

warnings.filterwarnings("ignore")

datasets = test_SitesDataset(transform=transforms.ToTensor())
test_loader = DataLoader(datasets, batch_size=128, shuffle=False, num_workers=4)
device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# the following two line should be modified for the change of model architecture
#####################################################################################
model = SimpleModel()
model.load_state_dict(torch.load("simple/ModelAt0Epoch.pt"))


#####################################################################################


def test(model, test_loader, device):
    model.to(device)
    model.eval()
    pred_label = []
    pred_list = []
    with torch.no_grad():
        for _, batch in tqdm(enumerate(test_loader)):
            test_x = batch["image"].to(device)
            test_name = batch["dir"]

            output = model(test_x)
            _, pred_y = output.data.max(1)

            pred_label.extend(pred_y.cpu().numpy())
            pred_list.extend(test_name)

    return pred_label, pred_list


pred_y, pred_list = test(model=model, test_loader=test_loader, device=device0)
pred_df = pd.DataFrame({"Image name": pred_list, "predict": pred_y})
pred_df.to_csv("predict_result.csv", index=False)
