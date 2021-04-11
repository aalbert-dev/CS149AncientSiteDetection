import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

from test_AncientSites import test_SitesDataset


def parse_args():
    parser = argparse.ArgumentParser(description="AncientSites Detection")
    parser.add_argument(
        "--root-dir",
        type=str,
        default="",
        help="path where test folder is on",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="path where previously trained model is located",
    )
    args = parser.parse_args()
    return args


def test(device, loader):
    model = models.resnet101(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.model))

    results = pd.DataFrame(columns=["Image name", "predict"])

    with torch.no_grad():
        for _, batch in enumerate(loader):
            test_x = batch["image"]
            test_x = test_x.to(device)
            output = model(test_x)
            _, pred_y = output.data.max(1)
            batch_df = pd.DataFrame(
                {"Image name": batch["dir"], "predict": pred_y.cpu().detach().numpy()}
            )
            print(batch_df.head())

            results = results.append(batch_df, sort=False, ignore_index=True)

    return results


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform_data = transforms.Compose([transforms.ToTensor()])
    testset = test_SitesDataset(root_dir=args.root_dir, transform=transform_data)
    test_loader = DataLoader(testset, batch_size=128, shuffle=True, num_workers=5)

    results = test(device, test_loader)
    results.to_csv("predict_result.csv")


if __name__ == "__main__":
    args = parse_args()
    main()
