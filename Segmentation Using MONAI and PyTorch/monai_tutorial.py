# Some imports

import monai
import torch
from torch.utils.data import DataLoader
from monai.transforms import (
    EnsureChannelFirstd,
    AsDiscreted,
    Compose,
    LoadImaged,
    Orientationd,
    Randomizable,
    Resized,
    ScaleIntensityd,
    Spacingd,
    EnsureTyped,
    Lambda
)
import monai.metrics as metrics
import os
import tempfile
from utils.decathlon_dataset import get_decathlon_dataloader
from utils.unet import UNET
from utils.train import train

import matplotlib.pyplot as plt


root_dir = './utils/datasets'
task = "Task04_Hippocampus"

train_loader = get_decathlon_dataloader(root_dir, task, "training", batch_size=4, num_workers=0, shuffle=True)
val_loader = get_decathlon_dataloader(root_dir, task, "validation", batch_size=4, num_workers=0, shuffle=False)


# Load UNET model from MONAI

model = monai.networks.nets.UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=3,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)

# Define loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Define device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


from utils.train import train
# Your code here
trained_model = None

trained_model = train(model, train_loader, loss_fn, optimizer, device, epochs=20)


import torch
from monai.metrics import HausdorffDistanceMetric

hausdorff_distance = HausdorffDistanceMetric(include_background=False, reduction="mean", percentile=95)

for val_data in val_loader:
    val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
    val_outputs = torch.nn.functional.one_hot(trained_model(val_inputs).argmax(dim=1), num_classes=3).permute(0, 3, 1, 2)
    val_labels = torch.nn.functional.one_hot(val_labels.long(), num_classes=3).squeeze(1).permute(0, 3, 1, 2)

    # Identify valid classes
    valid_classes = torch.logical_and(torch.any(val_labels, dim=(0, 2, 3)), torch.any(val_outputs, dim=(0, 2, 3)))

    # Only compute Hausdorff distance for valid classes
    if torch.any(valid_classes):
        hausdorff_distance(val_outputs[:, valid_classes], val_labels[:, valid_classes])
    else:
        print("No valid classes found for Hausdorff distance calculation.")

result = hausdorff_distance.aggregate().item()
print(f'95th percentile Hausdorff distance: {result:.4f}')