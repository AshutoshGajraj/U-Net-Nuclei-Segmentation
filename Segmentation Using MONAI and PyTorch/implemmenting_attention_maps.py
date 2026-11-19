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
import os
import tempfile
from utils.decathlon_dataset import get_decathlon_dataloader
from utils.unet import UNET
from utils.train import train

import matplotlib.pyplot as plt


# Load the data
root_dir = r'utils\datasets'
task = "Task04_Hippocampus"

train_loader = get_decathlon_dataloader(root_dir, task, "training", batch_size=4, num_workers=0, shuffle=True)
val_loader = get_decathlon_dataloader(root_dir, task, "validation", batch_size=4, num_workers=0)

# Load the model

model = UNET(in_channels=1, out_channels=3)

# Load the model weights

model.load_state_dict(torch.load("trained_unet.pth"))

# Set the model to evaluation mode

model.eval()


from utils.attention_maps import GradCAM, apply_cam
# Use the model to generate attention maps
target_layer = model.downs[-1].conv[-3]  # Using the last ReLU in the last downsampling block

grad_cam = GradCAM(model, target_layer)

# Find a sample image with labels
for sample_data in val_loader:
    sample_label = sample_data['label'][20:21]

    if sample_label.sum() > 10:  # Check if the label contains any positive values
        sample_image = sample_data['image'][20:21] # Add batch dimension
        break

# Generate prediction
with torch.no_grad():
    sample_prediction = model(sample_image).argmax(dim=1)

# Generate CAM for each class
for class_idx in range(3):  # Assuming 3 classes (background, anterior, posterior)
    cam = grad_cam.generate_cam(sample_image, class_idx)
    print(f"Attention map for class {class_idx}")
    apply_cam(sample_image[0], sample_label, cam, class_idx, sample_prediction)