import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os

from dataset import HipMRISegmentationDataset
from modules import ImprovedUNet
from modules import dice_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_image_dir = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test"
test_mask_dir = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_test"
model_path = "unet_model_hipmri.pth"

num_classes = 6
batch_size = 4

# Load test dataset
test_dataset = HipMRISegmentationDataset(test_image_dir, test_mask_dir)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = ImprovedUNet(input_channels=1, output_channels=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

dice_per_class = []

# Evaluate model on test set
with torch.no_grad():
    for imgs, masks in test_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)
        batch_dice = dice_score(outputs, masks, num_classes)
        dice_per_class.append(batch_dice)

# Compute mean dice score 
dice_per_class = np.array(dice_per_class)
mean_dice = dice_per_class.mean(axis=0)

print("Dice per class:")
for i, d in enumerate(mean_dice):
    print(f"Class {i}: ", round(d, 4))
