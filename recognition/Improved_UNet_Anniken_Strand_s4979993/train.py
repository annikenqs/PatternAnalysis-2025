import torch
import torch.nn as nn
from torch.utils.data import random_split
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from dataset import HipMRISegmentationDataset
from modules import ImprovedUNet

# Paths and hyperparameters
train_image_dir = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train"
train_mask_dir = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_train"
val_image_dir = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate"
val_mask_dir = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_validate"
save_path = "unet_model_hipmri.pth"
num_classes = 6
epochs = 20
batch_size = 4
lr = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
train_dataset = HipMRISegmentationDataset(train_image_dir, train_mask_dir)
val_dataset = HipMRISegmentationDataset(val_image_dir, val_mask_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = ImprovedUNet(input_channels=1, output_channels=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Dice loss function
def dice_loss(pred, target, eps=1e-6):
    pred = torch.softmax(pred, dim=1)
    target_onehot = torch.zeros_like(pred)
    target_onehot.scatter_(1, target.unsqueeze(1), 1)
    intersection = (pred * target_onehot).sum(dim=(0, 2, 3))
    union = pred.sum(dim=(0, 2, 3)) + target_onehot.sum(dim=(0, 2, 3))
    dice = (2 * intersection + eps)/(union + eps)
    loss = 1 - dice.mean()
    return loss

train_losses = []
val_losses = []
best_val_loss = float('inf')

# Training and validation loop
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks) + 0.5 * dice_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    scheduler.step()
    epoch_loss = total_loss/len(train_loader)
    train_losses.append(epoch_loss)
    
    # validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks) + 0.5 * dice_loss(outputs, masks)
            val_loss += loss.item()
    val_loss = val_loss/len(val_loader)
    val_losses.append(val_loss)

    print(epoch+1, "/", epochs, " | training loss:", round(epoch_loss, 4), " | validation loss:", round(val_loss, 4), flush=True)

    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)


