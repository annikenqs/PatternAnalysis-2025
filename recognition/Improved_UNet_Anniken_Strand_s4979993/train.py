import torch
import torch.nn as nn
from torch.utils.data import random_split
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from dataset import OASISSegmentationDataset
from modules import UNet

# Paths and hyperparameters
train_image_dir = "/home/groups/comp3710/OASIS/keras_png_slices_train"
train_mask_dir = "/home/groups/comp3710/OASIS/keras_png_slices_seg_train"
val_image_dir = "/home/groups/comp3710/OASIS/keras_png_slices_validate"
val_mask_dir = "/home/groups/comp3710/OASIS/keras_png_slices_seg_validate"
save_path = "unet_model.pth"
num_classes = 4
epochs = 20
batch_size = 4
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
train_dataset = OASISSegmentationDataset(train_image_dir, train_mask_dir)
val_dataset = OASISSegmentationDataset(val_image_dir, val_mask_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = UNet(input_channels=1, output_channels=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

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
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    epoch_loss = total_loss/len(train_loader)
    train_losses.append(epoch_loss)
    
    # validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    val_loss = val_loss/len(val_loader)
    val_losses.append(val_loss)

    print(epoch+1, "/", epochs, "training loss:", epoch_loss, "validation loss:", val_loss)
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)


