import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from dataset import OASISSegmentationDataset
from modules import UNet

# Paths and hyperparameters
image_dir = "/home/groups/comp3710/OASIS/keras_png_slices_train"
mask_dir = "/home/groups/comp3710/OASIS/keras_png_slices_seg_train"
save_path = "unet_model.pth"
num_classes = 4
epochs = 20
batch_size = 4
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
train_dataset = OASISSegmentationDataset(image_dir, mask_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = UNet(input_channels=1, output_channels=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

train_losses = []

# Training loop
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

    print(epoch+1, "/", epochs, "loss:", epoch_loss)

# Save trained model
torch.save(model.state_dict(), save_path)

