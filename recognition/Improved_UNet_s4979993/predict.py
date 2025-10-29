import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from dataset import HipMRISegmentationDataset
from modules import ImprovedUNet
from modules import dice_score
from matplotlib.colors import ListedColormap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths and hyperparameters
test_image_dir = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test"
test_mask_dir = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_test"
model_path = "unet_model_hipmri.pth"
num_classes = 6
batch_size = 4

# Load test dataset
test_dataset = HipMRISegmentationDataset(test_image_dir, test_mask_dir)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load trained model
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

print("Dice similarity coefficient per class:")
for i, d in enumerate(mean_dice):
    print(f"Class {i}: ", round(d, 4))


# Define class colors and labels for visualisation
colors = [
    "#000000",  # 0 – Background (black)
    "#c9c9ff",  # 1 – Body (purple)
    "#bb6688",  # 2 – Bones (red)
    "#2bc191",  # 3 – Bladder (green)
    "#65a3df",  # 4 - Rectum (blue)
    "#ffd966"   # 5 – Prostate (yellow)
]
class_labels = [
    "0: Background",
    "1: Body",
    "2: Bones",
    "3: Bladder",
    "4: Rectum",
    "5: Prostate"
]

custom_cmap = ListedColormap(colors)

# Create output directory for predictions
output_dir = "hipmri_predictions"
os.makedirs(output_dir, exist_ok=True)

# Visualise predictions on a few examples
num_examples = 3
found_samples = []

with torch.no_grad():
    for imgs, masks in test_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)
        preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

        for j in range(imgs.size(0)):
            mask = masks[j].cpu()
            unique_classes = torch.unique(mask)

            # Only keep samples containing all 6 classes
            if len(unique_classes) == num_classes and all(
                c in unique_classes for c in range(num_classes)
            ):
                found_samples.append((imgs[j].cpu().squeeze(), mask, preds[j].cpu()))

            if len(found_samples) >= num_examples:
                break
        if len(found_samples) >= num_examples:
            break

# Plot input, ground truth and prediction
fig, axes = plt.subplots(num_examples, 3, figsize=(9, 3 * num_examples))

for i, (img, mask, pred) in enumerate(found_samples):
    ax_img, ax_gt, ax_pred = axes[i]
    ax_img.imshow(img, cmap="gray")
    ax_gt.imshow(mask, cmap=custom_cmap, vmin=0, vmax=len(colors)-1)
    ax_pred.imshow(pred, cmap=custom_cmap, vmin=0, vmax=len(colors)-1)

    ax_img.set_title("Input MRI", fontsize=11)
    ax_gt.set_title("Ground Truth", fontsize=11)
    ax_pred.set_title("Prediction", fontsize=11)

    for ax in (ax_img, ax_gt, ax_pred):
        ax.axis("off")

# Add color legend for classes
patches = [mpatches.Patch(color=colors[i], label=class_labels[i]) for i in range(len(colors))]
fig.legend(
    handles=patches,
    loc="lower center",
    ncol=3,
    fontsize=9,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02)
)

plt.tight_layout(pad=0.5, w_pad=0.3, h_pad=0.5)
plt.subplots_adjust(bottom=0.1) 
plt.savefig(os.path.join(output_dir, "hipmri_predictions.png"), bbox_inches="tight", dpi=200)
plt.close()
    
# Dice coefficient visualisation
classes = ["Background", "Body", "Bones", "Bladder", "Rectum", "Prostate"]
plt.figure(figsize=(6, 4))
bars = plt.bar(classes, mean_dice, color="#b19cd8")
plt.ylim(0, 1)
plt.ylabel("Dice coefficient", fontsize=10)
plt.title("Dice per class", fontsize=10)
plt.xticks(rotation=30, fontsize=8)
plt.yticks(fontsize=8)

# Add value labels above bars
for bar, val in zip(bars, mean_dice):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}", ha="center", fontsize=9)
    
plt.tight_layout()
dice_plot_path = os.path.join(output_dir, "dice_coefficients.png")
plt.savefig(dice_plot_path, bbox_inches="tight")
plt.close()

