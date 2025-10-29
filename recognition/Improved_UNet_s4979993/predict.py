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

print("Dice similarity coefficient per class:")
for i, d in enumerate(mean_dice):
    print(f"Class {i}: ", round(d, 4))


# Visualisation of predictions
colors = [
    "#000000",  # Class 0 – black
    "#c9c9ff",  # Class 1 – purple
    "#bb6688",  # Class 2 – red
    "#2bc191",  # Class 3 – green
    "#65a3df",  # Class 4 – blue
    "#ffd966"   # Class 5 – yellow
]
class_labels = [
    "Class 0", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5"
]
custom_cmap = ListedColormap(colors)

output_dir = "hipmri_predictions"
os.makedirs(output_dir, exist_ok=True)

num_examples = 3
count = 0
fig, axes = plt.subplots(num_examples, 3, figsize=(9, 3 * num_examples))

# Show input, ground truth, and predictions
with torch.no_grad():
    for imgs, masks in test_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)
        preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

        for j in range(imgs.size(0)):
            if count >= num_examples:
                break

            img = imgs[j].cpu().squeeze()
            mask = masks[j].cpu()
            pred = preds[j].cpu()

            ax_img, ax_gt, ax_pred = axes[count]
            ax_img.imshow(img, cmap="gray")
            ax_gt.imshow(mask, cmap=custom_cmap, vmin=0, vmax=len(colors)-1)
            ax_pred.imshow(pred, cmap=custom_cmap, vmin=0, vmax=len(colors)-1)

            ax_img.set_title("Input MRI", fontsize=11)
            ax_gt.set_title("Ground Truth", fontsize=11)
            ax_pred.set_title("Prediction", fontsize=11)

            for ax in (ax_img, ax_gt, ax_pred):
                ax.axis("off")

            count += 1

        if count >= num_examples:
            break

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
classes = [f"Class {i}" for i in range(num_classes)]
plt.figure(figsize=(5, 4))
bars = plt.bar(classes, mean_dice, color="#b19cd8")
plt.ylim(0, 1)
plt.ylabel("Dice coefficient", fontsize=10)
plt.title(f"Dice per class", fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
# Add value labels above bars
for bar, val in zip(bars, mean_dice):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}", ha="center", fontsize=9)
plt.tight_layout()
dice_plot_path = os.path.join(output_dir, "dice_coefficients.png")
plt.savefig(dice_plot_path, bbox_inches="tight")
plt.close()

