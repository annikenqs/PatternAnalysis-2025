import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class OASISSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mask_files = sorted(os.listdir(self.mask_dir))
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            # normalise around 0
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.mask_files)

    def __getitem__(self, idx):
        
        # get matching image and mask names
        mask_name = self.mask_files[idx]
        img_name = mask_name.replace("seg_", "case_")

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        img = Image.open(img_path)
        mask = Image.open(mask_path)
        img = self.transform(img)
        mask_np = np.array(mask, dtype=np.uint8)

        # map pixel values to class IDs
        mapping = {0: 0, 85: 1, 170: 2, 255: 3}
        mask_np = np.vectorize(mapping.get)(mask_np)

        mask = torch.from_numpy(mask_np).long()

        return img, mask
