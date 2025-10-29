import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import torch.nn.functional as F
import random
import torchvision.transforms.functional as TF

class HipMRISegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, target_size=(256, 128), transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mask_files = sorted(os.listdir(self.mask_dir))
        self.target_size = target_size
        self.transform = transform

    def __len__(self):
        return len(self.mask_files)

    def __getitem__(self, idx):
        
        # get matching image and mask names
        mask_name = self.mask_files[idx]
        img_name = mask_name.replace("seg_", "case_")
        
        # load image and mask
        img = nib.load(os.path.join(self.image_dir, img_name)).get_fdata()
        mask = nib.load(os.path.join(self.mask_dir, mask_name)).get_fdata()

        # convert to tensor
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) 
        mask = torch.tensor(mask, dtype=torch.int64)               

        # resize image to 256x128
        img = img.unsqueeze(0)                          
        img = F.interpolate(img, size=self.target_size, mode="bilinear", align_corners=False)
        img = img.squeeze(0)                             

        # resize mask to 256x128
        mask = mask.unsqueeze(0).unsqueeze(0).float()        
        mask = F.interpolate(mask, size=self.target_size, mode="nearest")
        mask = mask.squeeze().long()

        # normalise
        img = (img - img.mean())/(img.std() + 1e-8)
        
        # apply data augmentation
        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask
    
class RandomFlip:
    def __call__(self, img, mask):
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        if random.random() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        return img, mask
