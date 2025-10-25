import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import nibabel as nib
import numpy as np
import torch.nn.functional as F

class HipMRISegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mask_files = sorted(os.listdir(self.mask_dir))
        self.transform = transform or transforms.Normalize(mean=[0.5], std=[0.5])
        self.target_size = target_size

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
        img = torch.tensor(np.squeeze(img), dtype=torch.float32).unsqueeze(0)  
        mask = torch.tensor(np.squeeze(mask).astype(np.int64))               

        # resize image to 256x256 
        img = img.unsqueeze(0)                          
        img = F.interpolate(img, size=self.target_size, mode="bilinear", align_corners=False)
        img = img.squeeze(0)                             

        # resize mask to 256x256 
        mask = mask.unsqueeze(0).unsqueeze(0).float()        
        mask = F.interpolate(mask, size=self.target_size, mode="nearest")
        mask = mask.squeeze().long()

        # normalize
        img = (img - img.mean())/(img.std() + 1e-8)

        return img, mask
