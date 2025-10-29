import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDoubleConv(nn.Module):
    """
    Basic block with two 3x3 conv layers, InstanceNorm and LeakyReLU 
    Includes a residual (skip) connection for improved gradient flow
    """
    def __init__(self, in_channels, out_channels):
        """Initialises the ResidualDoubleConv block"""
        super().__init__()
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Dropout2d(0.3),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )
        
        self.residual = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        """Forward pass through the ResidualDoubleConv block"""
        residual = self.residual(x)
        out = self.conv(x)
        return out + residual

class ImprovedUNet(nn.Module):
    """
    Improved U-Net for image segmentation, inspired by Isensee et al. (2018)
    includes encoder-decoder structure with skip connections, 
    using InstanceNorm, LeakyReLU and pre-activation residual blocks
    """
    def __init__(self, input_channels=1, output_channels=6):
        """Initialises the Improved U-Net architecture"""
        super(ImprovedUNet, self).__init__()
        # encoder
        self.enc1 = ResidualDoubleConv(input_channels, 16)
        self.down1 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)

        self.enc2 = ResidualDoubleConv(32, 32)
        self.down2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        self.enc3 = ResidualDoubleConv(64, 64)
        self.down3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.enc4 = ResidualDoubleConv(128, 128)
        self.down4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        # bottleneck
        self.bottleneck = ResidualDoubleConv(256, 256)

        # decoder
        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = ResidualDoubleConv(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = ResidualDoubleConv(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ResidualDoubleConv(64, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = ResidualDoubleConv(32, 16)

        # outputs
        self.seg3 = nn.Conv2d(64, output_channels, kernel_size=1)
        self.seg2 = nn.Conv2d(32, output_channels, kernel_size=1)
        self.seg1 = nn.Conv2d(16, output_channels, kernel_size=1)


    def forward(self, x):
        """Forward pass through encoder, bottleneck and decoder with skip connections"""
        # encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        b = self.bottleneck(self.down4(e4))

        # decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # deep supervision outputs
        seg3 = F.interpolate(self.seg3(d3), size=d1.shape[2:], mode="bilinear", align_corners=False)
        seg2 = F.interpolate(self.seg2(d2), size=d1.shape[2:], mode="bilinear", align_corners=False)
        seg1 = self.seg1(d1)

        # sum up the predictions
        out = seg1 + seg2 + seg3
        return out

def dice_loss(pred, target, eps=1e-6):
    """Computes multi-class Dice loss"""
    pred = torch.softmax(pred, dim=1)
    target_onehot = torch.zeros_like(pred)
    target_onehot.scatter_(1, target.unsqueeze(1), 1)
    intersection = (pred * target_onehot).sum(dim=(0, 2, 3))
    union = pred.sum(dim=(0, 2, 3)) + target_onehot.sum(dim=(0, 2, 3))
    dice = (2 * intersection + eps)/(union + eps)
    loss = 1 - dice.mean()
    return loss


def dice_score(pred, target, num_classes=6, eps=1e-6):
    """Computes Dice coefficient per class"""
    pred = torch.softmax(pred, dim=1)
    pred_classes = torch.argmax(pred, dim=1)
    dice_scores = []

    for c in range(num_classes):
        pred_c = (pred_classes == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice = (2 * intersection + eps)/(union + eps)
        dice_scores.append(dice.item())
    return dice_scores