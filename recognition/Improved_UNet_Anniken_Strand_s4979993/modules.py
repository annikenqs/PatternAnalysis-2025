import torch
import torch.nn as nn

class ResidualDoubleConv(nn.Module):
    """
    Basic block with two 3x3 conv layers, InstanceNorm, and LeakyReLU. Used throughout the network
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )
        
        self.residual = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv(x)
        return out + residual

class ImprovedUNet(nn.Module):
    """
    Improved U-Net for image segmentation, inspired by Isensee et al. (2018)
    includes encoder-decoder structure with skip connections, 
    using InstanceNorm, LeakyReLU, and pre-activation residual blocks
    """
    def __init__(self, input_channels=1, output_channels=4, base=64):
        super(ImprovedUNet, self).__init__()
        # encoder
        self.enc1 = ResidualDoubleConv(input_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ResidualDoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ResidualDoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = ResidualDoubleConv(256, 512)
        self.dropout = nn.Dropout2d(0.3)

        # decoder
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ResidualDoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ResidualDoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ResidualDoubleConv(128, 64)

        # output layer
        self.out = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        # encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.dropout(self.bottleneck(self.pool3(e3)))


        # decoder (skip connections)
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)
