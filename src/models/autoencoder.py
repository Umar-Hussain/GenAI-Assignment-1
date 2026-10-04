import torch
import torch.nn as nn

class DenoisingAutoencoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, bottleneck_dim=256, dropout=0.1, **kwargs):
        super().__init__()
        if in_channels > 3 and base_channels == 32:
            base_channels = in_channels
            in_channels = 3

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout)
        )

        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Sequential(
            nn.Conv2d(base_channels, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)                           # (B, 32, 64, 64)
        e2 = self.enc2(e1)                          # (B, 64, 32, 32)
        e3 = self.enc3(e2)                          # (B, 128, 16, 16)
        e4 = self.enc4(e3)                          # (B, 256, 8, 8)

        b = self.bottleneck(e4)                     # (B, 256, 8, 8)

        d4 = self.dec4(b)                           # (B, 128, 16, 16)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))  # (B, 64, 32, 32)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))  # (B, 32, 64, 64)
        d1 = self.dec1(torch.cat([d2, e1], dim=1))  # (B, 32, 128, 128)
        return self.final(d1)

UniversalAutoencoder = DenoisingAutoencoder
