import torch
import torch.nn as nn

class DenoisingAutoencoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, bottleneck_dim=256, dropout=0.2, **kwargs):
        super().__init__()
        if in_channels > 3 and base_channels == 32:
            base_channels = in_channels
            in_channels = 3
        # Input: in_channels x 128 x 128 -> (base_channels*8) x 8 x 8
        self.encoder = nn.Sequential(
            self._make_enc_block(in_channels, base_channels), # 64x64
            self._make_enc_block(base_channels, base_channels * 2), # 32x32
            self._make_enc_block(base_channels * 2, base_channels * 4), # 16x16
            self._make_enc_block(base_channels * 4, base_channels * 8)  # 8x8
        )
        
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 8 * 8 * 8, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, base_channels * 8 * 8 * 8),
            nn.ReLU(inplace=True)
        )
        self.base_ch8 = base_channels * 8
        
        # Input: (base_channels*8)x8x8 -> Output: 3x128x128
        self.decoder = nn.Sequential(
            self._make_dec_block(base_channels * 8, base_channels * 4), # 16x16
            self._make_dec_block(base_channels * 4, base_channels * 2), # 32x32
            self._make_dec_block(base_channels * 2, base_channels), # 64x64
            nn.ConvTranspose2d(base_channels, 3, kernel_size=4, stride=2, padding=1), # 128x128
            nn.Sigmoid()
        )

    def _make_enc_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def _make_dec_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = x.view(-1, self.base_ch8, 8, 8)
        x = self.decoder(x)
        return x

UniversalAutoencoder = DenoisingAutoencoder

