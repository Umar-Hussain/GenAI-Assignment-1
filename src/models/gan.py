import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=64, num_styles=3, style_embed_dim=16, dropout=0.5):
        super().__init__()
        self.style_emb = nn.Embedding(num_styles, style_embed_dim)
        
        # Input will be image (in_channels) + style_emb expanded
        inc = in_channels + style_embed_dim
        
        # Encoder (downsampling)
        self.enc1 = self._conv_block(inc, base_channels, normalize=False)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self._conv_block(base_channels * 4, base_channels * 8)
        
        # Decoder (upsampling) with skip connections
        self.dec1 = self._upconv_block(base_channels * 8, base_channels * 4, dropout)
        self.dec2 = self._upconv_block(base_channels * 8, base_channels * 2, dropout)
        self.dec3 = self._upconv_block(base_channels * 4, base_channels, 0.0)
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def _conv_block(self, in_c, out_c, normalize=True):
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def _upconv_block(self, in_c, out_c, dropout_rate=0.0):
        layers = [
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        ]
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x, style_id):
        b, _, h, w = x.shape
        style = self.style_emb(style_id) # (B, E)
        style = style.view(b, -1, 1, 1).expand(b, -1, h, w)
        x = torch.cat([x, style], dim=1)
        
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        d1 = self.dec1(e4)
        d1 = torch.cat([d1, e3], dim=1)
        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        
        return self.final(d3)

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=4, base_channels=64, num_styles=3, style_embed_dim=16):
        super().__init__()
        self.style_emb = nn.Embedding(num_styles, style_embed_dim)
        inc = in_channels + style_embed_dim
        
        self.model = nn.Sequential(
            self._conv_block(inc, base_channels, normalize=False),
            self._conv_block(base_channels, base_channels * 2),
            self._conv_block(base_channels * 2, base_channels * 4),
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 8, 1, 4, 1, 1)
        )

    def _conv_block(self, in_c, out_c, normalize=True):
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x, style_id):
        b, _, h, w = x.shape
        style = self.style_emb(style_id)
        style = style.view(b, -1, 1, 1).expand(b, -1, h, w)
        x = torch.cat([x, style], dim=1)
        return self.model(x)

ConditionalGenerator = UNetGenerator
Discriminator = PatchGANDiscriminator

