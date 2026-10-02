import torch
import torch.nn as nn

class CorruptionClassifier(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, base_channels=32, dropout=0.3, **kwargs):
        super().__init__()
        
        # Handle alternate positional call (base_channels, dropout, num_classes)
        if isinstance(num_classes, float) and 0.0 <= num_classes <= 1.0:
            dropout = num_classes
            num_classes = 4

        self.features = nn.Sequential(
            self._make_block(in_channels, base_channels),
            self._make_block(base_channels, base_channels * 2),
            self._make_block(base_channels * 2, base_channels * 4),
            self._make_block(base_channels * 4, base_channels * 8),
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 8, num_classes)
        )

    def _make_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x
