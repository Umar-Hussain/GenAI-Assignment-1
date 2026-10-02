import torch
import torch.nn as nn
import torch.nn.functional as F

class GatingNetwork(nn.Module):
    def __init__(self, base_channels=32, num_experts=4, temperature=1.0, **kwargs):
        super().__init__()
        self.temperature = temperature
        
        self.features = nn.Sequential(
            nn.Conv2d(3, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 4, num_experts)

    def forward(self, x, return_logits=False):
        feat = self.features(x)
        pooled = self.pool(feat)
        flat = pooled.view(pooled.size(0), -1)
        logits = self.fc(flat)
        weights = F.softmax(logits / self.temperature, dim=1)
        if return_logits:
            return logits, weights
        return weights

class SoftMoE(nn.Module):
    def __init__(self, gate, experts, **kwargs):
        super().__init__()
        self.gate = gate
        self.experts = nn.ModuleList(experts)
        
    def forward(self, x, return_logits=False):
        if return_logits:
            logits, weights = self.gate(x, return_logits=True)
        else:
            weights = self.gate(x, return_logits=False)
        
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
            
        expert_outputs = torch.stack(expert_outputs, dim=1)
        weights_expanded = weights.view(weights.size(0), -1, 1, 1, 1)
        output = torch.sum(weights_expanded * expert_outputs, dim=1)
        
        if return_logits:
            return output, logits, weights
        return output, weights
