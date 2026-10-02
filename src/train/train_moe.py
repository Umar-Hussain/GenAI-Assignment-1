import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import optuna
import os
from src.models.moe import SoftMoE
from src.models.classifier import CorruptionClassifier
from src.models.autoencoder import UniversalAutoencoder
from src.data.dataset import get_dataloader
from src.utils import AverageMeter, save_checkpoint, ssim_loss

def train_epoch(model, loader, optimizer, l1_w, ssim_w, ce_w, bal_w, device):
    model.train()
    losses = AverageMeter()
    criterion_ce = nn.CrossEntropyLoss()
    for clean, corrupted, labels in loader:
        clean, corrupted, labels = clean.to(device), corrupted.to(device), labels.to(device)
        optimizer.zero_grad()
        
        recon, gate_logits, weights = model(corrupted, return_logits=True)
        l1 = F.l1_loss(recon, clean)
        ssim = ssim_loss(recon, clean)
        ce = criterion_ce(gate_logits, labels)
        
        # Balance loss
        mean_weights = weights.mean(dim=0)
        bal_loss = (mean_weights * torch.log(mean_weights + 1e-8)).sum()
        
        loss = l1_w * l1 + ssim_w * ssim + ce_w * ce + bal_w * bal_loss
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), clean.size(0))
    return losses.avg

def validate(model, loader, l1_w, ssim_w, ce_w, bal_w, device):
    model.eval()
    losses = AverageMeter()
    criterion_ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for clean, corrupted, labels in loader:
            clean, corrupted, labels = clean.to(device), corrupted.to(device), labels.to(device)
            recon, gate_logits, weights = model(corrupted, return_logits=True)
            l1 = F.l1_loss(recon, clean)
            ssim = ssim_loss(recon, clean)
            ce = criterion_ce(gate_logits, labels)
            
            mean_weights = weights.mean(dim=0)
            bal_loss = (mean_weights * torch.log(mean_weights + 1e-8)).sum()
            loss = l1_w * l1 + ssim_w * ssim + ce_w * ce + bal_w * bal_loss
            losses.update(loss.item(), clean.size(0))
    return losses.avg

def objective(trial, args):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    temp = trial.suggest_float("temp", 0.5, 2.0)
    l1_w = trial.suggest_float("l1_w", 0.1, 1.0)
    ssim_w = trial.suggest_float("ssim_w", 0.1, 1.0)
    ce_w = trial.suggest_float("ce_w", 0.1, 1.0)
    bal_w = trial.suggest_float("bal_w", 0.01, 0.5)
    batch_size = 32
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    classifier = CorruptionClassifier(3, 4, 32, 0.1)
    experts = [UniversalAutoencoder(3, 32, 64, 0.1) for _ in range(4)]
    model = SoftMoE(classifier, experts, temp).to(device)
    
    train_loader = get_dataloader(args.data_dir, batch_size, split='train')
    val_loader = get_dataloader(args.data_dir, batch_size, split='val')
    
    # Warmup
    optimizer_gate = optim.Adam(model.gate.parameters(), lr=1e-3)
    for epoch in range(3):
        train_epoch(model, train_loader, optimizer_gate, 0, 0, 1.0, 0, device)
        
    # Joint
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, l1_w, ssim_w, ce_w, bal_w, device)
        val_loss = validate(model, val_loader, l1_w, ssim_w, ce_w, bal_w, device)
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs('checkpoints', exist_ok=True)
            save_checkpoint(model, f'checkpoints/best_moe_trial_{trial.number}.pt')
            
    return best_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n_trials", type=int, default=5)
    args = parser.parse_args()
    
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
    print("Best trial:", study.best_trial.params)

if __name__ == "__main__":
    main()
