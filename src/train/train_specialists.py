import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import optuna
import os
from src.models.autoencoder import UniversalAutoencoder
from src.data.dataset import get_dataloader
from src.utils import AverageMeter, save_checkpoint, ssim_loss

def train_epoch(model, loader, optimizer, alpha, corruption_type, device):
    model.train()
    losses = AverageMeter()
    for clean, corrupted, labels in loader:
        mask = (labels == corruption_type)
        if not mask.any(): continue
        
        c, corr = clean[mask].to(device), corrupted[mask].to(device)
        optimizer.zero_grad()
        recon = model(corr)
        l1 = F.l1_loss(recon, c)
        ssim = ssim_loss(recon, c)
        loss = alpha * l1 + (1 - alpha) * ssim
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), c.size(0))
    return losses.avg

def validate(model, loader, alpha, corruption_type, device):
    model.eval()
    losses = AverageMeter()
    with torch.no_grad():
        for clean, corrupted, labels in loader:
            mask = (labels == corruption_type)
            if not mask.any(): continue
            c, corr = clean[mask].to(device), corrupted[mask].to(device)
            recon = model(corr)
            l1 = F.l1_loss(recon, c)
            ssim = ssim_loss(recon, c)
            loss = alpha * l1 + (1 - alpha) * ssim
            losses.update(loss.item(), c.size(0))
    return losses.avg

def objective(trial, args):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    bottleneck_dim = trial.suggest_categorical("bottleneck_dim", [64, 128])
    base_channels = trial.suggest_categorical("base_channels", [32, 64])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    alpha = trial.suggest_float("alpha", 0.5, 0.99)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = get_dataloader(args.data_dir, batch_size, split='train')
    val_loader = get_dataloader(args.data_dir, batch_size, split='val')
    
    models = {
        1: UniversalAutoencoder(3, base_channels, bottleneck_dim, dropout).to(device), # Salt-pepper
        2: UniversalAutoencoder(3, base_channels, bottleneck_dim, dropout).to(device), # Blur
        3: UniversalAutoencoder(3, base_channels, bottleneck_dim, dropout).to(device)  # Occlusion
    }
    optimizers = {k: optim.Adam(v.parameters(), lr=lr) for k, v in models.items()}
    
    total_val_loss = 0
    os.makedirs('checkpoints', exist_ok=True)
    for c_type, model in models.items():
        best_loss = float('inf')
        for epoch in range(args.epochs):
            train_epoch(model, train_loader, optimizers[c_type], alpha, c_type, device)
            val_loss = validate(model, val_loader, alpha, c_type, device)
            
            if val_loss and val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(model, f'checkpoints/best_specialist_{c_type}_trial_{trial.number}.pt')
        total_val_loss += best_loss
                
    return total_val_loss

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
