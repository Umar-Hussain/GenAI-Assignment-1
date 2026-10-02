import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import optuna
import os
from src.models.autoencoder import UniversalAutoencoder
from src.data.dataset import get_dataloader
from src.utils import AverageMeter, save_checkpoint, ssim_loss

def train_epoch(model, loader, optimizer, alpha, device):
    model.train()
    losses = AverageMeter()
    for clean, corrupted, _ in loader:
        clean, corrupted = clean.to(device), corrupted.to(device)
        optimizer.zero_grad()
        recon = model(corrupted)
        l1 = F.l1_loss(recon, clean)
        ssim = ssim_loss(recon, clean)
        loss = alpha * l1 + (1 - alpha) * ssim
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), clean.size(0))
    return losses.avg

def validate(model, loader, alpha, device):
    model.eval()
    losses = AverageMeter()
    with torch.no_grad():
        for clean, corrupted, _ in loader:
            clean, corrupted = clean.to(device), corrupted.to(device)
            recon = model(corrupted)
            l1 = F.l1_loss(recon, clean)
            ssim = ssim_loss(recon, clean)
            loss = alpha * l1 + (1 - alpha) * ssim
            losses.update(loss.item(), clean.size(0))
    return losses.avg

def objective(trial, args):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    bottleneck_dim = trial.suggest_categorical("bottleneck_dim", [64, 128, 256])
    base_channels = trial.suggest_categorical("base_channels", [32, 64])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    alpha = trial.suggest_float("alpha", 0.5, 0.99)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniversalAutoencoder(3, base_channels, bottleneck_dim, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_loader = get_dataloader(args.data_dir, batch_size, split='train')
    val_loader = get_dataloader(args.data_dir, batch_size, split='val')
    
    best_loss = float('inf')
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, alpha, device)
        val_loss = validate(model, val_loader, alpha, device)
        
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs('checkpoints', exist_ok=True)
            save_checkpoint(model, f'checkpoints/best_uae_trial_{trial.number}.pt')
            
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return best_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()
    
    if args.use_wandb:
        import wandb
        wandb.init(project="universal_autoencoder")
        
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
    print("Best trial:", study.best_trial.params)

if __name__ == "__main__":
    main()
