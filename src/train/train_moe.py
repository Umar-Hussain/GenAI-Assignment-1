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
    
def build_moe_model(ckpt_dir, temp=1.0, device='cpu'):
    from src.models.moe import GatingNetwork
    gate = GatingNetwork(base_channels=32, num_experts=4, temperature=temp).to(device)
    
    cls_ckpt = os.path.join(ckpt_dir, "classifier.pth")
    if os.path.exists(cls_ckpt):
        try:
            cls_sd = torch.load(cls_ckpt, map_location=device)
            # Copy compatible feature extractor layers to gate
            gate_sd = gate.state_dict()
            matched = {k: v for k, v in cls_sd.items() if k in gate_sd and v.shape == gate_sd[k].shape}
            gate_sd.update(matched)
            gate.load_state_dict(gate_sd)
            print(f"Loaded {len(matched)} layers from classifier.pth into GatingNetwork.")
        except Exception as e:
            print("Note: GatingNetwork init:", e)
            
    # Load specialists
    salt_ae = UniversalAutoencoder().to(device)
    if os.path.exists(os.path.join(ckpt_dir, "specialist_salt.pth")):
        salt_ae.load_state_dict(torch.load(os.path.join(ckpt_dir, "specialist_salt.pth"), map_location=device))
        
    blur_ae = UniversalAutoencoder().to(device)
    if os.path.exists(os.path.join(ckpt_dir, "specialist_blur.pth")):
        blur_ae.load_state_dict(torch.load(os.path.join(ckpt_dir, "specialist_blur.pth"), map_location=device))
        
    occ_ae = UniversalAutoencoder().to(device)
    if os.path.exists(os.path.join(ckpt_dir, "specialist_occlusion.pth")):
        occ_ae.load_state_dict(torch.load(os.path.join(ckpt_dir, "specialist_occlusion.pth"), map_location=device))
        
    experts = [nn.Identity(), salt_ae, blur_ae, occ_ae]
    model = SoftMoE(gate, experts).to(device)
    return model

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_moe_model(args.ckpt_dir, args.temp, device)
    
    train_loader = get_dataloader(args.data_dir, args.batch_size, split='train')
    val_loader = get_dataloader(args.data_dir, args.batch_size, split='val')
    os.makedirs(args.ckpt_dir, exist_ok=True)
    save_path = os.path.join(args.ckpt_dir, "soft_moe.pth")
    
    # Warmup gate
    print(f"\n--- Warmup Gating Network on {device} (1 epoch) ---")
    opt_gate = optim.Adam(model.gate.parameters(), lr=1e-3)
    train_epoch(model, train_loader, opt_gate, 0.0, 0.0, 1.0, 0.0, device)
    
    # Joint training
    print(f"\n--- Joint Soft-MoE Fine-Tuning ({args.epochs} epochs) ---")
    opt_joint = optim.Adam(model.parameters(), lr=args.lr)
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        t_loss = train_epoch(model, train_loader, opt_joint, args.l1_w, args.ssim_w, args.ce_w, args.bal_w, device)
        v_loss = validate(model, val_loader, args.l1_w, args.ssim_w, args.ce_w, args.bal_w, device)
        print(f"[SoftMoE] Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")
        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(model.state_dict(), save_path)
            print(f"  [+] Saved new best Soft-MoE checkpoint -> {save_path}")
            
    print(f"Soft-MoE training completed. Best Val Loss: {best_loss:.4f}")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--ckpt_dir", type=str, default="models/checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--l1_w", type=float, default=0.8)
    parser.add_argument("--ssim_w", type=float, default=0.2)
    parser.add_argument("--ce_w", type=float, default=0.1)
    parser.add_argument("--bal_w", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--run_optuna", action="store_true")
    parser.add_argument("--n_trials", type=int, default=5)
    args = parser.parse_args()
    
    if args.run_optuna:
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
        print("Best trial:", study.best_trial.params)
    else:
        train_model(args)

if __name__ == "__main__":
    main()
