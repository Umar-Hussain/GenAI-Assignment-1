import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import optuna
import os
from src.models.autoencoder import UniversalAutoencoder
from src.data.dataset import get_dataloader
from src.utils import AverageMeter, save_checkpoint, ssim_loss

def train_epoch_parallel(specialists, loader, optimizers, alpha, device):
    for model in specialists.values():
        model.train()
    losses = {k: AverageMeter() for k in specialists}

    for clean, corrupted, labels in loader:
        clean, corrupted = clean.to(device), corrupted.to(device)
        for c_type, model in specialists.items():
            mask = (labels == c_type)
            if not mask.any():
                continue
            c, corr = clean[mask], corrupted[mask]
            opt = optimizers[c_type]
            opt.zero_grad()
            recon = model(corr)
            l1 = F.l1_loss(recon, c)
            ssim = ssim_loss(recon, c)
            loss = alpha * l1 + (1 - alpha) * ssim
            loss.backward()
            opt.step()
            losses[c_type].update(loss.item(), c.size(0))

    return {k: v.avg for k, v in losses.items()}

def validate_parallel(specialists, loader, alpha, device):
    for model in specialists.values():
        model.eval()
    losses = {k: AverageMeter() for k in specialists}

    with torch.no_grad():
        for clean, corrupted, labels in loader:
            clean, corrupted = clean.to(device), corrupted.to(device)
            for c_type, model in specialists.items():
                mask = (labels == c_type)
                if not mask.any():
                    continue
                c, corr = clean[mask], corrupted[mask]
                recon = model(corr)
                l1 = F.l1_loss(recon, c)
                ssim = ssim_loss(recon, c)
                loss = alpha * l1 + (1 - alpha) * ssim
                losses[c_type].update(loss.item(), c.size(0))

    return {k: v.avg for k, v in losses.items()}

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = get_dataloader(args.data_dir, args.batch_size, split='train')
    val_loader = get_dataloader(args.data_dir, args.batch_size, split='val')
    os.makedirs(args.ckpt_dir, exist_ok=True)

    specialist_specs = {
        1: ("salt", "specialist_salt.pth"),
        2: ("blur", "specialist_blur.pth"),
        3: ("occlusion", "specialist_occlusion.pth")
    }

    specialists = {}
    optimizers = {}
    best_losses = {k: float('inf') for k in specialist_specs}

    uae_ckpt = os.path.join(args.ckpt_dir, "universal_ae.pth")
    has_uae = os.path.exists(uae_ckpt)

    for c_type, (name, _) in specialist_specs.items():
        model = UniversalAutoencoder(3, args.base_channels, args.bottleneck_dim, args.dropout).to(device)
        if has_uae:
            try:
                model.load_state_dict(torch.load(uae_ckpt, map_location=device))
                print(f"Initialized specialist {name} from {uae_ckpt}")
            except Exception as e:
                print(f"Could not load UAE for {name}: {e}")
        specialists[c_type] = model
        optimizers[c_type] = optim.Adam(model.parameters(), lr=args.lr)

    print(f"\nTraining all 3 Specialists in parallel on {device} ({args.epochs} epochs)...")
    for epoch in range(1, args.epochs + 1):
        train_losses = train_epoch_parallel(specialists, train_loader, optimizers, args.alpha, device)
        val_losses = validate_parallel(specialists, val_loader, args.alpha, device)

        msg = f"Epoch {epoch:02d}/{args.epochs:02d} | "
        for c_type, (name, filename) in specialist_specs.items():
            t_loss = train_losses.get(c_type, 0.0)
            v_loss = val_losses.get(c_type, 0.0)
            msg += f"{name.capitalize()}: [T={t_loss:.4f}, V={v_loss:.4f}] "
            if v_loss < best_losses[c_type] and v_loss > 0:
                best_losses[c_type] = v_loss
                save_path = os.path.join(args.ckpt_dir, filename)
                torch.save(specialists[c_type].state_dict(), save_path)
        print(msg)

    print("\nAll specialists trained and best checkpoints saved.")
    return specialists

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--ckpt_dir", type=str, default="models/checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--bottleneck_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.82)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_model(args)

if __name__ == "__main__":
    main()
