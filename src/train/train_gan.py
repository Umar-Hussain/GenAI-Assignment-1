import argparse
import os
import json
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
from src.models.gan import ConditionalGenerator, Discriminator
from src.utils import AverageMeter

class FS2KDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        anno_file = os.path.join(root_dir, f"anno_{split}.json")
        if not os.path.exists(anno_file):
            alt_anno = os.path.join(root_dir, "FS2K", f"anno_{split}.json")
            if os.path.exists(alt_anno):
                self.root_dir = os.path.join(root_dir, "FS2K")
                anno_file = alt_anno

        with open(anno_file, 'r') as f:
            raw_annos = json.load(f)

        self.samples = []
        for item in raw_annos:
            img_name = item['image_name']
            p_path = os.path.join(self.root_dir, 'photo', f"{img_name}.jpg")
            sk_name = img_name.replace('photo', 'sketch').replace('image', 'sketch')
            s_path = os.path.join(self.root_dir, 'sketch', f"{sk_name}.jpg")
            style = int(item.get('style', 0))
            if os.path.exists(p_path) and os.path.exists(s_path):
                self.samples.append((p_path, s_path, style))

        self.transform_photo = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.transform_sketch = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p_path, s_path, style = self.samples[idx]
        photo = Image.open(p_path).convert('RGB')
        sketch = Image.open(s_path).convert('L')
        return self.transform_photo(photo), self.transform_sketch(sketch), torch.tensor(style, dtype=torch.long)

def train_epoch(netG, netD, loader, optG, optD, criterionBCE, criterionL1, lambda_l1, device):
    netG.train()
    netD.train()
    loss_G_meter = AverageMeter()
    loss_D_meter = AverageMeter()

    for real_A, real_B, style in loader:
        real_A, real_B, style = real_A.to(device), real_B.to(device), style.to(device)

        # Train Discriminator
        optD.zero_grad()
        fake_B = netG(real_A, style)
        pred_fake = netD(torch.cat((real_A, fake_B.detach()), dim=1), style)
        loss_D_fake = criterionBCE(pred_fake, torch.zeros_like(pred_fake))

        pred_real = netD(torch.cat((real_A, real_B), dim=1), style)
        loss_D_real = criterionBCE(pred_real, torch.ones_like(pred_real))

        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_D.backward()
        optD.step()
        loss_D_meter.update(loss_D.item(), real_A.size(0))

        # Train Generator
        optG.zero_grad()
        pred_fake_g = netD(torch.cat((real_A, fake_B), dim=1), style)
        loss_G_bce = criterionBCE(pred_fake_g, torch.ones_like(pred_fake_g))
        loss_G_l1 = criterionL1(fake_B, real_B) * lambda_l1
        loss_G = loss_G_bce + loss_G_l1
        loss_G.backward()
        optG.step()
        loss_G_meter.update(loss_G.item(), real_A.size(0))

    return loss_G_meter.avg, loss_D_meter.avg

def validate(netG, loader, criterionL1, device):
    netG.eval()
    val_l1 = AverageMeter()
    with torch.no_grad():
        for real_A, real_B, style in loader:
            real_A, real_B, style = real_A.to(device), real_B.to(device), style.to(device)
            fake_B = netG(real_A, style)
            l1 = criterionL1(fake_B, real_B)
            val_l1.update(l1.item(), real_A.size(0))
    return val_l1.avg

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fs2k_dir = os.path.join(args.data_dir, "fs2k", "FS2K")
    if not os.path.exists(fs2k_dir):
        fs2k_dir = os.path.join(args.data_dir, "fs2k")

    train_dataset = FS2KDataset(fs2k_dir, split='train')
    val_dataset = FS2KDataset(fs2k_dir, split='test')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    netG = ConditionalGenerator(
        in_channels=3,
        out_channels=1,
        base_channels=args.base_channels,
        num_styles=3,
        style_embed_dim=args.style_embed_dim,
        dropout=args.dropout
    ).to(device)

    netD = Discriminator(
        in_channels=4,
        base_channels=args.base_channels,
        num_styles=3,
        style_embed_dim=args.style_embed_dim
    ).to(device)

    optG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))
    criterionBCE = nn.BCEWithLogitsLoss()
    criterionL1 = nn.L1Loss()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    save_path = os.path.join(args.ckpt_dir, "generator.pth")
    best_val_l1 = float('inf')

    print(f"Training Face-to-Sketch GAN on {device} ({args.epochs} epochs, {len(train_dataset)} train samples)...")
    for epoch in range(1, args.epochs + 1):
        g_loss, d_loss = train_epoch(netG, netD, train_loader, optG, optD, criterionBCE, criterionL1, args.lambda_l1, device)
        val_l1 = validate(netG, val_loader, criterionL1, device)
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | G Loss: {g_loss:.4f} | D Loss: {d_loss:.4f} | Val L1: {val_l1:.4f}")

        if val_l1 < best_val_l1:
            best_val_l1 = val_l1
            torch.save(netG.state_dict(), save_path)
            print(f"  [+] Saved new best generator -> {save_path} (Val L1: {best_val_l1:.4f})")

    print(f"GAN Generator training completed. Best Val L1: {best_val_l1:.4f}")
    return netG

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--ckpt_dir", type=str, default="models/checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--style_embed_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lambda_l1", type=float, default=100.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    train_model(args)

if __name__ == "__main__":
    main()
