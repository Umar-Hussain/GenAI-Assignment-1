import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import optuna
import os
from src.models.gan import ConditionalGenerator, Discriminator
from src.utils import save_checkpoint
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import glob
from PIL import Image

class FS2KDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.files = glob.glob(os.path.join(root_dir, split, '*', '*.jpg'))
        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        # Assume style is 0,1,2 based on parent folder name
        style = hash(os.path.basename(os.path.dirname(self.files[idx]))) % 3
        return self.transform(img), style

def objective(trial, args):
    gen_lr = trial.suggest_float("gen_lr", 1e-5, 1e-3, log=True)
    disc_lr = trial.suggest_float("disc_lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16])
    base_channels = trial.suggest_categorical("base_channels", [32, 64])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    style_embed_dim = trial.suggest_categorical("style_embed_dim", [16, 32])
    lambda_l1 = trial.suggest_float("lambda_l1", 10, 100)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = ConditionalGenerator(3, 3, 3, style_embed_dim, base_channels, dropout).to(device)
    netD = Discriminator(6, base_channels).to(device)
    
    optG = optim.Adam(netG.parameters(), lr=gen_lr, betas=(0.5, 0.999))
    optD = optim.Adam(netD.parameters(), lr=disc_lr, betas=(0.5, 0.999))
    criterionBCE = nn.BCEWithLogitsLoss()
    criterionL1 = nn.L1Loss()
    
    dataset = FS2KDataset(os.path.join(args.data_dir, 'fs2k'), split='train')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    best_loss = float('inf')
    for epoch in range(args.epochs):
        for i, (real_B, style) in enumerate(loader):
            real_A, real_B = real_B, real_B.to(device) # using autoencoder mode as placeholder
            real_A, style = real_A.to(device), style.to(device)
            
            # Train Discriminator
            optD.zero_grad()
            fake_B = netG(real_A, style)
            pred_fake = netD(torch.cat((real_A, fake_B.detach()), 1))
            loss_D_fake = criterionBCE(pred_fake, torch.zeros_like(pred_fake))
            pred_real = netD(torch.cat((real_A, real_B), 1))
            loss_D_real = criterionBCE(pred_real, torch.ones_like(pred_real))
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D.backward()
            optD.step()
            
            # Train Generator
            optG.zero_grad()
            pred_fake = netD(torch.cat((real_A, fake_B), 1))
            loss_G_GAN = criterionBCE(pred_fake, torch.ones_like(pred_fake))
            loss_G_L1 = criterionL1(fake_B, real_B) * lambda_l1
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optG.step()
            
        if loss_G.item() < best_loss:
            best_loss = loss_G.item()
            os.makedirs('checkpoints', exist_ok=True)
            save_checkpoint(netG, f'checkpoints/best_gan_G_trial_{trial.number}.pt')
            
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
