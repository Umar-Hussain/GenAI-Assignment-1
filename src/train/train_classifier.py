import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import optuna
import os
from src.models.classifier import CorruptionClassifier
from src.data.dataset import get_dataloader
from src.utils import AverageMeter, save_checkpoint
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = AverageMeter()
    for _, corrupted, labels in loader:
        corrupted, labels = corrupted.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(corrupted)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), corrupted.size(0))
    return losses.avg

def validate(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for _, corrupted, labels in loader:
            corrupted, labels = corrupted.to(device), labels.to(device)
            preds = model(corrupted)
            loss = criterion(preds, labels)
            losses.update(loss.item(), corrupted.size(0))
            all_preds.extend(preds.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    return losses.avg, acc, p, r, f1, cm

def objective(trial, args):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    base_channels = trial.suggest_categorical("base_channels", [16, 32, 64])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CorruptionClassifier(3, 4, base_channels, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_loader = get_dataloader(args.data_dir, batch_size, split='train') # Needs balanced sampling in dataset
    val_loader = get_dataloader(args.data_dir, batch_size, split='val')
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, acc, p, r, f1, cm = validate(model, val_loader, criterion, device)
        
        if acc > best_acc:
            best_acc = acc
            os.makedirs('checkpoints', exist_ok=True)
            save_checkpoint(model, f'checkpoints/best_cls_trial_{trial.number}.pt')
            
        trial.report(acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return best_acc

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CorruptionClassifier(3, 4, args.base_channels, args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    train_loader = get_dataloader(args.data_dir, args.batch_size, split='train')
    val_loader = get_dataloader(args.data_dir, args.batch_size, split='val')
    
    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_acc = 0.0
    save_path = os.path.join(args.ckpt_dir, "classifier.pth")
    
    print(f"Training Corruption Classifier on {device} ({args.epochs} epochs)...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, acc, p, r, f1, cm = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Acc: {acc*100:.2f}% | F1: {f1:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            print(f"  [+] Saved new best checkpoint -> {save_path} (Val Acc: {best_acc*100:.2f}%)")
            
    print(f"Corruption Classifier training completed. Best Val Acc: {best_acc*100:.2f}%")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--ckpt_dir", type=str, default="models/checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--run_optuna", action="store_true")
    parser.add_argument("--n_trials", type=int, default=5)
    args = parser.parse_args()
        
    if args.run_optuna:
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
        print("Best trial:", study.best_trial.params)
    else:
        train_model(args)

if __name__ == "__main__":
    main()
