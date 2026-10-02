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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n_trials", type=int, default=10)
    args = parser.parse_args()
        
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
    print("Best trial:", study.best_trial.params)

if __name__ == "__main__":
    main()
