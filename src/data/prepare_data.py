import os
from pathlib import Path
from torchvision.datasets import OxfordIIITPet
from sklearn.model_selection import train_test_split
import json
import argparse

try:
    from .manifest import generate_manifest
except ImportError:
    from manifest import generate_manifest

def prepare(data_dir=None):
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    else:
        data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading trainval split...")
    trainval = OxfordIIITPet(root=data_dir, split='trainval', download=True)
    print("Downloading test split...")
    test = OxfordIIITPet(root=data_dir, split='test', download=True)
    
    indices = list(range(len(trainval)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    with open(splits_dir / "train_indices.json", 'w') as f:
        json.dump(train_idx, f)
    with open(splits_dir / "val_indices.json", 'w') as f:
        json.dump(val_idx, f)
        
    print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}, Test size: {len(test)}")
    
    manifest_dir = data_dir / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    
    generate_manifest(len(val_idx), 'val', seed=42, save_path=manifest_dir / "val_manifest.json")
    generate_manifest(len(test), 'test', seed=42, save_path=manifest_dir / "test_manifest.json")
    print("Manifests generated.")

if __name__ == "__main__":
    prepare()
