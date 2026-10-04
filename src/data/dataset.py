import torch
import os
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.datasets import OxfordIIITPet
import random
import json

from .corruptions import apply_salt_pepper, apply_gaussian_blur, apply_rectangular_occlusion

class PetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, corruption_manifest=None):
        super().__init__()
        self.split = split
        dataset_split = 'trainval' if split in ['train', 'val'] else 'test'
        self.dataset = OxfordIIITPet(root=root_dir, split=dataset_split, download=True)
        
        self.base_transform = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor(),
        ])
        self.transform = transform
        
        self.indices = None
        splits_file = os.path.join(root_dir, "splits", f"{split}_indices.json")
        if split in ['train', 'val'] and os.path.exists(splits_file):
            with open(splits_file, 'r') as f:
                self.indices = json.load(f)

        self.manifest = None
        if corruption_manifest and os.path.exists(corruption_manifest):
            with open(corruption_manifest, 'r') as f:
                self.manifest = json.load(f)
        else:
            default_manifest = os.path.join(root_dir, "manifests", f"{split}_manifest.json")
            if os.path.exists(default_manifest):
                with open(default_manifest, 'r') as f:
                    self.manifest = json.load(f)

        self.corruption_types = ["clean", "salt_pepper", "gaussian_blur", "occlusion"]

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.dataset)

    def _apply_corruption(self, img, params):
        c_type = params["type"]
        if c_type == "clean":
            return img, 0
        elif c_type == "salt_pepper":
            corrupted, _ = apply_salt_pepper(img, params["prob"])
            return corrupted, 1
        elif c_type == "gaussian_blur":
            corrupted, _ = apply_gaussian_blur(img, params["kernel_size"], params["sigma"])
            return corrupted, 2
        elif c_type == "occlusion":
            corrupted, _ = apply_rectangular_occlusion(img, params["num_rects"], params["coverage"])
            return corrupted, 3
        return img, 0

    def __getitem__(self, idx):
        actual_idx = self.indices[idx] if self.indices is not None else idx
        img, _ = self.dataset[actual_idx]
        img = self.base_transform(img)
        
        if self.transform:
            img = self.transform(img)

        if self.split == 'train':
            c_type = random.choice(self.corruption_types)
            params = {"type": c_type}
            if c_type == "salt_pepper":
                params["prob"] = random.uniform(0.02, 0.15)
            elif c_type == "gaussian_blur":
                params["kernel_size"] = random.choice([3, 5, 7])
                params["sigma"] = random.uniform(0.5, 2.5)
            elif c_type == "occlusion":
                params["num_rects"] = random.randint(1, 3)
                params["coverage"] = random.uniform(0.10, 0.35)
        else:
            if self.manifest:
                params = self.manifest[idx]
            else:
                params = {"type": "clean"}

        corrupted_img, label = self._apply_corruption(img, params)
        
        return img, corrupted_img, label

def get_dataloader(data_dir, batch_size=32, split='train', shuffle=None, num_workers=0):
    from torch.utils.data import DataLoader
    if shuffle is None:
        shuffle = (split == 'train')
    ds = PetDataset(root_dir=data_dir, split=split)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

