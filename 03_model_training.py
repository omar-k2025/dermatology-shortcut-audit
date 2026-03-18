"""
03_model_training.py
Train EfficientNet-B3, ResNet50, ViT-B/16 on HAM10000 binary melanoma classification.
Produces: EfficientNet_B3.pth, ResNet50.pth, ViT_B16.pth
"""

import os, glob, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from config import PATHS, TRAIN_CONFIG, MODEL_PATHS, IMAGENET_MEAN, IMAGENET_STD, RANDOM_SEED

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Transforms ────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((TRAIN_CONFIG["img_size"], TRAIN_CONFIG["img_size"])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
val_transform = transforms.Compose([
    transforms.Resize((TRAIN_CONFIG["img_size"], TRAIN_CONFIG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class SkinDataset(Dataset):
    def __init__(self, df, transform=None):
        self.paths = df['path'].tolist()
        self.labels = df['label'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def build_model(mtype):
    if mtype == "efficientnet_b3":
        m = models.efficientnet_b3(weights='IMAGENET1K_V1')
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    elif mtype == "resnet50":
        m = models.resnet50(weights='IMAGENET1K_V1')
        m.fc = nn.Linear(m.fc.in_features, 1)
    elif mtype == "vit_b_16":
        m = models.vit_b_16(weights='IMAGENET1K_V1')
        m.heads.head = nn.Linear(m.heads.head.in_features, 1)
    return m.to(device)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labs = labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += ((torch.sigmoid(out) > 0.5).float() == labs).sum().item()
        total += labs.size(0)
    return total_loss / len(loader), correct / total


def val_epoch(model, loader, criterion):
    model.eval()
    total_loss, all_probs, all_labels = 0, [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labs = labels.float().unsqueeze(1).to(device)
            out = model(imgs)
            total_loss += criterion(out, labs).item()
            all_probs.extend(torch.sigmoid(out).cpu().numpy().flatten())
            all_labels.extend(labels.numpy())
    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / len(loader), auc, all_labels, all_probs


def main():
    # ── Load HAM10000 ──────────────────────────
    ham_images = (
        glob.glob(f"{PATHS['HAM10000']}/HAM10000_images_part_1/*.jpg") +
        glob.glob(f"{PATHS['HAM10000']}/HAM10000_images_part_2/*.jpg")
    )
    meta = pd.read_csv(f"{PATHS['HAM10000']}/HAM10000_metadata.csv")
    meta['label'] = (meta['dx'] == 'mel').astype(int)
    id2p = {os.path.basename(p).replace('.jpg', ''): p for p in ham_images}
    meta['path'] = meta['image_id'].map(id2p)
    meta = meta[meta['path'].notna()].copy()

    train_df, val_df = train_test_split(
        meta, test_size=0.2, random_state=RANDOM_SEED, stratify=meta['label']
    )
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # Class weight
    pos_weight = torch.tensor([8902 / 1113]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(
        SkinDataset(train_df, train_transform),
        batch_size=TRAIN_CONFIG["batch_size"], shuffle=True,
        num_workers=TRAIN_CONFIG["num_workers"]
    )
    val_loader = DataLoader(
        SkinDataset(val_df, val_transform),
        batch_size=TRAIN_CONFIG["batch_size"], shuffle=False,
        num_workers=TRAIN_CONFIG["num_workers"]
    )

    # ── Train 3 models ─────────────────────────
    model_configs = [
        ("EfficientNet-B3", "efficientnet_b3", MODEL_PATHS["EfficientNet_B3"]),
        ("ResNet50",        "resnet50",         MODEL_PATHS["ResNet50"]),
        ("ViT-B/16",        "vit_b_16",         MODEL_PATHS["ViT_B16"]),
    ]

    results = {}
    for name, mtype, save_path in model_configs:
        print(f"\n{'='*55}\n📦 {name}\n{'='*55}")
        model = build_model(mtype)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=TRAIN_CONFIG["lr"],
            weight_decay=TRAIN_CONFIG["weight_decay"]
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=TRAIN_CONFIG["epochs"])

        best_auc = 0
        print(f"{'Ep':<5}{'TrLoss':<12}{'TrAcc':<12}{'ValLoss':<12}{'ValAUC'}")
        print("-" * 55)

        for ep in range(TRAIN_CONFIG["epochs"]):
            tl, ta = train_epoch(model, train_loader, optimizer, criterion)
            vl, va, lbs, pbs = val_epoch(model, val_loader, criterion)
            scheduler.step()

            if va > best_auc:
                best_auc = va
                torch.save(model.state_dict(), save_path)
                flag = " ⭐"
            else:
                flag = ""
            print(f"{ep+1:<5}{tl:<12.4f}{ta:<12.3f}{vl:<12.4f}{va:.4f}{flag}")

        print(f"✅ Best AUC: {best_auc:.4f} → saved to {save_path}")
        results[name] = best_auc
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 40)
    print("Final Results:")
    for name, auc in results.items():
        print(f"  {name}: AUC = {auc:.4f}")


if __name__ == "__main__":
    main()
