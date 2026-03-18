"""
08_ablation_resolution.py
Resolution ablation study: disentangle resolution vs artifact effects.
Produces: fig_ablation_resolution.png
"""

import os, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from config import PATHS, MODEL_PATHS, IMAGENET_MEAN, IMAGENET_STD, RANDOM_SEED
from utils.delong import delong_ci
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transform_base = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def get_transform_with_resize(target_size=224):
    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.Resize((224, 224)),  # Always upscale to 224 for model
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_probs_resized(model, paths, downscale_to=None):
    """Predict with optional downsampling before feeding to model."""
    probs = []
    transform = get_transform_with_resize(downscale_to) if downscale_to else \
                get_transform_with_resize(224)
    for p in paths:
        try:
            img = Image.open(p).convert('RGB')
            t   = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                probs.append(torch.sigmoid(model(t)).item())
        except Exception:
            probs.append(0.5)
    return probs


def main():
    # Load HAM10000 val set
    ham_images = (
        glob.glob(f"{PATHS['HAM10000']}/HAM10000_images_part_1/*.jpg") +
        glob.glob(f"{PATHS['HAM10000']}/HAM10000_images_part_2/*.jpg")
    )
    meta = pd.read_csv(f"{PATHS['HAM10000']}/HAM10000_metadata.csv")
    meta['label'] = (meta['dx'] == 'mel').astype(int)
    id2p = {os.path.basename(p).replace('.jpg', ''): p for p in ham_images}
    meta['path'] = meta['image_id'].map(id2p)
    meta = meta[meta['path'].notna()].copy()
    _, val_df = train_test_split(meta, test_size=0.2, random_state=RANDOM_SEED,
                                  stratify=meta['label'])
    val_paths  = val_df['path'].tolist()
    val_labels = val_df['label'].tolist()
    print(f"Val set: {len(val_df)} images")

    # Load EfficientNet-B3
    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model.load_state_dict(torch.load(MODEL_PATHS["EfficientNet_B3"], map_location=device))
    model = model.to(device).eval()

    # Resolution conditions
    # ISIC2024 ≈ 141×141 (0.02 MP)
    resolutions = [
        ("Original (~450px)", None),
        ("224px (0.05 MP)",   224),
        ("141px (0.02 MP)\n[ISIC2024 level]", 141),
        ("100px (0.01 MP)",   100),
        ("64px (0.004 MP)",   64),
    ]

    print(f"\n{'='*70}")
    print("RESOLUTION ABLATION STUDY")
    print(f"{'='*70}")

    results = []
    baseline_auc = None

    for label, res in resolutions:
        label_short = label.split('\n')[0].split('(')[0].strip()
        print(f"  {label_short}...", end=" ", flush=True)
        probs = get_probs_resized(model, val_paths, downscale_to=res)
        auc, lo, hi, _ = delong_ci(val_labels, probs)

        if baseline_auc is None:
            baseline_auc = auc
            delta_str = "—"
        else:
            d = auc - baseline_auc
            delta_str = f"{d:+.4f} ({d/baseline_auc*100:+.1f}%)"

        print(f"AUC={auc:.4f}  Δ={delta_str}")
        results.append({"label": label, "res": res, "auc": auc, "lo": lo, "hi": hi})

    # ISIC2024 actual (out-of-distribution)
    isic_actual_auc = 0.5582  # From 06_cross_dataset_eval.py
    isic_res_auc    = results[2]["auc"]  # 141px condition
    total_deg       = baseline_auc - isic_actual_auc
    res_effect      = baseline_auc - isic_res_auc
    art_effect      = isic_res_auc - isic_actual_auc

    print(f"\n{'='*60}")
    print(f"Degradation Decomposition (HAM → ISIC2024):")
    print(f"  Total:              {total_deg:.4f} (100%)")
    print(f"  Resolution-only:    {res_effect:.4f} ({res_effect/total_deg*100:.0f}%)")
    print(f"  Artifact/dist:      {art_effect:.4f} ({art_effect/total_deg*100:.0f}%)")
    print(f"{'='*60}")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Resolution Ablation: Disentangling Resolution vs Artifact Effects",
                 fontsize=13, fontweight='bold')

    labels_short = ["Original", "224px", "141px\n(ISIC2024\nlevel)", "100px", "64px"]
    aucs = [r["auc"] for r in results]
    los  = [r["auc"] - r["lo"] for r in results]
    his  = [r["hi"] - r["auc"] for r in results]
    bar_colors = ['#2ECC71', '#F39C12', '#E67E22', '#E74C3C', '#8E44AD']

    ax = axes[0]
    x  = np.arange(len(aucs))
    bars = ax.bar(x, aucs, color=bar_colors, alpha=0.85, edgecolor='black', width=0.6)
    ax.errorbar(x, aucs, yerr=[los, his], fmt='none',
                color='black', capsize=5, capthick=1.5)
    ax.axhline(isic_actual_auc, color='red', ls='--', lw=2, alpha=0.8,
               label=f'ISIC2024 actual ({isic_actual_auc})')
    ax.axhline(baseline_auc, color='green', ls='--', lw=1.5, alpha=0.6,
               label=f'HAM original ({baseline_auc:.4f})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, fontsize=9)
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0.45, 1.02)
    ax.set_title("AUC vs Downsampling Level")
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{auc:.4f}', ha='center', fontsize=9, fontweight='bold')

    ax2 = axes[1]
    cats   = ['Total\nDegradation', 'Resolution\nEffect', 'Artifact+Distribution\nEffect']
    vals   = [total_deg, res_effect, art_effect]
    colors = ['#E74C3C', '#F39C12', '#9B59B6']
    bars2  = ax2.bar(cats, vals, color=colors, alpha=0.85, edgecolor='black', width=0.5)
    ax2.set_ylabel("AUC Degradation (absolute)")
    ax2.set_title("Degradation Decomposition")
    ax2.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars2, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f'{val:.4f}\n({val/total_deg*100:.0f}%)',
                 ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig("/kaggle/working/fig_ablation_resolution.png",
                dpi=200, bbox_inches='tight')
    print("✅ fig_ablation_resolution.png saved")


if __name__ == "__main__":
    main()
