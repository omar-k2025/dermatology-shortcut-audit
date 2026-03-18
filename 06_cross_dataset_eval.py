"""
06_cross_dataset_eval.py
Multi-model cross-dataset generalization evaluation.
DeLong CI + inter-model tests + bootstrap ΔAUC CI.
Produces: fig6_multimodel_crossdataset.png, fig_delong_bcn_analysis.png
"""

import os, glob, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from config import PATHS, MODEL_PATHS, IMAGENET_MEAN, IMAGENET_STD, RANDOM_SEED, EVAL_CONFIG
from utils.delong import delong_ci, delong_compare, auc_vs_chance
from utils.bootstrap import bootstrap_delta_auc_ci
import matplotlib.pyplot as plt

random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_model(model_type, path):
    if model_type == "eff":
        m = models.efficientnet_b3(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    elif model_type == "res":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 1)
    elif model_type == "vit":
        m = models.vit_b_16(weights=None)
        m.heads.head = nn.Linear(m.heads.head.in_features, 1)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.to(device).eval()


def get_probs(model, paths):
    probs = []
    for p in paths:
        try:
            img = Image.open(p).convert('RGB')
            t   = val_transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                probs.append(torch.sigmoid(model(t)).item())
        except Exception:
            probs.append(0.5)
    return probs


def setup_datasets():
    """Load all dataset paths and labels."""
    # HAM10000 val set
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

    # ISIC2024
    isic_images = glob.glob(f"{PATHS['ISIC2024']}/train-image/image/*.jpg")
    isic_meta   = pd.read_csv(f"{PATHS['ISIC2024']}/train-metadata.csv", low_memory=False)
    isic_id2p   = {os.path.basename(p).replace('.jpg', ''): p for p in isic_images}
    isic_v      = isic_meta[isic_meta['isic_id'].isin(isic_id2p)].copy()
    isic_v['path'] = isic_v['isic_id'].map(isic_id2p)
    isic_sample = pd.concat([
        isic_v[isic_v['target'] == 1].sample(EVAL_CONFIG["isic2024_pos"], random_state=RANDOM_SEED),
        isic_v[isic_v['target'] == 0].sample(EVAL_CONFIG["isic2024_neg"], random_state=RANDOM_SEED),
    ])

    # PAD-UFES
    pad_images = (
        glob.glob(f"{PATHS['PAD_UFES']}/imgs_part_1/imgs_part_1/*.png") +
        glob.glob(f"{PATHS['PAD_UFES']}/imgs_part_2/imgs_part_2/*.png") +
        glob.glob(f"{PATHS['PAD_UFES']}/imgs_part_3/imgs_part_3/*.png")
    )
    pad_meta = pd.read_csv(f"{PATHS['PAD_UFES']}/metadata.csv")
    pad_meta['label'] = (pad_meta['diagnostic'] == 'MEL').astype(int)
    pid2p = {os.path.basename(p): p for p in pad_images}
    pad_meta['path'] = pad_meta['img_id'].map(pid2p)
    pad_v = pad_meta[pad_meta['path'].notna()].copy()

    # BCN20000 (no labels)
    bcn_images = glob.glob(f"{PATHS['BCN20000']}/*.jpg")
    bcn_sample = random.sample(bcn_images, 300)

    return {
        "ham":  (val_df['path'].tolist(), val_df['label'].tolist()),
        "isic": (isic_sample['path'].tolist(), isic_sample['target'].tolist()),
        "pad":  (pad_v['path'].tolist(), pad_v['label'].tolist()),
        "bcn":  (bcn_sample, None),
    }


def main():
    datasets = setup_datasets()
    model_configs = [
        ("EfficientNet-B3", "eff", MODEL_PATHS["EfficientNet_B3"]),
        ("ResNet50",        "res", MODEL_PATHS["ResNet50"]),
        ("ViT-B/16",        "vit", MODEL_PATHS["ViT_B16"]),
    ]

    # Collect predictions
    all_probs = {}
    for mname, mtype, mpath in model_configs:
        print(f"\n{mname} predictions...")
        model = load_model(mtype, mpath)
        all_probs[mname] = {}
        for ds_key, (paths, _) in datasets.items():
            all_probs[mname][ds_key] = get_probs(model, paths)
            print(f"  {ds_key}: done ({len(paths)} images)")
        del model
        torch.cuda.empty_cache()

    # AUC table
    print(f"\n{'='*90}")
    print(f"{'Model':<18} {'HAM [95%CI]':<30} {'ISIC2024 [95%CI]':<30} {'PAD-UFES [95%CI]'}")
    print("-" * 90)

    auc_table = {}
    for mname, _, _ in model_configs:
        row = {}
        for ds_key in ["ham", "isic", "pad"]:
            paths, labels = datasets[ds_key]
            auc, lo, hi, se = delong_ci(labels, all_probs[mname][ds_key])
            z_c, p_c = auc_vs_chance(auc, se)
            row[ds_key] = {"auc": auc, "lo": lo, "hi": hi, "se": se,
                            "probs": all_probs[mname][ds_key], "labels": labels}
        auc_table[mname] = row
        print(f"{mname:<18} "
              f"{row['ham']['auc']:.4f} [{row['ham']['lo']:.4f}–{row['ham']['hi']:.4f}]   "
              f"{row['isic']['auc']:.4f} [{row['isic']['lo']:.4f}–{row['isic']['hi']:.4f}]   "
              f"{row['pad']['auc']:.4f} [{row['pad']['lo']:.4f}–{row['pad']['hi']:.4f}]")
    print("=" * 90)

    # DeLong inter-model
    print("\nDeLong Inter-Model (HAM val):")
    model_names = [m[0] for m in model_configs]
    pairs = [("EfficientNet-B3", "ResNet50"),
             ("EfficientNet-B3", "ViT-B/16"),
             ("ResNet50", "ViT-B/16")]
    ham_labels = datasets["ham"][1]
    for m1, m2 in pairs:
        a1, a2, z, p = delong_compare(ham_labels,
                                       auc_table[m1]["ham"]["probs"],
                                       auc_table[m2]["ham"]["probs"])
        sig = "✅" if p < 0.05 else "NS"
        print(f"  {m1} vs {m2}: ΔAUC={a1-a2:+.4f}, Z={z:.3f}, p={p:.4f} {sig}")

    # Bootstrap ΔAUC CI
    print("\nBootstrap ΔAUC CI (HAM → ISIC2024):")
    for mname, _, _ in model_configs:
        d, lo, hi = bootstrap_delta_auc_ci(
            datasets["ham"][1], auc_table[mname]["ham"]["probs"],
            datasets["isic"][1], auc_table[mname]["isic"]["probs"],
            n_boot=EVAL_CONFIG["bootstrap_n"]
        )
        print(f"  {mname}: ΔAUC={d:.4f} [95%CI: {lo:.4f}–{hi:.4f}]")

    # Figure
    colors_m = {"EfficientNet-B3": "#E74C3C", "ResNet50": "#3498DB", "ViT-B/16": "#2ECC71"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle("Cross-Dataset Generalization: Shortcut Learning Evidence",
                 fontsize=14, fontweight='bold')

    ds_keys   = ["ham", "isic", "pad"]
    ds_labels = ["HAM10000\n(in-dist)", "ISIC2024\n(out-dist)", "PAD-UFES\n(out-dist)"]

    for col, (dk, dl) in enumerate(zip(ds_keys, ds_labels)):
        ax = axes[col]
        x  = np.arange(len(model_names))
        aucs = [auc_table[m][dk]["auc"] for m in model_names]
        los  = [auc_table[m][dk]["auc"] - auc_table[m][dk]["lo"] for m in model_names]
        his  = [auc_table[m][dk]["hi"] - auc_table[m][dk]["auc"] for m in model_names]
        bars = ax.bar(x, aucs, color=[colors_m[m] for m in model_names],
                      alpha=0.85, edgecolor='black', width=0.55)
        ax.errorbar(x, aucs, yerr=[los, his], fmt='none',
                    color='black', capsize=5, capthick=1.5)
        ax.axhline(0.5, color='red', ls='--', alpha=0.4)
        ax.set_ylim((0.70, 1.02) if col == 0 else (0.40, 1.02))
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel("AUC-ROC")
        ax.set_title(dl, fontweight='bold',
                     color="#2ECC71" if col == 0 else "#E74C3C")
        ax.grid(True, axis='y', alpha=0.3)
        for bar, auc in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{auc:.4f}', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig("/kaggle/working/fig6_multimodel_crossdataset.png",
                dpi=200, bbox_inches='tight')
    print("✅ fig6_multimodel_crossdataset.png saved")


if __name__ == "__main__":
    main()
