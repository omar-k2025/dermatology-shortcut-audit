"""
01_dataset_analysis.py
Visual distribution analysis across 5 datasets.
Produces: fig1_dataset_bias.png
"""

import os, glob, random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from config import PATHS, RANDOM_SEED

random.seed(RANDOM_SEED)

# ── Image paths ───────────────────────────────
IMG_PATHS = {
    "HAM10000": (
        glob.glob(f"{PATHS['HAM10000']}/HAM10000_images_part_1/*.jpg") +
        glob.glob(f"{PATHS['HAM10000']}/HAM10000_images_part_2/*.jpg")
    ),
    "BCN20000": glob.glob(f"{PATHS['BCN20000']}/*.jpg"),
    "ISIC2024": glob.glob(f"{PATHS['ISIC2024']}/train-image/image/*.jpg"),
    "ISIC2018": glob.glob(f"{PATHS['ISIC2018']}/ISIC2018_Task1-2_Training_Input/*.jpg"),
    "PAD_UFES": (
        glob.glob(f"{PATHS['PAD_UFES']}/imgs_part_1/imgs_part_1/*.png") +
        glob.glob(f"{PATHS['PAD_UFES']}/imgs_part_2/imgs_part_2/*.png") +
        glob.glob(f"{PATHS['PAD_UFES']}/imgs_part_3/imgs_part_3/*.png")
    ),
}

DATASET_NAMES = ["HAM10000", "BCN20000", "ISIC2024", "ISIC2018", "PAD_UFES"]
COLORS = ["#3498DB","#E74C3C","#2ECC71","#F39C12","#9B59B6"]
N_SAMPLE = 500


def analyze_dataset(paths, n=N_SAMPLE):
    sample = random.sample(paths, min(n, len(paths)))
    brightness, resolutions, r_vals, g_vals, b_vals = [], [], [], [], []
    for p in sample:
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness.append(np.mean(gray))
        h, w = img.shape[:2]
        resolutions.append((h * w) / 1e6)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        r_vals.append(np.mean(img_rgb[:,:,0]))
        g_vals.append(np.mean(img_rgb[:,:,1]))
        b_vals.append(np.mean(img_rgb[:,:,2]))
    return {
        "brightness": brightness,
        "resolution": resolutions,
        "R": r_vals, "G": g_vals, "B": b_vals,
    }


def main():
    print("Analyzing datasets (n=500 each)...")
    stats = {}
    for name in DATASET_NAMES:
        print(f"  {name}...")
        stats[name] = analyze_dataset(IMG_PATHS[name])

    # ── Figure ──────────────────────────────────
    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
    fig.suptitle("Dataset Visual Distribution Analysis (n=500 per dataset)",
                 fontsize=16, fontweight='bold')

    # Brightness histograms
    ax1 = fig.add_subplot(gs[0, :2])
    for name, col in zip(DATASET_NAMES, COLORS):
        ax1.hist(stats[name]["brightness"], bins=40, alpha=0.6,
                 label=name, color=col, density=True)
    ax1.set_xlabel("Mean Brightness"); ax1.set_ylabel("Density")
    ax1.set_title("Brightness Distribution"); ax1.legend(fontsize=8)

    # Resolution boxplot
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.boxplot([stats[n]["resolution"] for n in DATASET_NAMES],
                labels=DATASET_NAMES, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax2.set_ylabel("Resolution (MP)")
    ax2.set_title("Spatial Resolution")
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)

    # RGB bar chart
    ax3 = fig.add_subplot(gs[1, :])
    x = np.arange(len(DATASET_NAMES)); w = 0.25
    ax3.bar(x - w, [np.mean(stats[n]["R"]) for n in DATASET_NAMES], w,
            label='R', color='#E74C3C', alpha=0.85, edgecolor='black')
    ax3.bar(x,     [np.mean(stats[n]["G"]) for n in DATASET_NAMES], w,
            label='G', color='#2ECC71', alpha=0.85, edgecolor='black')
    ax3.bar(x + w, [np.mean(stats[n]["B"]) for n in DATASET_NAMES], w,
            label='B', color='#3498DB', alpha=0.85, edgecolor='black')
    ax3.set_xticks(x); ax3.set_xticklabels(DATASET_NAMES)
    ax3.set_ylabel("Mean Channel Value"); ax3.set_title("RGB Channel Distribution")
    ax3.legend(); ax3.grid(True, axis='y', alpha=0.3)

    # Summary table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    rows = []
    for name in DATASET_NAMES:
        s = stats[name]
        rows.append([
            name,
            f"{np.mean(s['brightness']):.1f} ± {np.std(s['brightness']):.1f}",
            f"{np.mean(s['resolution']):.2f} MP",
            f"{np.mean(s['R']):.0f}",
            f"{np.mean(s['G']):.0f}",
            f"{np.mean(s['B']):.0f}",
        ])
    tbl = ax4.table(
        cellText=rows,
        colLabels=["Dataset", "Brightness (mean±SD)", "Resolution", "R", "G", "B"],
        loc='center', cellLoc='center'
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    tbl.scale(1, 1.8)
    ax4.set_title("Summary Statistics", fontweight='bold', pad=20)

    plt.savefig("/kaggle/working/fig1_dataset_bias.png", dpi=200, bbox_inches='tight')
    print("✅ fig1_dataset_bias.png saved")

    # Print summary
    print("\nSummary:")
    for name in DATASET_NAMES:
        s = stats[name]
        print(f"  {name}: brightness={np.mean(s['brightness']):.1f}±{np.std(s['brightness']):.1f} "
              f"res={np.mean(s['resolution']):.2f}MP "
              f"RGB=({np.mean(s['R']):.0f}/{np.mean(s['G']):.0f}/{np.mean(s['B']):.0f})")


if __name__ == "__main__":
    main()
