"""
02_artifact_detection.py
Morphological artifact detection across 5 datasets.
Produces: fig2_artifact_prevalence.png, fig3_artifact_examples.png
"""

import os, glob, random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from config import PATHS, ARTIFACT_CONFIG, RANDOM_SEED

random.seed(RANDOM_SEED)

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

C = ARTIFACT_CONFIG


def detect_artifacts(img_path):
    """Detect 5 artifact types using morphological image processing."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    results = {}

    # 1. Hair — black-hat morphological transform
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                        (C["hair_kernel_size"], C["hair_kernel_size"]))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, C["hair_threshold"], 255, cv2.THRESH_BINARY)
    results["hair"] = int((np.sum(hair_mask > 0) / hair_mask.size) > C["hair_min_area_pct"])

    # 2. Ruler — Hough line transform
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=C["ruler_hough_threshold"],
                             minLineLength=C["ruler_min_line_length"],
                             maxLineGap=10)
    results["ruler"] = int(lines is not None and len(lines) > C["ruler_min_lines"])

    # 3. Vignette — center vs border brightness
    center = gray[h // 4:3 * h // 4, w // 4:3 * w // 4]
    border = np.concatenate([
        gray[:h // 8, :].flatten(), gray[-h // 8:, :].flatten(),
        gray[:, :w // 8].flatten(), gray[:, -w // 8:].flatten()
    ])
    results["vignette"] = int((np.mean(center) - np.mean(border)) > C["vignette_threshold"])

    # 4. Frame — dark corners
    corners = [gray[:20, :20], gray[:20, -20:], gray[-20:, :20], gray[-20:, -20:]]
    results["frame"] = int(
        np.mean([np.mean(c) for c in corners]) < C["frame_corner_threshold"])

    # 5. Marker — blue ink detection in HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(
        hsv,
        (C["marker_hue_lo"], 50, 50),
        (C["marker_hue_hi"], 255, 255)
    )
    results["marker"] = int(
        (np.sum(blue_mask > 0) / blue_mask.size) > C["marker_min_area_pct"])

    return results


def compute_prevalence(paths, n=300, seed=RANDOM_SEED):
    random.seed(seed)
    sample = random.sample(paths, min(n, len(paths)))
    artifacts = ["hair", "ruler", "vignette", "frame", "marker"]
    counts = {a: 0 for a in artifacts}
    valid = 0
    for p in sample:
        res = detect_artifacts(p)
        if res:
            valid += 1
            for a in artifacts:
                counts[a] += res[a]
    return {a: counts[a] / valid * 100 for a in artifacts}


def main():
    datasets = ["HAM10000", "BCN20000", "ISIC2024", "ISIC2018", "PAD_UFES"]
    artifacts = ["hair", "ruler", "vignette", "frame", "marker"]
    colors_art = {"hair": "#3498DB", "ruler": "#E74C3C", "vignette": "#2ECC71",
                  "frame": "#F39C12", "marker": "#9B59B6"}

    print("Computing artifact prevalence (n=300 per dataset)...")
    prev = {}
    for ds in datasets:
        print(f"  {ds}...")
        prev[ds] = compute_prevalence(IMG_PATHS[ds])
        for a in artifacts:
            print(f"    {a}: {prev[ds][a]:.1f}%")

    # ── Fig 2: Prevalence ──────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Clinical Artifact Prevalence Across Datasets (n=300 per dataset)",
                 fontsize=14, fontweight='bold')

    # Grouped bar chart
    ax = axes[0]
    x = np.arange(len(datasets))
    w = 0.15
    for i, (art, col) in enumerate(colors_art.items()):
        vals = [prev[ds][art] for ds in datasets]
        ax.bar(x + i * w, vals, w, label=art.capitalize(),
               color=col, alpha=0.85, edgecolor='black')
    ax.set_xticks(x + 2 * w)
    ax.set_xticklabels(datasets, rotation=15, ha='right')
    ax.set_ylabel("Prevalence (%)")
    ax.set_title("Artifact Prevalence by Dataset and Type")
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    # Heatmap
    ax2 = axes[1]
    matrix = np.array([[prev[ds][a] for a in artifacts] for ds in datasets])
    im = ax2.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
    ax2.set_xticks(range(len(artifacts)))
    ax2.set_xticklabels([a.capitalize() for a in artifacts])
    ax2.set_yticks(range(len(datasets)))
    ax2.set_yticklabels(datasets)
    ax2.set_title("Prevalence Heatmap (%)")
    plt.colorbar(im, ax=ax2, label='Prevalence (%)')
    for i in range(len(datasets)):
        for j in range(len(artifacts)):
            ax2.text(j, i, f"{matrix[i,j]:.0f}",
                     ha='center', va='center', fontsize=10, fontweight='bold',
                     color='white' if matrix[i, j] > 60 else 'black')

    plt.tight_layout()
    plt.savefig("/kaggle/working/fig2_artifact_prevalence.png", dpi=200, bbox_inches='tight')
    print("✅ fig2_artifact_prevalence.png saved")

    # Save results as CSV
    df = pd.DataFrame(prev).T
    df.index.name = "Dataset"
    df.to_csv("/kaggle/working/artifact_prevalence.csv")
    print("✅ artifact_prevalence.csv saved")


if __name__ == "__main__":
    main()
