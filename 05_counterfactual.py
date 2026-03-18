"""
05_counterfactual.py
Counterfactual hair removal experiment (n=100).
Produces: fig5_counterfactual.png, fig_waterfall_counterfactual.png
"""

import os, glob, random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr
from config import PATHS, MODEL_PATHS, IMAGENET_MEAN, IMAGENET_STD, RANDOM_SEED, EVAL_CONFIG

random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def detect_hair(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return False, None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    has_hair = (np.sum(mask > 0) / mask.size) > 0.02
    return has_hair, mask, img


def remove_hair(img, mask):
    return cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def get_prob(model, img_array):
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    t = val_transform(Image.fromarray(img_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.sigmoid(model(t)).item()


def main():
    # Load model
    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model.load_state_dict(torch.load(MODEL_PATHS["EfficientNet_B3"], map_location=device))
    model = model.to(device).eval()

    # Find hair-positive images
    ham_images = (
        glob.glob(f"{PATHS['HAM10000']}/HAM10000_images_part_1/*.jpg") +
        glob.glob(f"{PATHS['HAM10000']}/HAM10000_images_part_2/*.jpg")
    )
    print("Finding hair-positive images (n=100)...")
    random.shuffle(ham_images)
    hair_samples = []
    for p in ham_images:
        has_hair, _, _ = detect_hair(p)
        if has_hair:
            hair_samples.append(p)
        if len(hair_samples) >= 100:
            break
    print(f"Found: {len(hair_samples)}")

    # Counterfactual analysis
    print("Running counterfactual analysis...")
    results = []
    for i, path in enumerate(hair_samples):
        has_hair, mask, img = detect_hair(path)
        if not has_hair or img is None:
            continue
        img_clean = remove_hair(img, mask)
        p_orig  = get_prob(model, img)
        p_clean = get_prob(model, img_clean)
        shift   = p_orig - p_clean
        results.append({
            "path": path,
            "prob_orig":  p_orig,
            "prob_clean": p_clean,
            "shift":      shift,
            "abs_shift":  abs(shift),
            "significant": abs(shift) > EVAL_CONFIG["sig_shift_threshold"],
            "reversal":   (p_orig >= 0.5) != (p_clean >= 0.5),
        })
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/100 done")

    shifts = [r["shift"] for r in results]
    probs  = [r["prob_orig"] for r in results]

    # Flip analysis
    mal_to_ben = [r for r in results if r["prob_orig"] >= 0.5 and r["prob_clean"] < 0.5]
    ben_to_mal = [r for r in results if r["prob_orig"] < 0.5  and r["prob_clean"] >= 0.5]
    predicted_mal = [r for r in results if r["prob_orig"] >= 0.5]

    print(f"\n{'='*55}")
    print(f"COUNTERFACTUAL RESULTS (n={len(results)})")
    print(f"{'='*55}")
    print(f"Mean shift:        {np.mean(shifts):+.4f} ± {np.std(shifts):.4f}")
    print(f"Max shift:         {max(shifts):+.4f}")
    print(f"Positive shifts:   {sum(s>0 for s in shifts)}/100")
    print(f"Significant:       {sum(r['significant'] for r in results)}/100")
    print(f"Total flips:       {len(mal_to_ben)+len(ben_to_mal)}/100")
    print(f"Mal→Ben flips:     {len(mal_to_ben)}/100")
    if predicted_mal:
        print(f"Flip in pred-mal:  {len(mal_to_ben)}/{len(predicted_mal)} "
              f"({len(mal_to_ben)/len(predicted_mal)*100:.1f}%)")
    r_val, p_val = pearsonr(probs, shifts)
    print(f"Pearson r:         {r_val:.3f}, p={p_val:.4f}")
    print(f"{'='*55}")

    # Waterfall plot
    sorted_data   = sorted(zip(shifts, probs), key=lambda x: x[0], reverse=True)
    sorted_shifts = [d[0] for d in sorted_data]
    sorted_probs  = [d[1] for d in sorted_data]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Counterfactual Hair Removal: Prediction Shift Analysis (n={len(results)})",
                 fontsize=14, fontweight='bold')

    ax = axes[0]
    bar_colors = []
    sig_thresh = EVAL_CONFIG["sig_shift_threshold"]
    for s in sorted_shifts:
        if abs(s) <= sig_thresh:
            bar_colors.append('#95A5A6')
        elif s > 0:
            bar_colors.append('#E74C3C')
        else:
            bar_colors.append('#3498DB')

    ax.bar(range(len(sorted_shifts)), sorted_shifts, color=bar_colors,
           alpha=0.85, edgecolor='black', lw=0.3, width=1.0)
    ax.axhline(0,          color='black', lw=1.2)
    ax.axhline(sig_thresh, color='#E74C3C', lw=1, ls='--', alpha=0.5)
    ax.axhline(-sig_thresh, color='#3498DB', lw=1, ls='--', alpha=0.5)
    ax.set_xlabel("Image (sorted by shift magnitude)")
    ax.set_ylabel("Prediction Shift (Δ = P_orig − P_clean)")
    ax.set_title("Waterfall Plot: Hair Artifact Impact", fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    legend_patches = [
        mpatches.Patch(color='#E74C3C', alpha=0.85, label='Hair inflated (Δ>0.05)'),
        mpatches.Patch(color='#3498DB', alpha=0.85, label='Hair suppressed (Δ<-0.05)'),
        mpatches.Patch(color='#95A5A6', alpha=0.85, label='Insignificant (|Δ|≤0.05)'),
    ]
    ax.legend(handles=legend_patches, fontsize=9)

    textstr = (f"Mean={np.mean(shifts):+.3f}±{np.std(shifts):.3f}\n"
               f"Significant: {sum(abs(s)>sig_thresh for s in shifts)}/100\n"
               f"Reversals: {len(mal_to_ben)+len(ben_to_mal)}/100")
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=9,
            va='bottom', bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))

    ax2 = axes[1]
    colors2 = ['#E74C3C' if p >= 0.5 else '#3498DB' for p in probs]
    ax2.scatter(probs, shifts, c=colors2, alpha=0.7, s=60,
                edgecolors='black', lw=0.5)
    ax2.axhline(0, color='black', lw=1)
    ax2.axvline(0.5, color='purple', ls=':', alpha=0.5)
    ax2.set_xlabel("Original Predicted Probability")
    ax2.set_ylabel("Prediction Shift (Δ)")
    ax2.set_title(f"Prob vs Shift (r={r_val:.3f}, p={p_val:.3f})", fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/kaggle/working/fig_waterfall_counterfactual.png",
                dpi=200, bbox_inches='tight')
    print("✅ fig_waterfall_counterfactual.png saved")


if __name__ == "__main__":
    main()
