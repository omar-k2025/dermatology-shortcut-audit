"""
04_gradcam_analysis.py
Grad-CAM explainability + quantitative artifact focus scoring.
Produces: fig4_gradcam.png, fig_gradcam_quantification.png
"""

import os, glob, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from config import PATHS, MODEL_PATHS, IMAGENET_MEAN, IMAGENET_STD, RANDOM_SEED

random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, m, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, m, gi, go):
        self.gradients = go[0].detach()

    def generate(self, img_tensor):
        self.model.zero_grad()
        out = self.model(img_tensor)
        out[0, 0].backward()
        w   = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = F.relu((w * self.activations).sum(1, keepdim=True))
        cam = F.interpolate(cam, (224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def detect_hair(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return False, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    has_hair = (np.sum(mask > 0) / mask.size) > 0.02
    return has_hair, mask


def artifact_focus_score(cam, hair_mask):
    """Proportion of CAM activation on hair artifact regions."""
    hair_r = cv2.resize(hair_mask, (224, 224)) > 0
    return float(cam[hair_r].sum() / (cam.sum() + 1e-8))


def main():
    # Load model
    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model.load_state_dict(torch.load(MODEL_PATHS["EfficientNet_B3"], map_location=device))
    model = model.to(device).eval()
    gradcam = GradCAM(model, model.features[-1])

    # HAM10000 images
    ham_images = (
        glob.glob(f"{PATHS['HAM10000']}/HAM10000_images_part_1/*.jpg") +
        glob.glob(f"{PATHS['HAM10000']}/HAM10000_images_part_2/*.jpg")
    )
    meta = pd.read_csv(f"{PATHS['HAM10000']}/HAM10000_metadata.csv")
    meta['label'] = (meta['dx'] == 'mel').astype(int)
    id2p = {os.path.basename(p).replace('.jpg', ''): p for p in ham_images}
    meta['path'] = meta['image_id'].map(id2p)
    meta = meta[meta['path'].notna()].copy()

    # Sample 50 melanoma + 50 non-melanoma
    random.seed(RANDOM_SEED)
    mel_paths = meta[meta['label'] == 1]['path'].sample(50, random_state=RANDOM_SEED).tolist()
    ben_paths = meta[meta['label'] == 0]['path'].sample(50, random_state=RANDOM_SEED).tolist()
    all_paths = mel_paths + ben_paths
    all_labels = [1] * 50 + [0] * 50

    print("Computing Grad-CAM artifact focus scores (n=100)...")
    results = []
    for i, (path, label) in enumerate(zip(all_paths, all_labels)):
        img = cv2.imread(str(path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = val_transform(Image.fromarray(img_rgb)).unsqueeze(0).to(device)

        try:
            cam = gradcam.generate(t)
        except Exception:
            continue

        has_hair, hair_mask = detect_hair(path)
        focus = artifact_focus_score(cam, hair_mask) if hair_mask is not None else 0.0

        with torch.no_grad():
            prob = torch.sigmoid(model(t)).item()

        results.append({
            "path": path, "label": label, "has_hair": has_hair,
            "focus_score": focus, "prob": prob,
        })
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/100 done")

    df = pd.DataFrame(results)

    # ── Statistics ─────────────────────────────
    print(f"\n{'='*55}")
    print("GRAD-CAM QUANTIFICATION RESULTS")
    print(f"{'='*55}")
    print(f"Mean focus (all):    {df['focus_score'].mean():.4f} ± {df['focus_score'].std():.4f}")
    print(f"Mean focus (Hair+):  {df[df['has_hair']]['focus_score'].mean():.4f}")
    print(f"Mean focus (Hair-):  {df[~df['has_hair']]['focus_score'].mean():.4f}")
    print(f"Mean focus (MEL):    {df[df['label']==1]['focus_score'].mean():.4f}")
    print(f"Mean focus (BEN):    {df[df['label']==0]['focus_score'].mean():.4f}")
    print(f"High focus (>0.30):  {(df['focus_score']>0.30).sum()}/100")

    if df['has_hair'].sum() > 0 and (~df['has_hair']).sum() > 0:
        stat, p = mannwhitneyu(
            df[df['has_hair']]['focus_score'],
            df[~df['has_hair']]['focus_score'],
            alternative='greater'
        )
        print(f"Mann-Whitney (Hair+ vs Hair-): U={stat:.1f}, p={p:.4f}")

    # Save results
    df.to_csv("/kaggle/working/gradcam_results.csv", index=False)

    # ── Figure ─────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Grad-CAM Artifact Focus Quantification", fontsize=14, fontweight='bold')

    axes[0].hist(df[df['has_hair']]['focus_score'], bins=20, alpha=0.7,
                 color='#E74C3C', label=f"Hair+ (n={df['has_hair'].sum()})")
    axes[0].hist(df[~df['has_hair']]['focus_score'], bins=20, alpha=0.7,
                 color='#2ECC71', label=f"Hair- (n={(~df['has_hair']).sum()})")
    axes[0].axvline(0.3, color='black', ls='--', alpha=0.7)
    axes[0].set_xlabel("Hair Artifact Focus Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Focus Distribution by Hair Status")
    axes[0].legend()

    axes[1].boxplot([df[df['label']==1]['focus_score'], df[df['label']==0]['focus_score']],
                    labels=['Melanoma', 'Non-melanoma'], patch_artist=True,
                    boxprops=dict(facecolor='#E74C3C', alpha=0.7))
    axes[1].set_ylabel("Hair Artifact Focus Score")
    axes[1].set_title("Focus by Diagnosis")
    axes[1].grid(True, axis='y', alpha=0.3)

    sc = axes[2].scatter(df['prob'], df['focus_score'],
                         c=df['has_hair'].astype(int), cmap='RdYlGn',
                         alpha=0.7, s=60, edgecolors='black', lw=0.5)
    axes[2].set_xlabel("Predicted Malignancy Probability")
    axes[2].set_ylabel("Hair Artifact Focus Score")
    axes[2].set_title("Prob vs Artifact Focus")
    plt.colorbar(sc, ax=axes[2], label='Has Hair')

    plt.tight_layout()
    plt.savefig("/kaggle/working/fig_gradcam_quantification.png", dpi=200, bbox_inches='tight')
    print("✅ fig_gradcam_quantification.png saved")


if __name__ == "__main__":
    main()
