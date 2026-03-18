"""
09_artifact_cooccurrence.py
Artifact-malignancy co-occurrence analysis in HAM10000.
Chi-square + Fisher's exact tests.
Produces: artifact_cooccurrence_results.csv
"""

import os, glob, random
import numpy as np
import pandas as pd
import cv2
from scipy.stats import chi2_contingency, fisher_exact
import matplotlib.pyplot as plt
from config import PATHS, ARTIFACT_CONFIG, RANDOM_SEED

random.seed(RANDOM_SEED)
C = ARTIFACT_CONFIG


def detect_artifacts(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    results = {}

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                        (C["hair_kernel_size"], C["hair_kernel_size"]))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, C["hair_threshold"], 255, cv2.THRESH_BINARY)
    results["hair"] = int((np.sum(hair_mask > 0) / hair_mask.size) > C["hair_min_area_pct"])

    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=C["ruler_hough_threshold"],
                             minLineLength=C["ruler_min_line_length"], maxLineGap=10)
    results["ruler"] = int(lines is not None and len(lines) > C["ruler_min_lines"])

    center = gray[h // 4:3 * h // 4, w // 4:3 * w // 4]
    border = np.concatenate([
        gray[:h // 8, :].flatten(), gray[-h // 8:, :].flatten(),
        gray[:, :w // 8].flatten(), gray[:, -w // 8:].flatten()
    ])
    results["vignette"] = int((np.mean(center) - np.mean(border)) > C["vignette_threshold"])

    corners = [gray[:20, :20], gray[:20, -20:], gray[-20:, :20], gray[-20:, -20:]]
    results["frame"] = int(
        np.mean([np.mean(c) for c in corners]) < C["frame_corner_threshold"])

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv,
                             (C["marker_hue_lo"], 50, 50),
                             (C["marker_hue_hi"], 255, 255))
    results["marker"] = int(
        (np.sum(blue_mask > 0) / blue_mask.size) > C["marker_min_area_pct"])

    return results


def main():
    ham_images = (
        glob.glob(f"{PATHS['HAM10000']}/HAM10000_images_part_1/*.jpg") +
        glob.glob(f"{PATHS['HAM10000']}/HAM10000_images_part_2/*.jpg")
    )
    meta = pd.read_csv(f"{PATHS['HAM10000']}/HAM10000_metadata.csv")
    meta['label'] = (meta['dx'] == 'mel').astype(int)
    id2p = {os.path.basename(p).replace('.jpg', ''): p for p in ham_images}
    meta['path'] = meta['image_id'].map(id2p)
    meta = meta[meta['path'].notna()].copy()

    sample = meta.sample(1000, random_state=RANDOM_SEED)
    print(f"Sample: n={len(sample)}, melanoma={sample['label'].sum()} ({sample['label'].mean()*100:.1f}%)")

    print("Detecting artifacts (n=1000)...")
    artifact_data = []
    for i, (_, row) in enumerate(sample.iterrows()):
        res = detect_artifacts(row['path'])
        if res:
            res['label'] = row['label']
            res['dx']    = row['dx']
            artifact_data.append(res)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/1000 done")

    df = pd.DataFrame(artifact_data)
    artifacts = ["hair", "ruler", "vignette", "frame", "marker"]

    print(f"\n{'='*65}")
    print("ARTIFACT-MALIGNANCY CO-OCCURRENCE (Chi-square + Fisher Exact)")
    print(f"{'='*65}")

    results = []
    for art in artifacts:
        ct = pd.crosstab(df[art], df['label'])
        if ct.shape != (2, 2):
            continue
        ct_arr = ct.values

        chi2, p_chi2, dof, _ = chi2_contingency(ct_arr)
        or_val, p_fisher = fisher_exact(ct_arr)

        art_pos = ct_arr[1].sum()
        art_neg = ct_arr[0].sum()
        mel_rate_pos = ct_arr[1, 1] / art_pos * 100 if art_pos > 0 else 0
        mel_rate_neg = ct_arr[0, 1] / art_neg * 100 if art_neg > 0 else 0

        sig = "✅ p<0.05" if p_fisher < 0.05 else "NS"
        print(f"\n{art.upper()}:")
        print(f"  Art+: {ct_arr[1,1]}/{art_pos} melanoma ({mel_rate_pos:.1f}%)")
        print(f"  Art-: {ct_arr[0,1]}/{art_neg} melanoma ({mel_rate_neg:.1f}%)")
        print(f"  OR={or_val:.3f}, χ²={chi2:.3f}, p={p_chi2:.4f}, Fisher p={p_fisher:.4f}  {sig}")

        results.append({
            "artifact": art,
            "mel_rate_pos": mel_rate_pos,
            "mel_rate_neg": mel_rate_neg,
            "odds_ratio": or_val,
            "chi2": chi2,
            "p_chi2": p_chi2,
            "p_fisher": p_fisher,
            "significant": p_fisher < 0.05,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv("/kaggle/working/artifact_cooccurrence_results.csv", index=False)
    print("\n✅ artifact_cooccurrence_results.csv saved")

    # Figure: vignette highlighted
    sig_art = results_df[results_df['significant']]
    print(f"\nSignificant associations:")
    for _, row in sig_art.iterrows():
        print(f"  {row['artifact']}: OR={row['odds_ratio']:.3f}, p={row['p_fisher']:.4f}")


if __name__ == "__main__":
    main()
