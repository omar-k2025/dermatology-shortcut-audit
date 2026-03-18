# Shortcut Signals and Dataset Bias in Dermatology AI

**Replication code for:**
> "Shortcut Signals and Dataset Bias in Dermatology AI: A Multi-Dataset Artifact Audit, Counterfactual Evaluation, and Cross-Architecture Generalization Study"  
> *Submitted to npj Digital Medicine, 2026*

---

## Key Findings

- Vignette artifacts show statistically significant co-occurrence with melanoma (OR = 2.74, p = 0.004)
- All three architectures (EfficientNet-B3, ResNet50, ViT-B/16) show severe out-of-distribution degradation (AUC 0.93 → 0.56–0.69)
- Resolution ablation: **83% of ISIC2024 degradation is artifact/distribution-driven**, not resolution
- Counterfactual hair removal causes diagnostic flips in **63.2% of model-predicted malignant cases**
- Attention Rollout reveals ViT-B/16 robustness is **not** explained by reduced artifact focus (negative result)

## Repository Structure

| File | Description |
|------|-------------|
| `config.py` | Global config: seeds, paths, hyperparameters |
| `requirements.txt` | Python dependencies |
| `01_dataset_analysis.py` | Visual distribution analysis (Fig 1) |
| `02_artifact_detection.py` | Morphological artifact detection (Fig 2–3) |
| `03_model_training.py` | EfficientNet-B3, ResNet50, ViT-B/16 training |
| `04_gradcam_analysis.py` | Grad-CAM + artifact focus scoring (Fig 4) |
| `05_counterfactual.py` | Hair removal counterfactual (Fig 5, Table 5) |
| `06_cross_dataset_eval.py` | Multi-model cross-dataset AUC (Fig 6, Table 6) |
| `07_attention_rollout.py` | ViT Attention Rollout vs Grad-CAM (Fig 7) |
| `08_ablation_resolution.py` | Resolution ablation study (Fig 8) |
| `09_artifact_cooccurrence.py` | Artifact-malignancy Fisher's exact tests |
| `utils/delong.py` | DeLong AUC CI and two-sample test |
| `utils/bootstrap.py` | Bootstrap ΔAUC confidence intervals |

## Datasets

All datasets are publicly available on Kaggle:

| Dataset | n | Kaggle Path |
|---------|---|-------------|
| HAM10000 | 10,015 | `kmader/skin-cancer-mnist-ham10000` |
| ISIC2024 | 401,059 | `isic-2024-challenge` |
| BCN20000 | 18,946 | `radwahashiesh/bcn20000` |
| ISIC2018 | 2,594 | `tschandl/isic2018-challenge-task1-data-segmentation` |
| PAD-UFES-20 | 2,298 | `mahdavi1202/skin-cancer` |

## Reproducibility

All experiments use a fixed random seed (`seed = 42`).

- **Hardware:** NVIDIA Tesla P100 PCIe 16 GB
- **Framework:** PyTorch 2.9, Python 3.12
- **Training:** 7 epochs, batch size 32, Adam lr=1e-4

## Installation & Usage
```bash
pip install -r requirements.txt

# Run in order:
python 01_dataset_analysis.py
python 02_artifact_detection.py
python 03_model_training.py
python 04_gradcam_analysis.py
python 05_counterfactual.py
python 06_cross_dataset_eval.py
python 07_attention_rollout.py
python 08_ablation_resolution.py
python 09_artifact_cooccurrence.py
```

## License

MIT — see `LICENSE` file.
## Citation

If you use this code, please cite:
```bibtex
@article{karakoyun2026shortcut,
  title={Shortcut Signals and Dataset Bias in Dermatology AI: A Multi-Dataset Artifact Audit, Counterfactual Evaluation, and Cross-Architecture Generalization Study},
  author={Karakoyun, {\"O}mer},
  journal={npj Digital Medicine},
  year={2026},
  publisher={Nature Publishing Group},
  url={https://github.com/omar-k2025/dermatology-shortcut-audit}
}
```

## Contact

**Dr. Ömer Karakoyun**  
Department of Dermatology and Venereology  
Gazi Yaşargil Training and Research Hospital  
Diyarbakır, Türkiye  
📧 omerkarakoyun@gmail.com  
🔗 ORCID: [0009-0008-8196-7470](https://orcid.org/0009-0008-8196-7470)
