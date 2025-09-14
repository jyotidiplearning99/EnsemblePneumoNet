# EnsemblePneumoNet — Reproducible Pipeline
Patch-based residual CNNs and an ensemble pipeline for pneumonia detection on chest X‑rays (CXRs). This repository contains preprocessing utilities, training/evaluation notebooks, and a final, end‑to‑end notebook to reproduce the paper results.

> **TL;DR:** Preprocess CXRs (NLM → CLAHE → DWT / resizing), extract **25×25** patches from **600×600** CXRs, train patch‑based CNNs with **5‑fold CV** on Guangzhou, and evaluate on an external “Pneumonia Analysis (PA)” set; optional S3 data lake utilities are included.

---

## Table of Contents
- [What’s New](#whats-new)
- [Repo Structure](#repo-structure)
- [Quickstart](#quickstart)
- [Environment & Dependencies](#environment--dependencies)
- [Data & Directory Layout](#data--directory-layout)
- [Preprocessing](#preprocessing)
- [Patch Extraction & Aggregation](#patch-extraction--aggregation)
- [Training & Cross‑Validation](#training--cross-validation)
- [External Test Set (PA)](#external-test-set-pa)
- [Ablations (Patch Size & Resolution)](#ablations-patch-size--resolution)
- [Reproducing Tables & Figures](#reproducing-tables--figures)
- [Headless Execution with Papermill](#headless-execution-with-papermill)
- [Expected Artifacts](#expected-artifacts)
- [Configuration (Optional S3)](#configuration-optional-s3)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Citation](#citation)
- [Changelog](#changelog)

---

## What’s New
- **Final end‑to‑end notebook:** `Pneumonia Code v4.0 final .ipynb` (single entry point).
- **Reproducibility files:** [`environment.yml`](./environment.yml) (conda), [`requirements.txt`](./requirements.txt) (pip).
- **Notebook guide consolidated into README:** run order, parameters, artifacts, papermill usage.
- **Reviewer‑aligned details:** mean±SD reporting, ablations pointers, patch selection rationale, PA evaluation and confusion matrix outputs.

---

## Repo Structure

```
EnsemblePneumoNet/
├─ Pneumonia Code v4.0 final .ipynb               # FINAL end‑to‑end pipeline
├─ CXR_Model_and_Ensemble_Model/
│  ├─ Chest X Ray Pre Processing.ipynb            # preprocessing notebook (optional)
│  ├─ Chest X Ray Pneumonia_Model_and_Training.ipynb  # model training / CV (optional)
│  ├─ cxr_classify.ipynb                          # example inference (optional)
│  └─ ensemble.py                                 # ensemble architecture & helpers
├─ preprocessors/
│  ├─ image_processor.py                          # NLM/BM3D/wavelet/contrast utilities
│  ├─ image_reader.py                             # DICOM/PNG loader & normalization
│  ├─ data_arranger.py                            # splitting & augmentation helpers
│  └─ findings_processor.py                       # label preparation utilities
├─ pipelines/
│  └─ pipelines.py                                # read → denoise → enhance → resize (+optional S3 upload)
├─ constant/
│  └─ constants.py                                # constants & config stubs
├─ service/
│  └─ createDataLake.py                           # (optional) S3 data lake bootstrap
├─ notebooks/                                     # sandbox/demo notebooks
│  ├─ Sandbox.ipynb
│  └─ dmeo.ipynb
├─ environment.yml                                # conda env (TF/Keras 2.12)
├─ requirements.txt                               # pip env
└─ README.md
```

---

## Quickstart

1) **Create env** (Conda recommended):
```bash
conda env create -f environment.yml
conda activate pneumo
# or with pip
pip install -r requirements.txt
```

2) **Organize data** as in [Data & Directory Layout](#data--directory-layout).

3) **Open & run** `Pneumonia Code v4.0 final .ipynb` top‑to‑bottom.  
   - Configure the first “Parameters/Paths” cell.  
   - Run sections in order: Preprocessing → Patch Extraction → CV Training → Aggregation/Inference → PA Evaluation → Ablations (optional).

4) **Find outputs** in `models/`, `results/`, `figures/` (see [Expected Artifacts](#expected-artifacts)).

---

## Environment & Dependencies

**Python:** 3.9–3.11 (tested with 3.10)  
**Deep learning:** TensorFlow 2.12 + Keras 2.12

Install via Conda:
```bash
conda env create -f environment.yml
conda activate pneumo
```

Or via pip:
```bash
pip install -r requirements.txt
```

> **Note:** The code mixes `keras` and `tensorflow.keras`. Pinning **TF/Keras 2.12** avoids import issues. If needed, change `from keras...` to `from tensorflow.keras...` in local copies.

GPU (optional): install CUDA/CuDNN compatible with your TF build.

---

## Data & Directory Layout

Two datasets are expected:

1. **Guangzhou pediatric CXR dataset** (a.k.a. “Chest X‑Ray Images (Pneumonia)”) — after preprocessing:
```
<DATA_ROOT>/Guangzhou_X_ray_dataset/processed_chest_xray/
├─ train/
│  ├─ NORMAL/      *.png
│  └─ PNEUMONIA/   *.png
├─ val/
│  ├─ NORMAL/      *.png
│  └─ PNEUMONIA/   *.png
└─ test/
   ├─ NORMAL/      *.png
   └─ PNEUMONIA/   *.png
```

2. **External “Pneumonia Analysis (PA)” dataset** (independent test set):
```
<DATA_ROOT>/Pneumonia_Analysis_Dataset/
├─ normal/       *.png|*.jpg
└─ pneumonia/    *.png|*.jpg
```

> Acquire datasets from original sources and follow their licenses. This repo does **not** redistribute data.

---

## Preprocessing

**Notebook section:** *Preprocessing* (or `CXR_Model_and_Ensemble_Model/Chest X Ray Pre Processing.ipynb`)

Pipeline:
1. Convert to **8‑bit** (consistent dynamic range).  
2. **Non‑Local Means** denoising (`cv2.fastNlMeansDenoising`), optional **BM3D**.  
3. **CLAHE** (clipLimit=2.0, tileGridSize=8×8).  
4. **DWT (Haar)**, keep **LL** band.  
5. Resize: either **600×600** (for patch extraction), and/or 64×64 for compact patch nets.  
6. Save to `processed_chest_xray` keeping class folders.

> **Consistency:** Apply the **same pipeline** to the PA set before evaluation.

---

## Patch Extraction & Aggregation

**Reviewer‑ready explanation:** We **do not** manually mine “pneumonia‑relevant” patches. Each 600×600 image is partitioned into a **fixed 24×24 grid** of **25×25** patches (**576** patches/image), each inheriting the image label. Patch‑level CNNs are trained on all patches. At inference, we average per‑patch probabilities across 576 patches to get an **image‑level** score, then threshold to label the image.

---

## Training & Cross‑Validation

**Notebook section:** *Training / Cross‑Validation*

- **Model:** residual CNN (per patch) — blocks with BN+ReLU; 32→64→128 filters; GAP → Dense(128) → softmax.  
- **CV:** **5‑fold stratified** on Guangzhou; early stopping + ReduceLROnPlateau.  
- **Typical hyperparams:** `Adam(lr=1e-4)`, batch size 64, up to 50 epochs.

**Reporting:** Save per‑fold metrics and report **mean ± SD** (Accuracy, Precision, Recall; AUROC if computed).

**Artifacts:** `models/cnn_model_fold_{k}.keras`, `results/cv_metrics.csv`.

---

## External Test Set (PA)

**Notebook section:** *External Evaluation (PA)*

- Preprocess PA with **identical** pipeline.  
- Run inference with fold models or ensemble; aggregate per‑patch → image‑level.  
- Save **classification report** and **confusion matrix** (image‑level).

**Artifacts:** `results/pa_report.txt`, `results/pa_metrics.csv`, `figures/pa_confusion_matrix.png`.

---

## Ablations (Patch Size & Resolution)

**Notebook section:** *Ablations* (optional but recommended)

- **Patch size sweep:** 10, 15, 20, 25, 32, 48, 64 px — **25×25** typically best in our runs.  
- **Input resolution sweep:** 256, 600, 768 px — 600×600 often yields strongest AUC/recall trade‑off.

**Artifacts:** `figures/ablation_patchsize.png`, `figures/ablation_resolution.png`, CSV exports.

---

## Reproducing Tables & Figures

- **Cross‑validation mean ± SD** → `results/cv_metrics.csv` → compute mean±SD → manuscript Results + Supplementary (e.g., S3/S4).  
- **Ablations** → Patch size & resolution plots (Supplementary S1–S2).  
- **PA confusion matrix** → Supplementary Figure (e.g., S1).

> Include exact seeds and environment versions in the exported reports for traceability.

---

## Configuration (Optional S3)

Some utilities use `configparser` for local/S3 paths (see `pipelines/pipelines.py`, `service/createDataLake.py`). Example:

```ini
# s3.cfg
[S3]
s3_bucket = bucket-name
s3_image_prefix = cxr/images/
```

Requires valid AWS credentials if used.

---

## Troubleshooting

- **`ImportError: ... from keras`** → Use TF/Keras 2.12, or change `from keras...` → `from tensorflow.keras...`.  
- **BM3D not found** → `pip install bm3d`; or disable and use NLM only.  
- **CUDA not detected** → Falls back to CPU; install driver/CUDA/CuDNN compatible with your TF build.  
- **OOM / slow patch extraction** → Use generators / process in batches; avoid loading all patches into RAM.  
- **Inconsistent preprocessing** → Ensure PA uses identical NLM → CLAHE → DWT(LL) + resize steps.

---

## License

Code is provided **for research purposes**. Respect dataset licenses. For commercial use, contact the authors/owners.

---

## Citation

If you use this repository or its ideas in your research, please cite the manuscript when available. For now:

```
@software{ensemblepneumonet,
  title = {EnsemblePneumoNet: Patch-based Residual CNNs for Pneumonia Detection on CXRs},
  author = {<Author list>},
  year = {2025},
  url = {<repo-url>},
  note = {Version 2025-09-14}
}
```

---

## Changelog
- **2025‑09‑14:** Unified README; added notebook‑centric run guide, env files, papermill usage; clarified patch extraction; added links to ablations & PA outputs.
