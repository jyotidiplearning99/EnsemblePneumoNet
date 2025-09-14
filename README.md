# EnsemblePneumoNet
Patch-based residual CNNs and an ensemble pipeline for pneumonia detection on chest X‑rays (CXRs). This repository contains preprocessing utilities, training/evaluation notebooks, and an ensemble model definition.

> **TL;DR:** Preprocess CXRs (NLM → CLAHE → DWT / resizing), train patch‑based CNNs with 5‑fold CV on the Guangzhou pediatric dataset, and evaluate on an external “Pneumonia Analysis (PA)” set; optional S3 data lake utilities are included.

---

## Table of Contents
- [Repo Structure](#repo-structure)
- [Environment & Dependencies](#environment--dependencies)
- [Data & Directory Layout](#data--directory-layout)
- [Preprocessing](#preprocessing)
- [Training & Cross‑Validation](#training--cross-validation)
- [Ensembling & Inference](#ensembling--inference)
- [External Test Set (PA)](#external-test-set-pa)
- [Reproducing Tables & Figures](#reproducing-tables--figures)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Citation](#citation)

---

## Repo Structure

```
EnsemblePneumoNet/
├─ CXR_Model_and_Ensemble_Model/
│  ├─ Chest X Ray Pre Processing.ipynb            # end-to-end preprocessing notebook
│  ├─ Chest X Ray Pneumonia_Model_and_Training.ipynb  # model training / CV
│  ├─ cxr_classify.ipynb                          # example inference/classification
│  └─ ensemble.py                                 # ensemble architecture & helpers
├─ preprocessors/
│  ├─ image_processor.py                          # NLM/BM3D/wavelet/contrast utilities
│  ├─ image_reader.py                             # DICOM/PNG loader & normalization
│  ├─ data_arranger.py                            # splitting & augmentation helpers
│  └─ findings_processor.py                       # label preparation utilities
├─ pipelines/
│  └─ pipelines.py                                # reading → denoise → enhance → resize (+optional S3 upload)
├─ constant/                                      # constants config (column names, params, keys)
│  └─ constants.py
├─ service/
│  └─ createDataLake.py                           # (optional) S3 data lake bootstrap
├─ notebooks/                                     # sandbox/demo notebooks
│  ├─ Sandbox.ipynb
│  └─ dmeo.ipynb
└─ Pneumonia Code v4.0 final .ipynb               # legacy/notebook variant
```

---

## Environment & Dependencies

**Python:** 3.9–3.11 recommended

Install via Conda (recommended):

```bash
conda create -n pneumo python=3.10 -y
conda activate pneumo
pip install -U pip wheel setuptools
pip install tensorflow==2.12.* keras==2.12.*
pip install numpy pandas scikit-learn scikit-image matplotlib seaborn tqdm opencv-python pywavelets pydicom bm3d boto3
```

> **Note:** The code mixes `keras` and `tensorflow.keras` imports. Using **TF 2.12 + Keras 2.12** keeps them compatible. If you hit import issues, replace `from keras...` with `from tensorflow.keras...` in your local copy of the scripts.

GPU (optional): install CUDA/CuDNN versions compatible with your TF build.

---

## Data & Directory Layout

This repo expects two datasets:

1. **Guangzhou pediatric CXR dataset** (often referred to as “Chest X‑Ray Images (Pneumonia)”)
   - Train/Val/Test split structure (after preprocessing):
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

2. **External “Pneumonia Analysis (PA)” dataset** (independent test set you curated)
   ```
   <DATA_ROOT>/Pneumonia_Analysis_Dataset/
   ├─ normal/       *.png|*.jpg
   └─ pneumonia/    *.png|*.jpg
   ```

> **Licensing:** Please acquire datasets from their original sources and follow their licenses/terms. This repository does not redistribute data.

Configure base paths inside notebooks or pass them as variables (see sections below).

---

## Preprocessing

Run the notebook:
```
CXR_Model_and_Ensemble_Model/Chest X Ray Pre Processing.ipynb
```

Pipeline summary:
- **Convert to 8‑bit** for consistent dynamic range.
- **Denoise** (Non‑Local Means; optional BM3D).
- **Contrast-limited adaptive histogram equalization (CLAHE)**.
- **DWT** (Haar), retain **LL** band.
- **Resize** to **64×64** for patch‑level networks (or 600×600 before patching for patch extraction), and save to `processed_chest_xray`.

Key functions live in `preprocessors/image_processor.py` and `image_reader.py`. The `pipelines/pipelines.py` file shows how to chain read → denoise → enhance → resize. You can adapt the batch readers there to your storage (local/DICOM/PNG).

**Outputs:** PNG images in the directory layout shown above, preserving class labels.

---

## Training & Cross‑Validation

Run the notebook:
```
CXR_Model_and_Ensemble_Model/Chest X Ray Pneumonia_Model_and_Training.ipynb
```

This notebook trains residual CNNs on **25×25 patches** extracted from **600×600** resized CXRs (24×24 grid → 576 patches/image). It supports **5‑fold stratified CV** on the Guangzhou dataset and saves best models per fold.

**Typical hyperparameters:**
- Optimizer: `Adam(lr=1e-4)`, ReduceLROnPlateau, EarlyStopping
- Batch size: 64
- Epochs: up to 50 (early stopping on val accuracy)
- Loss: categorical cross‑entropy; metrics: accuracy (and AUROC tracked in analysis cells)
- Class weighting if imbalanced

**Artifacts:**
- Fold models in `models/` (e.g., `cnn_model_fold_{k}.keras`)
- Training logs/history per fold
- Optionally confusion matrices and classification reports

---

## Ensembling & Inference

`CXR_Model_and_Ensemble_Model/ensemble.py` defines the ensemble architecture and helper utilities (SeparableConv2D branches + residual blocks; global average pooling → dense). You can either:

1) **Average** per‑patch probabilities across 576 patches (simple aggregation), or  
2) **Use an ensemble head** that fuses multiple backbones (see `ensemble.py`).

Example (sketch):
```python
from tensorflow.keras.models import load_model
import numpy as np

# Load k-fold models
fold_models = [load_model(f"models/cnn_model_fold_{k}.keras") for k in range(1, 6)]

def predict_image_patches(patches):  # patches: (576, 25, 25, 1)
    probs = [m.predict(patches, verbose=0) for m in fold_models]  # list of (576, 2)
    mean_prob_per_patch = np.mean(np.stack(probs, axis=0), axis=0)  # (576, 2)
    image_prob = mean_prob_per_patch.mean(axis=0)                  # (2,)
    return image_prob
```

There is a reference inference notebook:
```
CXR_Model_and_Ensemble_Model/cxr_classify.ipynb
```

---

## External Test Set (PA)

To evaluate on your independent PA dataset, run your adapted evaluation notebook (or replicate the logic from `cxr_classify.ipynb`) using the **same preprocessing** (NLM → CLAHE → DWT → 600×600 → 25×25 patches) and the **saved fold models / ensemble**. Record **Accuracy, Precision, Recall, AUROC**, and the **confusion matrix** at the **image level** (after patch aggregation).

Example directory:
```
<PARENT>/Pneumonia_Analysis_Dataset/{normal,pneumonia}/*.png
```

---

## Reproducing Tables & Figures

- **Supplementary S1–S2 (Ablations):** Patch size (10–64 px) and input resolution (256–768 px) sweeps. Use your ablation notebook to run and export CSVs → plot with matplotlib/seaborn.
- **Cross‑val table:** Export fold‑wise metrics and compute mean ± SD.
- **PA confusion matrix:** Save as `Supplementary Figure S1` and reference in the manuscript.

> If you need automation, place small helper scripts under `notebooks/` or add a `scripts/` folder with CLIs that wrap the notebook code.

---

## Configuration

Some utilities use `configparser` for local/S3 paths. If you use S3 (optional), create a config like:

```ini
# s3.cfg
[S3]
s3_bucket = your-bucket-name
s3_image_prefix = cxr/images/
```

And pass it to the upload pipeline in `pipelines/pipelines.py` (requires valid AWS credentials/profile).

---

## Troubleshooting

- **`ImportError: cannot import name '...' from keras`** → Ensure `tensorflow==2.12.*` and `keras==2.12.*`. If issues persist, change `from keras...` to `from tensorflow.keras...` or install `pip install keras==2.12.*` explicitly.
- **BM3D not found** → `pip install bm3d`. You can disable BM3D and rely on NLM (`cv2.fastNlMeansDenoising`) if needed.
- **CUDA not detected** → Training will fall back to CPU. Install CUDA/CuDNN versions matching your TensorFlow build.
- **DICOM reading errors** → Ensure `pydicom` installed and paths point to actual files; otherwise switch to PNG/JPG with `PNGReader` in `image_reader.py`.

---

## License

This repository’s code is provided **for research purposes**. Check dataset licenses before use. If you intend to use this code commercially, please contact the authors/owners.

---

## Citation

If you use this repository or its ideas in your research, please cite the manuscript when available. For now, you may cite this codebase as:

```
@software{ensemblepneumonet,
  title = {EnsemblePneumoNet: Patch-based Residual CNNs for Pneumonia Detection on CXRs},
  author = {Your Author List},
  year = {2025},
  url = {<repo-url>},
  note = {Version 2025-09-14}
}
```
