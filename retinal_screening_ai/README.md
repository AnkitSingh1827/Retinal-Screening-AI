# AI-Based Retinal Imaging and Ophthalmic Screening System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit)
![Dataset](https://img.shields.io/badge/Dataset-ODIR--5K-green)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

**B.Tech Final Year Project**  
*AI-Assisted Retinal Image Screening Using Deep Learning and Transfer Learning*

</div>

---

> ⚠️ **Medical Disclaimer** — This application is an AI-assisted screening prototype
> developed for educational and research purposes. It does **not** provide a medical
> diagnosis. All results must be reviewed by a qualified, licensed ophthalmologist
> before any clinical decision is made.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Proposed Solution](#proposed-solution)
4. [Features](#features)
5. [Dataset](#dataset)
6. [Project Structure](#project-structure)
7. [Model Architecture](#model-architecture)
8. [Installation](#installation)
9. [Step-by-Step Workflow](#step-by-step-workflow)
10. [Google Colab Training](#google-colab-training)
11. [Grad-CAM Explainability](#grad-cam-explainability)
12. [Evaluation Metrics](#evaluation-metrics)
13. [Limitations](#limitations)
14. [Future Scope](#future-scope)
15. [Medical Disclaimer](#medical-disclaimer)

---

## 🔭 Project Overview

This system performs automated first-level screening of retinal fundus photographs using
deep learning. It classifies images into four categories:

| Class | Description |
|---|---|
| ✅ Normal | No sign of the targeted diseases |
| 🩸 Diabetes | Diabetic retinopathy indicators |
| 👁 Glaucoma | Optic nerve / intraocular pressure signs |
| 🌅 AMD | Age-related macular degeneration |

A **Grad-CAM** visual explanation is generated for each prediction so clinicians can see
which regions of the retina influenced the result.

---

## 🌍 Problem Statement

- Eye diseases like diabetic retinopathy, glaucoma, and AMD are leading causes of
  preventable blindness worldwide.
- Ophthalmologists are scarce, especially in rural/developing regions.
- Manual screening is slow and costly, delaying early detection.
- High-risk patients are identified too late, when treatment options are limited.

---

## 💡 Proposed Solution

An AI-powered web application that:

1. Accepts a retinal fundus image uploaded by a healthcare worker.
2. Performs image quality validation.
3. Runs the image through a trained **EfficientNetB0** deep learning model.
4. Returns the predicted condition with confidence score and all class probabilities.
5. Generates a **Grad-CAM heatmap** to highlight the influential retinal regions.
6. Provides a screening recommendation with prominent medical disclaimers.

---

## ✨ Features

- 🔬 **Transfer learning** — EfficientNetB0 with ImageNet pretrained weights
- 📊 **Patient-level data splitting** — prevents data leakage between splits
- ⚖️ **Class imbalance handling** — class weights + targeted augmentation
- 🖼 **Retinal-safe augmentation** — small rotations, flips, zoom, brightness
- 🌡 **Grad-CAM explainability** — visual heatmap of influential regions
- 🔍 **Image quality check** — brightness, blur, minimum dimension validation
- ⚠️ **Confidence thresholding** — warns on low-confidence predictions
- 📈 **Full evaluation suite** — accuracy, F1, recall, ROC-AUC, confusion matrix
- 🏥 **Streamlit dashboard** — professional medical UI with 6 tabs
- 💻 **Windows compatible** — tested on Windows + Python 3.12
- ☁️ **Google Colab** training notebook included

---

## 📦 Dataset

**ODIR-5K — Ocular Disease Intelligent Recognition**

- **Source**: [bumbledeep/odir on Hugging Face](https://huggingface.co/datasets/bumbledeep/odir)
- **Patients**: 5,000
- **Images**: Bilateral fundus photographs (left + right eye)
- **Labels**: Free-text diagnostic keywords per patient

### Dataset Columns

| Column | Type | Description |
|---|---|---|
| `patient_id` | Int64 | Unique patient identifier |
| `age` | Int64 | Patient age in years |
| `sex` | Text | Patient sex |
| `label` | Text | Free-text diagnostic label |
| `label_code` | Int64 | Numeric label code |
| `image` | ImageObject | PIL Image (fundus photograph) |

### Multi-Label Handling Policy

The original ODIR dataset contains **free-text labels** that can describe multiple
diseases simultaneously (e.g., "diabetes, glaucoma"). This project uses a transparent
**keyword-matching + exclusion policy**:

1. Each label is scanned for disease keywords (case-insensitive substring match).
2. Records matching **exactly one** target class → included.
3. Records matching **multiple** target classes → **excluded** from the four-class prototype and reported.
4. Records matching **no** target class → excluded.

This decision is documented in `reports/metrics/labeling_stats.json` and
`notebooks/01_dataset_exploration.ipynb`.

---

## 🗂 Project Structure

```
retinal_screening_ai/
│
├── app.py                        ← Streamlit web application
├── requirements.txt              ← Python dependencies
├── README.md                     ← This file
├── .gitignore
│
├── config/
│   └── config.py                 ← All configuration (paths, hyperparameters)
│
├── data/
│   ├── raw/                      ← (empty — dataset downloaded from HuggingFace)
│   ├── processed/                ← (reserved for future processed files)
│   └── splits/
│       ├── full_dataset.csv      ← Labelled metadata (generated)
│       ├── train.csv             ← Training split indices
│       ├── val.csv               ← Validation split indices
│       └── test.csv              ← Test split indices
│
├── models/
│   ├── best_model.keras          ← Best trained model (generated by train.py)
│   ├── class_names.json          ← Ordered class names
│   └── training_metadata.json   ← Training hyperparameters and history
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── dataset.py                ← HuggingFace dataset loading & inspection
│   ├── preprocessing.py          ← Image preprocessing, splitting, augmentation
│   ├── train.py                  ← Model building and training
│   ├── evaluate.py               ← Evaluation metrics and plots
│   ├── predict.py                ← Single-image inference pipeline
│   ├── explainability.py         ← Grad-CAM implementation
│   └── utils.py                  ← Shared utilities
│
├── assets/
│   └── logo/
│
└── reports/
    ├── figures/
    │   ├── class_distribution.png
    │   ├── training_accuracy.png
    │   ├── training_loss.png
    │   └── confusion_matrix.png
    └── metrics/
        ├── labeling_stats.json
        ├── classification_report.json
        └── evaluation_metrics.json
```

---

## 🏗 Model Architecture

```
Input (224 × 224 × 3)
        │
        ▼
EfficientNetB0 backbone (ImageNet weights)
        │   [Frozen in Phase 1 / Top layers unfrozen in Phase 2]
        ▼
GlobalAveragePooling2D
        │
        ▼
Dense(256, activation='relu')
        │
        ▼
Dropout(0.40)
        │
        ▼
Dense(4, activation='softmax')
        │
        ▼
Predicted Class Probabilities
```

### Training Phases

| Phase | Backbone | Learning Rate | Epochs |
|---|---|---|---|
| 1 — Head Training | Frozen | 1e-3 | 20 |
| 2 — Fine-tuning | Top 20 layers unfrozen | 1e-5 | 10 |

---

## ⚙️ Installation

### Prerequisites

- Python 3.12 (recommended)
- pip
- Windows 10/11 or Linux/macOS
- (Optional) NVIDIA GPU with CUDA for faster training

### Windows Setup

```bash
# 1. Clone / download the project
cd retinal_screening_ai

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment (Windows)
.venv\Scripts\activate

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt
```

### Linux / macOS Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Step-by-Step Workflow

### Step 1 — Inspect the Dataset

```bash
python -m src.dataset
```

This will:
- Download and cache the ODIR-5K dataset from HuggingFace
- Print the schema, sample records, label distribution
- Assign four-class labels and report exclusions
- Save `data/splits/full_dataset.csv`
- Save `reports/figures/class_distribution.png`

### Step 2 — Train the Model

```bash
python -m src.train
```

This will:
- Load the HuggingFace dataset
- Split by patient (70/15/15)
- Build EfficientNetB0 with ImageNet weights
- Train Phase 1 (frozen backbone, head only)
- Fine-tune Phase 2 (top layers)
- Save `models/best_model.keras`
- Save training plots to `reports/figures/`

⏱ **Expected time**: ~30–60 min on CPU | ~5–15 min on GPU

> 💡 **Tip**: Use Google Colab with GPU for faster training. See the [Colab section](#google-colab-training).

### Step 3 — Evaluate the Model

```bash
python -m src.evaluate
```

This will:
- Load the saved model and the test split
- Compute accuracy, F1, recall, ROC-AUC, confusion matrix
- Save metrics to `reports/metrics/`
- Save confusion matrix plot to `reports/figures/`

### Step 4 — Run the Streamlit App

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### Step 5 — Predict a Single Image (CLI)

```bash
python -m src.predict path/to/your/fundus_image.jpg
```

---

## ☁️ Google Colab Training

Use the included notebook for GPU-accelerated training in Google Colab:

```
notebooks/03_model_training.ipynb
```

### Colab Workflow

1. Open the notebook in Google Colab
2. Enable GPU: **Runtime → Change runtime type → GPU**
3. Run all cells (installs dependencies, loads dataset, trains, saves model)
4. Download `models/best_model.keras` and `models/class_names.json`
5. Place them in your local `models/` directory
6. Run `streamlit run app.py` locally

### Colab Quick-Start Commands (in a Colab cell)

```python
# Install dependencies
!pip install -q datasets huggingface_hub tensorflow keras opencv-python scikit-learn matplotlib seaborn streamlit plotly

# Clone your repo (if using GitHub)
# !git clone https://github.com/your-repo/retinal_screening_ai

# Or upload the src/ and config/ folders manually, then:
import sys
sys.path.insert(0, '/content/retinal_screening_ai')

from src.train import train
train()
```

---

## 🌡 Grad-CAM Explainability

**Grad-CAM** (Gradient-weighted Class Activation Mapping) uses the gradients of the
predicted class flowing back into the final convolutional layer to produce a heatmap
showing which spatial regions were most important for the prediction.

- 🔴 **Red/Yellow** → High activation (most influential regions)
- 🔵 **Blue** → Low activation (less influential regions)

> ⚠️ Grad-CAM shows model attention — it does **not** prove the disease is present.
> The highlighted regions may correspond to clinically relevant structures
> (optic disc, macula, vasculature) but must be validated by an ophthalmologist.

---

## 📊 Evaluation Metrics

After running `python -m src.evaluate`, the following are reported:

| Metric | Description |
|---|---|
| Accuracy | Overall correct predictions / total |
| Macro Precision | Average precision across all classes |
| **Macro Recall** | **Average recall (sensitivity) — primary metric** |
| Macro F1-Score | Harmonic mean of precision and recall |
| ROC-AUC | Area under ROC curve (one-vs-rest) |
| Confusion Matrix | Per-class prediction breakdown |

> ⚠️ **Why recall matters most**: In medical screening, missing a disease case
> (false negative) is more dangerous than a false alarm (false positive).
> High recall = fewer missed cases.

Metrics are saved to `reports/metrics/evaluation_metrics.json`.

---

## 🔧 Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: datasets` | Package not installed | `pip install datasets` |
| `FileNotFoundError: best_model.keras` | Model not trained | Run `python -m src.train` |
| `CUDA out of memory` | GPU VRAM too small | Reduce `BATCH_SIZE` in `config/config.py` |
| `ValueError: No data in split` | Dataset loading issue | Check internet / re-run `python -m src.dataset` |
| `ImportError: cv2` | OpenCV missing | `pip install opencv-python` |
| Streamlit `ModuleNotFoundError` | Wrong working directory | Run from the project root |
| `AttributeError` on Grad-CAM | Layer name mismatch | Check `GRADCAM_LAYER_NAME` in `config.py` |
| Low dataset size | Multi-label exclusions | Change `MULTI_LABEL_POLICY = "first"` in `config.py` |

---

## ⚠️ Limitations

1. Trained only on ODIR-5K — may not generalise to all fundus camera types.
2. Four-class prototype only — does not detect all possible eye diseases.
3. AI confidence is **not** diagnostic certainty.
4. Poor-quality fundus images may give unreliable predictions.
5. Not validated in a prospective clinical trial.
6. Should always be used as a **support tool**, not a standalone diagnostic system.

---

## 🚀 Future Scope

- Multi-label classification for all 8 ODIR disease categories
- Integration with other retinal imaging modalities (OCT, fluorescein angiography)
- Mobile app deployment (Android / iOS)
- Clinical validation study partnership
- Longitudinal patient tracking
- Integration with hospital EHR systems
- Support for low-bandwidth / offline environments

---

## 📜 Medical Disclaimer

This tool is an AI-assisted screening prototype for **educational and research purposes only**.
It does not provide medical diagnoses, treatment recommendations, or prescriptions.
Retinal images and any AI-generated predictions must be reviewed and interpreted by a
qualified, licensed ophthalmologist before any clinical decision is made.
The developers and institution assume **no medical or legal liability** for outcomes
based on this tool's predictions.

---

*B.Tech Final Year Project — AI-Based Retinal Imaging and Ophthalmic Screening System*
