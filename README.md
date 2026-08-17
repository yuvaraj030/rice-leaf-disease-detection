# 🌾 PRCP-1001: Rice Leaf Disease Detection

> **Deep Learning Classification System for Rice Leaf Disease Identification**

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Transfer%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

---

## 📋 Project Overview

| Field | Detail |
|---|---|
| **Project ID** | PRCP-1001 |
| **Type** | Classification (Computer Vision) |
| **Student** | Yuvaraj S |
| **Course** | AIE |
| **Team ID** | PTID-AIE-APR-26-11148 |
| **Contribution** | Individual |

Rice is one of the world's most important staple crops — feeding over half the global population. Disease outbreaks cause **significant yield losses every season**. This project builds a deep learning pipeline to **automatically identify three major rice leaf diseases** from field/greenhouse photographs, enabling early intervention before crop losses escalate.

---

## 🎯 Problem Statement

Three classification tasks are addressed:

| Task | Description |
|---|---|
| **Task 1** | Comprehensive data analysis — class balance, image properties, RGB channel characteristics |
| **Task 2** | Multi-class disease classification: Bacterial Leaf Blight, Brown Spot, Leaf Smut |
| **Task 3** | Data augmentation analysis — justify applied transforms and their effect on generalisation |

---

## 🦠 Disease Classes

| Class | Description |
|---|---|
| 🔴 **Bacterial Leaf Blight** | Water-soaked lesions turning yellow then white; caused by *Xanthomonas oryzae* |
| 🟡 **Brown Spot** | Circular brown spots with yellow halos; caused by *Helminthosporium oryzae* |
| 🟢 **Leaf Smut** | Dark, slightly raised angular spots on leaf surface; caused by *Entyloma oryzae* |

---

## 📁 Repository Structure

```
rice-leaf-disease-detection/
│
├── 📓 PRCP-1001-RiceLeaf-Disease-Detection-Executed-Yuvaraj S.ipynb   # Fully executed notebook
│
├── 📂 dataset/
│   └── Data/
│       ├── Bacterial leaf blight/   # 40 JPEG images
│       ├── Brown spot/              # 40 JPEG images
│       └── Leaf smut/               # 39 JPEG images
│
└── 📄 README.md
```

---

## 🗃️ Dataset

| Property | Value |
|---|---|
| **Total Images** | 119 |
| **Classes** | 3 |
| **Distribution** | Bacterial Leaf Blight: 40 · Brown Spot: 40 · Leaf Smut: 39 |
| **Balance Ratio** | 1.03× (approximately balanced) |
| **Format** | JPEG (RGB, 3-channel) |
| **Source** | Official field/greenhouse collection |
| **Integrity** | ✅ All 119 images readable — zero corruption detected |

> **Dataset path:** `dataset/Data/` — each sub-directory corresponds to one disease class.

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `tensorflow` | 2.21.0 | Model building, training, pretrained architectures |
| `keras` | (bundled) | `ImageDataGenerator`, callbacks, layers API |
| `numpy` | 2.4.2 | Array operations |
| `pandas` | latest | Tabular metadata & EDA |
| `opencv-python` | latest | Image I/O, pixel-level analysis |
| `Pillow` | latest | PIL compatibility with Keras utilities |
| `matplotlib` | latest | Visualisations |
| `seaborn` | latest | Statistical charts |
| `scikit-learn` | latest | Train/val/test split, metrics, cross-validation |

---

## 📊 Exploratory Data Analysis

### 2.1 Class Distribution
The dataset is **approximately balanced** (40 / 40 / 39) — no class-weighted loss or oversampling was required.

### 2.2 Image Properties
- **Dimensions** vary across the dataset; all images resized to **224 × 224** before training
- **File sizes** range from small to medium JPEG — no outlier files detected
- All images are **RGB (3-channel)** — no grayscale anomalies

### 2.3 RGB Channel Analysis
Per-class RGB intensity distributions reveal distinct **colour signatures** for each disease:
- Bacterial Leaf Blight → yellow/white lesion regions lift red & green channels
- Brown Spot → dark brown patches suppress all three channels
- Leaf Smut → subtle dark angular spots visible in all channels

> 📓 All EDA charts (sample images, class distribution, RGB histograms, color analysis) are embedded as outputs inside the executed notebook.

---

## 🔄 Data Augmentation

Augmentation expands the small training set (~67 images after 70/15/15 split) and prevents overfitting.

| Transform | Applied | Reason |
|---|---|---|
| Rotation (±20°) | ✅ Yes | Leaf angle varies in field photography |
| Horizontal Flip | ✅ Yes | No directional bias for disease detection |
| Vertical Flip | ✅ Yes | Overhead images may appear inverted |
| Width/Height Shift (±15%) | ✅ Yes | Leaf not always centred |
| Zoom (±15%) | ✅ Yes | Variable camera distance |
| Brightness adjustment | ✅ Yes | Lighting varies in the field |
| Hue / Saturation shift | ❌ **Excluded** | Colour is diagnostically critical — disease identification is colour-dependent |

> 📓 Augmentation examples are visualised inside the executed notebook.

---

## 🤖 Models Trained

Five architectures were trained and compared:

| # | Model | Parameters | Notes |
|---|---|---|---|
| 1 | **Custom CNN** | ~500K | Baseline from scratch |
| 2 | **VGG16** | ~14.7M | Transfer learning, frozen base |
| 3 | **ResNet50** | ~23.6M | Transfer learning, frozen base |
| 4 | **MobileNetV2** | ~2.6M | Deployment-efficient |
| 5 | **InceptionV3** | ~22.4M | Highest test accuracy |

### Architecture Details
- **Input size:** 224 × 224 × 3 (299 × 299 × 3 for InceptionV3)
- **Base layers:** Frozen pretrained ImageNet weights
- **Head:** GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.5) → Dense(3, Softmax)
- **Optimizer:** Adam
- **Callbacks:** EarlyStopping (patience=10), ReduceLROnPlateau, ModelCheckpoint

---

## 📈 Results

### Model Comparison (Hold-out Test Set — 18 images)

| Model | Test Accuracy | Remarks |
|---|---|---|
| Custom CNN | ~33% | Baseline, limited by dataset size |
| VGG16 | ~33–38% | Overfits; large param count for small dataset |
| ResNet50 | ~33–38% | Residual connections help slightly |
| MobileNetV2 | ~38–44% | Efficient; chosen for fine-tuning |
| **InceptionV3** | **~44.44%** | **Highest test accuracy** |

> ⚠️ **Note on accuracy:** The small dataset (only ~100 training images) sets a realistic ceiling on performance. Accuracy figures reflect the true difficulty of generalising from ~67 training samples across 3 classes.

> 📓 Training history, model comparison bar charts, and confusion matrices are all embedded as outputs inside the executed notebook.

---

## 🔁 Cross-Validation

5-Fold Cross-Validation was performed on **MobileNetV2** to provide a statistically robust performance estimate:

| Metric | Value |
|---|---|
| **CV Mean Accuracy** | ~38–44% |
| **CV Std Dev** | High (expected for n=119) |
| **Interpretation** | High std reflects data split sensitivity — normal for very small datasets |

> The CV mean is more reliable than any single hold-out split at this dataset size.
>
> 📓 5-Fold CV charts are embedded inside the executed notebook.

---

## 🎯 Fine-Tuning

**MobileNetV2** was selected for fine-tuning (despite InceptionV3 having higher raw test accuracy) because:
- **2.6M parameters** vs InceptionV3's 22.4M — far more deployment-efficient
- Optimised for **on-device / mobile inference** (ideal for field farmers)
- Fine-tuning with **LR = 1e-5** to gently adapt ImageNet features without catastrophic forgetting

---

## 🔍 Sample Predictions

> 📓 Sample prediction grids (with ground truth vs predicted labels) are embedded inside the executed notebook.

---

## 🚀 How to Run

### Prerequisites

```bash
pip install tensorflow numpy pandas matplotlib seaborn opencv-python pillow scikit-learn
```

### Dataset Setup

1. Place the dataset in `dataset/Data/` with the following structure:
```
dataset/Data/
├── Bacterial leaf blight/
├── Brown spot/
└── Leaf smut/
```

2. Open and run the notebook:
```bash
jupyter notebook "PRCP-1001-RiceLeaf-Disease-Detection-Executed-Yuvaraj S.ipynb"
```

### Run All Cells
The notebook is fully executed and self-contained — all outputs, charts, and model results are embedded.

---

## 📐 Data Split

| Split | Size | % |
|---|---|---|
| Training | ~83 images | 70% |
| Validation | ~18 images | 15% |
| Test (Hold-out) | ~18 images | 15% |

**Stratified split** ensures each class is proportionally represented in every partition.

---

## 🔑 Key Findings

1. **Dataset is balanced** — no class-weighting required during training
2. **RGB colour** is a critical diagnostic feature — hue-altering augmentations deliberately excluded
3. **InceptionV3** achieved the highest hold-out accuracy (~44.44%) among all models
4. **MobileNetV2** is the recommended deployment model due to its efficiency (~2.6M params)
5. **Dataset size** (~119 images) is the primary bottleneck — performance ceiling is expected around 44–50% without additional data
6. **5-Fold CV** confirms ~38–44% is a realistic generalisation estimate, not a lucky split

---

## 📚 References

- Xanthomonas oryzae pv. oryzae — Bacterial Leaf Blight pathogen
- Helminthosporium oryzae — Brown Spot pathogen
- Entyloma oryzae — Leaf Smut pathogen
- Simonyan & Zisserman (2014) — VGG16
- He et al. (2015) — ResNet50
- Sandler et al. (2018) — MobileNetV2
- Szegedy et al. (2015) — InceptionV3

---

## 👤 Author

**Yuvaraj S**
- Course: AIE
- Team ID: PTID-AIE-APR-26-11148
- Project: PRCP-1001 (Individual Contribution)

---

*Built with ❤️ using TensorFlow, Keras, and Python*
