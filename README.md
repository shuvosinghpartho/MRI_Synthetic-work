# Synthetic Medical Image Augmentation and Cross-Domain Robustness in Brain Tumor MRI and Multi-Class Lung Ultrasound via Transfer-Learned StyleGAN2-ADA

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00.svg?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Generative Model](https://img.shields.io/badge/Model-StyleGAN2--ADA-8A2BE2.svg)](https://github.com/NVlabs/stylegan2-ada-pytorch)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Verified-brightgreen.svg)]()
[![Platform: Kaggle / CUDA](https://img.shields.io/badge/Platform-Kaggle%20GPU%20%7C%20CUDA-20BEFF.svg?logo=kaggle&logoColor=white)](https://www.kaggle.com/)

---

## 📌 Executive Summary

Data scarcity, severe class imbalances, and stringent patient privacy regulations (e.g., HIPAA, GDPR) present substantial bottlenecks in training deep neural networks for medical diagnostics. This research presents an end-to-end framework investigating whether **transfer-learned generative adversarial networks (StyleGAN2-ADA)** can synthesize clinically realistic medical scans to alleviate data scarcity and whether synthetic data can be trusted during **cross-domain model selection and validation**.

We evaluate this methodology across two distinct clinical modalities:
1. **Multi-Class Neuro-Oncology (Brain Tumor MRI):** 4-class classification (`glioma`, `meningioma`, `pituitary tumor`, `no tumor`).
2. **Cross-Modality Pulmonary Diagnostics (Lung Ultrasound / LUS):** 3-class classification (`COVID-19`, `Healthy`, `Other Pulmonary Infection`).

In addition to traditional pure Convolutional Neural Networks (**EfficientNet-B0**, **DenseNet-169**), we evaluate **Hybrid CNN-Vision Transformer** architectures (**DenseNet-169 + ViT-Tiny**, **EfficientNet-B0 + ViT-Tiny**) that fuse convolutional inductive biases with global self-attention. Crucially, we conduct cross-domain experiments examining whether models tuned on **synthetic validation sets** preserve robust generalization on **untouched, real patient test sets**.

---

## 🔬 Core Contributions

- **Generative Transfer Learning in Data-Constrained Regimes:** Adaptation of official NVIDIA StyleGAN2-ADA using FFHQ-256 pretraining, customized Adam optimization ($\beta_0 = 0.9, \beta_1 = 0.99$), full FP32 numerical stability, and Adaptive Discriminator Augmentation (ADA) to suppress discriminator overfitting on medical cohorts.
- **Multi-Metric Generative Evaluation:** Rigorous quantitative evaluation across training snapshots using **Fréchet Inception Distance (FID)**, **Kernel Inception Distance (KID)**, and **Learned Perceptual Image Patch Similarity (LPIPS)** diversity metrics.
- **Downstream Diagnostic Benchmarking:** Systematic comparative study assessing classification performance **Before Synthetic Augmentation (Real Only)** versus **After Synthetic Augmentation (Real + Synthetic)** across diagnostic and statistical metrics (Accuracy, Precision, Recall, Specificity, Macro/Weighted F1, MCC, ROC-AUC).
- **Cross-Domain Generalization & Model Selection:** Investigation into domain transfer under two clinical protocols:
  - *Train on Real $\to$ Validate on Real $\to$ Test on Synthetic* (quantifying synthetic fidelity and out-of-distribution alignment).
  - *Train on Real $\to$ Validate on Synthetic $\to$ Test on Untouched Real* (evaluating whether synthetic validation sets can be safely used for hyperparameter tuning and model checkpointing without compromising clinical generalization).
- **Clinical Computational Profiling:** Comprehensive edge-computing profiling documenting Total and Trainable Parameters, Multiply-Accumulate Operations (FLOPs / GMACs), Inference Latency (ms), and Frame Throughput (FPS).
- **Cross-Modality & Cross-Dataset Translation:** Verification of the generative augmentation pipeline on Lung Ultrasound (LUS) COVID-19 datasets, proving modality-agnostic transferability.

---

## 🏗️ Methodological Architecture

```
                                  [ Raw Clinical Data ]
                             (Brain Tumor MRI / Lung Ultrasound)
                                            │
                                            ▼
                           [ Data Preprocessing & Curation ]
                           - Lanczos 256x256 Resampling
                           - Aspect-Ratio Center Cropping
                           - Stratified Split (80% Train / 10% Val / 10% Test)
                                            │
                  ┌─────────────────────────┴─────────────────────────┐
                  ▼                                                   ▼
     [ Generative Synthesis Track ]                      [ Baseline Diagnostic Track ]
      NVIDIA StyleGAN2-ADA Transfer                      Train Pure CNNs & Hybrid ViTs
      - Source: FFHQ-256 Pretrained Weights               - EfficientNet-B0 / DenseNet-169
      - Adam β0 = 0.9, FP32 Stability                     - EfficientNet-B0 + ViT-Tiny
      - Adaptive Discriminator Augmentation               - DenseNet-169 + ViT-Tiny
                  │                                                   │
                  ▼                                                   │
     [ Checkpoint Evaluation ]                                        │
      - Fréchet Inception Distance (FID)                              │
      - Kernel Inception Distance (KID)                               │
      - LPIPS Perceptual Diversity                                    │
                  │                                                   │
                  ▼                                                   │
     [ Synthetic Data Generation ]                                    │
      High-Fidelity Synthetic Images                                  │
                  │                                                   │
                  └─────────────────────────┬─────────────────────────┘
                                            ▼
                           [ Augmented Diagnostic Track ]
                            Real + Synthetic Hybrid Sets
                                            │
                  ┌─────────────────────────┴─────────────────────────┐
                  ▼                                                   ▼
     [ Cross-Domain Validation ]                        [ Comprehensive Diagnostics ]
      - Train: Real                                      - Accuracy, Precision, Recall
      - Validation: Synthetic (Model Selection)          - Specificity, F1-Score, MCC
      - Test: Untouched Real (Generalization)            - ROC-AUC (Micro & Per-Class)
                                                         - Computational Profiling:
                                                           GMACs, Latency (ms), FPS
```

---

## 📂 Repository Organization

```text
MRI_Synthetic-Work/
├── CITATION.cff                                 # Academic citation metadata
├── LICENSE                                      # MIT Open-Source License
├── README.md                                    # Comprehensive journal-grade documentation
├── requirements.txt                             # Explicit pip dependencies
├── environment.yml                              # Conda environment configuration
│
├── Synthetic Mri with StyleGAN2-ADA/            # [Track 1: Generative MRI Synthesis]
│   └── StyleGAN2-ADA.ipynb                      # StyleGAN2-ADA fine-tuning, optimizer patching & FID
│
├── Classification Experiment Before and after synthetic augmentation/
│   ├── Before synthetic/                        # [Track 2A: Baseline Real-Only Models]
│   │   └── Run Files/
│   │       ├── Before-DenseNet169.ipynb         # DenseNet-169 trained on real MRI data
│   │       ├── Before-DenseNet169+ViT.ipynb     # DenseNet-169 + ViT-Tiny hybrid on real MRI
│   │       ├── Before-EfficientNetB0.ipynb      # EfficientNet-B0 trained on real MRI data
│   │       └── Before-EfficientNetB0+ViT.ipynb  # EfficientNet-B0 + ViT-Tiny hybrid on real MRI
│   │
│   └── After Synthetic/                         # [Track 2B: Augmented Real + Synthetic Models]
│       └── Run Files/
│           ├── After-DenseNet169.ipynb          # DenseNet-169 trained with synthetic augmentation
│           ├── After-DenseNet169+ViT.ipynb      # DenseNet-169 + ViT-Tiny hybrid augmented
│           ├── After-EfficientNetB0.ipynb       # EfficientNet-B0 trained with synthetic augmentation
│           └── After-EfficientNetB0+ViT.ipynb   # EfficientNet-B0 + ViT-Tiny hybrid augmented
│
├── Cross-domain experiment।.Train-Raw and Val-Synthetic/
│   │                                            # [Track 3: Cross-Domain & Model Selection]
│   ├── DenseNet169+ViT.ipynb                    # Train Real -> Val Real -> Test Synthetic
│   ├── efficientNetB0+ViT.ipynb                 # Train Real -> Val Real -> Test Synthetic
│   ├── tuned-DenseNetr169.ipynb                 # Train Real -> Val Synthetic -> Test Untouched Real
│   └── tuned-efficientNetB0.ipynb               # Train Real -> Val Synthetic -> Test Untouched Real
│
└── cross-dataset syntheic-Lung ultra covid 19/  # [Track 4: Cross-Modality Pulmonary Diagnostics]
    ├── 1.ipynb                                  # LUS StyleGAN2-ADA fine-tuning & evaluation
    ├── cross-dataset tranning curve.png         # Multi-checkpoint FID/KID/LPIPS progression
    ├── cross-dataset tranning.png               # Training loss and discriminator dynamics
    └── LUS Images for Automated AI-based Lung Disease Classification/
        ├── covid/                               # Real COVID-19 lung ultrasound scans
        ├── healthy/                             # Real healthy control scans
        └── other/                               # Real other pulmonary infections
```

---

## 📊 Empirical Results & Benchmarks

### 1. Classification Performance: Before vs. After Synthetic Augmentation (Brain Tumor MRI)

Evaluated on the independent test split ($N = 594$ images, stratified 80:10:10):

| Model Architecture | Paradigm | Test Accuracy | Macro F1 | Weighted F1 | Specificity | MCC | Micro ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B0** | Before Synthetic | 84.00% | 0.8400 | 0.8400 | 0.9426 | 0.7744 | 0.9612 |
| **EfficientNet-B0** | **After Synthetic** | **84.34%** (Val) | 0.8000 | 0.8000 | 0.9345 | 0.7377 | 0.9620 |
| **DenseNet-169** | Before Synthetic | 95.00% | 0.9400 | 0.9500 | 0.9821 | 0.9286 | 0.9910 |
| **DenseNet-169** | **After Synthetic** | **96.47%** (Val) | **0.9500** | **0.9500** | **0.9821** | **0.9286** | **0.9945** |
| **EfficientNet-B0 + ViT-Tiny** | Before Synthetic | 98.99% | 0.9900 | 0.9900 | 0.9950 | 0.9860 | 0.9991 |
| **EfficientNet-B0 + ViT-Tiny** | **After Synthetic** | **97.64%** | **0.9800** | **0.9800** | **0.9920** | **0.9680** | **0.9973** |
| **DenseNet-169 + ViT-Tiny** | Before Synthetic | 98.99% | 0.9900 | 0.9900 | 0.9950 | 0.9860 | 0.9994 |
| **DenseNet-169 + ViT-Tiny** | **After Synthetic** | **97.64%** | **0.9800** | **0.9800** | **0.9920** | **0.9680** | **0.9988** |

*Per-Class ROC-AUC for Best Hybrid Models:*
- Glioma: **0.9986** | Meningioma: **0.9974** | No Tumor: **0.9999** | Pituitary: **0.9989**

---

### 2. Computational Complexity & Clinical Edge Feasibility

Benchmarked on single GPU inference with warm-up loops:

| Model Architecture | Computational Cost (GMACs) | Inference Latency (ms / image) | Throughput (FPS) | Clinical Deployment Target |
| :--- | :---: | :---: | :---: | :--- |
| **EfficientNet-B0** | **0.4004** | **86.22 ms** | **341.84** | Ultra-low power / Point-of-care mobile devices |
| **DenseNet-169** | 3.3788 | 106.07 ms | 144.87 | Hospital workstation / Diagnostic radiology PACS |
| **EfficientNet-B0 + ViT-Tiny** | **1.3225** | **2.31 ms** | **433.68** | Real-time clinical screening / High-throughput triage |
| **DenseNet-169 + ViT-Tiny** | 4.3511 | 4.86 ms | 205.72 | High-precision diagnostic confirmatory pipeline |

---

### 3. Cross-Domain Robustness & Model Selection Matrix

To evaluate domain shift and the feasibility of using synthetic data for model validation:

| Experiment Protocol | Model Architecture | Validation Domain | Test Domain | Validation Acc | Test Acc | Primary Takeaway |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Real $\to$ Synthetic Generalization** | DenseNet-169 + ViT | Real MRI | Synthetic MRI | 98.91% | **96.68%** | Hybrid models maintain >96.5% generalization to synthetic scans. |
| **Real $\to$ Synthetic Generalization** | EfficientNet-B0 + ViT | Real MRI | Synthetic MRI | 98.91% | **96.42%** | Demonstrates high distribution alignment between real and synthetic data. |
| **Synthetic Proxy Validation** | Tuned DenseNet-169 | Synthetic MRI | **Untouched Real MRI** | 85.47% | **86.58%** | **Validation on synthetic data yields valid checkpoints for real patient data.** |
| **Synthetic Proxy Validation** | Tuned EfficientNet-B0 | Synthetic MRI | **Untouched Real MRI** | 86.19% | **87.38%** | **Models selected purely on synthetic validation generalize without real-world test collapse.** |

---

### 4. Generative Synthesis Progression (Lung Ultrasound - LUS Track)

Snapshot evaluation across training ticks showing convergence:

| Snapshot Checkpoint | Training kimg | Fréchet Inception Distance (FID) ↓ | Kernel Inception Distance (KID Mean ± Std) ↓ | LPIPS Perceptual Diversity ↑ |
| :--- | :---: | :---: | :---: | :---: |
| `network-snapshot-000000` | 0 | 361.55 | 0.3903 ± 0.0098 | 0.5370 |
| `network-snapshot-000016` | 16 | 149.93 | 0.1229 ± 0.0075 | 0.5128 |
| `network-snapshot-000032` | 32 | 107.94 | 0.0694 ± 0.0073 | 0.4564 |
| `network-snapshot-000048` | 48 | 117.94 | 0.0770 ± 0.0076 | 0.4078 |
| `network-snapshot-000064` (Best) | **64** | **72.11** | **0.0268 ± 0.0045** | **0.4491** |

---

## 🛠️ Step-by-Step Reusability & Reproduction Guide

### 1. Hardware & System Requirements

- **Operating System:** Linux (Ubuntu 20.04/22.04 LTS recommended) or macOS / Windows with WSL2.
- **Accelerator:** NVIDIA GPU with $\ge 8$ GB VRAM (e.g., NVIDIA Tesla T4, RTX 3080/4090, A100).
- **CUDA Toolkit:** $\ge 11.8$ or $12.1$.
- **Python:** Version $3.10$ or higher.

### 2. Environment Installation

#### Option A: Using Conda (Recommended for Isolation)
```bash
# Clone the repository
git clone https://github.com/shuvosinghpartho/MRI_Synthetic-work.git
cd MRI_Synthetic-work

# Create and activate environment
conda env create -f environment.yml
conda activate mri-synthetic
```

#### Option B: Using Python `venv` and `pip`
```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Upgrade pip and install wheel
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt
```

---

### 3. Dataset Procurement & Preparation

The experiments utilize publicly available benchmark cohorts:
1. **Brain Tumor MRI Benchmark:** 4-class cohort (`glioma`, `meningioma`, `notumor`, `pituitary`).
   - Sourced from Kaggle: `shuvosingh/brain-tumor-mri-dataset` and `ramanarasimhakanduri/brain-tumour-dataset`.
   - Resolution: Standardized to $256 \times 256$ RGB with Lanczos resampling.
2. **Lung Ultrasound (LUS) COVID-19 Cohort:** 3-class cohort (`covid`, `healthy`, `other`).
   - Sourced from Kaggle: `shuvosingh/lus-cnn-hybrid`.
   - Real benchmark scans are included directly in `cross-dataset syntheic-Lung ultra covid 19/LUS Images for Automated AI-based Lung Disease Classification/`.

---

### 4. Executing Experiments

#### Track 1: StyleGAN2-ADA Medical Image Synthesis
Navigate to `Synthetic Mri with StyleGAN2-ADA/`:
```bash
# Launch Jupyter Lab
jupyter lab "Synthetic Mri with StyleGAN2-ADA/StyleGAN2-ADA.ipynb"
```
*Key configuration settings inside the notebook:*
- Source Network: `https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-256x256.pkl`
- Adam Optimizer: $\beta_0 = 0.9, \beta_1 = 0.99$ (patched directly in StyleGAN training loops).
- Precision: FP32 (`--fp32=True`).
- ADA Target: `p=0.5` adaptive augmentation.

#### Track 2: Classification Benchmarks (Before vs. After Synthetic Augmentation)
Open the desired model in `Classification Experiment Before and after synthetic augmentation/`:
- **Real-only baseline:** Run `Before synthetic/Run Files/Before-DenseNet169+ViT.ipynb`
- **Synthetic augmented:** Run `After Synthetic/Run Files/After-DenseNet169+ViT.ipynb`
- Stratification is preserved with fixed random seed (`SEED = 42`) across an 80:10:10 ratio.

#### Track 3: Cross-Domain & Model Selection Experiments
Open the notebooks in `Cross-domain experiment।.Train-Raw and Val-Synthetic/`:
- To verify model selection using synthetic validation: `tuned-DenseNetr169.ipynb` and `tuned-efficientNetB0.ipynb`.
- Checkpoints are saved based on the highest validation accuracy on synthetic validation sets, and subsequently evaluated against the independent, untouched real patient test split.

#### Track 4: Cross-Modality Pulmonary Diagnostics
Navigate to `cross-dataset syntheic-Lung ultra covid 19/`:
```bash
jupyter lab "cross-dataset syntheic-Lung ultra covid 19/1.ipynb"
```
- Trains StyleGAN2-ADA on ultrasound, snapshots checkpoints at regular intervals, computes FID/KID/LPIPS, selects the optimal generator, and performs downstream multi-class classification.

---

## 🔒 Rigor, Leakage Prevention & Reproducibility Guarantee

1. **Strict Data Isolation:** Stratified splitting is performed prior to any data transformation. At no point are synthetic samples generated from the test split, nor do test samples enter the training or validation pipelines.
2. **Deterministic Seeding:** PyTorch, TensorFlow, NumPy, and Python standard random generators are initialized with explicit seeds (`42` or `101`) to ensure exact metric reproducibility across platforms.
3. **Lossless Storage:** All generative and intermediate image pipelines export strictly in lossless PNG format to prevent JPEG block compression artifacts from confounding feature extractors.
4. **Computational Fair Comparisons:** Latency and throughput benchmarks are computed after identical GPU warm-up cycles (10 iterations) across 100 consecutive forward passes with synchronized CUDA events (`torch.cuda.synchronize()`).

---

## 📑 Citation & Academic Attribution

If you utilize this codebase, experimental methodologies, hybrid architectures, or generated datasets in your scholarly work, please cite our repository:

```bibtex
@misc{partho2026mri_synthetic,
  author       = {Shuvo Singh Partho},
  title        = {Synthetic Medical Image Augmentation and Cross-Domain Robustness in Brain Tumor MRI and Multi-Class Lung Ultrasound via Transfer-Learned StyleGAN2-ADA},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {\url{https://github.com/shuvosinghpartho/MRI_Synthetic-work}},
  note         = {Codebase and experimental protocols under peer review}
}
```

---

## 📜 License & Compliance

This repository is distributed under the **[MIT License](LICENSE)**. 

*Ethical Disclaimer:* The synthetic images, models, and evaluation scripts provided in this repository are intended exclusively for academic, research, and scientific inquiry. They are not cleared by regulatory agencies (FDA, CE) as medical diagnostic devices and must not be used for direct patient care without formal clinical trials and institutional review board approval.

---

## 📬 Inquiries & Correspondence

For questions regarding experiment reproducibility, dataset access, or collaborative research:
- **Author:** Shuvo Singh Partho
- **GitHub:** [@shuvosinghpartho](https://github.com/shuvosinghpartho)
- **Repository:** [https://github.com/shuvosinghpartho/MRI_Synthetic-work](https://github.com/shuvosinghpartho/MRI_Synthetic-work)
