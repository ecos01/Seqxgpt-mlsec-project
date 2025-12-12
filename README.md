# AI Text Detection: SeqXGPT + BERT Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Machine Learning Security Project** - A unified pipeline for detecting AI-generated text using two complementary approaches:

* **SeqXGPT-style detector** – CNN + self-attention over GPT-2 token log-probabilities
* **BERT-based classifier** – Fine-tuned transformer (DistilBERT) for human vs AI classification

Compare performance, robustness, and reliability for applications such as **plagiarism detection, content moderation, and misinformation analysis**.

---

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Results Summary](#-results-summary)
- [Overview](#1-overview)
- [Project Structure](#2-project-structure)
- [Installation](#3-installation-and-setup)
- [Datasets](#4-datasets)
- [How It Works](#5-how-it-works)
- [Training](#6-training)
- [Evaluation](#7-evaluation)
- [Inference](#8-inference)
- [Configuration](#9-configuration)
- [Detailed Results](#10-detailed-results)
  - [SeqXGPT Results](#101-seqxgpt-validation-results)
  - [BERT Results](#102-bert-detector-results)
  - [Comparative Evaluation](#103-comparative-evaluation-test-set)
- [Reproducibility](#11-reproducibility)
- [License](#12-license)
- [Acknowledgements](#13-acknowledgements)

---

## ✨ Features

This project provides:

* ✅ **Full training and evaluation pipeline** for SeqXGPT and BERT detectors
* ✅ **Unified dataset loaders** with consistent preprocessing and train/val/test splits
* ✅ **Log-probability feature extraction** from GPT-2 with efficient caching
* ✅ **Modular architecture** - Clean separation of data, models, features, and configs
* ✅ **Comparative evaluation** on SeqXGPT-Bench with accuracy, precision, recall, F1, AUROC
* ✅ **Robustness tests** via paraphrasing and back-translation (optional)
* ✅ **Exportable metrics** - JSON logs, ROC curves, confusion matrices, checkpoints
* ✅ **Production-ready inference** - Simple API for text classification

**Key Innovations vs. Original SeqXGPT**:
- Added BERT baseline for direct comparison
- Optimized feature extraction with batch processing (2x speedup)
- Fixed critical NaN handling issues
- CPU-friendly training with DistilBERT (15h → 15min)
- Comprehensive documentation and FAQ

---

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone <repo-url>
cd Seqxgpt-mlsec-project
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows (or source venv/bin/activate on Linux/macOS)
pip install -r requirements.txt

# 2. Verify setup
python verify_setup.py

# 3. Train models
python train_seqxgpt.py  # ~2.5h on CPU
python train_bert.py     # ~15min on CPU

# 4. Evaluate
python eval.py           # Comparative evaluation
```

**Output**: 
- Trained models in `checkpoints/`
- Evaluation results, plots in `results/`

---

## 🏆 Results Summary

### Performance Comparison (Test Set)

| Model | Accuracy | Precision | Recall | F1-Score | AUROC | Winner |
|-------|----------|-----------|--------|----------|-------|--------|
| **SeqXGPT** | **88.14%** | **92.23%** | 93.65% | **92.93%** | **91.45%** | ✅ |
| BERT (DistilBERT) | 86.22% | 87.39% | **97.53%** | 92.18% | 88.41% | - |

**Key Takeaways**:
- ✅ **SeqXGPT wins overall** with superior precision (+4.84%), F1 (+0.75%), and AUROC (+3.04%)
- ✅ **BERT wins on recall** (+3.88%) – catches 97.5% of AI text but with more false positives
- 💡 **Use SeqXGPT** for content moderation, plagiarism detection (precision critical)
- 💡 **Use BERT** for security screening, spam filtering (recall critical)

📊 **Visualizations**: [ROC Curves](results/roc_curves.png) | [Confusion Matrices](results/confusion_matrices.png)

📖 **Detailed Documentation**: See [SPIEGAZIONE.md](SPIEGAZIONE.md) for comprehensive technical details, FAQ, and study guide.

---

## 1. Overview

### What This Project Does

Implements and compares two state-of-the-art approaches for **AI-generated text detection**:

1. **SeqXGPT-style Model**: 
   - Feature-based approach using GPT-2 log-probabilities
   - Architecture: CNN layers → Multi-head self-attention → Pooling → MLP classifier
   - Captures statistical "fingerprints" of AI-generated text
   - Lightweight: 225,922 parameters

2. **BERT Detector**:
   - End-to-end fine-tuned transformer (DistilBERT)
   - Learns patterns directly from raw text
   - Pre-trained knowledge: 66M parameters
   - Fast inference and training

### Why It Matters

**Machine Learning Security Applications**:
- 🔒 **Plagiarism Detection** - Identify AI-assisted academic dishonesty
- 🛡️ **Content Moderation** - Filter AI-generated spam/misinformation
- 🔍 **Research Integrity** - Verify authenticity of scientific writing
- ⚖️ **Legal/Forensic** - Detect AI-generated documents in investigations

**Testing Ground**: Uses **SeqXGPT-Bench** benchmark with texts from GPT-2, GPT-3, GPT-J, GPT-Neo, and LLaMA.

---

## 2. Project Structure

```
SeqXGPT-MLSEC-Project/
├── data/                      # Dataset loaders
│   ├── seqxgpt_dataset.py     # SeqXGPT-Bench loader
│   └── extra_dataset.py       # Generic dataset loader (optional)
├── dataset/                   # Raw datasets (from original SeqXGPT repo)
│   ├── SeqXGPT-Bench/         # Main benchmark (sentence-level)
│   │   ├── en_human_lines.jsonl
│   │   ├── en_gpt2_lines.jsonl
│   │   ├── en_gpt3_lines.jsonl
│   │   ├── en_gptj_lines.jsonl
│   │   ├── en_gptneo_lines.jsonl
│   │   └── en_llama_lines.jsonl
│   ├── document-level detection dataset/      # Optional
│   └── OOD sentence-level detection dataset/  # Optional
├── models/                    # Model architectures
│   ├── seqxgpt.py             # SeqXGPT CNN + Self-Attention
│   └── bert_detector.py       # BERT-based detector
├── features/                  # Feature extraction from LLMs
│   ├── llm_probs.py           # GPT-2 log-probability extraction
│   └── cache/                 # Cached features (generated)
├── attacks/                   # Evasion attacks / text augmentation
│   └── text_augmentation.py   # Paraphrasing & back-translation
├── configs/                   # Configuration files (YAML)
│   ├── seqxgpt_config.yaml
│   └── bert_config.yaml
├── checkpoints/               # Saved model checkpoints (generated)
│   ├── seqxgpt/
│   │   ├── best_model.pt
│   │   └── history.json
│   └── bert/
│       └── best_model/
├── results/                   # Evaluation results, plots, tables (generated)
│   ├── roc_curves.png
│   └── confusion_matrices.png
├── train_seqxgpt.py           # Training script for SeqXGPT
├── train_bert.py              # Training script for BERT
├── eval.py                    # Comparative evaluation script
├── run_evasion_attacks.py     # Robustness testing (optional)
├── verify_setup.py            # Environment sanity checks
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── SPIEGAZIONE.md             # Detailed technical documentation (Italian)
```

**Note**: Raw datasets under `dataset/` are from the official [SeqXGPT repository](https://github.com/Jihuai-wpy/SeqXGPT).

---

## 3. Installation and Setup

### 3.1 Requirements

- **Python**: 3.8 or higher
- **Hardware**: CPU sufficient (GPU optional for faster training)
- **OS**: Windows, Linux, or macOS

### 3.2 Create Environment and Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate      # Linux/macOS
.\venv\Scripts\Activate.ps1   # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

**Main Dependencies**: 
- `torch` (PyTorch)
- `transformers` (HuggingFace)
- `datasets`
- `scikit-learn`
- `pyyaml`
- `numpy`
- `tqdm`
- `matplotlib`
- `tabulate`

### 3.3 Verify Setup

```bash
python verify_setup.py
```

This checks:
- ✅ All dependencies installed
- ✅ Datasets accessible
- ✅ GPU availability (if present)
- ✅ Model loading works

---

## 4. Datasets

### 4.1 SeqXGPT-Bench

Sentence-level benchmark containing human- and AI-generated text (GPT‑2, GPT‑3, GPT‑J, GPT‑Neo, LLaMA).

Loader: `SeqXGPTDataset` → binary labels (0 = human, 1 = AI).

Default split: **80 / 10 / 10**.

### 4.2 Optional datasets

* Document-level dataset
* OOD sentence-level dataset

---

## 5. Training

### 5.1 Train SeqXGPT

```bash
python train_seqxgpt.py
```

**Process**:
1. Loads SeqXGPT-Bench dataset (28,722 train, 3,591 val)
2. Extracts GPT-2 features (log-prob, surprisal, entropy) → cached
3. Trains CNN + Attention model for 20 epochs
4. Saves best model to `checkpoints/seqxgpt/best_model.pt`

**Training Time**: ~2.5 hours on CPU (1.5h features + 1h training)

**Configuration**: [`configs/seqxgpt_config.yaml`](configs/seqxgpt_config.yaml)

### 5.2 Train BERT

```bash
python train_bert.py
```

**Process**:
1. Loads 5k train samples (stratified subset for speed)
2. Fine-tunes DistilBERT for binary classification
3. Early stopping at epoch 1 (converged)
4. Saves model to `checkpoints/bert/best_model/`

**Training Time**: ~15 minutes on CPU

**Configuration**: [`configs/bert_config.yaml`](configs/bert_config.yaml)

---

## 6. Evaluation

### 6.1 Comparative Evaluation

```bash
python eval.py
```

**Output**:
- Comparative metrics table (Accuracy, Precision, Recall, F1, AUROC)
- ROC curves: `results/roc_curves.png`
- Confusion matrices: `results/confusion_matrices.png`

### 6.2 Evasion Attacks (Optional)

```bash
python run_evasion_attacks.py
```

Tests robustness against:
- Paraphrasing (T5-based)
- Back-translation (en→it→en)

```python
from attacks.text_augmentation import TextAugmenter

augmenter = TextAugmenter()
paraphrased = augmenter.paraphrase(text)
back_translated = augmenter.back_translate(text, intermediate_lang="it")
```

---

## 7. Configuration

Configuration files in [`configs/`](configs/) define all hyperparameters.

### 7.1 SeqXGPT Configuration

**File**: [`configs/seqxgpt_config.yaml`](configs/seqxgpt_config.yaml)

```yaml
model:
  input_dim: 3                    # log-prob, surprisal, entropy
  hidden_dim: 128
  num_cnn_layers: 3
  num_attention_heads: 4
  dropout: 0.1

training:
  batch_size: 16
  learning_rate: 0.00005          # 5e-5
  num_epochs: 20
  early_stopping_patience: 5
  gradient_clip_max_norm: 1.0

llm:
  model_name: "gpt2"
  max_length: 256
  cache_dir: "features/cache"
```

### 7.2 BERT Configuration

**File**: [`configs/bert_config.yaml`](configs/bert_config.yaml)

```yaml
model:
  model_name: "distilbert-base-uncased"  # 2x faster than BERT
  num_labels: 2
  dropout: 0.1

training:
  batch_size: 32
  learning_rate: 0.00003          # 3e-5
  num_epochs: 3
  max_length: 256                 # Reduced for speed
  early_stopping_patience: 1
  max_train_samples: 5000         # Subset for fast training
  max_val_samples: 1000
```

---

## 8. Detailed Results

### 8.1 SeqXGPT Validation Results

**Dataset Statistics**:
- **Train**: 28,722 samples (4,800 human, 23,922 AI) - 83.3% imbalanced
- **Validation**: 3,591 samples (600 human, 2,991 AI)

**Model Configuration**:
- Architecture: 1D CNN (3 layers) + Multi-Head Attention (4 heads) + Attention Pooling
- Parameters: **225,922**
- Training: 20 epochs with early stopping (patience=5)

**Feature Extraction**:
- GPT-2 log-probabilities, surprisal, entropy per token
- Cached and z-score normalized with clipping [-5, +5]

**Epoch-wise Validation Metrics**:

| Epoch |    Val Acc |     Val F1 |  Val AUROC |
| ----: | ---------: | ---------: | ---------: |
|     1 |     0.8329 |     0.9088 |     0.8360 |
|     2 |     0.8329 |     0.9088 |     0.8649 |
|     3 |     0.8329 |     0.9088 |     0.8709 |
|     4 |     0.8491 |     0.9131 |     0.8768 |
|     5 |     0.8624 |     0.9186 |     0.8878 |
|     6 |     0.8563 |     0.9176 |     0.8836 |
|     7 |     0.8544 |     0.9105 |     0.8936 |
|     8 |     0.8733 |     0.9249 |     0.9001 |
|    10 |     0.8777 |     0.9277 |     0.9030 |
|    11 |     0.8791 |     0.9282 |     0.9059 |
|    12 |     0.8755 |     0.9254 |     0.9073 |
|    13 |     0.8808 |     0.9295 |     0.9048 |
|    14 |     0.8761 |     0.9245 |     0.9106 |
|    15 |     0.8669 |     0.9173 |     0.9106 |
|    16 |     0.8842 |     0.9308 |     0.9107 |
|    17 |     0.8842 |     0.9311 |     0.9110 |
|    18 |     0.8816 |     0.9282 |     0.9137 |
|    19 |     0.8844 |     0.9302 |     0.9126 |
|    20 | **0.8858** | **0.9319** | **0.9153** |

**Best Model**: F1 = **93.19%** at epoch 20

**Checkpoint**: [`checkpoints/seqxgpt/best_model.pt`](checkpoints/seqxgpt/)

---

### 8.2 BERT Detector Results

| Parameter | Value |
| :-------- | :---- |
| Model | DistilBERT (distilbert-base-uncased) |
| Parameters | ~66M |
| Max Length | 256 tokens |
| Batch Size | 32 |
| Learning Rate | 3e-5 |
| Training Samples | 5,000 (subset) |
| Validation Samples | 1,000 |
| Epochs Completed | 1 (early stopping) |

### 12.2 Validation Metrics

| Metric | Value |
| :----- | ----: |
| **Accuracy** | 0.8665 (86.65%) |
| **Precision** | 0.8762 (87.62%) |
| **Recall** | 0.9778 (97.78%) |
| **F1-Score** | **0.9242 (92.42%)** |
| **AUROC** | 0.8825 (88.25%) |

**Observations**:
- ✅ High recall (97.78%) → Catches almost all AI-generated text
- ✅ Good precision (87.62%) → Few false positives
- ✅ Fast convergence (1 epoch) due to pre-trained weights
- ⚠️ Trained on 5k subset → May improve with full dataset

**Checkpoint**: [`checkpoints/bert/best_model/`](checkpoints/bert/)

---

### 8.3 Comparative Evaluation (Test Set)

**Test Dataset**:
- **Source**: SeqXGPT-Bench Test Split
- **Total Samples**: 3,591 (600 Human, 2,991 AI)
- **Class Distribution**: 16.7% Human, 83.3% AI (**imbalanced**)

**Model Performance Comparison**:

| Model | Accuracy | Precision | Recall | F1-Score | AUROC |
| :---- | -------: | --------: | -----: | -------: | ----: |
| **SeqXGPT** | **88.14%** | **92.23%** | 93.65% | **92.93%** | **91.45%** |
| **BERT (DistilBERT)** | 86.22% | 87.39% | **97.53%** | 92.18% | 88.41% |

**Analysis**:

🏆 **Winner: SeqXGPT**

**SeqXGPT Advantages**:
- ✅ **Higher Accuracy** (+1.92%): Better overall performance
- ✅ **Superior Precision** (+4.84%): Fewer false positives (human wrongly classified as AI)
- ✅ **Better F1-Score** (+0.75%): Optimal precision-recall balance
- ✅ **Higher AUROC** (+3.04%): Superior discrimination capability
- 💡 **Why?** GPT-2 log-probability features capture the "statistical fingerprint" of AI-generated text

**BERT Advantages**:
- ✅ **Higher Recall** (+3.88%): Catches 97.5% of AI text (misses only 2.5%)
- 💡 **When to use?** When cost of missing AI text > cost of false accusations

**Key Insights**:
1. **SeqXGPT's GPT-2 features** provide superior discrimination between human and AI text
2. **BERT is more conservative** (higher recall) but produces more false alarms (lower precision)
3. **Both models achieve >92% F1**, demonstrating excellent detection capability
4. **Imbalanced dataset** (83.3% AI) makes **precision critical** to avoid over-predicting AI class

**Use Case Recommendations**:
- **Use SeqXGPT** for: Content moderation, plagiarism detection, research (precision matters)
- **Use BERT** for: Security screening, spam filtering (recall matters)

**Visualizations**:
- 📈 [ROC Curves](results/roc_curves.png) - Compare discrimination at different thresholds
- 📊 [Confusion Matrices](results/confusion_matrices.png) - See false positive/negative breakdown

---

## 9. Reproducibility

**Reproducibility Features**:
- ✅ **Fixed seed** (42) for consistent train/val/test splits
- ✅ **YAML configs** for all hyperparameters
- ✅ **Checkpoints** saved with training statistics
- ✅ **Feature caching** for deterministic results

**To Reproduce Results**:

```bash
# Ensure same environment
pip install -r requirements.txt

# Same data splits (seed=42)
python train_seqxgpt.py  # Saves to checkpoints/seqxgpt/
python train_bert.py     # Saves to checkpoints/bert/

# Same evaluation
python eval.py           # Uses saved checkpoints
```

**System Info** (for reference):
- OS: Windows 10/11
- Python: 3.8+
- PyTorch: 2.0+
- Hardware: CPU (no GPU required)

---

## 10. License

MIT License - See [LICENSE](LICENSE) file for details.

---

## 11. Acknowledgements
