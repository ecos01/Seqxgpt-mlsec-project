# AI-Generated Text Detection with SeqXGPT-Style Features and a DistilBERT Baseline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

##  Overview

This project implements a **complete, production-ready pipeline** for AI-generated text detection, extending and significantly improving upon the original [SeqXGPT](https://github.com/Jihuai-wpy/SeqXGPT) research. We compare two complementary approaches:

*   **SeqXGPT-style detector** – A CNN + self-attention model leveraging GPT-2 token log-probabilities as features.
*   **BERT-based classifier** – A fine-tuned DistilBERT model for human vs. AI text classification.

###  Project Goals

This project was developed with four key objectives:

1. **Dataset & Modular Pipeline** – Reorganize the original SeqXGPT codebase into a clean, maintainable architecture using the SeqXGPT-Bench dataset.
2. **SeqXGPT-Style Detector** – Implement a robust feature-based detector using GPT-2 log-probabilities + CNN + Self-Attention classifier.
3. **BERT Baseline** – Add a fine-tuned DistilBERT model trained on the same data splits for direct comparison of feature-based vs. end-to-end fine-tuning approaches.
4. **Unified Evaluation** – Create a single evaluation pipeline with comprehensive metrics (Accuracy, Precision, Recall, F1-score, AUROC) and visualizations.

###  Key Improvements Over Original SeqXGPT

This implementation resolves **critical issues** in the original research code and adds substantial new functionality:

| **Aspect** | **Original SeqXGPT** | **This Project** |
|------------|---------------------|------------------|
| **Architecture** | Monolithic code (553-line files) | Modular components (data, models, features, configs) |
| **Feature Extraction** | Serial processing, slow | Batch processing with caching (2x faster) |
| **Feature Normalization** |  None (training crashes after 2-3 batches) |  Z-score + clipping (stable training) |
| **NaN Handling** |  None (frequent crashes) |  5-layer protection system (0 crashes) |
| **Evaluation Normalization** |  Bug: test features not normalized (AUROC ~50%) |  Fixed: uses training statistics (AUROC 91.45%) |
| **BERT Baseline** |  Not included (only RoBERTa mentioned) |  Full DistilBERT implementation + comparison |
| **Configuration** | Hardcoded hyperparameters | External YAML configs for easy experimentation |
| **Training Stability** | Unstable, crashes frequently | Robust with gradient clipping, early stopping |
| **Reproducibility** | Low (no fixed seeds, no saved stats) | High (fixed seeds, saved normalization stats) |
| **Documentation** | Minimal README | Complete README + detailed [explanation.md](explanation.md) |

**Critical Fix Example**: The original evaluation code did not normalize test features using training statistics, resulting in random AUROC (~50%). This project fixes this bug by saving and reusing training mean/std, achieving **91.45% AUROC** – a **+41.45%** improvement.

The primary goal is to provide a **reliable, well-documented foundation** for AI text detection research and applications such as **plagiarism detection, content moderation, and misinformation analysis**.


##  Features

This project offers:

*   A **complete training and evaluation pipeline** for both SeqXGPT and BERT detectors.
*   **Unified dataset loaders** with consistent preprocessing and reproducible data splits (80/10/10).
*   **Efficient log-probability feature extraction** from GPT-2 with automatic caching (saves hours on re-runs).
*   **Modular architecture** – clean separation of data loaders, models, features, and configs.
*   **Comparative evaluation** on SeqXGPT-Bench, reporting accuracy, precision, recall, F1-score, and AUROC.
*   **Robustness testing framework** via paraphrasing and back-translation attacks.
*   **Exportable metrics** including JSON logs, ROC curves, confusion matrices, and model checkpoints.
*   **Production-ready inference** with simple APIs for both models.

###  Technical Implementation Highlights

Three critical fixes that make this project work:

1. **Feature Normalization** – The original code didn't normalize GPT-2 features, causing training loss to explode after 2-3 batches. We implement Z-score normalization + clipping, enabling stable 20-epoch training.

2. **Evaluation Fix** – Original evaluation didn't normalize test features using training statistics, resulting in random AUROC (~50%). We save mean/std in checkpoints and apply them at test time, achieving correct 91.45% AUROC.

3. **Multi-Layer NaN Protection** – We implement 5 layers of NaN handling (feature extraction → normalization → forward pass → loss → gradients), eliminating all training crashes.

**For complete technical details, see [explanation.md](explanation.md)** (includes full architecture diagrams, implementation code, epoch-by-epoch training logs, and FAQ).

---

##  What's Included

This project provides a **complete reimplementation** of SeqXGPT with extensive improvements:

```
Seqxgpt-mlsec-project/
├── data/                          #  Modular dataset loaders
│   ├── seqxgpt_dataset.py         # SeqXGPT-Bench loader (consistent splits)
│   └── extra_dataset.py           # Support for additional datasets
│
├── models/                        #  Separated model architectures
│   ├── seqxgpt.py                 # SeqXGPT (CNN + Attention)
│   └── bert_detector.py           # BERT-based classifier
│
├── features/                      #  Feature extraction module
│   └── llm_probs.py               # GPT-2 log-prob extraction with caching
│
├── attacks/                       #  Robustness testing
│   └── text_augmentation.py      # Paraphrasing & back-translation attacks
│
├── configs/                       #  YAML configurations
│   ├── seqxgpt_config.yaml        # SeqXGPT hyperparameters
│   └── bert_config.yaml           # BERT hyperparameters
│
├── checkpoints/                   #  Saved models
│   ├── seqxgpt/                   # Includes feature_mean/std for eval
│   └── bert/                      # HuggingFace format
│
├── results/                       #  Evaluation outputs
│   ├── results.json               # Detailed metrics
│   ├── roc_curves.png             # ROC comparison
│   └── confusion_matrices.png    # Side-by-side confusion matrices
│
├── train_seqxgpt.py               # Training script (robust, with early stopping)
├── train_bert.py                  # BERT training (CPU-optimized)
├── eval.py                        # Unified evaluation for both models
├── run_evasion_attacks.py         # Robustness testing script
└── verify_setup.py                # Environment verification
```

**vs. Original SeqXGPT**:
-  **Everything from the original** (SeqXGPT model, GPT-2 features, SeqXGPT-Bench dataset)
-  **+ Critical bug fixes** (normalization, NaN handling, evaluation correctness)
-  **+ BERT baseline** for direct comparison
-  **+ Modular architecture** for maintainability
-  **+ Production features** (configs, caching, checkpoints with stats)
-  **+ Complete documentation** (this README + [explanation.md](explanation.md))

---

## Quick Start

```bash
# 1. Clone the repository and set up the environment
git clone https://github.com/ecos01/Seqxgpt-mlsec-project.git
cd Seqxgpt-mlsec-project
python -m venv venv
.\venv\Scripts\Activate.ps1  
pip install -r requirements.txt

# 2. Verify your setup
python verify_setup.py

# 3. Train the models
python train_seqxgpt.py  
python train_bert.py     

# 4. Evaluate both models
python eval.py           
```

**Output**:
-   Trained models are saved in `checkpoints/`
-   Evaluation results and plots are found in `results/`

---

##  Results Summary

### Final Performance (Test Set - 3,591 samples)

| Model | Accuracy | Precision | Recall | F1-Score | AUROC | Best For |
|-------|----------|-----------|--------|----------|-------|----------|
| **SeqXGPT** | **88.14%** | **92.23%** | 93.65% | **92.93%** | **91.45%** | **Precision-critical apps** |
| BERT (DistilBERT) | 86.22% | 87.39% | **97.53%** | 92.18% | 88.41% | **Recall-critical apps** |

###  Key Findings

**SeqXGPT wins overall** with superior precision (+4.84%), F1-score (+0.75%), and AUROC (+3.04%):
-  **Fewer false positives** – Better at avoiding misclassifying human text as AI
-  **Higher confidence** – Stronger distinction between classes (higher AUROC)
-  **Best for**: Content moderation, plagiarism detection, academic integrity (where false accusations are costly)

**BERT excels at recall** (+3.88%), catching 97.53% of AI-generated text:
-  **Catches more AI text** – Identifies nearly all AI-generated content
-  **More false positives** – Some human text flagged as AI
-  **Best for**: Security screening, spam filtering (where missing AI content is more costly than false alarms)

Both models achieve **>92% F1-score**, demonstrating robust detection capabilities. The choice depends on your application's cost/benefit analysis of false positives vs. false negatives.

**Visualizations**: [ROC Curves](results/roc_curves.png) | [Confusion Matrices](results/confusion_matrices.png)

**For detailed epoch-by-epoch training logs and technical analysis, see [explanation.md](explanation.md).**

---

## 1. Dataset

### SeqXGPT-Bench (Primary Dataset)

**Description**: A sentence-level benchmark dataset comprising human-written text and AI-generated text from various models (GPT-2, GPT-3, GPT-J, GPT-Neo, LLaMA).

**Composition**:
- **Total**: 36,004 samples
- **Labels**: `0` = human, `1` = AI-generated
- **Split**: 80% Train (28,722), 10% Val (3,591), 10% Test (3,591)
- **Imbalance**: ~83% AI, ~17% Human (reflects real-world scenarios)
- **Seed**: 42 (for reproducibility)

**Note**: All datasets originate from the official [SeqXGPT repository](https://github.com/Jihuai-wpy/SeqXGPT).

---

## 2. How It Works

### 2.1 GPT-2 Feature Extraction (for SeqXGPT)

The SeqXGPT pipeline transforms input text into numerical features based on GPT-2's predictions:

```
Input Text → GPT-2 Tokenization → For each token:
                                  ├─ log P(token|context)  [log-probability]
                                  ├─ -log P(token|context) [surprisal]
                                  └─ H(P)                  [entropy]
                                  
Features [batch, 256, 3] → Cached to disk → Used in training
```

These features capture the "statistical fingerprint" of AI-generated text, where AI models tend to produce more predictable tokens (higher log-probability, lower surprisal and entropy) compared to human writers. This process includes batch processing, automatic caching, and robust handling of numerical instabilities.

### 2.2 SeqXGPT-Style Model Architecture

The SeqXGPT model is a lightweight (225,922 parameters) network designed to process the GPT-2 derived features:

```
Features [batch, 256, 3]
    ↓
1D CNN Layers (3 layers, kernel=3)
    ↓ [batch, 256, 128]
Multi-Head Self-Attention (4 heads)
    ↓
Attention-Based Pooling (weighted sum)
    ↓
MLP Classifier (128 → 64 → 1)
    ↓
Sigmoid → Probability
```

It combines CNN layers for local pattern recognition with multi-head self-attention for long-range dependencies, followed by an attention-based pooling mechanism to aggregate relevant information for classification.

### 2.3 BERT Detector Architecture

The BERT Detector utilizes a fine-tuned DistilBERT model (`distilbert-base-uncased`) for end-to-end binary classification:

*   **Base model**: DistilBERT, chosen for its efficiency (40% fewer parameters than BERT-base, 2x faster).
*   **Input**: Raw text, tokenized by DistilBERT's tokenizer.
*   **Output**: Logits for two classes (Human, AI), converted to probabilities via softmax.

DistilBERT's pre-trained knowledge allows it to learn patterns directly from raw text with fast inference and training times.

---

## 3. Installation and Setup

### 3.1 Requirements

-   **Python**: 3.8 or higher.
-   **Hardware**: A CPU is sufficient; a GPU is optional for faster training.
-   **OS**: Windows, Linux, or macOS.

### 3.2 Create Environment and Install Dependencies

```bash
# Create a Python virtual environment
python -m venv venv

# Activate the environment
source venv/bin/activate      # Linux/macOS
.\venv\Scripts\Activate.ps1   # Windows PowerShell

# Install required Python packages
pip install -r requirements.txt
```

**Main Dependencies**: `torch`, `transformers`, `datasets`, `scikit-learn`, `pyyaml`, `numpy`, `tqdm`, `matplotlib`, `tabulate`.

### 3.3 Verify Setup

```bash
python verify_setup.py
```

This script verifies essential components: installed dependencies, dataset accessibility, GPU availability (if applicable), and correct model loading.

---

## 4. Training

### 4.1 Train SeqXGPT

```bash
python train_seqxgpt.py
```

This script trains the SeqXGPT model on the SeqXGPT-Bench dataset. It first extracts and caches GPT-2 features, normalizes them, and then trains the CNN+Attention model for up to 20 epochs. The best model, based on validation F1-score, is saved. Training takes approximately 2.5 hours on a CPU (including one-time feature extraction). For configuration details, see [`configs/seqxgpt_config.yaml`](configs/seqxgpt_config.yaml).

### 4.2 Train BERT

```bash
python train_bert.py
```

This script fine-tunes DistilBERT. To optimize for CPU training speed, it uses a stratified subset of 5,000 training samples. Due to DistilBERT's pre-trained weights, the model typically converges quickly, often within 1 epoch. The best model is saved in HuggingFace format. Training completes in about 15 minutes on a CPU. For configuration details, refer to [`configs/bert_config.yaml`](configs/bert_config.yaml).

---

## 5. Evaluation

### 5.1 Comparative Evaluation

```bash
python eval.py
```

This script loads both trained SeqXGPT and BERT models, evaluates them on the test set (3,591 samples), computes all relevant metrics (accuracy, precision, recall, F1-score, AUROC), and generates comparative visualizations.

**Output**:
-   A comparative metrics table (printed to console).
-   ROC curves: `results/roc_curves.png`.
-   Confusion matrices: `results/confusion_matrices.png`.
-   JSON logs with detailed metrics.

---

## 6. Inference

Examples for classifying new text using the trained models.

### 6.1 SeqXGPT Inference

```python
import torch
import yaml
from models.seqxgpt import SeqXGPTModel
from features.llm_probs import LLMProbExtractor

# Load configuration and model
with open("configs/seqxgpt_config.yaml") as f:
    config = yaml.safe_load(f)
checkpoint = torch.load("checkpoints/seqxgpt/best_model.pt", map_location="cpu")
model = SeqXGPTModel(**config['model'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load feature extractor
extractor = LLMProbExtractor(model_name=config['llm']['model_name'], max_length=config['llm']['max_length'])

# Extract and normalize features from input text
text = "Your text to classify here."
features, mask = extractor.extract_single(text)
features = (features - checkpoint['feature_mean']) / checkpoint['feature_std']
features = torch.clamp(features, -5, 5)

# Predict
with torch.no_grad():
    prob = model.predict(features.unsqueeze(0), mask.unsqueeze(0))
    pred = 1 if prob.item() > 0.5 else 0

print(f"Prediction: {'AI-generated' if pred == 1 else 'Human-written'}")
print(f"Confidence: {prob.item():.4f}")
```

### 6.2 BERT Inference

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("checkpoints/bert/best_model")
tokenizer = AutoTokenizer.from_pretrained("checkpoints/bert/best_model")
model.eval()

# Tokenize input text and predict
text = "Your text to classify here."
inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True, padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred].item()

print(f"Prediction: {'AI-generated' if pred == 1 else 'Human-written'}")
print(f"Confidence: {confidence:.4f}")
```

---

## 7. Configuration

All hyperparameters are defined in YAML files within the [`configs/`](configs/) directory.

### 7.1 SeqXGPT Configuration Example

```yaml
model:
  input_dim: 3                    # log-prob, surprisal, entropy
  hidden_dim: 128
  num_cnn_layers: 3
  num_attention_heads: 4
  dropout: 0.1
training:
  batch_size: 16
  learning_rate: 0.00005
  num_epochs: 20
  early_stopping_patience: 5
  gradient_clip_max_norm: 1.0
llm:
  model_name: "gpt2"              # GPT-2 for feature extraction
  max_length: 256
  cache_dir: "features/cache"
data:
  data_dir: "dataset/SeqXGPT-Bench"
  seed: 42
```

### 7.2 BERT Configuration Example

```yaml
model:
  model_name: "distilbert-base-uncased"
  num_labels: 2
  dropout: 0.1
training:
  batch_size: 32
  learning_rate: 0.00003
  num_epochs: 3
  max_length: 256
  early_stopping_patience: 1
  max_train_samples: 5000         # Subset for fast CPU training
  max_val_samples: 1000
  gradient_accumulation_steps: 2
data:
  data_dir: "dataset/SeqXGPT-Bench"
  seed: 42
```

---

## 8. Reproducibility

The project prioritizes reproducibility with:
-   A **fixed seed** (42) for consistent data splits.
-   **YAML configurations** for all hyperparameters.
-   **Saved checkpoints** with training statistics (including feature_mean/std).
-   **Feature caching** for deterministic results.

To reproduce the results, simply run the training scripts (`train_seqxgpt.py`, `train_bert.py`) followed by the evaluation script (`eval.py`).

**System Info** (for reference): OS: Windows 10/11, Python: 3.8+, PyTorch: 2.0+, Hardware: CPU.

---

## 9. License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

This project is intended for **academic and research purposes**. For production use, ensure compliance with model licenses (GPT-2, DistilBERT), dataset terms of use, and ethical AI detection guidelines.

---

