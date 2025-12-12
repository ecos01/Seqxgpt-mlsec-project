# AI Text Detection: SeqXGPT + BERT Pipeline

This repository implements a turnkey pipeline for **AI-generated text detection**, starting from the original [SeqXGPT](https://arxiv.org/abs/2310.08903) work.
The goal is to compare a **SeqXGPT-style detector** against a **BERT-based classifier** on the same benchmark and study their behavior in a security-relevant setting (plagiarism, spam, misinformation, etc.).

---

## 1. Overview

The project provides:

* A **SeqXGPT-style model**: CNN + self-attention operating on token-level log-probabilities from GPT‑2.
* A **BERT detector**: fine-tuned `bert-base-uncased` for binary classification (human vs AI).
* A **unified training and evaluation pipeline** on **SeqXGPT-Bench**.
* Optional **robustness / evasion tests** via paraphrasing and round-trip translation.

This addresses a machine-learning security problem: **reliably detecting AI-generated text** to support use cases such as plagiarism detection, content moderation, and misinformation control.

---

## 2. Project Structure

```
SeqXGPT-MLSEC-Project/
├── data/                      # Dataset loaders
│   ├── seqxgpt_dataset.py     # SeqXGPT-Bench loader
│   └── extra_dataset.py       # Generic dataset loader (optional)
├── dataset/                   # Raw datasets (from original SeqXGPT repo)
│   ├── SeqXGPT-Bench/         # Main benchmark (sentence-level)
│   ├── document-level detection dataset/      # optional
│   └── OOD sentence-level detection dataset/  # optional
├── models/                    # Model architectures
│   ├── seqxgpt.py             # SeqXGPT-style CNN + Self-Attention
│   └── bert_detector.py       # BERT-based detector
├── features/                  # Feature extraction from LLMs
│   └── llm_probs.py           # GPT-2 log-probability extraction
├── attacks/                   # Evasion attacks / text augmentation
│   └── text_augmentation.py   # Paraphrasing & back-translation
├── configs/                   # Configuration files (YAML)
│   ├── seqxgpt_config.yaml
│   └── bert_config.yaml
├── checkpoints/               # Saved model checkpoints (generated)
├── results/                   # Evaluation results, plots, tables (generated)
├── train_seqxgpt.py           # Training script for SeqXGPT-style detector
├── train_bert.py              # Training script for BERT detector
├── eval.py                    # Comparative evaluation script
├── verify_setup.py            # Sanity checks for environment & data
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

The raw datasets under `dataset/` are taken from the official SeqXGPT repository.

---

## 3. Installation and Setup

### 3.1 Create environment and install dependencies

```
python -m venv venv
source venv/bin/activate      # Linux/macOS
# .\venv\Scripts\Activate.ps1  # Windows PowerShell

pip install -r requirements.txt
```

Main dependencies: `torch`, `transformers`, `datasets`, `scikit-learn`, `pyyaml`, `numpy`, `tqdm`, `matplotlib`.

### 3.2 Verify setup

```
python verify_setup.py
```

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

## 5. Project Plan and Methodology

### 5.1 Setup

Modular components: data loaders, GPT‑2 log-prob features, SeqXGPT model, BERT model, training scripts, evaluation, configs.

### 5.2 LLM Feature Extraction

* Tokenize with GPT‑2
* Compute log-probabilities per token
* Pad / truncate to fixed length
* Cache for speed

### 5.3 SeqXGPT-style Model

CNN layers → Multi-head self-attention → Pooling → MLP.

Training: `train_seqxgpt.py`.

### 5.4 BERT Detector

Binary classifier based on `bert-base-uncased`.

Training: `train_bert.py`.

### 5.5 Evaluation

Metrics: accuracy, precision, recall, F1, AUROC.

Optional plots: ROC, confusion matrix.

### 5.6 Evasion & Robustness

Paraphrasing + back-translation (`text_augmentation.py`).

---

## 6. Usage

### 6.1 Quick Start

```
python verify_setup.py
python train_seqxgpt.py
python train_bert.py
python eval.py
```

### 6.2 Text Augmentation

```
from attacks.text_augmentation import TextAugmenter
augmenter = TextAugmenter()
para = augmenter.paraphrase(text)
bt = augmenter.back_translate(text, intermediate_lang="it")
```

---

## 7. Configuration

### SeqXGPT

```
model:
  hidden_dim: 128
  num_cnn_layers: 3
  num_attention_heads: 4
training:
  batch_size: 16
  learning_rate: 0.0001
  num_epochs: 20
llm:
  model_name: "gpt2"
  max_length: 256
```

### BERT

```
model:
  model_name: "bert-base-uncased"
training:
  batch_size: 16
  learning_rate: 0.00002
  num_epochs: 10
  max_length: 512
```

---

## 8. Reproducibility

Seeds, checkpoints, configs, results (CSV/JSON).

---

## 9. License

Add your license.

---

## 10. Acknowledgements

SeqXGPT authors, original datasets, HuggingFace, PyTorch.

---

## 11. SeqXGPT Validation Results on SeqXGPT-Bench

### 11.1 Dataset statistics

**Train**: 28722 total (4800 human, 23922 AI)
**Val**: 3591 total (600 human, 2991 AI)

### 11.2 Features

GPT‑2 features, cached and standardized.

### 11.3 Model configuration

Parameters: **225,922**
Epochs: **20**

### 11.4 Epoch-wise Validation Metrics

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

### 11.5 Best Model

Best F1: **0.9319** (epoch 20)


---

## 12. BERT Detector Results on SeqXGPT-Bench

### 12.1 Model Configuration

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

### 12.3 Observations

- High recall (97.78%) indicates the model catches almost all AI-generated text
- Good precision (87.62%) with few false positives
- F1-score of 92.42% is excellent for AI text detection
- Training converged quickly (1 epoch) due to pre-trained DistilBERT weights


---

## 13. Comparative Evaluation Results (Test Set)

### 13.1 Test Dataset

**Dataset:** SeqXGPT-Bench Test Split  
**Total Samples:** 3,591 (600 Human, 2,991 AI)  
**Class Distribution:** 16.7% Human, 83.3% AI (imbalanced)

### 13.2 Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | AUROC |
| :---- | -------: | --------: | -----: | -------: | ----: |
| **SeqXGPT** | **88.14%** | **92.23%** | 93.65% | **92.93%** | **91.45%** |
| **BERT (DistilBERT)** | 86.22% | 87.39% | **97.53%** | 92.18% | 88.41% |

### 13.3 Analysis

**Winner: SeqXGPT** 🏆

- **SeqXGPT** outperforms BERT on most metrics:
  - Higher accuracy (+1.92%)
  - Better precision (+4.84%) - fewer false positives
  - Better F1-score (+0.75%)
  - Higher AUROC (+3.04%) - better discrimination capability

- **BERT** advantages:
  - Higher recall (97.53% vs 93.65%) - catches more AI-generated text
  - Better at minimizing false negatives (misses fewer AI texts)

**Key Findings:**
1. SeqXGPT's use of GPT-2 log-probability features provides superior discrimination between human and AI text
2. BERT is more conservative (higher recall) but produces more false alarms (lower precision)
3. Both models achieve >92% F1-score, demonstrating excellent detection capability
4. The imbalanced dataset (83.3% AI) makes precision particularly important to avoid over-predicting AI class

### 13.4 Visualizations

- ROC curves: [results/roc_curves.png](results/roc_curves.png)
- Confusion matrices: [results/confusion_matrices.png](results/confusion_matrices.png)
