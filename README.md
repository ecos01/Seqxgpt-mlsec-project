# AI Text Detection: SeqXGPT + BERT Pipeline

Complete turnkey pipeline for AI-generated text detection using SeqXGPT and BERT models.

This repository implements a comprehensive comparison between:
- **SeqXGPT**: CNN + Self-Attention model using LLM log-probabilities
- **BERT Detector**: Fine-tuned BERT for binary classification

Based on the paper ["SeqXGPT: Sentence-Level AI-Generated Text Detection"](https://arxiv.org/abs/2310.08903).

## 📁 Project Structure

```
SeqXGPT/
├── data/                      # Dataset loaders
│   ├── seqxgpt_dataset.py    # SeqXGPT-Bench loader
│   └── extra_dataset.py      # Generic dataset loader
├── dataset/                   # Raw datasets
│   ├── SeqXGPT-Bench/        # Main benchmark dataset
│   ├── document-level detection dataset/
│   └── OOD sentence-level detection dataset/
├── models/                    # Model architectures
│   ├── seqxgpt.py            # SeqXGPT (CNN + Self-Attention)
│   └── bert_detector.py      # BERT-based detector
├── features/                  # Feature extraction
│   └── llm_probs.py          # LLM log-probability extraction
├── attacks/                   # Evasion attacks
│   └── text_augmentation.py  # Paraphrasing & translation
├── configs/                   # Configuration files
│   ├── seqxgpt_config.yaml
│   └── bert_config.yaml
├── train_seqxgpt.py          # Training script for SeqXGPT
├── train_bert.py             # Training script for BERT
├── eval.py                    # Comparative evaluation
├── checkpoints/               # Saved models
├── results/                   # Evaluation results
└── requirements.txt           # Python dependencies
```

## 🚀 Quick Start

### 1. Installation

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Verify setup
python verify_setup.py
```

If `verify_setup.py` passes all checks, you're ready to go! 🎉

### 2. Dataset

The pipeline uses the **SeqXGPT-Bench** dataset included in the repository:
- Location: `dataset/SeqXGPT-Bench/`
- Contains texts from: GPT-2, GPT-3, GPT-J, GPT-Neo, LLaMA, and human-written
- Automatic train/val/test split (80/10/10)

### 3. Training

#### Train SeqXGPT Model

```bash
python train_seqxgpt.py
```

This will:
- Extract log-probability features from GPT-2
- Train CNN + Self-Attention model
- Save best checkpoint to `checkpoints/seqxgpt/`

Configuration: `configs/seqxgpt_config.yaml`

#### Train BERT Detector

```bash
python train_bert.py
```

This will:
- Fine-tune BERT on text classification
- Save best checkpoint to `checkpoints/bert/`

Configuration: `configs/bert_config.yaml`

### 4. Evaluation

```bash
python eval.py
```

Generates:
- Comparative metrics (Accuracy, F1, AUROC)
- ROC curves
- Confusion matrices
- Results table

Output saved to `results/`

## 📊 Models

### SeqXGPT
- **Architecture**: 1D CNN + Multi-Head Self-Attention + Attention Pooling
- **Input**: Log-probability features from LLM (GPT-2)
- **Features**: log_probs, surprisal, entropy
- **Strengths**: Captures sequential patterns in LLM outputs

### BERT Detector
- **Architecture**: Fine-tuned BERT-base
- **Input**: Raw text
- **Strengths**: Strong semantic understanding, transfer learning

## 🎯 Evasion Attacks

Test model robustness with text augmentation:

```python
from attacks.text_augmentation import TextAugmenter

augmenter = TextAugmenter()

# Paraphrasing
paraphrases = augmenter.paraphrase(text)

# Back-translation (en -> it -> en)
back_translated = augmenter.back_translate(text, intermediate_lang='it')
```

## ⚙️ Configuration

### SeqXGPT Config (`configs/seqxgpt_config.yaml`)
```yaml
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

### BERT Config (`configs/bert_config.yaml`)
```yaml
model:
  model_name: "bert-base-uncased"
  
training:
  batch_size: 16
  learning_rate: 0.00002
  num_epochs: 10
  max_length: 512
```

