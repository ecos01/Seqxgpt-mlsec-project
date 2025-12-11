# AI Text Detection: SeqXGPT + BERT Pipeline

This repository implements a turnkey pipeline for **AI-generated text detection**, starting from the original [SeqXGPT](https://arxiv.org/abs/2310.08903) work.[1][2]
The goal is to compare a **SeqXGPT-style detector** against a **BERT-based classifier** on the same benchmark and study their behavior in a security-relevant setting (plagiarism, spam, misinformation, etc.).[2][3][1]

## 1. Overview

The project provides:

- A **SeqXGPT-style model**: CNN + self-attention operating on token-level log-probabilities from GPT‑2.[1][2]
- A **BERT detector**: fine-tuned `bert-base-uncased` for binary classification (human vs AI).   
- A **unified training and evaluation pipeline** on **SeqXGPT-Bench**.[4][2][1]
- Optional **robustness / evasion tests** via paraphrasing and round-trip translation.

This addresses a machine-learning security problem: **reliably detecting AI-generated text** to support use cases such as plagiarism detection, content moderation, and misinformation control.[3][2][1]

***

## 2. Project Structure

```text
SeqXGPT-MLSEC-Project/
├── data/                      # Dataset loaders
│   ├── seqxgpt_dataset.py     # SeqXGPT-Bench loader
│   └── extra_dataset.py       # Generic dataset loader (optional)
├── dataset/                   # Raw datasets (from original SeqXGPT repo)
│   ├── SeqXGPT-Bench/         # Main benchmark (sentence-level)
│   ├── document-level detection dataset/     # optional
│   └── OOD sentence-level detection dataset/ # optional
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

The raw datasets under `dataset/` are taken from the official SeqXGPT repository.[5][4]

***

## 3. Installation and Setup

### 3.1 Create environment and install dependencies

```bash
# (optional) create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/macOS
# .\venv\Scripts\Activate.ps1  # Windows PowerShell

# install requirements
pip install -r requirements.txt
```

Dependencies include:

- `torch`, `torchvision` (if needed)  
- `transformers`, `datasets`  
- `scikit-learn`  
- `pyyaml`  
- `numpy`, `tqdm`, `matplotlib`  

### 3.2 Verify setup

```bash
python verify_setup.py
```

This script checks:

- that all required Python packages are installed,  
- that the expected project structure is present,  
- that `SeqXGPT-Bench` JSONL files exist and can be read,  
- that key modules (dataset loaders, models, feature extractor, attacks) can be imported.

If all checks pass, the project is ready to use.

***

## 4. Datasets

### 4.1 Main benchmark: SeqXGPT-Bench

- Location: `dataset/SeqXGPT-Bench/`  
- Content: sentence-level texts from multiple generators and human authors: GPT‑2, GPT‑3, GPT‑J, GPT‑Neo, LLaMA, and human-written samples.[2][1]
- Typical file names:
  - `en_human_lines.jsonl`  
  - `en_gpt2_lines.jsonl`  
  - `en_gpt3_lines.jsonl`  
  - `en_gptj_lines.jsonl`  
  - `en_gptneo_lines.jsonl`  
  - `en_llama_lines.jsonl`  

The loader `SeqXGPTDataset` (in `data/seqxgpt_dataset.py`) merges these files and builds a binary classification dataset (label 0 = human, 1 = AI).[4][1][2]

By default, an **80 / 10 / 10** split (train/val/test) is created with a fixed random seed.

### 4.2 Optional datasets

The additional folders:

- `dataset/document-level detection dataset/`  
- `dataset/OOD sentence-level detection dataset/`  

are kept from the original repository and can be wired through `extra_dataset.py` if cross-domain or out-of-distribution experiments are desired.[2][4]

***

## 5. Project Plan and Methodology

### 5.1 Setup

The repository is organized into modular components:

- `data/` – data loading and splitting utilities  
- `features/llm_probs.py` – GPT‑2-based feature extraction  
- `models/seqxgpt.py` – SeqXGPT-style CNN + self-attention model[1][2]
- `models/bert_detector.py` – BERT-based binary classifier   
- `train_seqxgpt.py`, `train_bert.py` – training entry points  
- `eval.py` – unified evaluation and comparison script  
- `configs/` – hyperparameter configurations

The environment relies on Python 3.x, PyTorch, HuggingFace Transformers, and scikit-learn.

### 5.2 LLM feature extraction (SeqXGPT-style)

File: `features/llm_probs.py`

- Loads a causal language model (default: `gpt2`) via `transformers`.  
- For each input sentence:
  - tokenizes with the GPT‑2 tokenizer;  
  - computes per-token log-probabilities / surprisal given the left context;  
  - pads or truncates to a fixed maximum length (e.g., 256 tokens).[1][2]

The resulting tensor has shape `(batch_size, seq_len, feature_dim)` and is used as input to the SeqXGPT-style model.

Features can optionally be cached on disk to speed up repeated training runs.

### 5.3 SeqXGPT-style model

File: `models/seqxgpt.py`

Architecture (following the SeqXGPT idea):[2][1]

- 1D convolutional layers over the feature sequence  
- Multi-head self-attention layer  
- Global pooling (attention-weighted or max/mean)  
- Final MLP classifier for binary output (human vs AI)

Training script: `train_seqxgpt.py`

- Loads data via `SeqXGPTDataset` and applies GPT‑2 feature extraction.  
- Uses `CrossEntropyLoss` and Adam/AdamW optimizer.  
- Tracks accuracy, F1, and optionally AUROC on the validation set.  
- Saves the best-performing model to `checkpoints/seqxgpt/`.

### 5.4 BERT-based detector

File: `models/bert_detector.py`

- Wraps `transformers.AutoModelForSequenceClassification` with `bert-base-uncased` as default backbone.   
- Uses standard BERT tokenization (max length, padding, attention mask).  

Training script: `train_bert.py`

- Loads the same (text, label) pairs and tokenizes on-the-fly.  
- Uses `CrossEntropyLoss` and AdamW with a small learning rate (e.g., 2e‑5).  
- Shares the same train/val/test split as SeqXGPT for fair comparison.  

Best checkpoints are stored in `checkpoints/bert/`.

### 5.5 Base evaluation

File: `eval.py`

- Loads trained SeqXGPT and BERT detectors.  
- Evaluates both models on the test split(s) and reports:
  - accuracy, precision, recall, F1, AUROC.[1][2]
- Optionally produces:
  - ROC curves,  
  - confusion matrices,  
  - a Markdown/CSV summary table in `results/`.

Example table:

| Model   | Dataset        | Acc | F1  | AUROC |
|--------|----------------|-----|-----|-------|
| SeqXGPT | SeqXGPT-Bench |  –  |  –  |   –   |
| BERT    | SeqXGPT-Bench |  –  |  –  |   –   |

### 5.6 Generalization and optional evasion

Optionally, the project can be extended to:

- **Generalization to other generators**:  
  Evaluate detectors trained on SeqXGPT-Bench on texts generated by a different LLM (e.g., another open-source model) to study robustness under generator shift.[6][2][1]

- **Evasion / robustness tests** (file: `attacks/text_augmentation.py`):  
  - Paraphrasing using a light paraphrasing model or sequence-to-sequence transformer.  
  - Back-translation (e.g., English → Italian → English).  
  The performance drop compared to the original texts quantifies robustness against simple rewriting strategies.[7]

***

## 6. Usage

### 6.1 Quick start

1. **Verify setup**

```bash
python verify_setup.py
```

2. **Train SeqXGPT-style detector**

```bash
python train_seqxgpt.py
```

3. **Train BERT detector**

```bash
python train_bert.py
```

4. **Run comparative evaluation**

```bash
python eval.py
```

Metrics and plots are written to `results/`.

### 6.2 Text augmentation and evasion tests

Example usage:

```python
from attacks.text_augmentation import TextAugmenter

augmenter = TextAugmenter()

text = "This is an example AI-generated sentence."

# Paraphrasing
para = augmenter.paraphrase(text)

# Back-translation (en -> it -> en)
bt = augmenter.back_translate(text, intermediate_lang="it")
```

These transformed texts can be fed to the trained detectors to measure robustness.

***

## 7. Configuration

### SeqXGPT config (`configs/seqxgpt_config.yaml`)

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

### BERT config (`configs/bert_config.yaml`)

```yaml
model:
  model_name: "bert-base-uncased"

training:
  batch_size: 16
  learning_rate: 0.00002
  num_epochs: 10
  max_length: 512
```

Hyperparameters can be adjusted to match available compute resources.

***

## 8. Reproducibility

- Random seeds are set in the training scripts to improve reproducibility of results.  
- Checkpoints are saved under `checkpoints/` with informative names (model type, epoch, best metric).  
- Configurations live under `configs/` and are tracked in version control.  
- Evaluation scripts produce machine-readable logs (CSV/JSON) in `results/`.

***

## 9. License

Add your chosen license here (e.g., MIT) and include a `LICENSE` file in the root directory.

***

## 10. Acknowledgements

- **SeqXGPT**: Jihuai Wang et al., “SeqXGPT: Sentence-Level AI-Generated Text Detection”.[8][2][1]
- The original **SeqXGPT repository**, which provides datasets and baseline code.[5][4]
- HuggingFace Transformers and the PyTorch ecosystem for tooling and models.   

