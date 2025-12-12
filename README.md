# AI Text Detection: SeqXGPT + BERT Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This project presents a unified pipeline for detecting AI-generated text, comparing two state-of-the-art approaches:

*   **SeqXGPT-style detector** – A CNN + self-attention model leveraging GPT-2 token log-probabilities as features.
*   **BERT-based classifier** – A fine-tuned DistilBERT model for human vs. AI text classification.

The primary goal is to compare their performance, robustness, and reliability for critical applications such as **plagiarism detection, content moderation, and misinformation analysis**.


##  Features

This project offers:

*   A **complete training and evaluation pipeline** for both SeqXGPT and BERT detectors.
*   **Unified dataset loaders** with consistent preprocessing and data splits.
*   **Efficient log-probability feature extraction** from GPT-2 with optimized caching.
*   **Modular architecture** for clear separation of components.
*   **Comparative evaluation** on SeqXGPT-Bench, reporting accuracy, precision, recall, F1-score, and AUROC.
*   (Optional) **Robustness tests** via paraphrasing and back-translation.
*   **Exportable metrics** including JSON logs, ROC curves, confusion matrices, and model checkpoints.
*   **Production-ready inference** with a simple API for text classification.

**Key Innovations vs. Original SeqXGPT**:
*   Incorporated a BERT baseline for direct comparison.
*   Optimized feature extraction with batch processing (2x speedup).
*   Resolved critical NaN handling issues.
*   Achieved CPU-friendly training with DistilBERT (reduced from 15h to 15min).
*   Provided comprehensive documentation and FAQs.

---

## Quick Start

```bash
# 1. Clone the repository and set up the environment
git clone https://github.com/ecos01/Seqxgpt-mlsec-project.git
cd Seqxgpt-mlsec-project
python -m venv venv
.\venv\Scripts\Activate.ps1  # For Windows (use 'source venv/bin/activate' on Linux/macOS)
pip install -r requirements.txt

# 2. Verify your setup
python verify_setup.py

# 3. Train the models
python train_seqxgpt.py  # Estimated time: ~2.5h on CPU
python train_bert.py     # Estimated time: ~15min on CPU

# 4. Evaluate both models
python eval.py           # Runs comparative evaluation and generates reports
```

**Output**:
-   Trained models are saved in `checkpoints/`
-   Evaluation results and plots are found in `results/`

---

## Results Summary

### Performance Comparison (Test Set)

| Model | Accuracy | Precision | Recall | F1-Score | AUROC | Winner |
|-------|----------|-----------|--------|----------|-------|--------|
| **SeqXGPT** | **88.14%** | **92.23%** | 93.65% | **92.93%** | **91.45%** | Yes |
| BERT (DistilBERT) | 86.22% | 87.39% | **97.53%** | 92.18% | 88.41% | - |

**Key Takeaways**:
-   **SeqXGPT achieves superior overall performance** with higher precision (+4.84%), F1-score (+0.75%), and AUROC (+3.04%). This makes it ideal for applications where minimizing false positives is crucial (e.g., content moderation, plagiarism detection).
-   **BERT excels in recall** (+3.88%), effectively identifying 97.5% of AI-generated text, albeit with a higher rate of false positives. It is better suited for scenarios where missing AI-generated content is more costly than false alarms (e.g., security screening, spam filtering).

**Visualizations**: [ROC Curves](results/roc_curves.png) | [Confusion Matrices](results/confusion_matrices.png)

**For in-depth technical details, FAQs, and a study guide, refer to [SPIEGAZIONE.md](SPIEGAZIONE.md).**

---

## 1. Project Structure

```
SeqXGPT-MLSEC-Project/
├── data/                      # Dataset loaders and preprocessing scripts.
├── dataset/                   # Raw datasets, including SeqXGPT-Bench.
├── models/                    # Model architectures for SeqXGPT and BERT.
├── features/                  # Code for GPT-2 log-probability feature extraction and cache.
├── attacks/                   # Implementations of evasion attacks (paraphrasing, back-translation).
├── configs/                   # YAML configuration files for model hyperparameters.
├── checkpoints/               # Directory for saved model checkpoints.
├── results/                   # Stores evaluation results, plots, and metrics.
├── train_seqxgpt.py           # Script to train the SeqXGPT model.
├── train_bert.py              # Script to train the BERT-based detector.
├── eval.py                    # Script for comparative evaluation of trained models.
├── run_evasion_attacks.py     # Script to test model robustness against attacks.
├── verify_setup.py            # Utility for environment and dependency checks.
├── requirements.txt           # Python dependency list.
├── README.md                  # This document.
└── SPIEGAZIONE.md             # Detailed technical documentation (in Italian).
```

---

## 2. Installation and Setup

### 2.1 Requirements

-   **Python**: 3.8 or higher.
-   **Hardware**: A CPU is sufficient; a GPU is optional for faster training.
-   **OS**: Windows, Linux, or macOS.

### 2.2 Create Environment and Install Dependencies

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

### 2.3 Verify Setup

```bash
python verify_setup.py
```

This script verifies essential components: installed dependencies, dataset accessibility, GPU availability (if applicable), and correct model loading.

---

## 3. Datasets

### 3.1 SeqXGPT-Bench (Primary Dataset)

**Description**: A sentence-level benchmark dataset comprising human-written text and AI-generated text from various models (GPT-2, GPT-3, GPT-J, GPT-Neo, LLaMA).
**Labels**: `0` for human-written, `1` for AI-generated.
**Default Split**: 80% Train (28,722 samples), 10% Validation (3,591 samples), 10% Test (3,591 samples).
**Imbalance**: The dataset is imbalanced (approx. 83% AI, 17% human), reflecting real-world scenarios.

**Note**: All datasets originate from the official [SeqXGPT repository](https://github.com/Jihuai-wpy/SeqXGPT).

---

## 4. How It Works

### 4.1 GPT-2 Feature Extraction (for SeqXGPT)

The SeqXGPT pipeline transforms input text into numerical features based on GPT-2's predictions:

```
Input Text → GPT-2 Tokenization → For each token:
                                  ├─ log P(token|context)  [log-probability]
                                  ├─ -log P(token|context) [surprisal]
                                  └─ H(P)                  [entropy]
                                  
Features [batch, 256, 3] → Cached to disk → Used in training
```

These features capture the "statistical fingerprint" of AI-generated text, where AI models tend to produce more predictable tokens (higher log-probability, lower surprisal and entropy) compared to human writers. This process includes batch processing, automatic caching, and robust handling of numerical instabilities.

### 4.2 SeqXGPT-Style Model Architecture

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

### 4.3 BERT Detector Architecture

The BERT Detector utilizes a fine-tuned DistilBERT model (`distilbert-base-uncased`) for end-to-end binary classification:

*   **Base model**: DistilBERT, chosen for its efficiency (40% fewer parameters than BERT-base, 2x faster).
*   **Input**: Raw text, tokenized by DistilBERT's tokenizer.
*   **Output**: Logits for two classes (Human, AI), converted to probabilities via softmax.

DistilBERT's pre-trained knowledge allows it to learn patterns directly from raw text with fast inference and training times.

---

## 5. Training

### 5.1 Train SeqXGPT

```bash
python train_seqxgpt.py
```

This script trains the SeqXGPT model on the SeqXGPT-Bench dataset. It first extracts and caches GPT-2 features, normalizes them, and then trains the CNN+Attention model for up to 20 epochs. The best model, based on validation F1-score, is saved. Training takes approximately 2.5 hours on a CPU (including one-time feature extraction). For configuration details, see [`configs/seqxgpt_config.yaml`](configs/seqxgpt_config.yaml).

### 5.2 Train BERT

```bash
python train_bert.py
```

This script fine-tunes DistilBERT. To optimize for CPU training speed, it uses a stratified subset of 5,000 training samples. Due to DistilBERT's pre-trained weights, the model typically converges quickly, often within 1 epoch. The best model is saved in HuggingFace format. Training completes in about 15 minutes on a CPU. For configuration details, refer to [`configs/bert_config.yaml`](configs/bert_config.yaml).

---

## 6. Evaluation

### 6.1 Comparative Evaluation

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

## 7. Inference

Examples for classifying new text using the trained models.

### 7.1 SeqXGPT Inference

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

### 7.2 BERT Inference

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

## 8. Configuration

All hyperparameters are defined in YAML files within the [`configs/`](configs/) directory.

### 8.1 SeqXGPT Configuration Example

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

### 8.2 BERT Configuration Example

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

## 9. Detailed Results

This section provides a more in-depth look at the model performance beyond the initial summary.

### 9.1 SeqXGPT Validation Results Summary

The SeqXGPT model was trained for 20 epochs, achieving its best validation F1-score of **93.19%** at epoch 20, with an AUROC of 91.53%. Full epoch-wise metrics are available in [SPIEGAZIONE.md](SPIEGAZIONE.md). The model configuration and training process, including feature extraction from GPT-2 and normalization, are detailed in Section 4.1 and 5.1.

### 9.2 BERT Detector Validation Results Summary

The DistilBERT detector was fine-tuned efficiently, converging in just 1 epoch on a 5k sample subset. It achieved a validation F1-score of **92.42%** and an AUROC of 88.25%. This high performance with rapid convergence underscores the effectiveness of leveraging pre-trained transformer models. Detailed metrics are below:

| Metric     | Value    |
| :--------- | :------- |
| **Accuracy** | 0.8665  |
| **Precision**| 0.8762  |
| **Recall**   | 0.9778  |
| **F1-Score** | **0.9242** |
| **AUROC**    | 0.8825  |

### 9.3 Comparative Evaluation (Test Set Analysis)

The final evaluation on the imbalanced test set (16.7% Human, 83.3% AI) clearly highlights the strengths of each model.

| Model             | Accuracy | Precision | Recall | F1-Score | AUROC  |
| :---------------- | -------: | --------: | -----: | -------: | -----: |
| **SeqXGPT**       | **88.14%** | **92.23%** | 93.65% | **92.93%** | **91.45%** |
| **BERT (DistilBERT)** | 86.22% | 87.39% | **97.53%** | 92.18% | 88.41% |

**Analysis**:
*   **SeqXGPT's Advantage**: Demonstrates superior precision, F1-score, and AUROC. Its reliance on GPT-2 log-probability features allows it to capture subtle "statistical fingerprints" of AI-generated text, leading to fewer false positives. This makes it highly suitable for applications where misclassifying human text as AI is costly.
*   **BERT's Advantage**: Achieves exceptional recall, identifying nearly all AI-generated content. While its precision is slightly lower, its high recall makes it valuable in scenarios where the priority is to catch as much AI text as possible, even at the risk of some false alarms.

Both models offer robust detection capabilities, achieving F1-scores above 92%. The choice between them depends on the specific requirements of the application, particularly the trade-off between precision and recall.

---

## 10. Reproducibility

The project prioritizes reproducibility with:
-   A **fixed seed** (42) for consistent data splits.
-   **YAML configurations** for all hyperparameters.
-   **Saved checkpoints** with training statistics.
-   **Feature caching** for deterministic results.

To reproduce the results, simply run the training scripts (`train_seqxgpt.py`, `train_bert.py`) followed by the evaluation script (`eval.py`).

**System Info** (for reference): OS: Windows 10/11, Python: 3.8+, PyTorch: 2.0+, Hardware: CPU.

---

## 11. License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

This project is intended for **academic and research purposes**. For production use, ensure compliance with model licenses (GPT-2, DistilBERT), dataset terms of use, and ethical AI detection guidelines.

---

## 12. Acknowledgements

This project builds upon and extends the original [SeqXGPT work](https://arxiv.org/abs/2310.08903).

**Research**:
-   **SeqXGPT Authors** - Original paper and architecture.
-   [SeqXGPT GitHub Repository](https://github.com/Jihuai-wpy/SeqXGPT) - Datasets and baseline code.

**Frameworks & Libraries**: HuggingFace Transformers, PyTorch, scikit-learn.
**Models**: OpenAI GPT-2, DistilBERT.
**Contributors**: Eugenio (Project Lead), Sapienza University of Rome (Machine Learning Security Course).

**Citation**:

If you use this project in your research, please cite:

```bibtex
@misc{seqxgpt-mlsec-2025,
  author = {Eugenio},
  title = {AI Text Detection: SeqXGPT + BERT Pipeline},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/your-repo-url}
}
```

And the original SeqXGPT paper:

```bibtex
@article{wang2023seqxgpt,
  title={SeqXGPT: Sentence-Level AI-Generated Text Detection},
  author={Wang, Jihuai and others},
  journal={arXiv preprint arXiv:2310.08903},
  year={2023}
}
```
