# AI Text Detection: SeqXGPT + BERT Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Machine Learning Security Project** - A unified pipeline for detecting AI-generated text using two complementary approaches:

* **SeqXGPT-style detector** – CNN + self-attention over GPT-2 token log-probabilities
* **BERT-based classifier** – Fine-tuned transformer (DistilBERT) for human vs AI classification

Compare performance, robustness, and reliability for applications such as **plagiarism detection, content moderation, and misinformation analysis**.


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

### 4.1 SeqXGPT-Bench (Primary Dataset)

**Description**: Sentence-level benchmark containing human-written and AI-generated text from multiple models.

**Sources**:
- Human-written text
- GPT-2 generated
- GPT-3 generated
- GPT-J generated
- GPT-Neo generated
- LLaMA generated

**Labels**: 
- `0` = Human-written
- `1` = AI-generated (all AI models)

**Default Split**: 
- Train: 80% (28,722 samples)
- Validation: 10% (3,591 samples)
- Test: 10% (3,591 samples)

**Loader**: `SeqXGPTDataset` class in `data/seqxgpt_dataset.py`

**Imbalanced Classes**: ~83% AI, ~17% Human (reflects real-world scenario)

### 4.2 Optional Datasets

* **Document-level detection dataset** - Longer texts (paragraphs/documents)
* **OOD sentence-level dataset** - Out-of-distribution test set

**Note**: All datasets are from the original [SeqXGPT repository](https://github.com/Jihuai-wpy/SeqXGPT).

---

## 5. How It Works

### 5.1 GPT-2 Feature Extraction

**SeqXGPT Pipeline**:

```
Input Text → GPT-2 Tokenization → For each token:
                                    ├─ log P(token|context)  [log-probability]
                                    ├─ -log P(token|context) [surprisal]
                                    └─ H(P)                  [entropy]
                                    
Features [batch, 256, 3] → Cache to disk → Use in training
```

**Why These Features?**
- **Log-probability**: Measures token predictability
  - AI text: High probability (more predictable)
  - Human text: More variable
- **Surprisal**: Information content (`-log P`)
  - AI: Low surprisal (expected tokens)
  - Human: High surprisal (creative choices)
- **Entropy**: Uncertainty of distribution
  - AI: Low entropy (confident predictions)
  - Human: High entropy (more ambiguous)

**Implementation**: `features/llm_probs.py`
- Batch processing (16-32 samples)
- Automatic caching
- NaN/Inf handling with clipping

### 5.2 SeqXGPT-Style Model

**Architecture**:

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
Sigmoid → Probability [0, 1]
```

**Key Components**:
- **CNN**: Captures local patterns (n-grams)
- **Self-Attention**: Captures long-range dependencies
- **Attention Pooling**: Model learns which positions are important
- **Parameters**: 225,922 (lightweight!)

**Training Script**: `train_seqxgpt.py`

### 5.3 BERT Detector

**Architecture**:
- Base model: DistilBERT (distilbert-base-uncased)
- Fine-tuned for binary classification
- Input: Raw text → BERT tokenization
- Output: 2 classes (Human, AI) → Softmax

**Why DistilBERT?**
- 40% fewer parameters than BERT-base (66M vs 110M)
- 2x faster inference and training
- 97% of BERT performance
- Perfect for CPU training

**Training Script**: `train_bert.py`

### 5.4 Evaluation

**Metrics Computed**:
- **Accuracy**: Overall correctness
- **Precision**: Of predicted AI, how many are truly AI?
- **Recall**: Of all AI texts, how many did we catch?
- **F1-Score**: Harmonic mean of precision and recall
- **AUROC**: Area Under ROC Curve (discrimination quality)

**Plots Generated**:
- ROC curves (True Positive Rate vs False Positive Rate)
- Confusion matrices (visual breakdown of predictions)

---

## 6. Training

### 6.1 Train SeqXGPT

```bash
python train_seqxgpt.py
```

**Process**:
1. ✅ Loads SeqXGPT-Bench dataset (28,722 train, 3,591 val)
2. ✅ Extracts GPT-2 features (log-prob, surprisal, entropy) → Cached for reuse
3. ✅ Normalizes features (z-score + clipping [-5, +5])
4. ✅ Trains CNN + Attention model for 20 epochs
5. ✅ Saves best model based on validation F1 to `checkpoints/seqxgpt/best_model.pt`

**Training Time**: ~2.5 hours on CPU
- Feature extraction: ~1.5 hours (only once, then cached)
- Training: ~1 hour (20 epochs × 3 min/epoch)

**Configuration**: [`configs/seqxgpt_config.yaml`](configs/seqxgpt_config.yaml)

**Key Optimizations**:
- Feature caching (no re-extraction)
- Gradient clipping (prevents NaN)
- Early stopping (patience=5 epochs)
- LR scheduler (ReduceLROnPlateau)

### 6.2 Train BERT

```bash
python train_bert.py
```

**Process**:
1. ✅ Loads 5k train samples (stratified subset for speed on CPU)
2. ✅ Fine-tunes DistilBERT for binary classification (Human vs AI)
3. ✅ Early stopping at epoch 1 (converged due to pre-training)
4. ✅ Saves model to `checkpoints/bert/best_model/` (HuggingFace format)

**Training Time**: ~15 minutes on CPU

**Configuration**: [`configs/bert_config.yaml`](configs/bert_config.yaml)

**Optimizations for CPU**:
- DistilBERT instead of BERT (2x faster)
- Reduced samples: 5k/28k (stratified)
- Shorter sequences: 256 tokens (vs 512)
- Larger batch size: 32 (efficiency)

**Performance Note**: Despite using only 5k samples, achieves F1=92.42% (excellent!)

---

## 7. Evaluation

### 7.1 Comparative Evaluation

```bash
python eval.py
```

**What It Does**:
- Loads both trained models (SeqXGPT and BERT)
- Evaluates on test set (3,591 samples)
- Computes all metrics for both models
- Generates visualizations

**Output**:
- 📊 Comparative metrics table (printed to console)
- 📈 ROC curves: `results/roc_curves.png`
- 📊 Confusion matrices: `results/confusion_matrices.png`
- 💾 JSON logs with detailed metrics

**Metrics Computed**:
- Accuracy
- Precision
- Recall
- F1-Score
- AUROC
- Confusion Matrix

### 7.2 Evasion Attacks (Optional)

Test model robustness against adversarial modifications:

```bash
python run_evasion_attacks.py
```

**Attack Methods**:
- **Paraphrasing**: Rephrase AI text using T5 model
- **Back-translation**: en→it→en to change style while preserving meaning

**Code Example**:

```python
from attacks.text_augmentation import TextAugmenter

augmenter = TextAugmenter()

# Original AI text
original = "The quick brown fox jumps over the lazy dog."

# Apply attacks
paraphrased = augmenter.paraphrase(original)
back_translated = augmenter.back_translate(original, intermediate_lang="it")

# Test if detector still catches it
# ...
```

**Purpose**: Measure performance drop under evasion (not included in main results).

---

## 8. Inference

### 8.1 SeqXGPT Inference

```python
import torch
import yaml
from models.seqxgpt import SeqXGPTModel
from features.llm_probs import LLMProbExtractor

# Load model
with open("configs/seqxgpt_config.yaml") as f:
    config = yaml.safe_load(f)

checkpoint = torch.load("checkpoints/seqxgpt/best_model.pt", map_location="cpu", weights_only=False)
model = SeqXGPTModel(**config['model'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load feature extractor
extractor = LLMProbExtractor(
    model_name=config['llm']['model_name'],
    max_length=config['llm']['max_length']
)

# Extract features
text = "Your text to classify here."
features, mask = extractor.extract_single(text)

# Normalize features (CRITICAL!)
feature_mean = checkpoint['feature_mean']
feature_std = checkpoint['feature_std']
features = (features - feature_mean) / feature_std
features = torch.clamp(features, -5, 5)

# Predict
with torch.no_grad():
    prob = model.predict(features.unsqueeze(0), mask.unsqueeze(0))
    pred = 1 if prob.item() > 0.5 else 0

print(f"Prediction: {'AI-generated' if pred == 1 else 'Human-written'}")
print(f"Confidence: {prob.item():.4f}")
```

### 8.2 BERT Inference

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("checkpoints/bert/best_model")
tokenizer = AutoTokenizer.from_pretrained("checkpoints/bert/best_model")
model.eval()

# Tokenize and predict
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

### 8.3 Production API Example

```python
class AITextDetector:
    def __init__(self, model_type="seqxgpt"):
        """Load model at initialization."""
        self.model_type = model_type
        if model_type == "seqxgpt":
            # Load SeqXGPT (see above)
            pass
        else:
            # Load BERT (see above)
            pass
    
    def predict(self, text):
        """Classify text as human (0) or AI (1)."""
        # Implementation...
        return {
            "prediction": 1,  # or 0
            "confidence": 0.95,
            "label": "AI-generated"  # or "Human-written"
        }

# Usage
detector = AITextDetector(model_type="seqxgpt")
result = detector.predict("This is a test sentence.")
print(result)
```

---

## 9. Configuration

Configuration files in [`configs/`](configs/) directory define all hyperparameters. Edit these files to experiment with different settings.

### 9.1 SeqXGPT Configuration

**File**: [`configs/seqxgpt_config.yaml`](configs/seqxgpt_config.yaml)

```yaml
model:
  input_dim: 3                    # 3 features: log-prob, surprisal, entropy
  hidden_dim: 128                 # Hidden dimension for CNN/Attention
  num_cnn_layers: 3               # Number of 1D CNN layers
  num_attention_heads: 4          # Multi-head attention heads
  dropout: 0.1                    # Dropout rate

training:
  batch_size: 16                  # Training batch size
  learning_rate: 0.00005          # 5e-5, reduced from 1e-4
  num_epochs: 20                  # Maximum epochs
  early_stopping_patience: 5      # Stop if no improvement for 5 epochs
  gradient_clip_max_norm: 1.0     # Gradient clipping (critical!)

llm:
  model_name: "gpt2"              # GPT-2 for feature extraction
  max_length: 256                 # Max sequence length
  cache_dir: "features/cache"     # Where to cache extracted features

data:
  data_dir: "dataset/SeqXGPT-Bench"
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  seed: 42                        # Fixed seed for reproducibility
```

### 9.2 BERT Configuration

**File**: [`configs/bert_config.yaml`](configs/bert_config.yaml)

```yaml
model:
  model_name: "distilbert-base-uncased"  # 2x faster than BERT-base
  num_labels: 2                          # Binary classification
  dropout: 0.1

training:
  batch_size: 32                         # Larger for efficiency
  learning_rate: 0.00003                 # 3e-5 (standard for BERT)
  num_epochs: 3                          # Few epochs needed
  max_length: 256                        # Reduced from 512 for speed
  early_stopping_patience: 1             # Aggressive early stopping
  max_train_samples: 5000                # Subset for fast CPU training
  max_val_samples: 1000                  # Subset for validation
  gradient_accumulation_steps: 2         # Simulate larger batch

data:
  data_dir: "dataset/SeqXGPT-Bench"
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  seed: 42
```

**Tip**: Increase `max_train_samples` to 28722 (full dataset) if you have GPU or more time.

---

## 10. Detailed Results

### 10.1 SeqXGPT Validation Results

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

### 10.2 BERT Detector Results

**Model Configuration**:
| :-------- | :---- |
| Model | DistilBERT (distilbert-base-uncased) |
| Parameters | ~66M |
| Max Length | 256 tokens |
| Batch Size | 32 |
| Learning Rate | 3e-5 |
| Training Samples | 5,000 (subset) |
| Validation Samples | 1,000 |
| Epochs Completed | 1 (early stopping) |

**Why DistilBERT?** 
- 2x faster than BERT-base on CPU
- Similar performance (97% of BERT quality)
- Training time: 15h → 15min

**Validation Metrics** (on 1k validation samples):

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

### 10.3 Comparative Evaluation (Test Set)

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

## 11. Reproducibility

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

## 12. License

MIT License - See [LICENSE](LICENSE) file for details.

This project is for **academic and research purposes**. If using in production, ensure compliance with:
- Model licenses (GPT-2, DistilBERT)
- Dataset terms of use
- Ethical AI detection guidelines

---

## 13. Acknowledgements

This project builds upon and extends the original [SeqXGPT work](https://arxiv.org/abs/2310.08903):

**Research**:
- **SeqXGPT authors** - Original paper and architecture
- [SeqXGPT GitHub Repository](https://github.com/Jihuai-wpy/SeqXGPT) - Datasets and baseline code

**Frameworks & Libraries**:
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) - BERT models and tokenizers
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [scikit-learn](https://scikit-learn.org/) - Metrics and evaluation tools

**Models**:
- OpenAI GPT-2 - Feature extraction
- DistilBERT - Efficient transformer baseline

**Contributors**:
- Eugenio (Project Lead)
- Sapienza University of Rome - Machine Learning Security Course

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

---

**Questions?** Open an issue or see [SPIEGAZIONE.md](SPIEGAZIONE.md) for detailed FAQ and technical documentation.

**Star ⭐ this repo** if you find it useful!
