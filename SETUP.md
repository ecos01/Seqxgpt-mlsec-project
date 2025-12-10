# Quick Setup Guide

## Initial Setup (run once)

```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## Training

```powershell
# Train SeqXGPT model
python train_seqxgpt.py

# Train BERT model
python train_bert.py
```

## Evaluation

```powershell
# Run comparative evaluation
python eval.py
```

## Testing Components

```powershell
# Test dataset loader
python data/seqxgpt_dataset.py

# Test feature extraction
python features/llm_probs.py

# Test models
python models/seqxgpt.py
python models/bert_detector.py
```

## Project Structure

```
SeqXGPT/
├── data/          # Dataset loaders
├── dataset/       # Raw data (SeqXGPT-Bench)
├── models/        # Model implementations
├── features/      # Feature extraction
├── attacks/       # Evasion attacks
├── configs/       # YAML configurations
├── checkpoints/   # Saved models (created during training)
├── results/       # Evaluation outputs (created during eval)
├── train_*.py     # Training scripts
└── eval.py        # Evaluation script
```

## Configuration

Edit YAML files in `configs/` to change hyperparameters:
- `seqxgpt_config.yaml` - SeqXGPT model settings
- `bert_config.yaml` - BERT model settings

## GPU Usage

The scripts automatically use GPU if available. To force CPU:
```python
# In config YAML, or set environment variable:
$env:CUDA_VISIBLE_DEVICES = "-1"
```

## Troubleshooting

**Out of memory?**
- Reduce `batch_size` in config files
- Use `gpt2` instead of `gpt2-medium` for feature extraction

**Slow training?**
- Features are cached in `features/cache/` after first extraction
- To recompute: set `force_recompute_features: true` in config
