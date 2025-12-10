# Project Overview - AI Text Detection Pipeline

## 📋 Summary

This is a complete, production-ready pipeline for comparing **SeqXGPT** and **BERT** models for AI-generated text detection.

## 🎯 What This Project Does

1. **Loads Data**: SeqXGPT-Bench dataset with texts from GPT-2, GPT-3, GPT-J, GPT-Neo, LLaMA, and humans
2. **Trains Two Models**:
   - SeqXGPT: Uses LLM log-probabilities + CNN + Self-Attention
   - BERT: Fine-tuned transformer for binary classification
3. **Evaluates**: Compares models with metrics (Accuracy, F1, AUROC) and visualizations
4. **Tests Robustness**: Includes evasion attacks (paraphrasing, translation)

## 📂 Clean Structure

```
SeqXGPT/
├── 📁 data/              Dataset loaders (Python classes)
├── 📁 dataset/           Raw data (JSONL files)
├── 📁 models/            Model architectures
├── 📁 features/          Feature extraction
├── 📁 attacks/           Evasion attacks
├── 📁 configs/           YAML configurations
├── 📁 checkpoints/       Saved models (created during training)
├── 📁 results/           Evaluation outputs (created during eval)
├── 📄 train_seqxgpt.py   Train SeqXGPT
├── 📄 train_bert.py      Train BERT
├── 📄 eval.py            Compare models
├── 📄 verify_setup.py    Check if everything works
├── 📄 requirements.txt   Python dependencies
├── 📄 README.md          Full documentation
└── 📄 SETUP.md           Quick start guide
```

## 🚀 Quick Start (3 Steps)

### 1. Install Dependencies
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Verify Setup
```powershell
python verify_setup.py
```

### 3. Train & Evaluate
```powershell
# Train models (can run in parallel)
python train_seqxgpt.py  # ~30-60 min on GPU
python train_bert.py     # ~20-40 min on GPU

# Compare results
python eval.py
```

## 📊 Expected Outputs

### During Training
- Progress bars with loss and accuracy
- Automatic early stopping when validation F1 stops improving
- Best model checkpoints saved automatically

### After Evaluation
- `results/results.json` - All metrics
- `results/results_table.txt` - Formatted comparison table
- `results/roc_curves.png` - ROC curves
- `results/confusion_matrices.png` - Confusion matrices

## ⚙️ Configuration

All hyperparameters in `configs/*.yaml`:

**SeqXGPT** (`configs/seqxgpt_config.yaml`):
- Model: hidden_dim, CNN layers, attention heads
- Training: batch_size, learning_rate, epochs
- LLM: which model to use for features (gpt2, gpt2-medium, etc.)

**BERT** (`configs/bert_config.yaml`):
- Model: bert-base-uncased, roberta-base, etc.
- Training: batch_size, learning_rate, epochs

## 🔬 Advanced Usage

### Test Individual Components
```powershell
python data/seqxgpt_dataset.py    # Test dataset loader
python features/llm_probs.py      # Test feature extraction
python models/seqxgpt.py          # Test SeqXGPT model
python models/bert_detector.py    # Test BERT model
```

### Use Different Datasets
```powershell
# Add your CSV/JSONL in data/
# Use extra_dataset.py loader
```

### Test Robustness
```powershell
python attacks/text_augmentation.py
```

## 💡 Key Features

✅ **Modular Design**: Each component can be tested independently
✅ **Automatic Caching**: LLM features cached to speed up experiments
✅ **Early Stopping**: Prevents overfitting automatically
✅ **GPU Support**: Automatically uses GPU if available
✅ **Flexible**: Easy to add new models or datasets
✅ **Well Documented**: Clear code with docstrings

## 📝 For Your Report/Presentation

Include:
1. **Methods**: SeqXGPT (log-probs + CNN) vs BERT (fine-tuning)
2. **Dataset**: SeqXGPT-Bench (6 sources, train/val/test split)
3. **Metrics**: Accuracy, Precision, Recall, F1, AUROC
4. **Results**: Comparison table from `results/`
5. **Robustness**: Results with paraphrasing/translation
6. **Analysis**: Which model works better and why

## 🐛 Troubleshooting

**Problem**: Out of memory
**Solution**: Reduce batch_size in configs, use smaller LLM (gpt2 vs gpt2-medium)

**Problem**: Slow feature extraction
**Solution**: Features are cached after first run, delete `features/cache/` to recompute

**Problem**: Import errors
**Solution**: Run `python verify_setup.py` to check everything

## 📧 Support

For issues:
1. Run `python verify_setup.py`
2. Check error messages
3. Review SETUP.md for detailed instructions

---

**Ready to start?** Run `python verify_setup.py` to check everything is working!
