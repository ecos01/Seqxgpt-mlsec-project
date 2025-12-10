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

## 📈 Expected Results

| Model      | Dataset         | Acc   | F1    | AUROC |
|-----------|-----------------|-------|-------|-------|
| SeqXGPT   | SeqXGPT-Bench   | 0.85+ | 0.83+ | 0.90+ |
| BERT      | SeqXGPT-Bench   | 0.88+ | 0.86+ | 0.92+ |

*Note: Results may vary based on dataset split and hyperparameters*

## 🔬 Experiments

### 1. Basic Evaluation
```bash
python eval.py
```

### 2. Cross-Generator Testing
Test on texts from unseen generators (e.g., newer models)

### 3. Robustness Testing
Evaluate against augmented/paraphrased texts:
```bash
python attacks/text_augmentation.py
```

## 📚 Dataset Formats

### SeqXGPT-Bench (JSONL)
```json
{"text": "Sample text here..."}
```

### Custom Dataset (CSV)
```csv
text,label
"Human written text",0
"AI generated text",1
```

Use `data/extra_dataset.py` for custom datasets.

## 🛠️ Development

### Test Individual Components

```bash
# Test dataset loader
python data/seqxgpt_dataset.py

# Test feature extraction
python features/llm_probs.py

# Test models
python models/seqxgpt.py
python models/bert_detector.py

# Test augmentation
python attacks/text_augmentation.py
```

## 📝 Citation

If you use this pipeline, please cite:

```bibtex
@article{seqxgpt2023,
  title={SeqXGPT: Sentence-Level AI-Generated Text Detection},
  author={Wang, Pengyu and others},
  journal={arXiv preprint arXiv:2310.08903},
  year={2023}
}
```

## 📄 License

See LICENSE file for details

## 💡 Tips

- **GPU Recommended**: Training is much faster on GPU
- **Memory**: BERT requires ~8GB GPU memory with batch_size=16
- **Feature Caching**: LLM features are cached automatically to `features/cache/`
- **Early Stopping**: Training stops automatically when validation F1 stops improving

---

## Original SeqXGPT Paper

For more details about the original SeqXGPT method, please refer to the [paper](https://arxiv.org/abs/2310.08903).

## Open-Source List

We list all the datasets, models, training and testing codes related to SeqXGPT [here](https://github.com/Jihuai-wpy/SeqXGPT/tree/main/SeqXGPT). 

## Performance

All the values listed in our table are **F1 scores**, and **Macro-F1 scores** to consider the overall performance. For detailed precision and recall scores, please refer to our [paper](https://arxiv.org/abs/2310.08903).

### Results of Particular-Model Binary AIGT Detection

SeqXGPT performs much better than two zero-shot methods and supervised method Sent-RoBERTa.

| Method           |  GPT-2   |  Human   | Macro-F1 | Method           | GPT-Neo  |  Human   | Macro-F1 |
| ---------------- | :------: | :------: | :------: | ---------------- | :------: | :------: | :------: |
| **$log$ $p(x)$** |   78.4   |   47.9   |   63.1   | **$log$ $p(x)$** |   73.9   |   41.2   |   57.5   |
| **DetectGPT**    |   65.8   |   42.9   |   54.3   | **DetectGPT**    |   57.6   |   41.3   |   49.4   |
| **Sent-RoBERTa** |   92.9   |   75.8   |   84.4   | **Sent-RoBERTa** |   92.6   |   73.4   |   83.0   |
| **SeqXGPT**      | **98.6** | **95.8** | **97.2** | **SeqXGPT**      | **98.8** | **96.4** | **97.6** |

| Method           |  GPT-J   |  Human   | Macro-F1 | Method           |  LLaMA   |  Human   | Macro-F1 |
| ---------------- | :------: | :------: | :------: | ---------------- | :------: | :------: | :------: |
| **$log$ $p(x)$** |   76.5   |   34.4   |   55.5   | **$log$ $p(x)$** |   69.1   |   27.1   |   48.1   |
| **DetectGPT**    |   66.8   |   37.0   |   51.9   | **DetectGPT**    |   52.8   |   47.6   |   50.2   |
| **Sent-RoBERTa** |   93.1   |   71.8   |   82.4   | **Sent-RoBERTa** |   89.7   |   69.6   |   79.6   |
| **SeqXGPT**      | **97.9** | **92.9** | **95.4** | **SeqXGPT**      | **96.0** | **89.9** | **92.9** |

### Results of Mixed-Model Binary AIGT Detection

SeqXGPT shows the best performance among these four methods. In contrast, the performance of Sniffer is noticeably inferior, which emphasizes that document-level AIGT detection methods cannot be effectively modified for sentence-level AIGT detection. Interestingly, we find that the performance of both RoBERTa-based methods is slightly inferior to SeqXGPT in overall performance. This suggests that the semantic features of RoBERTa might be helpful to discriminate human-created sentences.

| Method           |    AI    |  Human   | Macro-F1 |
| ---------------- | :------: | :------: | :------: |
| **Sniffer**      |   87.7   |   54.3   |   71.0   |
| **Sent-RoBERTa** | **97.6** |   92.6   |   95.1   |
| **Seq-RoBERTa**  |   97.4   |   91.8   |   94.6   |
| **SeqXGPT**      | **97.6** | **92.9** |   95.3   |

### Results of Mixed-Model Multiclass AIGC Detection

SeqXGPT can accurately discriminate sentences generated by various models and those authored by humans, demonstrating its strength in multi-class detection. It is noteworthy that RoBERTa-based methods perform significantly worse than binary AIGT detection.

| Method           |  GPT-2   | GPT-Neo  |  GPT-J   |  LLaMA   |  GPT-3   |  Human   | Macro-F1 |
| :--------------- | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| **Sniffer**      |   51.5   |   45.5   |   36.0   |   23.1   |   54.1   |   58.1   |   44.7   |
| **Sent-RoBERTa** |   43.1   |   31.6   |   31.5   |   42.4   |   78.2   |   90.5   |   52.9   |
| **Seq-RoBERTa**  |   55.5   |   36.7   |   32.0   |   78.6   | **94.4** |   92.3   |   64.9   |
| **SeqXGPT**      | **98.5** | **98.7** | **97.2** | **93.2** |   93.9   | **92.9** | **95.7** |

### Results of Document-Level AIGT Detection

sentence-level detection methods can be transformed and directly applied to document-level detection, and the performance is positively correlated with their performance on sentence-level detection. Overall, SeqXGPT exhibits excellent performance in document-level detection.

| Method           |  GPT-2   | GPT-Neo  |  GPT-J   |  LLaMA   |  GPT-3   |  Human   | Macro-F1 |
| :--------------- | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| **Sniffer**      |   85.4   |   84.5   |   74.6   |   13.0   |   81.3   |   66.2   |   67.5   |
| **Sent-RoBERTa** |   55.8   |   42.8   |   24.4   |   18.3   |   84.7   | **94.6** |   53.4   |
| **Seq-RoBERTa**  |   63.4   |   41.0   |   32.6   |   67.0   | **91.9** |   51.2   |   57.9   |
| **SeqXGPT**      | **99.5** | **99.5** | **98.0** | **93.3** |   90.4   |   84.8   | **94.2** |

### Results of Out-of-Distribution Sentence-Level AIGT Detection

The great performance of SeqXGPT on OOD data reflects the strong generalization capabilities of SeqXGPT.

| Method           |  GPT-2   | GPT-2-Neo |  GPT-J   |  LLaMA   |  GPT-3   |  Human   | Macro-F1 |
| ---------------- | :------: | :-------: | :------: | :------: | :------: | :------: | :------: |
| **Sniffer**      |   7.8    |   50.8    |   28.3   |   22.9   |   61.9   |   44.8   |   36.1   |
| **Sent-RoBERTa** |   32.5   |   18.2    |   24.8   |   30.8   |   74.6   |   30.5   |   35.2   |
| **Seq-RoBERTa**  |   53.7   |   29.0    |   29.9   |   75.2   |   92.3   |   83.6   |   60.6   |
| **SeqXGPT**      | **98.9** | **90.7**  | **95.2** | **90.3** | **93.7** | **88.2** | **92.8** |

## Contribution

In this paper, we first introduce the challenge of sentence-level AIGT detection and propose three tasks based on existing research in AIGT detection. Further, we introduce a strong approach, SeqXGPT, as well as a benchmark to solve this challenge. Through extensive experiments, our proposed SeqXGPT can obtain promising results in both sentence and document-level AIGT detection. On the OOD testset, SeqXGPT also exhibits strong generalization. We hope that SeqXGPT will inspire future research in AIGT detection, and may also provide insightful references for the detection of content generated by models in other fields.

## Limitations

Despite SeqXGPT exhibits excellent performance in both sentence and document-level AIGT detection challenge and exhibits strong generalization, it still present certain limitations:

1. We did not incorporate semantic features, which could potentially assist our model further in the sentence recognition process, particularly in cases involving human-like sentence generation. We leave this exploration for future work.
2. During the construction of GPT-3.5-turbo data, we did not extensively investigate the impact of more diversified instructions. Future research will dive into exploring the influence of instructions on AIGT detection.
3. Due to limitations imposed by the model's context length and generation patterns, our samples only consist of two distinct sources of sentences. In future studies, we aim to explore more complex scenarios where a document contains sentences from multiple sources.

## Acknowledgements

- We sincerely thank Qipeng Guo for his valuable discussions and insightful suggestions.
- This work was supported by the National Natural Science Foundation of China (No. 62236004 and No. 62022027).
- This work's logo was initially generated using OpenAI's [DALL-E 3](https://openai.com/research/dall-e-3-system-card) and further refined by Youqi Sun. Special thanks to both.

## Citation

If you find SeqXGPT useful for your research and applications, please cite using the Bibtex:

```
@misc{wang2023seqxgpt,
      title={SeqXGPT: Sentence-Level AI-Generated Text Detection}, 
      author={Pengyu Wang and Linyang Li and Ke Ren and Botian Jiang and Dong Zhang and Xipeng Qiu},
      year={2023},
      eprint={2310.08903},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

