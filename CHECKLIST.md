# Checklist Progetto SeqXGPT + BERT

## ✅ Setup Completato

### Configurazione
- [x] GPT-2 standard per log-probabilities
- [x] BERT-base-uncased per classificazione
- [x] Batch size SeqXGPT: 64
- [x] Batch size BERT: 16
- [x] Evasion attacks: paraphrasing + back-translation (en→it→en)

### Struttura Progetto
```
SeqXGPT/
├── data/              ✅ Dataset loaders implementati
├── dataset/           ✅ 36,000 samples (6,000 per source)
├── models/            ✅ SeqXGPT e BERT implementati
├── features/          ✅ LLM log-prob extraction
├── attacks/           ✅ Paraphrasing + Back-translation
├── configs/           ✅ YAML con iperparametri
├── train_*.py         ✅ Training scripts
├── eval.py            ✅ Evaluation script
├── test_components.py ✅ Component testing
└── run_evasion_attacks.py ✅ Evasion testing
```

---

## 📋 Workflow di Esecuzione

### 1. Test Ambiente (PRIMA DI TUTTO)
```bash
python test_components.py
```
**Output atteso:** Tutti i test passano ✓

### 2. Verifica Setup
```bash
python verify_setup.py
```
**Output atteso:** Dependencies, structure, dataset OK

### 3. Training SeqXGPT
```bash
python train_seqxgpt.py
```
**Tempo stimato:** 30-60 min su GPU (8-12 GB)

**Output:**
- `checkpoints/seqxgpt/best_model.pt`
- `checkpoints/seqxgpt/history.json`

**Cosa fa:**
1. Estrae log-prob da GPT-2 (cache in `features/cache/`)
2. Allena CNN + Self-Attention
3. Early stopping su validation F1
4. Salva best model

### 4. Training BERT
```bash
python train_bert.py
```
**Tempo stimato:** 20-40 min su GPU (8-12 GB)

**Output:**
- `checkpoints/bert/best_model/`
- `checkpoints/bert/history.json`

**Cosa fa:**
1. Fine-tuning BERT-base-uncased
2. Early stopping su validation F1
3. Salva best model

### 5. Evaluation Comparativa
```bash
python eval.py
```
**Tempo stimato:** 5-10 min

**Output:**
- `results/results.json` - Metriche dettagliate
- `results/results_table.txt` - Tabella formattata
- `results/roc_curves.png` - Curve ROC
- `results/confusion_matrices.png` - Confusion matrices

**Metriche calcolate:**
- Accuracy
- Precision
- Recall
- F1 Score
- AUROC

### 6. Evasion Attacks
```bash
# Default: 100 samples
python run_evasion_attacks.py

# O con più samples
python run_evasion_attacks.py --num_samples 200
```
**Tempo stimato:** 20-40 min (dipende da num_samples)

**Output:**
- `results/evasion_results.json`

**Test eseguiti:**
1. **Original:** Detection rate su testi originali
2. **Paraphrasing:** Detection rate dopo parafrasamento
3. **Back-translation:** Detection rate dopo en→it→en

---

## 📊 Risultati Attesi

### Training Metrics
- **SeqXGPT:** F1 ~0.83-0.87, AUROC ~0.90-0.93
- **BERT:** F1 ~0.86-0.90, AUROC ~0.92-0.95

### Evasion Impact
- **Paraphrasing:** Drop detection ~10-20%
- **Back-translation:** Drop detection ~15-30%
- **BERT** generalmente più robusto di SeqXGPT

---

## 🐛 Troubleshooting

### Out of Memory (OOM)
**Problema:** CUDA out of memory durante training

**Soluzioni:**
```yaml
# In configs/seqxgpt_config.yaml
training:
  batch_size: 32  # Riduci da 64

# In configs/bert_config.yaml
training:
  batch_size: 8  # Riduci da 16
```

### Feature Extraction Lenta
**Problema:** Prima estrazione feature molto lenta

**Soluzione:** 
- È normale la prima volta
- Le feature vengono cachate in `features/cache/`
- Esecuzioni successive saranno veloci

**Per ricominciare da capo:**
```bash
# Elimina cache
rm -rf features/cache/
```

### Model Loading Error
**Problema:** Errore loading modelli in eval.py

**Causa:** Modelli non ancora allenati

**Soluzione:**
```bash
# Verifica che esistano:
ls checkpoints/seqxgpt/best_model.pt
ls checkpoints/bert/best_model/
```

### Import Errors
**Problema:** ModuleNotFoundError

**Soluzione:**
```bash
# Reinstalla dipendenze
pip install -r requirements.txt

# Verifica installazione
python verify_setup.py
```

---

## 📈 Monitoraggio Training

### Durante il Training
Vedrai:
```
Epoch 1/20
Training: 100%|████████| 150/150 [02:15<00:00]
Train Loss: 0.4521, Train Acc: 0.7892
Val - Acc: 0.8234, F1: 0.8156, AUROC: 0.8945
✓ Saved best model (F1: 0.8156)
```

### Segnali di Buon Training
- ✅ Loss diminuisce progressivamente
- ✅ Validation metrics migliorano
- ✅ Early stopping dopo 3-5 patience
- ✅ No overfitting (train/val gap piccolo)

### Segnali di Problemi
- ⚠️ Loss NaN o inf → riduci learning rate
- ⚠️ No improvement dopo molte epoche → learning rate troppo basso
- ⚠️ Train acc alta, val acc bassa → overfitting

---

## 📝 Per il Report

### Sezioni da Includere

**1. Metodologia**
- SeqXGPT: Log-prob GPT-2 + CNN + Self-Attention
- BERT: Fine-tuning bert-base-uncased
- Dataset: SeqXGPT-Bench (6 sources, 36K samples)
- Split: 80/10/10 train/val/test

**2. Risultati**
- Tabella comparativa (da `results/results_table.txt`)
- Curve ROC (da `results/roc_curves.png`)
- Confusion matrices

**3. Analisi Robustezza**
- Detection rate su original vs attacked texts
- Tabella evasion results (da `results/evasion_results.json`)
- Confronto SeqXGPT vs BERT su attacks

**4. Discussione**
- Quale modello performa meglio?
- Perché BERT più robusto?
- Trade-off: SeqXGPT più interpretabile, BERT più accurato
- Limiti: dependency su reference LLM (GPT-2)

---

## 🎯 Quick Commands Reference

```bash
# 1. Test tutto
python test_components.py

# 2. Train entrambi (parallelizzabile)
python train_seqxgpt.py &
python train_bert.py &
wait

# 3. Evaluate
python eval.py

# 4. Test evasion
python run_evasion_attacks.py --num_samples 100

# 5. Controlla risultati
cat results/results_table.txt
cat results/evasion_results.json
```

---

## ✅ Status Finale

- [x] Tutti i componenti implementati
- [x] Configurazione ottimizzata per GPU 8-12 GB
- [x] GPT-2 + BERT-base-uncased
- [x] Evasion attacks inclusi
- [x] Scripts di testing e verifica
- [x] Documentazione completa

**Pronto per l'esecuzione! 🚀**

Esegui nell'ordine:
1. `python test_components.py`
2. `python train_seqxgpt.py`
3. `python train_bert.py`
4. `python eval.py`
5. `python run_evasion_attacks.py`
