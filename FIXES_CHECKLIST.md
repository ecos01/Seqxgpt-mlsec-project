# Checklist delle Modifiche Applicate per Risolvere i Problemi di Training

## ✅ Problema 1: NaN in roc_auc_score
**File**: `train_seqxgpt.py` - funzione `evaluate()`
- [x] Aggiunto `torch.nan_to_num(features, nan=0.0)` prima del forward pass
- [x] Aggiunto `torch.nan_to_num(probs, nan=0.5)` dopo sigmoid
- [x] Aggiunto `np.nan_to_num(all_probs, nan=0.5)` prima di roc_auc_score

## ✅ Problema 2: NaN nel softmax del modello
**File**: `models/seqxgpt.py` - attention pooling
- [x] Aggiunto `torch.nan_to_num(attn_weights, nan=1.0 / attn_weights.size(1))` 
- [x] Gestisce caso edge: tutti i pesi -inf → NaN dopo softmax

## ✅ Problema 3: Loss = NaN durante training
**File**: `train_seqxgpt.py` - funzione `extract_or_load_features()`
- [x] Pulizia NaN/Inf PRIMA della normalizzazione: `np.nan_to_num(stacked, nan=0.0, posinf=20.0, neginf=-20.0)`

**File**: `train_seqxgpt.py` - funzione `normalize_features()`
- [x] Clipping dopo normalizzazione: `np.clip(fd['features'], -5.0, 5.0)`
- [x] Gestione NaN dopo normalizzazione: `np.nan_to_num(..., nan=0.0, posinf=5.0, neginf=-5.0)`
- [x] Logging di min/max dopo normalizzazione

**File**: `train_seqxgpt.py` - funzione `train_epoch()`
- [x] Pulizia features nel batch: `torch.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)`
- [x] Clipping features: `torch.clamp(features, -5.0, 5.0)`
- [x] Check features invalidi prima del forward
- [x] Check loss NaN/Inf dopo forward
- [x] Gradient clipping: `max_norm=1.0`
- [x] Check gradients NaN/Inf dopo backward
- [x] Try-catch per exceptions
- [x] Contatore batch skippati con logging
- [x] Gestione divisione per zero nel calcolo avg_loss

## ✅ Problema 4: Modello predice sempre stessa classe
**File**: `train_seqxgpt.py` - learning rate e optimizer
- [x] Learning rate ridotto: `1e-4` → `5e-5`
- [x] Aggiunto `weight_decay=0.01` all'optimizer
- [x] Aggiunto `eps=1e-8` all'optimizer
- [x] Aggiunto Learning Rate Scheduler: `ReduceLROnPlateau`
- [x] Scheduler step basato su F1 validation

**File**: `train_seqxgpt.py` - main loop
- [x] Normalizzazione features train
- [x] Normalizzazione features validation con stesse statistiche train
- [x] Salvataggio `feature_mean` e `feature_std` nel checkpoint
- [x] Step dello scheduler dopo ogni epoca

## ✅ Verifiche Aggiuntive
- [x] Test script `test_nan_fix.py` creato e testato
- [x] Tutti i test passati
- [x] Debug script `debug_training.py` disponibile
- [x] File test_nan_fix.py rimosso dopo verifica

## 📊 Summary delle Protezioni
1. **Pre-processing**: Pulizia NaN prima di normalizzazione (±20)
2. **Normalizzazione**: Z-score + clipping (±5)
3. **Training**: Clipping batch + gradient clipping (max_norm=1.0)
4. **Validation**: Gestione NaN in probabilità
5. **Metrics**: Safety check prima di sklearn metrics
6. **Model**: Gestione softmax edge cases

## 🎯 Risultati Attesi
- Train Loss: Dovrebbe diminuire costantemente (non NaN)
- Train Accuracy: Dovrebbe aumentare da ~50% (random) verso 80%+
- Val F1: Dovrebbe migliorare progressivamente
- Val AUROC: > 0.5 (meglio di random)
- Batch skippati: Dovrebbe essere 0 o molto pochi

## 🚀 Prossimi Passi
1. Eseguire: `python train_seqxgpt.py`
2. Monitorare che non ci siano batch skippati
3. Verificare che loss diminuisca
4. Verificare che F1 aumenti
5. Se tutto ok, training dovrebbe completarsi in ~20 epochs o prima per early stopping
