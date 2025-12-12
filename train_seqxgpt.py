"""
Training Script for SeqXGPT Model
Trains the SeqXGPT model using log-probability features extracted from LLM.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import yaml

from data.seqxgpt_dataset import SeqXGPTDataset
from features.llm_probs import LLMProbExtractor
from models.seqxgpt import SeqXGPTModel


class FeatureDataset(Dataset):
    """Dataset that returns pre-extracted features."""
    
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def collate_fn(batch):
    """Custom collate function for batching."""
    features_list, labels = zip(*batch)
    
    # Stack features
    features = torch.stack([torch.from_numpy(f['features']) for f in features_list])
    
    # Create masks
    masks = torch.zeros(len(features_list), features.shape[1])
    for i, f in enumerate(features_list):
        actual_len = f['actual_length']
        masks[i, :actual_len] = 1
    
    labels = torch.tensor(labels, dtype=torch.float32)
    
    return features, masks, labels


def extract_or_load_features(texts, labels, extractor, cache_name, config):
    """Extract features or load from cache."""
    features = extractor.extract_and_cache(
        texts,
        cache_name=cache_name,
        force_recompute=config.get('force_recompute_features', False)
    )
    
    # Convert to format for dataset
    feature_dicts = []
    for feat in features:
        # Stack selected features
        feature_types = config.get('feature_types', ['log_probs', 'surprisal', 'entropy'])
        feat_list = [feat[ft] for ft in feature_types]
        stacked = np.stack(feat_list, axis=-1)  # [max_length, num_features]
        
        # Clean NaN/Inf BEFORE normalization
        stacked = np.nan_to_num(stacked, nan=0.0, posinf=20.0, neginf=-20.0)
        
        feature_dicts.append({
            'features': stacked.astype(np.float32),
            'actual_length': feat['actual_length']
        })
    
    return feature_dicts


def normalize_features(feature_dicts):
    """Normalize features to have zero mean and unit variance."""
    # Collect all features
    all_features = []
    for fd in feature_dicts:
        actual_len = fd['actual_length']
        # Only use actual features (not padding)
        all_features.append(fd['features'][:actual_len])
    
    # Concatenate and compute stats
    all_features = np.concatenate(all_features, axis=0)  # [total_tokens, num_features]
    
    mean = np.mean(all_features, axis=0, keepdims=True)  # [1, num_features]
    std = np.std(all_features, axis=0, keepdims=True)    # [1, num_features]
    std = np.where(std < 1e-8, 1.0, std)  # Avoid division by zero
    
    print(f"  Feature normalization stats:")
    print(f"    Mean: {mean.flatten()}")
    print(f"    Std:  {std.flatten()}")
    
    # Normalize all features
    for fd in feature_dicts:
        fd['features'] = (fd['features'] - mean) / std
        # Clean any remaining NaN/Inf
        fd['features'] = np.nan_to_num(fd['features'], nan=0.0, posinf=5.0, neginf=-5.0)
        # Clip to reasonable range after normalization
        fd['features'] = np.clip(fd['features'], -5.0, 5.0)
    
    print(f"  After normalization: min={np.min([fd['features'].min() for fd in feature_dicts]):.4f}, max={np.max([fd['features'].max() for fd in feature_dicts]):.4f}")
    
    return feature_dicts, mean, std


def evaluate(model, dataloader, device):
    """Evaluate model on a dataset."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for features, masks, labels in dataloader:
            features = features.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            # Replace NaN in features with 0
            features = torch.nan_to_num(features, nan=0.0)
            
            logits = model(features, masks)
            probs = torch.sigmoid(logits).squeeze(-1)
            
            # Replace NaN probabilities with 0.5 (neutral)
            probs = torch.nan_to_num(probs, nan=0.5)
            
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    import numpy as np
    all_probs = np.array(all_probs)
    all_probs = np.nan_to_num(all_probs, nan=0.5)  # Extra safety check
    
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    auroc = roc_auc_score(all_labels, all_probs)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc
    }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    skipped_batches = 0
    
    for batch_idx, (features, masks, labels) in enumerate(tqdm(dataloader, desc="Training")):
        features = features.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        
        # Clean and clip features
        features = torch.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)
        features = torch.clamp(features, -5.0, 5.0)
        
        # Check for invalid data
        if torch.isnan(features).any() or torch.isinf(features).any():
            print(f"\nBatch {batch_idx}: Invalid features detected, skipping")
            skipped_batches += 1
            continue
        
        optimizer.zero_grad()
        
        try:
            logits = model(features, masks).squeeze(-1)
            loss = criterion(logits, labels)
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nBatch {batch_idx}: Invalid loss={loss.item()}, skipping")
                skipped_batches += 1
                continue
            
            loss.backward()
            
            # Gradient clipping to prevent explosion
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Check gradients
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"\nBatch {batch_idx}: Invalid gradients, skipping")
                skipped_batches += 1
                continue
            
            optimizer.step()
            
            total_loss += loss.item()
            
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        except Exception as e:
            print(f"\nBatch {batch_idx}: Exception {e}, skipping")
            skipped_batches += 1
            continue
    
    if skipped_batches > 0:
        print(f"\nSkipped {skipped_batches}/{len(dataloader)} batches due to invalid data")
    
    valid_batches = len(dataloader) - skipped_batches
    avg_loss = total_loss / max(valid_batches, 1)
    acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
    
    return avg_loss, acc


def main():
    # Load configuration
    config_path = Path("configs/seqxgpt_config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            'model': {
                'input_dim': 3,
                'hidden_dim': 128,
                'num_cnn_layers': 3,
                'kernel_size': 3,
                'num_attention_heads': 4,
                'dropout': 0.3,
                'max_seq_length': 256
            },
            'training': {
                'batch_size': 16,
                'learning_rate': 5e-5,
                'num_epochs': 20,
                'early_stopping_patience': 5
            },
            'data': {
                'data_dir': 'dataset/SeqXGPT-Bench',
                'train_ratio': 0.8,
                'val_ratio': 0.1,
                'test_ratio': 0.1,
                'seed': 42
            },
            'llm': {
                'model_name': 'gpt2',
                'max_length': 256,
                'cache_dir': 'features/cache'
            },
            'feature_types': ['log_probs', 'surprisal', 'entropy'],
            'force_recompute_features': False
        }
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path("checkpoints/seqxgpt")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize feature extractor
    print("\n=== Initializing Feature Extractor ===")
    extractor = LLMProbExtractor(
        model_name=config['llm']['model_name'],
        max_length=config['llm']['max_length'],
        cache_dir=config['llm']['cache_dir'],
        batch_size=32  # Larger batch on CPU to amortize overhead
    )
    
    # Load datasets
    print("\n=== Loading Datasets ===")
    train_dataset = SeqXGPTDataset(
        data_dir=config['data']['data_dir'],
        split='train',
        **{k: v for k, v in config['data'].items() if k != 'data_dir'}
    )
    val_dataset = SeqXGPTDataset(
        data_dir=config['data']['data_dir'],
        split='val',
        **{k: v for k, v in config['data'].items() if k != 'data_dir'}
    )
    
    # Extract features
    print("\n=== Extracting Features ===")
    train_texts, train_labels = train_dataset.get_texts_and_labels()
    val_texts, val_labels = val_dataset.get_texts_and_labels()
    
    train_features = extract_or_load_features(train_texts, train_labels, extractor, "train", config)
    val_features = extract_or_load_features(val_texts, val_labels, extractor, "val", config)
    
    # Normalize features using train statistics
    print("\n=== Normalizing Features ===")
    train_features, mean, std = normalize_features(train_features)
    
    # Apply same normalization to validation
    for fd in val_features:
        fd['features'] = (fd['features'] - mean) / std
        fd['features'] = np.nan_to_num(fd['features'], nan=0.0, posinf=5.0, neginf=-5.0)
        fd['features'] = np.clip(fd['features'], -5.0, 5.0)
    
    # Create dataloaders
    train_loader = DataLoader(
        FeatureDataset(train_features, train_labels),
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        FeatureDataset(val_features, val_labels),
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Initialize model
    print("\n=== Initializing Model ===")
    model = SeqXGPTModel(**config['model']).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=0.01,
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    # Training loop
    print("\n=== Training ===")
    best_f1 = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_metrics': []}
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        history['val_metrics'].append(val_metrics)
        
        # Check for NaN in training
        if np.isnan(train_loss):
            print("ERROR: Training loss is NaN! Stopping training.")
            break
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val - Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, AUROC: {val_metrics['auroc']:.4f}")
        
        # Step scheduler
        scheduler.step(val_metrics['f1'])
        
        # Early stopping
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'metrics': val_metrics,
                'feature_mean': mean,
                'feature_std': std
            }, output_dir / 'best_model.pt')
            print(f"✓ Saved best model (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config['training']['early_stopping_patience']:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    # Save training history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✓ Training completed! Best F1: {best_f1:.4f}")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
