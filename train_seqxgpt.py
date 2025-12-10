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
        
        feature_dicts.append({
            'features': stacked.astype(np.float32),
            'actual_length': feat['actual_length']
        })
    
    return feature_dicts


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
            
            logits = model(features, masks)
            probs = torch.sigmoid(logits).squeeze(-1)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
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
    
    for features, masks, labels in tqdm(dataloader, desc="Training"):
        features = features.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        logits = model(features, masks).squeeze(-1)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = (torch.sigmoid(logits) > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    
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
                'learning_rate': 1e-4,
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
        cache_dir=config['llm']['cache_dir']
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
        lr=config['training']['learning_rate']
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
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val - Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, AUROC: {val_metrics['auroc']:.4f}")
        
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
                'metrics': val_metrics
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
