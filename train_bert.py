"""
Training Script for BERT Detector
Fine-tunes BERT/RoBERTa for AI text detection.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import yaml

from data.seqxgpt_dataset import SeqXGPTDataset
from models.bert_detector import BERTDetector


class TextDataset(Dataset):
    """Dataset that returns raw texts and labels."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


def evaluate(model, dataloader, device):
    """Evaluate model on a dataset."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model.forward(input_ids, attention_mask)
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1 (AI)
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


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model.forward(input_ids, attention_mask, labels)
        loss = outputs['loss']
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        logits = outputs['logits']
        preds = torch.argmax(logits, dim=-1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, acc


def main():
    # Load configuration
    config_path = Path("configs/bert_config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            'model': {
                'model_name': 'bert-base-uncased',
                'num_labels': 2,
                'dropout': 0.1
            },
            'training': {
                'batch_size': 16,
                'learning_rate': 2e-5,
                'num_epochs': 10,
                'max_length': 512,
                'early_stopping_patience': 3
            },
            'data': {
                'data_dir': 'dataset/SeqXGPT-Bench',
                'train_ratio': 0.8,
                'val_ratio': 0.1,
                'test_ratio': 0.1,
                'seed': 42
            }
        }
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path("checkpoints/bert")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    print("\n=== Loading Datasets ===")
    train_dataset_raw = SeqXGPTDataset(
        data_dir=config['data']['data_dir'],
        split='train',
        **{k: v for k, v in config['data'].items() if k != 'data_dir'}
    )
    val_dataset_raw = SeqXGPTDataset(
        data_dir=config['data']['data_dir'],
        split='val',
        **{k: v for k, v in config['data'].items() if k != 'data_dir'}
    )
    
    # Initialize model
    print("\n=== Initializing Model ===")
    model = BERTDetector(**config['model']).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create datasets with tokenization
    train_texts, train_labels = train_dataset_raw.get_texts_and_labels()
    val_texts, val_labels = val_dataset_raw.get_texts_and_labels()
    
    train_dataset = TextDataset(
        train_texts, train_labels, model.tokenizer,
        max_length=config['training']['max_length']
    )
    val_dataset = TextDataset(
        val_texts, val_labels, model.tokenizer,
        max_length=config['training']['max_length']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    # Training setup
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
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
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
            model.model.save_pretrained(output_dir / 'best_model')
            model.tokenizer.save_pretrained(output_dir / 'best_model')
            
            # Save config and metrics
            with open(output_dir / 'config.json', 'w') as f:
                json.dump(config, f, indent=2)
            with open(output_dir / 'metrics.json', 'w') as f:
                json.dump(val_metrics, f, indent=2)
            
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
