"""
Evaluation Script
Comparative evaluation of SeqXGPT and BERT detectors on multiple datasets.
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
import numpy as np
from tabulate import tabulate
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import yaml

from data.seqxgpt_dataset import SeqXGPTDataset
from data.extra_dataset import ExtraDataset
from features.llm_probs import LLMProbExtractor
from models.seqxgpt import SeqXGPTModel
from models.bert_detector import BERTDetector
from train_seqxgpt import FeatureDataset, collate_fn, extract_or_load_features
from train_bert import TextDataset


def evaluate_seqxgpt(model, dataloader, device):
    """Evaluate SeqXGPT model."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for features, masks, labels in dataloader:
            features = features.to(device)
            masks = masks.to(device)
            
            probs = model.predict(features, masks)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return calculate_metrics(all_labels, all_preds, all_probs)


def evaluate_bert(model, dataloader, device):
    """Evaluate BERT model."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            
            probs = model.predict(input_ids, attention_mask)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return calculate_metrics(all_labels, all_preds, all_probs)


def calculate_metrics(labels, preds, probs):
    """Calculate evaluation metrics."""
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    auroc = roc_auc_score(labels, probs)
    cm = confusion_matrix(labels, preds)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc,
        'confusion_matrix': cm,
        'predictions': preds,
        'probabilities': probs,
        'labels': labels
    }


def plot_roc_curves(results, output_dir):
    """Plot ROC curves for all models and datasets."""
    plt.figure(figsize=(10, 8))
    
    for model_name, datasets in results.items():
        for dataset_name, metrics in datasets.items():
            if 'labels' in metrics and 'probabilities' in metrics:
                fpr, tpr, _ = roc_curve(metrics['labels'], metrics['probabilities'])
                auroc = metrics['auroc']
                plt.plot(fpr, tpr, label=f"{model_name} - {dataset_name} (AUROC={auroc:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Model Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves.png', dpi=300)
    print(f"Saved ROC curves to {output_dir / 'roc_curves.png'}")


def plot_confusion_matrices(results, output_dir):
    """Plot confusion matrices."""
    n_models = len(results)
    n_datasets = max(len(datasets) for datasets in results.values())
    
    fig, axes = plt.subplots(n_models, n_datasets, figsize=(5*n_datasets, 5*n_models))
    if n_models == 1:
        axes = [axes]
    if n_datasets == 1:
        axes = [[ax] for ax in axes]
    
    for i, (model_name, datasets) in enumerate(results.items()):
        for j, (dataset_name, metrics) in enumerate(datasets.items()):
            if 'confusion_matrix' in metrics:
                cm = metrics['confusion_matrix']
                ax = axes[i][j] if n_models > 1 else axes[i]
                
                im = ax.imshow(cm, cmap='Blues')
                ax.set_title(f"{model_name}\n{dataset_name}")
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                
                # Add text annotations
                for x in range(2):
                    for y in range(2):
                        ax.text(y, x, str(cm[x, y]), ha='center', va='center')
                
                plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=300)
    print(f"Saved confusion matrices to {output_dir / 'confusion_matrices.png'}")


def load_seqxgpt_model(checkpoint_path, config, device):
    """Load trained SeqXGPT model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SeqXGPTModel(**config['model']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def load_bert_model(checkpoint_dir, device):
    """Load trained BERT model."""
    model = BERTDetector()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model.to(device)
    model.eval()
    return model


def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Load configurations
    with open("configs/seqxgpt_config.yaml", 'r') as f:
        seqxgpt_config = yaml.safe_load(f)
    with open("configs/bert_config.yaml", 'r') as f:
        bert_config = yaml.safe_load(f)
    
    # Initialize feature extractor for SeqXGPT
    print("=== Initializing Feature Extractor ===")
    extractor = LLMProbExtractor(
        model_name=seqxgpt_config['llm']['model_name'],
        max_length=seqxgpt_config['llm']['max_length'],
        cache_dir=seqxgpt_config['llm']['cache_dir']
    )
    
    # Load models
    print("\n=== Loading Models ===")
    
    seqxgpt_checkpoint = Path("checkpoints/seqxgpt/best_model.pt")
    bert_checkpoint = Path("checkpoints/bert/best_model")
    
    if not seqxgpt_checkpoint.exists():
        print(f"SeqXGPT checkpoint not found: {seqxgpt_checkpoint}")
        print("Please train the model first using train_seqxgpt.py")
        return
    
    if not bert_checkpoint.exists():
        print(f"BERT checkpoint not found: {bert_checkpoint}")
        print("Please train the model first using train_bert.py")
        return
    
    seqxgpt_model = load_seqxgpt_model(seqxgpt_checkpoint, seqxgpt_config, device)
    bert_model = load_bert_model(bert_checkpoint, device)
    
    print("✓ Models loaded successfully")
    
    # Define datasets to evaluate
    datasets_config = [
        {
            'name': 'SeqXGPT-Bench',
            'type': 'seqxgpt',
            'data_dir': 'dataset/SeqXGPT-Bench'
        }
    ]
    
    # Check for extra dataset
    extra_dataset_file = Path("data/extra_dataset.csv")
    if extra_dataset_file.exists():
        datasets_config.append({
            'name': 'ExtraDataset',
            'type': 'extra',
            'data_file': str(extra_dataset_file)
        })
    
    # Evaluate on all datasets
    results = {
        'SeqXGPT': {},
        'BERT': {}
    }
    
    for dataset_config in datasets_config:
        dataset_name = dataset_config['name']
        print(f"\n{'='*60}")
        print(f"Evaluating on: {dataset_name}")
        print('='*60)
        
        # Load test dataset
        if dataset_config['type'] == 'seqxgpt':
            test_dataset_raw = SeqXGPTDataset(
                data_dir=dataset_config['data_dir'],
                split='test',
                **{k: v for k, v in seqxgpt_config['data'].items() if k != 'data_dir'}
            )
        else:
            test_dataset_raw = ExtraDataset(
                data_file=dataset_config['data_file'],
                split='test',
                seed=seqxgpt_config['data']['seed']
            )
        
        test_texts, test_labels = test_dataset_raw.get_texts_and_labels()
        
        # Evaluate SeqXGPT
        print(f"\n--- SeqXGPT ---")
        test_features = extract_or_load_features(
            test_texts, test_labels, extractor,
            f"test_{dataset_name.lower()}", seqxgpt_config
        )
        seqxgpt_loader = DataLoader(
            FeatureDataset(test_features, test_labels),
            batch_size=seqxgpt_config['training']['batch_size'],
            shuffle=False,
            collate_fn=collate_fn
        )
        seqxgpt_metrics = evaluate_seqxgpt(seqxgpt_model, seqxgpt_loader, device)
        results['SeqXGPT'][dataset_name] = seqxgpt_metrics
        
        print(f"Accuracy:  {seqxgpt_metrics['accuracy']:.4f}")
        print(f"Precision: {seqxgpt_metrics['precision']:.4f}")
        print(f"Recall:    {seqxgpt_metrics['recall']:.4f}")
        print(f"F1:        {seqxgpt_metrics['f1']:.4f}")
        print(f"AUROC:     {seqxgpt_metrics['auroc']:.4f}")
        
        # Evaluate BERT
        print(f"\n--- BERT ---")
        bert_dataset = TextDataset(
            test_texts, test_labels, bert_model.tokenizer,
            max_length=bert_config['training']['max_length']
        )
        bert_loader = DataLoader(
            bert_dataset,
            batch_size=bert_config['training']['batch_size'],
            shuffle=False
        )
        bert_metrics = evaluate_bert(bert_model, bert_loader, device)
        results['BERT'][dataset_name] = bert_metrics
        
        print(f"Accuracy:  {bert_metrics['accuracy']:.4f}")
        print(f"Precision: {bert_metrics['precision']:.4f}")
        print(f"Recall:    {bert_metrics['recall']:.4f}")
        print(f"F1:        {bert_metrics['f1']:.4f}")
        print(f"AUROC:     {bert_metrics['auroc']:.4f}")
    
    # Create results table
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print('='*60)
    
    table_data = []
    for model_name, datasets in results.items():
        for dataset_name, metrics in datasets.items():
            table_data.append([
                model_name,
                dataset_name,
                f"{metrics['accuracy']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1']:.4f}",
                f"{metrics['auroc']:.4f}"
            ])
    
    headers = ['Model', 'Dataset', 'Acc', 'Prec', 'Rec', 'F1', 'AUROC']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Save results
    results_json = {
        model: {
            dataset: {k: v for k, v in metrics.items() 
                     if k not in ['predictions', 'probabilities', 'labels', 'confusion_matrix']}
            for dataset, metrics in datasets.items()
        }
        for model, datasets in results.items()
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    # Save table
    with open(output_dir / 'results_table.txt', 'w') as f:
        f.write(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Plot results
    try:
        plot_roc_curves(results, output_dir)
        plot_confusion_matrices(results, output_dir)
    except Exception as e:
        print(f"\nWarning: Could not generate plots: {e}")
    
    print(f"\n✓ Evaluation completed!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
