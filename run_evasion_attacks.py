"""
Test and run evasion attacks on trained models.
Tests paraphrasing and back-translation attacks.
"""

import torch
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

from attacks.text_augmentation import TextAugmenter
from data.seqxgpt_dataset import SeqXGPTDataset
from models.seqxgpt import SeqXGPTModel
from models.bert_detector import BERTDetector
from features.llm_probs import LLMProbExtractor
import yaml


def load_models(device):
    """Load trained models."""
    print("Loading models...")
    
    # Load SeqXGPT
    seqxgpt_checkpoint = Path("checkpoints/seqxgpt/best_model.pt")
    if not seqxgpt_checkpoint.exists():
        print("⚠ SeqXGPT checkpoint not found. Train the model first.")
        seqxgpt_model = None
    else:
        with open("configs/seqxgpt_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        checkpoint = torch.load(seqxgpt_checkpoint, map_location=device)
        seqxgpt_model = SeqXGPTModel(**config['model']).to(device)
        seqxgpt_model.load_state_dict(checkpoint['model_state_dict'])
        seqxgpt_model.eval()
        print("✓ SeqXGPT loaded")
    
    # Load BERT
    bert_checkpoint = Path("checkpoints/bert/best_model")
    if not bert_checkpoint.exists():
        print("⚠ BERT checkpoint not found. Train the model first.")
        bert_model = None
    else:
        bert_model = BERTDetector()
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        bert_model.model = AutoModelForSequenceClassification.from_pretrained(bert_checkpoint)
        bert_model.tokenizer = AutoTokenizer.from_pretrained(bert_checkpoint)
        bert_model.to(device)
        bert_model.eval()
        print("✓ BERT loaded")
    
    return seqxgpt_model, bert_model


def test_on_augmented(texts, labels, augmented_texts, model, model_name, extractor=None):
    """Test model on augmented texts."""
    device = next(model.parameters()).device
    
    if model_name == "SeqXGPT":
        # Extract features
        print(f"  Extracting features for {len(augmented_texts)} texts...")
        features = extractor.extract_batch(augmented_texts, show_progress=False)
        
        # Prepare tensors
        feature_list = []
        for feat in features:
            stacked = np.stack([
                feat['log_probs'],
                feat['surprisal'],
                feat['entropy']
            ], axis=-1)
            feature_list.append(torch.from_numpy(stacked))
        
        features_tensor = torch.stack(feature_list).to(device)
        masks = torch.ones(len(features), features_tensor.shape[1]).to(device)
        
        # Predict
        with torch.no_grad():
            probs = model.predict(features_tensor, masks).cpu().numpy()
    
    else:  # BERT
        # Tokenize
        encoding = model.tokenizer(
            augmented_texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        
        # Predict
        with torch.no_grad():
            probs = model.predict(input_ids, attention_mask).cpu().numpy()
    
    # Calculate metrics
    preds = (probs > 0.5).astype(int)
    accuracy = (preds == labels).mean()
    
    # For AI texts only
    ai_mask = labels == 1
    if ai_mask.sum() > 0:
        ai_accuracy = (preds[ai_mask] == labels[ai_mask]).mean()
        ai_detected = (preds[ai_mask] == 1).mean()
    else:
        ai_accuracy = 0
        ai_detected = 0
    
    return {
        'accuracy': accuracy,
        'ai_accuracy': ai_accuracy,
        'ai_detected': ai_detected,
        'predictions': preds
    }


def run_evasion_tests(num_samples=100):
    """Run evasion attack tests."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load models
    seqxgpt_model, bert_model = load_models(device)
    
    if seqxgpt_model is None and bert_model is None:
        print("No trained models found. Please train models first.")
        return
    
    # Initialize feature extractor for SeqXGPT
    if seqxgpt_model is not None:
        with open("configs/seqxgpt_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        extractor = LLMProbExtractor(
            model_name=config['llm']['model_name'],
            max_length=config['llm']['max_length'],
            cache_dir="features/cache"
        )
    else:
        extractor = None
    
    # Load test data (AI-generated only for attacks)
    print("\nLoading test dataset...")
    test_dataset = SeqXGPTDataset(split="test")
    all_texts, all_labels = test_dataset.get_texts_and_labels()
    
    # Filter AI texts only
    ai_indices = [i for i, label in enumerate(all_labels) if label == 1]
    ai_indices = ai_indices[:num_samples]  # Limit number of samples
    
    original_texts = [all_texts[i] for i in ai_indices]
    original_labels = np.array([all_labels[i] for i in ai_indices])
    
    print(f"Selected {len(original_texts)} AI-generated texts for attack testing")
    
    # Initialize augmenter
    print("\nInitializing text augmenter...")
    augmenter = TextAugmenter(device=str(device))
    
    # Run attacks
    results = {
        'original': {},
        'paraphrase': {},
        'back_translate': {}
    }
    
    print("\n" + "="*60)
    print("TESTING ORIGINAL TEXTS")
    print("="*60)
    
    if seqxgpt_model:
        print("\nSeqXGPT on original:")
        metrics = test_on_augmented(
            original_texts, original_labels, original_texts,
            seqxgpt_model, "SeqXGPT", extractor
        )
        results['original']['SeqXGPT'] = metrics
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  AI Detection Rate: {metrics['ai_detected']:.3f}")
    
    if bert_model:
        print("\nBERT on original:")
        metrics = test_on_augmented(
            original_texts, original_labels, original_texts,
            bert_model, "BERT"
        )
        results['original']['BERT'] = metrics
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  AI Detection Rate: {metrics['ai_detected']:.3f}")
    
    # Paraphrasing attack
    print("\n" + "="*60)
    print("ATTACK 1: PARAPHRASING")
    print("="*60)
    
    print("\nGenerating paraphrases...")
    paraphrased_texts = []
    for text in tqdm(original_texts):
        try:
            paraphrases = augmenter.paraphrase(text, num_return_sequences=1)
            paraphrased_texts.append(paraphrases[0] if paraphrases else text)
        except:
            paraphrased_texts.append(text)  # Fallback to original
    
    if seqxgpt_model:
        print("\nSeqXGPT on paraphrased:")
        metrics = test_on_augmented(
            original_texts, original_labels, paraphrased_texts,
            seqxgpt_model, "SeqXGPT", extractor
        )
        results['paraphrase']['SeqXGPT'] = metrics
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  AI Detection Rate: {metrics['ai_detected']:.3f}")
        print(f"  Drop: {results['original']['SeqXGPT']['ai_detected'] - metrics['ai_detected']:.3f}")
    
    if bert_model:
        print("\nBERT on paraphrased:")
        metrics = test_on_augmented(
            original_texts, original_labels, paraphrased_texts,
            bert_model, "BERT"
        )
        results['paraphrase']['BERT'] = metrics
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  AI Detection Rate: {metrics['ai_detected']:.3f}")
        print(f"  Drop: {results['original']['BERT']['ai_detected'] - metrics['ai_detected']:.3f}")
    
    # Back-translation attack
    print("\n" + "="*60)
    print("ATTACK 2: BACK-TRANSLATION (EN→IT→EN)")
    print("="*60)
    
    print("\nGenerating back-translations...")
    back_translated_texts = []
    for text in tqdm(original_texts):
        try:
            bt_text = augmenter.back_translate(text, intermediate_lang='it')
            back_translated_texts.append(bt_text)
        except:
            back_translated_texts.append(text)  # Fallback to original
    
    if seqxgpt_model:
        print("\nSeqXGPT on back-translated:")
        metrics = test_on_augmented(
            original_texts, original_labels, back_translated_texts,
            seqxgpt_model, "SeqXGPT", extractor
        )
        results['back_translate']['SeqXGPT'] = metrics
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  AI Detection Rate: {metrics['ai_detected']:.3f}")
        print(f"  Drop: {results['original']['SeqXGPT']['ai_detected'] - metrics['ai_detected']:.3f}")
    
    if bert_model:
        print("\nBERT on back-translated:")
        metrics = test_on_augmented(
            original_texts, original_labels, back_translated_texts,
            bert_model, "BERT"
        )
        results['back_translate']['BERT'] = metrics
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  AI Detection Rate: {metrics['ai_detected']:.3f}")
        print(f"  Drop: {results['original']['BERT']['ai_detected'] - metrics['ai_detected']:.3f}")
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Convert numpy to native Python types for JSON
    results_json = {}
    for attack_type, models_dict in results.items():
        results_json[attack_type] = {}
        for model_name, metrics in models_dict.items():
            results_json[attack_type][model_name] = {
                'accuracy': float(metrics['accuracy']),
                'ai_accuracy': float(metrics['ai_accuracy']),
                'ai_detected': float(metrics['ai_detected'])
            }
    
    with open(output_dir / "evasion_results.json", 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir / 'evasion_results.json'}")
    
    # Summary table
    print("\n" + "="*60)
    print("SUMMARY: AI DETECTION RATES")
    print("="*60)
    print(f"{'Attack':<20} {'SeqXGPT':<15} {'BERT':<15}")
    print("-" * 60)
    
    for attack_type in ['original', 'paraphrase', 'back_translate']:
        attack_name = attack_type.replace('_', ' ').title()
        seqxgpt_rate = results_json.get(attack_type, {}).get('SeqXGPT', {}).get('ai_detected', 0)
        bert_rate = results_json.get(attack_type, {}).get('BERT', {}).get('ai_detected', 0)
        print(f"{attack_name:<20} {seqxgpt_rate:.3f} ({seqxgpt_rate*100:.1f}%)  {bert_rate:.3f} ({bert_rate*100:.1f}%)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run evasion attacks on trained models")
    parser.add_argument("--num_samples", type=int, default=100,
                      help="Number of AI samples to test (default: 100)")
    args = parser.parse_args()
    
    run_evasion_tests(num_samples=args.num_samples)
