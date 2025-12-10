"""
Quick test script to verify all components work correctly.
Run this before starting training to ensure everything is set up properly.
"""

import torch
import numpy as np
from pathlib import Path

def test_dataset():
    """Test dataset loading."""
    print("\n" + "="*60)
    print("TEST 1: Dataset Loading")
    print("="*60)
    
    try:
        from data.seqxgpt_dataset import SeqXGPTDataset
        
        # Test with small subset
        dataset = SeqXGPTDataset(split="train", max_samples_per_source=10)
        print(f"✓ Loaded {len(dataset)} samples")
        
        # Test getting item
        text, label = dataset[0]
        print(f"✓ Sample text length: {len(text)} chars")
        print(f"✓ Label: {label} ({'AI' if label == 1 else 'Human'})")
        
        return True
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False


def test_feature_extraction():
    """Test LLM feature extraction."""
    print("\n" + "="*60)
    print("TEST 2: Feature Extraction (GPT-2)")
    print("="*60)
    
    try:
        from features.llm_probs import LLMProbExtractor
        
        extractor = LLMProbExtractor(
            model_name="gpt2",
            max_length=128,
            cache_dir="features/test_cache"
        )
        
        test_text = "This is a simple test sentence."
        features = extractor.extract_features(test_text)
        
        print(f"✓ Extracted features shape: {features['log_probs'].shape}")
        print(f"✓ Actual length: {features['actual_length']}")
        print(f"✓ Mean log-prob: {features['log_probs'][:features['actual_length']].mean():.3f}")
        
        # Cleanup test cache
        import shutil
        if Path("features/test_cache").exists():
            shutil.rmtree("features/test_cache")
        
        return True
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_seqxgpt_model():
    """Test SeqXGPT model."""
    print("\n" + "="*60)
    print("TEST 3: SeqXGPT Model")
    print("="*60)
    
    try:
        from models.seqxgpt import SeqXGPTModel
        
        model = SeqXGPTModel(
            input_dim=3,
            hidden_dim=128,
            num_cnn_layers=3,
            max_seq_length=256
        )
        
        # Test forward pass
        batch_size = 4
        seq_len = 256
        x = torch.randn(batch_size, seq_len, 3)
        mask = torch.ones(batch_size, seq_len)
        
        logits = model(x, mask)
        probs = model.predict(x, mask)
        
        print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"✓ Forward pass successful")
        print(f"✓ Output shape: {logits.shape}")
        print(f"✓ Predictions shape: {probs.shape}")
        
        return True
    except Exception as e:
        print(f"✗ SeqXGPT model failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bert_model():
    """Test BERT model."""
    print("\n" + "="*60)
    print("TEST 4: BERT Model")
    print("="*60)
    
    try:
        from models.bert_detector import BERTDetector
        
        model = BERTDetector(model_name="bert-base-uncased")
        
        # Test with sample texts
        texts = ["This is a test.", "Another test sentence."]
        encoding = model.tokenizer(
            texts,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        outputs = model.forward(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"]
        )
        
        print(f"✓ Model loaded: bert-base-uncased")
        print(f"✓ Forward pass successful")
        print(f"✓ Logits shape: {outputs['logits'].shape}")
        
        return True
    except Exception as e:
        print(f"✗ BERT model failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configs():
    """Test configuration files."""
    print("\n" + "="*60)
    print("TEST 5: Configuration Files")
    print("="*60)
    
    try:
        import yaml
        
        # Test SeqXGPT config
        with open("configs/seqxgpt_config.yaml", 'r') as f:
            seqxgpt_config = yaml.safe_load(f)
        print(f"✓ SeqXGPT config loaded")
        print(f"  - Batch size: {seqxgpt_config['training']['batch_size']}")
        print(f"  - LLM: {seqxgpt_config['llm']['model_name']}")
        
        # Test BERT config
        with open("configs/bert_config.yaml", 'r') as f:
            bert_config = yaml.safe_load(f)
        print(f"✓ BERT config loaded")
        print(f"  - Batch size: {bert_config['training']['batch_size']}")
        print(f"  - Model: {bert_config['model']['model_name']}")
        
        return True
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False


def test_gpu():
    """Test GPU availability."""
    print("\n" + "="*60)
    print("TEST 6: GPU/CUDA")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"  - Device: {torch.cuda.get_device_name(0)}")
        print(f"  - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Test GPU memory
        try:
            x = torch.randn(100, 100).cuda()
            print(f"✓ GPU memory test passed")
        except:
            print(f"⚠ GPU available but test failed")
    else:
        print(f"⚠ CUDA not available - will use CPU")
        print(f"  Training will be slower but should work")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "🔬 RUNNING COMPONENT TESTS")
    print("="*60)
    
    tests = [
        ("Configuration Files", test_configs),
        ("GPU/CUDA", test_gpu),
        ("Dataset Loading", test_dataset),
        ("Feature Extraction", test_feature_extraction),
        ("SeqXGPT Model", test_seqxgpt_model),
        ("BERT Model", test_bert_model),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready to train.")
        print("\nNext steps:")
        print("  1. python train_seqxgpt.py")
        print("  2. python train_bert.py")
        print("  3. python eval.py")
    else:
        print("\n⚠ Some tests failed. Fix issues before training.")


if __name__ == "__main__":
    main()
