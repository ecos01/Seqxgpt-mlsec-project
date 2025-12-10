"""
Utility script to verify the project setup and test all components.
"""

import sys
from pathlib import Path

def check_imports():
    """Check if all required packages are installed."""
    print("=" * 60)
    print("CHECKING DEPENDENCIES")
    print("=" * 60)
    
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'HuggingFace Transformers',
        'sklearn': 'scikit-learn',
        'yaml': 'PyYAML',
        'tqdm': 'tqdm',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib'
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - NOT INSTALLED")
            missing.append(name)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies installed!")
    return True


def check_dataset():
    """Check if dataset exists."""
    print("\n" + "=" * 60)
    print("CHECKING DATASET")
    print("=" * 60)
    
    dataset_dir = Path("dataset/SeqXGPT-Bench")
    
    if not dataset_dir.exists():
        print(f"✗ Dataset not found at {dataset_dir}")
        return False
    
    required_files = [
        "en_human_lines.jsonl",
        "en_gpt2_lines.jsonl",
        "en_gpt3_lines.jsonl",
        "en_gptj_lines.jsonl",
        "en_gptneo_lines.jsonl",
        "en_llama_lines.jsonl"
    ]
    
    all_found = True
    for file in required_files:
        file_path = dataset_dir / file
        if file_path.exists():
            # Count lines
            with open(file_path, 'r', encoding='utf-8') as f:
                num_lines = sum(1 for _ in f)
            print(f"✓ {file}: {num_lines} samples")
        else:
            print(f"✗ {file} - NOT FOUND")
            all_found = False
    
    if not all_found:
        return False
    
    print("\n✓ Dataset found and valid!")
    return True


def check_structure():
    """Check project structure."""
    print("\n" + "=" * 60)
    print("CHECKING PROJECT STRUCTURE")
    print("=" * 60)
    
    required_dirs = [
        "data",
        "models",
        "features",
        "attacks",
        "configs",
        "dataset"
    ]
    
    required_files = [
        "train_seqxgpt.py",
        "train_bert.py",
        "eval.py",
        "requirements.txt",
        "configs/seqxgpt_config.yaml",
        "configs/bert_config.yaml"
    ]
    
    all_ok = True
    
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✓ {dir_name}/")
        else:
            print(f"✗ {dir_name}/ - NOT FOUND")
            all_ok = False
    
    for file_name in required_files:
        if Path(file_name).exists():
            print(f"✓ {file_name}")
        else:
            print(f"✗ {file_name} - NOT FOUND")
            all_ok = False
    
    if not all_ok:
        return False
    
    print("\n✓ Project structure is correct!")
    return True


def test_imports_modules():
    """Test importing project modules."""
    print("\n" + "=" * 60)
    print("TESTING MODULE IMPORTS")
    print("=" * 60)
    
    modules = [
        ('data.seqxgpt_dataset', 'SeqXGPTDataset'),
        ('models.seqxgpt', 'SeqXGPTModel'),
        ('models.bert_detector', 'BERTDetector'),
        ('features.llm_probs', 'LLMProbExtractor'),
        ('attacks.text_augmentation', 'TextAugmenter')
    ]
    
    all_ok = True
    for module_name, class_name in modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✓ {module_name}.{class_name}")
        except Exception as e:
            print(f"✗ {module_name}.{class_name} - ERROR: {e}")
            all_ok = False
    
    if not all_ok:
        return False
    
    print("\n✓ All modules can be imported!")
    return True


def main():
    """Run all checks."""
    print("\n🔍 VERIFYING PROJECT SETUP\n")
    
    checks = [
        ("Dependencies", check_imports),
        ("Project Structure", check_structure),
        ("Dataset", check_dataset),
        ("Module Imports", test_imports_modules)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 All checks passed! Project is ready to use.")
        print("\nNext steps:")
        print("  1. Train SeqXGPT: python train_seqxgpt.py")
        print("  2. Train BERT: python train_bert.py")
        print("  3. Evaluate: python eval.py")
    else:
        print("\n⚠ Some checks failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
