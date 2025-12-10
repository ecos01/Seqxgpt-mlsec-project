"""
SeqXGPT-Bench Dataset Loader
Loads JSONL files from the SeqXGPT/dataset directory and provides train/val/test splits.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset


class SeqXGPTDataset(Dataset):
    """
    Dataset loader for SeqXGPT-Bench.
    Loads human and AI-generated texts from JSONL files.
    """
    
    def __init__(
        self,
        data_dir: str = "dataset/SeqXGPT-Bench",
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        max_samples_per_source: int = None
    ):
        """
        Args:
            data_dir: Path to dataset directory
            split: One of 'train', 'val', 'test'
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            seed: Random seed for reproducibility
            max_samples_per_source: Max samples to load per source (None for all)
        """
        assert split in ["train", "val", "test"], f"Split must be train/val/test, got {split}"
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.seed = seed
        
        # Load data
        self.texts = []
        self.labels = []
        
        # AI-generated sources (label = 1)
        ai_sources = ["en_gpt2_lines.jsonl", "en_gpt3_lines.jsonl", "en_gptj_lines.jsonl", 
                      "en_gptneo_lines.jsonl", "en_llama_lines.jsonl"]
        
        # Human-written (label = 0)
        human_sources = ["en_human_lines.jsonl"]
        
        # Load human texts
        for source in human_sources:
            file_path = self.data_dir / source
            if file_path.exists():
                texts = self._load_jsonl(file_path, max_samples_per_source)
                self.texts.extend(texts)
                self.labels.extend([0] * len(texts))
        
        # Load AI texts
        for source in ai_sources:
            file_path = self.data_dir / source
            if file_path.exists():
                texts = self._load_jsonl(file_path, max_samples_per_source)
                self.texts.extend(texts)
                self.labels.extend([1] * len(texts))
        
        # Create train/val/test split
        self._create_split(train_ratio, val_ratio, test_ratio)
        
        print(f"Loaded {len(self.texts)} samples for {split} split")
        print(f"Label distribution - Human: {self.labels.count(0)}, AI: {self.labels.count(1)}")
    
    def _load_jsonl(self, file_path: Path, max_samples: int = None) -> List[str]:
        """Load texts from JSONL file."""
        texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                data = json.loads(line.strip())
                text = data.get('text', data.get('content', ''))
                if text:
                    texts.append(text)
        return texts
    
    def _create_split(self, train_ratio: float, val_ratio: float, test_ratio: float):
        """Create train/val/test splits."""
        # First split: separate test set
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            self.texts, self.labels, 
            test_size=test_ratio,
            random_state=self.seed,
            stratify=self.labels
        )
        
        # Second split: separate train and val from remaining data
        relative_val_ratio = val_ratio / (train_ratio + val_ratio)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts, train_val_labels,
            test_size=relative_val_ratio,
            random_state=self.seed,
            stratify=train_val_labels
        )
        
        # Assign to correct split
        if self.split == "train":
            self.texts = train_texts
            self.labels = train_labels
        elif self.split == "val":
            self.texts = val_texts
            self.labels = val_labels
        else:  # test
            self.texts = test_texts
            self.labels = test_labels
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[str, int]:
        """
        Returns:
            text: Raw text string
            label: 0 for human, 1 for AI
        """
        return self.texts[idx], self.labels[idx]
    
    def get_texts_and_labels(self) -> Tuple[List[str], List[int]]:
        """Return all texts and labels as lists."""
        return self.texts, self.labels


def test_dataset():
    """Test the dataset loader."""
    print("Testing SeqXGPT Dataset Loader...")
    
    # Test train split
    train_dataset = SeqXGPTDataset(split="train")
    print(f"\nTrain set: {len(train_dataset)} samples")
    
    # Test val split
    val_dataset = SeqXGPTDataset(split="val")
    print(f"Val set: {len(val_dataset)} samples")
    
    # Test test split
    test_dataset = SeqXGPTDataset(split="test")
    print(f"Test set: {len(test_dataset)} samples")
    
    # Print example
    text, label = train_dataset[0]
    print(f"\nExample text (label={label}):")
    print(text[:200] + "...")


if __name__ == "__main__":
    test_dataset()
