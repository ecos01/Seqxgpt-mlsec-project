"""
Extra Dataset Loader
Generic loader for additional datasets (e.g., Kaggle, custom datasets).
Provides same interface as SeqXGPTDataset for compatibility.
"""

import json
import csv
from pathlib import Path
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class ExtraDataset(Dataset):
    """
    Generic dataset loader for additional datasets.
    Supports CSV and JSONL formats.
    """
    
    def __init__(
        self,
        data_file: str,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        text_column: str = "text",
        label_column: str = "label",
        max_samples: int = None
    ):
        """
        Args:
            data_file: Path to CSV or JSONL file
            split: One of 'train', 'val', 'test'
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            seed: Random seed for reproducibility
            text_column: Name of column containing text
            label_column: Name of column containing label (0=human, 1=AI)
            max_samples: Maximum samples to load (None for all)
        """
        assert split in ["train", "val", "test"], f"Split must be train/val/test, got {split}"
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        self.data_file = Path(data_file)
        self.split = split
        self.seed = seed
        
        # Load data
        if self.data_file.suffix == ".csv":
            self.texts, self.labels = self._load_csv(text_column, label_column, max_samples)
        elif self.data_file.suffix == ".jsonl":
            self.texts, self.labels = self._load_jsonl(text_column, label_column, max_samples)
        else:
            raise ValueError(f"Unsupported file format: {self.data_file.suffix}")
        
        # Create train/val/test split
        self._create_split(train_ratio, val_ratio, test_ratio)
        
        print(f"Loaded {len(self.texts)} samples for {split} split")
        print(f"Label distribution - Human: {self.labels.count(0)}, AI: {self.labels.count(1)}")
    
    def _load_csv(self, text_col: str, label_col: str, max_samples: int = None) -> Tuple[List[str], List[int]]:
        """Load data from CSV file."""
        texts = []
        labels = []
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if max_samples and i >= max_samples:
                    break
                
                text = row.get(text_col, '')
                label = int(row.get(label_col, -1))
                
                if text and label in [0, 1]:
                    texts.append(text)
                    labels.append(label)
        
        return texts, labels
    
    def _load_jsonl(self, text_col: str, label_col: str, max_samples: int = None) -> Tuple[List[str], List[int]]:
        """Load data from JSONL file."""
        texts = []
        labels = []
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                
                data = json.loads(line.strip())
                text = data.get(text_col, '')
                label = int(data.get(label_col, -1))
                
                if text and label in [0, 1]:
                    texts.append(text)
                    labels.append(label)
        
        return texts, labels
    
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


def create_example_dataset():
    """Create an example CSV dataset for testing."""
    example_file = Path("data/example_dataset.csv")
    example_file.parent.mkdir(exist_ok=True)
    
    with open(example_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'label'])
        
        # Add some example data
        writer.writerow(['This is a human-written text sample.', 0])
        writer.writerow(['Another human text example here.', 0])
        writer.writerow(['AI-generated text sample number one.', 1])
        writer.writerow(['Another AI-generated example text.', 1])
    
    print(f"Created example dataset at {example_file}")
    return str(example_file)


if __name__ == "__main__":
    # Create and test with example dataset
    example_file = create_example_dataset()
    
    print("\nTesting Extra Dataset Loader...")
    dataset = ExtraDataset(example_file, split="train")
    print(f"Loaded {len(dataset)} samples")
    
    if len(dataset) > 0:
        text, label = dataset[0]
        print(f"\nExample: (label={label})")
        print(text)
