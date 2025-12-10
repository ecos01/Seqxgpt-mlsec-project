"""
LLM Log-Probability Feature Extraction
Extracts log-probabilities from LLM (GPT-2) for each token in the input text.
These features are used as input to the SeqXGPT model.
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional
from pathlib import Path
import pickle
from tqdm import tqdm


class LLMProbExtractor:
    """
    Extract log-probabilities and related features from an LLM.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = None,
        max_length: int = 256,
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            model_name: HuggingFace model name (e.g., 'gpt2', 'gpt2-medium')
            device: Device to run on ('cuda' or 'cpu')
            max_length: Maximum sequence length
            cache_dir: Directory to cache extracted features
        """
        self.model_name = model_name
        self.max_length = max_length
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Loading model {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Model loaded successfully!")
    
    def extract_features(self, text: str) -> Dict[str, np.ndarray]:
        """
        Extract log-probability features from a single text.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary containing:
                - log_probs: Log-probabilities for each token [seq_len]
                - surprisal: Surprisal (negative log-prob) [seq_len]
                - entropy: Token-level entropy [seq_len]
                - tokens: Token IDs [seq_len]
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [1, seq_len, vocab_size]
        
        # Calculate log probabilities
        log_probs_all = F.log_softmax(logits, dim=-1)  # [1, seq_len, vocab_size]
        
        # Get log-prob of actual tokens (shifted by 1 for causal LM)
        seq_len = input_ids.shape[1]
        log_probs = []
        surprisal = []
        entropy = []
        
        for i in range(1, seq_len):  # Start from 1 because first token has no context
            # Log-prob of token i given previous tokens
            token_id = input_ids[0, i]
            log_prob = log_probs_all[0, i-1, token_id].item()
            log_probs.append(log_prob)
            surprisal.append(-log_prob)
            
            # Entropy at position i-1
            probs = torch.exp(log_probs_all[0, i-1, :])
            ent = -(probs * log_probs_all[0, i-1, :]).sum().item()
            entropy.append(ent)
        
        # Pad to max_length if needed
        actual_len = len(log_probs)
        if actual_len < self.max_length:
            pad_len = self.max_length - actual_len
            log_probs.extend([0.0] * pad_len)
            surprisal.extend([0.0] * pad_len)
            entropy.extend([0.0] * pad_len)
        
        return {
            "log_probs": np.array(log_probs[:self.max_length], dtype=np.float32),
            "surprisal": np.array(surprisal[:self.max_length], dtype=np.float32),
            "entropy": np.array(entropy[:self.max_length], dtype=np.float32),
            "actual_length": actual_len
        }
    
    def extract_batch(self, texts: List[str], show_progress: bool = True) -> List[Dict[str, np.ndarray]]:
        """
        Extract features from a batch of texts.
        
        Args:
            texts: List of text strings
            show_progress: Whether to show progress bar
            
        Returns:
            List of feature dictionaries
        """
        features = []
        iterator = tqdm(texts, desc="Extracting features") if show_progress else texts
        
        for text in iterator:
            feat = self.extract_features(text)
            features.append(feat)
        
        return features
    
    def extract_and_cache(
        self,
        texts: List[str],
        cache_name: str,
        force_recompute: bool = False
    ) -> List[Dict[str, np.ndarray]]:
        """
        Extract features and cache to disk.
        
        Args:
            texts: List of text strings
            cache_name: Name for cache file
            force_recompute: Whether to recompute even if cache exists
            
        Returns:
            List of feature dictionaries
        """
        if self.cache_dir is None:
            print("No cache directory specified, computing features...")
            return self.extract_batch(texts)
        
        cache_file = self.cache_dir / f"{cache_name}.pkl"
        
        # Try to load from cache
        if cache_file.exists() and not force_recompute:
            print(f"Loading cached features from {cache_file}...")
            with open(cache_file, "rb") as f:
                features = pickle.load(f)
            print(f"Loaded {len(features)} cached features")
            return features
        
        # Compute features
        print(f"Computing features and caching to {cache_file}...")
        features = self.extract_batch(texts)
        
        # Save to cache
        with open(cache_file, "wb") as f:
            pickle.dump(features, f)
        print(f"Cached {len(features)} features")
        
        return features
    
    def features_to_tensor(self, features: List[Dict[str, np.ndarray]], feature_types: List[str] = None) -> torch.Tensor:
        """
        Convert feature dictionaries to tensor for model input.
        
        Args:
            features: List of feature dictionaries
            feature_types: Which features to include (default: ['log_probs', 'surprisal', 'entropy'])
            
        Returns:
            Tensor of shape [batch_size, max_length, num_features]
        """
        if feature_types is None:
            feature_types = ['log_probs', 'surprisal', 'entropy']
        
        batch_features = []
        for feat_dict in features:
            # Stack selected features
            feat_list = [feat_dict[ft] for ft in feature_types]
            stacked = np.stack(feat_list, axis=-1)  # [max_length, num_features]
            batch_features.append(stacked)
        
        # Convert to tensor
        tensor = torch.from_numpy(np.array(batch_features))  # [batch_size, max_length, num_features]
        return tensor


def test_extractor():
    """Test the feature extractor."""
    print("Testing LLM Probability Extractor...")
    
    # Initialize extractor
    extractor = LLMProbExtractor(
        model_name="gpt2",
        max_length=128,
        cache_dir="features/cache"
    )
    
    # Test texts
    texts = [
        "This is a test sentence written by a human.",
        "The artificial intelligence model generated this text automatically."
    ]
    
    # Extract features
    features = extractor.extract_batch(texts, show_progress=True)
    
    # Print results
    for i, (text, feat) in enumerate(zip(texts, features)):
        print(f"\nText {i+1}: {text[:50]}...")
        print(f"Log-probs shape: {feat['log_probs'].shape}")
        print(f"Actual length: {feat['actual_length']}")
        print(f"Mean log-prob: {feat['log_probs'][:feat['actual_length']].mean():.3f}")
        print(f"Mean surprisal: {feat['surprisal'][:feat['actual_length']].mean():.3f}")
    
    # Test tensor conversion
    tensor = extractor.features_to_tensor(features)
    print(f"\nFeature tensor shape: {tensor.shape}")


if __name__ == "__main__":
    test_extractor()
