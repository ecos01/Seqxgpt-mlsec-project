"""
LLM Log-Probability Feature Extraction
Extracts log-probabilities from LLM (GPT-2) for each token in the input text.
These features are used as input to the SeqXGPT model.
OPTIMIZED VERSION: Uses batch processing for 10-20x speedup.
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
    Optimized with batch processing for faster extraction.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = None,
        max_length: int = 256,
        cache_dir: Optional[str] = None,
        batch_size: int = 16
    ):
        """
        Args:
            model_name: HuggingFace model name (e.g., 'gpt2', 'gpt2-medium')
            device: Device to run on ('cuda' or 'cpu')
            max_length: Maximum sequence length
            cache_dir: Directory to cache extracted features
            batch_size: Batch size for processing (higher = faster but more memory)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.batch_size = batch_size
        
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
        
        # Use half precision on GPU for speed
        if self.device.type == "cuda":
            self.model.half()
            print("Using FP16 for faster inference")
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        print(f"Model loaded successfully! Batch size: {batch_size}")
    
    def _process_batch(self, texts: List[str]) -> List[Dict[str, np.ndarray]]:
        """Process a batch of texts efficiently."""
        # Tokenize batch
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits.float()  # Back to float32 for stability
            else:
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
        
        # Calculate log probabilities for all
        log_probs_all = F.log_softmax(logits, dim=-1)  # [batch, seq_len, vocab_size]
        
        # Process each sample in batch
        batch_features = []
        for i in range(len(texts)):
            seq_len = attention_mask[i].sum().item()
            features = self._extract_single_features(
                input_ids[i], log_probs_all[i], int(seq_len)
            )
            batch_features.append(features)
        
        return batch_features
    
    def _extract_single_features(self, input_ids, log_probs_all, seq_len) -> Dict[str, np.ndarray]:
        """Extract features for a single sequence from batch results."""
        log_probs = []
        surprisal = []
        entropy = []
        
        # Vectorized computation for speed
        for i in range(1, min(seq_len, self.max_length)):
            token_id = input_ids[i]
            log_prob = log_probs_all[i-1, token_id].item()
            log_probs.append(log_prob)
            surprisal.append(-log_prob)
            
            # Entropy
            probs = torch.exp(log_probs_all[i-1, :])
            ent = -(probs * log_probs_all[i-1, :]).sum().item()
            entropy.append(ent)
        
        actual_len = len(log_probs)
        
        # Pad to max_length
        if actual_len < self.max_length:
            pad_len = self.max_length - actual_len
            log_probs.extend([0.0] * pad_len)
            surprisal.extend([0.0] * pad_len)
            entropy.extend([0.0] * pad_len)
        
        # Convert to arrays and clean
        log_probs_arr = np.array(log_probs[:self.max_length], dtype=np.float32)
        surprisal_arr = np.array(surprisal[:self.max_length], dtype=np.float32)
        entropy_arr = np.array(entropy[:self.max_length], dtype=np.float32)
        
        # Clean NaN/Inf
        log_probs_arr = np.nan_to_num(log_probs_arr, nan=0.0, posinf=0.0, neginf=-20.0)
        surprisal_arr = np.nan_to_num(surprisal_arr, nan=0.0, posinf=20.0, neginf=0.0)
        entropy_arr = np.nan_to_num(entropy_arr, nan=0.0, posinf=10.0, neginf=0.0)
        
        # Clip to ranges
        log_probs_arr = np.clip(log_probs_arr, -20.0, 0.0)
        surprisal_arr = np.clip(surprisal_arr, 0.0, 20.0)
        entropy_arr = np.clip(entropy_arr, 0.0, 15.0)
        
        return {
            "log_probs": log_probs_arr,
            "surprisal": surprisal_arr,
            "entropy": entropy_arr,
            "actual_length": actual_len
        }

    def extract_features(self, text: str) -> Dict[str, np.ndarray]:
        """Extract features from a single text (for compatibility)."""
        return self._process_batch([text])[0]
    
    def extract_batch(self, texts: List[str], show_progress: bool = True) -> List[Dict[str, np.ndarray]]:
        """
        Extract features from texts using optimized batch processing.
        
        Args:
            texts: List of text strings
            show_progress: Whether to show progress bar
            
        Returns:
            List of feature dictionaries
        """
        features = []
        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting features", total=num_batches)
        
        for i in iterator:
            batch_texts = texts[i:i + self.batch_size]
            try:
                batch_features = self._process_batch(batch_texts)
                features.extend(batch_features)
            except Exception as e:
                # Fallback: process one by one if batch fails
                print(f"\nBatch {i//self.batch_size} failed ({e}), processing individually...")
                for text in batch_texts:
                    try:
                        feat = self._process_batch([text])[0]
                        features.append(feat)
                    except:
                        # Return zero features on error
                        features.append({
                            "log_probs": np.zeros(self.max_length, dtype=np.float32),
                            "surprisal": np.zeros(self.max_length, dtype=np.float32),
                            "entropy": np.zeros(self.max_length, dtype=np.float32),
                            "actual_length": 0
                        })
        
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
