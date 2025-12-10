"""
BERT-based Detector
Fine-tuned BERT/RoBERTa for AI text detection.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Tuple


class BERTDetector(nn.Module):
    """
    BERT-based detector for AI-generated text.
    Uses pre-trained BERT/RoBERTa with a classification head.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            model_name: HuggingFace model name (e.g., 'bert-base-uncased', 'roberta-base')
            num_labels: Number of output classes (2 for binary classification)
            dropout: Dropout probability for classification head
        """
        super(BERTDetector, self).__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load pre-trained model
        print(f"Loading {model_name}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Model loaded successfully!")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Optional labels [batch_size]
            
        Returns:
            Dictionary with 'logits', 'loss' (if labels provided)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        result = {
            "logits": outputs.logits
        }
        
        if labels is not None:
            result["loss"] = outputs.loss
        
        return result
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get predictions (probabilities for class 1).
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Probabilities for AI class [batch_size]
        """
        outputs = self.forward(input_ids, attention_mask)
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=-1)
        return probs[:, 1]  # Probability of class 1 (AI)
    
    def predict_texts(
        self,
        texts: List[str],
        max_length: int = 512,
        batch_size: int = 8,
        device: str = "cuda"
    ) -> List[float]:
        """
        Predict on raw texts.
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            batch_size: Batch size for inference
            device: Device to run on
            
        Returns:
            List of probabilities
        """
        self.model.to(device)
        self.model.eval()
        
        all_probs = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            encoding = self.tokenizer(
                batch_texts,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            
            # Predict
            with torch.no_grad():
                probs = self.predict(input_ids, attention_mask)
            
            all_probs.extend(probs.cpu().tolist())
        
        return all_probs


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Test the BERT detector."""
    print("Testing BERT Detector...")
    
    # Create model
    model = BERTDetector(model_name="bert-base-uncased")
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test texts
    texts = [
        "This is a human-written text sample.",
        "AI-generated text example here."
    ]
    
    # Tokenize
    encoding = model.tokenizer(
        texts,
        max_length=128,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    # Forward pass
    outputs = model.forward(
        input_ids=encoding["input_ids"],
        attention_mask=encoding["attention_mask"]
    )
    
    print(f"Logits shape: {outputs['logits'].shape}")
    
    # Predict
    probs = model.predict(
        input_ids=encoding["input_ids"],
        attention_mask=encoding["attention_mask"]
    )
    
    print(f"Predictions shape: {probs.shape}")
    print(f"Predictions: {probs}")
    
    # Test predict_texts
    device = "cuda" if torch.cuda.is_available() else "cpu"
    probs_list = model.predict_texts(texts, device=device)
    print(f"\nPredictions from texts: {probs_list}")


if __name__ == "__main__":
    test_model()
