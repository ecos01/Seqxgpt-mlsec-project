"""
Text Augmentation for Evasion Attacks
Implements paraphrasing and translation attacks to test detector robustness.
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    MarianMTModel, MarianTokenizer
)
from typing import List, Dict
import os


class TextAugmenter:
    """
    Text augmentation for evasion attacks.
    Supports paraphrasing and back-translation.
    """
    
    def __init__(self, device: str = None):
        """
        Args:
            device: Device to run on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Will load models on demand
        self.paraphrase_model = None
        self.paraphrase_tokenizer = None
        self.translation_models = {}
    
    def _load_paraphrase_model(self):
        """Load paraphrase model (T5-based)."""
        if self.paraphrase_model is None:
            print("Loading paraphrase model (T5)...")
            model_name = "Vamsi/T5_Paraphrase_Paws"
            try:
                self.paraphrase_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                self.paraphrase_model.to(self.device)
                self.paraphrase_model.eval()
                print("✓ Paraphrase model loaded")
            except Exception as e:
                print(f"Warning: Could not load paraphrase model: {e}")
                print("Falling back to simple paraphrase (word replacement)")
                self.paraphrase_model = "fallback"
    
    def _load_translation_model(self, source_lang: str, target_lang: str):
        """Load translation model."""
        key = f"{source_lang}-{target_lang}"
        
        if key not in self.translation_models:
            print(f"Loading translation model ({key})...")
            model_name = f"Helsinki-NLP/opus-mt-{key}"
            
            try:
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)
                model.to(self.device)
                model.eval()
                
                self.translation_models[key] = {
                    'tokenizer': tokenizer,
                    'model': model
                }
                print(f"✓ Translation model loaded ({key})")
            except Exception as e:
                print(f"Warning: Could not load translation model {key}: {e}")
                self.translation_models[key] = None
    
    def paraphrase(self, text: str, num_return_sequences: int = 1) -> List[str]:
        """
        Paraphrase text.
        
        Args:
            text: Input text
            num_return_sequences: Number of paraphrases to generate
            
        Returns:
            List of paraphrased texts
        """
        self._load_paraphrase_model()
        
        if self.paraphrase_model == "fallback":
            # Simple fallback: no actual paraphrase
            return [text]
        
        if self.paraphrase_model is None:
            return [text]
        
        # Prepare input
        input_text = f"paraphrase: {text} </s>"
        encoding = self.paraphrase_tokenizer(
            input_text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.paraphrase_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=512,
                num_return_sequences=num_return_sequences,
                num_beams=num_return_sequences,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )
        
        # Decode
        paraphrases = []
        for output in outputs:
            paraphrase = self.paraphrase_tokenizer.decode(output, skip_special_tokens=True)
            paraphrases.append(paraphrase)
        
        return paraphrases
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text.
        
        Args:
            text: Input text
            source_lang: Source language code (e.g., 'en')
            target_lang: Target language code (e.g., 'it')
            
        Returns:
            Translated text
        """
        key = f"{source_lang}-{target_lang}"
        self._load_translation_model(source_lang, target_lang)
        
        if key not in self.translation_models or self.translation_models[key] is None:
            print(f"Translation model not available for {key}")
            return text
        
        tokenizer = self.translation_models[key]['tokenizer']
        model = self.translation_models[key]['model']
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Translate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=512
            )
        
        # Decode
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation
    
    def back_translate(self, text: str, intermediate_lang: str = 'it') -> str:
        """
        Back-translate text (en -> intermediate -> en).
        
        Args:
            text: Input text in English
            intermediate_lang: Intermediate language code (e.g., 'it', 'de', 'fr')
            
        Returns:
            Back-translated text
        """
        # Translate to intermediate language
        translated = self.translate(text, 'en', intermediate_lang)
        
        # Translate back to English
        back_translated = self.translate(translated, intermediate_lang, 'en')
        
        return back_translated
    
    def augment_batch(
        self,
        texts: List[str],
        methods: List[str] = ['paraphrase', 'back_translate']
    ) -> Dict[str, List[str]]:
        """
        Apply multiple augmentation methods to a batch of texts.
        
        Args:
            texts: List of input texts
            methods: List of augmentation methods to apply
            
        Returns:
            Dictionary mapping method names to augmented texts
        """
        results = {'original': texts}
        
        if 'paraphrase' in methods:
            print("\nGenerating paraphrases...")
            paraphrases = []
            for text in texts:
                paraphrased = self.paraphrase(text, num_return_sequences=1)
                paraphrases.append(paraphrased[0])
            results['paraphrase'] = paraphrases
        
        if 'back_translate' in methods:
            print("\nGenerating back-translations...")
            back_translations = []
            for text in texts:
                back_translated = self.back_translate(text, intermediate_lang='it')
                back_translations.append(back_translated)
            results['back_translate'] = back_translations
        
        return results


def test_augmenter():
    """Test the text augmenter."""
    print("Testing Text Augmenter...")
    
    augmenter = TextAugmenter()
    
    # Test texts
    texts = [
        "This is a simple test sentence for paraphrasing.",
        "AI-generated text detection is an important research area."
    ]
    
    print("\n=== Testing Paraphrase ===")
    for text in texts:
        print(f"\nOriginal: {text}")
        paraphrases = augmenter.paraphrase(text, num_return_sequences=2)
        for i, para in enumerate(paraphrases, 1):
            print(f"Paraphrase {i}: {para}")
    
    print("\n=== Testing Back-Translation ===")
    for text in texts[:1]:  # Test only first text
        print(f"\nOriginal: {text}")
        
        # Try en->it->en
        try:
            back_trans = augmenter.back_translate(text, intermediate_lang='it')
            print(f"Back-translated (via Italian): {back_trans}")
        except Exception as e:
            print(f"Back-translation failed: {e}")


def create_attack_dataset(
    original_texts: List[str],
    original_labels: List[int],
    augmenter: TextAugmenter,
    output_file: str
):
    """
    Create an attack dataset with augmented samples.
    
    Args:
        original_texts: Original texts
        original_labels: Original labels
        augmenter: TextAugmenter instance
        output_file: Output file path
    """
    import json
    
    # Filter only AI-generated texts (label=1)
    ai_texts = [text for text, label in zip(original_texts, original_labels) if label == 1]
    
    print(f"\nCreating attack dataset with {len(ai_texts)} AI-generated texts...")
    
    # Augment
    augmented = augmenter.augment_batch(ai_texts[:100], methods=['paraphrase', 'back_translate'])
    
    # Save
    attack_data = []
    for i, text in enumerate(augmented['original']):
        attack_data.append({
            'original': text,
            'paraphrase': augmented.get('paraphrase', [None])[i],
            'back_translate': augmented.get('back_translate', [None])[i],
            'label': 1  # All are AI-generated
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(attack_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Attack dataset saved to {output_file}")


if __name__ == "__main__":
    test_augmenter()
