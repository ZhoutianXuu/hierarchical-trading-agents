"""
Model Evaluator

Evaluates fine-tuned models on test datasets.
"""

import torch
from typing import List, Dict, Tuple
import logging
import numpy as np
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate fine-tuned models"""
    
    def __init__(self, model, tokenizer, device: str = None):
        """
        Initialize evaluator
        
        Args:
            model: Trained model
            tokenizer: Model tokenizer
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model = model
        self.tokenizer = tokenizer
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate_examples(self, examples: List[Dict], max_length: int = 512) -> Dict:
        """
        Evaluate model on a set of examples
        
        Args:
            examples: List of test examples with 'prompt' and 'completion'
            max_length: Maximum generation length
            
        Returns:
            Dictionary with evaluation metrics
        """
        total_loss = 0.0
        predictions = []
        references = []
        
        with torch.no_grad():
            for example in examples:
                prompt = example['prompt']
                reference = example['completion']
                
                # Generate prediction
                prediction = self.generate(prompt, max_length=max_length)
                
                predictions.append(prediction)
                references.append(reference)
                
                # Compute loss (if possible)
                try:
                    loss = self._compute_loss(prompt, reference)
                    total_loss += loss
                except:
                    pass
        
        # Compute metrics
        avg_loss = total_loss / len(examples) if total_loss > 0 else None
        
        metrics = {
            'num_examples': len(examples),
            'average_loss': avg_loss,
            'predictions': predictions,
            'references': references
        }
        
        return metrics
    
    def generate(self, prompt: str, max_length: int = 512, 
                temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generate response for a prompt
        
        Args:
            prompt: Input prompt
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_length,
            truncation=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt from generated text
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def _compute_loss(self, prompt: str, reference: str) -> float:
        """
        Compute loss for a prompt-reference pair
        
        Args:
            prompt: Input prompt
            reference: Reference completion
            
        Returns:
            Loss value
        """
        full_text = f"{prompt}\n\n{reference}"
        
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        
        return loss.item()
    
    def compute_accuracy(self, predictions: List[str], references: List[str]) -> float:
        """
        Compute accuracy (exact match)
        
        Args:
            predictions: List of predictions
            references: List of references
            
        Returns:
            Accuracy score
        """
        correct = sum(1 for pred, ref in zip(predictions, references) 
                     if pred.strip().lower() == ref.strip().lower())
        accuracy = correct / len(predictions) if predictions else 0.0
        
        return accuracy
    
    def compute_bleu(self, predictions: List[str], references: List[str]) -> float:
        """
        Compute BLEU score (requires nltk)
        
        Args:
            predictions: List of predictions
            references: List of references
            
        Returns:
            BLEU score
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu
            
            scores = []
            for pred, ref in zip(predictions, references):
                score = sentence_bleu([ref.split()], pred.split())
                scores.append(score)
            
            return np.mean(scores) if scores else 0.0
        
        except ImportError:
            logger.warning("NLTK not installed. BLEU score unavailable.")
            return 0.0
    
    def evaluate_comprehensive(self, test_examples: List[Dict], 
                              output_path: str = None) -> Dict:
        """
        Comprehensive evaluation with multiple metrics
        
        Args:
            test_examples: List of test examples
            output_path: Optional path to save results
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Evaluating on {len(test_examples)} examples...")
        
        # Get predictions
        results = self.evaluate_examples(test_examples)
        predictions = results['predictions']
        references = results['references']
        
        # Compute metrics
        metrics = {
            'num_examples': len(test_examples),
            'average_loss': results['average_loss'],
            'exact_match_accuracy': self.compute_accuracy(predictions, references),
            'bleu_score': self.compute_bleu(predictions, references)
        }
        
        # Add sample predictions
        metrics['sample_predictions'] = [
            {
                'prompt': test_examples[i]['prompt'][:100] + '...',
                'reference': references[i][:100] + '...',
                'prediction': predictions[i][:100] + '...'
            }
            for i in range(min(5, len(test_examples)))
        ]
        
        logger.info(f"Evaluation complete. Accuracy: {metrics['exact_match_accuracy']:.2%}")
        
        # Save results
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Results saved to {output_path}")
        
        return metrics