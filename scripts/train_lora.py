"""
LoRA Fine-tuning Script for Trading Agents

Usage:
    # Train single agent
    python scripts/train_lora.py --agent fundamental --data data/training/fundamental.jsonl
    
    # Train with custom config
    python scripts/train_lora.py --agent sentiment --data data/training/sentiment.jsonl \
        --epochs 5 --batch-size 8 --learning-rate 5e-4
    
    # Train all agents
    python scripts/train_all_agents.py
"""

import argparse
import logging
from pathlib import Path
import sys
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LoRATrainingConfig:
    """Configuration for LoRA fine-tuning"""
    
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        
        # LoRA parameters
        self.lora_r = 8
        self.lora_alpha = 16
        self.lora_dropout = 0.05 if agent_type != 'sentiment' else 0.10
        self.target_modules = ["q_proj", "v_proj"]
        
        # Training parameters
        self.learning_rate = 3e-4
        self.num_epochs = 3
        self.batch_size = 4
        self.gradient_accumulation_steps = 8
        self.warmup_steps = 100
        self.max_length = 512
        
        # Model
        self.base_model = "Qwen/Qwen3-4B" 
        self.output_dir = f"models/adapters/{agent_type}_agent"
        
        # Optimization
        self.optimizer = "adamw_torch"
        self.weight_decay = 0.01
        self.max_grad_norm = 1.0
        
        # Logging
        self.logging_steps = 10
        self.save_steps = 500
        self.eval_steps = 500
        self.save_total_limit = 3


class TrainingDataset:
    """Prepare dataset for training"""
    
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def load_from_jsonl(self, filepath: str):
        """Load training data from JSONL file"""
        examples = []
        with open(filepath, 'r') as f:
            for line in f:
                examples.append(json.loads(line))
        
        logger.info(f"Loaded {len(examples)} examples from {filepath}")
        return examples
    
    def format_example(self, example: dict) -> str:
        """Format example as prompt + completion"""
        prompt = example.get('prompt', '')
        completion = example.get('completion', '')
        
        # Add instruction formatting
        formatted = f"### Instruction:\n{prompt}\n\n### Response:\n{completion}"
        return formatted
    
    def tokenize_examples(self, examples: list):
        """Tokenize examples"""
        texts = [self.format_example(ex) for ex in examples]
        
        tokenized = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors=None
        )
        
        # Create labels (same as input_ids for causal LM)
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        return Dataset.from_dict(tokenized)


class LoRATrainer:
    """LoRA Trainer for agent fine-tuning"""
    
    def __init__(self, config: LoRATrainingConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
    
    def load_model_and_tokenizer(self):
        """Load base model and apply LoRA"""
        logger.info(f"Loading base model: {self.config.base_model}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True
        )
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            load_in_8bit=True  # 8-bit quantization for memory efficiency
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none"
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, peft_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
    
    def prepare_datasets(self, train_data_path: str, eval_data_path: str = None):
        """Prepare training and evaluation datasets"""
        dataset_builder = TrainingDataset(self.tokenizer, self.config.max_length)
        
        # Load and tokenize training data
        train_examples = dataset_builder.load_from_jsonl(train_data_path)
        self.train_dataset = dataset_builder.tokenize_examples(train_examples)
        
        # Load evaluation data if provided
        if eval_data_path:
            eval_examples = dataset_builder.load_from_jsonl(eval_data_path)
            self.eval_dataset = dataset_builder.tokenize_examples(eval_examples)
        else:
            # Split training data
            split = int(len(self.train_dataset) * 0.9)
            self.eval_dataset = self.train_dataset.select(range(split, len(self.train_dataset)))
            self.train_dataset = self.train_dataset.select(range(split))
        
        logger.info(f"Train dataset size: {len(self.train_dataset)}")
        logger.info(f"Eval dataset size: {len(self.eval_dataset)}")
    
    def train(self):
        """Train the model"""
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=self.config.save_total_limit,
            fp16=True,
            optim=self.config.optimizer,
            max_grad_norm=self.config.max_grad_norm,
            report_to="none",  # Disable wandb/tensorboard
            remove_unused_columns=False
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        logger.info(f"Saving final model to {self.config.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Save config
        with open(f"{self.config.output_dir}/training_config.json", 'w') as f:
            json.dump(vars(self.config), f, indent=2)
        
        logger.info("Training completed!")
        
        return trainer


def main():
    parser = argparse.ArgumentParser(description='Train LoRA adapter for trading agent')
    parser.add_argument('--agent', required=True, 
                       choices=['fundamental', 'sentiment', 'technical', 'risk'],
                       help='Agent type to train')
    parser.add_argument('--data', required=True, 
                       help='Path to training data (JSONL format)')
    parser.add_argument('--eval-data', 
                       help='Path to evaluation data (optional)')
    parser.add_argument('--output', 
                       help='Output directory for adapter')
    parser.add_argument('--base-model', default='Qwen/Qwen3-4B',
                       help='Base model to fine-tune')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--lora-r', type=int, default=8,
                       help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=16,
                       help='LoRA alpha')
    
    args = parser.parse_args()
    
    # Create config
    config = LoRATrainingConfig(args.agent)
    
    # Override with command line args
    if args.output:
        config.output_dir = args.output
    if args.base_model:
        config.base_model = args.base_model
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.lora_r = args.lora_r
    config.lora_alpha = args.lora_alpha
    
    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    trainer = LoRATrainer(config)
    
    # Load model
    trainer.load_model_and_tokenizer()
    
    # Prepare data
    trainer.prepare_datasets(args.data, args.eval_data)
    
    # Train
    trainer.train()
    
    logger.info(f"✓ Training complete!")
    logger.info(f"✓ Model saved to: {config.output_dir}")
    logger.info(f"✓ Load with: AutoPeftModelForCausalLM.from_pretrained('{config.output_dir}')")


if __name__ == '__main__':
    main()