"""
Training Module for LoRA Fine-tuning

This module provides components for fine-tuning trading agents using
LoRA (Low-Rank Adaptation) for parameter-efficient training.

Components:
- LoRATrainingConfig: Configuration for LoRA training
- TrainingDataPreparator: Prepare datasets for fine-tuning
- LoRATrainer: Train LoRA adapters
- ModelEvaluator: Evaluate trained models
"""

from .lora_config import LoRATrainingConfig, AGENT_CONFIGS
from .data_preparation import TrainingDataPreparator
from .trainer import LoRATrainer
from .evaluator import ModelEvaluator

__all__ = [
    'LoRATrainingConfig',
    'AGENT_CONFIGS',
    'TrainingDataPreparator',
    'LoRATrainer',
    'ModelEvaluator'
]

__version__ = '1.0.0'