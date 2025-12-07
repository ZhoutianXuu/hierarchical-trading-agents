"""
LoRA Configuration for Fine-tuning

Defines configuration classes and presets for LoRA fine-tuning
of different agent types.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LoRATrainingConfig:
    """
    Configuration object for LoRA fine-tuning on Qwen models.
    """

    # Agent metadata
    agent_type: str = "fundamental"

    # LoRA hyperparameters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    bias: str = "none"  # Options: ["none", "all", "lora_only"]

    # Model backend
    base_model: str = "Qwen/Qwen3-4B"
    max_length: int = 512

    # Training hyperparameters
    learning_rate: float = 3e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 100

    # Optimization
    optimizer: str = "adamw_torch"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"

    # Quantization
    load_in_8bit: bool = True
    load_in_4bit: bool = False

    # Logging & checkpointing
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3

    # Output path
    output_dir: str = "models/adapters/agent"

    # Data I/O
    train_data_path: Optional[str] = None
    eval_data_path: Optional[str] = None

    # Misc
    seed: int = 42
    fp16: bool = True
    bf16: bool = False

    def __post_init__(self):
        """Automatically create agent-specific output directory."""
        if self.output_dir == "models/adapters/agent":
            self.output_dir = f"models/adapters/{self.agent_type}_agent"

    def to_dict(self) -> dict:
        """Return a lightweight export dictionary."""
        return {
            "agent_type": self.agent_type,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "base_model": self.base_model,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "output_dir": self.output_dir,
        }

# Predefined configurations for each agent type
AGENT_CONFIGS = {
    'fundamental': LoRATrainingConfig(
        agent_type='fundamental',
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        learning_rate=3e-4,
        num_epochs=3,
        output_dir="models/adapters/fundamental_agent"
    ),
    
    'sentiment': LoRATrainingConfig(
        agent_type='sentiment',
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.10,  # Higher dropout for sentiment
        learning_rate=3e-4,
        num_epochs=3,
        output_dir="models/adapters/sentiment_agent"
    ),
    
    'technical': LoRATrainingConfig(
        agent_type='technical',
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        learning_rate=3e-4,
        num_epochs=3,
        output_dir="models/adapters/technical_agent"
    ),
    
    'risk': LoRATrainingConfig(
        agent_type='risk',
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        learning_rate=3e-4,
        num_epochs=3,
        output_dir="models/adapters/risk_agent"
    ),
    
    # Coordinator agents (if needed)
    'reviewer': LoRATrainingConfig(
        agent_type='reviewer',
        lora_r=16,  # Larger rank for coordinator
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "o_proj"],
        learning_rate=2e-4,
        num_epochs=3,
        output_dir="models/adapters/reviewer_agent"
    ),
    
    'decision': LoRATrainingConfig(
        agent_type='decision',
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "o_proj"],
        learning_rate=2e-4,
        num_epochs=3,
        output_dir="models/adapters/decision_agent"
    ),
}


def get_agent_config(agent_type: str) -> LoRATrainingConfig:
    """
    Get configuration for a specific agent type
    
    Args:
        agent_type: Type of agent ('fundamental', 'sentiment', 'technical', 'risk')
        
    Returns:
        LoRATrainingConfig for the agent
    """
    if agent_type not in AGENT_CONFIGS:
        raise ValueError(f"Unknown agent type: {agent_type}. "
                        f"Available: {list(AGENT_CONFIGS.keys())}")
    
    return AGENT_CONFIGS[agent_type]


def create_custom_config(agent_type: str, **kwargs) -> LoRATrainingConfig:
    """
    Create custom configuration by overriding defaults
    
    Args:
        agent_type: Type of agent
        **kwargs: Configuration parameters to override
        
    Returns:
        Custom LoRATrainingConfig
    """
    base_config = get_agent_config(agent_type)
    
    # Update with custom parameters
    for key, value in kwargs.items():
        if hasattr(base_config, key):
            setattr(base_config, key, value)
    
    return base_config