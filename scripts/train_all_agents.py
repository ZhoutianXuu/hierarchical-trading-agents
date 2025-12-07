"""
Train All Trading Agents

This script trains LoRA adapters for all four specialist agents:
- Fundamental Agent
- Sentiment Agent
- Technical Agent
- Risk Agent

Usage:
    python scripts/train_all_agents.py --data-dir data/training
    
    # With custom settings
    python scripts/train_all_agents.py --data-dir data/training --epochs 5 --base-model Qwen/Qwen3-4B
"""

import argparse
import sys
import logging
from pathlib import Path
import subprocess

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_agent(agent_type: str, data_path: str, args):
    """
    Train a single agent
    
    Args:
        agent_type: Type of agent to train
        data_path: Path to training data
        args: Command line arguments
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {agent_type.upper()} Agent")
    logger.info(f"{'='*60}\n")
    
    # Build command
    cmd = [
        sys.executable,
        "scripts/train_lora.py",
        "--agent", agent_type,
        "--data", data_path
    ]
    
    # Add optional arguments
    if args.base_model:
        cmd.extend(["--base-model", args.base_model])
    if args.epochs:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.batch_size:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.learning_rate:
        cmd.extend(["--learning-rate", str(args.learning_rate)])
    if args.lora_r:
        cmd.extend(["--lora-r", str(args.lora_r)])
    if args.lora_alpha:
        cmd.extend(["--lora-alpha", str(args.lora_alpha)])
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True)
        logger.info(f"✓ {agent_type.upper()} agent training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {agent_type.upper()} agent training failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Train all trading agents')
    
    # Data arguments
    parser.add_argument('--data-dir', required=True,
                       help='Directory containing training data for all agents')
    
    # Model arguments
    parser.add_argument('--base-model', default='Qwen/Qwen3-4B',
                       help='Base model to fine-tune')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    
    # LoRA arguments
    parser.add_argument('--lora-r', type=int, default=8,
                       help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=16,
                       help='LoRA alpha')
    
    # Agent selection
    parser.add_argument('--agents', nargs='+',
                       choices=['fundamental', 'sentiment', 'technical', 'risk', 'all'],
                       default=['all'],
                       help='Which agents to train (default: all)')
    
    # Other
    parser.add_argument('--skip-errors', action='store_true',
                       help='Continue training other agents if one fails')
    
    args = parser.parse_args()
    
    # Determine which agents to train
    if 'all' in args.agents:
        agents_to_train = ['fundamental', 'sentiment', 'technical', 'risk']
    else:
        agents_to_train = args.agents
    
    # Validate data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    # Check for training data files
    missing_data = []
    for agent_type in agents_to_train:
        data_file = data_dir / f"{agent_type}.jsonl"
        if not data_file.exists():
            missing_data.append(agent_type)
            logger.warning(f"Training data not found: {data_file}")
    
    if missing_data:
        logger.error(f"Missing training data for: {', '.join(missing_data)}")
        logger.info("Prepare training data with: python scripts/prepare_training_data.py")
        return
    
    # Train each agent
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {len(agents_to_train)} Agents")
    logger.info(f"Agents: {', '.join([a.upper() for a in agents_to_train])}")
    logger.info(f"Base Model: {args.base_model}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"{'='*60}\n")
    
    results = {}
    for agent_type in agents_to_train:
        data_path = str(data_dir / f"{agent_type}.jsonl")
        success = train_agent(agent_type, data_path, args)
        results[agent_type] = success
        
        if not success and not args.skip_errors:
            logger.error("Training failed. Use --skip-errors to continue with other agents.")
            break
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Training Summary")
    logger.info(f"{'='*60}")
    
    successful = [agent for agent, success in results.items() if success]
    failed = [agent for agent, success in results.items() if not success]
    
    logger.info(f"Total agents: {len(results)}")
    logger.info(f"Successful: {len(successful)} - {', '.join(successful)}")
    if failed:
        logger.info(f"Failed: {len(failed)} - {', '.join(failed)}")
    
    logger.info(f"\n{'='*60}")
    logger.info("All Training Complete!")
    logger.info(f"{'='*60}")
    
    # Print next steps
    logger.info("\nNext steps:")
    logger.info("1. Evaluate models: python scripts/evaluate_agents.py")
    logger.info("2. Test with examples: python examples/run_finetuned_model.py")
    logger.info("3. Deploy to system: Update multi_agent_trading_system.py\n")


if __name__ == '__main__':
    main()