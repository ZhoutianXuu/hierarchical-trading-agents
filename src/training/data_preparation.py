"""
Training Data Preparation

Prepares datasets for fine-tuning trading agents.
"""

import pandas as pd
import json
from typing import List, Dict, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TrainingDataPreparator:
    """Prepare datasets for agent fine-tuning"""
    
    def __init__(self, data_dir: str = "data/training"):
        """
        Initialize data preparator
        
        Args:
            data_dir: Directory for training data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_fundamental_data(self, df: pd.DataFrame) -> List[Dict]:
        """
        Prepare fundamental analysis training examples
        
        Args:
            df: DataFrame with columns: ticker, date, pe, pb, ps, fundamental_analysis
            
        Returns:
            List of training examples
        """
        examples = []
        
        for _, row in df.iterrows():
            prompt = self._create_fundamental_prompt(row)
            completion = row.get('fundamental_analysis', '')
            
            if completion:  # Only include if we have a label
                examples.append({
                    'prompt': prompt,
                    'completion': completion,
                    'metadata': {
                        'ticker': row.get('ticker', ''),
                        'date': str(row.get('date', ''))
                    }
                })
        
        logger.info(f"Prepared {len(examples)} fundamental training examples")
        return examples
    
    def prepare_sentiment_data(self, df: pd.DataFrame) -> List[Dict]:
        """
        Prepare sentiment analysis training examples
        
        Args:
            df: DataFrame with columns: ticker, date, news, sentiment_label
            
        Returns:
            List of training examples
        """
        examples = []
        
        for _, row in df.iterrows():
            prompt = self._create_sentiment_prompt(row)
            completion = row.get('sentiment_label', '')
            
            if completion:
                examples.append({
                    'prompt': prompt,
                    'completion': completion,
                    'metadata': {
                        'ticker': row.get('ticker', ''),
                        'date': str(row.get('date', ''))
                    }
                })
        
        logger.info(f"Prepared {len(examples)} sentiment training examples")
        return examples
    
    def prepare_technical_data(self, df: pd.DataFrame) -> List[Dict]:
        """
        Prepare technical analysis training examples
        
        Args:
            df: DataFrame with OHLCV data and technical_analysis label
            
        Returns:
            List of training examples
        """
        examples = []
        
        for _, row in df.iterrows():
            prompt = self._create_technical_prompt(row)
            completion = row.get('technical_analysis', '')
            
            if completion:
                examples.append({
                    'prompt': prompt,
                    'completion': completion,
                    'metadata': {
                        'ticker': row.get('ticker', ''),
                        'date': str(row.get('date', ''))
                    }
                })
        
        logger.info(f"Prepared {len(examples)} technical training examples")
        return examples
    
    def prepare_risk_data(self, df: pd.DataFrame) -> List[Dict]:
        """
        Prepare risk assessment training examples
        
        Args:
            df: DataFrame with risk metrics and risk_assessment label
            
        Returns:
            List of training examples
        """
        examples = []
        
        for _, row in df.iterrows():
            prompt = self._create_risk_prompt(row)
            completion = row.get('risk_assessment', '')
            
            if completion:
                examples.append({
                    'prompt': prompt,
                    'completion': completion,
                    'metadata': {
                        'ticker': row.get('ticker', ''),
                        'date': str(row.get('date', ''))
                    }
                })
        
        logger.info(f"Prepared {len(examples)} risk training examples")
        return examples
    
    def _create_fundamental_prompt(self, row: pd.Series) -> str:
        """Create prompt for fundamental analysis"""
        ticker = row.get('ticker', 'UNKNOWN')
        pe = row.get('pe', 'N/A')
        pb = row.get('pb', 'N/A')
        ps = row.get('ps', 'N/A')
        pcf = row.get('pcf_ttm', 'N/A')
        
        prompt = f"""Analyze the fundamental metrics for {ticker}:

Valuation Ratios:
- P/E Ratio: {pe}
- P/B Ratio: {pb}
- P/S Ratio: {ps}
- Price/Cash Flow: {pcf}

Provide a fundamental analysis assessment."""
        
        return prompt
    
    def _create_sentiment_prompt(self, row: pd.Series) -> str:
        """Create prompt for sentiment analysis"""
        ticker = row.get('ticker', 'UNKNOWN')
        news = row.get('news', '[]')
        
        # Parse news if it's a JSON string
        if isinstance(news, str):
            try:
                news_items = json.loads(news)
                news_text = "\n".join([f"- {item.get('title', '')}" for item in news_items[:5]])
            except:
                news_text = news[:500]
        else:
            news_text = str(news)[:500]
        
        prompt = f"""Analyze market sentiment for {ticker} based on recent news:

News Headlines:
{news_text}

Provide a sentiment analysis (positive, negative, or neutral)."""
        
        return prompt
    
    def _create_technical_prompt(self, row: pd.Series) -> str:
        """Create prompt for technical analysis"""
        ticker = row.get('ticker', 'UNKNOWN')
        open_price = row.get('open', 'N/A')
        high = row.get('high', 'N/A')
        low = row.get('low', 'N/A')
        close = row.get('close', 'N/A')
        volume = row.get('volume', 'N/A')
        
        prompt = f"""Analyze the technical indicators for {ticker}:

Price Action:
- Open: {open_price}
- High: {high}
- Low: {low}
- Close: {close}
- Volume: {volume}

Provide a technical analysis assessment."""
        
        return prompt
    
    def _create_risk_prompt(self, row: pd.Series) -> str:
        """Create prompt for risk assessment"""
        ticker = row.get('ticker', 'UNKNOWN')
        volatility = row.get('volatility', 'N/A')
        beta = row.get('beta', 'N/A')
        
        prompt = f"""Assess the risk profile for {ticker}:

Risk Metrics:
- Volatility: {volatility}
- Beta: {beta}

Provide a risk assessment (low, medium, or high risk)."""
        
        return prompt
    
    def save_dataset(self, examples: List[Dict], output_path: str):
        """
        Save prepared dataset as JSONL
        
        Args:
            examples: List of training examples
            output_path: Path to save file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
        
        logger.info(f"Saved {len(examples)} examples to {output_path}")
    
    def load_dataset(self, input_path: str) -> List[Dict]:
        """
        Load dataset from JSONL file
        
        Args:
            input_path: Path to JSONL file
            
        Returns:
            List of training examples
        """
        examples = []
        with open(input_path, 'r') as f:
            for line in f:
                examples.append(json.loads(line))
        
        logger.info(f"Loaded {len(examples)} examples from {input_path}")
        return examples
    
    def train_test_split(self, examples: List[Dict], test_size: float = 0.2, 
                        shuffle: bool = True) -> Tuple[List[Dict], List[Dict]]:
        """
        Split dataset into train and test
        
        Args:
            examples: List of examples
            test_size: Fraction for test set
            shuffle: Whether to shuffle before splitting
            
        Returns:
            Tuple of (train_examples, test_examples)
        """
        if shuffle:
            import random
            random.shuffle(examples)
        
        split_idx = int(len(examples) * (1 - test_size))
        train = examples[:split_idx]
        test = examples[split_idx:]
        
        logger.info(f"Split into {len(train)} train and {len(test)} test examples")
        return train, test
    
    def prepare_all_agents(self, data_path: str, output_dir: str = None):
        """
        Prepare training data for all agent types
        
        Args:
            data_path: Path to source data CSV
            output_dir: Output directory (defaults to self.data_dir)
        """
        if output_dir is None:
            output_dir = self.data_dir
        else:
            output_dir = Path(output_dir)
        
        # Load data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data from {data_path}: {len(df)} rows")
        
        # Prepare each agent type
        agent_preparers = {
            'fundamental': self.prepare_fundamental_data,
            'sentiment': self.prepare_sentiment_data,
            'technical': self.prepare_technical_data,
            'risk': self.prepare_risk_data
        }
        
        for agent_type, preparer in agent_preparers.items():
            examples = preparer(df)
            if examples:
                output_path = output_dir / f"{agent_type}.jsonl"
                self.save_dataset(examples, str(output_path))
        
        logger.info(f"Prepared training data for all agents in {output_dir}")