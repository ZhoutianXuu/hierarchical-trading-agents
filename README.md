# Hierarchical Multi-Agent LLM Architecture for Stock Trading

## Executive Summary

This project implements a modular, retrieval-augmented multi-agent LLM architecture with hierarchical reasoning for stock trading assistance. The system addresses limitations of traditional machine learning and monolithic LLM approaches by introducing:

1. **Modular lightweight LLM agents** specialized by task
2. **Layered reasoning structure** for interpretable coordination
3. **Retrieval-augmented generation (RAG)** for dynamic knowledge grounding

## Project Structure

```
.
├── multi_agent_trading_system.py  # Core multi-agent framework
├── data_pipeline.py                # Data collection and preprocessing
├── evaluation.py                   # Performance evaluation metrics
├── run_examples.py                 # Example usage and demonstrations
├── requirements.txt                # Python dependencies
├── config.yaml                     # Configuration settings
└── README.md                       # This file
```

## Architecture Overview

### Three-Layer Hierarchical Structure

#### Layer 1: Modular Multi-Agent Layer
- **Fundamental Agent**: Analyzes valuation ratios, financial health, earnings
- **Sentiment Agent**: Processes news sentiment and market psychology
- **Technical Agent**: Evaluates price action, volume, support/resistance
- **Risk Management Agent**: Assesses portfolio risk and volatility

Each agent operates as an adapter-tuned or LoRA-based lightweight variant of a shared base model.

#### Layer 2: Hierarchical Reasoning Layer
- **Reviewer Agent**: Synthesizes intermediate reasoning traces using Chain-of-Thought aggregation
- **Decision Agent**: Integrates synthesized insights to produce final trading signals

#### Layer 3: Retrieval-Augmented Knowledge Layer
- **RAG Pipeline**: Retrieves relevant financial documents, company filings, and news
- **Dynamic Grounding**: Enhances factual accuracy and adaptability to real-time data

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for LLM inference)
- 16GB+ RAM
- Internet connection (for data fetching)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd multi-agent-trading
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install additional models:
```bash
# For LLaMA models
huggingface-cli login

# For LoRA adapters
pip install peft
```

## Usage

### Quick Start

```python
from multi_agent_trading_system import HierarchicalMultiAgentSystem, MarketData
from data_pipeline import DataPipeline
from datetime import datetime

# Initialize system
system = HierarchicalMultiAgentSystem(
    model_name="meta-llama/Llama-3-8B",
    use_quantization=True
)

# Fetch market data
pipeline = DataPipeline()
market_data = pipeline.prepare_market_data_object(
    ticker='AAPL',
    date=datetime(2023, 6, 15)
)

# Process trading decision
decision = system.process_trading_decision(market_data)

print(f"Action: {decision.action}")
print(f"Confidence: {decision.confidence:.2%}")
print(f"Rationale: {decision.rationale}")
```

### Running Examples

```bash
python run_examples.py
```

This launches an interactive menu with five demonstration examples:
1. Single stock analysis
2. Batch processing multiple stocks
3. Building complete dataset (20 stocks)
4. Custom analysis configuration
5. Architecture comparison framework

### Building Datasets

```python
from data_pipeline import DatasetBuilder

# Define stocks
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Build dataset
builder = DatasetBuilder(
    tickers=tickers,
    start_date='2023-01-01',
    end_date='2023-12-31'
)

dataset = builder.build_dataset('market_data_2023.csv')
```

### Running Evaluation

```python
from evaluation import run_backtesting_evaluation

# Run comprehensive backtesting
evaluator = run_backtesting_evaluation(
    system=system,
    test_data=test_market_data,
    output_dir="evaluation_results"
)

# Generate report
evaluator.generate_report("evaluation_report.txt")
evaluator.plot_performance("performance_plots.png")
```

## Data Sources

The system integrates multiple financial information sources:

1. **Historical Market Data**: Daily OHLC, volume, valuation ratios (P/E, P/B, P/S, PCF)
2. **Fundamental Data**: Company financials, earnings, balance sheets
3. **News and Sentiment**: Daily financial news and corporate announcements

**Primary Data Provider**: Yahoo Finance (via `yfinance`)  
**Coverage**: 20 major U.S. equities for 2023 calendar year

## Evaluation Metrics

### Accuracy Metrics
- Prediction accuracy
- Precision, Recall, F1 Score
- Confusion matrix analysis

### Coherence Metrics
- Agent consensus rate
- Contradiction detection
- Confidence calibration

### Efficiency Metrics
- Average latency per decision
- Total processing time
- Token usage statistics

### Financial Metrics
- Total return
- Sharpe ratio
- Maximum drawdown
- Win rate

## Model Configuration

### Supported Base Models
- LLaMA 3 (8B, 70B)
- Mistral (7B)
- FinGPT (7B)
- Custom fine-tuned models

### Optimization Techniques
- 8-bit quantization (via bitsandbytes)
- LoRA adapters for specialization
- Gradient checkpointing
- Flash Attention 2

### Compute Requirements

| Configuration | GPU Memory | Processing Time |
|--------------|------------|-----------------|
| Quantized 8B | 8-12 GB | ~30s per decision |
| Full 8B | 16-20 GB | ~20s per decision |
| Quantized 70B | 40-48 GB | ~60s per decision |

## Research Questions

This implementation addresses two key research questions:

### 1. Architectural Efficiency
**Question**: Does modularizing LLM components and introducing hierarchical coordination improve reasoning coherence, interpretability, and computational efficiency compared to a monolithic model?

**Evaluation Approach**:
- Compare modular vs. monolithic architectures
- Measure reasoning coherence via consensus rates
- Benchmark computational costs

### 2. Retrieval-Augmented Adaptability
**Question**: Can RAG-enhanced agents produce more accurate, transparent, and adaptive trading decisions under dynamic market conditions?

**Evaluation Approach**:
- Test with/without RAG enhancement
- Measure adaptability to new information
- Evaluate decision transparency

## Key Features

✅ **Modular Design**: Specialized agents for different analysis types  
✅ **Hierarchical Reasoning**: Multi-stage decision-making process  
✅ **RAG Integration**: Dynamic knowledge retrieval and grounding  
✅ **Comprehensive Evaluation**: Multiple metrics for system assessment  
✅ **Real Data Integration**: Yahoo Finance API for live market data  
✅ **Efficient Inference**: Quantization and optimization for GPU efficiency  
✅ **Extensible Framework**: Easy to add new agents or modify reasoning flow

## Limitations and Future Work

### Current Limitations
- Requires GPU for efficient LLM inference
- Limited to publicly available market data
- Simplified technical indicator calculations
- No real-time streaming data support

### Future Enhancements
1. **Real-time Trading Integration**: Connect to brokerage APIs
2. **Advanced Risk Models**: Implement portfolio optimization
3. **Multi-timeframe Analysis**: Support for intraday and long-term strategies
4. **Enhanced RAG**: Use domain-specific embeddings and vector stores
5. **Agent Communication**: Implement inter-agent dialogue for consensus building
6. **Explainable AI**: Add attention visualization and reasoning traceability

## License

This project is for academic research purposes only. Not intended for actual trading or financial advice.

## Contact

For questions or collaboration:
- Zhoutian Xu: zx348@cornell.edu
