# Hierarchical Multi-Agent LLM Architecture for Stock Trading

## 📄 Abstract

This repository implements a novel **Hierarchical Multi-Agent LLM Architecture** for stock trading that decomposes financial decision-making into specialized reasoning subspaces. Our system achieves **68.4% directional prediction accuracy** (9.7% improvement over monolithic baselines) and a **Sharpe Ratio of 0.89** on 20 major U.S. equities throughout 2023.

**Key Features:**
- ✅ Modular specialization via LoRA fine-tuning (99.6% parameter reduction)
- ✅ Hierarchical coordination with Chain-of-Thought synthesis
- ✅ Retrieval-Augmented Generation (RAG) for real-time knowledge grounding
- ✅ 85.2% inter-agent consensus rate, 33% latency reduction

---

## 🚀 Quick Start (5 Minutes)

### Prerequisites

- **Python 3.8+**
- **Ollama** (for local LLM inference)
- **8GB+ RAM**

### Step 1: Install Ollama

```bash
# Download from https://ollama.ai and install

# Pull Qwen3:4b model
ollama pull qwen3:4b

# Start Ollama service
ollama serve
```

### Step 2: Setup Repository

```bash
# Clone repository
git clone https://github.com/ZhoutianXuu/hierarchical-trading-agents.git
cd hierarchical-trading-agents

# Install dependencies
pip install -r requirements.txt

# Place your data file
mkdir -p data
cp /path/to/your/data.csv data/example_data.csv
```

### Step 3: Run Example

```bash
cd examples
python run_examples.py
# Select option 1: Single Stock Analysis
```

**Done!** You should see a trading decision in ~10-20 seconds.

---
## Data Availability Notice
We include example_data.csv strictly for demonstration purposes to help users understand the expected data format and workflow of our system. This sample file is intentionally lightweight and synthetic, making it easy to explore the pipeline without requiring any paid data access. In addition, we provide randomly_sampled_data.csv, which offers a small preview of the structure and characteristics of our final dataset.

The full dataset used in our experiments cannot be distributed due to licensing restrictions. Our project relies on data obtained through paid APIs such as EODHD and Yahoo Finance as well as proprietary datasets from brokerage firms, all of which prohibit redistribution. For this reason, only limited example files are included in the repository. Users wishing to reproduce the full results should acquire their own data sources in accordance with the respective providers’ terms of service.

---


## 🧪 Reproducing Paper Results (After Fine-tuning Your LLMs)

### Table 1: Overall Performance (Main Results)

```bash
# Run comprehensive evaluation
python -m src.evaluation_runner

# Or run step-by-step:
cd examples
python run_examples.py
# Select option 2: Batch Processing (Multiple Days)
```

**Expected Results:**

| Method | Accuracy | Sharpe Ratio | Latency | Memory |
|--------|----------|--------------|---------|--------|
| Traditional-ML | 52.3% | 0.42 | 0.1s | 2GB |
| Monolithic-LLM | 58.7% | 0.61 | 6.2s | 16GB |
| **Ours (Full)** | **68.4%** | **0.89** | **4.2s** | **10GB** |

### Table 2: Ablation Studies

**Without Fine-tuning:**
```python
# Edit src/multi_agent_trading_system.py
# Comment out LoRA loading sections
# Expected: 62.1% accuracy (-6.3% drop)
```

**Without RAG:**
```python
system = HierarchicalMultiAgentSystem(use_rag=False)
# Expected: 64.7% accuracy (-3.7% drop)
```

**Without Hierarchical Coordination:**
```python
# Use simple majority voting instead of Reviewer Agent
# Expected: 63.5% accuracy, 42.1% consensus
```

### Table 3: Per-Agent Performance

```bash
# Test individual agents
cd examples
python run_examples.py
# Select option 1, observe agent outputs
```

**Expected per-agent accuracy gains:**
- Fundamental: +5.6% (64.2% → 69.8%)
- Sentiment: +9.2% (58.9% → 68.1%)
- Technical: +4.7% (61.5% → 66.2%)
- Risk: +3.4% (66.7% → 70.1%)

---

## 📁 Repository Structure

```
hierarchical-trading-agents/
├── README.md                          ← This file
├── requirements.txt                   ← Dependencies
├── LICENSE                            ← MIT License
│
├── src/                               ← Source code
│   ├── multi_agent_trading_system.py  ← Main system
│   ├── data_loader.py                 ← Data loading
│   ├── evaluation.py                  ← Metrics & backtesting
│   ├── rag/                           ← RAG components
│   └── training/                      ← LoRA training
│
├── examples/                          ← Demo scripts
│   ├── run_examples.py                ← Main demo (5 examples)
│
├── scripts/                           ← Utility scripts
│   ├── train_lora.py                  ← Fine-tune agents
│   └── build_rag_index.py             ← Build vector store
│
├── configs/                           ← Configuration
│   ├── lora_config.yaml
│   └── rag_config.yaml
│
└── data/                              ← Dataset
    ├── example_data.csv               ← Example dataset for demo script
    ├── randomly_sampled_data.csv      ← Randomly selected data from our final dataset for demonstration
    ├── news_sentiment_data.ipynb      ← News data extraction script
    ├── valuation_data.ipynb           ← Technical data extraction script e.g.(P/E)
    ├── yahoo_data.ipynb               ← OHLC data extraction script e.g.(open, high, low and close)
    ├── training/                      ← Fine-tuning data
    └── documents/                     ← RAG corpus
```

---

## 📊 System Architecture

### Layer 1: Specialized Agents

Four domain-specific agents process distinct data subsets:

1. **Fundamental Agent:** Valuation ratios (P/E, P/B, P/S, P/CF)
2. **Sentiment Agent:** News articles, analyst commentary
3. **Technical Agent:** OHLCV data, price patterns
4. **Risk Agent:** Volatility, beta, correlation

```python
from src.multi_agent_trading_system import (
    FundamentalAgent,
    SentimentAgent,
    TechnicalAgent,
    RiskManagementAgent
)
```

### Layer 2: Hierarchical Coordination

Two meta-agents synthesize Layer 1 outputs:

5. **Reviewer Agent:** Chain-of-Thought synthesis, conflict resolution
6. **Decision Agent:** Final trading action (BUY/SELL/HOLD)

### Layer 3: RAG Knowledge Base (Optional)

Dynamic retrieval from:
- SEC filings
- Earnings call transcripts
- News feeds

---

## 🔧 Configuration

### Core Parameters

Edit in `src/multi_agent_trading_system.py` or via arguments:

```python
system = HierarchicalMultiAgentSystem(
    model="qwen3:4b",              # Base LLM
    base_url="http://localhost:11434",  # Ollama endpoint
    use_rag=False,                 # Enable RAG retrieval
    no_think=True                  # Disable reasoning output
)
```

### LoRA Training Parameters

From paper (Section 3.2.2):

```yaml
lora:
  rank: 8                   # r = 8
  alpha: 16                 # α = 16
  dropout: 0.05
  target_modules: ["q_proj", "v_proj"]

training:
  learning_rate: 3e-4
  num_epochs: 3
  batch_size: 4
  optimizer: "AdamW"
```

---

## 📈 Data Format

### Required CSV Columns

Your `data/sub.csv` must include:

```
stock_code    : Ticker symbol (e.g., 'AAPL.O')
date          : YYYY-MM-DD format
open          : Opening price
high          : Daily high
low           : Daily low
close         : Closing price
volume        : Trading volume
prev_close    : Previous day's close
pe, pb, ps    : Valuation ratios
pe_ttm, ps_ttm, pcf_ttm : Trailing twelve month ratios
news          : JSON array of news articles
```

### News Field Format

```json
[
  {
    "title": "Apple Reports Strong Q1 Earnings",
    "published": "2023-01-05T16:30:00-05:00",
    "body": "Full article text..."
  }
]
```

### Example Data Loading

```python
from src.data_loader import LocalDatasetLoader

loader = LocalDatasetLoader('data/sub.csv')
market_data = loader.get_market_data('AAPL.O', '2023-01-05')
```

---

## 🎯 Reproducing Core Experiments

### Experiment 1: Single Stock Decision

**Objective:** Generate one trading decision with full agent reasoning  
**Runtime:** ~10-20 seconds  
**Paper Reference:** Section 5.1

```bash
cd examples
python run_examples.py
# Select: 1. Single Stock Analysis
```

**Expected Output:**
```
DECISION: BUY
Confidence: 73.5%
Risk Level: MEDIUM
Supporting Agents: Fundamental_Agent, Sentiment_Agent, Risk_Management_Agent
Processing Time: 14.2s

Rationale: 
Strong fundamental valuation (P/E below sector average), positive 
sentiment from recent earnings beat, moderate technical momentum. 
Risk assessment indicates acceptable volatility profile.
```

### Experiment 2: Batch Processing (252 Days)

**Objective:** Process full 2023 trading year  
**Runtime:** ~15-20 minutes for 20 stocks  
**Paper Reference:** Table 1, Figure 2

```bash
cd examples
python run_examples.py
# Select: 2. Batch Processing (Multiple Days)
```

**Generates:**
- Day-by-day decisions
- Accuracy metrics
- Sharpe ratio calculation
- Cumulative returns

### Experiment 3: Multi-Ticker Comparison

**Objective:** Compare decisions across stocks on same date  
**Runtime:** ~1-2 minutes  
**Paper Reference:** Section 5.4

```bash
cd examples
python run_examples.py
# Select: 3. Compare Multiple Tickers
```

### Experiment 4: Full Backtesting

**Objective:** Reproduce Table 1 results  
**Runtime:** ~1-2 hours for full evaluation  
**Paper Reference:** Section 5

```python
from src.evaluation import run_backtesting_evaluation
from src.multi_agent_trading_system import HierarchicalMultiAgentSystem
from src.data_loader import LocalDatasetLoader

# Load all test data
loader = LocalDatasetLoader('data/sub.csv')
test_data = []
for ticker in ['AAPL.O', 'MSFT.O', 'GOOGL.O']:  # Add all 20 tickers
    history = loader.get_ticker_history(ticker, '2023-01-01', '2023-12-31')
    test_data.extend(history)

# Initialize system
system = HierarchicalMultiAgentSystem(model='qwen3:4b', no_think=True)

# Run evaluation
evaluator = run_backtesting_evaluation(
    system=system,
    test_data=test_data,
    output_dir='evaluation_results'
)
```

---

## 🧬 Advanced Features

### LoRA Fine-tuning

**Note:** Paper reports results with fine-tuned agents. Our submission uses base Qwen3:4b.

To fine-tune your own agents:

```bash
# Prepare training data
python scripts/prepare_data.py --input data/your_data_path.csv --output data/training/

# Fine-tune all agents (requires GPU)
python scripts/train_all_agents.py --data-dir data/training/ --epochs 3

# Or fine-tune individual agent
python scripts/train_lora.py \
    --agent fundamental \
    --data data/training/fundamental.jsonl \
    --epochs 3 \
    --batch-size 4
```

**Requirements:**
- NVIDIA A100 (40GB+) or 4× RTX 3090
- PyTorch with CUDA
- ~8 GPU-hours training time

### RAG Setup

To enable document retrieval (optional):

```bash
# Install RAG dependencies
pip install langchain-community sentence-transformers faiss-cpu

# Build FAISS index
python scripts/build_rag_index.py \
    --documents data/documents/ \
    --output rag_store/faiss_index

# Enable in system
system = HierarchicalMultiAgentSystem(use_rag=True)
```

---

## ⚙️ Hardware Requirements

### Minimum (CPU-only, slow)

- 8-core CPU
- 8GB RAM
- 10GB disk
- ~30s per decision

### Recommended (GPU)

- NVIDIA RTX 3090/4090 (24GB VRAM)
- 16GB RAM
- 50GB disk
- ~4s per decision

### Optimal (Paper Results)

- NVIDIA A100 (80GB)
- 32GB RAM
- 100GB disk
- ~4.2s per decision

---

## 🐛 Troubleshooting

### "Cannot connect to Ollama"

```bash
# Start Ollama in background
ollama serve &

# Verify
curl http://localhost:11434/api/tags
```

### "Model not found"

```bash
ollama pull qwen3:4b
ollama list
```

### "ModuleNotFoundError: No module named 'src'"

```bash
# Run from examples/ directory
cd examples
python run_examples.py

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### "Out of memory"

Reduce batch size or use CPU:
```python
# In src/multi_agent_trading_system.py
# Reduce max_tokens from 512 to 256
```

### "File not found: your_data_path.csv"

```bash
# Ensure data file is in correct location
ls data/your_data_path.csv

# Check path in run_examples.py
# Should be: '../data/sub.csv' or 'data/sub.csv' from root
```

---

## 📊 Expected Performance

### Accuracy Metrics

- **Directional Accuracy:** 68.4% (paper) / 60-65% (base model expected)
- **Consensus Rate:** 85.2% (stable across volatility regimes)
- **Precision:** ~0.72
- **Recall:** ~0.69
- **F1 Score:** ~0.70

### Financial Metrics

- **Sharpe Ratio:** 0.89 (paper) / 0.6-0.7 (base model expected)
- **Max Drawdown:** ~12%
- **Win Rate:** ~62%
- **Annualized Return:** ~25%

### Efficiency Metrics

- **Inference Latency:** 4.2s per ticker-day (GPU)
- **Throughput:** ~850 ticker-days/hour (4× A100)
- **Memory Usage:** 10GB VRAM (parallel agents)

---

## 📝 Citation

```bibtex
@article{xu2025hierarchical,
  title={Hierarchical Multi-Agent LLM Architecture with Modular and 
         Retrieval-Augmented Reasoning for Stock Trading},
  author={Xu, Zhoutian and Tao, Raymond...},
  journal={Cornell University},
  year={2025}
}
```

---

## 📧 Contact
- Zhoutian Xu: zx348@cornell.edu (Corresponding Author)

**Questions?** Open an issue on GitHub or email team lead.

---

## 📄 License

MIT License - See LICENSE file