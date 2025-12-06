"""
Example Runner for Ollama-based Multi-Agent Trading System
Uses local CSV data and local Qwen3:4b model
"""

from multi_agent_trading_system import HierarchicalMultiAgentSystem
from data_loader import LocalDatasetLoader
from datetime import datetime
import time


def example_1_single_analysis():
    """Example 1: Analyze a single stock on a specific date"""
    
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Stock Analysis with Ollama")
    print("="*70 + "\n")
    
    # Load dataset
    print("Loading dataset...")
    loader = LocalDatasetLoader('sub.csv')
    
    # Get available tickers
    tickers = loader.get_available_tickers()
    print(f"Available tickers: {tickers}\n")
    
    # Choose first ticker and a date
    ticker = tickers[0] if tickers else 'AAPL.O'
    date = '2023-01-05'
    
    # Load market data
    print(f"Loading data for {ticker} on {date}...")
    market_data = loader.get_market_data(ticker, date)
    
    if not market_data:
        print("No data available for this ticker/date")
        return
    
    print(f"✓ Data loaded: Close=${market_data.ohlc['close']:.2f}, Volume={market_data.volume:,.0f}\n")
    
    # Initialize system (RAG disabled by default)
    print("Initializing Multi-Agent System...")
    system = HierarchicalMultiAgentSystem(
        model="qwen3:4b",
        base_url="http://localhost:11434",
        use_rag=False  # RAG infrastructure kept but not used
    )
    
    # Process decision
    print("\nProcessing trading decision...\n")
    start_time = time.time()
    
    decision = system.process_trading_decision(market_data)
    
    elapsed = time.time() - start_time
    
    # Display results
    print("\n" + "="*70)
    print("TRADING DECISION RESULTS")
    print("="*70)
    print(f"Ticker: {decision.ticker}")
    print(f"Action: {decision.action}")
    print(f"Confidence: {decision.confidence:.1%}")
    print(f"Risk Level: {decision.risk_assessment['level']}")
    print(f"Supporting Agents: {', '.join(decision.supporting_agents)}")
    print(f"Processing Time: {elapsed:.2f}s")
    print(f"\nRationale:")
    print(f"{decision.rationale}")
    print("="*70)


def example_2_batch_processing():
    """Example 2: Process multiple days for a single ticker"""
    
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Processing (Multiple Days)")
    print("="*70 + "\n")
    
    # Load dataset
    loader = LocalDatasetLoader('/mnt/user-data/uploads/sub.csv')
    
    # Get first ticker
    tickers = loader.get_available_tickers()
    ticker = tickers[0] if tickers else 'AAPL.O'
    
    # Load 5 days of data
    print(f"Loading 5 days of data for {ticker}...")
    history = loader.get_ticker_history(ticker, start_date='2023-01-03', end_date='2023-01-09')
    
    if len(history) < 1:
        print("No data available")
        return
    
    print(f"✓ Loaded {len(history)} days\n")
    
    # Initialize system
    system = HierarchicalMultiAgentSystem(
        model="qwen3:4b",
        use_rag=False
    )
    
    # Process each day
    results = []
    
    for i, market_data in enumerate(history[:5], 1):  # Limit to 5 days for demo
        print(f"\n{'='*60}")
        print(f"Day {i}/{min(5, len(history))}: {market_data.date.strftime('%Y-%m-%d')}")
        print(f"{'='*60}")
        
        decision = system.process_trading_decision(market_data)
        
        results.append({
            'date': market_data.date,
            'close': market_data.ohlc['close'],
            'action': decision.action,
            'confidence': decision.confidence
        })
        
        print(f"Decision: {decision.action} ({decision.confidence:.0%})")
    
    # Summary
    print("\n" + "="*70)
    print("BATCH PROCESSING SUMMARY")
    print("="*70)
    print(f"{'Date':<12} {'Close':>10} {'Action':>8} {'Confidence':>12}")
    print("-"*70)
    for r in results:
        print(f"{r['date'].strftime('%Y-%m-%d'):<12} ${r['close']:>9.2f} {r['action']:>8} {r['confidence']:>11.0%}")
    print("="*70)


def example_3_compare_tickers():
    """Example 3: Compare multiple tickers on the same date"""
    
    print("\n" + "="*70)
    print("EXAMPLE 3: Compare Multiple Tickers")
    print("="*70 + "\n")
    
    # Load dataset
    loader = LocalDatasetLoader('/mnt/user-data/uploads/sub.csv')
    
    # Get up to 3 tickers
    tickers = loader.get_available_tickers()[:3]
    date = '2023-01-05'
    
    print(f"Comparing {len(tickers)} tickers on {date}:")
    print(f"Tickers: {', '.join(tickers)}\n")
    
    # Initialize system
    system = HierarchicalMultiAgentSystem(
        model="qwen3:4b",
        use_rag=False
    )
    
    # Process each ticker
    results = []
    
    for ticker in tickers:
        print(f"\n{'='*60}")
        print(f"Analyzing: {ticker}")
        print(f"{'='*60}")
        
        market_data = loader.get_market_data(ticker, date)
        
        if not market_data:
            print(f"No data for {ticker}")
            continue
        
        decision = system.process_trading_decision(market_data)
        
        results.append({
            'ticker': ticker,
            'close': market_data.ohlc['close'],
            'action': decision.action,
            'confidence': decision.confidence,
            'risk': decision.risk_assessment['level']
        })
    
    # Comparison table
    print("\n" + "="*70)
    print("TICKER COMPARISON")
    print("="*70)
    print(f"{'Ticker':<12} {'Close':>10} {'Action':>8} {'Confidence':>12} {'Risk':>10}")
    print("-"*70)
    for r in results:
        print(f"{r['ticker']:<12} ${r['close']:>9.2f} {r['action']:>8} {r['confidence']:>11.0%} {r['risk']:>10}")
    print("="*70)


def example_4_data_exploration():
    """Example 4: Explore the dataset"""
    
    print("\n" + "="*70)
    print("EXAMPLE 4: Dataset Exploration")
    print("="*70 + "\n")
    
    # Load dataset
    loader = LocalDatasetLoader('/mnt/user-data/uploads/sub.csv')
    
    # Get overall stats
    stats = loader.get_summary_stats()
    
    print("Dataset Statistics:")
    print("-"*70)
    print(f"Total Records: {stats['total_records']:,}")
    print(f"Unique Tickers: {stats['unique_tickers']}")
    print(f"Date Range: {stats['date_range'][0]} to {stats['date_range'][1]}")
    print(f"Average Volume: {stats['avg_volume']:,.0f}")
    print(f"Average Close Price: ${stats['avg_close_price']:.2f}")
    print(f"Price Range: ${stats['price_range'][0]:.2f} - ${stats['price_range'][1]:.2f}")
    
    # List all tickers
    tickers = loader.get_available_tickers()
    print(f"\nAvailable Tickers ({len(tickers)}):")
    print(", ".join(tickers))
    
    # Sample data for first ticker
    if tickers:
        ticker = tickers[0]
        print(f"\n{'='*70}")
        print(f"Sample Data for {ticker}")
        print(f"{'='*70}")
        
        ticker_stats = loader.get_summary_stats(ticker)
        print(f"Records: {ticker_stats['total_records']}")
        print(f"Date Range: {ticker_stats['date_range'][0]} to {ticker_stats['date_range'][1]}")
        
        # Get a few days
        history = loader.get_ticker_history(ticker)[:5]
        print(f"\nFirst 5 Days:")
        print(f"{'Date':<12} {'Close':>10} {'Volume':>15} {'P/E':>8}")
        print("-"*50)
        for data in history:
            pe = data.valuation_ratios.get('pe', 'N/A')
            pe_str = f"{pe:.2f}" if isinstance(pe, (int, float)) else str(pe)
            print(f"{data.date.strftime('%Y-%m-%d'):<12} ${data.ohlc['close']:>9.2f} {data.volume:>14,.0f} {pe_str:>8}")


def example_5_train_test_split():
    """Example 5: Create train/test split for evaluation"""
    
    print("\n" + "="*70)
    print("EXAMPLE 5: Train/Test Split for Backtesting")
    print("="*70 + "\n")
    
    # Load dataset
    loader = LocalDatasetLoader('/mnt/user-data/uploads/sub.csv')
    
    # Get first ticker
    tickers = loader.get_available_tickers()
    ticker = tickers[0] if tickers else 'AAPL.O'
    
    print(f"Creating train/test split for {ticker}")
    print(f"Test size: 20% of data\n")
    
    # Create split
    train_data, test_data = loader.create_train_test_split(ticker, test_size=0.2)
    
    print(f"\nTrain set date range:")
    print(f"  Start: {train_data[0].date.strftime('%Y-%m-%d')}")
    print(f"  End: {train_data[-1].date.strftime('%Y-%m-%d')}")
    
    print(f"\nTest set date range:")
    print(f"  Start: {test_data[0].date.strftime('%Y-%m-%d')}")
    print(f"  End: {test_data[-1].date.strftime('%Y-%m-%d')}")
    
    print(f"\n{'='*70}")
    print("Ready for backtesting evaluation!")
    print("You can now:")
    print("  1. Train/tune on train_data")
    print("  2. Evaluate on test_data")
    print("  3. Calculate performance metrics")
    print("="*70)


def main():
    """Main menu"""
    
    while True:
        print("\n" + "="*70)
        print("OLLAMA MULTI-AGENT TRADING SYSTEM")
        print("Using Qwen3:4b + Local Dataset")
        print("="*70)
        print("\nAvailable Examples:")
        print("  1. Single Stock Analysis")
        print("  2. Batch Processing (Multiple Days)")
        print("  3. Compare Multiple Tickers")
        print("  4. Dataset Exploration")
        print("  5. Train/Test Split Demo")
        print("  0. Exit")
        print("-"*70)
        
        try:
            choice = input("\nSelect an example (0-5): ").strip()
            
            if choice == '0':
                print("\nExiting. Thank you!")
                break
            elif choice == '1':
                example_1_single_analysis()
            elif choice == '2':
                example_2_batch_processing()
            elif choice == '3':
                example_3_compare_tickers()
            elif choice == '4':
                example_4_data_exploration()
            elif choice == '5':
                example_5_train_test_split()
            else:
                print("\nInvalid choice. Please select 0-5.")
            
            input("\nPress Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n\nExiting. Thank you!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║  HIERARCHICAL MULTI-AGENT LLM TRADING SYSTEM                        ║
    ║  Using Local Ollama Qwen3:4b                                        ║
    ║                                                                      ║
    ║  Team 9: Zhoutian Xu, Raymond Tao, Jiashuo Xu                       ║
    ║  Course: SYSEN 5530 - Fall 2025                                     ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\nIMPORTANT: Make sure Ollama is running!")
    print("Start Ollama with: ollama serve")
    print("Pull model with: ollama pull qwen3:4b\n")
    
    response = input("Ready to proceed? (y/n): ").strip().lower()
    
    if response == 'y':
        main()
    else:
        print("\nSetup Instructions:")
        print("1. Install Ollama: https://ollama.ai")
        print("2. Start Ollama: ollama serve")
        print("3. Pull model: ollama pull qwen3:4b")
        print("4. Run this script again")