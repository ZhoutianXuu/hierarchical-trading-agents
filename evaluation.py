"""
Evaluation Module for Multi-Agent Trading System
Measures reasoning coherence, signal accuracy, and computational efficiency
"""

import time
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Container for system performance metrics"""
    # Accuracy metrics
    correct_predictions: int
    total_predictions: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Coherence metrics
    avg_confidence: float
    consensus_rate: float
    contradiction_rate: float
    
    # Efficiency metrics
    avg_latency: float
    total_time: float
    tokens_used: int
    
    # Financial metrics
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float


class TradingSimulator:
    """Simulates trading based on agent decisions and calculates returns"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}  # ticker -> (shares, avg_price)
        self.trade_history = []
        self.portfolio_values = []
        
    def execute_trade(
        self,
        decision,
        current_price: float,
        date: datetime
    ):
        """Execute a trading decision"""
        ticker = decision.ticker
        action = decision.action
        
        trade_size = self.capital * 0.1  # Use 10% of capital per trade
        
        if action == "BUY":
            shares = trade_size / current_price
            
            if ticker in self.positions:
                old_shares, old_price = self.positions[ticker]
                new_shares = old_shares + shares
                new_avg_price = ((old_shares * old_price) + (shares * current_price)) / new_shares
                self.positions[ticker] = (new_shares, new_avg_price)
            else:
                self.positions[ticker] = (shares, current_price)
            
            self.capital -= trade_size
            
            self.trade_history.append({
                'date': date,
                'ticker': ticker,
                'action': 'BUY',
                'shares': shares,
                'price': current_price,
                'value': trade_size,
                'confidence': decision.confidence
            })
            
        elif action == "SELL" and ticker in self.positions:
            shares, avg_price = self.positions[ticker]
            sale_value = shares * current_price
            profit = (current_price - avg_price) * shares
            
            self.capital += sale_value
            del self.positions[ticker]
            
            self.trade_history.append({
                'date': date,
                'ticker': ticker,
                'action': 'SELL',
                'shares': shares,
                'price': current_price,
                'value': sale_value,
                'profit': profit,
                'confidence': decision.confidence
            })
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio value"""
        position_value = sum(
            shares * current_prices.get(ticker, avg_price)
            for ticker, (shares, avg_price) in self.positions.items()
        )
        return self.capital + position_value
    
    def calculate_metrics(self) -> Dict:
        """Calculate trading performance metrics"""
        if not self.portfolio_values:
            return {}
        
        values = np.array(self.portfolio_values)
        returns = np.diff(values) / values[:-1]
        
        total_return = (values[-1] - values[0]) / values[0]
        
        # Sharpe ratio (annualized, assuming daily returns)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        cumulative = np.maximum.accumulate(values)
        drawdown = (cumulative - values) / cumulative
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        # Win rate
        trades_df = pd.DataFrame(self.trade_history)
        if 'profit' in trades_df.columns:
            winning_trades = len(trades_df[trades_df['profit'] > 0])
            total_trades = len(trades_df[trades_df['profit'].notna()])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        else:
            win_rate = 0.0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(self.trade_history),
            'final_value': values[-1]
        }


class CoherenceEvaluator:
    """Evaluates reasoning coherence and consistency"""
    
    @staticmethod
    def evaluate_agent_consensus(agent_outputs: List) -> float:
        """
        Measure consensus among agents
        
        Returns value between 0 and 1, where 1 means perfect agreement
        """
        if len(agent_outputs) < 2:
            return 1.0
        
        # Extract signals from each agent
        all_signals = [set(output.key_signals) for output in agent_outputs]
        
        # Calculate Jaccard similarity between all pairs
        similarities = []
        for i in range(len(all_signals)):
            for j in range(i + 1, len(all_signals)):
                intersection = len(all_signals[i] & all_signals[j])
                union = len(all_signals[i] | all_signals[j])
                if union > 0:
                    similarities.append(intersection / union)
        
        return np.mean(similarities) if similarities else 0.0
    
    @staticmethod
    def detect_contradictions(agent_outputs: List) -> Tuple[float, List[str]]:
        """
        Detect contradictory signals among agents
        
        Returns contradiction rate and list of contradictions
        """
        contradictions = []
        
        # Define contradictory signal pairs
        contradiction_pairs = [
            ('BULLISH', 'BEARISH'),
            ('POSITIVE_SENTIMENT', 'NEGATIVE_SENTIMENT'),
            ('UNDERVALUED', 'OVERVALUED'),
            ('LOW_RISK', 'HIGH_RISK'),
        ]
        
        # Collect all signals
        all_signals = []
        for output in agent_outputs:
            all_signals.extend(output.key_signals)
        
        # Check for contradictions
        for signal1, signal2 in contradiction_pairs:
            if signal1 in all_signals and signal2 in all_signals:
                contradictions.append(f"{signal1} vs {signal2}")
        
        contradiction_rate = len(contradictions) / len(contradiction_pairs)
        return contradiction_rate, contradictions
    
    @staticmethod
    def evaluate_confidence_calibration(
        decisions: List,
        outcomes: List[bool]
    ) -> Dict:
        """
        Evaluate how well confidence scores match actual outcomes
        
        Args:
            decisions: List of TradingDecision objects
            outcomes: List of boolean values indicating if decision was correct
        """
        if len(decisions) != len(outcomes):
            raise ValueError("Decisions and outcomes must have same length")
        
        # Group by confidence bins
        bins = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
        bin_accuracies = {f"{bins[i]:.1f}-{bins[i+1]:.1f}": [] 
                         for i in range(len(bins)-1)}
        
        for decision, outcome in zip(decisions, outcomes):
            conf = decision.confidence
            for i in range(len(bins)-1):
                if bins[i] <= conf < bins[i+1]:
                    bin_accuracies[f"{bins[i]:.1f}-{bins[i+1]:.1f}"].append(outcome)
                    break
        
        # Calculate accuracy per bin
        calibration = {}
        for bin_range, outcomes_list in bin_accuracies.items():
            if outcomes_list:
                calibration[bin_range] = sum(outcomes_list) / len(outcomes_list)
            else:
                calibration[bin_range] = None
        
        return calibration


class EfficiencyEvaluator:
    """Evaluates computational efficiency"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.operation_times = []
        
    def start_timer(self):
        """Start timing an operation"""
        self.start_time = time.time()
        
    def stop_timer(self) -> float:
        """Stop timer and record operation time"""
        if self.start_time is None:
            return 0.0
        
        elapsed = time.time() - self.start_time
        self.operation_times.append(elapsed)
        self.start_time = None
        return elapsed
    
    def get_statistics(self) -> Dict:
        """Get timing statistics"""
        if not self.operation_times:
            return {}
        
        return {
            'total_time': sum(self.operation_times),
            'avg_time': np.mean(self.operation_times),
            'median_time': np.median(self.operation_times),
            'min_time': np.min(self.operation_times),
            'max_time': np.max(self.operation_times),
            'std_time': np.std(self.operation_times)
        }


class SystemEvaluator:
    """Main evaluator for the multi-agent system"""
    
    def __init__(self):
        self.simulator = TradingSimulator()
        self.coherence_evaluator = CoherenceEvaluator()
        self.efficiency_evaluator = EfficiencyEvaluator()
        
        self.all_decisions = []
        self.all_agent_outputs = []
        self.ground_truth = []
        
    def evaluate_decision(
        self,
        decision,
        agent_outputs: List,
        actual_outcome: bool,
        execution_time: float
    ):
        """Evaluate a single trading decision"""
        self.all_decisions.append(decision)
        self.all_agent_outputs.append(agent_outputs)
        self.ground_truth.append(actual_outcome)
        self.efficiency_evaluator.operation_times.append(execution_time)
    
    def calculate_final_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        
        # Accuracy metrics
        correct = sum(self.ground_truth)
        total = len(self.ground_truth)
        accuracy = correct / total if total > 0 else 0.0
        
        # Calculate precision, recall, F1
        # Assuming BUY decisions are positive class
        true_positives = sum(
            1 for i, dec in enumerate(self.all_decisions)
            if dec.action == "BUY" and self.ground_truth[i]
        )
        false_positives = sum(
            1 for i, dec in enumerate(self.all_decisions)
            if dec.action == "BUY" and not self.ground_truth[i]
        )
        false_negatives = sum(
            1 for i, dec in enumerate(self.all_decisions)
            if dec.action != "BUY" and self.ground_truth[i]
        )
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Coherence metrics
        avg_confidence = np.mean([d.confidence for d in self.all_decisions])
        
        consensus_rates = [
            self.coherence_evaluator.evaluate_agent_consensus(outputs)
            for outputs in self.all_agent_outputs
        ]
        consensus_rate = np.mean(consensus_rates)
        
        contradiction_rates = [
            self.coherence_evaluator.detect_contradictions(outputs)[0]
            for outputs in self.all_agent_outputs
        ]
        contradiction_rate = np.mean(contradiction_rates)
        
        # Efficiency metrics
        efficiency_stats = self.efficiency_evaluator.get_statistics()
        
        # Financial metrics
        trading_metrics = self.simulator.calculate_metrics()
        
        return PerformanceMetrics(
            correct_predictions=correct,
            total_predictions=total,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            avg_confidence=avg_confidence,
            consensus_rate=consensus_rate,
            contradiction_rate=contradiction_rate,
            avg_latency=efficiency_stats.get('avg_time', 0.0),
            total_time=efficiency_stats.get('total_time', 0.0),
            tokens_used=0,  # Would need to track this separately
            total_return=trading_metrics.get('total_return', 0.0),
            sharpe_ratio=trading_metrics.get('sharpe_ratio', 0.0),
            max_drawdown=trading_metrics.get('max_drawdown', 0.0),
            win_rate=trading_metrics.get('win_rate', 0.0)
        )
    
    def generate_report(self, output_path: str = "evaluation_report.txt"):
        """Generate comprehensive evaluation report"""
        metrics = self.calculate_final_metrics()
        
        report = f"""
{'='*70}
MULTI-AGENT TRADING SYSTEM EVALUATION REPORT
{'='*70}

ACCURACY METRICS
{'-'*70}
Total Predictions:        {metrics.total_predictions}
Correct Predictions:      {metrics.correct_predictions}
Accuracy:                 {metrics.accuracy:.2%}
Precision:                {metrics.precision:.2%}
Recall:                   {metrics.recall:.2%}
F1 Score:                 {metrics.f1_score:.3f}

REASONING COHERENCE
{'-'*70}
Average Confidence:       {metrics.avg_confidence:.2%}
Agent Consensus Rate:     {metrics.consensus_rate:.2%}
Contradiction Rate:       {metrics.contradiction_rate:.2%}

COMPUTATIONAL EFFICIENCY
{'-'*70}
Total Processing Time:    {metrics.total_time:.2f} seconds
Average Latency:          {metrics.avg_latency:.2f} seconds
Tokens Used:              {metrics.tokens_used:,}

TRADING PERFORMANCE
{'-'*70}
Total Return:             {metrics.total_return:.2%}
Sharpe Ratio:             {metrics.sharpe_ratio:.3f}
Maximum Drawdown:         {metrics.max_drawdown:.2%}
Win Rate:                 {metrics.win_rate:.2%}

{'='*70}
"""
        
        print(report)
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"\nReport saved to {output_path}")
    
    def plot_performance(self, output_path: str = "performance_plots.png"):
        """Generate performance visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Portfolio value over time
        if self.simulator.portfolio_values:
            axes[0, 0].plot(self.simulator.portfolio_values)
            axes[0, 0].set_title('Portfolio Value Over Time')
            axes[0, 0].set_xlabel('Trading Day')
            axes[0, 0].set_ylabel('Portfolio Value ($)')
            axes[0, 0].grid(True)
        
        # Plot 2: Decision distribution
        actions = [d.action for d in self.all_decisions]
        action_counts = pd.Series(actions).value_counts()
        axes[0, 1].bar(action_counts.index, action_counts.values)
        axes[0, 1].set_title('Trading Decision Distribution')
        axes[0, 1].set_ylabel('Count')
        
        # Plot 3: Confidence vs Accuracy
        confidences = [d.confidence for d in self.all_decisions]
        axes[1, 0].scatter(confidences, self.ground_truth, alpha=0.5)
        axes[1, 0].set_title('Confidence vs Actual Outcome')
        axes[1, 0].set_xlabel('Confidence')
        axes[1, 0].set_ylabel('Correct (1) / Incorrect (0)')
        axes[1, 0].grid(True)
        
        # Plot 4: Processing time distribution
        if self.efficiency_evaluator.operation_times:
            axes[1, 1].hist(self.efficiency_evaluator.operation_times, bins=20)
            axes[1, 1].set_title('Processing Time Distribution')
            axes[1, 1].set_xlabel('Time (seconds)')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Performance plots saved to {output_path}")


def run_backtesting_evaluation(
    system,
    test_data: List,
    output_dir: str = "evaluation_results"
):
    """
    Run comprehensive backtesting evaluation
    
    Args:
        system: HierarchicalMultiAgentSystem instance
        test_data: List of MarketData objects for testing
        output_dir: Directory to save results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    evaluator = SystemEvaluator()
    
    print("Starting backtesting evaluation...")
    print(f"Testing on {len(test_data)} data points\n")
    
    for i, market_data in enumerate(test_data):
        print(f"Processing {i+1}/{len(test_data)}: {market_data.ticker} on {market_data.date}")
        
        # Time the decision process
        start_time = time.time()
        
        # Process decision
        decision = system.process_trading_decision(market_data)
        
        execution_time = time.time() - start_time
        
        # Simulate trading
        current_price = market_data.ohlc['close']
        evaluator.simulator.execute_trade(decision, current_price, market_data.date)
        
        # Record portfolio value (would need next day's prices in real scenario)
        evaluator.simulator.portfolio_values.append(
            evaluator.simulator.get_portfolio_value({market_data.ticker: current_price})
        )
        
        # Evaluate (outcome would be based on future price movement)
        # For demonstration, using a placeholder
        actual_outcome = np.random.choice([True, False])  # Replace with real outcome
        
        # Store evaluation data
        evaluator.evaluate_decision(
            decision,
            [],  # Would pass agent_outputs here
            actual_outcome,
            execution_time
        )
        
        print(f"  Decision: {decision.action} (confidence: {decision.confidence:.2%})")
        print(f"  Execution time: {execution_time:.2f}s\n")
    
    # Generate final report
    print("\n" + "="*70)
    print("GENERATING EVALUATION REPORT")
    print("="*70 + "\n")
    
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    evaluator.generate_report(report_path)
    
    plot_path = os.path.join(output_dir, "performance_plots.png")
    evaluator.plot_performance(plot_path)
    
    return evaluator


if __name__ == "__main__":
    print("Evaluation Module for Multi-Agent Trading System")
    print("This module provides comprehensive evaluation capabilities.")