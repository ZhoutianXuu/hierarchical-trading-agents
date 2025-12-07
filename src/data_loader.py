"""
Data Loader for Local CSV Dataset
"""

import pandas as pd
import json
import ast
from datetime import datetime
from typing import List, Dict, Optional

try:
    from .multi_agent_trading_system import MarketData
except ImportError:
    # Fallback for direct execution
    from multi_agent_trading_system import MarketData


class LocalDatasetLoader:
    """Load and process data from local CSV file"""
    
    def __init__(self, csv_path: str):
        """
        Initialize with path to CSV file
        
        Args:
            csv_path: Path to the CSV file (e.g., 'data/sub.csv')
        """
        self.csv_path = csv_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load the CSV file into a DataFrame"""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"✓ Loaded {len(self.df)} records from {self.csv_path}")
            print(f"✓ Date range: {self.df['date'].min()} to {self.df['date'].max()}")
            print(f"✓ Unique tickers: {self.df['stock_code'].nunique()}")
            print(f"✓ Columns: {list(self.df.columns)}\n")
        except Exception as e:
            print(f"Error loading CSV: {e}")
            raise
    
    def parse_news(self, news_str: str) -> List[Dict]:
        """Parse the news field from string to list of dictionaries"""
        if pd.isna(news_str) or news_str == '':
            return []
        
        try:
            # Try to parse as JSON/Python literal
            news_list = ast.literal_eval(news_str)
            if isinstance(news_list, list):
                return news_list
        except:
            pass
        
        # If parsing fails, return empty list
        return []
    
    def get_market_data(self, ticker: str, date: str) -> Optional[MarketData]:
        """
        Get MarketData object for a specific ticker and date
        
        Args:
            ticker: Stock ticker (e.g., 'AAPL.O')
            date: Date string in 'YYYY-MM-DD' format
            
        Returns:
            MarketData object or None if not found
        """
        # Filter for the specific ticker and date
        mask = (self.df['stock_code'] == ticker) & (self.df['date'] == date)
        row = self.df[mask]
        
        if row.empty:
            print(f"No data found for {ticker} on {date}")
            return None
        
        row = row.iloc[0]
        
        # Parse news
        news_list = self.parse_news(row['news'])
        
        # Create OHLC dictionary
        ohlc = {
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close'])
        }
        
        # Create valuation ratios dictionary
        valuation_ratios = {
            'pe': float(row['pe']) if pd.notna(row['pe']) else None,
            'pe_ttm': float(row['pe_ttm']) if pd.notna(row['pe_ttm']) else None,
            'pb': float(row['pb']) if pd.notna(row['pb']) else None,
            'ps': float(row['ps']) if pd.notna(row['ps']) else None,
            'ps_ttm': float(row['ps_ttm']) if pd.notna(row['ps_ttm']) else None,
            'pcf_ttm': float(row['pcf_ttm']) if pd.notna(row['pcf_ttm']) else None,
        }
        
        # Extract fundamentals
        fundamentals = {
            'pe_ratio': valuation_ratios['pe'],
            'price_to_book': valuation_ratios['pb'],
            'price_to_sales': valuation_ratios['ps']
        }
        
        # Convert date string to datetime
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        
        # Extract news titles for simplified view
        news_titles = []
        for news_item in news_list:
            if isinstance(news_item, dict):
                title = news_item.get('title', '')
                if title:
                    news_titles.append(title)
        
        return MarketData(
            ticker=ticker,
            date=date_obj,
            ohlc=ohlc,
            volume=float(row['volume']),
            valuation_ratios=valuation_ratios,
            fundamentals=fundamentals,
            news=news_titles,
            prev_close=float(row['prev_close']) if pd.notna(row['prev_close']) else None
        )
    
    def get_ticker_history(self, ticker: str, start_date: str = None, end_date: str = None) -> List[MarketData]:
        """
        Get historical data for a ticker
        
        Args:
            ticker: Stock ticker (e.g., 'AAPL.O')
            start_date: Optional start date in 'YYYY-MM-DD' format
            end_date: Optional end date in 'YYYY-MM-DD' format
            
        Returns:
            List of MarketData objects
        """
        # Filter by ticker
        ticker_df = self.df[self.df['stock_code'] == ticker].copy()
        
        # Filter by date range if provided
        if start_date:
            ticker_df = ticker_df[ticker_df['date'] >= start_date]
        if end_date:
            ticker_df = ticker_df[ticker_df['date'] <= end_date]
        
        # Sort by date
        ticker_df = ticker_df.sort_values('date')
        
        # Convert each row to MarketData
        market_data_list = []
        for _, row in ticker_df.iterrows():
            market_data = self.get_market_data(ticker, row['date'])
            if market_data:
                market_data_list.append(market_data)
        
        return market_data_list
    
    def get_available_tickers(self) -> List[str]:
        """Get list of all available tickers in the dataset"""
        return sorted(self.df['stock_code'].unique().tolist())
    
    def get_date_range(self, ticker: str = None) -> tuple:
        """
        Get the date range available in the dataset
        
        Args:
            ticker: Optional ticker to get date range for specific stock
            
        Returns:
            Tuple of (min_date, max_date) as strings
        """
        if ticker:
            ticker_df = self.df[self.df['stock_code'] == ticker]
            return (ticker_df['date'].min(), ticker_df['date'].max())
        else:
            return (self.df['date'].min(), self.df['date'].max())
    
    def get_summary_stats(self, ticker: str = None) -> Dict:
        """
        Get summary statistics
        
        Args:
            ticker: Optional ticker to get stats for specific stock
            
        Returns:
            Dictionary with summary statistics
        """
        df = self.df if ticker is None else self.df[self.df['stock_code'] == ticker]
        
        return {
            'total_records': len(df),
            'unique_tickers': df['stock_code'].nunique() if ticker is None else 1,
            'date_range': (df['date'].min(), df['date'].max()),
            'avg_volume': df['volume'].mean(),
            'avg_close_price': df['close'].mean(),
            'price_range': (df['close'].min(), df['close'].max())
        }
    
    def create_train_test_split(
        self,
        ticker: str,
        test_size: float = 0.2
    ) -> tuple:
        """
        Split data into train and test sets chronologically
        
        Args:
            ticker: Stock ticker
            test_size: Proportion of data for testing (e.g., 0.2 = 20%)
            
        Returns:
            Tuple of (train_data, test_data) as lists of MarketData objects
        """
        history = self.get_ticker_history(ticker)
        
        split_idx = int(len(history) * (1 - test_size))
        train_data = history[:split_idx]
        test_data = history[split_idx:]
        
        print(f"Split {ticker} data:")
        print(f"  Training: {len(train_data)} days")
        print(f"  Testing: {len(test_data)} days")
        
        return train_data, test_data


def demo_usage():
    """Demonstrate how to use the LocalDatasetLoader"""
    
    # Load the dataset
    # Updated path - assumes CSV is in data/ directory
    loader = LocalDatasetLoader('data/sub.csv')
    
    # Get available tickers
    tickers = loader.get_available_tickers()
    print(f"\nAvailable tickers: {tickers[:5]}... (showing first 5)")
    
    # Get summary stats
    stats = loader.get_summary_stats()
    print(f"\nDataset Summary:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Get data for a specific ticker and date
    ticker = tickers[0] if tickers else 'AAPL.O'
    date = '2023-01-05'
    
    print(f"\n{'='*60}")
    print(f"Example: Loading {ticker} on {date}")
    print(f"{'='*60}")
    
    market_data = loader.get_market_data(ticker, date)
    
    if market_data:
        print(f"\n✓ MarketData loaded successfully:")
        print(f"  Ticker: {market_data.ticker}")
        print(f"  Date: {market_data.date}")
        print(f"  Close: ${market_data.ohlc['close']:.2f}")
        print(f"  Volume: {market_data.volume:,.0f}")
        print(f"  P/E Ratio: {market_data.valuation_ratios.get('pe', 'N/A')}")
        print(f"  News items: {len(market_data.news)}")
        
        if market_data.news:
            print(f"\n  Sample news:")
            for i, news in enumerate(market_data.news[:2], 1):
                print(f"    {i}. {news[:80]}...")
    
    # Get historical data
    print(f"\n{'='*60}")
    print(f"Loading historical data for {ticker}")
    print(f"{'='*60}")
    
    history = loader.get_ticker_history(ticker, start_date='2023-01-03', end_date='2023-01-10')
    print(f"\n✓ Loaded {len(history)} days of historical data")
    
    if history:
        print(f"\nPrice progression:")
        for data in history[:5]:
            print(f"  {data.date.strftime('%Y-%m-%d')}: ${data.ohlc['close']:.2f}")
    
    # Create train/test split
    print(f"\n{'='*60}")
    print("Creating train/test split")
    print(f"{'='*60}")
    
    train_data, test_data = loader.create_train_test_split(ticker, test_size=0.2)


if __name__ == "__main__":
    demo_usage()