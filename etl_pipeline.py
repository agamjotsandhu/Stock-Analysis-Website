import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_rsi(data, window=14):
    """
    Calculate the Relative Strength Index (RSI) for a given series.
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def run_pipeline():
    # Define tickers and timeframe
    tickers = ["SPY", "QQQ", "VAS.AX"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    print(f"Downloading data from {start_date.date()} to {end_date.date()}...")
    
    master_df = pd.DataFrame()
    
    for ticker in tickers:
        print(f"Processing {ticker}...")
        # Download historical data
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
        
        if df.empty:
            print(f"Warning: No data found for {ticker}")
            continue
            
        # Handle MultiIndex columns if necessary (yfinance > 0.2.40 returns MultiIndex for multiple tickers or even single ones sometimes)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 1. Clean data: Forward fill missing values
        df = df.ffill()
        
        # 2. Calculate Indicators
        # 50-day Moving Average
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # 14-day RSI
        df['RSI_14'] = calculate_rsi(df['Close'], window=14)
        
        # Add metadata
        df['Ticker'] = ticker
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Keep only relevant columns
        available_cols = [c for c in ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'SMA_50', 'RSI_14'] if c in df.columns]
        df = df[available_cols]
        
        # Append to master dataframe
        master_df = pd.concat([master_df, df], ignore_index=True)
        
    # Output to CSV
    output_file = "master_etf_data.csv"
    master_df.to_csv(output_file, index=False)
    print(f"Successfully saved cleaned data to {output_file}")

if __name__ == "__main__":
    run_pipeline()
