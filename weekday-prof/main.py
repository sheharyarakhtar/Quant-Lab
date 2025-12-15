import yfinance as yf
import pandas as pd
import calendar
import os
import pickle
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

# Configuration
TICKERS = ['AAPL', 'GOOG', 'MSFT', 'TSLA']  # Add or remove tickers as needed
START_DATE = '2007-01-01'
END_DATE = '2025-11-01'
PCT_THRESHOLD = 0
CACHE_DIR = 'cache'
CONTENTS_DIR = 'contents'

# Create necessary directories
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CONTENTS_DIR, exist_ok=True)


def get_cache_path(ticker):
    """Get cache file path for a ticker."""
    return os.path.join(CACHE_DIR, f'{ticker}_{START_DATE}_{END_DATE}.pkl')


def load_cached_data(ticker):
    """Load cached data if available."""
    cache_path = get_cache_path(ticker)
    if os.path.exists(cache_path):
        print(f"Loading cached data for {ticker}...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None


def save_cached_data(ticker, df):
    """Save data to cache."""
    cache_path = get_cache_path(ticker)
    with open(cache_path, 'wb') as f:
        pickle.dump(df, f)
    print(f"Cached data saved for {ticker}")


def download_ticker_data(ticker):
    """Download or load cached ticker data."""
    cached_df = load_cached_data(ticker)
    if cached_df is not None:
        return cached_df
    
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE, multi_level_index=False)
    save_cached_data(ticker, df)
    return df


def process_ticker_data(df):
    """Process ticker data to calculate weekday statistics."""
    pct_df = df['Close'].pct_change().dropna()
    binary_df = pct_df.apply(lambda x: 1 if x > 0 else -1)
    
    binary_df = binary_df.reset_index()
    binary_df['day_of_week'] = binary_df['Date'].dt.dayofweek
    binary_df['year'] = binary_df['Date'].dt.year
    binary_df['weekday_name'] = binary_df['day_of_week'].apply(lambda x: calendar.day_name[x])
    
    return binary_df


def calculate_weekday_stats(binary_df):
    """Calculate overall weekday statistics."""
    stats = binary_df.groupby('day_of_week').agg({
        'Close': ['sum', 'size']
    }).reset_index()
    stats.columns = ['day_of_week', 'sum', 'size']
    stats['percentage'] = stats['sum'] * 100 / stats['size']
    stats['weekday_name'] = stats['day_of_week'].apply(lambda x: calendar.day_name[x])
    return stats.sort_values('percentage')


def plot_heatmap(ticker, binary_df):
    """Create and save heatmap for a single ticker."""
    stats2 = binary_df.groupby(['year', 'weekday_name']).agg({
        'Close': ['sum', 'size']
    }).reset_index()
    stats2.columns = ['year', 'weekday_name', 'sum', 'size']
    stats2['percentage'] = stats2['sum'] * 100 / stats2['size']
    
    pivot_df = stats2.pivot(index='year', columns='weekday_name', values='percentage')
    
    # Reorder columns to display weekdays chronologically
    ordered_weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    pivot_df = pivot_df[[col for col in ordered_weekdays if col in pivot_df.columns]]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, cmap='coolwarm', fmt=".1f", linewidths=.5, center=50)
    if PCT_THRESHOLD > 0:
        plt.title(f"{ticker} - Win Rate ({PCT_THRESHOLD*100}% Up Days) by Year and Weekday")
    else:
        plt.title(f'{ticker} - Percentage of Up Days by Year and Weekday')
    plt.xlabel('Weekday')
    plt.ylabel('Year')
    plt.tight_layout()
    plt.savefig(os.path.join(CONTENTS_DIR, f'{ticker}_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap for {ticker}")


def plot_weekday_comparison(all_stats):
    """Compare weekday performance across all tickers."""
    plt.figure(figsize=(14, 6))
    
    ordered_weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    x = np.arange(len(ordered_weekdays))
    width = 0.8 / len(TICKERS)
    
    for i, ticker in enumerate(TICKERS):
        stats = all_stats[ticker]
        percentages = [stats[stats['weekday_name'] == day]['percentage'].values[0] 
                      if day in stats['weekday_name'].values else 0 
                      for day in ordered_weekdays]
        plt.bar(x + i * width, percentages, width, label=ticker, alpha=0.8)
    
    plt.xlabel('Weekday', fontsize=12)
    plt.ylabel('% of Up Days', fontsize=12)
    plt.title('Weekday Profitability Comparison Across Tickers', fontsize=14, fontweight='bold')
    plt.xticks(x + width * (len(TICKERS) - 1) / 2, ordered_weekdays)
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CONTENTS_DIR, 'weekday_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved weekday comparison chart")


def plot_ticker_correlation(all_binary_dfs):
    """Plot correlation of weekday patterns between tickers."""
    # Create a matrix of weekday percentages for each ticker
    weekday_data = {}
    ordered_weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    for ticker, binary_df in all_binary_dfs.items():
        stats = calculate_weekday_stats(binary_df)
        weekday_data[ticker] = [stats[stats['weekday_name'] == day]['percentage'].values[0] 
                               if day in stats['weekday_name'].values else 0 
                               for day in ordered_weekdays]
    
    # Create correlation matrix
    df_corr = pd.DataFrame(weekday_data).T
    df_corr.columns = ordered_weekdays
    correlation = df_corr.T.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", 
                linewidths=.5, center=0, vmin=-1, vmax=1)
    plt.title('Correlation of Weekday Patterns Between Tickers', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(CONTENTS_DIR, 'ticker_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved ticker correlation heatmap")


def plot_best_worst_days(all_stats):
    """Identify and visualize best and worst days for each ticker."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    best_days = {}
    worst_days = {}
    
    for ticker in TICKERS:
        stats = all_stats[ticker]
        best_day = stats.loc[stats['percentage'].idxmax()]
        worst_day = stats.loc[stats['percentage'].idxmin()]
        best_days[ticker] = (best_day['weekday_name'], best_day['percentage'])
        worst_days[ticker] = (worst_day['weekday_name'], worst_day['percentage'])
    
    # Best days
    tickers_list = list(best_days.keys())
    best_percentages = [best_days[t][1] for t in tickers_list]
    best_labels = [f"{best_days[t][0]}\n{best_days[t][1]:.1f}%" for t in tickers_list]
    
    bars1 = ax1.bar(tickers_list, best_percentages, color='green', alpha=0.7)
    ax1.set_ylabel('% of Up Days', fontsize=12)
    ax1.set_title('Best Weekday for Each Ticker', fontsize=14, fontweight='bold')
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, label in zip(bars1, best_labels):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                label.split('\n')[0], ha='center', va='bottom', fontsize=10)
    
    # Worst days
    worst_percentages = [worst_days[t][1] for t in tickers_list]
    worst_labels = [f"{worst_days[t][0]}\n{worst_days[t][1]:.1f}%" for t in tickers_list]
    
    bars2 = ax2.bar(tickers_list, worst_percentages, color='red', alpha=0.7)
    ax2.set_ylabel('% of Up Days', fontsize=12)
    ax2.set_title('Worst Weekday for Each Ticker', fontsize=14, fontweight='bold')
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, label in zip(bars2, worst_labels):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                label.split('\n')[0], ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONTENTS_DIR, 'best_worst_days.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved best/worst days comparison")


def plot_volatility_by_weekday(all_binary_dfs):
    """Compare volatility patterns by weekday across tickers."""
    plt.figure(figsize=(14, 6))
    
    ordered_weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    x = np.arange(len(ordered_weekdays))
    width = 0.8 / len(TICKERS)
    
    for i, ticker in enumerate(TICKERS):
        binary_df = all_binary_dfs[ticker]
        # Get original price data to calculate volatility
        df = download_ticker_data(ticker)
        pct_df = df['Close'].pct_change().dropna().reset_index()
        pct_df['day_of_week'] = pct_df['Date'].dt.dayofweek
        pct_df['weekday_name'] = pct_df['day_of_week'].apply(lambda x: calendar.day_name[x])
        
        volatility = pct_df.groupby('weekday_name')['Close'].std() * 100
        volatilities = [volatility[day] if day in volatility.index else 0 
                       for day in ordered_weekdays]
        plt.bar(x + i * width, volatilities, width, label=ticker, alpha=0.8)
    
    plt.xlabel('Weekday', fontsize=12)
    plt.ylabel('Volatility (Std Dev %)', fontsize=12)
    plt.title('Daily Volatility by Weekday Across Tickers', fontsize=14, fontweight='bold')
    plt.xticks(x + width * (len(TICKERS) - 1) / 2, ordered_weekdays)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CONTENTS_DIR, 'volatility_by_weekday.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved volatility comparison chart")


def main():
    """Main execution function."""
    print(f"\n{'='*60}")
    print(f"Weekday Profitability Analysis")
    print(f"Tickers: {', '.join(TICKERS)}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"{'='*60}\n")
    
    all_stats = {}
    all_binary_dfs = {}
    
    # Process each ticker
    for ticker in TICKERS:
        print(f"\n--- Processing {ticker} ---")
        df = download_ticker_data(ticker)
        binary_df = process_ticker_data(df)
        stats = calculate_weekday_stats(binary_df)
        
        all_stats[ticker] = stats
        all_binary_dfs[ticker] = binary_df
        
        print(f"\n{ticker} Weekday Statistics:")
        print(stats[['weekday_name', 'percentage', 'size']].to_string(index=False))
        
        # Create individual heatmap
        plot_heatmap(ticker, binary_df)
    
    # Create comparative analyses
    print(f"\n--- Creating Comparative Analyses ---")
    plot_weekday_comparison(all_stats)
    plot_ticker_correlation(all_binary_dfs)
    plot_best_worst_days(all_stats)
    plot_volatility_by_weekday(all_binary_dfs)
    
    print(f"\n{'='*60}")
    print(f"✓ Analysis complete! All figures saved to '{CONTENTS_DIR}/' folder")
    print(f"✓ Cached data saved to '{CACHE_DIR}/' folder")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
