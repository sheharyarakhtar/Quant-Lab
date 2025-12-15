"""
Standalone script to plot overlayed volatility smile (calls + puts on same plot)
Uses log-moneyness to center ATM at 0 for symmetric comparison of left/right tails.

Usage:
    python plot_smile_overlay.py

Or import and use programmatically:
    from plot_smile_overlay import plot_smile_overlay_standalone
    plot_smile_overlay_standalone("AAPL", expiry_index=0)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime


def get_options_data(ticker_symbol):
    """Fetch options data for a given ticker"""
    ticker = yf.Ticker(ticker_symbol)
    options_dates = ticker.options
    
    all_calls = []
    all_puts = []
    
    for expiry_date in options_dates:
        option_chain = ticker.option_chain(expiry_date)
        calls = option_chain.calls.copy()
        puts = option_chain.puts.copy()
        
        calls['expiryDate'] = expiry_date
        puts['expiryDate'] = expiry_date
        calls['type'] = 'C'
        puts['type'] = 'P'
        
        all_calls.append(calls)
        all_puts.append(puts)
    
    calls_df = pd.concat(all_calls, ignore_index=True)
    puts_df = pd.concat(all_puts, ignore_index=True)
    
    return calls_df, puts_df


def clean_options_data(df, min_iv=0.005, max_iv=1.0, require_liquidity=True):
    """Clean options data with basic filters"""
    df = df.copy()
    
    # Remove invalid IVs
    df = df.dropna(subset=['impliedVolatility', 'strike'])
    df = df[(df['impliedVolatility'] > min_iv) & (df['impliedVolatility'] < max_iv)]
    
    # Liquidity filter
    if require_liquidity:
        df['volume'] = df['volume'].fillna(0)
        df['openInterest'] = df['openInterest'].fillna(0)
        df = df[(df['volume'] > 0) | (df['openInterest'] > 0)]
        
        # Remove low-liquidity high-IV combinations (likely bad data)
        low_liq = (df['volume'] <= 1) & (df['openInterest'] <= 5)
        high_iv = df['impliedVolatility'] > 0.5
        df = df[~(low_liq & high_iv)]
    
    return df


def plot_smile_overlay_standalone(ticker_symbol, expiry_index=0, bin_count=60, 
                                  save_path=None, show_plot=True):
    """
    Plot overlayed calls and puts volatility smile using log-moneyness
    
    Args:
        ticker_symbol: Stock ticker (e.g., "AAPL", "META")
        expiry_index: Which expiry to plot (0=nearest, 1=next, etc.)
        bin_count: Number of bins for smoothed median curve
        save_path: Path to save plot (None = auto-generate)
        show_plot: Whether to display plot interactively
    
    Returns:
        dict with analysis results
    """
    print(f"\nFetching options data for {ticker_symbol}...")
    
    # Get current price
    ticker = yf.Ticker(ticker_symbol)
    spot = ticker.history(period="1d")['Close'].iloc[-1]
    print(f"Current Price: ${spot:.2f}")
    
    # Fetch options
    calls_df, puts_df = get_options_data(ticker_symbol)
    print(f"Fetched {len(calls_df)} calls, {len(puts_df)} puts")
    
    # Clean data
    calls_clean = clean_options_data(calls_df)
    puts_clean = clean_options_data(puts_df)
    print(f"After cleaning: {len(calls_clean)} calls, {len(puts_clean)} puts")
    
    # Combine
    df = pd.concat([calls_clean, puts_clean], ignore_index=True)
    
    # Compute moneyness
    df['moneyness'] = df['strike'] / spot
    df['log_moneyness'] = np.log(df['moneyness'])
    
    # Select expiry
    expiries = sorted(df['expiryDate'].unique())
    if expiry_index >= len(expiries):
        expiry_index = 0
    
    plot_expiry = expiries[expiry_index]
    subset = df[df['expiryDate'] == plot_expiry].copy()
    print(f"\nPlotting expiry: {plot_expiry}")
    
    # Separate calls and puts
    calls_subset = subset[subset['type'] == 'C'].copy()
    puts_subset = subset[subset['type'] == 'P'].copy()
    
    # Convert IV to percent
    calls_subset['IV_pct'] = calls_subset['impliedVolatility'] * 100
    puts_subset['IV_pct'] = puts_subset['impliedVolatility'] * 100
    
    # Binning for smoothed median curve
    all_x = np.concatenate([calls_subset['log_moneyness'].values, 
                           puts_subset['log_moneyness'].values])
    xmin, xmax = np.nanpercentile(all_x, [1, 99])
    bins = np.linspace(xmin, xmax, bin_count)
    
    def binned_median(x, y, bins):
        """Compute binned median for smoothing"""
        inds = np.digitize(x, bins)
        x_mid = []
        y_med = []
        for b in range(1, len(bins)):
            sel = inds == b
            if sel.sum() > 0:
                x_mid.append(np.median(x[sel]))
                y_med.append(np.median(y[sel]))
        return np.array(x_mid), np.array(y_med)
    
    cx, cy = binned_median(calls_subset['log_moneyness'].values, 
                          calls_subset['IV_pct'].values, bins)
    px, py = binned_median(puts_subset['log_moneyness'].values, 
                          puts_subset['IV_pct'].values, bins)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Raw data points
    ax.scatter(puts_subset['log_moneyness'], puts_subset['IV_pct'], 
              s=18, alpha=0.45, label='Puts (raw)', marker='x', color='tab:blue')
    ax.scatter(calls_subset['log_moneyness'], calls_subset['IV_pct'], 
              s=18, alpha=0.45, label='Calls (raw)', marker='o', color='tab:orange')
    
    # Smoothed median curves
    if len(cx) > 0:
        ax.plot(cx, cy, '-', color='tab:orange', lw=2.5, 
               label='Calls (binned median)', zorder=10)
    if len(px) > 0:
        ax.plot(px, py, '-', color='tab:blue', lw=2.5, 
               label='Puts (binned median)', zorder=10)
    
    # ATM vertical line
    ax.axvline(0, color='k', linestyle='--', alpha=0.6, linewidth=1.5, label='ATM')
    
    # X-axis: symmetric ticks in moneyness space
    xticks_log = np.linspace(np.min(bins), np.max(bins), 9)
    xticks_labels = [f"{np.exp(x):.2f}×" for x in xticks_log]
    ax.set_xticks(xticks_log)
    ax.set_xticklabels(xticks_labels, rotation=0)
    
    ax.set_xlabel('Moneyness (Strike / Spot), log-scale; 0 = ATM', fontsize=12)
    ax.set_ylabel('Implied Volatility (%)', fontsize=12)
    ax.set_title(f'{ticker_symbol} — Overlayed Calls & Puts Volatility Smile\nExpiry: {plot_expiry}', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.35)
    
    plt.tight_layout()
    
    # Save
    if save_path is None:
        save_path = f'Images/{ticker_symbol}_overlayed_smile_{plot_expiry}.png'
    plt.savefig(save_path, dpi=220, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # Analysis
    results = {
        'ticker': ticker_symbol,
        'spot': spot,
        'expiry': plot_expiry,
        'num_calls': len(calls_subset),
        'num_puts': len(puts_subset)
    }
    
    # Compute skew
    if len(px) > 0 and len(cx) > 0:
        left_tail_iv = np.median(py[px < -0.1]) if np.any(px < -0.1) else np.nan
        right_tail_iv = np.median(cy[cx > 0.1]) if np.any(cx > 0.1) else np.nan
        
        if not np.isnan(left_tail_iv) and not np.isnan(right_tail_iv):
            skew = left_tail_iv - right_tail_iv
            results['left_tail_iv'] = left_tail_iv
            results['right_tail_iv'] = right_tail_iv
            results['skew'] = skew
            
            print(f"\n{'='*60}")
            print(f"Smile Interpretation:")
            print(f"  Left tail (OTM puts): {left_tail_iv:.1f}%")
            print(f"  Right tail (OTM calls): {right_tail_iv:.1f}%")
            print(f"  Skew: {skew:+.1f}%")
            
            if skew > 5:
                interpretation = "Classic downside skew (puts more expensive)"
            elif skew < -5:
                interpretation = "Upside skew (calls more expensive, bullish speculation)"
            else:
                interpretation = "Relatively balanced tail risk"
            
            print(f"  → {interpretation}")
            print(f"{'='*60}\n")
            results['interpretation'] = interpretation
    
    return results


if __name__ == "__main__":
    # Configuration
    TICKER = "META"
    EXPIRY_INDEX = 0  # 0 = nearest expiry
    
    # Run
    results = plot_smile_overlay_standalone(
        ticker_symbol=TICKER,
        expiry_index=EXPIRY_INDEX,
        bin_count=60,
        save_path=None,  # Auto-generate filename
        show_plot=True
    )
    
    print("\nResults:")
    for key, value in results.items():
        print(f"  {key}: {value}")

