import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import shutil

# Clean up Images folder before starting
if os.path.exists('Images'):
    shutil.rmtree('Images')
os.makedirs('Images')

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
        
        all_calls.append(calls)
        all_puts.append(puts)
    
    calls_df = pd.concat(all_calls, ignore_index=True)
    puts_df = pd.concat(all_puts, ignore_index=True)
    
    return calls_df, puts_df

def calculate_time_to_expiry(expiry_date):
    """Calculate time to expiry in years"""
    expiry = pd.to_datetime(expiry_date)
    today = pd.Timestamp.now()
    days_to_expiry = (expiry - today).days
    return max(days_to_expiry / 365.0, 0.001)  # Avoid division by zero

def diagnose_spikes(options_df, current_price, option_type='Call', iv_threshold=1.0, moneyness_range=(0, 0.6)):
    """Diagnose suspicious IV spikes to identify data quality issues
    
    Args:
        options_df: DataFrame with options data
        current_price: Current underlying price
        option_type: 'Call' or 'Put'
        iv_threshold: IV threshold above which to flag (default 1.0 = 100%)
        moneyness_range: Tuple of (min, max) moneyness to check for spikes
    
    Returns:
        DataFrame of suspicious rows
    """
    df = options_df.copy()
    df['moneyness'] = df['strike'] / current_price
    
    # Find suspicious rows
    spike_rows = df[
        (df['moneyness'] >= moneyness_range[0]) & 
        (df['moneyness'] <= moneyness_range[1]) & 
        (df['impliedVolatility'] > iv_threshold)
    ].sort_values(['expiryDate', 'moneyness', 'impliedVolatility'])
    
    if len(spike_rows) > 0:
        print(f"\n{'='*80}")
        print(f"DIAGNOSTIC: Found {len(spike_rows)} suspicious {option_type} options")
        print(f"  Criteria: Moneyness {moneyness_range[0]}-{moneyness_range[1]}, IV > {iv_threshold*100}%")
        print(f"{'='*80}\n")
        
        # Display key columns
        cols_to_show = ['expiryDate', 'strike', 'moneyness', 'impliedVolatility', 
                       'volume', 'openInterest', 'lastPrice', 'bid', 'ask']
        available_cols = [col for col in cols_to_show if col in spike_rows.columns]
        
        print(spike_rows[available_cols].head(20).to_string())
        print(f"\n... showing first 20 of {len(spike_rows)} suspicious rows\n")
    else:
        print(f"\nNo suspicious {option_type} options found with current criteria.\n")
    
    return spike_rows

def prepare_surface_data(options_df, current_price, strict_filtering=True, 
                        moneyness_range=(0.7, 1.5), iv_percentile_cap=98):
    """Prepare data for volatility surface plotting with comprehensive data cleaning
    
    Args:
        options_df: DataFrame with options data
        current_price: Current underlying price
        strict_filtering: If True, apply stricter filters for volume/OI
        moneyness_range: Tuple (min, max) to filter strikes (default 0.7-1.5)
        iv_percentile_cap: Remove IV values above this percentile (default 98)
    """
    options_df = options_df.copy()
    options_df['timeToExpiry'] = options_df['expiryDate'].apply(calculate_time_to_expiry)
    options_df['moneyness'] = options_df['strike'] / current_price
    
    # Fill NaN values with 0 for volume and open interest early
    options_df['volume'] = options_df['volume'].fillna(0)
    options_df['openInterest'] = options_df['openInterest'].fillna(0)
    
    print(f"  Initial rows: {len(options_df)}")
    
    # Basic filters - remove missing/invalid data
    options_df = options_df.dropna(subset=['impliedVolatility', 'strike', 'timeToExpiry'])
    print(f"  After removing NaN: {len(options_df)}")
    
    # 1. MONEYNESS FILTER - Focus on liquid strikes
    options_df = options_df[
        (options_df['moneyness'] >= moneyness_range[0]) & 
        (options_df['moneyness'] <= moneyness_range[1])
    ]
    print(f"  After moneyness filter ({moneyness_range[0]:.2f}-{moneyness_range[1]:.2f}): {len(options_df)}")
    
    # 2. Remove near-zero IVs
    options_df = options_df[options_df['impliedVolatility'] > 0.005]  # Remove < 0.5% IV
    print(f"  After removing IV < 0.5%: {len(options_df)}")
    
    # 3. PERCENTILE-BASED IV OUTLIER REMOVAL
    if len(options_df) > 0:
        iv_cap = options_df['impliedVolatility'].quantile(iv_percentile_cap / 100)
        options_df = options_df[options_df['impliedVolatility'] <= iv_cap]
        print(f"  After removing IV outliers (>{iv_percentile_cap}th percentile, {iv_cap*100:.1f}%): {len(options_df)}")
    
    # 4. LIQUIDITY FILTERS
    if strict_filtering:
        # Require either volume > 0 OR open interest > 0
        options_df = options_df[
            (options_df['volume'] > 0) | (options_df['openInterest'] > 0)
        ]
        print(f"  After requiring volume or OI > 0: {len(options_df)}")
        
        # 5. STRICTER FILTERS FOR DEEP OTM OPTIONS
        # Define deep OTM regions
        deep_otm = (options_df['moneyness'] < 0.8) | (options_df['moneyness'] > 1.3)
        
        # For deep OTM, require higher liquidity
        low_liquidity_otm = (options_df['volume'] < 5) & (options_df['openInterest'] < 20)
        options_df = options_df[~(deep_otm & low_liquidity_otm)]
        print(f"  After removing illiquid deep OTM options: {len(options_df)}")
        
        # Remove any remaining low-liquidity high-IV combinations
        low_liquidity = (options_df['volume'] <= 1) & (options_df['openInterest'] <= 5)
        high_iv = options_df['impliedVolatility'] > 0.5  # > 50%
        suspicious = low_liquidity & high_iv
        options_df = options_df[~suspicious]
        print(f"  After removing low-liquidity high-IV options: {len(options_df)}")
    
    # Mark liquid vs illiquid for potential future use
    options_df['liquid'] = (
        (options_df['volume'] > 10) | 
        (options_df['openInterest'] > 10)
    )
    
    print(f"  Final rows: {len(options_df)}")
    
    return options_df

def plot_volatility_surface_3d(calls_surface, puts_surface, ticker_symbol):
    """Plot interactive 3D volatility surfaces for both calls and puts side by side"""
    # Create subplots with 3D scenes
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=('Call Options', 'Put Options')
    )
    
    # Process Calls
    x_calls = calls_surface['moneyness'].values
    y_calls = calls_surface['timeToExpiry'].values
    z_calls = calls_surface['impliedVolatility'].values * 100
    
    xi_calls = np.linspace(x_calls.min(), x_calls.max(), 50)
    yi_calls = np.linspace(y_calls.min(), y_calls.max(), 50)
    xi_grid_calls, yi_grid_calls = np.meshgrid(xi_calls, yi_calls)
    zi_calls = griddata((x_calls, y_calls), z_calls, (xi_grid_calls, yi_grid_calls), method='cubic')
    
    # Process Puts
    x_puts = puts_surface['moneyness'].values
    y_puts = puts_surface['timeToExpiry'].values
    z_puts = puts_surface['impliedVolatility'].values * 100
    
    xi_puts = np.linspace(x_puts.min(), x_puts.max(), 50)
    yi_puts = np.linspace(y_puts.min(), y_puts.max(), 50)
    xi_grid_puts, yi_grid_puts = np.meshgrid(xi_puts, yi_puts)
    zi_puts = griddata((x_puts, y_puts), z_puts, (xi_grid_puts, yi_grid_puts), method='cubic')
    
    # Add Call surface
    fig.add_trace(
        go.Surface(x=xi_grid_calls, y=yi_grid_calls, z=zi_calls, colorscale='Viridis', name='Call Surface'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter3d(x=x_calls, y=y_calls, z=z_calls, mode='markers',
                     marker=dict(size=2, color='red', opacity=0.5), name='Call Data'),
        row=1, col=1
    )
    
    # Add Put surface
    fig.add_trace(
        go.Surface(x=xi_grid_puts, y=yi_grid_puts, z=zi_puts, colorscale='Plasma', name='Put Surface'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter3d(x=x_puts, y=y_puts, z=z_puts, mode='markers',
                     marker=dict(size=2, color='blue', opacity=0.5), name='Put Data'),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f'{ticker_symbol} Implied Volatility Surfaces (3D)',
        scene=dict(
            xaxis_title='Moneyness',
            yaxis_title='Time to Expiry (Years)',
            zaxis_title='IV (%)',
        ),
        scene2=dict(
            xaxis_title='Moneyness',
            yaxis_title='Time to Expiry (Years)',
            zaxis_title='IV (%)',
        ),
        width=1800,
        height=800
    )
    
    filename = f'Images/{ticker_symbol}_3D_Surfaces.html'
    fig.write_html(filename)
    print(f"Saved: {filename}")

def plot_volatility_surface_3d_static(calls_surface, puts_surface, ticker_symbol):
    """Plot static 3D volatility surfaces for both calls and puts as JPEG"""
    # Process Calls
    x_calls = calls_surface['moneyness'].values
    y_calls = calls_surface['timeToExpiry'].values
    z_calls = calls_surface['impliedVolatility'].values * 100
    
    xi_calls = np.linspace(x_calls.min(), x_calls.max(), 50)
    yi_calls = np.linspace(y_calls.min(), y_calls.max(), 50)
    xi_grid_calls, yi_grid_calls = np.meshgrid(xi_calls, yi_calls)
    zi_calls = griddata((x_calls, y_calls), z_calls, (xi_grid_calls, yi_grid_calls), method='cubic')
    
    # Process Puts
    x_puts = puts_surface['moneyness'].values
    y_puts = puts_surface['timeToExpiry'].values
    z_puts = puts_surface['impliedVolatility'].values * 100
    
    xi_puts = np.linspace(x_puts.min(), x_puts.max(), 50)
    yi_puts = np.linspace(y_puts.min(), y_puts.max(), 50)
    xi_grid_puts, yi_grid_puts = np.meshgrid(xi_puts, yi_puts)
    zi_puts = griddata((x_puts, y_puts), z_puts, (xi_grid_puts, yi_grid_puts), method='cubic')
    
    # Create figure with two 3D subplots
    fig = plt.figure(figsize=(20, 9), facecolor='white')
    
    # Call Options Surface
    ax1 = fig.add_subplot(121, projection='3d', facecolor='#f8f9fa')
    surf1 = ax1.plot_surface(xi_grid_calls, yi_grid_calls, zi_calls, 
                             cmap='viridis', alpha=0.85, edgecolor='none',
                             linewidth=0, antialiased=True)
    ax1.scatter(x_calls, y_calls, z_calls, c='red', marker='o', s=8, alpha=0.4)
    
    ax1.set_xlabel('Moneyness', fontsize=12, fontweight='semibold', labelpad=10)
    ax1.set_ylabel('Time to Expiry (Years)', fontsize=12, fontweight='semibold', labelpad=10)
    ax1.set_zlabel('IV (%)', fontsize=12, fontweight='semibold', labelpad=10)
    ax1.set_title('Call Options', fontsize=14, fontweight='bold', pad=15, color='#2c3e50')
    ax1.view_init(elev=25, azim=135)
    
    # Add colorbar for calls
    cbar1 = fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5, pad=0.08)
    cbar1.set_label('IV (%)', fontsize=10, fontweight='semibold')
    
    # Put Options Surface
    ax2 = fig.add_subplot(122, projection='3d', facecolor='#f8f9fa')
    surf2 = ax2.plot_surface(xi_grid_puts, yi_grid_puts, zi_puts, 
                             cmap='plasma', alpha=0.85, edgecolor='none',
                             linewidth=0, antialiased=True)
    ax2.scatter(x_puts, y_puts, z_puts, c='blue', marker='o', s=8, alpha=0.4)
    
    ax2.set_xlabel('Moneyness', fontsize=12, fontweight='semibold', labelpad=10)
    ax2.set_ylabel('Time to Expiry (Years)', fontsize=12, fontweight='semibold', labelpad=10)
    ax2.set_zlabel('IV (%)', fontsize=12, fontweight='semibold', labelpad=10)
    ax2.set_title('Put Options', fontsize=14, fontweight='bold', pad=15, color='#2c3e50')
    ax2.view_init(elev=25, azim=135)
    
    # Add colorbar for puts
    cbar2 = fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5, pad=0.08)
    cbar2.set_label('IV (%)', fontsize=10, fontweight='semibold')
    
    # Main title
    fig.suptitle(f'{ticker_symbol} — Implied Volatility Surfaces (3D)', 
                fontsize=18, fontweight='bold', y=0.98, color='#2c3e50')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    filename = f'Images/{ticker_symbol}_3D_Surfaces_Static.jpg'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', format='jpg')
    plt.close()
    print(f"Saved: {filename}")

def plot_volatility_strike_view(calls_surface, puts_surface, ticker_symbol, ylim=(20, 90), bin_count=60):
    """Plot overlayed calls and puts volatility smile for multiple expiries on one chart
    
    Similar to plot_overlayed_smile but shows all expiries together
    
    Args:
        ylim: Tuple (min, max) for y-axis limits in percentage (default 20-90%)
        bin_count: Number of bins for smoothed median curve
    """
    # Combine calls and puts
    calls = calls_surface.copy()
    puts = puts_surface.copy()
    calls['type'] = 'Call'
    puts['type'] = 'Put'
    
    df = pd.concat([calls, puts], ignore_index=True)
    df['log_moneyness'] = np.log(df['moneyness'])
    df['IV_pct'] = df['impliedVolatility'] * 100
    
    # Get up to 5 nearest expiries
    expiries = sorted(df['expiryDate'].unique())[:5]
    
    # Vibrant, contrasting color palette (like a professional dashboard)
    color_palette = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
    ]
    colors = color_palette[:len(expiries)]
    
    # Binning helper function
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
    
    # Create single plot with white background
    fig, ax = plt.subplots(figsize=(18, 10), facecolor='white')
    ax.set_facecolor('#f8f9fa')
    
    # Get global x-range for consistent binning
    all_x = df['log_moneyness'].values
    xmin, xmax = np.nanpercentile(all_x, [1, 99])
    bins = np.linspace(xmin, xmax, bin_count)
    
    for idx, expiry in enumerate(expiries):
        color = colors[idx]
        expiry_data = df[df['expiryDate'] == expiry].copy()
        
        # Separate calls and puts
        calls_subset = expiry_data[expiry_data['type'] == 'Call'].copy()
        puts_subset = expiry_data[expiry_data['type'] == 'Put'].copy()
        
        # Plot raw data points (smaller, more transparent)
        ax.scatter(calls_subset['log_moneyness'], calls_subset['IV_pct'],
                  s=15, alpha=0.2, marker='o', color=color, edgecolors='none')
        ax.scatter(puts_subset['log_moneyness'], puts_subset['IV_pct'],
                  s=15, alpha=0.2, marker='o', color=color, edgecolors='none')
        
        # Compute binned medians for calls and puts
        if len(calls_subset) > 0:
            cx, cy = binned_median(calls_subset['log_moneyness'].values,
                                  calls_subset['IV_pct'].values, bins)
            if len(cx) > 0:
                ax.plot(cx, cy, '-o', color=color, lw=2.8, markersize=3.5,
                       label=f'{expiry}', zorder=10, markeredgewidth=0)
        
        if len(puts_subset) > 0:
            px, py = binned_median(puts_subset['log_moneyness'].values,
                                  puts_subset['IV_pct'].values, bins)
            if len(px) > 0:
                ax.plot(px, py, '-o', color=color, lw=2.8, markersize=3.5,
                       zorder=10, markeredgewidth=0)
    
    # ATM vertical line - more prominent
    ax.axvline(0, color='#2c3e50', linestyle='--', alpha=0.7, linewidth=2, label='ATM', zorder=5)
    
    # X-axis: log-moneyness with readable labels
    xticks_log = np.linspace(xmin, xmax, 11)
    xticks_labels = [f"{np.exp(x):.2f}×" for x in xticks_log]
    ax.set_xticks(xticks_log)
    ax.set_xticklabels(xticks_labels, rotation=0, fontsize=11)
    
    # Enhanced styling
    ax.set_xlabel('Moneyness (Strike / Spot), log-scale; 0 = ATM', 
                 fontsize=14, fontweight='semibold', labelpad=10)
    ax.set_ylabel('Implied Volatility (%)', 
                 fontsize=14, fontweight='semibold', labelpad=10)
    ax.set_ylim(ylim)
    ax.set_title(f'{ticker_symbol} — Volatility Smile Across Expiries', 
                fontsize=18, fontweight='bold', pad=20, color='#2c3e50')
    
    # Enhanced legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95, 
             edgecolor='#bdc3c7', fancybox=True, shadow=True)
    
    # Enhanced grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, color='#95a5a6')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#7f8c8d')
    ax.spines['bottom'].set_color('#7f8c8d')
    ax.tick_params(colors='#2c3e50', which='both')
    
    plt.tight_layout()
    
    filename = f'Images/{ticker_symbol}_Strike_vs_IV.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filename}")

def plot_volatility_time_view(calls_surface, puts_surface, ticker_symbol, ylim=(20, 90)):
    """Plot 2D view: Time to Expiry vs IV for both calls and puts
    
    Args:
        ylim: Tuple (min, max) for y-axis limits in percentage (default 20-90%)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))
    
    moneyness_buckets = [
        (0.90, 0.95, 'OTM (0.90-0.95)'),
        (0.95, 1.00, 'Near ATM (0.95-1.00)'),
        (1.00, 1.05, 'ATM (1.00-1.05)'),
        (1.05, 1.10, 'Near ITM (1.05-1.10)'),
        (1.10, 1.15, 'ITM (1.10-1.15)')
    ]
    
    # Calls
    for lower, upper, label in moneyness_buckets:
        bucket_data = calls_surface[
            (calls_surface['moneyness'] >= lower) & 
            (calls_surface['moneyness'] < upper)
        ]
        
        if len(bucket_data) > 0:
            time_series = bucket_data.groupby('timeToExpiry').agg({
                'impliedVolatility': 'mean'
            }).sort_index()
            
            ax1.plot(time_series.index, time_series['impliedVolatility'] * 100, 
                    marker='o', label=label, linewidth=2, markersize=4)
    
    ax1.set_xlabel('Time to Expiry (Years)', fontsize=12)
    ax1.set_ylabel('Implied Volatility (%)', fontsize=12)
    ax1.set_title('Call Options - Time to Expiry vs IV', fontsize=14)
    ax1.set_ylim(ylim)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Puts
    for lower, upper, label in moneyness_buckets:
        bucket_data = puts_surface[
            (puts_surface['moneyness'] >= lower) & 
            (puts_surface['moneyness'] < upper)
        ]
        
        if len(bucket_data) > 0:
            time_series = bucket_data.groupby('timeToExpiry').agg({
                'impliedVolatility': 'mean'
            }).sort_index()
            
            ax2.plot(time_series.index, time_series['impliedVolatility'] * 100, 
                    marker='o', label=label, linewidth=2, markersize=4)
    
    ax2.set_xlabel('Time to Expiry (Years)', fontsize=12)
    ax2.set_ylabel('Implied Volatility (%)', fontsize=12)
    ax2.set_title('Put Options - Time to Expiry vs IV', fontsize=14)
    ax2.set_ylim(ylim)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'{ticker_symbol} - Time to Expiry vs IV Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    
    filename = f'Images/{ticker_symbol}_Time_vs_IV.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def plot_overlayed_smile(calls_df, puts_df, current_price, ticker_symbol, expiry_index=0, bin_count=60, ylim=(20, 90)):
    """Plot overlayed calls and puts volatility smile using log-moneyness
    
    This creates a single plot showing both calls and puts on the same axes,
    centered at ATM (log-moneyness = 0). This makes it easy to compare
    left tail (put) vs right tail (call) skew.
    
    Args:
        calls_df: DataFrame with call options data
        puts_df: DataFrame with put options data
        current_price: Current underlying price (spot)
        ticker_symbol: Stock ticker symbol
        expiry_index: Which expiry to plot (0 = nearest, 1 = next, etc.)
        bin_count: Number of bins for smoothed median curve
        ylim: Tuple (min, max) for y-axis limits in percentage (default 20-90%)
    """
    # Combine calls and puts with type labels
    calls = calls_df.copy()
    puts = puts_df.copy()
    calls['type'] = 'C'
    puts['type'] = 'P'
    
    df = pd.concat([calls, puts], ignore_index=True)
    
    # Compute moneyness and log-moneyness
    df['moneyness'] = df['strike'] / current_price
    df['log_moneyness'] = np.log(df['moneyness'])
    
    # Select expiry
    expiries = sorted(df['expiryDate'].unique())
    if expiry_index >= len(expiries):
        expiry_index = 0
    
    plot_expiry = expiries[expiry_index]
    subset = df[df['expiryDate'] == plot_expiry].copy()
    
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
    fig, ax = plt.subplots(figsize=(14, 8))
    
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
    ax.set_ylim(ylim)
    ax.set_title(f'{ticker_symbol} — Overlayed Calls & Puts Volatility Smile\nExpiry: {plot_expiry}', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.35)
    
    plt.tight_layout()
    
    filename = f'Images/{ticker_symbol}_Overlayed_Smile_{plot_expiry}.png'
    plt.savefig(filename, dpi=220, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")
    
    # Print interpretation
    print(f"\n  Smile Interpretation for {plot_expiry}:")
    if len(px) > 0 and len(cx) > 0:
        left_tail_iv = np.median(py[px < -0.1]) if np.any(px < -0.1) else np.nan
        right_tail_iv = np.median(cy[cx > 0.1]) if np.any(cx > 0.1) else np.nan
        
        if not np.isnan(left_tail_iv) and not np.isnan(right_tail_iv):
            skew = left_tail_iv - right_tail_iv
            print(f"    Left tail (OTM puts): {left_tail_iv:.1f}%")
            print(f"    Right tail (OTM calls): {right_tail_iv:.1f}%")
            print(f"    Skew: {skew:+.1f}%")
            
            if skew > 5:
                print(f"    → Classic downside skew (puts more expensive)")
            elif skew < -5:
                print(f"    → Upside skew (calls more expensive, bullish speculation)")
            else:
                print(f"    → Relatively balanced tail risk")

def plot_volume_oi_analysis(calls_df, puts_df, ticker_symbol):
    """Plot volume and open interest analysis for both calls and puts"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(22, 14))
    
    # Call Volume by strike
    volume_by_strike_calls = calls_df.groupby('strike')['volume'].sum().sort_index()
    ax1.bar(volume_by_strike_calls.index, volume_by_strike_calls.values, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Strike Price', fontsize=12)
    ax1.set_ylabel('Total Volume', fontsize=12)
    ax1.set_title('Call Options - Volume by Strike', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Put Volume by strike
    volume_by_strike_puts = puts_df.groupby('strike')['volume'].sum().sort_index()
    ax2.bar(volume_by_strike_puts.index, volume_by_strike_puts.values, alpha=0.7, color='coral')
    ax2.set_xlabel('Strike Price', fontsize=12)
    ax2.set_ylabel('Total Volume', fontsize=12)
    ax2.set_title('Put Options - Volume by Strike', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Call Open Interest by strike
    oi_by_strike_calls = calls_df.groupby('strike')['openInterest'].sum().sort_index()
    ax3.bar(oi_by_strike_calls.index, oi_by_strike_calls.values, alpha=0.7, color='steelblue')
    ax3.set_xlabel('Strike Price', fontsize=12)
    ax3.set_ylabel('Total Open Interest', fontsize=12)
    ax3.set_title('Call Options - Open Interest by Strike', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Put Open Interest by strike
    oi_by_strike_puts = puts_df.groupby('strike')['openInterest'].sum().sort_index()
    ax4.bar(oi_by_strike_puts.index, oi_by_strike_puts.values, alpha=0.7, color='coral')
    ax4.set_xlabel('Strike Price', fontsize=12)
    ax4.set_ylabel('Total Open Interest', fontsize=12)
    ax4.set_title('Put Options - Open Interest by Strike', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'{ticker_symbol} - Volume and Open Interest Analysis', fontsize=16, y=1.00)
    plt.tight_layout()
    
    filename = f'Images/{ticker_symbol}_Volume_OI.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def analyze_ticker(ticker_symbol, run_diagnostics=True, strict_filtering=True):
    """Complete analysis for a single ticker
    
    Args:
        ticker_symbol: Stock ticker symbol
        run_diagnostics: If True, run spike diagnostics before filtering
        strict_filtering: If True, apply strict data quality filters
    """
    print(f"\n{'='*60}")
    print(f"Analyzing {ticker_symbol}...")
    print(f"{'='*60}\n")
    
    # Get current price
    ticker = yf.Ticker(ticker_symbol)
    current_price = ticker.history(period='1d')['Close'].iloc[-1]
    print(f"Current Price: ${current_price:.2f}\n")
    
    # Fetch options data
    print("Fetching options data...")
    calls_df, puts_df = get_options_data(ticker_symbol)
    print(f"Fetched {len(calls_df)} call options and {len(puts_df)} put options\n")
    
    # Run diagnostics to identify problematic data (optional)
    if run_diagnostics:
        print("Running diagnostics to identify suspicious data points...")
        diagnose_spikes(calls_df, current_price, option_type='Call', 
                       iv_threshold=1.0, moneyness_range=(0, 0.6))
        diagnose_spikes(puts_df, current_price, option_type='Put', 
                       iv_threshold=1.0, moneyness_range=(1.4, 2.0))
    
    # Prepare data for visualization with cleaning
    print("\nCleaning Call Options Data:")
    calls_surface = prepare_surface_data(calls_df, current_price, strict_filtering=strict_filtering)
    
    print("\nCleaning Put Options Data:")
    puts_surface = prepare_surface_data(puts_df, current_price, strict_filtering=strict_filtering)
    
    # Generate visualizations
    print("\nGenerating visualizations...\n")
    
    # 3D Volatility Surfaces (Plotly) - Interactive HTML
    plot_volatility_surface_3d(calls_surface, puts_surface, ticker_symbol)
    
    # 3D Volatility Surfaces (Matplotlib) - Static JPEG
    plot_volatility_surface_3d_static(calls_surface, puts_surface, ticker_symbol)
    
    # 2D Views - Strike vs IV - Combined
    plot_volatility_strike_view(calls_surface, puts_surface, ticker_symbol)
    
    # 2D Views - Time vs IV - Combined
    plot_volatility_time_view(calls_surface, puts_surface, ticker_symbol)
    
    # Overlayed Volatility Smile (log-moneyness) - Multiple expiries
    print("\nGenerating overlayed volatility smiles...")
    num_expiries = min(3, len(sorted(calls_surface['expiryDate'].unique())))
    for i in range(num_expiries):
        plot_overlayed_smile(calls_surface, puts_surface, current_price, ticker_symbol, expiry_index=i)
    
    # Volume and Open Interest - Combined
    plot_volume_oi_analysis(calls_df, puts_df, ticker_symbol)
    
    # Summary statistics
    print(f"\n{ticker_symbol} Summary Statistics:")
    print(f"{'='*60}")
    print(f"\nCalls:")
    print(f"  Average IV: {calls_surface['impliedVolatility'].mean()*100:.2f}%")
    print(f"  IV Range: {calls_surface['impliedVolatility'].min()*100:.2f}% - {calls_surface['impliedVolatility'].max()*100:.2f}%")
    print(f"  Total Volume: {calls_df['volume'].sum():,.0f}")
    print(f"  Total Open Interest: {calls_df['openInterest'].sum():,.0f}")
    
    print(f"\nPuts:")
    print(f"  Average IV: {puts_surface['impliedVolatility'].mean()*100:.2f}%")
    print(f"  IV Range: {puts_surface['impliedVolatility'].min()*100:.2f}% - {puts_surface['impliedVolatility'].max()*100:.2f}%")
    print(f"  Total Volume: {puts_df['volume'].sum():,.0f}")
    print(f"  Total Open Interest: {puts_df['openInterest'].sum():,.0f}")
    print(f"\n{'='*60}\n")

def main(tickers):
    """Main function to analyze multiple tickers"""
    
    print("Options Volatility Surface Analysis")
    print("="*60)
    print(f"All plots will be saved to the 'Images' folder\n")
    
    for ticker in tickers:
        try:
            analyze_ticker(ticker)
        except Exception as e:
            print(f"Error analyzing {ticker}: {str(e)}\n")
            continue
    
    print("\nAnalysis complete! All plots saved to 'Images' folder.")

if __name__ == "__main__":
    tickers = ['AAPL','TSLA']
    main(tickers)