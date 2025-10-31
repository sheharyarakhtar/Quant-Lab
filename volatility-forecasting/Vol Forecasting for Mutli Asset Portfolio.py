#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 15:07:27 2025

@author: aadiljaved
"""

import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt

# Period under consideration:
    
start_date = '2018-01-01'
end_date = '2025-01-01'


# Asset Classes:  
assets = ['SPY', 'TLT', 'GLD','USO']


# Fetching data:

data = yf.download(assets,start_date,end_date)['Close']


"""
    Data Sanity & Cleaning Checks:
"""
  
## Check for NANs:
print("Number of NAs:" , data.isna().sum())

## Check for zeroes in dataset:
print("\nCheck for Zero Values for each ticker:\n", (data == 0).sum())


## Check to see if all tickers share the same exact dates & all columns have data on those dates: 

print(data.index.nunique() == len(data))

print(data.notna().all(axis=1).all())


## Check to find dates if any asset has missing data:
missing_dates = data[data.isna().any(axis=1)]
print("Missing Dates:", missing_dates.head())


data_stats = data.describe()
print("\nData Statistics\n\n", data_stats)


"""
    Visualizing Data:
"""

# Plotting prices:
data.plot(title='Price History of Assets', figsize=(10,5))


# Calculating & Plotting Returns:

returns = data.pct_change().dropna()
returns.plot(title='Returns of Assets', figsize=(10,5))

# Calculating & Plotting log returns:
log_returns = np.log(data / data.shift(1))
log_returns = log_returns.dropna()

log_returns.plot(title='Log Returns of Assets', figsize=(10,5))


"""
    Correlations among Assets:
"""

simple_corr = returns.corr()
log_corr = log_returns.corr()

print("Simple return correlation:\n", simple_corr)
print("\nLog return correlation:\n", log_corr)

## Heatmaps:

### Simple Returns Correlations Heatmap:
plt.figure(figsize=(7,5))
sns.heatmap(simple_corr, annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Heatmap of Log Returns", fontsize=14)
plt.show()


### Log Returns Correlations Heatmap:
plt.figure(figsize=(7,5))
sns.heatmap(log_corr, annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Heatmap of Log Returns", fontsize=14)
plt.show()

## Combined Heatmaps:
fig, axes = plt.subplots(1, 2, figsize=(12,5))
sns.heatmap(simple_corr, annot=True, cmap='coolwarm', center=0, ax=axes[0])
axes[0].set_title('Simple Return Correlation')

sns.heatmap(log_corr, annot=True, cmap='coolwarm', center=0, ax=axes[1])
axes[1].set_title('Log Return Correlation')

plt.tight_layout()
plt.show()


"""
    Summary Statistics:
"""

R = returns   # use simple for portfolio compounding; use log_ret for modeling

ann_fac = 252 # number of days in a trading year.

stats = pd.DataFrame({
    "Mean(daily)": R.mean(),
    "Vol(daily)":  R.std(),
    "Ann.Return":  (1 + R).prod()**(ann_fac/len(R)) - 1,   # geometric
    "Ann.Vol":     R.std()*np.sqrt(ann_fac),
    "Skew":        R.skew(),
    "Kurtosis":    R.kurt()
})

print("\nAsset Statistics:\n", stats.round(4))



"""
    Portfolio Construction & Analysis:
"""

# Weights:
    
weights = np.array([0.25,0.25,0.25,0.25]) # equal weights for now.

portfolio_returns = (returns * weights).sum(axis=1)             #daily portfolio retruns
cum_portfolio_value = (1 + portfolio_returns).cumprod()     #Cumulative Compounding of Returns 

portfolio_df = pd.DataFrame({
    'Portfolio_Return': portfolio_returns,
    'Portfolio_Value': cum_portfolio_value
})

plt.figure(figsize=(10,5))
plt.plot(portfolio_df['Portfolio_Value'], label='Portfolio Growth')
plt.title('Portfolio Growth Over Time ($1 Initial Investment)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend()
plt.grid(True)
plt.show()


