import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
data = yf.download(tickers, start='2020-01-01', end='2024-01-01',
                   auto_adjust=True, progress=False)['Close']
returns = data.pct_change().dropna()

def sharpe(rets, rf=0.02/252):
    excess = rets - rf
    return np.sqrt(252) * excess.mean() / excess.std()

equal_weight = returns.mean(axis=1)
np.random.seed(42)
random_weight = returns.apply(
    lambda x: x @ np.random.dirichlet(np.ones(len(tickers))), axis=1)
inv_vol = 1 / returns.std()
riskaware_weight = inv_vol / inv_vol.sum()
riskaware = returns @ riskaware_weight

print('Sharpe (Equal Weight):', round(float(sharpe(equal_weight)), 3))
print('Sharpe (Random):      ', round(float(sharpe(random_weight)), 3))
print('Sharpe (Risk-Aware):  ', round(float(sharpe(riskaware)), 3))

fig, ax = plt.subplots(figsize=(10, 5))
(1 + pd.DataFrame({
    'Equal Weight': equal_weight,
    'Random': random_weight,
    'Risk-Aware (proxy)': riskaware
})).cumprod().plot(ax=ax)
ax.set_title('Cumulative Returns Comparison (2020--2024)', fontsize=14)
ax.set_ylabel('Portfolio Value (start = 1)')
ax.set_xlabel('Date')
ax.axhline(1, color='black', lw=0.8, ls='--', alpha=0.5)
ax.legend()
plt.tight_layout()
plt.savefig('results/cumulative_returns.png', dpi=150)
print('Saved: results/cumulative_returns.png')
