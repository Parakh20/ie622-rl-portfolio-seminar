"""Feature engineering for single-asset signal trading.

Reproduces the 9-feature set described in the seminar report: trend,
momentum, mean-reversion, and volatility signals derived from Close price.
"""
import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "log_return",
    "mom_5d",
    "mom_15d",
    "mom_30d",
    "rsi_14",
    "vol_10d",
    "vol_30d",
    "bollinger_pct_b",
    "abs_mom_5d",
]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_features(close: pd.Series) -> pd.DataFrame:
    """Compute the 9-feature matrix from a Close-price series.

    Returns a DataFrame aligned to `close.index`, with the warm-up rows
    (where rolling windows are undefined) dropped.
    """
    log_return = np.log(close / close.shift(1))

    mom_5d = (close / close.shift(5) - 1).clip(-0.15, 0.15)
    mom_15d = (close / close.shift(15) - 1).clip(-0.15, 0.15)
    mom_30d = (close / close.shift(30) - 1).clip(-0.20, 0.20)

    rsi_14 = _rsi(close, 14) / 100.0

    vol_10d = log_return.rolling(10).std()
    vol_30d = log_return.rolling(30).std()

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper, lower = mid + 2 * std, mid - 2 * std
    bollinger_pct_b = (close - lower) / (upper - lower).replace(0, np.nan)

    abs_mom_5d = log_return.abs().rolling(5).mean()

    features = pd.DataFrame({
        "log_return": log_return,
        "mom_5d": mom_5d,
        "mom_15d": mom_15d,
        "mom_30d": mom_30d,
        "rsi_14": rsi_14,
        "vol_10d": vol_10d,
        "vol_30d": vol_30d,
        "bollinger_pct_b": bollinger_pct_b,
        "abs_mom_5d": abs_mom_5d,
    })
    return features.dropna()
