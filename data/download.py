"""Download and cache OHLCV price data for the project's asset universe.

Usage:
    python data/download.py                  # download all assets, full history
    python data/download.py --tickers AAPL MSFT
"""
import argparse
import os
import sys

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.universe import ALL_ASSETS

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")


def download_ticker(ticker: str, start: str = "2016-01-01", end: str | None = None,
                     cache_dir: str = RAW_DIR, force: bool = False) -> pd.DataFrame:
    """Download one ticker's OHLCV history, caching to a local CSV."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{ticker}.csv")

    if os.path.exists(cache_path) and not force:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if not df.empty:
            return df

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"yfinance returned no data for ticker '{ticker}'")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.to_csv(cache_path)
    return df


def load_close(ticker: str, start: str, end: str) -> pd.Series:
    """Return the cached daily Close series for `ticker`, sliced to [start, end].

    Always fetches/caches the full history (from download_ticker's default
    start) so the cache is valid for every period. Requesting from `start`
    would cache only that window, and a later call for an earlier period
    (e.g. training after a backtest) would silently get no data.
    """
    df = download_ticker(ticker)
    close = df["Close"].astype(float)
    close.index = pd.to_datetime(close.index)
    return close.loc[start:end]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=ALL_ASSETS)
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()

    for ticker in args.tickers:
        try:
            df = download_ticker(ticker, start=args.start, end=args.end, force=args.force)
            print(f"{ticker}: {len(df)} rows cached -> {os.path.join(RAW_DIR, ticker + '.csv')}")
        except Exception as exc:
            print(f"{ticker}: FAILED ({exc})")


if __name__ == "__main__":
    main()
