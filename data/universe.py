"""Asset universe and backtest period definitions, matching the seminar report."""

EQUITIES = [
    "AAPL", "MSFT", "GOOGL", "AMD", "INTC", "IBM", "GE",
    "META", "NFLX", "TSLA", "PYPL", "COIN",
]

CRYPTO = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD"]

ALL_ASSETS = EQUITIES + CRYPTO

# Two training/test splits used for equities in the report.
PERIOD_A = {
    "train_start": "2018-01-01", "train_end": "2021-12-31",
    "test_start": "2022-01-01", "test_end": "2023-12-31",
}
PERIOD_B = {
    "train_start": "2019-01-01", "train_end": "2022-12-31",
    "test_start": "2023-01-01", "test_end": "2024-06-30",
}

CRYPTO_PERIOD = {
    "train_start": "2020-01-01", "train_end": "2022-06-30",
    "test_start": "2022-07-01", "test_end": "2023-12-31",
}

PERIODS = {"A": PERIOD_A, "B": PERIOD_B, "crypto": CRYPTO_PERIOD}
