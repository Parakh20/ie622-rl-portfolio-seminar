"""Shared synthetic-data fixtures. No network access or trained models needed."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def close() -> pd.Series:
    """A 300-day geometric random-walk price series."""
    rng = np.random.default_rng(0)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 300)))
    return pd.Series(prices, index=pd.bdate_range("2020-01-01", periods=300), name="Close")


@pytest.fixture
def asset_returns() -> pd.DataFrame:
    """Daily simple returns for 4 assets with different volatilities."""
    rng = np.random.default_rng(1)
    vols = np.array([0.005, 0.01, 0.02, 0.03])
    data = rng.normal(0.0003, vols, size=(200, len(vols)))
    return pd.DataFrame(data, columns=["A", "B", "C", "D"],
                        index=pd.bdate_range("2021-01-01", periods=200))
