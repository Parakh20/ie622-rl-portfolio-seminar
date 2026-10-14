import numpy as np
import pytest

from baselines.baselines import (
    _min_variance_weights,
    _risk_parity_weights,
    buy_and_hold_returns,
    equal_weight_returns,
    markowitz_returns,
    risk_parity_returns,
)


def test_equal_weight_is_row_mean(asset_returns):
    out = equal_weight_returns(asset_returns)
    np.testing.assert_allclose(out.values, asset_returns.mean(axis=1).values)


@pytest.mark.parametrize("weight_fn", [_min_variance_weights, _risk_parity_weights])
def test_weights_are_long_only_and_sum_to_one(asset_returns, weight_fn):
    w = weight_fn(asset_returns.iloc[:60].cov().values)
    assert w.shape == (4,)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert (w >= -1e-9).all()


def test_min_variance_favours_lowest_volatility_asset(asset_returns):
    w = _min_variance_weights(asset_returns.cov().values)
    assert np.argmax(w) == 0  # asset A has the smallest volatility


def test_risk_parity_weights_inverse_to_volatility(asset_returns):
    w = _risk_parity_weights(asset_returns.cov().values)
    assert np.all(np.diff(w) < 0)  # A (low vol) > B > C > D (high vol)


@pytest.mark.parametrize("strategy", [markowitz_returns, risk_parity_returns])
def test_rolling_strategies_return_aligned_series(asset_returns, strategy):
    out = strategy(asset_returns)
    assert out.index.equals(asset_returns.index)
    assert np.isfinite(out.values).all()
    # before the first rebalance the portfolio is equal weight
    np.testing.assert_allclose(out.iloc[:60].values, asset_returns.iloc[:60].mean(axis=1).values)


def test_buy_and_hold_returns(close):
    out = buy_and_hold_returns(close)
    assert len(out) == len(close) - 1
    assert out.iloc[0] == pytest.approx(close.iloc[1] / close.iloc[0] - 1)
