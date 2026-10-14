import math

import pandas as pd
import pytest

from agents.metrics import ann_return, calmar, cumulative_return, max_drawdown, sharpe, summarize


def test_sharpe_of_constant_returns_is_zero():
    assert sharpe(pd.Series([0.001] * 50)) == 0.0


def test_max_drawdown_known_path():
    # 1.0 -> 1.1 -> 0.88 -> 0.968 : peak 1.1, trough 0.88 => -20%
    rets = pd.Series([0.10, -0.20, 0.10])
    assert max_drawdown(rets) == pytest.approx(-0.20)


def test_cumulative_and_annual_return():
    rets = pd.Series([0.01] * 252)
    assert cumulative_return(rets) == pytest.approx(1.01 ** 252 - 1)
    assert ann_return(rets) == pytest.approx(1.01 ** 252 - 1)
    assert ann_return(pd.Series([], dtype=float)) == 0.0


def test_calmar_is_nan_without_drawdown():
    assert math.isnan(calmar(pd.Series([0.01] * 10)))


def test_summarize_keys():
    out = summarize(pd.Series([0.01, -0.02, 0.015]))
    assert set(out) == {"ann_return", "cumulative_return", "sharpe", "max_drawdown", "calmar"}
