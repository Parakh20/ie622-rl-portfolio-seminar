import numpy as np
import pandas as pd

import data.download as download


def test_load_close_caches_full_history_and_slices(monkeypatch):
    calls = []
    idx = pd.bdate_range("2016-01-01", "2024-12-31")
    full = pd.DataFrame({"Close": np.arange(len(idx), dtype=float)}, index=idx)

    def fake_download_ticker(ticker, **kwargs):
        calls.append((ticker, kwargs))
        return full

    monkeypatch.setattr(download, "download_ticker", fake_download_ticker)

    test = download.load_close("AAPL", "2022-01-01", "2023-12-31")
    train = download.load_close("AAPL", "2018-01-01", "2021-12-31")

    # no period-specific start is passed, so one cache serves every period
    assert calls == [("AAPL", {}), ("AAPL", {})]
    assert test.index.min() >= pd.Timestamp("2022-01-01")
    assert test.index.max() <= pd.Timestamp("2023-12-31")
    assert train.index.min() >= pd.Timestamp("2018-01-01")
    assert train.index.max() <= pd.Timestamp("2021-12-31")
