import numpy as np

from envs.features import FEATURE_NAMES, compute_features


def test_compute_features_returns_nine_named_columns(close):
    feats = compute_features(close)
    assert list(feats.columns) == FEATURE_NAMES
    assert len(FEATURE_NAMES) == 9


def test_compute_features_drops_warmup_and_has_no_nans(close):
    feats = compute_features(close)
    assert 0 < len(feats) < len(close)
    assert not feats.isna().any().any()
    assert feats.index.isin(close.index).all()


def test_momentum_and_rsi_are_bounded(close):
    feats = compute_features(close)
    assert feats["mom_5d"].abs().max() <= 0.15 + 1e-12
    assert feats["mom_30d"].abs().max() <= 0.20 + 1e-12
    assert feats["rsi_14"].between(0, 1).all()
    assert np.isfinite(feats.values).all()
