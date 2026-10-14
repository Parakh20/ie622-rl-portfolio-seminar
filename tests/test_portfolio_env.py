import numpy as np
import pandas as pd
import pytest

from envs.portfolio_env import N_FEATURES, WINDOW, PortfolioEnv


def test_observation_and_action_spaces(close):
    env = PortfolioEnv(close)
    assert env.observation_space.shape == (WINDOW * N_FEATURES + 1,) == (181,)
    assert env.action_space.n == 3


def test_reset_returns_flat_position_and_correct_shape(close):
    env = PortfolioEnv(close)
    obs, info = env.reset(seed=0)
    assert obs.shape == (181,)
    assert obs.dtype == np.float32
    assert obs[-1] == 0.0  # starts flat
    assert info == {}


@pytest.mark.parametrize("action, position", [(0, -1.0), (1, 0.0), (2, 1.0)])
def test_step_maps_action_to_position(close, action, position):
    env = PortfolioEnv(close)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (181,)
    assert isinstance(reward, float)
    assert not terminated and not truncated
    assert info["position"] == position
    assert obs[-1] == position
    assert info["strategy_return"] == pytest.approx(position * info["asset_return"])


def test_flat_policy_keeps_equity_at_one(close):
    env = PortfolioEnv(close)
    env.reset(seed=0)
    for _ in range(10):
        _, _, _, _, info = env.step(1)
    assert info["equity"] == pytest.approx(1.0)


def test_episode_terminates_after_available_steps(close):
    env = PortfolioEnv(close)
    env.reset(seed=0)
    steps, terminated = 0, False
    while not terminated:
        obs, _, terminated, _, _ = env.step(env.action_space.sample())
        steps += 1
    assert steps == env.n_steps - env.window
    assert obs.shape == (181,)


def test_too_little_data_raises():
    short = pd.Series(np.linspace(100, 110, 40), index=pd.bdate_range("2020-01-01", periods=40))
    with pytest.raises(ValueError):
        PortfolioEnv(short)
