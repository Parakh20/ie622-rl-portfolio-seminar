"""Builds (but never trains) the RecurrentPPO agent to check wiring and hyperparameters."""
import numpy as np
import pytest

pytest.importorskip("sb3_contrib")

from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from agents.recurrent_ppo import HYPERPARAMS, build_agent  # noqa: E402
from envs.portfolio_env import PortfolioEnv  # noqa: E402


def test_build_agent_uses_report_hyperparameters(close):
    venv = DummyVecEnv([lambda: PortfolioEnv(close)])
    model = build_agent(venv, seed=0, verbose=0)
    assert model.n_steps == HYPERPARAMS["n_steps"] == 252
    assert model.batch_size == 63
    assert model.policy.lstm_actor.hidden_size == 128


def test_untrained_agent_predicts_valid_action(close):
    venv = DummyVecEnv([lambda: PortfolioEnv(close)])
    model = build_agent(venv, seed=0, verbose=0)
    obs = venv.reset()
    action, state = model.predict(obs, state=None, episode_start=np.array([True]), deterministic=True)
    assert action.shape == (1,)
    assert int(action[0]) in (0, 1, 2)
