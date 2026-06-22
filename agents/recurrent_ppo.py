"""RecurrentPPO (LSTM) agent factory, with hyperparameters from the seminar report.

Tuning notes (see results/tuning_notes.md):
- n_steps raised 128->252 (~one trading year) so the LSTM's truncated
  backprop-through-time window covers enough sequential structure to learn
  momentum/volatility regimes instead of fragmenting into short chunks.
- batch_size set to 63 (252 / 4) so it evenly divides n_steps for a single
  env, giving 4 minibatches per epoch.
- ent_coef now decays 0.05 -> 0.01 over training via EntCoefDecayCallback,
  since stable-baselines3 PPO does not support a schedule for this
  parameter natively.
"""
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback

HYPERPARAMS = dict(
    learning_rate=1e-4,
    n_steps=252,
    batch_size=63,
    n_epochs=10,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.15,
    ent_coef=0.05,
    policy_kwargs=dict(
        lstm_hidden_size=128,
        n_lstm_layers=1,
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
    ),
)

DEFAULT_TOTAL_TIMESTEPS = 300_000


class EntCoefDecayCallback(BaseCallback):
    """Linearly decay `model.ent_coef` from `start` to `end` over training."""

    def __init__(self, start: float = 0.05, end: float = 0.01, verbose: int = 0):
        super().__init__(verbose)
        self.start = start
        self.end = end

    def _on_step(self) -> bool:
        progress = min(self.num_timesteps / max(self.locals.get("total_timesteps", 1), 1), 1.0)
        self.model.ent_coef = self.start + progress * (self.end - self.start)
        return True


def build_agent(env, seed: int | None = None, verbose: int = 1, **overrides) -> RecurrentPPO:
    """Build a RecurrentPPO agent over `env` using the report's hyperparameters."""
    kwargs = {**HYPERPARAMS, **overrides}
    return RecurrentPPO(
        "MlpLstmPolicy",
        env,
        seed=seed,
        verbose=verbose,
        **kwargs,
    )
