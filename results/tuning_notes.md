# Tuning Notes

Five tuning changes were proposed against the original report's RecurrentPPO
setup; the user approved four of them (everything except "keep defaults
as-is"). Each rationale was grounded in the report's own findings or directly
observed behavior in the reimplemented pipeline (the original training code
did not survive — only the report and figures did — so the env/agent were
rebuilt from the report's MDP spec; these are the changes made on top of
that faithful reimplementation).

## Applied

1. **Transaction-cost penalty** (`envs/portfolio_env.py`, `TRANSACTION_COST = 0.0005`)
   The original reward had no cost of trading, so the agent could flip
   short/flat/long every day for free. Added `-0.0005 * |Δposition|` per step
   so turnover is no longer free, pushing the policy toward results closer
   to what would survive real trading frictions.

2. **Observation normalization** (`scripts/train.py`, `VecNormalize(norm_obs=True)`)
   The 9 input features have heterogeneous scales (log-returns ~0.01,
   RSI normalized to [0,1], volatility ~0.01-0.05). PPO is sensitive to this;
   running mean/std normalization is a standard, low-risk stabilizer. The
   normalization statistics are frozen (`training=False`) and reused at
   evaluation/backtest time via the saved `*_vecnormalize.pkl` file.

3. **Longer rollouts** (`agents/recurrent_ppo.py`, `n_steps` 128 -> 252)
   RecurrentPPO truncates backprop-through-time to `n_steps`. At 128 steps,
   the LSTM never observed a full trading year before each update cut it
   off. Raising `n_steps` to 252 (one trading year) lets the recurrent state
   learn momentum/volatility persistence over a more representative horizon;
   `batch_size` was adjusted to 63 (252 / 4) so it evenly divides `n_steps`.

4. **Entropy decay** (`agents/recurrent_ppo.py`, `EntCoefDecayCallback`,
   0.05 -> 0.01 over training)
   A flat entropy coefficient of 0.05 keeps the policy close to uniform
   random (`entropy_loss` near `-ln(3) ≈ -1.10`) for a large fraction of the
   300k-step budget, as observed directly in early smoke tests. Decaying it
   linearly lets the agent explore early and commit to a more decisive
   policy later in training. stable-baselines3's PPO does not support a
   built-in schedule for `ent_coef`, so this is implemented as a callback
   that updates `model.ent_coef` each step.

5. **Rebalanced reward coefficients** (`envs/portfolio_env.py`,
   `SHARPE_COEF` 0.1 -> 0.5, `DD_COEF` 2.0 -> 1.0)
   In the original design, the drawdown term (`2.0 * min(drawdown, 0)`)
   numerically dominates the Sharpe term (`0.1 * rolling_sharpe`) by roughly
   one to two orders of magnitude per step. The report itself documents this
   as the cause of a degenerate always-flat policy, patched over by the
   direction bonus. Rebalancing the coefficients makes the Sharpe term a
   real competing signal rather than a rounding error, while the direction
   bonus and (now smaller) drawdown penalty still discourage reckless
   risk-taking.
