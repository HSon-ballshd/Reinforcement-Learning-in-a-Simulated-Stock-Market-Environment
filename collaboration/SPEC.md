# SPEC — AIE301M Project, agent-facing

This restates `../Aie Project Overview.md` for the agents. If anything here
disagrees with the original, **the original wins** until the human says otherwise.

## Goal

Build a Python port of the Cookie Clicker stock market (see
`../minigameMarket.js`), then on top of it:

1. Train a regime classifier that infers the hidden market mode from observable
   price behavior.
2. Train a DQN trader in two variants (baseline, regime-aware).
3. Compare both against three heuristic traders (Random, Buy-and-Hold,
   Mean-Reversion).

## Hypotheses to test

- **H1** — DQN beats Random and Buy-and-Hold on return.
- **H2** — Regime classifier beats random guessing (1/4 = 25.0%) by a meaningful
  margin.
- **H3** — DQN + regime beats plain DQN. **This is the academic contribution.**

## Evaluation parameters

- **Episode length:** 500 ticks (1000 ticks caused ~15M% compounded returns; 500 → ~1100%)
- **Train seeds:** [42, 123, 456] — one DQN trained per seed
- **Eval seeds:** [789, 1024, 2048, 4096, 8192] — each trained agent evaluated on all 5
- **Total evaluation runs:** 15 per agent type (3 agents × 5 eval seeds)
- **Checkpoint files:** `models/dqn_agent_{seed}.pkl`, `models/ra_dqn_agent_{seed}.pkl`
- **Training logs:** `outputs/{exp}_seed{seed}_log.csv` (incremental, crash-proof)
- **Plots:** `outputs/{exp}_seed{seed}.png`, `outputs/h3_comparison.png`
- **Baseline MeanReversion threshold:** 5% (was 1% — never triggered trades before)

## Environment

- Four hidden macro-regimes: Stable, Bull (Bullish+Strong Bull merged), Bear (Bearish+Strong Bear merged), Chaotic.
  The underlying simulator still has 6 modes internally; labels are the 4 macro-regimes.
- Regime is not observable to any trading agent.
- Simulator exposes the regime only when `reveal=True` (used for dataset generation
  and Exp 2/3 evaluation).
- Price dynamics, transition rule, and base noise follow `../minigameMarket.js`
  (lines 780–872) verbatim, with `dragonBoost = 0`.

## Stack

- Python 3.11+
- NumPy / Pandas for the simulator
- scikit-learn for the regime classifier (LogReg, RandomForest, ExtraTrees, GradientBoosting, MLP)
- **PyTorch** for the DQN
- Matplotlib for plots
- pytest for tests

## Out of scope (for the current build)

- Multi-asset portfolios (we use a single stock for v1)
- Realistic transaction-cost models beyond a flat percentage
- Live deployment
- Comparison to commercial trading systems
