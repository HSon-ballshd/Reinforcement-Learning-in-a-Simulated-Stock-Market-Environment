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
- **H2** — Regime classifier beats random guessing (1/6 ≈ 16.7%) by a meaningful
  margin.
- **H3** — DQN + regime beats plain DQN. **This is the academic contribution.**

## Environment

- Six hidden regimes: Stable, Bullish, Bearish, Strong Bull, Strong Bear, Chaotic.
- Regime is not observable to any trading agent.
- Simulator exposes the regime only when `reveal=True` (used for dataset generation
  and Exp 2/3 evaluation).
- Price dynamics, transition rule, and base noise follow `../minigameMarket.js`
  (lines 780–872) verbatim, with `dragonBoost = 0`.

## Stack

- Python 3.11+
- NumPy / Pandas for the simulator
- scikit-learn for the regime classifier (LogReg, RandomForest, MLP)
- **PyTorch** for the DQN
- Matplotlib for plots
- pytest for tests

## Out of scope (for the current build)

- Multi-asset portfolios (we use a single stock for v1)
- Realistic transaction-cost models beyond a flat percentage
- Live deployment
- Comparison to commercial trading systems
