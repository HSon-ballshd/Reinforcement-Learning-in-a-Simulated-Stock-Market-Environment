# AIE301M — Reinforcement Learning in a Simulated Stock Market

End-to-end prototype: Cookie Clicker stock-market simulator + regime classifier
+ DQN trader + heuristic baselines.

> **Status:** This branch (`Prototype`) is a surprise working prototype that
> follows the interface contract in [`collaboration/CONTRACT.md`](collaboration/CONTRACT.md).
> It is meant to be merged or replaced by the Sim / Eval / Lead agents.

## Quick start

```bash
pip install -r requirements.txt

# Run all tests (~30 s)
pytest tests/ -q

# Walk through the demo notebook (~5-10 min on CPU)
jupyter nbconvert --to notebook --execute notebooks/prototype_demo.ipynb --inplace
```

Results (PNGs + CSV tables) land in `results/`. The regime dataset (parquet) and
the trained classifier land in `data/`.

## What's here

```
sim/ Market simulator (matches CONTRACT.md §1, §2)
 market_sim/simulator.py class CookieClickerMarket
 market_sim/dataset.py generate_regime_dataset()

eval/ Trading environment + agents + harness
 env/trading_env.py class TradingEnv (matches CONTRACT.md §3)
 baselines/heuristics.py Random, Buy-and-Hold, Mean-Reversion traders
 classifier/regime_classifier.py LogReg / RandomForest / MLP
 agents/dqn.py DQNAgent (baseline + regime-augmented)
 agents/train.py train_dqn()
 harness/run_experiments.py Exp 1 / Exp 2 / Exp 3

tests/ pytest (test_sim.py, test_eval.py)
notebooks/prototype_demo.ipynb End-to-end walkthrough
data/, results/ Generated artifacts (gitignored)
```

## Hypotheses

- **H1** — DQN beats Random and Buy-and-Hold on mean return.
- **H2** — Regime classifier beats 1/6 ≈ 16.7% random guessing.
- **H3** — DQN + regime beats plain DQN.

See the notebook's final section for the headline verdict on a single seed.
For multi-seed numbers, run `python -m eval.harness.run_experiments`.

## See also

- [Aie Project Overview.md](Aie%20Project%20Overview.md) — human-authored spec.
- [collaboration/](collaboration/) — PROTOCOL, SPEC, CONTRACT, STATUS for the
 three-agent (Sim / Eval / Lead) workflow.