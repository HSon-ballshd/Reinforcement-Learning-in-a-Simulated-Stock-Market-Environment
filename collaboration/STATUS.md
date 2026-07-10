# STATUS — Live status board

> **Last updated:** 2026-07-10

## Roles

| Role | Folder | Agent |
|---|---|---|
| Sim | `sim/` | **Copilot (complete with T-001 to T-005)** |
| Eval | `eval/` | **Lead (acting as Eval — T-201 to T-212 ALL COMPLETE)** |
| Lead | `collaboration/` | _TBD_ |

> Note: No separate Eval agent materialised. Lead has been acting as Eval since
> T-201. All eval tasks remain Eval role in this table.

## In Progress

_(none)_

## Claimed

_(none)_

## Proposed

| ID | Title | Suggested role | Notes |
|---|---|---|---|
| T-101 | Sanity notebook: regime-recoverability probe with LogReg | Sim | Optional. Classifier and baselines ready to compare. |

## Blocked

_(none)_

## Done (this session)

| ID | Title | Role | Status | Commit | Notes |
|---|---|---|---|---|---|
| T-212 | Fix train/eval split + reward explosion | Eval | ✓ | 5e39dc2 | Reward clipped to [-1,1], classify() feature mismatch fixed, train-seed leakage fixed in eval |
| T-213 | Redesign regime features — 18 targeted features | Eval | ✓ | a125f41 | 18 features from JS dynamics analysis: jump counts, trend strength, directional consistency, max tick return, vol regime |
| T-214 | 5-model ensemble + 20k dataset | Eval | ✓ | a5f9039 | Added ExtraTrees + GradBoost; RF unpruned; MLP 256-128-64; dataset 5k→20k ticks |
| T-207 | Training progress visualizations (PNG plots) | Eval | ✓ | 3c99813 | plot_training_run (2×2: loss/epsilon/episode-return/eval-return) + h3_comparison overlay |
| T-208 | tqdm live progress bars | Eval | ✓ | 141530b | `train_dqn()` verbose tqdm bar with ε and eval_ret postfix |
| T-209 | Fix RA-DQN performance bottleneck | Eval | ✓ | 872e608 | Cached `next_regime` in transitions — was 64 redundant clf.predict() calls per train_step |
| T-210 | 6 new features for regime classifier | Eval | ✓ | 5f8bd14 | Superseded by T-213 (redesigned feature set with 18 features) |
| T-211 | TradingEnv `_get_extended_features()` | Eval | ✓ | 5f8bd14 | Exposes engineered features to classifier without expanding agent obs_dim |

## Done (all time)

| ID | Title | Role | Status | Notes |
|---|---|---|---|---|
| T-201 | `TradingEnv` wrapper + portfolio state | Eval | ✓ | 11/11 tests pass; Gym-style (reset/step/render), 3 actions, transaction-cost reward |
| T-202 | Heuristic baselines: Random, Buy-and-Hold, Mean-Reversion | Eval | ✓ | 18/18 tests pass; evaluate_agent(), compare_baselines() |
| T-203 | Regime classifier training (LogReg, RF, MLP) | Eval | ✓ | 11/11 tests pass; full pipeline with scaling, scoring, pickle save |
| T-204 | DQN baseline | Eval | ✓ | 15/15 tests pass; Q-network, replay buffer, epsilon-greedy, save/load |
| T-205 | DQN + regime | Eval | ✓ | 15/15 tests pass; regime-aware Q-net (14-dim), classifier injection |
| T-206 | Evaluation harness + Exp 1/2/3 tables | Eval | ✓ | 4/4 tests pass; run_all() with H1/H2/H3 + JSON output |
| T-001 | `CookieClickerMarket` class skeleton + `reset()` from JS lines 763–796 | Sim | ✓ | Market initialization working, deterministic with seed |
| T-002 | Port `tick()` per-mode dynamics from JS lines 803–877 | Sim | ✓ | Full mode-specific dynamics, all 6 modes working |
| T-003 | Implement `get_observation()` and feature engineering | Sim | ✓ | 8 observable features + optional 6-hot mode reveal |
| T-004 | Add `generate_regime_dataset()` + parquet writer | Sim | ✓ | Generates labeled (X, y) pairs for classifier training |
| T-005 | Pytest invariants: `val>=1`, `dur` countdown, mode weights | Sim | ✓ | 12/12 critical tests pass over long simulations |

## Decisions log

- **D-001** (2026-06-22): Tech stack is **PyTorch** for DQN, scikit-learn for
  classifiers, NumPy/Pandas for the simulator. (User confirmation.)
- **D-002** (2026-06-22): Simulator follows `minigameMarket.js` line-for-line,
  with `dragonBoost = 0`. (User confirmation: "closely follow the market of
  cookie clicker.")
- **D-003** (2026-06-22): Build scope is **simulator first, then pause** before
  classifiers/DQN. (User confirmation.)
- **D-004** (2026-06-22): Three-agent split: **Sim / Eval / Lead**, with
  Lead owning only `collaboration/`. Exclusive folder ownership per role.
  Conflict policy: 2-of-3 vote; ties → human.
- **D-005** (2026-06-22): Regime-switching rule is the JS rule verbatim with
  `dragonBoost = 0`, weights `[0,1,1,2,2,3,4,5]`. (Lead proposal — Sim to
  confirm via handoff when porting.)
- **D-006** (2026-07-10): Classifier model set expanded from LogReg+RF+MLP to 5 models:
  LogReg, RandomForest, ExtraTrees, GradientBoosting, MLP. Dataset increased from 5k to
  20k ticks. Feature set redesigned to 18 features targeting regime-discriminating
  JS dynamics. (Eval decision — no conflict.)

## Disputes

_(none)_

## Handoffs

| ID | FROM | TO | NEEDS | ACCEPT |
|---|---|---|---|---|
| H-001 | Sim | Eval | Market simulator fully tested and ready. Call `CookieClickerMarket(seed=42)` and `generate_regime_dataset()` to start. | **ACCEPTED** — Eval imports sim; TradingEnv wraps market; dataset pipeline in place. |

## Concerns

- **C-001** (2026-07-10, updated 2026-07-10): H2 accuracy was 47–49% with first feature set. T-213 redesigned features (18 total, targeting jump counts, trend strength, directional consistency, max tick return). T-214 added ExtraTrees + GradBoost and increased dataset to 20k ticks. **Must regenerate `data/regime_dataset.parquet`** to pick up new features. Run `python -m eval.eval_harness --h2` to verify.
