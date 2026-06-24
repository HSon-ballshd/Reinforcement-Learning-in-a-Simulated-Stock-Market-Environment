# STATUS — Live status board

> **Last updated:** 2026-06-24 (T-001 to T-005 complete by Sim; Awaiting Eval)

## Roles

| Role | Folder | Agent |
|---|---|---|
| Sim | `sim/` | **Copilot (complete with T-001 to T-005)** |
| Eval | `eval/` | _TBD_ (next phase) |
| Lead | `collaboration/` | _TBD_ |

## In Progress

_(none)_

## Claimed

_(none)_

## Proposed

| ID | Title | Suggested role | Notes |
|---|---|---|---|
| T-101 | Sanity notebook: regime-recoverability probe with LogReg | Sim | Optional next phase. Regime classifier ready to test. |
| T-201 | `TradingEnv` wrapper + portfolio state | Eval | READY - simulator fully functional with get_observation(). |
| T-202 | Heuristic baselines: Random, Buy-and-Hold, Mean-Reversion | Eval | Depends on T-201. |
| T-203 | Regime classifier training (LogReg, RF, MLP) | Eval | READY - dataset generation complete. Can train classifiers now. |
| T-204 | DQN baseline | Eval | Depends on T-201. |
| T-205 | DQN + regime | Eval | Depends on T-203, T-204. |
| T-206 | Evaluation harness + Exp 1/2/3 tables | Eval | Depends on T-202, T-204, T-205. |

## Blocked

_(none)_

## Done (this session)

| ID | Title | Role | Status | Notes |
|---|---|---|---|---|
| T-001 | Port `CookieClickerMarket` class skeleton + `reset()` from JS lines 763–796 | Sim | ✓ Complete | Market initialization working, deterministic with seed |
| T-002 | Port `tick()` per-mode dynamics from JS lines 803–877 | Sim | ✓ Complete | Full mode-specific dynamics, all 6 modes working |
| T-003 | Implement `get_observation()` and feature engineering | Sim | ✓ Complete | 8 observable features + optional 6-hot mode reveal |
| T-004 | Add `generate_regime_dataset()` + parquet writer | Sim | ✓ Complete | Generates labeled (X, y) pairs for classifier training |
| T-005 | Pytest invariants: `val>=1`, `dur` countdown, mode weights | Sim | ✓ Complete | 12/12 critical tests pass over long simulations |

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

## Disputes

_(none)_

## Handoffs

| ID | FROM | TO | NEEDS | ACCEPT |
|---|---|---|---|---|
| H-001 | Sim | Eval | Market simulator fully tested and ready. Call `CookieClickerMarket(seed=42)` and `generate_regime_dataset()` to start. | Eval imports sim module without errors; can generate dataset and wrap in `TradingEnv`. |

## Concerns

_(none)_
