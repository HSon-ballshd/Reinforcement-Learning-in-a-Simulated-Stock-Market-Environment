# STATUS — Live status board

> **Last updated:** 2026-06-22 (initial)

## Roles

| Role | Folder | Agent |
|---|---|---|
| Sim | `sim/` | _TBD_ |
| Eval | `eval/` | _TBD_ |
| Lead | `collaboration/` | _TBD_ |

## In Progress

_(none)_

## Claimed

_(none)_

## Proposed

| ID | Title | Suggested role | Notes |
|---|---|---|---|
| T-001 | Port `CookieClickerMarket` class skeleton + `reset()` from JS lines 763–796 | Sim | Foundation; everything else depends on this. |
| T-002 | Port `tick()` per-mode dynamics from JS lines 803–877 | Sim | Depends on T-001. |
| T-003 | Implement `get_observation()` and feature engineering | Sim | Depends on T-002. |
| T-004 | Add `generate_regime_dataset()` + parquet writer | Sim | Depends on T-003. |
| T-005 | Pytest invariants: `val>=1`, `dur` countdown, mode weights | Sim | Depends on T-001. |
| T-101 | Sanity notebook: regime-recoverability probe with LogReg | Sim | Depends on T-004. |
| T-201 | `TradingEnv` wrapper + portfolio state | Eval | Depends on T-003. |
| T-202 | Heuristic baselines: Random, Buy-and-Hold, Mean-Reversion | Eval | Depends on T-201. |
| T-203 | Regime classifier training (LogReg, RF, MLP) | Eval | Depends on T-004. |
| T-204 | DQN baseline | Eval | Depends on T-201. |
| T-205 | DQN + regime | Eval | Depends on T-203, T-204. |
| T-206 | Evaluation harness + Exp 1/2/3 tables | Eval | Depends on T-202, T-204, T-205. |

## Blocked

_(none)_

## Done (this session)

_(none)_

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

_(none)_

## Concerns

_(none)_
