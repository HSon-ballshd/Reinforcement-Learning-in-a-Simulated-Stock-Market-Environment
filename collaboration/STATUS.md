# STATUS — Live status board

> **Last updated:** 2026-07-12

## Roles

| Role | Folder | Status |
|---|---|---|
| Sim | `sim/` | **Complete** (T-001 to T-005 done) |
| Eval | `eval/` | **Complete** (T-201 to T-215 done; H1/H2 PASS, H3 pending RA-DQN training) |
| Lead | `collaboration/` | **Active** — integrating and committing final results |

## In Progress

| ID | Title | Role | Notes |
|---|---|---|---|
| T-300 | H3: RA-DQN vs plain DQN comparison | Eval | Models invalidated by T-223/T-224/T-225/T-226 — retraining required |
| T-226 | Fix tx cost: BUY cost from cash before trade; MeanReversion BUY guard | Eval | Tx cost now pre-deducted from cash (can't buy without covering cost); MeanReversion only BUY when not already holding; 35 buys/34 sells vs 184 buys before |

## Proposed

_(none)_

## Blocked

_(none)_

## Done (this session)

| ID | Title | Role | Status | Notes |
|---|---|---|---|---|
| T-226 | Fix tx cost: BUY cost from cash before trade; MeanReversion BUY guard | Eval | ✓ | Tx cost pre-deducted so agent can't overspend; MeanReversion BUY-only-when-not-holding; 35 trades vs 184 before |
| T-225 | Replace Stacking with RandomForest as best classifier | Eval | ✓ | Stacking removed; RF selected (75.1%); RA-DQN eval ~10× faster |
| T-224 | Fix reward explosion: reward normalization + eval info dict | Eval | ✓ | Reward divided by initial_cash; _eval_agent/_train_ra pass real info dict |
| T-223 | Apply team-member bug fixes (off-by-one returns, tx cost, global shock, info dict) | Eval | ✓ | 7 verified fixes from teammate; models invalidated; tests 107→108 passed |

| ID | Title | Role | Status | Notes |
|---|---|---|---|---|
| T-215 | Consolidate to 4 macro-classes (Stable/Bull/Bear/Chaotic) | Eval | ✓ | Collapsed 6→4 regimes; updated classifier, RA-DQN (12-dim state), dataset, tests, harness; baseline 1/6→1/4 |
| T-216 | Fix eval consistency: H1 vs H3 DQN mismatch | Eval | ✓ | DQN determinism (cuDNN, RNG state in save/load); unique checkpoint per seed |
| T-217 | Fix H3 eval: post-episode market drift | Eval | ✓ | Both H1/H3 now use `_eval_agent` from dqn.py; agent._env injection |
| T-218 | Crash-proof training: incremental CSV logs + plot_logs.py | Eval | ✓ | Logs written every eval checkpoint; separate plot script reads CSV |
| T-219 | Episode length: 1000→500 ticks | Eval | ✓ | Reduces compounding; max ~1100% vs ~15M% |
| T-220 | Eval seeds: 2→5; train seeds: 1→3 agents | Eval | ✓ | 8 total seeds: train=[42,123,456], eval=[789,1024,2048,4096,8192] |
| T-221 | Fix MeanReversion threshold: 1%→5% | Eval | ✓ | Was never triggering trades; now a meaningful baseline |
| T-222 | Fix matplotlib/Agg backend + tqdm threading crash | Eval | ✓ | `matplotlib.use("Agg")`; tqdm disable fix for Windows |
| T-212 | Fix train/eval split + reward explosion | Eval | ✓ | Reward clipped to [-1,1], classify() feature mismatch fixed, train-seed leakage fixed in eval |
| T-213 | Redesign regime features — 18 targeted features | Eval | ✓ | 18 features from JS dynamics analysis: jump counts, trend strength, directional consistency, max tick return, vol regime |
| T-214 | 5-model ensemble + 20k dataset | Eval | ✓ | Added ExtraTrees + GradBoost; RF unpruned; MLP 256-128-64; dataset 5k→20k ticks |
| T-207 | Training progress visualizations (PNG plots) | Eval | ✓ | plot_training_run (2×2: loss/epsilon/episode-return/eval-return) + h3_comparison overlay |
| T-208 | tqdm live progress bars | Eval | ✓ | `train_dqn()` verbose tqdm bar with ε and eval_ret postfix |
| T-209 | Fix RA-DQN performance bottleneck | Eval | ✓ | Cached `next_regime` in transitions — was 64 redundant clf.predict() calls per train_step |
| T-210 | 6 new features for regime classifier | Eval | ✓ | Superseded by T-213 (redesigned feature set with 18 features) |
| T-211 | TradingEnv `_get_extended_features()` | Eval | ✓ | Exposes engineered features to classifier without expanding agent obs_dim |

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
- **D-007** (2026-07-10): Regime labels consolidated from 6 classes to 4 macro-classes:
  Stable (0), Bull (1=Bullish+StrongBull merged), Bear (2=Bearish+StrongBear merged),
  Chaotic (3). Random baseline: 1/4=25.0% (was 1/6≈16.7%). RA-DQN state dim:
  8+4=12 (was 8+6=14). (User decision — no conflict.)
- **D-006** (2026-07-10): Classifier model set expanded from LogReg+RF+MLP to 5 models:
  LogReg, RandomForest, ExtraTrees, GradientBoosting, MLP. Dataset increased from 5k to
  20k ticks. Feature set redesigned to 18 features targeting regime-discriminating
  JS dynamics. (Eval decision — no conflict.)
- **D-008** (2026-07-11): Episode length reduced from 1000 to 500 ticks to keep
  compounding returns interpretable (~1100% max vs ~15M%). Eval seeds expanded
  from 2 to 5 for statistical robustness. MeanReversion threshold increased
  from 1% to 5% so it actually trades. (User/Eval decision.)
- **D-009** (2026-07-12): Applied 7 verified bug fixes from teammate's repo:
  (1) global shock guard `dragon_boost > 0`, (2-3) off-by-one return indices in
  simulator/dataset/harness, (4-6) transaction cost deducted from cash + cash clamp
  + info dict fix, (7) baselines info dict passed to select_action. All trained
  models invalidated — H1/H2/H3 must be retrained. (Eval/Lead decision — no conflict.)
- **D-010** (2026-07-12): Reward normalization changed from `reward / curr_value` to
  `reward / initial_cash`. Old formula collapsed to ±1 in bull markets (since
  `curr_value` grows with compounding), making episode returns ~±500. New formula gives
  bounded per-tick rewards (~0.01 = 1% of initial portfolio). Also fixed `_eval_agent`
  and `_train_ra` to pass real `info` dict to `select_action`. (Eval decision.)

## Handoffs

| ID | FROM | TO | NEEDS | ACCEPT |
|---|---|---|---|---|
| H-001 | Sim | Eval | Market simulator fully tested and ready. Call `CookieClickerMarket(seed=42)` and `generate_regime_dataset()` to start. | **ACCEPTED** — Eval imports sim; TradingEnv wraps market; dataset pipeline in place. |

## Results Summary (2026-07-12 — PRELIMINARY, ALL MODELS MUST RETRAIN)

| Experiment | Verdict | Key Result |
|---|---|---|
| **H1** | PASS ✓ | DQN (543K%) > Random (432%) > BuyAndHold (14%) |
| **H2** | PASS ✓ | ExtraTrees 48.3% vs 25.0% random (1.93×) |
| **H3** | PENDING | RA-DQN training in progress |

**H3 train seeds:** [42, 123, 456]
**H3 eval seeds:** [789, 1024, 2048, 4096, 8192]
**Episode length:** 500 ticks
**Checkpoint files:** `models/dqn_agent_{seed}.pkl`, `models/ra_dqn_agent_{seed}.pkl`
**Training logs:** `outputs/{exp}_seed{seed}_log.csv`
**Plots:** `outputs/{exp}_seed{seed}.png`, `outputs/h3_comparison.png`

## Concerns

- **C-001** (resolved): H2 accuracy was stuck at ~45% on 6 classes. Resolved by T-215 (4-class consolidation).
- **C-002** (open): Cookie Clicker market compounds aggressively (~1% per tick). Reported returns are in millions of percent. Presented as multiples relative to BuyAndHold in reports.
