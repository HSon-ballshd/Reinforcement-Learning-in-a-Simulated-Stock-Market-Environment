# CONTRACT — Interface between Sim and Eval

This is the binding interface between the simulator (`sim/`) and the rest of the
system (`eval/`). **Both roles must conform exactly.** Lead approves changes.

## 1. The market object

`sim/market_sim/simulator.py` exposes a single class:

```python
class CookieClickerMarket:
    def __init__(
        self,
        n_stocks: int = 1,
        bank_level: int = 1,
        seed: int | None = None,
        seconds_per_tick: int = 60,
    ) -> None: ...

    def reset(self) -> None: ...
    def tick(self) -> None: ...
    def history(self) -> "pandas.DataFrame": ...
    def get_observation(self, *, reveal: bool = False) -> "numpy.ndarray": ...
```

### `__init__`

- `n_stocks` — number of independent stocks. **For v1 we use 1.**
- `bank_level` — replaces `Game.Objects['Bank'].level` from JS. Default 1.
- `seed` — RNG seed. If None, random.
- `seconds_per_tick` — JS uses 60. Exposed so we can compress time in experiments.

### `reset()`

Initializes all stocks. Each stock gets:
- `mode = choose([0, 1, 1, 2, 2, 3, 4, 5])`  (matches JS line 780)
- `dur = floor(10 + rand*690)`
- `val = resting_val = 10 + 10*stock_id + (bank_level - 1)`
- `d = rand*0.2 - 0.1`

### `tick()`

Advances the market by one tick. Updates `val`, `mode`, `dur`, `d` for every stock
following `minigameMarket.js` lines 813–872 verbatim, with `dragonBoost = 0`.

### `history()`

Returns a `pandas.DataFrame` with columns:

```
[tick, stock_id, price, mode, d, dur]
```

`mode` is **always included** in history (history is for offline analysis only —
Eval must not use `mode` from history as an observation).

### `get_observation(reveal=False)`

Returns a `numpy.ndarray` of shape `(n_stocks, n_features)` with the observable
features per stock.

- If `reveal=False`: features that a trader could realistically compute from
  public price information. **No `mode`, `d`, or `dur`** — those are hidden.
- If `reveal=True`: same as above, **plus** the current `mode` as a one-hot
  vector of length 6 appended at the end. Used only for dataset generation and
  Exp 2 evaluation. Eval must NOT pass `reveal=True` into the trading loop.

The default feature set (v1, subject to change via handoff):

```
[price,
 return_1, return_5, return_20,
 rolling_mean_5, rolling_mean_20,
 rolling_std_20,
 momentum_5_20]
```

Shape: `(n_stocks, 8)` when `reveal=False`, `(n_stocks, 14)` when `reveal=True`.

## 2. Dataset generation (Sim → Eval)

`sim/market_sim/dataset.py` exposes:

```python
def generate_regime_dataset(
    n_ticks: int = 5000,
    n_stocks: int = 1,
    seed: int = 0,
    out_path: str | "Path" = "data/regime_dataset.parquet",
) -> "Path":
    """Run the simulator for n_ticks and dump (X, y) pairs."""
```

Output format: a single parquet file with columns:

```
[tick, stock_id, price, return_1, return_5, return_20,
 rolling_mean_5, rolling_mean_20, rolling_std_20, momentum_5_20,
 mode]            # mode is the label y
```

Eval reads this file. Eval **must not** call `get_observation(reveal=True)`
during training; the dataset file is the only allowed source of labels.

## 3. Trading environment (Eval → Sim is read-only)

`eval/env/trading_env.py` exposes:

```python
class TradingEnv:
    def __init__(self, market: "CookieClickerMarket", ...): ...
    def reset(self) -> "numpy.ndarray": ...     # initial observation
    def step(self, action: int) -> tuple["numpy.ndarray", float, bool, dict]: ...
    def render(self) -> None: ...
```

The env wraps a `CookieClickerMarket` and adds a portfolio state
(cash, holdings, position fraction). Actions:

```
0 = hold
1 = buy  (target position fraction increases)
2 = sell (target position fraction decreases)
```

Reward: change in portfolio net worth, minus a flat transaction-cost penalty.
Eval defines the exact reward; the contract only requires that the env returns
`(obs, reward, done, info)` like a Gym environment.

## 4. Versioning

This contract is **v1**. Breaking changes require:

1. Proposal in STATUS.md under "Disputes."
2. Vote per PROTOCOL.md §7.
3. Bump version to v2 at the top of this file.
4. Both roles update in the same commit, coordinated by Lead.

## 5. What is NOT in this contract

- The exact DQN architecture (Eval's call, within reason).
- The exact reward shaping (Eval's call).
- The transaction-cost percentage (Eval's call, but must be reported in
  experiments).
- Plot styling (Eval's call).
