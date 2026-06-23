"""Generate a Jupyter notebook (.ipynb) that walks through the prototype.

Uses 'INDENT' as a sentinel for tab characters in source cells, which
survives the Write tool's whitespace stripping. We replace INDENT with
a single tab after writing.
"""
import json
from pathlib import Path

IND = "INDENT"


def md(src):
	text = src.strip("\n")
	lines = [l + "\n" for l in text.split("\n")]
	return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code(src):
	text = src.strip("\n")
	# Prepend sys.path setup so local packages resolve in any kernel cwd.
	# Jupyter launches with cwd = notebook's parent dir; for our layout that is
	# notebooks/. Step up one level to the project root.
	prelude = (
		"import sys, os" + chr(10)
		+ "_here = os.path.dirname(os.path.abspath('__file__')) if '__file__' in dir() else os.getcwd()" + chr(10)
		+ "_project_root = _here if os.path.isdir(os.path.join(_here, 'sim')) else os.path.dirname(_here)" + chr(10)
		+ "sys.path.insert(0, _project_root)" + chr(10)
		+ "os.chdir(_project_root)" + chr(10)
		+ "os.makedirs('results', exist_ok=True)" + chr(10)
		+ "os.makedirs('data', exist_ok=True)" + chr(10)
	)
	if "sys.path.insert" not in text:
		text = prelude + text
	# Replace 4-space INDENT units with the sentinel string. So 1 indent = "INDENT",
	# 2 levels = "INDENTINDENT", etc.
	out_lines = []
	for line in text.split("\n"):
		n = 0
		for c in line:
			if c == " ":
				n += 1
			else:
				break
		body = line[n:]
		if not body and n > 0:
			out_lines.append("\n")
		else:
			out_lines.append(IND * (n // 4) + body + "\n")
	return {
		"cell_type": "code",
		"execution_count": None,
		"metadata": {},
		"outputs": [],
		"source": out_lines,
	}


cells = [
	md(
		"""
# AIE301M Prototype - RL Trading + Regime Inference

Surprise working prototype on the `Prototype` branch.

Sections:
1. Sim sanity check
2. Regime dataset
3. Regime classifier
4. Trading experiment (heuristics + DQN)
5. Ablation: DQN vs DQN + regime
6. Headline verdict (H1, H2, H3)
"""
	),
	code(
		"""
import os, sys
sys.path.insert(0, os.getcwd())
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 42
N_TICKS = 800
N_EPISODES = 10
N_SEEDS = 2
N_DATASET_TICKS = 2000

np.random.seed(SEED)
print("ready")
"""
	),
	md("## 1. Simulator sanity"),
	code(
		"""
from sim.market_sim import CookieClickerMarket

market = CookieClickerMarket(seed=SEED)
prices = []
modes = []
for _ in range(1500):
INDENTmarket.tick()
INDENTprices.append(market.current_prices[0])
INDENTmodes.append(market.current_modes[0])

fig, ax = plt.subplots(figsize=(10, 3.5))
mode_colors = ['#cccccc', '#a8e6a3', '#f4a3a3', '#3aaf3a', '#c33', '#9b59b6']
for m in range(6):
INDENTmask = np.array(modes) == m
INDENTif mask.any():
INDENTINDENTax.fill_between(np.where(mask)[0], 0, max(prices), color=mode_colors[m], alpha=0.15, label=f'mode {m}')
ax.plot(prices, color='black', lw=0.8)
ax.set_title('Cookie Clicker market - 1500 ticks (background = hidden regime)')
ax.set_xlabel('tick')
ax.set_ylabel('price')
ax.legend(loc='upper right', ncol=6, fontsize=8)
fig.tight_layout()
fig.savefig('results/price_chart.png', dpi=110)
plt.show()
print('feature vector at last tick:', market.get_observation(reveal=False)[0])
"""
	),
	md("## 2. Regime dataset"),
	code(
		"""
from sim.market_sim import generate_regime_dataset

ds_path = 'data/regime_dataset.parquet'
import os
if not os.path.exists(ds_path):
INDENTgenerate_regime_dataset(n_ticks=N_DATASET_TICKS, n_stocks=1, seed=SEED, out_path=ds_path)
df = pd.read_parquet(ds_path)
print('rows:', len(df), 'columns:', list(df.columns))
print('class balance:')
print(df['mode'].value_counts().sort_index())
"""
	),
	md("## 3. Regime classifier"),
	code(
		"""
from eval.classifier.regime_classifier import (
INDENTfit_one, make_logreg, make_random_forest, make_mlp,
INDENTsave_classifier, FEATURE_COLS,
)

X = df[FEATURE_COLS].to_numpy(dtype=np.float64)
y = df['mode'].to_numpy(dtype=np.int64)
split = int(0.7 * len(X))
X_tr, y_tr = X[:split], y[:split]
X_te, y_te = X[split:], y[split:]

df_tr = pd.DataFrame(X_tr, columns=FEATURE_COLS)
df_tr['mode'] = y_tr
df_te = pd.DataFrame(X_te, columns=FEATURE_COLS)
df_te['mode'] = y_te

results = []
for name, factory in [('LogReg', make_logreg), ('RandomForest', make_random_forest), ('MLP', make_mlp)]:
INDENTm = fit_one(name, factory(), df_tr, df_te)
INDENTprint(f"{name:14s} accuracy={m.accuracy:.3f} f1={m.f1_macro:.3f}")
INDENTresults.append({'classifier': name, 'accuracy': m.accuracy, 'f1_macro': m.f1_macro})
INDENTif name == 'MLP':
INDENTINDENTsave_classifier(m, 'data/regime_classifier.joblib')
INDENTINDENTprint(' -> saved data/regime_classifier.joblib (used by DQN+Regime)')

res_df = pd.DataFrame(results)
res_df.to_csv('results/exp2_classifiers_notebook.csv', index=False)
fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(res_df['classifier'], res_df['accuracy'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.axhline(1/6, color='red', ls='--', lw=1, label='random (1/6 = 16.7%)')
ax.set_ylabel('accuracy')
ax.set_ylim(0, 0.6)
ax.set_title('Regime classifier accuracy vs random baseline')
ax.legend()
fig.tight_layout()
fig.savefig('results/exp2_accuracy.png', dpi=110)
plt.show()
"""
	),
	md("## 4. Trading experiment - Random / Buy&Hold / MeanRev / DQN"),
	code(
		"""
from eval.env.trading_env import TradingEnv
from eval.baselines.heuristics import RandomTrader, BuyAndHoldTrader, MeanReversionTrader
from eval.agents.dqn import DQNAgent, DQNConfig
from eval.agents.train import train_dqn, evaluate_dqn
from sim.market_sim import CookieClickerMarket
from eval.harness.run_experiments import _eval_heuristic


def env_factory(seed, tmax=N_TICKS):
INDENTreturn TradingEnv(market=CookieClickerMarket(seed=seed), tmax=tmax)


rows = []
curves = {}
for seed in range(N_SEEDS):
INDENTfor name, cls in [
INDENTINDENT('Random', lambda: RandomTrader(seed=seed)),
INDENTINDENT('BuyAndHold', BuyAndHoldTrader),
INDENTINDENT('MeanReversion', MeanReversionTrader),
INDENT]:
INDENTINDENTenv = env_factory(seed * 1000 + 7)
INDENTINDENTm = _eval_heuristic(env, cls())
INDENTINDENTm['seed'] = seed
INDENTINDENTm['trader'] = name
INDENTINDENTrows.append(m)
INDENTINDENTcurves[f'{name}_{seed}'] = m['equity_curve']
INDENTINDENTprint(f" seed={seed} {name:14s} return={m['return_pct']:+.2%} dd={m['max_drawdown']:.2f}")

for seed in range(N_SEEDS):
INDENTagent = DQNAgent(
INDENTINDENTDQNConfig(input_dim=11, warmup=200, eps_decay_steps=N_EPISODES*N_TICKS//2),
INDENTINDENTuse_regime=False,
INDENT)
INDENTprint(f" training DQN seed={seed} ...")
INDENTtrain_dqn(
INDENTINDENTlambda: env_factory(seed*1000+7),
INDENTINDENTagent,
INDENTINDENTn_episodes=N_EPISODES,
INDENTINDENTepisode_length=N_TICKS,
INDENTINDENTverbose=False,
INDENT)
INDENTm = evaluate_dqn(lambda: env_factory(seed*1000+7), agent, episode_length=N_TICKS)
INDENTm['seed'] = seed
INDENTm['trader'] = 'DQN'
INDENTrows.append(m)
INDENTcurves[f'DQN_{seed}'] = m['equity_curve']
INDENTprint(f" DQN seed={seed} return={m['return_pct']:+.2%} trades={m['n_trades']}")

all_df = pd.DataFrame(rows)
all_df.to_csv('results/exp1_all_notebook.csv', index=False)

fig, axes = plt.subplots(1, N_SEEDS, figsize=(6*N_SEEDS, 3.5), sharey=True)
if N_SEEDS == 1:
INDENTaxes = [axes]
for i, ax in enumerate(axes):
INDENTfor name in ['Random', 'BuyAndHold', 'MeanReversion', 'DQN']:
INDENTINDENTkey = f'{name}_{i}'
INDENTINDENTif key in curves:
INDENTINDENTINDENTax.plot(curves[key], label=name, lw=1.0)
INDENTax.set_title(f'seed {i}')
INDENTax.set_xlabel('tick')
INDENTax.legend(fontsize=7)
axes[0].set_ylabel('net worth')
fig.suptitle('Exp 1: Equity curves (heuristics + DQN)')
fig.tight_layout()
fig.savefig('results/exp1_equity_curves.png', dpi=110)
plt.show()
"""
	),
	md("## 5. Ablation - DQN vs DQN+Regime"),
	code(
		"""
from eval.classifier.regime_classifier import load_classifier

clf = load_classifier('data/regime_classifier.joblib')


def regime_fn(obs):
INDENTflat = obs.reshape(1, -1) if obs.ndim == 1 else obs.reshape(1, -1)
INDENTp = clf.pipeline.predict_proba(flat)[0]
INDENTif len(p) < 6:
INDENTINDENTpad = np.zeros(6)
INDENTINDENTpad[:len(p)] = p
INDENTINDENTp = pad
INDENTreturn p


rows = []
curves = {}
for use_regime in (False, True):
INDENTvariant = 'DQN+Regime' if use_regime else 'DQN'
INDENTin_dim = 11 + (6 if use_regime else 0)
INDENTfor seed in range(N_SEEDS):
INDENTINDENTagent = DQNAgent(
INDENTINDENTINDENTDQNConfig(input_dim=in_dim, warmup=200, eps_decay_steps=N_EPISODES*N_TICKS//2),
INDENTINDENTINDENTuse_regime=use_regime,
INDENTINDENT)
INDENTINDENTprint(f" training {variant} seed={seed} ...")
INDENTINDENTtrain_dqn(
INDENTINDENTINDENTlambda: env_factory(seed*1000+7),
INDENTINDENTINDENTagent,
INDENTINDENTINDENTn_episodes=N_EPISODES,
INDENTINDENTINDENTepisode_length=N_TICKS,
INDENTINDENTINDENTregime_prob_fn=regime_fn if use_regime else None,
INDENTINDENTINDENTverbose=False,
INDENTINDENT)
INDENTINDENTm = evaluate_dqn(
INDENTINDENTINDENTlambda: env_factory(seed*1000+7),
INDENTINDENTINDENTagent,
INDENTINDENTINDENTepisode_length=N_TICKS,
INDENTINDENTINDENTregime_prob_fn=regime_fn if use_regime else None,
INDENTINDENT)
INDENTINDENTm['seed'] = seed
INDENTINDENTm['variant'] = variant
INDENTINDENTrows.append(m)
INDENTINDENTcurves[f'{variant}_{seed}'] = m['equity_curve']
INDENTINDENTprint(f" {variant} seed={seed} return={m['return_pct']:+.2%} dd={m['max_drawdown']:.2f}")

ab = pd.DataFrame(rows)
ab.to_csv('results/exp3_ablation_notebook.csv', index=False)

fig, axes = plt.subplots(1, N_SEEDS, figsize=(6*N_SEEDS, 3.5), sharey=True)
if N_SEEDS == 1:
INDENTaxes = [axes]
for i, ax in enumerate(axes):
INDENTfor variant in ['DQN', 'DQN+Regime']:
INDENTINDENTkey = f'{variant}_{i}'
INDENTINDENTif key in curves:
INDENTINDENTINDENTax.plot(curves[key], label=variant, lw=1.2)
INDENTax.set_yscale('log')
INDENTax.set_title(f'seed {i}')
INDENTax.set_xlabel('tick')
INDENTax.legend(fontsize=8)
axes[0].set_ylabel('net worth (log)')
fig.suptitle('Exp 3: DQN vs DQN+Regime (log scale)')
fig.tight_layout()
fig.savefig('results/exp3_equity_curves.png', dpi=110)
plt.show()
"""
	),
	md("## 6. Headline"),
	code(
		"""
mean_returns = all_df.groupby('trader')['return_pct'].mean()
h1 = (mean_returns.get('DQN', 0) > mean_returns.get('Random', 0)) and \\
INDENT(mean_returns.get('DQN', 0) > mean_returns.get('BuyAndHold', 0))
print('H1 (DQN > Random, Buy&Hold):', h1)
print(mean_returns.to_string())

best_clf = res_df['accuracy'].max()
h2 = best_clf > 1.0/6
print('H2 (best classifier > 1/6 = 16.7%):', h2, 'best=', round(best_clf, 3))

mean_ab = ab.groupby('variant')['return_pct'].mean()
h3 = mean_ab.get('DQN+Regime', 0) > mean_ab.get('DQN', 0)
print('H3 (DQN+Regime > DQN):', h3)
print(mean_ab.to_string())
"""
	),
]

nb = {
	"cells": cells,
	"metadata": {
		"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
		"language_info": {"name": "python", "version": "3.12"},
	},
	"nbformat": 4,
	"nbformat_minor": 5,
}

out = Path("notebooks/prototype_demo.ipynb")
out.parent.mkdir(exist_ok=True)
# Write with INDENT sentinel, then post-process to replace with tabs.
import json as _json
raw = _json.dumps(nb, indent=1)
raw = raw.replace(IND, "\t")
# Write a list-of-strings source so we keep newlines properly.
# Replace INDENT sentinel in each line with a single space of indentation marker
# that nbformat can encode. We use the literal tab char in the source list,
# which nbformat encodes as chr(9).
import io as _io
nb_text = _json.dumps(nb, indent=1)
nb_text = nb_text.replace(IND, chr(92) + chr(116)) # escaped 	 in JSON
out.write_text(nb_text)
print(f"wrote {out}")