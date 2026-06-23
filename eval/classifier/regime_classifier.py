"""Regime classifier (LogReg / RandomForest / MLP) on observable features.

Per CONTRACT.md Sec.8: the classifier predicts the hidden market mode from
observable price features. Trained on the parquet dataset produced by
``sim.market_sim.dataset.generate_regime_dataset``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
 "price",
 "return_1",
 "return_5",
 "return_20",
 "rolling_mean_5",
 "rolling_mean_20",
 "rolling_std_20",
 "momentum_5_20",
]
N_CLASSES = 6


def make_logreg() -> Pipeline:
 return Pipeline([
  ("scaler", StandardScaler()),
  ("clf", LogisticRegression(max_iter=1000, n_jobs=1)),
 ])


def make_random_forest(n_estimators: int = 100) -> Pipeline:
 return Pipeline([
  ("scaler", StandardScaler()),
  ("clf", RandomForestClassifier(n_estimators=n_estimators, n_jobs=1, random_state=0)),
 ])


def make_mlp(hidden: tuple = (64, 32)) -> Pipeline:
 return Pipeline([
  ("scaler", StandardScaler()),
  ("clf", MLPClassifier(hidden_layer_sizes=hidden, max_iter=50, random_state=0)),
 ])


def _split_xy(df: pd.DataFrame) -> tuple:
 X = df[FEATURE_COLS].to_numpy(dtype=np.float64)
 y = df["mode"].to_numpy(dtype=np.int64)
 return X, y


@dataclass
class TrainedClassifier:
 name: str
 pipeline: Pipeline
 accuracy: float
 f1_macro: float

 def predict_proba(self, X: np.ndarray) -> np.ndarray:
  return self.pipeline.predict_proba(X)


def fit_one(
 name: str,
 pipeline: Pipeline,
 df_train: pd.DataFrame,
 df_test: Optional[pd.DataFrame] = None,
) -> TrainedClassifier:
 from sklearn.metrics import accuracy_score, f1_score

 X_train, y_train = _split_xy(df_train)
 pipeline.fit(X_train, y_train)
 if df_test is not None:
  X_test, y_test = _split_xy(df_test)
  y_pred = pipeline.predict(X_test)
  acc = float(accuracy_score(y_test, y_pred))
  f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
 else:
  acc = float(accuracy_score(y_train, pipeline.predict(X_train)))
  f1 = float(f1_score(y_train, pipeline.predict(X_train), average="macro", zero_division=0))
 return TrainedClassifier(name=name, pipeline=pipeline, accuracy=acc, f1_macro=f1)


def load_dataset(parquet_path: Union[str, Path]) -> pd.DataFrame:
 return pd.read_parquet(parquet_path)


def save_classifier(model: TrainedClassifier, out_path: Union[str, Path]) -> Path:
 out_path = Path(out_path)
 out_path.parent.mkdir(parents=True, exist_ok=True)
 joblib.dump({"name": model.name, "pipeline": model.pipeline}, out_path)
 return out_path


def load_classifier(path: Union[str, Path]) -> TrainedClassifier:
 blob = joblib.load(path)
 return TrainedClassifier(
  name=blob["name"],
  pipeline=blob["pipeline"],
  accuracy=-1.0,
  f1_macro=-1.0,
 )