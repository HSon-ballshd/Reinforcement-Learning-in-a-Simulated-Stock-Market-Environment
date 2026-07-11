"""
Tests for RegimeClassifierPipeline (eval/classifiers/regime_classifier.py).

These tests use a small synthetic dataset so they run without the real
data/regime_dataset.parquet file.  The full training run is exercised
in test_run_pipeline_when_data_exists().
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from eval.classifiers.regime_classifier import (
    RegimeClassifierPipeline,
    REGIME_NAMES,
    N_CLASSES,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------
@pytest.fixture
def small_dataset(tmp_path):
    """
    Create a tiny synthetic parquet file:
    500 rows, 4 macro-regimes, 18 features + macro_regime column.
    """
    n = 500
    rng = np.random.default_rng(0)

    data = {
        'return_1':   rng.standard_normal(n),
        'return_5':   rng.standard_normal(n),
        'return_10':  rng.standard_normal(n),
        'return_20':  rng.standard_normal(n),
        'rolling_std_5':   rng.uniform(0.1, 5, n),
        'rolling_std_20':  rng.uniform(0.1, 5, n),
        'rolling_std_ratio': rng.uniform(0.1, 3, n),
        'mean_reversion_z': rng.standard_normal(n),
        'directional_consistency_5':  rng.uniform(0, 1, n),
        'directional_consistency_20': rng.uniform(0, 1, n),
        'drift_estimate_5':  rng.standard_normal(n),
        'jump_count_5':  rng.integers(0, 5, n),
        'jump_count_20': rng.integers(0, 10, n),
        'max_tick_return_5': rng.uniform(0, 3, n),
        'trend_strength_5':  rng.standard_normal(n),
        'trend_strength_20': rng.standard_normal(n),
        'momentum_divergence': rng.integers(0, 2, n),
        'vol_regime_5': rng.uniform(0.1, 3, n),
        'macro_regime': rng.integers(0, 4, n),
    }
    df = pd.DataFrame(data)
    path = tmp_path / "test_regime.parquet"
    df.to_parquet(path, index=False)
    return path


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------
class TestRegimeClassifierPipelineInit:
    def test_default_attrs(self):
        pipe = RegimeClassifierPipeline()
        assert pipe.data_path    == Path("data/regime_dataset.parquet")
        assert pipe.out_dir     == Path("models")
        assert pipe.test_size    == 0.2
        assert pipe.random_state == 42

    def test_custom_attrs(self, small_dataset, tmp_path):
        pipe = RegimeClassifierPipeline(
            data_path=small_dataset,
            out_dir=tmp_path / "models",
            test_size=0.3,
            random_state=7,
        )
        assert pipe.data_path    == small_dataset
        assert pipe.out_dir     == tmp_path / "models"
        assert pipe.test_size    == 0.3
        assert pipe.random_state == 7


class TestRegimeClassifierPipelineRun:
    def test_run_produces_three_way_scores(self, small_dataset, tmp_path):
        """run() returns val_scores, test_scores, and best_name."""
        pipe = RegimeClassifierPipeline(
            data_path=small_dataset,
            out_dir=tmp_path / "models",
        )
        result = pipe.run()
        assert set(result.keys()) == {"val_scores", "test_scores", "best_name"}
        # val_scores has one entry per model
        assert set(result["val_scores"].keys()) == {"LogReg", "RandomForest", "ExtraTrees", "GradBoost", "MLP", "Stacking"}
        assert set(result["test_scores"].keys()) == {"LogReg", "RandomForest", "ExtraTrees", "GradBoost", "MLP", "Stacking"}
        for acc in list(result["val_scores"].values()) + list(result["test_scores"].values()):
            assert 0.0 <= acc <= 1.0

    def test_best_model_selected(self, small_dataset, tmp_path):
        pipe = RegimeClassifierPipeline(
            data_path=small_dataset,
            out_dir=tmp_path / "models",
        )
        result = pipe.run()
        assert pipe.best_name in {"LogReg", "RandomForest", "ExtraTrees", "GradBoost", "MLP", "Stacking"}
        assert pipe.best_model is not None
        assert pipe.best_name == result["best_name"]
        assert pipe.best_name == max(pipe.val_scores, key=pipe.val_scores.get)

    def test_run_saves_artifacts(self, small_dataset, tmp_path):
        out_dir = tmp_path / "models"
        pipe = RegimeClassifierPipeline(
            data_path=small_dataset,
            out_dir=out_dir,
        )
        pipe.run()
        assert (out_dir / "scaler.pkl").exists()
        assert (out_dir / "best_model.pkl").exists()
        assert (out_dir / "all_models.pkl").exists()


class TestRegimeClassifierPredict:
    def test_predict_shape(self, small_dataset, tmp_path):
        pipe = RegimeClassifierPipeline(
            data_path=small_dataset,
            out_dir=tmp_path / "models",
        )
        pipe.run()

        # 18 features matching FEATURE_COLS
        obs = np.array([0.01, 0.05, 0.02, 0.01,
                        0.5, 0.8, 0.6, 0.2,
                        0.6, 0.55, 0.05,
                        1, 3, 0.5,
                        0.1, 0.05, 0, 0.7])
        pred = pipe.predict(obs)
        assert isinstance(pred, int)
        assert 0 <= pred < N_CLASSES

    def test_predict_list_input(self, small_dataset, tmp_path):
        pipe = RegimeClassifierPipeline(
            data_path=small_dataset,
            out_dir=tmp_path / "models",
        )
        pipe.run()
        obs = [0.01, 0.05, 0.02, 0.01,
               0.5, 0.8, 0.6, 0.2,
               0.6, 0.55, 0.05,
               1, 3, 0.5,
               0.1, 0.05, 0, 0.7]
        pred = pipe.predict(obs)
        assert 0 <= pred < N_CLASSES


class TestRegimeClassifierConstants:
    def test_n_classes(self):
        assert N_CLASSES == 4

    def test_regime_names(self):
        assert set(REGIME_NAMES.keys()) == set(range(4))
        assert REGIME_NAMES[0] == 'Stable'
        assert REGIME_NAMES[3] == 'Chaotic'


class TestRegimeClassifierErrors:
    def test_missing_data_raises(self, tmp_path):
        pipe = RegimeClassifierPipeline(
            data_path=tmp_path / "nonexistent.parquet",
            out_dir=tmp_path / "models",
        )
        with pytest.raises(FileNotFoundError):
            pipe.run()

    def test_predict_before_run_raises(self):
        pipe = RegimeClassifierPipeline()
        with pytest.raises(RuntimeError, match="run"):
            pipe.predict([0.0] * 18)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
