"""
Tests for RegimeClassifierPipeline (eval/classifiers/regime_classifier.py).

These tests use a small synthetic dataset so they run without the real
data/regime_dataset.parquet file.  The full training run is exercised
in test_run_pipeline_when_data_exists().
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
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
    500 rows, 6 regimes, 7 features + mode column.
    """
    n = 500
    rng = np.random.default_rng(0)

    data = {
        'return_1':       rng.standard_normal(n),
        'return_5':       rng.standard_normal(n),
        'return_20':      rng.standard_normal(n),
        'rolling_mean_5': rng.uniform(5, 20, n),
        'rolling_mean_20': rng.uniform(5, 20, n),
        'rolling_std_20': rng.uniform(0.1, 5, n),
        'momentum_5_20': rng.standard_normal(n),
        'mode':          rng.integers(0, 6, n),
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
    def test_run_produces_scores(self, small_dataset, tmp_path):
        pipe = RegimeClassifierPipeline(
            data_path=small_dataset,
            out_dir=tmp_path / "models",
        )
        scores = pipe.run()
        assert isinstance(scores, dict)
        assert set(scores.keys()) == {"LogReg", "RandomForest", "MLP"}
        for acc in scores.values():
            assert 0.0 <= acc <= 1.0

    def test_best_model_selected(self, small_dataset, tmp_path):
        pipe = RegimeClassifierPipeline(
            data_path=small_dataset,
            out_dir=tmp_path / "models",
        )
        pipe.run()
        assert pipe.best_name in {"LogReg", "RandomForest", "MLP"}
        assert pipe.best_model is not None
        assert pipe.best_name == max(pipe.scores, key=pipe.scores.get)

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

        obs = np.array([0.01, 0.05, 0.02, 10.0, 10.5, 0.5, 0.1])
        pred = pipe.predict(obs)
        assert isinstance(pred, int)
        assert 0 <= pred < N_CLASSES

    def test_predict_list_input(self, small_dataset, tmp_path):
        pipe = RegimeClassifierPipeline(
            data_path=small_dataset,
            out_dir=tmp_path / "models",
        )
        pipe.run()
        pred = pipe.predict([0.01, 0.05, 0.02, 10.0, 10.5, 0.5, 0.1])
        assert 0 <= pred < N_CLASSES


class TestRegimeClassifierConstants:
    def test_n_classes(self):
        assert N_CLASSES == 6

    def test_regime_names(self):
        assert set(REGIME_NAMES.keys()) == set(range(6))
        assert REGIME_NAMES[0] == 'Stable'
        assert REGIME_NAMES[5] == 'Chaotic'


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
            pipe.predict([0.0] * 7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
