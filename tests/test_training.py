"""Tests for the ML training pipeline.

The committed model artifact is only trustworthy if it can be
regenerated, so these exercise the trainer end to end on a small
synthetic dataset and assert the result is loadable and servable.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from ml.train_enhanced import EnhancedCMLModelTrainer


@pytest.fixture
def training_csv(tmp_path):
    """A small but learnable dataset with both classes represented."""
    rng = np.random.default_rng(0)
    rows = 120
    corrosion = rng.uniform(0.02, 0.4, rows)
    thickness = rng.uniform(4.0, 14.0, rows)
    frame = pd.DataFrame(
        {
            "id_number": [f"CML-{i:03d}" for i in range(rows)],
            "average_corrosion_rate": corrosion,
            "thickness_mm": thickness,
            "commodity": rng.choice(["Crude Oil", "Natural Gas", "Steam"], rows),
            "feature_type": rng.choice(["Pipe", "Elbow", "Tee"], rows),
            "cml_shape": rng.choice(["Both", "Internal", "External"], rows),
            "last_inspection_date": ["2023-06-15"] * rows,
            "risk_score": rng.integers(0, 100, rows),
            "remaining_life_years": (thickness - 3.0) / corrosion,
        }
    )
    # A learnable rule: thick walls corroding slowly are elimination candidates.
    frame["elimination_flag"] = (
        (frame["average_corrosion_rate"] < 0.1) & (frame["thickness_mm"] > 9.0)
    ).astype(int)

    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)
    return path


class TestDataLoading:
    def test_loads_a_valid_dataset(self, training_csv, tmp_path):
        trainer = EnhancedCMLModelTrainer(training_csv, tmp_path / "models")
        assert len(trainer.load_data()) == 120

    def test_rejects_a_dataset_missing_the_target(self, tmp_path):
        path = tmp_path / "bad.csv"
        pd.DataFrame({"average_corrosion_rate": [0.1]}).to_csv(path, index=False)
        trainer = EnhancedCMLModelTrainer(path, tmp_path / "models")
        with pytest.raises(ValueError, match="Missing required columns"):
            trainer.load_data()


class TestFeaturePreparation:
    def test_engineered_columns_are_added(self, training_csv, tmp_path):
        trainer = EnhancedCMLModelTrainer(training_csv, tmp_path / "models")
        engineered = trainer.engineer_features(trainer.load_data())
        for column in ("corrosion_thickness_ratio", "days_since_inspection"):
            assert column in engineered.columns

    def test_target_is_separated_from_the_features(self, training_csv, tmp_path):
        trainer = EnhancedCMLModelTrainer(training_csv, tmp_path / "models")
        engineered = trainer.engineer_features(trainer.load_data())
        X, y, _ = trainer.prepare_features(engineered)
        assert "elimination_flag" not in X.columns
        assert len(X) == len(y)


@pytest.mark.slow
class TestEndToEndTraining:
    def test_produces_a_loadable_and_servable_model(self, training_csv, tmp_path):
        output = tmp_path / "models"
        trainer = EnhancedCMLModelTrainer(training_csv, output)

        engineered = trainer.engineer_features(trainer.load_data())
        X, y, preprocessor = trainer.prepare_features(engineered)
        # A single-point grid keeps the test fast; the search itself is
        # exercised by the CV plumbing, not by the size of the grid.
        metrics = trainer.train_model(X, y, preprocessor, param_grid={})
        model_path = trainer.save_model(metrics)

        assert model_path.exists()
        assert (output / "cml_elimination_model.joblib").exists()

        reloaded = joblib.load(output / "cml_elimination_model.joblib")
        predictions = reloaded.predict(X.head(5))
        assert set(predictions).issubset({0, 1})

        metadata = json.loads(next(output.glob("model_metadata_*.json")).read_text())
        assert metadata["feature_names"]
        assert 0.0 <= metadata["metrics"]["roc_auc"] <= 1.0

    def test_learns_the_signal(self, training_csv, tmp_path):
        """A model that cannot beat chance on a separable rule is broken."""
        trainer = EnhancedCMLModelTrainer(training_csv, tmp_path / "models")
        engineered = trainer.engineer_features(trainer.load_data())
        X, y, preprocessor = trainer.prepare_features(engineered)

        metrics = trainer.train_model(X, y, preprocessor, param_grid={})

        assert metrics["roc_auc"] > 0.8


class TestCalibration:
    """Calibration is available but off by default.

    It is off because it was measured and did not pay for itself on the
    bundled data -- see docs/MODEL_CARD.md. These tests pin the plumbing
    so the option stays usable when a larger dataset makes it worthwhile.
    """

    def test_off_by_default(self, training_csv, tmp_path):
        assert EnhancedCMLModelTrainer(training_csv, tmp_path / "models").calibration is None

    @pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
    def test_accepts_the_supported_methods(self, training_csv, tmp_path, method):
        trainer = EnhancedCMLModelTrainer(training_csv, tmp_path / "models", calibration=method)
        assert trainer.calibration == method

    def test_rejects_an_unknown_method(self, training_csv, tmp_path):
        with pytest.raises(ValueError, match="Unknown calibration method"):
            EnhancedCMLModelTrainer(training_csv, tmp_path / "models", calibration="magic")

    @pytest.mark.slow
    def test_a_calibrated_model_still_predicts(self, training_csv, tmp_path):
        trainer = EnhancedCMLModelTrainer(training_csv, tmp_path / "models", calibration="sigmoid")
        engineered = trainer.engineer_features(trainer.load_data())
        X, y, preprocessor = trainer.prepare_features(engineered)

        metrics = trainer.train_model(X, y, preprocessor, param_grid={})

        assert metrics["calibration"] == "sigmoid"
        assert set(trainer.model.predict(X.head(5))).issubset({0, 1})
        probabilities = trainer.model.predict_proba(X.head(5))[:, 1]
        assert ((probabilities >= 0) & (probabilities <= 1)).all()

    @pytest.mark.slow
    def test_calibration_quality_is_always_measured(self, training_csv, tmp_path):
        """An uncalibrated model must still report how uncalibrated it is."""
        trainer = EnhancedCMLModelTrainer(training_csv, tmp_path / "models")
        engineered = trainer.engineer_features(trainer.load_data())
        X, y, preprocessor = trainer.prepare_features(engineered)

        metrics = trainer.train_model(X, y, preprocessor, param_grid={})

        assert 0.0 <= metrics["brier_score"] <= 1.0
        assert 0.0 <= metrics["expected_calibration_error"] <= 1.0
        assert metrics["log_loss"] > 0
        assert metrics["calibration"] is None


class TestExpectedCalibrationError:
    def test_perfect_calibration_scores_zero(self):
        from ml.train_enhanced import _expected_calibration_error

        # Half the samples at p=0.0 are negative, half at p=1.0 positive.
        y_true = np.array([0, 0, 1, 1])
        probabilities = np.array([0.0, 0.0, 1.0, 1.0])
        assert _expected_calibration_error(y_true, probabilities) == pytest.approx(0.0)

    def test_confident_and_wrong_scores_one(self):
        from ml.train_enhanced import _expected_calibration_error

        y_true = np.array([0, 0, 0, 0])
        probabilities = np.array([1.0, 1.0, 1.0, 1.0])
        assert _expected_calibration_error(y_true, probabilities) == pytest.approx(1.0)
