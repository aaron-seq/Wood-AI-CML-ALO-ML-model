"""Enhanced CML Elimination Model Training Pipeline."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Allow running this file directly as `python ml/train_enhanced.py`.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.features import engineer_features  # noqa: E402


class EnhancedCMLModelTrainer:
    """Train and evaluate CML elimination prediction model with advanced features."""

    def __init__(
        self,
        data_path: str,
        model_output_dir: str = "models",
        calibration: str | None = None,
    ):
        """
        Args:
            data_path: CSV of labelled CML records.
            model_output_dir: Where artifacts and metadata are written.
            calibration: ``None`` (default), ``"sigmoid"`` or
                ``"isotonic"``. Wraps the classifier in
                CalibratedClassifierCV so predict_proba can be read as a
                probability. Off by default because on the bundled
                datasets it does not pay for itself -- see
                docs/MODEL_CARD.md for the measurement.
        """
        if calibration not in (None, "sigmoid", "isotonic"):
            raise ValueError(
                f"Unknown calibration method {calibration!r}; use None, 'sigmoid' or 'isotonic'."
            )
        self.calibration = calibration
        self.data_path = Path(data_path)
        self.model_output_dir = Path(model_output_dir)
        self.model_output_dir.mkdir(exist_ok=True)
        self.model: Pipeline | None = None
        self.feature_names: list[str] | None = None

    def load_data(self) -> pd.DataFrame:
        """Load and validate CML data."""
        print(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)

        required_cols = [
            "average_corrosion_rate",
            "thickness_mm",
            "commodity",
            "feature_type",
            "cml_shape",
            "elimination_flag",
        ]
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        print(f"Loaded {len(df)} records")
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create the derived features the model consumes.

        Delegates to app.features so training and serving compute these
        columns with exactly the same code. Previously each side had its
        own copy of the formulas, and only the serving copy guarded
        against division by zero.

        The earlier version also derived high_corrosion, thin_wall,
        risk_interaction and low_remaining_life. None of them were listed
        in prepare_features, so they were computed and discarded on every
        run; they have been removed rather than wired in, because adding
        them would change the model's inputs.
        """
        print("Engineering features...")
        return engineer_features(df)

    def prepare_features(self, df: pd.DataFrame):
        """Prepare feature matrix and target variable."""
        numerical_features = ["average_corrosion_rate", "thickness_mm", "corrosion_thickness_ratio"]

        categorical_features = ["commodity", "feature_type", "cml_shape"]

        # Add optional features if they exist
        if "days_since_inspection" in df.columns:
            numerical_features.append("days_since_inspection")
        if "risk_score" in df.columns:
            numerical_features.append("risk_score")
        if "remaining_life_years" in df.columns:
            numerical_features.append("remaining_life_years")

        # Create preprocessing pipelines
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        X = df[numerical_features + categorical_features]
        y = df["elimination_flag"]

        self.feature_names = numerical_features + categorical_features

        return X, y, preprocessor

    #: Searched when no explicit grid is supplied. Kept as a class
    #: attribute so callers (and tests) can narrow it without editing code.
    DEFAULT_PARAM_GRID = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [10, 20, None],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2],
        "classifier__class_weight": ["balanced", "balanced_subsample"],
    }

    def train_model(self, X, y, preprocessor, param_grid: dict | None = None):
        """Train a Random Forest classifier with hyperparameter tuning.

        Args:
            X: Feature matrix.
            y: Binary elimination target.
            preprocessor: Fitted-on-train ColumnTransformer.
            param_grid: Grid to search. Defaults to DEFAULT_PARAM_GRID;
                pass ``{}`` to fit a single estimator without a search.

        Returns:
            A dictionary of evaluation metrics.
        """
        print("Training model...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        classifier: Any = RandomForestClassifier(random_state=42)
        if self.calibration is not None:
            # Calibrate on cross-validated folds of the training split, so
            # the mapping is not fitted on the same predictions it corrects.
            classifier = CalibratedClassifierCV(classifier, method=self.calibration, cv=5)

        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])

        if param_grid is None:
            param_grid = self.DEFAULT_PARAM_GRID
        if self.calibration is not None and param_grid:
            # CalibratedClassifierCV nests the real estimator, so the grid
            # keys move with it.
            param_grid = {
                key.replace("classifier__", "classifier__estimator__"): value
                for key, value in param_grid.items()
            }

        # Grid search with cross-validation. Folds are capped by the size
        # of the smaller class so a small dataset cannot make CV fail.
        n_splits = max(2, min(5, int(y.value_counts().min())))
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=n_splits, scoring="f1", n_jobs=-1, verbose=1
        )

        grid_search.fit(X_train, y_train)

        # Bound to a local as well as the attribute: the attribute is
        # Optional, and an assert would vanish under `python -O`.
        model: Pipeline = grid_search.best_estimator_
        self.model = model

        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Keep", "Eliminate"]))
        print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
        print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
        print(f"\nBest Parameters: {grid_search.best_params_}")

        # Cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=n_splits, scoring="f1")
        print(f"\nCross-validation F1 scores: {cv_scores}")
        print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Calibration quality. Reported unconditionally: it is the number
        # that says whether elimination_probability can be read as a
        # probability, and it is only checkable if it is measured.
        brier = float(brier_score_loss(y_test, y_proba))
        expected_calibration_error = _expected_calibration_error(y_test.to_numpy(), y_proba)
        print(f"Brier score: {brier:.4f} (lower is better)")
        print(f"Expected calibration error: {expected_calibration_error:.4f}")

        # Feature importance
        if hasattr(model.named_steps["classifier"], "feature_importances_"):
            importances = model.named_steps["classifier"].feature_importances_
            feature_names_transformed = model.named_steps["preprocessor"].get_feature_names_out()
            feature_importance_df = pd.DataFrame(
                {"feature": feature_names_transformed, "importance": importances}
            ).sort_values("importance", ascending=False)

            print("\nTop 10 Feature Importances:")
            print(feature_importance_df.head(10).to_string(index=False))

        return {
            "test_accuracy": float(model.score(X_test, y_test)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "f1_score": float(f1_score(y_test, y_pred)),
            "cv_f1_mean": float(cv_scores.mean()),
            "cv_f1_std": float(cv_scores.std()),
            "brier_score": brier,
            "log_loss": float(log_loss(y_test, y_proba)),
            "expected_calibration_error": expected_calibration_error,
            "calibration": self.calibration,
            "best_params": grid_search.best_params_,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

    def save_model(self, metrics: dict):
        """Save trained model and metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_output_dir / f"cml_elimination_model_{timestamp}.joblib"
        latest_path = self.model_output_dir / "cml_elimination_model.joblib"

        # Save model
        joblib.dump(self.model, model_path)
        joblib.dump(self.model, latest_path)

        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "calibration": self.calibration,
            "feature_names": self.feature_names,
            "metrics": metrics,
            "model_path": str(model_path),
        }

        metadata_path = self.model_output_dir / f"model_metadata_{timestamp}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nModel saved to {model_path}")
        print(f"Latest model: {latest_path}")

        return model_path


def _expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    """Mean gap between predicted confidence and observed frequency.

    Zero means that among CMLs predicted at 0.7, exactly 70% really were
    eliminations. Reported alongside Brier because Brier mixes calibration
    with discrimination, and only the former is what "probability" claims.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (probabilities > low) & (probabilities <= high)
        if in_bin.sum():
            error += (
                in_bin.sum()
                / len(probabilities)
                * abs(y_true[in_bin].mean() - probabilities[in_bin].mean())
            )
    return float(error)


def train_enhanced_cml_model(
    data_path: str = "data/sample_cml_data.csv", calibration: str | None = None
) -> Path:
    """Main training function."""
    trainer = EnhancedCMLModelTrainer(data_path, calibration=calibration)

    # Load and prepare data
    df = trainer.load_data()
    df = trainer.engineer_features(df)
    X, y, preprocessor = trainer.prepare_features(df)

    # Train model
    metrics = trainer.train_model(X, y, preprocessor)

    # Save model
    model_path = trainer.save_model(metrics)

    return model_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the CML elimination model.")
    parser.add_argument(
        "data_path", nargs="?", default="data/sample_cml_data.csv", help="Labelled CML CSV"
    )
    parser.add_argument(
        "--calibrate",
        choices=["sigmoid", "isotonic"],
        default=None,
        help=(
            "Calibrate predicted probabilities. Off by default: on the bundled "
            "datasets it costs more F1 than it recovers in calibration error."
        ),
    )
    args = parser.parse_args()
    train_enhanced_cml_model(args.data_path, calibration=args.calibrate)
