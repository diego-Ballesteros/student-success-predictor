"""Tests for :class:`src.models.evaluator.ModelEvaluator`.

Covers final-fit, the one-time test evaluation contract, plot/persistence
artifacts, and the SHAP analysis path (global summary + per-instance
explanation). A fast LogisticRegression on 300 training rows stands in for
the real winner; correctness, not performance, is under test.

Run with: ``uv run pytest tests/test_evaluator.py -v``
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

import matplotlib

matplotlib.use("Agg")  # headless backend for plot tests

from sklearn.pipeline import Pipeline

from src.data.preprocessor import Preprocessor
from src.models.classifiers import LogisticRegressionModel
from src.models.evaluator import ModelEvaluator

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


@pytest.fixture(scope="session")
def train_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load the first 300 training rows.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: ``(X_train_small, y_train_small)``.
    """
    x = pd.read_parquet(PROCESSED_DIR / "X_train.parquet").head(300)
    y = pd.read_parquet(PROCESSED_DIR / "y_train.parquet").head(300)
    return x, y.squeeze("columns")


@pytest.fixture(scope="session")
def test_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load the first 100 test rows.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: ``(X_test_small, y_test_small)``.
    """
    x = pd.read_parquet(PROCESSED_DIR / "X_test.parquet").head(100)
    y = pd.read_parquet(PROCESSED_DIR / "y_test.parquet").head(100)
    return x, y.squeeze("columns")


@pytest.fixture(scope="session")
def fitted_pipeline(train_data: Tuple[pd.DataFrame, pd.Series]) -> Pipeline:
    """Fit a fast LogisticRegression pipeline once for the session.

    Returns:
        Pipeline: Fitted preprocessing + model pipeline.
    """
    x_train, y_train = train_data
    model = LogisticRegressionModel()
    preprocessor = Preprocessor().build_transformer()
    return model.fit(x_train, y_train, preprocessor)


@pytest.fixture(scope="session")
def shap_result(
    fitted_pipeline: Pipeline,
    train_data: Tuple[pd.DataFrame, pd.Series],
    test_data: Tuple[pd.DataFrame, pd.Series],
    tmp_path_factory: pytest.TempPathFactory,
) -> dict:
    """Compute SHAP values once (background=50 train, explain=10 test).

    Returns:
        dict: The result of ``compute_shap_values``.
    """
    x_train, _ = train_data
    x_test, _ = test_data
    evaluator = ModelEvaluator(tmp_path_factory.mktemp("shap"))
    return evaluator.compute_shap_values(
        fitted_pipeline, x_train.head(50), x_test.head(10)
    )


def test_fit_final_model_returns_fitted_pipeline(
    train_data: Tuple[pd.DataFrame, pd.Series], tmp_path: Path
) -> None:
    """fit_final_model returns a usable fitted Pipeline."""
    x_train, y_train = train_data
    evaluator = ModelEvaluator(tmp_path)
    pipeline = evaluator.fit_final_model(
        LogisticRegressionModel(),
        {"C": 1.0},
        Preprocessor().build_transformer(),
        x_train,
        y_train,
    )
    assert isinstance(pipeline, Pipeline)
    assert hasattr(pipeline, "predict")
    # Should not raise on a small sample.
    pipeline.predict(x_train.head(5))


def test_evaluate_on_test_structure(
    fitted_pipeline: Pipeline,
    test_data: Tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    """evaluate_on_test returns the full metric contract in valid ranges."""
    x_test, y_test = test_data
    evaluator = ModelEvaluator(tmp_path)
    result = evaluator.evaluate_on_test(fitted_pipeline, x_test, y_test)

    for key in (
        "f1",
        "roc_auc",
        "precision",
        "recall",
        "confusion_matrix",
        "classification_report",
    ):
        assert key in result

    for key in ("f1", "roc_auc", "precision", "recall"):
        assert isinstance(result[key], float)
        assert 0.0 <= result[key] <= 1.0

    assert result["confusion_matrix"].shape == (2, 2)


def test_plot_confusion_matrix_and_roc_curve_create_files(
    fitted_pipeline: Pipeline,
    test_data: Tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    """Both plotting methods write non-empty PNG files."""
    x_test, y_test = test_data
    evaluator = ModelEvaluator(tmp_path)
    result = evaluator.evaluate_on_test(fitted_pipeline, x_test, y_test)

    cm_path = evaluator.plot_confusion_matrix(
        result["confusion_matrix"], "cm.png"
    )
    y_proba = fitted_pipeline.predict_proba(x_test)[:, 1]
    roc_path = evaluator.plot_roc_curve(y_test, y_proba, "roc.png")

    for path in (cm_path, roc_path):
        assert path.exists()
        assert path.stat().st_size > 0


def test_save_model_and_metrics_csv(
    fitted_pipeline: Pipeline, tmp_path: Path
) -> None:
    """save_model and save_metrics_csv round-trip to disk."""
    evaluator = ModelEvaluator(tmp_path)
    model_path = evaluator.save_model(fitted_pipeline, "model.pkl")
    csv_path = evaluator.save_metrics_csv(
        [{"model_name": "test", "f1_mean": 0.8}], "metrics.csv"
    )

    assert model_path.exists()
    assert csv_path.exists()

    with open(model_path, "rb") as f:
        loaded = pickle.load(f)
    assert hasattr(loaded, "predict")


def test_compute_shap_values_structure(shap_result: dict) -> None:
    """SHAP values align with feature names and transformed explain set."""
    shap_values = shap_result["shap_values"]
    feature_names = shap_result["feature_names"]
    x_explain_t = shap_result["X_explain_transformed"]

    print(
        f"[test] shap_values shape={shap_values.shape}, "
        f"n_features={len(feature_names)}, "
        f"X_explain_transformed shape={x_explain_t.shape}"
    )

    assert shap_values.shape == (10, len(feature_names))
    assert shap_values.shape[1] == x_explain_t.shape[1]


def test_shap_summary_plot_creates_file(
    shap_result: dict, tmp_path: Path
) -> None:
    """shap_summary_plot writes a non-empty PNG."""
    evaluator = ModelEvaluator(tmp_path)
    path = evaluator.shap_summary_plot(shap_result, "shap_summary.png")
    assert path.exists()
    assert path.stat().st_size > 0


def test_explain_instance_structure(
    shap_result: dict,
    fitted_pipeline: Pipeline,
    test_data: Tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    """explain_instance returns a well-formed per-student explanation."""
    x_test, _ = test_data
    evaluator = ModelEvaluator(tmp_path)
    explanation = evaluator.explain_instance(
        shap_result, 0, fitted_pipeline, x_test.head(10), top_k=5
    )

    assert explanation["prediction"] in {"Dropout", "Graduate"}
    assert isinstance(explanation["probability_dropout"], float)
    assert 0.0 <= explanation["probability_dropout"] <= 1.0

    assert len(explanation["top_features"]) == 5
    for feat in explanation["top_features"]:
        assert set(feat.keys()) == {
            "feature",
            "shap_value",
            "feature_value",
            "direction",
        }
        assert feat["direction"] in {"increases_risk", "decreases_risk"}
