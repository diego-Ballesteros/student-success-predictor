"""Correctness tests for :class:`src.models.trainer.ModelTrainer`.

Exercises the cross-validation result contract, the model-comparison
ranking, the Optuna tuning loop, and — critically — that the Template
Method's ``sample_weight`` branch survives a full CV run for the
gradient-boosting models. Uses the first 300 training rows for speed;
these are interface/correctness tests, not full-scale benchmarks.

Run with: ``uv run pytest tests/test_trainer.py -v``
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.data.preprocessor import Preprocessor
from src.models.base_model import AbstractBaseModel
from src.models.classifiers import (
    GradientBoostingModel,
    HistGradientBoostingModel,
    LogisticRegressionModel,
    RandomForestModel,
)
from src.models.trainer import ModelTrainer

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
N_SPLITS = 5


@pytest.fixture(scope="session")
def small_training_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load the first 300 training rows for fast correctness checks.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: ``(X_small, y_small)``.
    """
    x_small = pd.read_parquet(PROCESSED_DIR / "X_train.parquet").head(300)
    y_small = pd.read_parquet(PROCESSED_DIR / "y_train.parquet").head(300)
    return x_small, y_small.squeeze("columns")


@pytest.fixture(scope="session")
def trainer() -> ModelTrainer:
    """Provide a ModelTrainer with a fresh unfitted preprocessor.

    Returns:
        ModelTrainer: Configured with ``N_SPLITS`` folds.
    """
    preprocessor: ColumnTransformer = Preprocessor().build_transformer()
    return ModelTrainer(preprocessor, n_splits=N_SPLITS, random_state=42)


def test_cross_validate_returns_expected_keys(
    trainer: ModelTrainer,
    small_training_data: Tuple[pd.DataFrame, pd.Series],
) -> None:
    """cross_validate returns the full metric contract and per-fold scores."""
    x_small, y_small = small_training_data
    result = trainer.cross_validate(LogisticRegressionModel(), x_small, y_small)

    expected_keys = {
        "model_name",
        "f1_mean",
        "f1_std",
        "roc_auc_mean",
        "roc_auc_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "fold_scores",
    }
    assert expected_keys.issubset(result.keys())
    assert len(result["fold_scores"]["f1"]) == N_SPLITS


def test_cross_validate_metrics_in_valid_range(
    trainer: ModelTrainer,
    small_training_data: Tuple[pd.DataFrame, pd.Series],
) -> None:
    """All mean metrics fall within the valid [0, 1] range."""
    x_small, y_small = small_training_data
    result = trainer.cross_validate(LogisticRegressionModel(), x_small, y_small)

    for key in ("f1_mean", "roc_auc_mean", "precision_mean", "recall_mean"):
        assert 0.0 <= result[key] <= 1.0


def test_compare_models_structure(
    trainer: ModelTrainer,
    small_training_data: Tuple[pd.DataFrame, pd.Series],
) -> None:
    """compare_models returns a ranked DataFrame, best F1 first."""
    x_small, y_small = small_training_data
    df = trainer.compare_models(
        [LogisticRegressionModel(), RandomForestModel()], x_small, y_small
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "f1_mean" in df.columns
    assert df.loc[0, "f1_mean"] >= df.loc[1, "f1_mean"]


def test_tune_with_optuna_returns_valid_params(
    trainer: ModelTrainer,
    small_training_data: Tuple[pd.DataFrame, pd.Series],
) -> None:
    """tune_with_optuna returns valid best params and a bounded score."""
    x_small, y_small = small_training_data
    result = trainer.tune_with_optuna(
        LogisticRegressionModel(), x_small, y_small, n_trials=3, metric="f1"
    )

    assert {"best_params", "best_score", "study"}.issubset(result.keys())
    assert "C" in result["best_params"]
    assert 0.0 <= result["best_score"] <= 1.0


@pytest.mark.parametrize(
    "model",
    [GradientBoostingModel(), HistGradientBoostingModel()],
    ids=lambda m: m.name,
)
def test_cross_validate_works_for_sample_weight_models(
    trainer: ModelTrainer,
    small_training_data: Tuple[pd.DataFrame, pd.Series],
    model: AbstractBaseModel,
) -> None:
    """The sample_weight branch runs end-to-end through real CV."""
    x_small, y_small = small_training_data
    result = trainer.cross_validate(model, x_small, y_small)
    assert 0.0 <= result["f1_mean"] <= 1.0
