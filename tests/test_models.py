"""Interface smoke tests for the model abstraction layer.

Verifies pipeline structure, the uniform fit/predict interface, the
sample-weight strategy flags, and Optuna search-space wiring across all
four concrete classifiers. These are fast interface checks (first 200
training rows only), not performance benchmarks.

Run with: ``uv run pytest tests/test_models.py -v``
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.data.preprocessor import Preprocessor
from src.models.base_model import AbstractBaseModel
from src.models.classifiers import (
    GradientBoostingModel,
    HistGradientBoostingModel,
    LogisticRegressionModel,
    RandomForestModel,
)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# Concrete model instances exercised by every parametrized test.
MODELS: List[AbstractBaseModel] = [
    LogisticRegressionModel(),
    RandomForestModel(),
    GradientBoostingModel(),
    HistGradientBoostingModel(),
]

# Representative in-range hyperparameters per model, used to build a
# FixedTrial so get_param_space is exercised deterministically.
FIXED_PARAMS = {
    "Logistic Regression": {"C": 1.0},
    "Random Forest": {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_leaf": 5,
    },
    "Gradient Boosting": {
        "n_estimators": 150,
        "learning_rate": 0.1,
        "max_depth": 3,
    },
    "Hist Gradient Boosting": {
        "max_iter": 150,
        "learning_rate": 0.1,
        "max_leaf_nodes": 31,
    },
}


@pytest.fixture(scope="session")
def small_training_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load the first 200 training rows for fast interface checks.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: ``(X_small, y_small)``.
    """
    x_small = pd.read_parquet(PROCESSED_DIR / "X_train.parquet").head(200)
    y_small = pd.read_parquet(PROCESSED_DIR / "y_train.parquet").head(200)
    # y is stored as a single-column frame; squeeze to a Series.
    return x_small, y_small.squeeze("columns")


@pytest.fixture(scope="session")
def preprocessor() -> ColumnTransformer:
    """Provide a fresh unfitted ColumnTransformer template.

    Returns:
        ColumnTransformer: Unfitted transformer from ``build_transformer``.
    """
    return Preprocessor().build_transformer()


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.name)
def test_build_pipeline_structure(
    model: AbstractBaseModel, preprocessor: ColumnTransformer
) -> None:
    """build_pipeline returns a Pipeline with the expected named steps."""
    pipeline = model.build_pipeline(preprocessor)
    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == ["preprocessor", "model"]


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.name)
def test_fit_and_predict(
    model: AbstractBaseModel,
    preprocessor: ColumnTransformer,
    small_training_data: Tuple[pd.DataFrame, pd.Series],
) -> None:
    """fit returns a fitted Pipeline that predicts valid binary labels."""
    x_small, y_small = small_training_data
    fitted = model.fit(x_small, y_small, preprocessor)

    assert isinstance(fitted, Pipeline)
    preds = fitted.predict(x_small)
    assert len(preds) == len(x_small)
    assert set(np.unique(preds)).issubset({0, 1})


def test_sample_weight_flag_consistency() -> None:
    """Only the gradient-boosting models flag USES_SAMPLE_WEIGHT."""
    assert GradientBoostingModel.USES_SAMPLE_WEIGHT is True
    assert HistGradientBoostingModel.USES_SAMPLE_WEIGHT is True
    assert LogisticRegressionModel.USES_SAMPLE_WEIGHT is False
    assert RandomForestModel.USES_SAMPLE_WEIGHT is False


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.name)
def test_get_param_space_returns_valid_dict(model: AbstractBaseModel) -> None:
    """get_param_space returns exactly the expected hyperparameter keys."""
    expected = FIXED_PARAMS[model.name]
    trial = optuna.trial.FixedTrial(expected)
    space = model.get_param_space(trial)
    assert set(space.keys()) == set(expected.keys())


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.name)
def test_build_pipeline_with_tuned_params(
    model: AbstractBaseModel, preprocessor: ColumnTransformer
) -> None:
    """Tuned params from get_param_space land on the estimator."""
    expected = FIXED_PARAMS[model.name]
    trial = optuna.trial.FixedTrial(expected)
    params = model.get_param_space(trial)

    pipeline = model.build_pipeline(preprocessor, **params)
    estimator_params = pipeline.named_steps["model"].get_params()
    for key, value in params.items():
        assert estimator_params[key] == value
