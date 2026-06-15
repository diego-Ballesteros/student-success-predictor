"""Regression tests for :class:`src.data.preprocessor.Preprocessor`.

Locks in the guarantees verified during implementation: the declared
feature groups match the persisted data, the shared ``ColumnTransformer``
expands the 39 raw features into a stable column space, that space is
finite (no NaN/inf), every categorical feature survives one-hot encoding,
and the binary passthrough columns remain 0/1.

Run with: ``uv run pytest tests/test_preprocessor.py -v``
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer

from src.data.preprocessor import Preprocessor

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


@pytest.fixture(scope="session")
def x_train() -> pd.DataFrame:
    """Load the resolved-outcome training features once per test session.

    Returns:
        pd.DataFrame: The 39-feature training matrix.
    """
    return pd.read_parquet(PROCESSED_DIR / "X_train.parquet")


@pytest.fixture(scope="session")
def enrolled_demo() -> pd.DataFrame:
    """Load the currently-enrolled inference features once per session.

    Returns:
        pd.DataFrame: The 39-feature matrix for the 794 enrolled students.
    """
    return pd.read_parquet(PROCESSED_DIR / "enrolled_demo.parquet")


@pytest.fixture(scope="session")
def fitted_transformer(x_train: pd.DataFrame) -> ColumnTransformer:
    """Build and fit the shared transformer on the training set once.

    Args:
        x_train: The training feature matrix fixture.

    Returns:
        ColumnTransformer: The transformer fitted on ``x_train``.
    """
    transformer = Preprocessor().build_transformer()
    transformer.fit(x_train)
    return transformer


@pytest.fixture(scope="session")
def transformed_arrays(
    fitted_transformer: ColumnTransformer,
    x_train: pd.DataFrame,
    enrolled_demo: pd.DataFrame,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Transform both datasets with the same fitted transformer.

    Args:
        fitted_transformer: The transformer fitted on the training set.
        x_train: The training feature matrix fixture.
        enrolled_demo: The enrolled inference feature matrix fixture.

    Returns:
        Tuple[NDArray, NDArray]: ``(X_train_transformed, enrolled_transformed)``.
    """
    return (
        fitted_transformer.transform(x_train),
        fitted_transformer.transform(enrolled_demo),
    )


def test_validate_against_passes(x_train: pd.DataFrame) -> None:
    """The declared feature groups exactly cover the training columns."""
    # Should return None silently; any mismatch raises ValueError.
    assert Preprocessor.validate_against(x_train.columns.tolist()) is None


def test_transform_shapes(
    transformed_arrays: Tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    """Row counts are preserved and both sets share the same column space."""
    x_train_t, enrolled_t = transformed_arrays

    assert x_train_t.shape[0] == 2904
    assert enrolled_t.shape[0] == 794
    # The exact column count is not hardcoded; the two must simply agree so
    # the enrolled inference set aligns with the training feature space.
    assert x_train_t.shape[1] == enrolled_t.shape[1]


def test_no_nan_or_inf(
    transformed_arrays: Tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    """Neither transformed array contains NaN or infinite values."""
    x_train_t, enrolled_t = transformed_arrays

    if not np.isfinite(x_train_t).all():
        print(f"X_train transformed shape (non-finite found): {x_train_t.shape}")
    if not np.isfinite(enrolled_t).all():
        print(f"enrolled transformed shape (non-finite found): {enrolled_t.shape}")

    assert np.isfinite(x_train_t).all()
    assert np.isfinite(enrolled_t).all()


def test_categorical_ohe_breakdown(fitted_transformer: ColumnTransformer) -> None:
    """Every categorical feature produces at least one one-hot column."""
    feature_names = fitted_transformer.get_feature_names_out()
    # One-hot outputs are prefixed with the "cat" transformer name; filtering
    # to that prefix avoids matching numeric columns such as
    # "num__Previous qualification (grade)".
    cat_names = [name for name in feature_names if name.startswith("cat__")]

    print("\nCategorical OHE breakdown (feature -> output columns):")
    counts = {}
    for feature in Preprocessor.CATEGORICAL_FEATURES:
        count = sum(1 for name in cat_names if feature in name)
        counts[feature] = count
        print(f"  {feature:<28} -> {count}")
    print(f"  {'TOTAL':<28} -> {sum(counts.values())}")

    for feature, count in counts.items():
        assert count >= 1, f"categorical feature produced no columns: {feature}"


def test_binary_passthrough_is_01(
    fitted_transformer: ColumnTransformer, x_train: pd.DataFrame
) -> None:
    """Passthrough binary columns retain strictly 0/1 values."""
    feature_names = fitted_transformer.get_feature_names_out()
    x_train_t = fitted_transformer.transform(x_train)

    # Passthrough columns keep their original name behind the "bin__" prefix.
    bin_indices = [
        i for i, name in enumerate(feature_names) if name.startswith("bin__")
    ]
    assert len(bin_indices) == len(Preprocessor.BINARY_FEATURES)

    binary_block = x_train_t[:, bin_indices]
    unique_values = set(np.unique(binary_block).tolist())
    assert unique_values <= {0.0, 1.0}, f"unexpected binary values: {unique_values}"
