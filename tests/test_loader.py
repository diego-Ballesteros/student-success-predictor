"""Tests for :class:`src.data.loader.DataLoader`.

Locks in the shapes, target dtype/values, and column alignment across the
train, test, and enrolled-inference splits.

Run with: ``uv run pytest tests/test_loader.py -v``
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pandas.api.types import is_integer_dtype

from src.data.loader import DataLoader

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


@pytest.fixture(scope="session")
def loader() -> DataLoader:
    """Provide a DataLoader pointed at the processed data directory.

    Returns:
        DataLoader: Configured loader instance.
    """
    return DataLoader(PROCESSED_DIR)


def test_load_train_test_shapes(loader: DataLoader) -> None:
    """Train/test splits have the expected shapes."""
    X_train, X_test, y_train, y_test = loader.load_train_test()
    assert X_train.shape == (2904, 39)
    assert X_test.shape == (726, 39)
    assert len(y_train) == 2904
    assert len(y_test) == 726


def test_load_train_test_y_values(loader: DataLoader) -> None:
    """Targets are integer Series restricted to {0, 1}."""
    _, _, y_train, y_test = loader.load_train_test()
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert is_integer_dtype(y_train)
    assert is_integer_dtype(y_test)
    assert set(y_train.unique()) <= {0, 1}
    assert set(y_test.unique()) <= {0, 1}


def test_load_enrolled_demo_shape(loader: DataLoader) -> None:
    """Enrolled set has 39 features and no target columns."""
    enrolled = loader.load_enrolled_demo()
    assert enrolled.shape == (794, 39)
    assert "Target" not in enrolled.columns
    assert "Target_binary" not in enrolled.columns


def test_train_test_enrolled_same_columns(loader: DataLoader) -> None:
    """All three feature matrices share identical column order."""
    X_train, X_test, _, _ = loader.load_train_test()
    enrolled = loader.load_enrolled_demo()
    assert (
        X_train.columns.tolist()
        == X_test.columns.tolist()
        == enrolled.columns.tolist()
    )
