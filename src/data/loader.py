"""Processed-data loading for the Student Success Predictor.

Defines :class:`DataLoader`, which reads the persisted parquet splits from
``data/processed/``. It performs acquisition only — no feature engineering
or transformation (that lives in :class:`~src.data.preprocessor.Preprocessor`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

TARGET_COLUMN = "Target_binary"


class DataLoader:
    """Loads the processed train/test/enrolled parquet files.

    Attributes:
        processed_dir: Directory containing the processed parquet files.
    """

    def __init__(self, processed_dir: Path) -> None:
        """Initialize the loader.

        Args:
            processed_dir: Path to the ``data/processed`` directory.
        """
        self.processed_dir = processed_dir

    def load_train_test(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load the binary-classification train/test split.

        The target files hold a single ``Target_binary`` column, returned
        as an integer ``pd.Series`` named ``"Target_binary"``.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            ``(X_train, X_test, y_train, y_test)``.
        """
        X_train = pd.read_parquet(self.processed_dir / "X_train.parquet")
        X_test = pd.read_parquet(self.processed_dir / "X_test.parquet")
        y_train = pd.read_parquet(self.processed_dir / "y_train.parquet")[
            TARGET_COLUMN
        ].astype(int)
        y_test = pd.read_parquet(self.processed_dir / "y_test.parquet")[
            TARGET_COLUMN
        ].astype(int)
        return X_train, X_test, y_train, y_test

    def load_enrolled_demo(self) -> pd.DataFrame:
        """Load the currently-enrolled inference set.

        Returns:
            pd.DataFrame: The 794 enrolled students (no ground-truth label).
        """
        return pd.read_parquet(self.processed_dir / "enrolled_demo.parquet")
