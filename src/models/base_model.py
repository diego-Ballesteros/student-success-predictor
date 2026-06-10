from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class AbstractBaseModel(ABC):
    """Open/Closed interface for all classification models.

    New models extend this class without modifying existing code (OCP).
    Concrete subclasses must implement fit, predict, and predict_proba.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AbstractBaseModel":
        """Train the model on labeled data."""
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return class predictions for X."""
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities for X."""
        ...
