from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.models.base_model import AbstractBaseModel


class ModelTrainer:
    """Orchestrates cross-validation, hyperparameter tuning, and model persistence.

    Depends on AbstractBaseModel — never on concrete implementations (DIP).
    """

    def __init__(
        self,
        artifacts_dir: Path,
        cv_folds: int = 5,
        random_state: int = 42,
    ) -> None:
        self.artifacts_dir = artifacts_dir
        self.cv_folds = cv_folds
        self.random_state = random_state
