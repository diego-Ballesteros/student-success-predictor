from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.base_model import AbstractBaseModel


class ModelEvaluator:
    """Computes offline and online metrics and generates evaluation reports.

    Single responsibility: evaluation only — no training or data loading.
    Primary metric is Macro F1-Score to handle class imbalance.
    """

    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir
