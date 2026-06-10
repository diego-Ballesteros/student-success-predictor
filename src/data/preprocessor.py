from __future__ import annotations

from pathlib import Path

import pandas as pd


class Preprocessor:
    """Applies feature engineering and train/test splitting to raw data.

    Handles encoding, scaling, and stratified splits — no model training.
    """

    def __init__(self, processed_dir: Path, test_size: float = 0.2, random_state: int = 42) -> None:
        self.processed_dir = processed_dir
        self.test_size = test_size
        self.random_state = random_state
