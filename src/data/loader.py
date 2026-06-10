from __future__ import annotations

from pathlib import Path

import pandas as pd


class DataLoader:
    """Loads raw student data from UCI repository or local cache.

    Responsible solely for data acquisition — no transformation happens here.
    """

    def __init__(self, raw_dir: Path, dataset_id: int = 697) -> None:
        self.raw_dir = raw_dir
        self.dataset_id = dataset_id
