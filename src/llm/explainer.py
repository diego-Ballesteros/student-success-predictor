from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")


class StudentRiskExplainer:
    """Generates natural-language risk reports for academic counselors via Claude.

    Combines ML predictions, SHAP feature attributions, and student context
    into a structured explanation using claude-haiku-4-5-20251001.
    Single responsibility: explanation only — no training or data loading.
    """

    MODEL: str = "claude-haiku-4-5-20251001"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
