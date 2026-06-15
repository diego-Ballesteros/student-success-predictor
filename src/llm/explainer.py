"""LLM-based risk explanation for the Student Success Predictor.

Defines :class:`StudentRiskExplainer`, which turns a SHAP-based dropout
prediction into a concise, counselor-actionable natural-language report
via the Anthropic Claude API.
"""

from __future__ import annotations

from pathlib import Path

import anthropic
from dotenv import load_dotenv


class StudentRiskExplainer:
    """LLM-based explainer that translates SHAP-based dropout risk
    predictions into counselor-actionable natural language reports.

    Per docs/DATA_INSIGHTS.md, students at risk fall into two profiles:
    Profile A (academic: low approval_rate_sem2 / 2nd-sem grade near 0)
    and Profile B (financial: reasonable grades but Tuition fees up to
    date = 0 or Debtor = 1). This explainer should identify which
    profile(s) a student's top SHAP features suggest, and recommend
    an appropriate intervention type (academic tutoring vs financial
    aid/counseling) accordingly.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001") -> None:
        """Loads ANTHROPIC_API_KEY from .env and initializes the client."""
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        self.client = anthropic.Anthropic()
        self.model = model

    def explain(self, explanation: dict) -> str:
        """Generates a natural-language risk report for one student.

        Args:
            explanation: output of ModelEvaluator.explain_instance() —
                {"prediction": "Dropout"|"Graduate",
                 "probability_dropout": float,
                 "top_features": [{"feature", "shap_value",
                 "feature_value", "direction"}, ...]}

        Returns:
            Counselor-facing explanation as plain text.
        """
        prompt = self._build_prompt(explanation)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _build_prompt(self, explanation: dict) -> str:
        """Builds the prompt sent to the LLM."""
        features_text = "\n".join(
            f"- {f['feature']}: value={f['feature_value']:.3f}, "
            f"SHAP={f['shap_value']:.3f} ({f['direction']})"
            for f in explanation["top_features"]
        )
        return (
            "You are an academic counselor's assistant analyzing a "
            "currently-enrolled student's dropout risk profile, based "
            "on a binary classifier (Dropout vs Graduate) and SHAP "
            "feature attributions.\n\n"
            f"Predicted profile: {explanation['prediction']}\n"
            f"Probability of dropout: "
            f"{explanation['probability_dropout']:.2%}\n\n"
            f"Top contributing factors (SHAP values; positive = "
            f"increases dropout risk, negative = decreases it):\n"
            f"{features_text}\n\n"
            "Write a concise report (max 150 words) for the academic "
            "counselor that:\n"
            "1. States the risk level in plain language\n"
            "2. Identifies whether this looks like an ACADEMIC risk "
            "profile (low approval rates / grades), a FINANCIAL risk "
            "profile (tuition/debt issues despite reasonable grades), "
            "both, or neither\n"
            "3. Recommends ONE concrete intervention type accordingly\n"
            "Be direct and actionable, avoid hedging language."
        )
