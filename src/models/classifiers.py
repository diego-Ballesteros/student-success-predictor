"""Concrete classifier wrappers for the Student Success Predictor.

Four estimators are wrapped behind :class:`AbstractBaseModel`. Each only
declares its sklearn estimator (with balanced defaults) and its Optuna
search space; all fitting logic lives in the base class.

Modeling rationale (see docs/DATA_INSIGHTS.md):
- Dropout has two distinct profiles — "academic" (collapsing
  ``approval_rate_sem2``) and "non-academic" (financial-distress signals
  despite passing grades). Tree ensembles can model this OR-pattern across
  feature groups, so Random Forest and Hist Gradient Boosting are expected
  to lead. Logistic Regression serves as an interpretable baseline that
  exploits the near-linear separability of ``approval_rate_sem2``.
"""

from __future__ import annotations

from typing import Any, Dict

import optuna
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression

from src.models.base_model import AbstractBaseModel


class LogisticRegressionModel(AbstractBaseModel):
    """Linear baseline classifier.

    Exploits the near-linear separability of ``approval_rate_sem2``
    (DATA_INSIGHTS.md §3) as an interpretable reference point against the
    tree ensembles. Handles class imbalance via ``class_weight='balanced'``.
    """

    name: str = "Logistic Regression"
    USES_SAMPLE_WEIGHT: bool = False

    def _build_estimator(self, **params: Any) -> BaseEstimator:
        """Build a balanced LogisticRegression, ``params`` overriding defaults.

        Args:
            **params: Hyperparameter overrides (e.g. ``C``).

        Returns:
            BaseEstimator: Unfitted ``LogisticRegression``.
        """
        defaults: Dict[str, Any] = {
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": 42,
        }
        return LogisticRegression(**{**defaults, **params})

    def get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample the inverse-regularization strength ``C``.

        Args:
            trial: Optuna trial used for sampling.

        Returns:
            Dict[str, Any]: ``{"C": ...}``.
        """
        return {"C": trial.suggest_float("C", 0.01, 10, log=True)}


class RandomForestModel(AbstractBaseModel):
    """Bagged decision-tree ensemble.

    Captures the OR-pattern between the academic and financial dropout
    profiles (DATA_INSIGHTS.md §2) and is robust to the multicollinearity
    among curricular-unit features. Balanced via ``class_weight='balanced'``.
    """

    name: str = "Random Forest"
    USES_SAMPLE_WEIGHT: bool = False

    def _build_estimator(self, **params: Any) -> BaseEstimator:
        """Build a balanced RandomForest, ``params`` overriding defaults.

        Args:
            **params: Hyperparameter overrides (e.g. ``n_estimators``).

        Returns:
            BaseEstimator: Unfitted ``RandomForestClassifier``.
        """
        defaults: Dict[str, Any] = {
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }
        return RandomForestClassifier(**{**defaults, **params})

    def get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample forest size and tree-shape hyperparameters.

        Args:
            trial: Optuna trial used for sampling.

        Returns:
            Dict[str, Any]: ``n_estimators``, ``max_depth``,
            ``min_samples_leaf``.
        """
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }


class GradientBoostingModel(AbstractBaseModel):
    """Sequentially boosted decision-tree ensemble.

    Models the same academic/financial OR-pattern as Random Forest
    (DATA_INSIGHTS.md §2) but stage-wise. It does not accept
    ``class_weight``, so imbalance is handled by ``sample_weight`` in the
    base-class ``fit()`` (``USES_SAMPLE_WEIGHT = True``).
    """

    name: str = "Gradient Boosting"
    USES_SAMPLE_WEIGHT: bool = True

    def _build_estimator(self, **params: Any) -> BaseEstimator:
        """Build a GradientBoosting estimator, ``params`` overriding defaults.

        Args:
            **params: Hyperparameter overrides (e.g. ``learning_rate``).

        Returns:
            BaseEstimator: Unfitted ``GradientBoostingClassifier``.
        """
        defaults: Dict[str, Any] = {"random_state": 42}
        return GradientBoostingClassifier(**{**defaults, **params})

    def get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample boosting depth, rate, and tree depth.

        Args:
            trial: Optuna trial used for sampling.

        Returns:
            Dict[str, Any]: ``n_estimators``, ``learning_rate``, ``max_depth``.
        """
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
        }


class HistGradientBoostingModel(AbstractBaseModel):
    """Histogram-based gradient-boosted ensemble.

    Fast, high-capacity learner that captures the OR-pattern between the
    academic and financial dropout profiles (DATA_INSIGHTS.md §2) and is a
    leading candidate alongside Random Forest. Imbalance is handled by
    ``sample_weight`` in the base-class ``fit()`` (``USES_SAMPLE_WEIGHT =
    True``).
    """

    name: str = "Hist Gradient Boosting"
    USES_SAMPLE_WEIGHT: bool = True

    def _build_estimator(self, **params: Any) -> BaseEstimator:
        """Build a HistGradientBoosting estimator, ``params`` over defaults.

        Args:
            **params: Hyperparameter overrides (e.g. ``max_iter``).

        Returns:
            BaseEstimator: Unfitted ``HistGradientBoostingClassifier``.
        """
        defaults: Dict[str, Any] = {"random_state": 42}
        return HistGradientBoostingClassifier(**{**defaults, **params})

    def get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample iterations, learning rate, and leaf budget.

        Args:
            trial: Optuna trial used for sampling.

        Returns:
            Dict[str, Any]: ``max_iter``, ``learning_rate``, ``max_leaf_nodes``.
        """
        return {
            "max_iter": trial.suggest_int("max_iter", 50, 300),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
        }
