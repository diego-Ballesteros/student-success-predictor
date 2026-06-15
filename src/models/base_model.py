"""Model abstraction layer for the Student Success Predictor.

Defines :class:`AbstractBaseModel`, the common interface every classifier
wrapper implements. The key design problem it solves: some estimators
support ``class_weight='balanced'`` in their constructor (LogisticRegression,
RandomForest) while others do not (GradientBoosting, HistGradientBoosting)
and must instead receive a ``sample_weight`` array at ``.fit()`` time. This
divergence is hidden behind a single ``fit()`` template method so the
downstream ``ModelTrainer`` can treat all models identically.

SOLID mapping
-------------
- **SRP** — Each concrete subclass knows only two things: how to build its
  sklearn estimator (``_build_estimator``) and how to describe its
  hyperparameter search space (``get_param_space``). It owns nothing else.
- **OCP** — Adding a fifth model is a new subclass; ``AbstractBaseModel``,
  ``ModelTrainer``, and the existing subclasses are never edited. The
  balanced-weighting strategy is selected by the ``USES_SAMPLE_WEIGHT``
  class attribute, not by branching in shared code.
- **LSP** — Every subclass is interchangeable through ``fit()``;
  ``ModelTrainer`` calls it the same way regardless of whether a model
  uses ``class_weight`` or ``sample_weight`` internally.
- **DIP** — ``ModelTrainer`` depends on this ``AbstractBaseModel``
  abstraction, never on concrete classifier classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import optuna
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight


class AbstractBaseModel(ABC):
    """Abstract interface and fit-template for all classifier wrappers.

    Subclasses supply an unfitted sklearn estimator and an Optuna search
    space; this base class assembles the preprocessing + model pipeline and
    applies balanced class weighting through whichever mechanism the
    underlying estimator supports.

    Attributes:
        name: Human-readable model name (e.g. ``"Logistic Regression"``).
        USES_SAMPLE_WEIGHT: ``True`` when the estimator does not accept
            ``class_weight`` and must instead receive a ``sample_weight``
            array at ``.fit()`` time. Overridden to ``True`` by the
            gradient-boosting subclasses; ``False`` by default.
    """

    name: str = "AbstractBaseModel"
    USES_SAMPLE_WEIGHT: bool = False

    @abstractmethod
    def _build_estimator(self, **params: Any) -> BaseEstimator:
        """Build the unfitted sklearn estimator for this model.

        Subclasses merge ``params`` over their own sensible defaults
        (always including ``random_state=42``, and
        ``class_weight='balanced'`` where the estimator supports it), so
        that values in ``params`` override the defaults on key collision.

        Args:
            **params: Hyperparameter overrides (e.g. from Optuna tuning).

        Returns:
            BaseEstimator: A configured but unfitted sklearn estimator.
        """
        ...

    @abstractmethod
    def get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define the Optuna hyperparameter search space for this model.

        Args:
            trial: The Optuna trial used to sample values via its
                ``suggest_*`` methods.

        Returns:
            Dict[str, Any]: Mapping of hyperparameter name to sampled value.
        """
        ...

    def build_pipeline(
        self, preprocessor: ColumnTransformer, **params: Any
    ) -> Pipeline:
        """Assemble an unfitted preprocessing + model pipeline.

        The preprocessor is cloned with :func:`sklearn.base.clone` so each
        call receives a fresh, unfitted transformer. This is essential for
        cross-validation, where every fold must fit its own transformer to
        avoid leakage across folds.

        Args:
            preprocessor: The (unfitted) ``ColumnTransformer`` template.
            **params: Hyperparameters forwarded to ``_build_estimator``.

        Returns:
            Pipeline: Unfitted pipeline with steps named ``'preprocessor'``
            and ``'model'``, in that order.
        """
        return Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", self._build_estimator(**params)),
            ]
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        preprocessor: ColumnTransformer,
        **params: Any,
    ) -> Pipeline:
        """Fit the full pipeline, applying balanced weighting uniformly.

        Template method called identically by ``ModelTrainer`` on every
        model. When :attr:`USES_SAMPLE_WEIGHT` is ``True``, a balanced
        ``sample_weight`` array is computed and routed to the ``'model'``
        step via the ``model__sample_weight`` keyword; otherwise the
        estimator's own ``class_weight='balanced'`` handles it.

        Args:
            X: Feature matrix.
            y: Target labels (binary: Dropout=1, Graduate=0).
            preprocessor: The (unfitted) ``ColumnTransformer`` template.
            **params: Hyperparameters forwarded to ``_build_estimator``.

        Returns:
            Pipeline: The fitted pipeline.
        """
        pipeline = self.build_pipeline(preprocessor, **params)
        if self.USES_SAMPLE_WEIGHT:
            sample_weight = compute_sample_weight("balanced", y)
            pipeline.fit(X, y, model__sample_weight=sample_weight)
        else:
            pipeline.fit(X, y)
        return pipeline
