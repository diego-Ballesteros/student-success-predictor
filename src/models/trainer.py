"""Training orchestration for the Student Success Predictor.

Defines :class:`ModelTrainer`, which drives the two Phase 2 steps:

- **Step A** — compare the four candidate models with stratified k-fold
  cross-validation (:meth:`ModelTrainer.compare_models`).
- **Step B** — tune the winning model's hyperparameters with Optuna
  (:meth:`ModelTrainer.tune_with_optuna`).

The trainer depends only on :class:`AbstractBaseModel` (DIP) and never on
concrete classifier classes, so new models require no changes here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.models.base_model import AbstractBaseModel

# Keep Optuna's per-trial chatter out of notebook/CI output.
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ModelTrainer:
    """Cross-validates and tunes models behind the AbstractBaseModel API.

    Attributes:
        preprocessor: The unfitted ``ColumnTransformer`` template shared by
            every model and every fold. It is never fitted directly here;
            each fold gets a fresh clone via ``model.fit`` (see
            :meth:`cross_validate`).
        n_splits: Number of stratified CV folds.
        random_state: Seed for fold shuffling and the Optuna sampler.
    """

    def __init__(
        self,
        preprocessor: ColumnTransformer,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> None:
        """Initialize the trainer.

        Args:
            preprocessor: Unfitted ``ColumnTransformer`` (e.g. from
                ``Preprocessor().build_transformer()``).
            n_splits: Number of stratified folds for cross-validation.
            random_state: Seed for reproducible folds and tuning.
        """
        self.preprocessor = preprocessor
        self.n_splits = n_splits
        self.random_state = random_state

    def cross_validate(
        self,
        model: AbstractBaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Stratified k-fold cross-validation with leakage-free pipelines.

        Each fold fits its OWN copy of the preprocessor on only that fold's
        training data. Fitting ``StandardScaler``/``OneHotEncoder`` once on
        all of ``X`` before the loop would leak validation-fold statistics
        into training. We avoid that by calling ``model.fit`` (the Template
        Method) inside each fold — it clones the preprocessor fresh via
        ``sklearn.base.clone`` — and never pre-transforming data outside
        the loop.

        Args:
            model: A concrete ``AbstractBaseModel`` subclass instance.
            X: Feature matrix.
            y: Binary target (Dropout=1, Graduate=0).
            params: Optional hyperparameter overrides for the estimator.

        Returns:
            Dict[str, Any]: ``model_name``; ``{metric}_mean`` and
            ``{metric}_std`` for each of f1, roc_auc, precision, recall;
            and ``fold_scores`` mapping each metric to its per-fold list.
        """
        params = params or {}
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        fold_scores: Dict[str, List[float]] = {
            "f1": [],
            "roc_auc": [],
            "precision": [],
            "recall": [],
        }

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            fitted_pipeline = model.fit(X_tr, y_tr, self.preprocessor, **params)
            y_pred = fitted_pipeline.predict(X_val)
            y_proba = fitted_pipeline.predict_proba(X_val)[:, 1]

            fold_scores["f1"].append(f1_score(y_val, y_pred, zero_division=0))
            fold_scores["roc_auc"].append(roc_auc_score(y_val, y_proba))
            fold_scores["precision"].append(
                precision_score(y_val, y_pred, zero_division=0)
            )
            fold_scores["recall"].append(
                recall_score(y_val, y_pred, zero_division=0)
            )

        result: Dict[str, Any] = {"model_name": model.name}
        for metric, scores in fold_scores.items():
            series = pd.Series(scores)
            result[f"{metric}_mean"] = float(series.mean())
            result[f"{metric}_std"] = float(series.std())
        result["fold_scores"] = fold_scores
        return result

    def compare_models(
        self,
        models: List[AbstractBaseModel],
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        """Cross-validate every model (defaults) and rank by mean F1.

        Args:
            models: Candidate model instances to compare.
            X: Feature matrix.
            y: Binary target (Dropout=1, Graduate=0).

        Returns:
            pd.DataFrame: One row per model with mean/std for each metric,
            sorted by ``f1_mean`` descending with a reset index.
        """
        columns = [
            "model_name",
            "f1_mean",
            "f1_std",
            "roc_auc_mean",
            "roc_auc_std",
            "precision_mean",
            "precision_std",
            "recall_mean",
            "recall_std",
        ]

        rows: List[Dict[str, Any]] = []
        for model in models:
            result = self.cross_validate(model, X, y)
            rows.append({col: result[col] for col in columns})

        df = pd.DataFrame(rows, columns=columns)
        df = df.sort_values("f1_mean", ascending=False).reset_index(drop=True)
        return df

    def tune_with_optuna(
        self,
        model: AbstractBaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 40,
        metric: str = "f1",
    ) -> Dict[str, Any]:
        """Tune a model's hyperparameters with Optuna over CV mean score.

        Each trial samples a configuration from ``model.get_param_space``
        and evaluates it via :meth:`cross_validate`, optimizing the chosen
        metric's cross-validated mean.

        Args:
            model: The model to tune.
            X: Feature matrix.
            y: Binary target (Dropout=1, Graduate=0).
            n_trials: Number of Optuna trials.
            metric: Metric to maximize (one of f1, roc_auc, precision,
                recall); the ``{metric}_mean`` from CV is optimized.

        Returns:
            Dict[str, Any]: ``best_params``, ``best_score`` (best CV mean),
            and ``study`` (the completed Optuna study).
        """

        def objective(trial: optuna.Trial) -> float:
            params = model.get_param_space(trial)
            result = self.cross_validate(model, X, y, params=params)
            return result[f"{metric}_mean"]

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "study": study,
        }
