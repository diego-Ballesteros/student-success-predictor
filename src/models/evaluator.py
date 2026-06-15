"""Final evaluation, persistence, and SHAP explainability.

Defines :class:`ModelEvaluator`, which owns Phase 2 Steps C and D:

- **Step C** — fit the chosen model on the full training set, perform the
  single one-time held-out evaluation on the test set, plot diagnostics,
  and persist the model + metrics.
- **Step D** — compute SHAP values on the fitted pipeline and produce both
  a global summary plot and per-instance explanations.

This class evaluates and reports only; it never selects models or loads
raw data (SRP).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from src.models.base_model import AbstractBaseModel


class ModelEvaluator:
    """Evaluates, visualizes, persists, and explains the final model.

    Attributes:
        artifacts_dir: Directory where models, metrics, and plots are
            written. Created on init if it does not exist.
    """

    def __init__(self, artifacts_dir: Path) -> None:
        """Initialize the evaluator and ensure the artifacts directory exists.

        Args:
            artifacts_dir: Target directory for all saved artifacts.
        """
        self.artifacts_dir = artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def fit_final_model(
        self,
        model: AbstractBaseModel,
        best_params: Dict[str, Any],
        preprocessor: ColumnTransformer,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Pipeline:
        """Fit the chosen model on the full training set.

        Delegates to the model's ``fit`` Template Method, which handles the
        ``sample_weight`` vs ``class_weight`` distinction transparently.

        Args:
            model: The winning ``AbstractBaseModel`` subclass instance.
            best_params: Tuned hyperparameters from Optuna.
            preprocessor: Unfitted ``ColumnTransformer`` template.
            X_train: Full training feature matrix.
            y_train: Full training target (Dropout=1, Graduate=0).

        Returns:
            Pipeline: The fitted preprocessing + model pipeline.
        """
        return model.fit(X_train, y_train, preprocessor, **best_params)

    def evaluate_on_test(
        self,
        fitted_pipeline: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Any]:
        """Run the one-time held-out evaluation on the test set.

        IMPORTANT: This is the ONLY method in the entire project that
        should ever be called on ``X_test``/``y_test``. The test set is
        touched exactly once, after model selection and tuning are fully
        complete, to obtain an unbiased estimate of generalization.

        Args:
            fitted_pipeline: A pipeline already fitted on the training set.
            X_test: Held-out test feature matrix.
            y_test: Held-out test target (Dropout=1, Graduate=0).

        Returns:
            Dict[str, Any]: ``f1``, ``roc_auc``, ``precision``, ``recall``
            (all for the positive class Dropout=1), ``confusion_matrix``
            (np.ndarray), and ``classification_report`` (dict).
        """
        y_pred = fitted_pipeline.predict(X_test)
        y_proba = fitted_pipeline.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            target_names=["Graduate", "Dropout"],
            zero_division=0,
        )

        return {
            "f1": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "precision": float(
                precision_score(y_test, y_pred, pos_label=1, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, y_pred, pos_label=1, zero_division=0)
            ),
            "confusion_matrix": cm,
            "classification_report": report,
        }

    def plot_confusion_matrix(self, cm: np.ndarray, filename: str) -> Path:
        """Render and save the test-set confusion matrix heatmap.

        Args:
            cm: A 2x2 confusion matrix array.
            filename: Output PNG filename within the artifacts dir.

        Returns:
            Path: Path to the saved PNG.
        """
        labels = ["Graduate", "Dropout"]
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix — Test Set")

        out_path = self.artifacts_dir / filename
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return out_path

    def plot_roc_curve(
        self, y_test: pd.Series, y_proba: np.ndarray, filename: str
    ) -> Path:
        """Render and save the test-set ROC curve.

        Args:
            y_test: Held-out test target.
            y_proba: Predicted probabilities for the positive class.
            filename: Output PNG filename within the artifacts dir.

        Returns:
            Path: Path to the saved PNG.
        """
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
        ax.plot(fpr, tpr, color="C0", label=f"ROC (AUC = {auc:.3f})")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — Test Set")
        ax.legend(loc="lower right")

        out_path = self.artifacts_dir / filename
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return out_path

    def save_model(
        self, fitted_pipeline: Pipeline, filename: str = "best_model.pkl"
    ) -> Path:
        """Pickle the fitted pipeline to disk.

        Args:
            fitted_pipeline: The fitted pipeline to serialize.
            filename: Output filename within the artifacts dir.

        Returns:
            Path: Path to the saved pickle file.
        """
        out_path = self.artifacts_dir / filename
        with open(out_path, "wb") as f:
            pickle.dump(fitted_pipeline, f)
        return out_path

    def save_metrics_csv(
        self, rows: List[Dict[str, Any]], filename: str = "metrics_comparison.csv"
    ) -> Path:
        """Write a list of metric rows to a CSV.

        Args:
            rows: One dict per model/run; keys become CSV columns.
            filename: Output filename within the artifacts dir.

        Returns:
            Path: Path to the saved CSV file.
        """
        out_path = self.artifacts_dir / filename
        pd.DataFrame(rows).to_csv(out_path, index=False)
        return out_path

    def compute_shap_values(
        self,
        fitted_pipeline: Pipeline,
        X_background: pd.DataFrame,
        X_explain: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Compute SHAP values for the model on transformed features.

        SHAP operates on the post-preprocessing feature space, so both the
        background and the to-explain frames are transformed with the
        already-fitted preprocessor before being passed to the explainer.
        For binary classifiers SHAP may return either a 2D array or a 3D
        array (samples x features x classes); in the latter case the
        class-1 (Dropout) slice is extracted.

        Args:
            fitted_pipeline: A fitted preprocessing + model pipeline.
            X_background: Reference sample for the explainer baseline.
            X_explain: Rows to explain.

        Returns:
            Dict[str, Any]: ``explainer``, ``shap_values`` (2D ndarray of
            shape ``(n_explain, n_features)``), ``feature_names``,
            ``X_explain_transformed``, and ``base_value`` (scalar expected
            value for class 1).
        """
        preprocessor = fitted_pipeline.named_steps["preprocessor"]
        model = fitted_pipeline.named_steps["model"]

        X_background_t = preprocessor.transform(X_background)
        X_explain_t = preprocessor.transform(X_explain)
        feature_names = preprocessor.get_feature_names_out().tolist()

        explainer = shap.Explainer(model, X_background_t)
        # check_additivity=False: documented HistGradientBoostingClassifier +
        # TreeExplainer limitation, does not affect SHAP value direction/ranking validity
        shap_values_obj = explainer(X_explain_t, check_additivity=False)

        raw_ndim = shap_values_obj.values.ndim
        if raw_ndim == 3:
            shap_values = shap_values_obj.values[..., 1]
        else:
            shap_values = shap_values_obj.values

        # Resolve the base/expected value for class 1, mirroring the ndim logic.
        expected = explainer.expected_value
        expected_arr = np.atleast_1d(expected)
        if expected_arr.shape[0] > 1:
            base_value = float(expected_arr[1])
        else:
            base_value = float(expected_arr[0])

        print(
            f"[SHAP] raw values.ndim={raw_ndim}, "
            f"raw shape={tuple(shap_values_obj.values.shape)}, "
            f"final shap_values shape={tuple(shap_values.shape)}, "
            f"base_value={base_value:.4f}"
        )

        return {
            "explainer": explainer,
            "shap_values": shap_values,
            "feature_names": feature_names,
            "X_explain_transformed": X_explain_t,
            "base_value": base_value,
        }

    def shap_summary_plot(self, shap_result: Dict[str, Any], filename: str) -> Path:
        """Render and save the global SHAP summary (beeswarm) plot.

        Args:
            shap_result: Output of :meth:`compute_shap_values`.
            filename: Output PNG filename within the artifacts dir.

        Returns:
            Path: Path to the saved PNG.
        """
        shap.summary_plot(
            shap_result["shap_values"],
            shap_result["X_explain_transformed"],
            feature_names=shap_result["feature_names"],
            show=False,
        )
        out_path = self.artifacts_dir / filename
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        return out_path

    def explain_instance(
        self,
        shap_result: Dict[str, Any],
        row_idx: int,
        fitted_pipeline: Pipeline,
        X_explain_original: pd.DataFrame,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Produce a per-student explanation from SHAP contributions.

        Args:
            shap_result: Output of :meth:`compute_shap_values`.
            row_idx: Positional index into ``X_explain`` for the student.
            fitted_pipeline: The fitted pipeline (for prediction).
            X_explain_original: The untransformed frame used for explaining.
            top_k: Number of top features (by absolute SHAP) to report.

        Returns:
            Dict[str, Any]: ``prediction`` ("Dropout"/"Graduate"),
            ``probability_dropout`` (float), and ``top_features`` — a list
            of dicts with ``feature``, ``shap_value``, ``feature_value``,
            and ``direction`` ("increases_risk"/"decreases_risk").
        """
        row = X_explain_original.iloc[[row_idx]]
        pred = int(fitted_pipeline.predict(row)[0])
        proba_dropout = float(fitted_pipeline.predict_proba(row)[0, 1])

        row_shap = shap_result["shap_values"][row_idx]
        feature_names = shap_result["feature_names"]
        transformed_row = shap_result["X_explain_transformed"][row_idx]

        order = np.argsort(np.abs(row_shap))[::-1][:top_k]

        top_features: List[Dict[str, Any]] = []
        for idx in order:
            shap_val = float(row_shap[idx])
            top_features.append(
                {
                    "feature": feature_names[idx],
                    "shap_value": shap_val,
                    "feature_value": float(transformed_row[idx]),
                    "direction": (
                        "increases_risk" if shap_val > 0 else "decreases_risk"
                    ),
                }
            )

        return {
            "prediction": "Dropout" if pred == 1 else "Graduate",
            "probability_dropout": proba_dropout,
            "top_features": top_features,
        }
