"""Preprocessing pipeline for the Student Success Predictor.

Defines the :class:`Preprocessor`, the single source of truth for the
feature column groups and the scikit-learn ``ColumnTransformer`` shared by
every Phase 2 model. Keeping the column lists and transformer construction
in one place guarantees all four models (LogisticRegression, RandomForest,
GradientBoosting, XGBoost) see an identical feature space.
"""

from __future__ import annotations

from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Preprocessor:
    """Builds the shared feature-transformation pipeline for modeling.

    The class declares the exact 39 feature columns (matching the names
    produced by ``fetch_ucirepo(id=697)`` after the feature engineering in
    ``notebooks/01_preprocessing.ipynb``) split into three groups:

    - ``NUMERIC_FEATURES`` (22): continuous/count features that are
      standardized.
    - ``CATEGORICAL_FEATURES`` (9): nominal codes that are one-hot encoded.
    - ``BINARY_FEATURES`` (8): already-binary 0/1 flags passed through
      untouched.

    Note the dataset's original ``"Nacionality"`` spelling is preserved
    verbatim so the column names align with the persisted parquet files.

    Attributes:
        NUMERIC_FEATURES: Names of the 22 numeric feature columns.
        CATEGORICAL_FEATURES: Names of the 9 categorical feature columns.
        BINARY_FEATURES: Names of the 8 binary (0/1) feature columns.
    """

    NUMERIC_FEATURES: List[str] = [
        # Curricular units — 1st semester (6)
        "Curricular units 1st sem (credited)",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 1st sem (without evaluations)",
        # Curricular units — 2nd semester (6)
        "Curricular units 2nd sem (credited)",
        "Curricular units 2nd sem (enrolled)",
        "Curricular units 2nd sem (evaluations)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Curricular units 2nd sem (without evaluations)",
        # Application / qualification grades
        "Application order",
        "Previous qualification (grade)",
        "Admission grade",
        # Demographic
        "Age at enrollment",
        # Macroeconomic
        "Unemployment rate",
        "Inflation rate",
        "GDP",
        # Engineered
        "approval_rate_sem1",
        "approval_rate_sem2",
        "grade_delta",
    ]

    CATEGORICAL_FEATURES: List[str] = [
        "Marital Status",
        "Application mode",
        "Course",
        "Previous qualification",
        "Nacionality",  # original dataset spelling preserved intentionally
        "Mother's qualification",
        "Father's qualification",
        "Mother's occupation",
        "Father's occupation",
    ]

    BINARY_FEATURES: List[str] = [
        "Daytime/evening attendance",
        "Displaced",
        "Educational special needs",
        "Debtor",
        "Tuition fees up to date",
        "Gender",
        "Scholarship holder",
        "International",
    ]

    def build_transformer(self) -> ColumnTransformer:
        """Build the unfitted feature-transformation pipeline.

        Assembles a :class:`~sklearn.compose.ColumnTransformer` with three
        branches:

        - ``num``: :class:`~sklearn.preprocessing.StandardScaler` over
          ``NUMERIC_FEATURES``.
        - ``cat``: :class:`~sklearn.preprocessing.OneHotEncoder`
          (``handle_unknown='ignore'``, ``min_frequency=0.01``, dense
          output) over ``CATEGORICAL_FEATURES``.
        - ``bin``: ``'passthrough'`` over ``BINARY_FEATURES``.

        The transformer is returned unfitted so callers control when and on
        which split it is fitted (fit on train, transform train/test/demo).

        Returns:
            ColumnTransformer: The unfitted transformer ready to be fitted.
        """
        # OneHotEncoder renamed ``sparse`` -> ``sparse_output`` in sklearn
        # 1.2. Detect which keyword the installed version supports so the
        # pipeline builds on both old and new versions.
        try:
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=0.01,
                sparse_output=False,
            )
        except TypeError:  # pragma: no cover - legacy sklearn (<1.2)
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=0.01,
                sparse=False,
            )

        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.NUMERIC_FEATURES),
                ("cat", encoder, self.CATEGORICAL_FEATURES),
                ("bin", "passthrough", self.BINARY_FEATURES),
            ]
        )

    @classmethod
    def validate_against(cls, columns: List[str]) -> None:
        """Validate the declared feature groups against actual columns.

        Confirms that the union of ``NUMERIC_FEATURES``,
        ``CATEGORICAL_FEATURES`` and ``BINARY_FEATURES`` is exactly equal,
        as a set, to ``columns`` — no missing groups members, no extra
        columns, and no duplicates across the groups.

        Args:
            columns: The actual feature column names (e.g. from
                ``X_train.columns.tolist()``).

        Raises:
            ValueError: If a column in ``columns`` is not covered by any
                group, if a group member is absent from ``columns``, or if
                a feature appears in more than one group.

        Returns:
            None: Returns silently when the groups match exactly.
        """
        grouped: List[str] = (
            cls.NUMERIC_FEATURES + cls.CATEGORICAL_FEATURES + cls.BINARY_FEATURES
        )

        # Detect duplicates across the groups before set comparison hides them.
        duplicates = sorted(
            {feat for feat in grouped if grouped.count(feat) > 1}
        )

        grouped_set = set(grouped)
        columns_set = set(columns)

        missing = sorted(columns_set - grouped_set)  # in columns, no group
        extra = sorted(grouped_set - columns_set)  # in groups, not in columns

        if duplicates or missing or extra:
            messages: List[str] = []
            if missing:
                messages.append(
                    f"columns not covered by any group ({len(missing)}): {missing}"
                )
            if extra:
                messages.append(
                    f"group features not present in columns ({len(extra)}): {extra}"
                )
            if duplicates:
                messages.append(
                    f"features assigned to multiple groups ({len(duplicates)}): "
                    f"{duplicates}"
                )
            raise ValueError(
                "Preprocessor feature groups do not match the provided columns. "
                + "; ".join(messages)
            )
