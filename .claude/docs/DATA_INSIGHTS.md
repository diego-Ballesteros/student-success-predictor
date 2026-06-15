# DATA_INSIGHTS — Student Success Predictor

Knowledge derived from EDA in notebooks/01_preprocessing.ipynb.
This is NOT a data dictionary (see data/data_dictionary.md for that).
This file captures PATTERNS and their MODELING IMPLICATIONS — read this
before building models, interpreting SHAP, or designing LLM explanations.

---

## 1. The Binary Pivot (see ADR_001 for full rationale)

- Original 3 classes: Dropout (1,421) / Enrolled (794) / Graduate (2,209)
- Enrolled ≈ Graduate academically (2nd sem grade: mean 11.12 vs 12.70,
  std 3.60 vs 2.69) — too similar to be a useful 3rd class
- Pivoted to BINARY: Dropout (1) vs Graduate (0), trained on 3,630
  resolved students (~39%/61%)
- The 794 Enrolled students are the INFERENCE target (enrolled_demo.parquet),
  not training data — they have no ground truth

---

## 2. TWO DISTINCT "DROPOUT" PROFILES — critical for SHAP & LLM explainer

Dropout is NOT one homogeneous group. EDA revealed two clearly different
populations sharing the same label:

### Profile A — "Academic Dropout" (~51% of Dropout students)
- 2nd semester grade = exactly 0 (727/1,421 students, 51.2%)
- approval_rate_sem2 near 0 (violin plot: dense mass at 0)
- Never engaged academically in 2nd semester — stopped attending/evaluating
- grade_delta near 0 (started at 0, stayed at 0) OR strongly negative if
  sem1 grade was normal

### Profile B — "Non-Academic Dropout" (~49% of Dropout students)
- 2nd semester grade in normal/passing range (~10-16, similar to Graduate)
- approval_rate may be reasonable
- BUT: strongly associated with financial distress signals (see Section 3)
- grade_delta strongly NEGATIVE (e.g. sem1=15, sem2=0 → delta=-15):
  visible as the horizontal line at Y=0 in the grade trajectory scatter plot
- These students were performing fine academically and left anyway

### Implication for LLM explainer (StudentRiskExplainer)
When explaining a high-risk "enrolled" student, the explanation should
identify WHICH profile the student resembles:
- Profile A → recommend academic tutoring / re-engagement intervention
- Profile B → recommend financial aid / counseling intervention
- A student can show signals of both — explainer should be able to say so

### Implication for SHAP validation
If SHAP analysis does NOT show financial features (Section 3) among
top contributors for at least some high-risk predictions, the model may
be over-indexing on academic features alone and missing Profile B entirely.
This would be worth flagging.

---

## 3. FEATURE SIGNAL MAP

### Strong signal — expect these near the top of SHAP importance

| Feature | Pattern |
|---|---|
| `approval_rate_sem2` | Near-perfect separator: Dropout mass at 0, Graduate mass at 1 (violin plot) |
| `Curricular units 2nd sem (approved)` | Dropout median=0, Graduate median=6 |
| `Curricular units 2nd sem (grade)` | Dropout median=0 (but bimodal — see Section 2) |
| `approval_rate_sem1` | Similar pattern to sem2, slightly less extreme |
| `Curricular units 1st sem (approved/grade)` | Same direction, weaker than sem2 |
| `Tuition fees up to date` | If =0 (not up to date): ~95% Dropout. Very strong binary signal |
| `Debtor` | If =1: ~3:1 ratio toward Dropout. If =0: ~2:1 toward Graduate |
| `Scholarship holder` | If =1: ~7:1 ratio toward Graduate (protective factor) |
| `grade_delta` | Captures trajectory; extreme negative values (-15 to -16) flag Profile B dropouts |

### Weak/no signal — expect these near the bottom of SHAP importance

| Feature | Pattern |
|---|---|
| `Unemployment rate` | Nearly identical distribution across all 3 original classes |
| `Inflation rate` | Nearly identical across classes (Graduate slightly lower median) |
| `GDP` | Nearly identical across classes (Graduate slightly higher median) |

These are country-level macro variables shared by all students enrolled
in the same period — they don't vary per-student, so they can't explain
per-student outcomes. Kept in the model for completeness but not expected
to drive predictions.

---

## 4. ENGINEERED FEATURES — rationale recap

- **approval_rate_sem1 / approval_rate_sem2** = approved / enrolled units
  (0 if enrolled=0). Normalizes for course load — more informative than
  raw counts. approval_rate_sem2 is the single strongest candidate feature
  in the dataset (see violin plot: Dropout mass at 0, Graduate mass at 1).

- **grade_delta** = sem2_grade - sem1_grade. Range observed: -16.14 to +16.
  - Near 0: stable performance (most students)
  - Strongly negative (-10 to -16): Profile B dropout signal
    (was performing fine, then disappeared — see grade trajectory scatter,
    the horizontal line at Y=0)
  - Strongly positive (+10 to +16): student who started at 0 in sem1
    but engaged in sem2 (visible as vertical line at X=0 in scatter plot)

---

## 5. MULTICOLLINEARITY (from correlation heatmap)

Two expected clusters of correlated features:
- All "Curricular units 1st/2nd sem (*)" features correlate strongly with
  each other (same underlying signal: academic engagement/performance)
- Mother's/Father's qualification and occupation correlate with each other
  (assortative pairing)
- Nacionality correlates with International

No action needed for tree-based models (RF, GBM, XGBoost) — they handle
redundant features natively. For Logistic Regression / SVM, L2
regularization (sklearn default) handles this — no manual feature removal
required.

---

## 6. QUICK REFERENCE FOR MODEL EVALUATION

- Primary metrics: F1-score, ROC-AUC (NOT accuracy — though class balance
  39/61 is much less severe than the original 18% Enrolled minority)
- Use `class_weight='balanced'` for LogisticRegression, RandomForest, SVM
- Expected ballpark: given how strong the academic signal is (near-perfect
  separation in approval_rate_sem2), F1 scores above 0.85-0.90 would not
  be surprising for tree-based models. If a model scores far below this,
  investigate — something may be wrong with feature alignment.