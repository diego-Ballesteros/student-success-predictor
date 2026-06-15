# Student Success Predictor

An **early-warning system** that flags currently-enrolled university students whose
academic and financial profile resembles that of students who historically dropped
out. A binary classifier (Dropout vs Graduate) is trained on students with resolved
outcomes, then applied to the currently-enrolled population — students whose future
is still undetermined — to estimate dropout risk. SHAP explains *why* each student is
flagged, and a Claude-based LLM turns those attributions into a concise,
counselor-actionable report. The result is a complete pipeline from raw data to
human-readable intervention guidance, combining classical ML with LLM explainability.

---

## Problem Statement

This is a supervised **binary classification** problem: **Dropout (1) vs Graduate (0)**.

The UCI dataset ships three target classes — Dropout, Enrolled, Graduate — but
"Enrolled" is not a resolved outcome; it is a snapshot of students whose academic
history was still open when the data was collected. EDA showed that Enrolled students
are academically much closer to Graduates than to Dropouts (2nd-semester grade mean
11.12 vs 12.70 for Graduate, vs 5.90 for Dropout), so a 3-class model is forced to
separate two classes that look alike, diluting the signal that actually distinguishes
positive from negative resolutions.

We therefore **reframed the task as binary classification on the 3,630 resolved
students** (39.1% Dropout / 60.9% Graduate) and treat the **794 Enrolled students as
the real-world inference set** — the population an early-warning system is meant to
serve. Their outcomes are unknown, so they are used for *inference*, never for
training or evaluation. Primary metrics are **F1** and **ROC-AUC** (not accuracy,
given the class imbalance), with class weighting to handle the 39/61 balance. See
[`docs/ADR_001_binary_classification.md`](docs/ADR_001_binary_classification.md) for
the full rationale and EDA evidence.

---

## Project Flow Diagram

```
        ┌─────────────────────────┐
        │   Raw Data (UCI #697)    │   4,424 students × 36 features
        └────────────┬────────────┘
                     │
                     ▼
   ┌──────────────────────────────────────┐
   │ EDA & Feature Engineering (Notebook 01)│  +approval_rate_sem1/2, grade_delta
   └────────────────────┬───────────────────┘
                     │
        ┌────────────┴─────────────┐
        ▼                          ▼
┌──────────────────┐      ┌──────────────────────┐
│ Resolved set     │      │ Enrolled set         │
│ Dropout/Graduate │      │ (inference target,   │
│ n = 3,630        │      │  no ground truth)    │
└────────┬─────────┘      │  n = 794             │
         │                └───────────┬──────────┘
         ▼                            │
┌──────────────────────────┐         │
│ 4-Model CV Comparison     │         │
│ LR · RF · GBM · HistGBM   │         │
└────────────┬──────────────┘         │
         ▼                            │
┌──────────────────────────┐         │
│ Hyperparameter Tuning     │         │
│ (top 2: GBM, HistGBM)     │         │
└────────────┬──────────────┘         │
         ▼                            │
┌──────────────────────────┐         │
│ Final Model + One-time    │         │
│ Held-out Test Evaluation  │         │
└────────────┬──────────────┘         │
         ▼                            │
┌──────────────────────────┐         │
│ SHAP Analysis             │         │
└────────────┬──────────────┘         │
         └─────────────┬──────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │ LLM Risk Explanations             │
        │ (StudentRiskExplainer → counselor)│
        └──────────────────────────────────┘
```

---

## Dataset

- **Source:** [UCI ML Repository — "Predict Students' Dropout and Academic Success"](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) (`fetch_ucirepo(id=697)`)
- **Size:** 4,424 students, 36 original features spanning demographics, application
  data, family background, financial/social factors, semester-level academic
  performance, and macroeconomic indicators.
- **Binary pivot:** 3,630 resolved students (Dropout/Graduate) for training and
  evaluation; 794 Enrolled students held aside as the inference target.
- **Engineered features:** `approval_rate_sem1`, `approval_rate_sem2`, `grade_delta`.
- Full column definitions: [`data/data_dictionary.md`](data/data_dictionary.md).

---

## Model Card

### Model Details
- **Architecture:** `HistGradientBoostingClassifier` wrapped in a scikit-learn
  `Pipeline` (preprocessing → model).
- **Task:** binary classification (Dropout=1 vs Graduate=0).
- **Preprocessing:** `StandardScaler` (22 numeric), `OneHotEncoder`
  (`handle_unknown='ignore'`, `min_frequency=0.01`; 9 categorical), passthrough
  (8 binary) → 39 raw features expand to ~108 model inputs.
- **Class imbalance:** handled via balanced `sample_weight` at fit time (HistGBM does
  not support `class_weight`).

### Intended Use
- An **early-warning aid for academic counselors** to prioritize outreach.
- **NOT** for automated decisions. Every flag requires human review; the model
  supports, not replaces, professional judgment.

### Training Data
- 2,904 students (80% of the resolved set), 39 features, **stratified** split.

### Evaluation Data
- 726 held-out test students (20% of the resolved set), touched **exactly once**.

### Metrics (from `artifacts/metrics_comparison.csv`)
| Stage | F1 | ROC-AUC | Precision | Recall |
|---|---|---|---|---|
| Offline CV (tuned, HistGBM) | 0.8786 | 0.9478 | 0.9068 | 0.8523 |
| **Online test (HistGBM)** | **0.9010** | **0.9698** | **0.8742** | **0.9296** |

### Ethical Considerations
- Predictions must **support, not replace** human judgment.
- Risk of **stigmatization** if a "risk" label is misused or shared inappropriately.
- The feature set includes **demographic variables** (nationality, parental
  occupation/qualification) — deployments should **monitor for disparate impact**
  across groups.

### Caveats
- Data comes from a **single Portuguese institution (collected ~2021)**; the model
  may not generalize to other institutions, countries, or cohorts without
  recalibration.

---

## Results

Offline (cross-validation) vs online (held-out test), from
`artifacts/metrics_comparison.csv`:

| Model | Evaluation | F1 | ROC-AUC | Precision | Recall |
|---|---|---|---|---|---|
| Hist Gradient Boosting | offline_cv_default | 0.8751 | 0.9452 | 0.9125 | 0.8408 |
| Gradient Boosting | offline_cv_default | 0.8749 | 0.9495 | 0.8858 | 0.8646 |
| Logistic Regression | offline_cv_default | 0.8679 | 0.9450 | 0.8751 | 0.8611 |
| Random Forest | offline_cv_default | 0.8677 | 0.9465 | 0.8956 | 0.8417 |
| Gradient Boosting | offline_cv_tuned | 0.8774 | 0.9475 | 0.8964 | 0.8593 |
| Hist Gradient Boosting | offline_cv_tuned | 0.8786 | 0.9478 | 0.9068 | 0.8523 |
| **Hist Gradient Boosting** | **online_test** | **0.9010** | **0.9698** | **0.8742** | **0.9296** |

**Interpretation.** All four models cluster tightly in the 0.87–0.88 F1 / ~0.95
ROC-AUC range — a near-tie that reflects the strong, almost linearly separable signal
in the data (see EDA findings). Gradient Boosting and Hist Gradient Boosting were the
two strongest candidates and were tuned; their tuned F1 scores landed within 0.0012,
so the final pick (Hist Gradient Boosting) was made on a ROC-AUC tiebreak. On the
held-out test set the chosen model **exceeded** its CV estimates (F1 0.90 vs 0.879,
ROC-AUC 0.97 vs 0.948), with **recall ≈ 0.93** — the metric that matters most for an
early-warning system, since missing an at-risk student is costlier than a false alarm.

---

## Key EDA Findings

From [`docs/DATA_INSIGHTS.md`](docs/DATA_INSIGHTS.md):

- **Two distinct dropout profiles** share the same label:
  - **Profile A — Academic dropout (~51%):** 2nd-semester grade of exactly 0,
    `approval_rate_sem2` near 0 — disengaged academically.
  - **Profile B — Non-academic dropout (~49%):** reasonable grades but strong
    **financial-distress signals** (`Tuition fees up to date = 0`, `Debtor = 1`),
    often with a sharply negative `grade_delta` — left despite performing fine.
- **Feature signal map:** strongest predictors are `approval_rate_sem2`,
  `Curricular units 2nd sem (approved/grade)`, `Tuition fees up to date` (≈95% Dropout
  when not up to date), `Debtor`, and `Scholarship holder` (protective). Macroeconomic
  variables (`Unemployment rate`, `Inflation rate`, `GDP`) carry almost no per-student
  signal — they are shared across all students in a cohort.
- **Implication for explanations:** a good explainer must distinguish the two profiles
  and recommend the matching intervention (academic tutoring vs financial aid).

---

## LLM Integration

`src/llm/explainer.py` defines **`StudentRiskExplainer`**, which takes the output of
`ModelEvaluator.explain_instance()` (prediction, dropout probability, and the top-5
SHAP features with direction) and prompts **Claude (`claude-haiku-4-5-20251001`)** to
produce a ≤150-word counselor-facing report. The report states the risk level,
classifies the student as an **academic** vs **financial** (vs both/neither) risk
profile per the two-profile framework above, and recommends **one concrete
intervention**. It is applied to the **794 enrolled students** — the real inference
use case — in Notebook 02, Section 10. The API key is loaded from `.env`
(`ANTHROPIC_API_KEY`) via `python-dotenv`.

---

## Repository Structure

```
student-success-predictor/
├── CLAUDE.md                  # Project instructions
├── PROGRESS.md                # Project status
├── ERRORS_AND_LEARNINGS.md    # Error log & learnings
├── README.md
├── pyproject.toml / uv.lock   # UV project definition
├── data/
│   ├── processed/             # X_train/X_test/y_*/enrolled_demo (parquet)
│   └── data_dictionary.md
├── notebooks/
│   ├── 01_preprocessing.ipynb # EDA + feature engineering
│   └── 02_ml_and_llm.ipynb    # Modeling + SHAP + LLM
├── src/
│   ├── data/                  # DataLoader, Preprocessor
│   ├── models/                # AbstractBaseModel, classifiers, trainer, evaluator
│   └── llm/                   # StudentRiskExplainer
├── artifacts/                 # best_model.pkl, metrics_comparison.csv, plots
├── docs/                      # ADR_001, DATA_INSIGHTS
└── tests/                     # pytest suites (preprocessor/models/trainer/evaluator/loader)
```

---

## Git Strategy

Follows **GitHub Flow**:
- `main` — production-ready only; never committed to directly.
- `dev` — active development branch.
- **Feature branches** (`feature/...`) for discrete units of work, merged into `dev`.
- All merges happen via **Pull Request** with a description.
- This project is delivered via a **`dev → main` PR** and tagged as **release
  `v1.0.0`**.

---

## How to Run

```bash
# 1. Install dependencies (UV — never use pip directly)
uv sync

# 2. Provide your Anthropic API key
cp .env.example .env   # then set ANTHROPIC_API_KEY=sk-ant-...

# 3. Launch Jupyter and run the notebooks in order
uv run jupyter lab
#   - notebooks/01_preprocessing.ipynb   (EDA + processed data)
#   - notebooks/02_ml_and_llm.ipynb      (modeling + SHAP + LLM)

# 4. Run the test suite
uv run pytest tests/ -v
```

---

## Conclusions

- **The binary pivot was validated by results.** Reframing to Dropout vs Graduate
  produced directly interpretable metrics and a genuine early-warning use case, with
  the model exceeding its CV estimates on the held-out test set.
- **Near-tied model performance reflects a strong, near-linear signal.** All four
  classifiers landed within ~0.007 F1; `approval_rate_sem2` alone nearly separates the
  classes, so model choice mattered less than feature quality.
- **Two dropout profiles were identified and addressed.** EDA surfaced an academic and
  a financial dropout profile; the LLM explainer distinguishes them and recommends a
  matching intervention (tutoring vs financial aid).
- **Recall was prioritized** (≈0.93 on test) so the system catches at-risk students,
  accepting more false positives as the safer trade-off for early warning.
- **The pipeline is reproducible and tested** — SOLID-structured `src/` modules,
  leakage-safe cross-validation, a one-time test evaluation, and a pytest suite.
- **Limitations:** a single institution and a single time period; demographic features
  warrant disparate-impact monitoring; enrolled-student predictions are inference and
  cannot be validated against ground truth.
