# ADR 001: Binary Classification (Dropout vs Graduate) Instead of 3-Class

## Status
Accepted

## Context

The UCI "Predict Students' Dropout and Academic Success" dataset has three
target classes: Dropout (1,421), Enrolled (794), Graduate (2,209).

During EDA, we analyzed the academic performance distributions (approved
units and grades, semesters 1 and 2) across the three classes. Two findings
drove this decision:

1. **Enrolled is not an outcome — it's a snapshot in time.** Unlike
   Dropout and Graduate, which represent closed, resolved trajectories,
   "Enrolled" simply means the student's academic history was still open
   at the moment the data was collected. Some of these students will
   eventually graduate, others will eventually drop out.

2. **Enrolled and Graduate show statistically similar academic profiles.**
   Descriptive statistics for 2nd semester grade show Enrolled
   (mean=11.12, std=3.60) and Graduate (mean=12.70, std=2.69) are much
   closer to each other than either is to Dropout (mean=5.90, std=6.12,
   with 51.2% of Dropout students having a 2nd-sem grade of exactly 0).

   This means a 3-class model is being asked to separate two classes
   that look academically alike, which dilutes the signal that clearly
   separates "resolved positively" from "resolved negatively".

## Decision

Reframe the problem as **binary classification: Dropout vs Graduate**,
trained only on students with a resolved (closed) academic outcome
(n=3,630, 39.1% Dropout / 60.9% Graduate).

The 794 "Enrolled" students are **not used for training**. Instead, they
become the **real-world application set**: the trained model is applied
to them to estimate how closely their current academic profile resembles
historically-dropout vs historically-graduate trajectories. This is the
exact use case of an early-warning system — predicting outcomes for
students whose future is still undetermined.

## Consequences

### Positive
- Metrics become directly interpretable (F1, ROC-AUC, precision/recall)
  instead of macro-F1 across 3 imbalanced classes.
- Class imbalance is less severe (39/61 vs the original 18% minority class).
- The LLM explainer component gains a genuinely meaningful demo: generating
  risk explanations for real students whose outcome is unknown, not for
  test-set students whose label we already have.
- The project's framing ("early warning system") becomes literal rather
  than aspirational — the model is applied exactly to the population it
  claims to serve.

### Negative / Trade-offs
- We no longer report a 3-class macro-F1, which was part of the original
  project framing. This is addressed by documenting this ADR and showing
  the EDA evidence that motivated the change.
- The Enrolled subset has no ground truth, so model performance on it
  cannot be measured — only offline (CV) and online (held-out test set)
  metrics on the Dropout/Graduate split are reportable. This is the
  correct framing: Enrolled predictions are *inference*, not *evaluation*.

## Evidence

See notebooks/01_preprocessing.ipynb, cells 10-11 and the descriptive
statistics table (cell added after 11), which show:

| Target   | 2nd sem grade — Median | 2nd sem grade — Mean | 2nd sem grade — Std |
|----------|------------------------|-----------------------|----------------------|
| Dropout  | 0.0                    | 5.90                  | 6.12                 |
| Enrolled | 12.0                   | 11.12                 | 3.60                 |
| Graduate | 13.0                   | 12.70                 | 2.69                 |

51.2% of Dropout students (727/1,421) have a 2nd-semester grade of exactly 0.
