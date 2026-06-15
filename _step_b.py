import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import pandas as pd

from src.data.preprocessor import Preprocessor
from src.models.trainer import ModelTrainer
from src.models.classifiers import GradientBoostingModel, HistGradientBoostingModel

proc = Path("data/processed")
X_train = pd.read_parquet(proc / "X_train.parquet")
y_train = pd.read_parquet(proc / "y_train.parquet").squeeze("columns")
print(f"X_train: {X_train.shape}, balance: {dict(y_train.value_counts())}\n")

trainer = ModelTrainer(Preprocessor().build_transformer(), n_splits=5, random_state=42)

candidates = [GradientBoostingModel(), HistGradientBoostingModel()]
records = []

for model in candidates:
    # Step A reference: default (untuned) CV metrics
    print(f"[default CV] {model.name} ...", flush=True)
    default_cv = trainer.cross_validate(model, X_train, y_train)

    # Step B: tune, then full CV with best params
    print(f"Tuning {model.name} (30 trials)...", flush=True)
    tuned = trainer.tune_with_optuna(model, X_train, y_train, n_trials=30, metric="f1")
    best_params = tuned["best_params"]
    print(f"  -> best_params: {best_params}", flush=True)

    tuned_cv = trainer.cross_validate(model, X_train, y_train, params=best_params)

    records.append({
        "model_name": model.name,
        "default_f1_mean": default_cv["f1_mean"],
        "default_roc_auc_mean": default_cv["roc_auc_mean"],
        "f1_mean": tuned_cv["f1_mean"],
        "f1_std": tuned_cv["f1_std"],
        "roc_auc_mean": tuned_cv["roc_auc_mean"],
        "roc_auc_std": tuned_cv["roc_auc_std"],
        "precision_mean": tuned_cv["precision_mean"],
        "recall_mean": tuned_cv["recall_mean"],
        "best_params": str(best_params),
    })

df = pd.DataFrame(records)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

print("\n=== BEFORE/AFTER TUNING (Step A default vs Step B tuned) ===")
before_after = df[["model_name", "default_f1_mean", "f1_mean",
                   "default_roc_auc_mean", "roc_auc_mean"]].copy()
before_after["f1_delta"] = before_after["f1_mean"] - before_after["default_f1_mean"]
before_after["roc_auc_delta"] = before_after["roc_auc_mean"] - before_after["default_roc_auc_mean"]
print(before_after.round(4).to_string(index=False))

print("\n=== TUNED CV METRICS (full) ===")
tuned_tbl = df[["model_name", "f1_mean", "f1_std", "roc_auc_mean",
                "roc_auc_std", "precision_mean", "recall_mean", "best_params"]]
print(tuned_tbl.round(4).to_string(index=False))

# Decide winner: primary f1_mean; tiebreak roc_auc_mean if |df1| < 0.002
ranked = df.sort_values("f1_mean", ascending=False).reset_index(drop=True)
top, second = ranked.iloc[0], ranked.iloc[1]
f1_gap = top["f1_mean"] - second["f1_mean"]

if f1_gap < 0.002:
    winner = ranked.sort_values("roc_auc_mean", ascending=False).iloc[0]
    reason = (f"f1_mean gap ({f1_gap:.4f}) < 0.002 -> tiebreak on roc_auc_mean; "
              f"{winner['model_name']} wins with roc_auc_mean={winner['roc_auc_mean']:.4f}")
else:
    winner = top
    reason = f"highest tuned f1_mean ({winner['f1_mean']:.4f}), gap {f1_gap:.4f} >= 0.002"

print("\n=== FINAL WINNER (CV-only decision, X_test untouched) ===")
print(f"Winner: {winner['model_name']}")
print(f"Why   : {reason}")
print(f"Tuned CV: f1_mean={winner['f1_mean']:.4f} (std {winner['f1_std']:.4f}), "
      f"roc_auc_mean={winner['roc_auc_mean']:.4f}")
print(f"best_params: {winner['best_params']}")
