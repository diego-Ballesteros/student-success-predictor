# PROGRESS — Student Success Predictor

## Current Status: PHASE 0 — Setup

---

## Phases

### Phase 0: Project Setup
- [x] Repository created and cloned
- [x] Context files designed (CLAUDE.md, PROGRESS.md, ERRORS_AND_LEARNINGS.md)
- [x] Directory structure created
- [x] pyproject.toml + UV environment initialized
- [x] .gitignore, .env.example created
- [x] Initial commit on dev branch

### Phase 1: Preprocessing Notebook (01_preprocessing.ipynb)
- [ ] Data loading via ucimlrepo
- [ ] EDA: class distribution, feature distributions, correlations
- [ ] Feature group analysis (demographic, academic, macroeconomic)
- [ ] Stratified train/test split (80/20)
- [ ] Feature engineering decisions documented
- [ ] Processed data saved to data/processed/
- [ ] ADR_001 created documenting binary classification pivot

### Phase 2: ML Models + Evaluation (02_ml_and_llm.ipynb — Part 1)
Binary classification (Dropout=1 vs Graduate=0) on the resolved-outcome set.
- [ ] src/ module classes implemented
- [ ] 4 models trained: LogisticRegression, RandomForest, GradientBoosting, XGBoost
      (with class_weight='balanced' where supported)
- [ ] Stratified 5-fold CV for each model
- [ ] Best model selected by F1 / ROC-AUC
- [ ] Hyperparameter tuning on best model
- [ ] Offline metrics: CV results table (F1 / ROC-AUC)
- [ ] Online metrics: test set evaluation (F1 / ROC-AUC)
- [ ] Confusion matrix + binary classification report
- [ ] SHAP analysis: summary plot + individual explanations
- [ ] Artifacts saved: best_model.pkl, metrics_comparison.csv, shap_summary.html

### Phase 3: LLM Integration (02_ml_and_llm.ipynb — Part 2)
Primary demo runs on the 794 Enrolled students (data/processed/enrolled_demo.parquet),
who have NO ground-truth label — the real-world inference target.
- [ ] StudentRiskExplainer class implemented
- [ ] ANTHROPIC_API_KEY loaded from .env
- [ ] 5 Enrolled students explained from enrolled_demo.parquet
      (range of estimated risk: high-risk, borderline, low-risk profiles)
- [ ] Secondary: test-set explanations used to validate explanation quality
      against known Dropout/Graduate labels
- [ ] Explanation quality validated

### Phase 4: Documentation
- [ ] data_dictionary.md completed
- [ ] README.md: problem, flow diagram, dataset, model card, results, conclusions
- [ ] Docstrings verified on all src/ classes

### Phase 5: Release
- [ ] PR dev → main merged
- [ ] Git strategy documented in README
- [ ] GitHub Release v1.0.0 with release notes

---

## Last Updated: [DATE]
## Current Branch: dev