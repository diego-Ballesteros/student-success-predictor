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

### Phase 2: ML Models + Evaluation (02_ml_and_llm.ipynb — Part 1)
- [ ] src/ module classes implemented
- [ ] 4 models trained: LogisticRegression, RandomForest, GradientBoosting, XGBoost
- [ ] Stratified 5-fold CV for each model
- [ ] Best model selected by Macro F1
- [ ] Hyperparameter tuning on best model
- [ ] Offline metrics: CV results table
- [ ] Online metrics: test set evaluation
- [ ] Confusion matrix + per-class report
- [ ] SHAP analysis: summary plot + individual explanations
- [ ] Artifacts saved: best_model.pkl, metrics_comparison.csv, shap_summary.html

### Phase 3: LLM Integration (02_ml_and_llm.ipynb — Part 2)
- [ ] StudentRiskExplainer class implemented
- [ ] ANTHROPIC_API_KEY loaded from .env
- [ ] 5 sample students explained (1 Dropout, 1 Enrolled, 1 Graduate, 2 edge cases)
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