# ERRORS AND LEARNINGS — Student Success Predictor

Template for each entry:
## ERROR: [short title]
- **Date**: 
- **Context**: where in the project this happened
- **Error message**: exact error
- **Root cause**: why it happened
- **Fix**: what solved it
- **Prevention**: how to avoid it going forward

---

## KNOWN CONSTRAINTS (pre-loaded from similar projects)

### Pickle serialization
- Custom classes defined in __main__ cause AttributeError when unpickled
  in a different Python process. Always use native sklearn components.

### UV on Windows
- Always use `uv run python script.py` not `python script.py`
- For Jupyter: `uv run jupyter lab` not `jupyter lab`
- Kernel registration: `uv run python -m ipykernel install --user --name=student-success`

### Anthropic SDK
- API key is read automatically from ANTHROPIC_API_KEY env variable
- Use load_dotenv() before any SDK client initialization
- Model name: claude-haiku-4-5-20251001

---
## ERROR: Hidden tab character in column name
- Context: Cell 8, FEATURE_GROUPS validator
- Error: 'Daytime/evening attendance\t' not matching 'Daytime/evening attendance'
- Root cause: UCI dataset has a \t tab character at the end of that column name
- Fix: Auto-fix loop in validator strips whitespace from column names
- Prevention: Always print df.columns before hardcoding any column name
  in dictionaries or selectors

## ERROR: Case sensitivity and typos in column names
- Context: Cell 8, FEATURE_GROUPS validator  
- Error: 'Marital status' vs 'Marital Status', 'Nationality' vs 'Nacionality'
- Root cause: Dataset has typo ('Nacionality') and different capitalization
- Fix: Match exact names from df.columns output
- Prevention: Same as above — print columns first, never assume

## ERROR: sys.stdout.reconfigure fails in Jupyter notebook
- **Date**: Notebook 01 (original), RECURRED 2026-06-14
- **Context**: First code cell of a notebook, added per CLAUDE.md CODE
  STANDARDS encoding rule
- **Error message**: `AttributeError: 'OutStream' object has no attribute 'reconfigure'`
- **Root cause**: Under a Jupyter kernel `sys.stdout` is an
  `ipykernel.iostream.OutStream`, which (unlike a plain script's
  `io.TextIOWrapper`) does not implement `.reconfigure()`. The CLAUDE.md
  rule was written for `.py` scripts and applied to notebooks by mistake.
- **Fix**: Remove `import sys` + `sys.stdout.reconfigure(encoding="utf-8")`
  from notebook cells. Jupyter already handles UTF-8.
- **Prevention**: CLAUDE.md CODE STANDARDS now distinguishes `.py` scripts
  (add the call) from `.ipynb` notebooks (never add it).
- **Note**: RECURRED in Notebook 02 cell 2 (Date: 2026-06-14) — root cause
  was CLAUDE.md's CODE STANDARDS section not distinguishing .py vs .ipynb
  explicitly. Fixed by restructuring that section (see CLAUDE.md).

## ERROR: SHAP additivity check fails on HistGradientBoostingClassifier
- Context: ModelEvaluator.compute_shap_values, TreeExplainer
- Fix: pass check_additivity=False to explainer() call
- Why safe: known numerical approximation issue with HistGB's binned
  splits; SHAP value signs/rankings remain valid for explanation purposes

*Add new entries above this line as errors are encountered*