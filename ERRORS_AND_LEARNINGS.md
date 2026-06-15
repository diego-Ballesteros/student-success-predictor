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

*Add new entries above this line as errors are encountered*