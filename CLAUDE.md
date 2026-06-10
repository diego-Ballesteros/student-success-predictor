# CLAUDE.md — Student Success Predictor

## PROJECT PURPOSE
Early warning system that predicts whether a university student will Graduate,
Dropout, or remain Enrolled, using enrollment data + 1st semester performance.
Enables academic counselors to intervene early with at-risk students.

Dataset: UCI "Predict Students' Dropout and Academic Success" (4,424 students, 36 features)
Problem type: Supervised multiclass classification (3 classes)
Primary metric: Macro F1-Score (class imbalance present)

---

## TECH STACK
- Package manager: UV — NEVER use pip directly
- Python: 3.11+
- ML: scikit-learn, xgboost, pandas, numpy
- Explainability: shap
- Visualization: matplotlib, seaborn, plotly
- LLM: anthropic SDK (model: claude-haiku-4-5-20251001)
- Data source: ucimlrepo (fetch_ucirepo id=697)
- Environment: python-dotenv for credentials

---

## REPOSITORY STRUCTURE

📋 Archivos de Configuración y Documentación
Raíz del Proyecto
student-success-predictor/
│
├── CLAUDE.md                  # Instrucciones para Claude Code (lectura automática)
├── PROGRESS.md                # Estado actual del proyecto
├── ERRORS_AND_LEARNINGS.md    # Registro de errores y soluciones
├── README.md                  # Documentación del proyecto
├── requirements.txt           # Dependencias pinadas
├── pyproject.toml             # Definición del proyecto UV
├── .env.example               # Template de credenciales (seguro para commit)
└── .gitignore

📂 Directorio: data/
Propósito: Almacenamiento de datos crudos y procesados
data/
├── raw/
│   └── .gitkeep              # Dataset original (no se commitea)
├── processed/
│   └── .gitkeep              # Datos transformados
└── data_dictionary.md         # Diccionario de variables

📔 Directorio: notebooks/
Propósito: Análisis exploratorio y experimentación
notebooks/
├── 01_preprocessing.ipynb     # EDA + limpieza + feature engineering
└── 02_ml_and_llm.ipynb        # Modelos + SHAP + integración LLM

🎯 Directorio: artifacts/
Propósito: Modelos entrenados y visualizaciones resultantes
artifacts/
├── best_model.pkl            # Pipeline serializado (mejor modelo)
├── metrics_comparison.csv     # Tabla comparativa de todos los modelos
└── shap_summary.html          # Visualización interactiva SHAP

🏗️ Directorio: src/ (Módulo OOP Reutilizable)
Propósito: Código modular siguiendo principios SOLID
Estructura General
src/
├── __init__.py
│
├── data/                      # Capa de datos
│   ├── __init__.py
│   ├── loader.py              # Clase DataLoader
│   └── preprocessor.py        # Clase Preprocessor
│
├── models/                    # Capa de modelos
│   ├── __init__.py
│   ├── base_model.py          # AbstractBaseModel (principio Open/Closed)
│   ├── trainer.py             # Clase ModelTrainer
│   └── evaluator.py           # Clase ModelEvaluator
│
└── llm/                       # Capa de integración LLM
    ├── __init__.py
    └── explainer.py           # Clase StudentRiskExplainer

    ---

## SOLID PRINCIPLES MAP

| Principle | Implementation |
|-----------|----------------|
| Single Responsibility | Each class has one job: DataLoader loads, Preprocessor transforms, Evaluator evaluates |
| Open/Closed | AbstractBaseModel defines interface; new models extend it without modifying existing code |
| Liskov Substitution | All model classes can replace AbstractBaseModel without breaking behavior |
| Interface Segregation | StudentRiskExplainer only knows about explaining, not about training or loading |
| Dependency Inversion | ModelTrainer depends on AbstractBaseModel abstraction, not concrete implementations |

---

## CODE STANDARDS
- Type hints required on ALL function signatures
- Google-style docstrings on ALL classes and public methods
- Use pathlib.Path for all file paths — never string concatenation
- No hardcoded credentials — always load from .env via python-dotenv
- Load .env with: load_dotenv(Path(__file__).parent.parent / ".env")
- sys.stdout.reconfigure(encoding="utf-8") at top of any script (Windows compatibility)
- Naming: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants

---

## LLM INTEGRATION DESIGN

Class: StudentRiskExplainer in src/llm/explainer.py
Model: claude-haiku-4-5-20251001 (fast + economical)
API key: loaded from ANTHROPIC_API_KEY in .env

Input to explainer:
  - student_features: dict with feature names and values
  - shap_values: dict with top 5 features and their SHAP contributions
  - prediction: str ("Dropout" | "Enrolled" | "Graduate")
  - probabilities: dict with probability per class

Output: structured natural language report for academic counselor

---

## GIT STRATEGY (GitHub Flow)
- main: production-ready only — never commit directly
- dev: active development branch
- Feature branches: feature/setup, feature/preprocessing, feature/ml-models,
  feature/llm-explainer, feature/readme
- Commit convention: feat: / fix: / docs: / refactor: / test:
- All merges via Pull Request with description

---

## ENVIRONMENT VARIABLES (.env)
ANTHROPIC_API_KEY=sk-ant-...

---

## NEVER DO
- Never use pip — always uv add or uv run
- Never hardcode API keys or paths
- Never commit .env files
- Never use custom classes defined in __main__ in pickled pipelines
- Never use accuracy as primary metric (class imbalance present)
- Never skip stratification in train/test splits