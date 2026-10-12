"""Central configuration: paths, column definitions and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw" / "heart_disease_raw.csv"
DATA_CLEAN = ROOT / "data" / "processed" / "heart_clean.csv"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
MODEL_INFO_PATH = MODELS_DIR / "model_info.json"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
TARGET = "target"
MLFLOW_EXPERIMENT = "heart-disease-risk"

# UCI "processed.cleveland.data" column order (last column "num" is 0-4 severity)
RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Created inside the pipeline by ClinicalFeatures (see src/features.py)
ENGINEERED_FEATURES = ["hr_reserve_pct", "chol_per_age"]

FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest pain type (1-4)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true)",
    "restecg": "Resting ECG result (0-2)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina (1 = yes)",
    "oldpeak": "ST depression induced by exercise",
    "slope": "Slope of peak exercise ST segment (1-3)",
    "ca": "Number of major vessels coloured by fluoroscopy (0-3)",
    "thal": "Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)",
}
