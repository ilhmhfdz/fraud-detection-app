import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "fraud_model.joblib"

joblib.dump(rf_smote, MODEL_PATH)
print("Model saved to:", MODEL_PATH)