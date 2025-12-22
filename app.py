import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "fraud_rf_smote.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ===============================
# UI
# ===============================
st.set_page_config(page_title="Fraud Detection App", layout="centered")

st.title("Credit Card Fraud Detection")
st.write(
    """
    This app detects whether a transaction is **Fraud** or **Non-Fraud**
    using a Machine Learning model (RandomForest + SMOTE).
    """
)

# ===============================
# Upload CSV
# ===============================
st.subheader(" Upload Transaction Data (CSV)")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("Preview Data:")
    st.dataframe(df.head())

    if st.button("Run Prediction"):
        X = df.drop(columns=["Class"], errors="ignore")
        X_scaled = scaler.transform(X)

        probs = model.predict_proba(X_scaled)[:, 1]
        preds = (probs >= 0.3).astype(int)  # custom threshold

        df["Fraud_Probability"] = probs
        df["Prediction"] = df["Fraud_Probability"].apply(
            lambda x: "Fraud" if x >= 0.3 else "Non-Fraud"
        )

        st.subheader("🔍 Prediction Result")
        st.dataframe(df[["Fraud_Probability", "Prediction"]].head(20))

        st.write("Fraud Count:")
        st.write(df["Prediction"].value_counts())

# ===============================
# Demo Mode
# ===============================
st.subheader("Demo Mode (Random Transaction)")

if st.button("Try Random Transaction"):
    sample = pd.read_csv(BASE_DIR / "data" / "creditcard.csv").sample(1)
    X_sample = sample.drop(columns=["Class"])
    X_scaled = scaler.transform(X_sample)

    prob = model.predict_proba(X_scaled)[0, 1]

    st.write("Transaction Data:")
    st.dataframe(X_sample)

    st.metric(
        label="Fraud Probability",
        value=f"{prob:.2%}"
    )

    if prob >= 0.3:
        st.error("⚠️ Fraud Detected")
    else:
        st.success("✅ Non-Fraud Transaction")