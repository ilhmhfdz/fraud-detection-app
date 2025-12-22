import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Fraud Detection App",
    page_icon="🛡️",
    layout="centered"
)

# =========================
# PATH SETUP
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "fraud_rf_smote.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# =========================
# LOAD MODEL & SCALER
# =========================
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_artifacts()

EXPECTED_COLUMNS = list(scaler.feature_names_in_)

# =========================
# UI HEADER
# =========================
st.title("🛡️ Credit Card Fraud Detection App")
st.write(
    "A machine learning application to predict whether a credit card transaction is **Fraud** or **Legitimate**."
)

# =========================
# DATA CONTRACT INFO
# =========================
with st.expander("📄 Required CSV Format (Click to expand)", expanded=True):
    st.markdown("""
    **Your CSV file must meet these requirements:**

    - Contains **30 columns**
    - **NO `Class` column**
    - Column order **must match exactly**
    - Numeric values only

    **Expected columns:**
    ```
    Time, V1, V2, V3, ..., V28, Amount
    ```
    """)

# =========================
# CSV TEMPLATE DOWNLOAD
# =========================
template_df = pd.DataFrame(columns=EXPECTED_COLUMNS)
template_csv = template_df.to_csv(index=False)

st.download_button(
    label="⬇️ Download CSV Template",
    data=template_csv,
    file_name="transaction_template.csv",
    mime="text/csv"
)

st.divider()

# =========================
# RANDOM TRANSACTION TEST
# =========================
st.subheader("🔄 Try Random Transaction")

if st.button("Generate Random Transaction"):
    random_data = pd.DataFrame(
        [np.random.normal(0, 1, size=len(EXPECTED_COLUMNS))],
        columns=EXPECTED_COLUMNS
    )

    scaled_data = scaler.transform(random_data)
    pred = model.predict(scaled_data)[0]
    prob = model.predict_proba(scaled_data)[0][1]

    st.success("Prediction completed!")
    st.write("### Result:")
    st.write("🚨 **FRAUD**" if pred == 1 else "✅ **LEGITIMATE**")
    st.write(f"Fraud Probability: **{prob:.2%}**")

st.divider()

# =========================
# FILE UPLOAD
# =========================
st.subheader("📤 Upload Transaction CSV")

uploaded_file = st.file_uploader(
    "Upload CSV file (using provided template)",
    type=["csv"]
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Validate columns
        if list(df.columns) != EXPECTED_COLUMNS:
            st.error("❌ CSV format is invalid.")
            st.write("Expected columns:")
            st.write(EXPECTED_COLUMNS)
            st.stop()

        # Scale data
        X_scaled = scaler.transform(df)

        # Predict
        preds = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)[:, 1]

        df_result = df.copy()
        df_result["Fraud_Prediction"] = preds
        df_result["Fraud_Probability"] = probs

        st.success("✅ Prediction successful!")
        st.write("### Prediction Preview")
        st.dataframe(df_result.head())

        # Download results
        result_csv = df_result.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Prediction Results",
            data=result_csv,
            file_name="fraud_prediction_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error("⚠️ An error occurred during processing.")
        st.code(str(e))

# =========================
# FOOTER
# =========================
st.divider()
st.caption(
    "Built by Ilham Hafidz | Machine Learning & Data Analysis Project"
)