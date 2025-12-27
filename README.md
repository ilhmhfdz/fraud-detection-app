
---

# 🛡️ Credit Card Fraud Detection App

An end-to-end **Machine Learning application** to detect fraudulent credit card transactions using an **imbalanced dataset**, deployed as an **interactive Streamlit web application**.

🔗 **Live Demo**: https://fraud-detection-app-bocamqysuxjd26xrwgzutq.streamlit.app/
📦 **Repository**: [https://github.com/ilhmhfdz/fraud-detection-app](https://github.com/ilhmhfdz/fraud-detection-app)

---

## 📌 Project Overview

Credit card fraud detection is a classic and challenging problem due to:

* Highly **imbalanced data**
* High cost of **false negatives** (fraud not detected)
* Need for **fast and reliable predictions**

This project focuses not only on building an accurate model, but also on deploying it as a **robust, user-friendly, and production-ready application**.

---

## 🎯 Objectives

* Build a fraud detection model using real-world credit card transaction data
* Handle severe class imbalance using **SMOTE**
* Evaluate model performance beyond accuracy (ROC-AUC, precision, recall)
* Deploy the trained model as an interactive web app
* Ensure **input validation and fail-safe mechanisms** for real users

---

## 🧠 Machine Learning Approach

### Dataset

* Credit Card Transactions Dataset
* Features:

  * `Time`
  * `V1` – `V28` (PCA-transformed features)
  * `Amount`
* Target:

  * `Class` (0 = Legitimate, 1 = Fraud)

> ⚠️ The dataset is used **only for training locally** and is **not included in deployment** for privacy and performance reasons.

---

### Model Pipeline

* **Preprocessing**:

  * Feature scaling using `StandardScaler`
* **Imbalance Handling**:

  * SMOTE (Synthetic Minority Oversampling Technique)
* **Model**:

  * Random Forest Classifier
* **Evaluation Metrics**:

  * ROC-AUC
  * Precision / Recall
  * Confusion Matrix

The final model and scaler are persisted using `joblib`.

---

## 🚀 Application Features

* ✅ Upload CSV transactions for batch prediction
* 🔄 Generate random transactions for quick testing
* 📄 Clear CSV format requirements & downloadable template
* 🧪 Input validation to prevent runtime errors
* 📊 Fraud probability score for each transaction
* ⬇️ Download prediction results as CSV

---

## 🖥️ Tech Stack

| Layer             | Technology                     |
| ----------------- | ------------------------------ |
| Language          | Python                         |
| ML                | Scikit-learn, Imbalanced-learn |
| Model Persistence | Joblib                         |
| Web App           | Streamlit                      |
| Data Handling     | Pandas, NumPy                  |
| Deployment        | Streamlit Cloud                |

---

## 📂 Project Structure

```
fraud-detection-app/
│
├── app.py                  # Streamlit application
├── train_model.py          # Model training script (local)
├── requirements.txt
├── README.md
│
├── models/
│   ├── fraud_rf_smote.pkl  # Trained model
│   └── scaler.pkl          # Feature scaler
│
└── notebooks/
    └── exploratory_analysis.ipynb
```

---

## ⚠️ Input Data Requirements

Uploaded CSV files **must**:

* Contain **exactly 30 columns**
* Match the following order:

```
Time, V1, V2, V3, ..., V28, Amount
```

* Contain **no `Class` column**
* Include only numeric values

A CSV template is provided directly in the application.

---

## 🧩 Key Engineering Considerations

* Strict **data contract enforcement**
* Input validation to prevent app crashes
* Separation between:

  * Training pipeline (offline)
  * Inference pipeline (online)
* No dependency on training data during deployment

---

## 📈 Results Summary

* Successfully handled extreme class imbalance
* Achieved strong ROC-AUC performance
* Reduced false negatives compared to baseline models
* Delivered a stable and deployable ML system

---

## 🧠 Lessons Learned

* High accuracy alone is misleading for imbalanced datasets
* Robust ML systems require strong validation and UX design
* Deployment constraints are as important as model performance

---

## 👤 Author

**Ilham Hafidz**
Fresh Graduate – Informatics
Interested in Data Science, Machine Learning, and System Analysis

📫 Email: [ilhamhafidz666@gmail.com](mailto:ilhamhafidz666@gmail.com)

---

## 📌 Notes

This project is intended for **educational and portfolio purposes** and demonstrates an end-to-end ML workflow from data preprocessing to deployment.

---

