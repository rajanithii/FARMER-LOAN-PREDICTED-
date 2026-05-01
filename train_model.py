"""
Train the Farmer Loan Default Prediction model and save it as loan_model.pkl
Run this script once to generate the model before launching the Streamlit app.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import os

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join("data", "loan_data.csv"))

# ── Step 3a: Handle Missing Values ───────────────────────────────────────────
print("Missing values before cleaning:")
print(df.isnull().sum())

df.fillna(df.median(numeric_only=True), inplace=True)
df.dropna(inplace=True)

print(f"\nMissing values after cleaning: {df.isnull().sum().sum()}")
print(f"Dataset shape after cleaning: {df.shape}")

# ── Step 3b: Encode Categorical Features using LabelEncoder ──────────────────
cat_cols = ['crop_type', 'repayment_history', 'soil_quality', 'irrigation_type', 'state']

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])
    print(f"{col}: {list(le.classes_)}")

# ── Features & target ─────────────────────────────────────────────────────────
FEATURES = [
    "age", "land_area_acres", "annual_income", "crop_type",
    "loan_amount", "loan_tenure_months", "previous_loans",
    "repayment_history", "soil_quality", "irrigation_type",
    "credit_score", "state", "rainfall_mm", "avg_temp_celsius"
]

X = df[FEATURES]
y = df["default"]

# ── Train / test split (80/20) ────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ── Model ─────────────────────────────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=4,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
print(f"\nAccuracy : {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

# ── Save model ────────────────────────────────────────────────────────────────
joblib.dump(model, "loan_model.pkl")
print("✅ Model saved to loan_model.pkl")
