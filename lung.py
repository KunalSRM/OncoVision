import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train_lung_model():
    df = pd.read_csv("../data/lung_cancer.csv")

    # Standardize string formatting
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Try to detect gender column
    gender_col = None
    for col in df.columns:
        if col.lower() in ["gender", "sex"]:
            gender_col = col
            break
    if gender_col:
        df[gender_col] = df[gender_col].map({'male': 0, 'female': 1, 'm': 0, 'f': 1})

    # Map Yes/No columns
    for col in df.columns:
        unique_vals = set(df[col].dropna().unique())
        if unique_vals <= {"yes", "no"}:
            df[col] = df[col].map({'no': 0, 'yes': 1})

    # Detect target column
    target_col = None
    for col in df.columns:
        if "cancer" in col.lower():
            target_col = col
            break
    if not target_col:
        raise ValueError("❌ No target column found containing 'cancer' in its name.")

    # Drop rows where target is missing
    df = df.dropna(subset=[target_col])

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42))
    pipeline.fit(X_train, y_train)

    print("✅ Lung cancer model accuracy:", pipeline.score(X_test, y_test))

    os.makedirs("../models", exist_ok=True)
    joblib.dump(pipeline, "../models/lung_model.pkl")
    print("💾 Saved lung cancer model to ../models/lung_model.pkl")

if __name__ == "__main__":
    train_lung_model()
