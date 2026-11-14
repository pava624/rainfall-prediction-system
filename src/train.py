
# =========================================================
# RAINFALL PREDICTION SYSTEM
# Complete Machine Learning Project
# =========================================================

# 1. Import Libraries

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# 2. Project Paths

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR, "data", "weatherAUS.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# 3. Load Dataset

print("\n======================================")
print("       RAINFALL PREDICTION SYSTEM")
print("======================================")

print("\nStep 1: Loading Dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Original Dataset Shape:", df.shape)


# 4. Data Preprocessing

print("\nStep 2: Data Preprocessing...")

df = df.drop_duplicates()

df = df.dropna(subset=["RainTomorrow"])

df["Date"] = pd.to_datetime(
    df["Date"], errors="coerce"
)

df = df.dropna(subset=["Date"])

df = df.sort_values("Date").reset_index(drop=True)

# Remove leakage column if present
df = df.drop(
    columns=["RISK_MM"],
    errors="ignore"
)

print("Duplicates removed.")
print("Missing target records removed.")
print("Date conversion completed.")


# 5. Feature Engineering

print("\nStep 3: Feature Engineering...")

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

df = df.drop(columns=["Date"])

df["RainTomorrow"] = (
    df["RainTomorrow"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"yes": 1, "no": 0})
)

df = df.dropna(subset=["RainTomorrow"])

print("Year, Month and Day features created.")
print("Target values converted into numbers.")


# 6. Separate Input and Output

print("\nStep 4: Separating Features and Target...")

X = df.drop(columns=["RainTomorrow"])
y = df["RainTomorrow"].astype(int)

print("Input features:", X.shape)
print("Target values:", y.shape)

print("\nRainfall Distribution:")
print(y.value_counts())


# 7. Chronological Train-Test Split

print("\nStep 5: Splitting Dataset...")

split_index = int(len(X) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# 8. Identify Feature Types

print("\nStep 6: Identifying Features...")

numerical_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("Numerical features:", len(numerical_features))
print("Categorical features:", len(categorical_features))


# 9. Handle Missing Values and Encoding

print("\nStep 7: Preparing Preprocessing Pipeline...")

numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])

print("Preprocessing pipeline created.")


# 10. Create Random Forest Classifier

print("\nStep 8: Building Random Forest Model...")

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

print("Random Forest model created.")


# 11. Train the Model

print("\nStep 9: Training Model...")

model.fit(X_train, y_train)

print("Model training completed successfully!")


# 12. Generate Predictions

print("\nStep 10: Generating Predictions...")

y_pred = model.predict(X_test)

print("Predictions generated successfully.")


# 13. Evaluate Model

print("\nStep 11: Evaluating Model...")

accuracy = accuracy_score(y_test, y_pred)

print("\n======================================")
print("           MODEL PERFORMANCE")
print("======================================")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["No Rain", "Rain"],
    zero_division=0
))


# 14. Generate Confusion Matrix

print("\nStep 12: Generating Confusion Matrix...")

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Rain", "Rain"]
)

display.plot(cmap="Blues")

plt.title("Rainfall Prediction Confusion Matrix")
plt.tight_layout()

plt.savefig(
    os.path.join(REPORT_DIR, "confusion_matrix.png")
)

plt.close()

print("Confusion matrix saved successfully.")


# 15. Save Trained Model

print("\nStep 13: Saving Trained Model...")

model_path = os.path.join(
    MODEL_DIR,
    "rainfall_model.pkl"
)

joblib.dump(model, model_path)

print("Trained model saved successfully!")
print("Model path:", model_path)


# 16. Save Sample Input

sample = X_test.iloc[[0]].copy()

sample.to_csv(
    os.path.join(REPORT_DIR, "sample_weather.csv"),
    index=False
)

print("Sample weather input saved.")


# 17. Final Summary

print("\n======================================")
print("       PROJECT EXECUTION COMPLETED")
print("======================================")

print("Data Preprocessing : Completed")
print("Feature Engineering: Completed")
print("Model Training     : Completed")
print("Model Evaluation   : Completed")
print("Model Saving       : Completed")

print(f"\nFinal Accuracy: {accuracy * 100:.2f}%")

print("\nAll operations completed successfully!")

