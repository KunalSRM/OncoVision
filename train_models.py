# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
# import joblib

# # -------------- Breast cancer model training ------------------

# print("Training model from ../data/breast_cancer.csv...")
# breast_df = pd.read_csv("../data/breast_cancer.csv")

# print(f"Breast cancer data shape: {breast_df.shape}")
# print(f"Columns: {breast_df.columns.tolist()}")

# print("Checking missing values in breast cancer dataset: ")
# print(breast_df.isnull().sum())

# # Drop the 'Unnamed: 32' column (all NaNs)
# if 'Unnamed: 32' in breast_df.columns:
#     breast_df = breast_df.drop(columns=['Unnamed: 32'])

# # Features and target
# X_breast = breast_df.drop(columns=['id', 'diagnosis'])
# # Map diagnosis 'M' and 'B' to 1 and 0
# breast_df['diagnosis'] = breast_df['diagnosis'].map({'B': 0, 'M': 1})
# y_breast = breast_df['diagnosis']

# print(f"Features shape: {X_breast.shape}, Target shape: {y_breast.shape}")

# Xb_train, Xb_test, yb_train, yb_test = train_test_split(X_breast, y_breast, test_size=0.2, random_state=42)

# print(f"Train shape: {Xb_train.shape}, Test shape: {Xb_test.shape}")

# breast_model = RandomForestClassifier(random_state=42)
# breast_model.fit(Xb_train, yb_train)

# yb_pred = breast_model.predict(Xb_test)
# print("Classification report for breast_model.pkl:")
# print(classification_report(yb_test, yb_pred))

# joblib.dump(breast_model, "../models/breast_model.pkl")
# print("✅ Model saved at: ../models/breast_model.pkl\n")

# # -------------- Lung cancer model training ------------------

# print("Training model from ../data/lung_cancer.csv...")
# lung_df = pd.read_csv("../data/lung_cancer.csv")

# print(f"Lung cancer data shape: {lung_df.shape}")
# print(f"Columns: {lung_df.columns.tolist()}")

# print("Missing values in lung cancer dataset:")
# print(lung_df.isnull().sum())

# print(f"Unique values in GENDER before mapping: {lung_df['GENDER'].unique()}")

# cols_to_map = ["SMOKING","YELLOW_FINGERS","ANXIETY","PEER_PRESSURE","CHRONIC DISEASE",
#                "FATIGUE ","ALLERGY ","WHEEZING","ALCOHOL CONSUMING","COUGHING",
#                "SHORTNESS OF BREATH","SWALLOWING DIFFICULTY","CHEST PAIN"]

# # Show unique values for these columns before mapping
# for col in cols_to_map:
#     if col in lung_df.columns:
#         print(f"{col} unique values before mapping: {lung_df[col].unique()}")

# print(f"Unique values in LUNG_CANCER before mapping: {lung_df['LUNG_CANCER'].unique()}")

# # Mapping dictionaries based on your dataset values
# mapping_gender = {"M": 0, "F": 1}
# mapping_binary = {1: 1, 2: 0}  # Assuming 1=Yes, 2=No based on your data
# mapping_target = {"NO": 0, "YES": 1}  # Exact uppercase mapping for target

# # Map gender
# lung_df['GENDER'] = lung_df['GENDER'].map(mapping_gender)

# # Map Yes/No columns with 1/0 (note: data has 1 and 2 integers, map accordingly)
# for col in cols_to_map:
#     if col in lung_df.columns:
#         lung_df[col] = lung_df[col].map(mapping_binary)

# # Map target column with exact uppercase keys
# lung_df['LUNG_CANCER'] = lung_df['LUNG_CANCER'].map(mapping_target)

# print("Missing values after mapping:")
# print(lung_df.isnull().sum())

# # Drop rows with any NaNs (just in case)
# lung_df = lung_df.dropna()

# # Features and target
# X_lung = lung_df.drop(columns=["LUNG_CANCER"])
# y_lung = lung_df["LUNG_CANCER"]

# print(f"Features shape: {X_lung.shape}, Target shape: {y_lung.shape}")

# Xl_train, Xl_test, yl_train, yl_test = train_test_split(X_lung, y_lung, test_size=0.2, random_state=42)

# print(f"Train shape: {Xl_train.shape}, Test shape: {Xl_test.shape}")

# lung_model = RandomForestClassifier(random_state=42)
# lung_model.fit(Xl_train, yl_train)

# yl_pred = lung_model.predict(Xl_test)
# print("Classification report for lung_model.pkl:")
# print(classification_report(yl_test, yl_pred))

# joblib.dump(lung_model, "../models/lung_model.pkl")
# print("✅ Model saved at: ../models/lung_model.pkl\n")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from imblearn.over_sampling import SMOTE

# ----- Breast Cancer Model Training -----

print("Training model from ../data/breast_cancer.csv...")
breast_df = pd.read_csv("../data/breast_cancer.csv")

print(f"Breast cancer data shape: {breast_df.shape}")
print(f"Columns: {breast_df.columns.tolist()}")

print("Checking missing values in breast cancer dataset:")   
print(breast_df.isnull().sum())

# Drop 'Unnamed: 32' column if exists
if 'Unnamed: 32' in breast_df.columns:
    breast_df = breast_df.drop(columns=['Unnamed: 32'])

X_breast = breast_df.drop(columns=['id', 'diagnosis'])
y_breast = breast_df['diagnosis'].map({'B': 0, 'M': 1})  # Benign=0, Malignant=1

print(f"Features shape: {X_breast.shape}, Target shape: {y_breast.shape}")

Xb_train, Xb_test, yb_train, yb_test = train_test_split(X_breast, y_breast, test_size=0.2, random_state=42)

print(f"Train shape: {Xb_train.shape}, Test shape: {Xb_test.shape}")

breast_model = RandomForestClassifier(random_state=42)
breast_model.fit(Xb_train, yb_train)

yb_pred = breast_model.predict(Xb_test)
print("Classification report for breast_model.pkl:")
print(classification_report(yb_test, yb_pred))

joblib.dump(breast_model, "../models/breast_model.pkl")
print("✅ Model saved at: ../models/breast_model.pkl\n")

# ----- Lung Cancer Model Training with SMOTE & GridSearch -----

print("Training model from ../data/lung_cancer.csv...")

lung_df = pd.read_csv("../data/lung_cancer.csv")
print(f"Lung cancer data shape: {lung_df.shape}")
print(f"Columns: {lung_df.columns.tolist()}")

print("Missing values in lung cancer dataset:")
print(lung_df.isnull().sum())

# Map GENDER
mapping_gender = {"M": 0, "F": 1}
lung_df["GENDER"] = lung_df["GENDER"].map(mapping_gender)

# List of binary columns with 1/2 encoding to map 1 -> 0 (No), 2 -> 1 (Yes)
cols_to_map = ["SMOKING","YELLOW_FINGERS","ANXIETY","PEER_PRESSURE","CHRONIC DISEASE",
               "FATIGUE ","ALLERGY ","WHEEZING","ALCOHOL CONSUMING","COUGHING",
               "SHORTNESS OF BREATH","SWALLOWING DIFFICULTY","CHEST PAIN"]

for col in cols_to_map:
    if col in lung_df.columns:
        lung_df[col] = lung_df[col].map({1:0, 2:1})

# Map target column (YES -> 1, NO -> 0)
lung_df["LUNG_CANCER"] = lung_df["LUNG_CANCER"].map({"NO": 0, "YES": 1})

print("Missing values after mapping:")
print(lung_df.isnull().sum())

# Drop any rows with missing values just in case
lung_df = lung_df.dropna()

X_lung = lung_df.drop(columns=["LUNG_CANCER"])
y_lung = lung_df["LUNG_CANCER"]

print(f"Features shape: {X_lung.shape}, Target shape: {y_lung.shape}")

Xl_train, Xl_test, yl_train, yl_test = train_test_split(X_lung, y_lung, test_size=0.2, random_state=42)

print(f"Train shape: {Xl_train.shape}, Test shape: {Xl_test.shape}")

# Apply SMOTE to balance classes in training set
print("Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
Xl_train_res, yl_train_res = smote.fit_resample(Xl_train, yl_train)
print(f"After SMOTE, train shapes: {Xl_train_res.shape}, {yl_train_res.shape}")

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

print("Starting GridSearchCV...")
grid_search.fit(Xl_train_res, yl_train_res)

print(f"Best params: {grid_search.best_params_}")
print(f"Best ROC AUC: {grid_search.best_score_:.4f}")

best_rf = grid_search.best_estimator_

# Evaluate on test set
yl_pred = best_rf.predict(Xl_test)
yl_proba = best_rf.predict_proba(Xl_test)[:, 1]

print("Classification report for lung_model_tuned.pkl:")
print(classification_report(yl_test, yl_pred))

# Confusion matrix
cm = confusion_matrix(yl_test, yl_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve and AUC
auc_score = roc_auc_score(yl_test, yl_proba)
fpr, tpr, _ = roc_curve(yl_test, yl_proba)

plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

print(f"AUC Score: {auc_score:.4f}")

# Save the tuned model
joblib.dump(best_rf, "../models/lung_model_tuned.pkl")
print("✅ Model saved at: ../models/lung_model_tuned.pkl")
