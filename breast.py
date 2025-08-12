# src/breast.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv("data/breast_cancer.csv")

# Keep only the features used in app.py
features = [
    "mean radius",
    "mean texture",
    "mean smoothness",
    "mean compactness",
    "mean symmetry"
]
X = df[features]
y = df["target"]  # Assuming your CSV has a 'target' column (0 = benign, 1 = malignant)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save the model
with open("models/breast_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Breast cancer model trained and saved!")
