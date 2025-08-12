import pickle
import numpy as np

# Load the saved models
with open("../models/breast_model.pkl", "rb") as f:
    breast_model = pickle.load(f)

with open("../models/lung_model.pkl", "rb") as f:
    lung_model = pickle.load(f)

print("✅ Models loaded successfully!")

# Example input (You can later replace this with actual user input)
# Breast Cancer Example: radius_mean, texture_mean, smoothness_mean
breast_sample = np.array([15.0, 20.0, 0.1]).reshape(1, -1)
lung_sample = np.array([65, 1, 2, 1, 0]).reshape(1, -1)  # Example features

# Predict breast cancer
breast_pred = breast_model.predict(breast_sample)
print("🩺 Breast Cancer Prediction:", "Malignant" if breast_pred[0] == 1 else "Benign")

# Predict lung cancer
lung_pred = lung_model.predict(lung_sample)
print("🩺 Lung Cancer Prediction:", "Cancer" if lung_pred[0] == 1 else "No Cancer")
