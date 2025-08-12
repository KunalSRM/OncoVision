# import streamlit as st
# import numpy as np
# import joblib

# st.title("OncoVision - Early Cancer Detection")

# # Load model
# breast_model = joblib.load("models/breast_model.pkl")

# menu = st.sidebar.selectbox("Choose Cancer Type", ("Breast Cancer",))  # Start with breast cancer only

# if menu == "Breast Cancer":
#     st.header("Breast Cancer Prediction")

#     mean_radius = st.number_input("Mean Radius", 0.0, 50.0, 14.0)
#     mean_texture = st.number_input("Mean Texture", 0.0, 50.0, 20.0)
#     mean_perimeter = st.number_input("Mean Perimeter", 0.0, 200.0, 90.0)
#     mean_area = st.number_input("Mean Area", 0.0, 3000.0, 650.0)
#     mean_smoothness = st.number_input("Mean Smoothness", 0.0, 0.3, 0.1)

#     if st.button("Predict Breast Cancer"):
#         input_data = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]])
#         prediction = breast_model.predict(input_data)[0]
#         proba = breast_model.predict_proba(input_data)[0][prediction]

#         if prediction == 1:
#             st.error(f"Malignant tumor detected (Confidence: {proba:.2%})")
#         else:
#             st.success(f"Benign tumor detected (Confidence: {proba:.2%})")

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)  # to silence warning

st.title("OncoVision - Early Cancer Detection")

# Load models and datasets once
@st.cache_data
def load_models():
    breast_model = joblib.load("../models/breast_model.pkl")
    lung_model = joblib.load("../models/lung_model_tuned.pkl")
    return breast_model, lung_model

@st.cache_data
def load_datasets():
    breast_df = pd.read_csv("../data/breast_cancer.csv")
    lung_df = pd.read_csv("../data/lung_cancer.csv")
    return breast_df, lung_df

breast_model, lung_model = load_models()
breast_df, lung_df = load_datasets()

menu = st.sidebar.selectbox("Choose Cancer Type", ("Breast Cancer", "Lung Cancer"))

def plot_distribution(df, feature, user_val, title=None):
    plt.figure(figsize=(8,4))
    sns.histplot(df[feature], kde=True, color='skyblue', bins=30)
    plt.axvline(user_val, color='red', linestyle='--', label='Your Input')
    plt.xlabel(feature)
    plt.title(title if title else f'Distribution of {feature}')
    plt.legend()
    st.pyplot()

if menu == "Breast Cancer":
    st.header("Breast Cancer Prediction & Explanation")

    # Features used for prediction (change as per your training)
    features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']

    user_inputs = {}
    for f in features:
        mean_val = float(breast_df[f].mean())
        median_val = float(breast_df[f].median())
        user_inputs[f] = st.number_input(f.capitalize().replace('_', ' '),
                                        min_value=float(breast_df[f].min()),
                                        max_value=float(breast_df[f].max()),
                                        value=mean_val,
                                        help=f"Dataset Mean: {mean_val:.2f}, Median: {median_val:.2f}")

    if st.button("Predict Breast Cancer"):
        input_data = np.array([list(user_inputs.values())])
        prediction = breast_model.predict(input_data)[0]
        proba = breast_model.predict_proba(input_data)[0][prediction]

        if prediction == 1:
            st.error(f"Malignant tumor detected (Confidence: {proba:.2%})")
        else:
            st.success(f"Benign tumor detected (Confidence: {proba:.2%})")

        # Show explanation graphs & comparisons
        st.markdown("### Feature Value Analysis:")
        for f in features:
            st.write(f"**{f}:** Your input = {user_inputs[f]:.2f}, Dataset mean = {breast_df[f].mean():.2f}")
            if user_inputs[f] > breast_df[f].mean():
                st.write(f"- Your value is *above* the dataset average, which might indicate increased risk for cancer.")
            else:
                st.write(f"- Your value is *below* or near the dataset average, which generally indicates lower risk.")

            plot_distribution(breast_df, f, user_inputs[f])

elif menu == "Lung Cancer":
    st.header("Lung Cancer Prediction & Explanation")

    # Lung features (1 = yes, 0 = no), encode gender as Male=1 Female=0
    lung_features = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
                     'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING',
                     'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
                     'SWALLOWING DIFFICULTY', 'CHEST PAIN']

    # Prepare user inputs
    gender = st.selectbox("Gender", ("Male", "Female"))
    gender_encoded = 1 if gender == "Male" else 0

    age = st.number_input("Age", min_value=int(lung_df['AGE'].min()), max_value=int(lung_df['AGE'].max()), value=int(lung_df['AGE'].mean()))

    binary_inputs = {}
    for feature in lung_features[2:]:  # excluding GENDER and AGE
        # original dataset uses 1 and 2, convert to 0 and 1 internally
        # but we ask user for 0/1 for simplicity
        binary_inputs[feature] = st.selectbox(feature.title().replace('_', ' '), options=[0,1])

    if st.button("Predict Lung Cancer"):
        # Rebuild input vector
        input_vector = [gender_encoded, age] + list(binary_inputs.values())
        input_np = np.array([input_vector])

        prediction = lung_model.predict(input_np)[0]
        proba = lung_model.predict_proba(input_np)[0][prediction]

        if prediction == 1:
            st.error(f"Lung cancer likely (Confidence: {proba:.2%})")
        else:
            st.success(f"Lung cancer unlikely (Confidence: {proba:.2%})")

        # Explanations
        st.markdown("### Feature Value Analysis:")
        for i, feature in enumerate(lung_features):
            if feature == 'GENDER':
                user_val = gender_encoded
                desc = "Male (1)" if user_val==1 else "Female (0)"
            elif feature == 'AGE':
                user_val = age
                desc = f"{user_val} years"
            else:
                user_val = binary_inputs[feature]
                desc = "Yes (1)" if user_val==1 else "No (0)"

            # For binary features, show value counts
            if feature in ['GENDER', 'AGE']:
                st.write(f"**{feature}:** Your input = {desc}, Dataset mean = {lung_df[feature].mean():.2f}")
                if feature == 'AGE':
                    st.write(f"- Age mean in dataset is {lung_df['AGE'].mean():.1f} years.")
                    if user_val > lung_df['AGE'].mean():
                        st.write("- Older age is generally associated with higher risk.")
                    else:
                        st.write("- Younger age generally means lower risk.")
            else:
                counts = lung_df[feature].value_counts()
                st.write(f"**{feature}:** Your input = {desc}, Dataset counts: {counts.to_dict()}")
                if user_val == 1 and counts.get(1,0) > counts.get(0,0):
                    st.write(f"- Majority of patients with this symptom have it (1). Presence may increase risk.")
                elif user_val == 1:
                    st.write(f"- This symptom is less common but present in your input, could indicate risk.")
                else:
                    st.write(f"- Symptom absent (0), which generally means lower risk.")

            # Plot distribution for age and gender
            if feature == 'AGE':
                plot_distribution(lung_df, 'AGE', user_val, title="Age distribution in Lung Cancer dataset")
            elif feature == 'GENDER':
                plt.figure(figsize=(6,3))
                sns.countplot(x=lung_df['GENDER'].map({1:'Male',0:'Female'}))
                plt.title("Gender distribution in dataset")
                plt.xlabel("Gender")
                plt.ylabel("Count")
                st.pyplot()

