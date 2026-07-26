# Install Streamlit 
!pip install streamlit
import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model and encoder once at startup (cached so they don't reload on every interaction)
@st.cache_resource
def load_artifacts():
    # Note: These filenames must match the ones used when saving the model and encoder.
    # Based on the previous cells, these should be 'churn_prediction_model.pkl' and 'one_hot_encoder.pkl'
    with open("churn_prediction_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("one_hot_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

model, encoder = load_artifacts()

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("Customer Renewal Probability Predictor")
st.write("Enter customer attributes to predict the likelihood of subscription renewal.")

# Dynamically get categories from the fitted encoder
income_level_options = list(encoder.categories_[0])
education_options = list(encoder.categories_[1])
device_type_options = list(encoder.categories_[2])

age               = st.number_input("Age", min_value=18, max_value=100, value=35)
income_level      = st.radio("Income Level",  income_level_options)
education         = st.radio("Education",     education_options)
device_type       = st.radio("Device Type",   device_type_options)
tech_comfort_score = st.number_input("Tech Comfort Score", min_value=1, max_value=10, value=5)

if st.button("Predict"):

    # Build categorical DataFrame — column names must match encoder exactly
    raw = pd.DataFrame([{
        'INCOME_LEVEL': income_level,
        'EDUCATION':    education,
        'DEVICE_TYPE':  device_type,
    }])

    # Apply the saved encoder (transform only — never fit_transform)
    encoded = encoder.transform(raw)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

    # Numeric features first, then encoded dummies — must match training column order
    numeric_df = pd.DataFrame([{
        'AGE':                age,
        'TECH_COMFORT_SCORE': tech_comfort_score,
    }])

    input_df = pd.concat([numeric_df, encoded_df], axis=1)

    # Column 1 = P(renewed), column 0 = P(churned)
    probability = model.predict_proba(input_df)[0][1]
    risk = "Low" if probability >= 0.6 else "Medium" if probability >= 0.4 else "High"

    st.metric("Renewal Probability", f"{probability:.2f}")
    if risk == "High":
        st.error(f"Churn Risk: {risk}")
    elif risk == "Medium":
        st.warning(f"Churn Risk: {risk}")
    else:
        st.success(f"Churn Risk: {risk}")