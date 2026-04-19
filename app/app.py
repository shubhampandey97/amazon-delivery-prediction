import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from pathlib import Path
import shap
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # go up one level
model_path = os.path.join(BASE_DIR, "models", "best_model.pkl")
final_data_path = os.path.join(BASE_DIR, "data", "processed", "final_data.csv")

model = joblib.load(model_path)

st.set_page_config(page_title="Delivery Time Prediction", layout="centered")
st.title("🚚 Smart Delivery Time Prediction")
st.markdown("Enter delivery details to estimate delivery time")
st.markdown("---")

# Inputs
age = st.slider("Agent Age", 18, 50)
rating = st.slider("Agent Rating", 1.0, 5.0)

distance = st.number_input("Distance (km)")
prep_time = st.number_input("Preparation Time (minutes)")

weather = st.selectbox("Weather", ["sunny", "rainy", "foggy", "Stormy"])
traffic = st.selectbox("Traffic", ["low", "medium", "high", "Jam"])
vehicle = st.selectbox("Vehicle", ["bike", "car", "scooter"])
area = st.selectbox("Area", ["urban", "Semi-Urban", "Rural", "metropolitan"])
category = st.selectbox("Category", ["food", "grocery", "electronics"])

st.markdown("---")

if st.button("Predict"):
    input_df = pd.DataFrame([{
        'Agent_Age': age,
        'Agent_Rating': rating,
        'distance_km': distance,
        'prep_time': prep_time,
        'Weather': weather,
        'Traffic': traffic,
        'Vehicle': vehicle,
        'Area': area,
        'Category': category
    }])
    prediction = model.predict(input_df)

    st.subheader("📊 Prediction Result")
    st.success(f"Estimated Delivery Time: {prediction[0]:.2f} minutes")

    st.markdown("---")

    # ---------------- SHAP ---------------- #
    st.subheader("🔍 Feature Impact (SHAP)")

    preprocessor = model.named_steps['preprocessor']
    reg_model = model.named_steps['model']

    # Transform user input
    X_transformed = preprocessor.transform(input_df)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

    # 🔥 BACKGROUND DATA (IMPORTANT)
    background = pd.read_csv(final_data_path).sample(100, random_state=42)

    background_transformed = preprocessor.transform(background)

    if hasattr(background_transformed, "toarray"):
        background_transformed = background_transformed.toarray()

    background_df = pd.DataFrame(background_transformed, columns=feature_names)

    # SHAP
    explainer = shap.KernelExplainer(reg_model.predict, background_df)
    shap_values = explainer.shap_values(X_transformed_df)

    # 🔥 USE BAR PLOT (WORKS BEST)
    fig, ax = plt.subplots()
    shap.summary_plot(
        shap_values, 
        X_transformed_df, 
        plot_type="bar", 
        show=False
    )
    st.pyplot(fig)
    plt.close(fig)