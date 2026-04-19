import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # go up one level
model_path = os.path.join(BASE_DIR, "models", "best_model.pkl")

model = joblib.load(model_path)

st.title("🚚 Smart Delivery Time Prediction")

# Inputs
age = st.slider("Agent Age", 18, 50)
rating = st.slider("Agent Rating", 1.0, 5.0)

distance = st.number_input("Distance (km)")
prep_time = st.number_input("Preparation Time (minutes)")

weather = st.selectbox("Weather", ["sunny", "rainy", "foggy"])
traffic = st.selectbox("Traffic", ["low", "medium", "high"])
vehicle = st.selectbox("Vehicle", ["bike", "car", "scooter"])
area = st.selectbox("Area", ["urban", "metropolitan"])
category = st.selectbox("Category", ["food", "grocery", "electronics"])

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

    st.success(f"Estimated Delivery Time: {prediction[0]:.2f} minutes")