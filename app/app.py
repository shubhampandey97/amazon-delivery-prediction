import streamlit as st
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # go up one level
model_path = os.path.join(BASE_DIR, "models", "model.pkl")

model = joblib.load(model_path)

st.title("🚚 Delivery Time Prediction")

age = st.slider("Agent Age", 18, 50)
rating = st.slider("Agent Rating", 1.0, 5.0)
distance = st.number_input("Distance (km)")
weekday = st.selectbox("Day of Week", list(range(7)))

if st.button("Predict"):
    input_data = np.array([[age, rating, distance, weekday]])
    prediction = model.predict(input_data)

    st.success(f"Estimated Delivery Time: {prediction[0]:.2f} hours")