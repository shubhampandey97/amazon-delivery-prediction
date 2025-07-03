import streamlit as st
import pandas as pd
import joblib

# Load the best saved model (GradientBoosting or any best performing one)
model = joblib.load(r"D:\Guvi\Projects\mini\Amazon Delivery Time Prediction\Amazon_delivery_time_prediction\models\best_model.pkl")

# Title
st.title("🚛 Amazon Delivery Time Predictor")
st.markdown("Predict the estimated delivery time using the best trained model (e.g., Gradient Boosting Regressor).")

# Input form
st.header("📋 Input Delivery Details")

agent_age = st.slider("Agent Age", 18, 60, 30)
agent_rating = st.slider("Agent Rating", 1.0, 5.0, 4.5, 0.1)
distance_km = st.slider("Distance (km)", 0.0, 50.0, 5.0, 0.1)
time_to_pickup = st.slider("Time to Pickup (minutes)", 0, 60, 10)
order_hour = st.slider("Order Hour", 0, 23, 12)
order_day = st.selectbox("Day of the Week", list(range(7)), format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])

weather = st.selectbox("Weather", ['Sunny', 'Stormy', 'Sandstorms', 'Cloudy', 'Windy', 'Fog'])
traffic = st.selectbox("Traffic", ['Low', 'Medium', 'High', 'Jam'])
vehicle = st.selectbox("Vehicle Type", ['bike', 'motorcycle', 'scooter', 'bicycle'])
area = st.selectbox("Delivery Area", ['Urban', 'Metropolitian'])
category = st.selectbox("Product Category", ['Clothing', 'Electronics', 'Toys', 'Cosmetics', 'Sports'])

# Create input DataFrame
input_data = pd.DataFrame([{
    'Agent_Age': agent_age,
    'Agent_Rating': agent_rating,
    'Distance_km': distance_km,
    'Time_To_Pickup': time_to_pickup,
    'Order_Hour': order_hour,
    'Order_DayOfWeek': order_day,
    'Weather': weather,
    'Traffic': traffic,
    'Vehicle': vehicle,
    'Area': area,
    'Category': category
}])

# Prediction
if st.button("📦 Predict Delivery Time"):
    prediction = model.predict(input_data)[0]
    st.success(f"⏱ Estimated Delivery Time: **{prediction:.2f} minutes**")
