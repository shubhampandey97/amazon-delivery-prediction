import streamlit as st
import pandas as pd
import joblib

# Title
st.title("📦 Amazon Delivery Time Predictor")
st.markdown("Predict delivery time based on multiple models like Linear Regression, Random Forest, and Gradient Boosting.")

# Model selection
st.sidebar.header("⚙️ Choose Model")
model_choice = st.sidebar.selectbox("Select Model", [
    "Linear Regression",
    "Random Forest",
    "Gradient Boosting"
])

# Map to model paths
model_paths = {
    "Linear Regression": r"D:\Guvi\Projects\mini\Amazon Delivery Time Prediction\Amazon_delivery_time_prediction\models\linear_regression_model.pkl",
    "Random Forest": r"D:\Guvi\Projects\mini\Amazon Delivery Time Prediction\Amazon_delivery_time_prediction\models\random_forest_model.pkl",
    "Gradient Boosting": r"D:\Guvi\Projects\mini\Amazon Delivery Time Prediction\Amazon_delivery_time_prediction\models\gradient_boosting_model.pkl"
}

# Load selected model
model = joblib.load(model_paths[model_choice])

# Input form
st.header("📝 Input Order Details")

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
if st.button("🔍 Predict Delivery Time"):
    prediction = model.predict(input_data)[0]
    st.success(f"⏱ Predicted Delivery Time using **{model_choice}**: **{prediction:.2f} minutes**")
