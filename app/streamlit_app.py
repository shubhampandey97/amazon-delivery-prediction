import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load(r"D:\Guvi\Projects\mini\Amazon Delivery Time Prediction\Amazon_delivery_time_prediction\models\linear_regression_model.pkl")

st.title("🚚 Amazon Delivery Time Predictor")
st.write("Predict the estimated delivery time based on order details.")

# User Inputs
st.header("📦 Order Details")

agent_age = st.slider("Agent Age", 18, 60, 30)
agent_rating = st.slider("Agent Rating", 1.0, 5.0, 4.5, 0.1)
distance_km = st.slider("Distance (km)", 0.0, 50.0, 5.0, 0.1)
time_to_pickup = st.slider("Time to Pickup (minutes)", 0, 60, 10)
order_hour = st.slider("Order Hour (24h format)", 0, 23, 12)
order_day = st.selectbox("Day of Week", list(range(7)), format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])

weather = st.selectbox("Weather", ['Sunny', 'Stormy', 'Sandstorms', 'Cloudy', 'Windy', 'Fog'])
traffic = st.selectbox("Traffic", ['Low', 'Medium', 'High', 'Jam'])
vehicle = st.selectbox("Vehicle Type", ['bike', 'motorcycle', 'scooter', 'bicycle'])
area = st.selectbox("Area Type", ['Urban', 'Metropolitian'])
category = st.selectbox("Product Category", ['Clothing', 'Electronics', 'Toys', 'Cosmetics', 'Sports'])

# Prediction
if st.button("🔍 Predict Delivery Time"):
    input_df = pd.DataFrame([{
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

    prediction = model.predict(input_df)[0]
    st.success(f"📦 Estimated Delivery Time: **{prediction:.2f} minutes**")


# # Make Sure You’re in the Right Conda Environment
# conda activate base  # or your existing conda env name

# # How to Run This App
# streamlit run streamlit_app.py
