import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os
from pathlib import Path


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("delivery_time_prediction")

BASE_DIR = Path(__file__).resolve().parent.parent

data_path = BASE_DIR / "data" / "processed" / "final_data.csv"
model_dir = BASE_DIR / "models"

# Create models folder if not exists
model_dir.mkdir(exist_ok=True)

df = pd.read_csv("data/processed/final_data.csv")

# Select features
features = ['Agent_Age', 'Agent_Rating', 'distance_km', 'weekday']
target = 'Delivery_Time'

X = df[features]
y = df[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

with mlflow.start_run():

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    # rmse = mean_squared_error(y_test, preds, squared=False)
    rmse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Log metrics
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    # Log model to MLflow
    mlflow.sklearn.log_model(model, "model")

    # ALSO save locally (for Streamlit)
    model_path = model_dir / "model.pkl"
    features_path = model_dir / "features.pkl"

    joblib.dump(model, model_path)
    joblib.dump(features, features_path)

    print(f"✅ Model saved at: {model_path}")
    print(f"✅ Features saved at: {features_path}")

    print("Model trained and logged in MLflow")