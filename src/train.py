import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os
from pathlib import Path


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor


mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("delivery_time_prediction")

BASE_DIR = Path(__file__).resolve().parent.parent

data_path = BASE_DIR / "data" / "processed" / "final_data.csv"
model_dir = BASE_DIR / "models"

# Create models folder if not exists
model_dir.mkdir(exist_ok=True)

df = pd.read_csv(data_path)

# Drop unnecessary columns
df = df.drop(columns=["Order_ID"], errors='ignore')

# Select features
# features = ['Agent_Age', 'Agent_Rating', 'distance_km', 'weekday']
target = 'Delivery_Time'

num_cols = [
    'Agent_Age',
    'Agent_Rating',
    'distance_km',
    'prep_time'
]

cat_cols = [
    'Weather',
    'Traffic',
    'Vehicle',
    'Area',
    'Category'
]

# X = df[features]
X = df[num_cols + cat_cols]
y = df[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ]
)

# Model
# model = GradientBoostingRegressor()
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6
)

# Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Hyperparameter tuning
param_grid = {
    'model__n_estimators': [100, 200],
    'model__learning_rate': [0.05, 0.1],
    'model__max_depth': [3, 5]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

with mlflow.start_run():

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    preds = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    # rmse = mean_squared_error(y_test, preds, squared=False)
    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Log metrics
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    # Log model to MLflow
    mlflow.sklearn.log_model(best_model, "model")

    # ALSO save locally (for Streamlit)
    model_path = model_dir / "best_model.pkl"
    features_path = model_dir / "features.pkl"

    joblib.dump(best_model, model_path)
    features = num_cols + cat_cols
    joblib.dump(features, features_path)

    print(f"✅ Model saved at: {model_path}")
    print(f"✅ Features saved at: {features_path}")

    print("Model trained and logged in MLflow")