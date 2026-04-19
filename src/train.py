import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import shap



mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("delivery_time_prediction")

BASE_DIR = Path(__file__).resolve().parent.parent

data_path = BASE_DIR / "data" / "processed" / "final_data.csv"
model_dir = BASE_DIR / "models"
outputs_dir = BASE_DIR / "assets"
outputs_dir.mkdir(exist_ok=True)

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


    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    importances = best_model.named_steps['model'].feature_importances_

    indices = np.argsort(importances)[-10:]

    plt.figure(figsize=(10,6))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title("Feature Importance")
    plt.savefig(outputs_dir / "feature_importance.png", bbox_inches='tight')

    print("Feature Impotance plot is added.")

    # Cross Validation
    cv_scores = cross_val_score(
        best_model,
        X,
        y,
        cv=5,
        scoring='neg_mean_squared_error'
    )

    rmse_scores = np.sqrt(-cv_scores)

    print("CV RMSE Scores:", rmse_scores)
    print("Mean CV RMSE:", rmse_scores.mean())

    # Log in MLflow
    mlflow.log_metric("CV_RMSE_MEAN", rmse_scores.mean())
    mlflow.log_metric("CV_RMSE_STD", rmse_scores.std())

    plt.figure(figsize=(8,5))
    plt.plot(rmse_scores, marker='o')
    plt.title("Cross Validation RMSE")
    plt.xlabel("Fold")
    plt.ylabel("RMSE")
    plt.grid()

    plt.savefig(outputs_dir / "cv_results.png", bbox_inches='tight')
    plt.close()


    # Extract components
    preprocessor = best_model.named_steps['preprocessor']
    model = best_model.named_steps['model']

    # Transform data
    X_transformed = preprocessor.transform(X)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

    # Sample for performance (VERY IMPORTANT)
    X_sample = X_transformed_df.sample(200, random_state=42)

    # 🔥 KernelExplainer (FINAL FIX)
    explainer = shap.KernelExplainer(model.predict, X_sample)

    shap_values = explainer.shap_values(X_sample)

    # Plot
    shap.summary_plot(shap_values, X_sample, show=False)

    plt.savefig(outputs_dir / "shap_summary.png", bbox_inches='tight')
    plt.close()