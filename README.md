# 🚚 Amazon Delivery Time Prediction

## 📌 Project Overview

This project focuses on predicting **delivery time for e-commerce orders** using Machine Learning.
It leverages real-world features such as **distance, traffic, weather, and agent performance** to build an accurate regression model.

The final solution includes:

* Data preprocessing & feature engineering
* Model training with MLflow tracking
* A user-friendly **Streamlit web application**

---

## 🎯 Business Objective

* Improve delivery time estimation accuracy
* Enhance customer satisfaction
* Optimize logistics and delivery operations
* Analyze impact of traffic, weather, and agent efficiency

---

## 🧠 Problem Statement

Predict the **Delivery_Time (in hours)** based on:

* Order details
* Delivery agent attributes
* External conditions (traffic, weather)
* Geographical distance

---

## 🗂️ Project Structure

```
amazon-delivery-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│
├── models/
│   └── best_model.pkl
│
├── app/
│   └── app.py
│
├── mlruns/        # MLflow logs
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

* **Programming:** Python
* **Libraries:** pandas, numpy, scikit-learn, xgboost
* **Visualization:** matplotlib, seaborn
* **Model Tracking:** MLflow
* **Web App:** Streamlit

---

## 🔄 Workflow

### 1. Data Preprocessing

* Removed duplicates & missing values
* Standardized categorical variables

### 2. Feature Engineering

* 📍 **Distance Calculation (Haversine Formula)**
* ⏱️ **Preparation Time (Order → Pickup)**
* 📅 Time-based features (day, weekday)

### 3. Exploratory Data Analysis (EDA)

* Delivery time distribution
* Distance vs delivery time relationship
* Impact of traffic and weather

### 4. Model Development

* Built regression models:

  * Gradient Boosting
  * XGBoost (optional)
* Used **Pipeline + ColumnTransformer**
* Applied **GridSearchCV** for tuning

### 5. Model Evaluation

* Metrics used:

  * MAE (Mean Absolute Error)
  * RMSE (Root Mean Squared Error)
  * R² Score

### 6. Experiment Tracking

* Used **MLflow** to:

  * Log metrics
  * Compare models
  * Track hyperparameters

### 7. Deployment

* Built interactive UI using **Streamlit**
* Users can input delivery conditions and get predictions

---

## 🚀 How to Run the Project

### 1️⃣ Clone Repository

```
git clone https://github.com/your-username/amazon-delivery-prediction.git
cd amazon-delivery-prediction
```

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run Data Pipeline

```
python src/data_preprocessing.py
python src/feature_engineering.py
```

### 4️⃣ Train Model

```
python src/train.py
```

### 5️⃣ Run MLflow

```
mlflow ui
```

### 6️⃣ Launch App

```
streamlit run app/app.py
```

---

## 📊 Key Features

* ✅ End-to-end ML pipeline
* ✅ Feature engineering using geospatial data
* ✅ Automated preprocessing with Pipeline
* ✅ Hyperparameter tuning (GridSearchCV)
* ✅ MLflow experiment tracking
* ✅ Interactive Streamlit UI

---

## 📈 Sample Input Features

* Agent Age & Rating
* Distance (calculated from coordinates)
* Preparation Time
* Weather Conditions
* Traffic Level
* Vehicle Type
* Area Type

---

## 🧪 Results

* Built multiple regression models and compared performance
* Identified key factors affecting delivery time:

  * Distance
  * Traffic conditions
  * Preparation time

---

## 🚀 Project Evolution

| Version | Improvement | Impact |
|--------|------------|--------|
| v1.0 | Baseline model | Initial pipeline |
| v1.1 | Feature engineering | Improved accuracy |
| v1.2 | Pipeline + preprocessing | Scalability |
| v1.3 | Hyperparameter tuning | Better performance |
| v1.4 | Code cleanup | Maintainability |
| v1.5 | Feature importance | Interpretability |

---

## 🔮 Future Improvements

* Add real-time traffic API integration
* Deploy on cloud (AWS / GCP)
* Add SHAP for model explainability
* Convert to REST API using FastAPI

---

## 👨‍💻 Author

**Shubham Pandey**
Data Science & ML Developer

---

## ⭐ If you found this project useful, consider giving it a star!
