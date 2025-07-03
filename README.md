---

## 📦 Amazon Delivery Time Prediction

This project predicts delivery time for Amazon-style e-commerce orders using machine learning models based on product, agent, and environmental conditions.
---

### 🚀 Live Demo (Optional)

If deployed:

> 🔗 [Streamlit App Demo](----)

---

## 📌 Project Overview

* **Objective**: Predict delivery time using features like distance, traffic, weather, etc.
* **Domain**: E-commerce & Logistics
* **Tech Stack**: Python, Pandas, Scikit-learn, MLflow, Streamlit, Geopy

---

## 🔍 Features

* Data Cleaning & Feature Engineering
* Exploratory Data Analysis (EDA)
* Multiple regression models:

  * Linear Regression
  * Random Forest
  * Gradient Boosting
* MLflow for model tracking
* Streamlit UI for real-time predictions
* Distance calculation using geospatial coordinates

---

## 🧾 Dataset Description

File: `amazon_delivery.csv`

| Feature            | Description                                      |
| ------------------ | ------------------------------------------------ |
| Agent\_Age         | Age of delivery agent                            |
| Agent\_Rating      | Rating of agent (1.0 to 5.0)                     |
| Store/Drop Lat/Lon | Geolocation for pickup and delivery              |
| Order/Pickup Time  | Timestamps for order and pickup                  |
| Weather, Traffic   | Environmental conditions                         |
| Vehicle, Area      | Delivery mode and zone                           |
| Category           | Product category                                 |
| Delivery\_Time     | Target variable – actual delivery time (minutes) |

---

## 🏗 Project Structure

```
amazon-delivery-prediction/
│
├── app/                      # Streamlit UI
│   ├── streamlit_app.py
|   ├── best_model_streamlit_app.py
|   └── multi_model_streamlit_app.py
│
├── data/                     # Data files
│   └── amazon_delivery_cleaned.csv
│
├── models/                   # Trained ML models
|   ├── best_models.pkl
│   ├── linear_regression_model.pkl
│   ├── random_forest_model.pkl
│   └── gradient_boosting_model.pkl
│
├── notebooks/                # EDA and preprocessing notebooks
│   ├── eda.ipynb
│   ├── data_preprocessing.ipynb
│   └── model_training.ipynb
│
├── scripts/                  # Model training scripts
│   ├── train_linear_model.py
│   ├── train_random_forest.py
│   └── train_gradient_boosting.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ How to Run Locally

1. **Clone the repository**

```bash
git clone https://github.com/your-username/amazon-delivery-prediction.git
cd amazon-delivery-prediction
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\\Scripts\\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the Streamlit App**

```bash
streamlit run app/streamlit_app.py
```

---

## 📊 MLflow Tracking

To start MLflow UI:

```bash
mlflow ui
```

Navigate to: `http://127.0.0.1:5000` to view experiment tracking results.

---

## 📈 Model Evaluation Metrics

| Model             | RMSE  | MAE   | R²   |
| ----------------- | ----- | ----- | ---- |
| Linear Regression | 21.54 | 17.62 | 0.72 |
| Random Forest     | 14.89 | 11.65 | 0.85 |
| Gradient Boosting | 13.42 | 10.98 | 0.88 |

*(Example values – update with actuals)*

---

## 👨‍💻 Author

**Shubham Pandey**
GitHub: [@shubhampandey97](https://github.com/shubhsmpandey97)

---

Would you like this file uploaded into your folder structure as `README.md` when tools are available? Or generate `requirements.txt` next?
