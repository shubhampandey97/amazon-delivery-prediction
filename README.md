# 🚚 Amazon Delivery Time Prediction

## 📌 Overview

This project focuses on predicting **Amazon delivery time** using Machine Learning. It demonstrates an end-to-end ML workflow — from data preprocessing and model training (in Jupyter Notebook) to deployment using a **Streamlit web application**.

---

## 🎯 Objective

To build a predictive model that estimates delivery time based on various input features such as order details, location, and logistics-related factors.

---

## 🧠 Tech Stack

* **Python**
* **Pandas, NumPy** – Data manipulation
* **Scikit-learn** – Model building
* **Joblib / Pickle** – Model serialization
* **Streamlit** – Deployment (UI)
* **MLflow** - Experiment tracking

---

## 📂 Project Structure

```
amazon-delivery-prediction/
│
├── data/
│   ├── raw/                  # Raw dataset
│   └── processed/            # Cleaned dataset
│
├── notebooks/
│   └── model_training.ipynb  # Initial model development
│
├── models/
│   └── model.pkl             # Trained model
│
├── src/                      # (In progress)
│   ├── train.py
│   └── preprocess.py
│
├── app.py                    # Streamlit application
├── requirements.txt
└── README.md
```

---

## ⚙️ Workflow

### 1. Data Preprocessing

* Handling missing values
* Encoding categorical variables
* Feature scaling

### 2. Model Training

* Model trained using **Scikit-learn**
* Initial experimentation done in Jupyter Notebook
* Performance evaluated using:

  * MAE (Mean Absolute Error)
  * RMSE (Root Mean Squared Error)
  * R² Score

### 3. Model Saving

* Final model saved as `.pkl` file using `joblib`

### 4. Deployment (Streamlit)

* Interactive UI built using Streamlit
* Users can input features and get delivery time predictions

---

## 🚀 How to Run the Project

### 1. Clone Repository

```bash
git clone https://github.com/shubhampandey97/amazon-delivery-prediction.git
cd amazon-delivery-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Streamlit App

```bash
streamlit run app.py
```

---

## 🔮 Future Improvements



---

## 👨‍💻 Author

**Shubham Pandey**
