---

## 🛒 Walmart Weekly Sales Forecasting

This repository contains a comprehensive time series forecasting project focused on predicting weekly sales for a specific Walmart department. It is the final capstone project for a time series course and includes traditional models, machine learning, deep learning, and deployment.

---

### 📊 Project Overview

This project aims to forecast weekly sales using the **Walmart cleaned dataset**. The workflow includes:

* 📈 Exploratory time series analysis
* 🔁 Trend & seasonality decomposition
* 📉 ARIMA & exponential smoothing methods
* 🌲 Machine learning regressors (Random Forest, XGBoost)
* 🤖 Deep learning models (ANN, CNN, RNN, LSTM)
* 🧙‍♂️ Facebook Prophet with holiday effects & regressors
* 🚀 Innovative technique: **N-BEATS**
* 📦 Deployment of best model via Flask API

---

### 📁 Repository Structure

```
├── Time_Series_Project_.ipynb       # Full notebook with analysis, models, and evaluation
├── rnn_model.keras      # Trained RNN model saved in Keras format
├── app.py               # Flask API to serve model predictions
├── test.py              # Pytest file to test API endpoints
└── README.md            # Project description
```

---

### 🚀 API Usage

**GET** `/`
Returns health message.

**POST** `/predict`
Accepts JSON with a 40-length sequence of sales values.

```json
{
  "sequence": [0.45, 0.51, ..., 0.39]  // length = 40
}
```

Returns predicted next sales value:

```json
{
  "prediction": [[0.42]]
}
```

---

### 🏆 Best Model

✅ RNN outperformed other models based on RMSE and deployment suitability.

💡 Effectively incorporated multiple external regressors: Temperature, Fuel Price, CPI, and Unemployment.

⚡ Provides fast inference and a lightweight architecture, ideal for real-time Flask API deployment.

🔁 Successfully captured short- and mid-term temporal dependencies in weekly sales data.

---

### 📽️ Presentation

A slide deck summarizing this project’s journey, challenges, and insights is included [here](#) (link if available).

---

### 👩‍💻 Author

**Mayar Mostafa Hassan**
Artificial Intelligence Engineer
Alexandria University

---
