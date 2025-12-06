# 📈 Stock Price Prediction Using LSTM  
### *A Deep Learning Approach for Forecasting NSE Stock Prices*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow" />
  <img src="https://img.shields.io/badge/NSE-Data-green" />
  <img src="https://img.shields.io/badge/Model-LSTM-purple" />
  <img src="https://img.shields.io/badge/Status-Completed-success" />
</p>

Predicting stock prices is one of the most challenging tasks in quantitative finance due to market volatility, noise, and complex temporal dependencies.  
This project leverages **Long Short-Term Memory (LSTM)** neural networks — a powerful variant of RNNs — to analyze historical data and forecast stock prices for **any NSE-listed company**.

The objective of this project is not only to build a predictive model but also to provide a **complete pipeline** including data acquisition, preprocessing, modeling, evaluation, and visualization.

---

# 🚀 Key Features

### ✔️ Fetches live historical data directly from **NSE**  
### ✔️ Complete preprocessing pipeline with scaling & cleaning  
### ✔️ Automated creation of time-series sequences  
### ✔️ Deep Learning model using **LSTM architecture**  
### ✔️ Predicts both **Open** and **Close** prices  
### ✔️ High-quality visualizations of predictions vs. actual prices  
### ✔️ Fully interactive — user inputs stock symbol & date range  

---

# 📊 Project Objectives

- **Acquire** historical stock data (Open, Close, High, Low, Volume)  
- **Clean & preprocess** the data for model compatibility  
- **Analyze** stock trends through visualizations  
- **Train** LSTM models on past patterns  
- **Evaluate** forecasting accuracy using MSE & MAE  
- **Visualize** predicted vs. actual price movement  

---

# 📦 Technologies & Libraries Used

- **Python 3.10+**
- **TensorFlow / Keras**
- **nselib** (NSE data fetching)
- **pandas**
- **NumPy**
- **Matplotlib**
- **StandardScaler (scikit-learn)**

---

# 🧠 Why LSTM?

Traditional feedforward networks fail to capture temporal relationships. LSTMs are designed to:

- Remember patterns across long sequences  
- Learn from time-dependent data  
- Handle financial market noise  
- Capture nonlinear price movements  

This makes LSTM an ideal choice for **time-series forecasting** like stock price prediction.

---

# 📥 Data Acquisition

Users provide:

- Stock symbol (e.g., `RELIANCE`)  
- Start date  
- End date  

Data is pulled using:

```python
from nselib import capital_market
stock = capital_market.price_volume_and_deliverable_position_data(...)
df = pd.DataFrame(stock)
```

# 🧹 Data Preprocessing
- ✔️ Remove commas from numerical columns
- ✔️ Convert text values to float
- ✔️ Scale values using StandardScaler
- ✔️ Create 100-day lookback sequences for LSTM


# 📊 Exploratory Data Analysis (EDA)

The scaled Open and Close prices are plotted to observe:

Trends

Volatility

Patterns before modeling


# 🏗️ LSTM Model Architecture

Both Open and Close prices are predicted using similar models.

-🔹 Model Layers:

LSTM(64, return_sequences=True)

Dropout(0.2)

LSTM(32)

Dense(16, activation='relu')

Dense(1, activation='linear')



# -🔹 Compilation:
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='mse',
    metrics=['mae']
)

# -🔹 Training:

Epochs: 50

Batch Size: 32

The model learns almost immediately with rapid loss reduction.

# 📈 Model Performance
- ✔️ Low MSE & MAE during training 
- ✔️ Strong convergence observed
- ✔️ Model learns long-term patterns effectively
- ✔️ Predictions closely follow actual values

# 📊 Prediction Visualization

The final graph compares:

- 🔵 Actual Prices

- 🔴 Predicted Prices (LSTM)

# 📂 Project Structure
📁 Stock-Price-Prediction-LSTM/
│── README.md
│── stock_prediction.ipynb
│── data/ (optional)
│── models/
│   ├── lstm_close_model.h5
│   ├── lstm_open_model.h5
│── plots/
│   ├── predicted_vs_actual.png
└── ...
