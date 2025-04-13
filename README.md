# Bank_of_Zambia_Exchange_Rate
Exchange Rate Prediction Project
Description
This project uses Python to analyze and predict exchange rates using machine learning techniques. It combines historical exchange rate data with advanced statistical methods to train a Random Forest model for time-series forecasting.

The model predicts MID Rates based on features like volatility, rolling averages, and lagged exchange rates. Residual analysis is used to evaluate model performance and pinpoint areas of improvement.

Features
Data Collection: Dynamically fetch exchange rate data from an Excel file hosted online.

Data Cleaning: Preprocess and handle missing values, create time-based features, and generate lagged/rolling features.

Machine Learning Model: Train a Random Forest Regressor to predict MID Rates.

Visualization Tools: Scatter plots and residual analysis to assess model accuracy.

Model Saving: Serialize the trained model using pickle for future use.

Installation
Prerequisites:
Python 3.2

Libraries: pandas, numpy, sklearn, matplotlib, pickle, tensorflow (for advanced models like LSTM)

Steps:
Clone the repository:

bash
git clone https://github.com/your-repo.git
cd your-repo
Install dependencies:

bash
pip install -r requirements.txt
Ensure the hosted Excel file is accessible:

URL: https://www.boz.zm/DAILY_RATES.xlsx

Usage
1. Run Data Analysis:
Use the following script to analyze and preprocess data:

python
import pandas as pd

url = 'https://www.boz.zm/DAILY_RATES.xlsx'
data = pd.read_excel(url)
print(data.head())
2. Train a Random Forest Model:
python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
3. Visualize Residuals:
python
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("Predicted vs Actual MID Rates")
plt.xlabel("Actual MID Rates")
plt.ylabel("Predicted MID Rates")
plt.show()
4. Save and Load the Model:
python
import pickle

with open('exchange_rate_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('exchange_rate_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
new_predictions = loaded_model.predict(X_test)
print(new_predictions[:10])

Contributing
Contributions are welcome! Feel free to open issues, submit bug reports, or improve the codebase with pull requests.
