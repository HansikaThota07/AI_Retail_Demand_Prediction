import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load trained model
model = joblib.load("retail_demand_model.pkl")

# Load real feature dataset
data = pd.read_csv("retail_demand_features.csv")

# Convert date
data["Date"] = pd.to_datetime(data["Date"])

# Sort by date
data = data.sort_values("Date")

# Use the same 100,000 records used for training
data = data.tail(100000).copy()

# Features
features = [
    "Description",
    "Average_Price",
    "Previous_Demand",
    "Rolling_7_Demand",
    "DayOfWeek",
    "Month"
]

target = "Demand"

# Time-based 80/20 split
split_index = int(len(data) * 0.8)

test_data = data.iloc[split_index:]

X_test = test_data[features]
y_test = test_data[target]

# Predictions
predictions = model.predict(X_test)

# Calculate errors
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\n==============================================")
print("             MODEL EVALUATION")
print("==============================================")

print("Test Records:", len(test_data))
print("Mean Absolute Error (MAE):", round(mae, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))

print("\nSample Predictions:")
print("----------------------------------------------")

for i in range(10):
    print(
        "Product:", test_data.iloc[i]["Description"],
        "| Actual:", round(y_test.iloc[i], 2),
        "| Predicted:", round(predictions[i], 2)
    )

print("==============================================")