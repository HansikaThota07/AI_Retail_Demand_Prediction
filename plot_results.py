import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("retail_demand_model.pkl")

# Load feature dataset
data = pd.read_csv("retail_demand_features.csv")
data["Date"] = pd.to_datetime(data["Date"])

# Sort by date
data = data.sort_values("Date")

# Use the same 100,000 records
data = data.tail(100000).copy()

# Features and target
features = [
    "Description",
    "Average_Price",
    "Previous_Demand",
    "Rolling_7_Demand",
    "DayOfWeek",
    "Month"
]

target = "Demand"

# Same 80/20 time-based split
split_index = int(len(data) * 0.8)

test_data = data.iloc[split_index:]

X_test = test_data[features]
y_test = test_data[target]

# Predict
predictions = model.predict(X_test)

# Plot first 100 test records
plt.figure(figsize=(12, 6))

plt.plot(
    range(100),
    y_test.iloc[:100].values,
    label="Actual Demand"
)

plt.plot(
    range(100),
    predictions[:100],
    label="Predicted Demand"
)

plt.xlabel("Test Record")
plt.ylabel("Demand (Units)")
plt.title("Actual vs Predicted Retail Demand")
plt.legend()
plt.grid(True)

plt.tight_layout()

# Save graph
plt.savefig("actual_vs_predicted.png")

print("\nGraph created successfully!")
print("Saved as: actual_vs_predicted.png")

# Display graph
plt.show()