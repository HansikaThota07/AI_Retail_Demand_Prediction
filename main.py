import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error


# Load the real feature dataset
data = pd.read_csv("retail_demand_features.csv")

print("Real retail dataset loaded successfully!")
print("Total records:", len(data))


# Convert Date column
data["Date"] = pd.to_datetime(data["Date"])

# Sort by date
data = data.sort_values("Date")


# Use the latest 100,000 records for faster training/testing
data = data.tail(100000).copy()

print("Records used for model:", len(data))


# Input features
features = [
    "Description",
    "Average_Price",
    "Previous_Demand",
    "Rolling_7_Demand",
    "DayOfWeek",
    "Month"
]

target = "Demand"


# Time-based split
split_index = int(len(data) * 0.8)

train_data = data.iloc[:split_index]
test_data = data.iloc[split_index:]


X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]


# Convert product names into numbers
preprocessor = ColumnTransformer(
    transformers=[
        ("product", OneHotEncoder(handle_unknown="ignore"), ["Description"])
    ],
    remainder="passthrough"
)


# Faster Random Forest
model = RandomForestRegressor(
    n_estimators=20,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)


# Complete ML pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# Train
print("Training model...")

pipeline.fit(X_train, y_train)

print("Model trained successfully!")


# Test
predictions = pipeline.predict(X_test)

error = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", round(error, 2))


# Save model
joblib.dump(pipeline, "retail_demand_model.pkl")

print("Model saved as retail_demand_model.pkl")