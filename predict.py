import pandas as pd
import joblib

# Load trained model
model = joblib.load("retail_demand_model.pkl")

# Load real retail data
data = pd.read_csv("retail_demand_features.csv")
data["Date"] = pd.to_datetime(data["Date"])

print("\n==============================================")
print("   AI-POWERED RETAIL DEMAND PREDICTION SYSTEM")
print("==============================================")

# Show example products
products = data["Description"].dropna().unique()

print("\nExample Products:")
for product in products[:10]:
    print(" -", product)

# Get product from user
product = input("\nEnter Product Name exactly as shown above: ")

# Find selected product
product_data = data[data["Description"] == product]

if len(product_data) == 0:
    print("\nProduct not found!")
    print("Please enter a product name exactly as shown above.")

else:
    # Sort product records by date
    product_data = product_data.sort_values("Date")

    # Latest historical record
    latest = product_data.iloc[-1]

    price = latest["Average_Price"]
    previous_demand = latest["Demand"]

    # Last 7 records
    recent_data = product_data.tail(7)
    rolling_demand = recent_data["Demand"].mean()

    # Current date
    prediction_date = pd.Timestamp.today()

    day = prediction_date.dayofweek
    month = prediction_date.month

    # Create input for model
    new_data = pd.DataFrame(
        [[
            product,
            price,
            previous_demand,
            rolling_demand,
            day,
            month
        ]],
        columns=[
            "Description",
            "Average_Price",
            "Previous_Demand",
            "Rolling_7_Demand",
            "DayOfWeek",
            "Month"
        ]
    )

    # Make prediction
    prediction = model.predict(new_data)

    print("\n==============================================")
    print("             PREDICTION RESULT")
    print("==============================================")

    print("Product:", product)
    print("Latest Historical Price:", round(price, 2))
    print("Previous Demand:", round(previous_demand, 2), "units")
    print("7-Record Average Demand:", round(rolling_demand, 2), "units")
    print("----------------------------------------------")
    print("PREDICTED DEMAND:", round(prediction[0], 2), "units")
    print("==============================================")