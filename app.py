import streamlit as st
import pandas as pd
import joblib

# Load model and data
model = joblib.load("retail_demand_model.pkl")
data = pd.read_csv("retail_demand_features.csv")

data["Date"] = pd.to_datetime(data["Date"])

# Page settings
st.set_page_config(
    page_title="AI Retail Demand Prediction",
    page_icon="🛒",
    layout="wide"
)

# Title
st.title("🛒 AI-Powered Retail Demand Prediction System")

st.write(
    "Predict retail product demand using historical sales data "
    "and a Random Forest machine learning model."
)

st.divider()

# Product list
products = sorted(data["Description"].dropna().unique())

# Product selection
product = st.selectbox(
    "🔎 Select a Product",
    products
)

# Selected product data
product_data = data[
    data["Description"] == product
].sort_values("Date")

# Latest record
latest = product_data.iloc[-1]

price = latest["Average_Price"]
previous_demand = latest["Demand"]

# Recent 7-record average
recent_data = product_data.tail(7)
rolling_demand = recent_data["Demand"].mean()

# Current date
prediction_date = pd.Timestamp.today()

day = prediction_date.dayofweek
month = prediction_date.month

# Product information
st.subheader("📊 Product Information")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Latest Price",
    f"{price:.2f}"
)

col2.metric(
    "Previous Demand",
    f"{previous_demand:.0f} units"
)

col3.metric(
    "7-Record Avg. Demand",
    f"{rolling_demand:.2f} units"
)

st.divider()

# Historical demand chart
st.subheader("📈 Historical Demand Trend")

chart_data = product_data[
    ["Date", "Demand"]
].tail(30).set_index("Date")

st.line_chart(chart_data)

st.divider()

# Prediction button
if st.button(
    "🔮 Predict Demand",
    use_container_width=True
):

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

    prediction = model.predict(new_data)[0]

    st.subheader("🎯 Prediction Result")

    st.success(
        f"Predicted Demand: {prediction:.2f} units"
    )

    st.info(
        f"Selected Product: {product}"
    )

st.divider()

st.caption(
    "Dataset: UCI Online Retail | "
    "Model: Random Forest Regressor"
)