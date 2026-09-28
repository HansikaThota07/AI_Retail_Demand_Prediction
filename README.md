AI-POWERED RETAIL DEMAND PREDICTION SYSTEM



1\. PROJECT TITLE

AI-Powered Retail Demand Prediction System



2\. OBJECTIVE

The objective of this project is to predict the future demand of retail products using historical sales data and machine learning.



3\. DATASET

Dataset: UCI Online Retail Dataset



Original Records: 541,909

Cleaned Valid Transactions: 530,104



The dataset contains real retail transaction information such as:

\- Invoice Number

\- Product Code

\- Product Description

\- Quantity

\- Invoice Date

\- Unit Price

\- Customer ID

\- Country



4\. DATA PREPROCESSING

The following cleaning steps were performed:

\- Removed records with missing product descriptions.

\- Removed cancelled invoices.

\- Removed transactions with negative quantities.

\- Removed transactions with zero or negative prices.



5\. FEATURE ENGINEERING

The transaction data was converted into product-level demand data.



Features used by the model:

\- Product Description

\- Average Price

\- Previous Demand

\- Rolling 7-Record Demand

\- Day of Week

\- Month



Target:

\- Demand



6\. MACHINE LEARNING MODEL

Algorithm: Random Forest Regressor



The model was trained using a chronological 80/20 train-test split.



For practical training time, the latest 100,000 feature records were used.



7\. MODEL EVALUATION

Test Records: 20,000



Mean Absolute Error (MAE): 17.27 units

Root Mean Squared Error (RMSE): 55.47



MAE represents the average absolute difference between actual and predicted demand.



8\. PROJECT OUTPUT

The system accepts a real product name from the dataset and predicts its demand.



Example:

Product: JUMBO BAG RED RETROSPOT

Predicted Demand: 182.92 units



9\. VISUALIZATION

The project includes an Actual vs Predicted Demand graph.



File:

actual\_vs\_predicted.png



10\. WEB APPLICATION

A Streamlit web application was created.



The application provides:

\- Product selection

\- Latest product price

\- Previous demand

\- Recent demand average

\- Historical demand trend

\- Predicted demand



11\. IMPORTANT FILES



main.py

Trains the Random Forest model.



predict.py

Predicts demand for a selected product.



evaluate.py

Evaluates the trained model using MAE and RMSE.



plot\_results.py

Creates the Actual vs Predicted graph.



app.py

Runs the Streamlit web application.



retail\_demand\_model.pkl

Saved trained machine learning model.



retail\_demand\_features.csv

Feature-engineered dataset.



cleaned\_retail\_sales.csv

Cleaned retail transaction dataset.



12\. HOW TO RUN THE PROJECT



Step 1:

Activate the virtual environment.



Step 2:

Run the prediction system:



python predict.py



Step 3:

Run the evaluation:



python evaluate.py



Step 4:

Run the Streamlit application:



streamlit run app.py



13\. TECHNOLOGIES USED



Python

Pandas

NumPy

Scikit-learn

Joblib

Matplotlib

Streamlit



14\. CONCLUSION

This project demonstrates how historical retail transaction data can be cleaned, transformed into useful demand features, and used to train a machine learning model for demand prediction.



The Streamlit application provides an interactive interface for selecting products and viewing demand predictions.

