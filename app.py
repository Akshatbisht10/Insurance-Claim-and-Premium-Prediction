import os
os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"

import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score


# Load and prepare the data
medical_df = pd.read_csv('insuranceClaims.csv')
medical_df.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
medical_df.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
medical_df.replace({'region': {'southeast': 0, 'southwest': 1, 'northwest': 2, 'northeast': 3}}, inplace=True)

# Log-transform charges to reduce skewness
medical_df['charges'] = np.log1p(medical_df['charges'])

# Splitting into features and targets for regression and classification
X = medical_df.drop(columns=['charges', 'insuranceclaim'])
y_reg = medical_df['charges']
y_clf = medical_df['insuranceclaim']

# Split training and testing data
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(X, y_reg, y_clf, test_size=0.1, random_state=2)

# Train regression model
xgb_reg = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=2)
xgb_reg.fit(X_train, y_reg_train)
y_reg_pred = xgb_reg.predict(X_test)

# Train classification model
xgb_clf = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=2)
xgb_clf.fit(X_train, y_clf_train)
y_clf_pred = xgb_clf.predict(X_test)

# Evaluate model
r2 = r2_score(y_reg_test, y_reg_pred)
mse = mean_squared_error(y_reg_test, y_reg_pred)
mae = mean_absolute_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mse)

# Reverse log transformation for better interpretability of regression
y_reg_test_exp = np.expm1(y_reg_test)
y_reg_pred_exp = np.expm1(y_reg_pred)
mse_exp = mean_squared_error(y_reg_test_exp, y_reg_pred_exp)
mae_exp = mean_absolute_error(y_reg_test_exp, y_reg_pred_exp)
rmse_exp = np.sqrt(mse_exp)

# Evaluate classification model
clf_accuracy = accuracy_score(y_clf_test, y_clf_pred)

# Initialize SHAP explainers
# explainer_reg = shap.Explainer(xgb_reg, X_train)
# explainer_clf = shap.Explainer(xgb_clf, X_train)
explainer_reg = shap.TreeExplainer(xgb_reg)
explainer_clf = shap.TreeExplainer(xgb_clf)

# Web app interface
st.title("Medical Insurance Prediction Model")
st.write("Enter the details below to predict medical insurance charges and claim status.")

# User inputs
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", options=["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", options=["yes", "no"])
region = st.selectbox("Region", options=["southeast", "southwest", "northwest", "northeast"])

# Convert categorical inputs to numerical values
sex = 0 if sex == "male" else 1
smoker = 0 if smoker == "yes" else 1
region = {"southeast": 0, "southwest": 1, "northwest": 2, "northeast": 3}[region]

# Create a prediction
input_features = np.array([age, sex, bmi, children, smoker, region]).reshape(1, -1)

# Predict charges
reg_prediction = xgb_reg.predict(input_features)
reg_prediction_exp = np.expm1(reg_prediction)

# Predict claim status
clf_prediction = xgb_clf.predict(input_features)

st.write("### Predicted Medical Insurance Premium Charges:")
st.write(f"${reg_prediction_exp[0]:,.2f}")

st.write("### Will a claim be made?")
st.write("Yes" if clf_prediction[0] == 1 else "No")


# Display regression model accuracy
st.write("### Model Accuracy (Normalized):")
st.write(f"R² Score: {r2:.2f}")
st.write(f"Mean Squared Error (MSE): {mse:.4f}")
st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Display classification model accuracy
st.write("### Classification Model Accuracy:")
st.write(f"Accuracy: {clf_accuracy:.2%}")

# SHAP explanation for regression
st.write("### Feature Impact on Premium Prediction:")
input_df = pd.DataFrame(input_features, columns=X.columns)

shap_values_reg = explainer_reg(input_df)
fig, ax = plt.subplots()
shap.waterfall_plot(shap_values_reg[0], max_display=6, show=False)
plt.tight_layout()
st.pyplot(fig)

# SHAP explanation for classification
st.write("### Feature Impact on Claim Prediction:")
shap_values_clf = explainer_clf(input_df)
fig, ax = plt.subplots()
shap.waterfall_plot(shap_values_clf[0], max_display=10, show=False)
plt.tight_layout()
st.pyplot(fig)
