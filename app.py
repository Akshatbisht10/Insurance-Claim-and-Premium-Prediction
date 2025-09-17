import os
os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"

import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score
from xgboost import XGBRegressor, XGBClassifier

# -------------------------------
# 1. Load and preprocess data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("insuranceClaims.csv")
    df.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
    df.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
    df.replace({'region': {'southeast': 0, 'southwest': 1, 'northwest': 2, 'northeast': 3}}, inplace=True)
    df['charges'] = np.log1p(df['charges'])  # log-transform
    return df

medical_df = load_data()

X = medical_df.drop(columns=['charges', 'insuranceclaim'])
y_reg = medical_df['charges']
y_clf = medical_df['insuranceclaim']

X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.1, random_state=2
)

# -------------------------------
# 2. Train models (cached)
# -------------------------------
@st.cache_resource
def get_models(X_train, y_reg_train, y_clf_train):
    xgb_reg = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=2)
    xgb_reg.fit(X_train, y_reg_train)

    xgb_clf = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=2)
    xgb_clf.fit(X_train, y_clf_train)

    return xgb_reg, xgb_clf

xgb_reg, xgb_clf = get_models(X_train, y_reg_train, y_clf_train)

# -------------------------------
# 3. SHAP explainers (cached)
# -------------------------------
@st.cache_resource
def get_explainers(xgb_reg, xgb_clf):
    explainer_reg = shap.TreeExplainer(xgb_reg)
    explainer_clf = shap.TreeExplainer(xgb_clf)
    return explainer_reg, explainer_clf

explainer_reg, explainer_clf = get_explainers(xgb_reg, xgb_clf)

# -------------------------------
# 4. Evaluate once (cached)
# -------------------------------
@st.cache_data
def evaluate_models():
    y_reg_pred = xgb_reg.predict(X_test)
    y_clf_pred = xgb_clf.predict(X_test)

    r2 = r2_score(y_reg_test, y_reg_pred)
    mse = mean_squared_error(y_reg_test, y_reg_pred)
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    rmse = np.sqrt(mse)

    # reverse log-transform for interpretability
    y_reg_test_exp = np.expm1(y_reg_test)
    y_reg_pred_exp = np.expm1(y_reg_pred)
    mse_exp = mean_squared_error(y_reg_test_exp, y_reg_pred_exp)
    mae_exp = mean_absolute_error(y_reg_test_exp, y_reg_pred_exp)
    rmse_exp = np.sqrt(mse_exp)

    clf_accuracy = accuracy_score(y_clf_test, y_clf_pred)

    return r2, mse, mae, rmse, mse_exp, mae_exp, rmse_exp, clf_accuracy

r2, mse, mae, rmse, mse_exp, mae_exp, rmse_exp, clf_accuracy = evaluate_models()

# -------------------------------
# 5. Streamlit UI
# -------------------------------
st.title("🏥 Medical Insurance Prediction Model")

st.write("Enter the details below to predict medical insurance charges and claim status.")

# Inputs
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", options=["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", options=["yes", "no"])
region = st.selectbox("Region", options=["southeast", "southwest", "northwest", "northeast"])

# Convert inputs
sex = 0 if sex == "male" else 1
smoker = 0 if smoker == "yes" else 1
region = {"southeast": 0, "southwest": 1, "northwest": 2, "northeast": 3}[region]

input_features = np.array([age, sex, bmi, children, smoker, region]).reshape(1, -1)
input_df = pd.DataFrame(input_features, columns=X.columns)

# Predictions
reg_prediction = xgb_reg.predict(input_features)
reg_prediction_exp = np.expm1(reg_prediction)
clf_prediction = xgb_clf.predict(input_features)

# -------------------------------
# 6. Display results
# -------------------------------
st.subheader("🔮 Predictions")
st.write(f"**Predicted Medical Insurance Premium Charges:** ${reg_prediction_exp[0]:,.2f}")
st.write(f"**Will a claim be made?** {'✅ Yes' if clf_prediction[0] == 1 else '❌ No'}")

st.subheader("📊 Model Performance (Regression)")
st.write(f"R² Score: {r2:.2f}")
st.write(f"MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
st.write(f"(After reversing log) MSE: {mse_exp:.2f}, MAE: {mae_exp:.2f}, RMSE: {rmse_exp:.2f}")

st.subheader("📊 Model Performance (Classification)")
st.write(f"Accuracy: {clf_accuracy:.2%}")

# -------------------------------
# 7. SHAP Visualizations
# -------------------------------
st.subheader("🔍 Feature Impact on Premium Prediction")
shap_values_reg = explainer_reg(input_df)
fig, ax = plt.subplots()
shap.waterfall_plot(shap_values_reg[0], max_display=6, show=False)
plt.tight_layout()
st.pyplot(fig)

st.subheader("🔍 Feature Impact on Claim Prediction")
shap_values_clf = explainer_clf(input_df)
fig, ax = plt.subplots()
shap.waterfall_plot(shap_values_clf[0], max_display=6, show=False)
plt.tight_layout()
st.pyplot(fig)
