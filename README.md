

---

# 🏥 Medical Insurance Prediction with Streamlit, XGBoost, and SHAP

This project is a **Streamlit web app** that predicts **medical insurance charges** (regression) and **insurance claim status** (classification) using **XGBoost models**. It also uses **SHAP explainability** to show how different features (age, BMI, smoker status, etc.) influence the predictions.

---

## 🚀 Features
- Predicts **medical insurance premium charges**.
- Predicts whether an **insurance claim will be made**.
- Displays **model evaluation metrics**:
  - R² Score, MSE, MAE, RMSE (for regression)
  - Accuracy (for classification)
- Provides **SHAP visualizations** (waterfall plots) to explain feature importance for each prediction.
- Interactive **Streamlit web app** interface.

---

## 📂 Project Structure
```

Insurance\_Project/
├── claims.py               # Main Streamlit app
├── insuranceClaims.csv     # Dataset
└── README.md               # Project documentation

````

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/insurance-prediction.git
cd insurance-prediction
````

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv myenv
myenv\Scripts\activate     # On Windows
source myenv/bin/activate  # On Linux/Mac
```

### 3. Install dependencies

```bash
pip install streamlit pandas numpy shap matplotlib scikit-learn xgboost
```

---

## ▶️ Running the App

Make sure you are inside the virtual environment and run:

```bash
python -m streamlit run claims.py
```

Then open the URL shown in the terminal (usually [http://localhost:8501](http://localhost:8501)).

---

## 📊 Dataset

The app uses **insuranceClaims.csv**, which contains:

* **age**: Age of the individual
* **sex**: Male/Female
* **bmi**: Body Mass Index
* **children**: Number of children
* **smoker**: Smoking status
* **region**: Residential region
* **charges**: Insurance charges (log-transformed for regression)
* **insuranceclaim**: Binary classification target (1 = claim made, 0 = no claim)

---

## 📈 Model Details

* **Regression**: `XGBRegressor` → Predicts medical insurance charges
* **Classification**: `XGBClassifier` → Predicts whether a claim is made
* Both trained with:

  * `n_estimators=500`
  * `learning_rate=0.05`
  * `max_depth=4`
* **SHAP** is used to explain predictions.

---

## 🖼️ Screenshots

### App Interface
<img width="1909" height="843" alt="Screenshot 2025-09-09 204828" src="https://github.com/user-attachments/assets/0a6e23c8-11f0-45cc-8e5d-a3c782009eaa" />
<img width="1904" height="912" alt="Screenshot 2025-09-09 204856" src="https://github.com/user-attachments/assets/0bcd141c-146a-4353-98c9-2943164ee2d7" />
<img width="1901" height="480" alt="Screenshot 2025-09-09 204911" src="https://github.com/user-attachments/assets/0db44534-4eca-44d9-9011-c92799cc61dc" />

---

## 📌 Future Improvements

* Deploy the app on **Streamlit Cloud / Heroku** for public access.
* Add more advanced models (e.g., neural networks).
* Perform hyperparameter tuning.
* Enhance dataset with additional health-related features.


