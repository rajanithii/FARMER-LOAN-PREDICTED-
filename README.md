# 🌾 Farmer Loan Default Predictor

> An AI-powered web application that assesses agricultural loan default risk in real time — built for financial institutions, rural banks, and credit analysts.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat)

---

## 📸 Application Preview

![Farmer Loan Default Predictor - App Screenshot](docs/screenshot.png)

> The application features a dark professional UI with a real-time prediction engine, interactive risk gauge, feature impact analysis, and an AI-powered recommendation system.

---

## 📌 Overview

Access to agricultural credit is essential for Indian farmers, yet loan default remains a persistent challenge for lenders operating in rural markets. This project applies machine learning to the problem — using a **Random Forest Classifier** trained on 14 financial, agricultural, and weather features to predict whether a given applicant is likely to default on a loan.

The result is a production-ready Streamlit dashboard that gives credit officers an interpretable, data-driven risk score within seconds — along with specific recommendations for each applicant.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔍 **Single Prediction** | Real-time risk assessment with confidence gauge and colour-coded verdict |
| 📊 **Feature Impact Analysis** | Bar chart showing which factors contributed most to the prediction |
| 💡 **Recommendation Engine** | Applicant-specific, actionable advice to reduce default risk |
| 📁 **Batch Prediction** | Upload a CSV of multiple applicants and download scored results |
| 📋 **Prediction History** | Session log of all predictions with summary statistics |
| 🔄 **What-If Simulator** | Adjust any input in real time and see how the risk score changes |
| 🎨 **Custom Dark UI** | Professional dark theme with gradient header, metric cards, and risk badges |

---

## 🗂️ Project Structure

```
farmer-loan-predictor/
├── data/
│   └── loan_data.csv              # Farmer loan dataset with weather features
├── notebooks/
│   ├── 01_EDA.ipynb               # Exploratory Data Analysis
│   └── 02_Model_Training.ipynb    # Preprocessing, model training & evaluation
├── docs/
│   └── screenshot.png             # App screenshot for README
├── app.py                         # Streamlit application (v2.0)
├── train_model.py                 # Model training script
├── loan_model.pkl                 # Saved Random Forest model (joblib)
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md
```

---

## ⚙️ Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/farmer-loan-predictor.git
cd farmer-loan-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the model** *(skip if `loan_model.pkl` already exists)*
```bash
python train_model.py
```

**4. Launch the app**
```bash
streamlit run app.py
```

The application will open automatically at `http://localhost:8501`.

> **Note:** If you are using Python 3.12+, use `pip install streamlit pandas numpy scikit-learn matplotlib seaborn plotly joblib jupyter` instead of the requirements file to avoid compilation errors with pinned versions.

---

## 🤖 Model Details

| Attribute | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Number of Trees | 200 |
| Max Depth | 8 |
| Class Weighting | Balanced |
| Train / Test Split | 80% / 20% |
| Preprocessing | `LabelEncoder` for categorical features, `fillna` median for missing values |
| Serialization | `joblib.dump()` → `loan_model.pkl` |
| Target Variable | `default` (0 = No Default, 1 = Default) |

---

## 📐 Input Features

| Feature | Type | Description |
|---|---|---|
| `age` | Numerical | Farmer's age in years |
| `land_area_acres` | Numerical | Total agricultural land owned |
| `annual_income` | Numerical | Yearly income in ₹ |
| `crop_type` | Categorical | Primary crop cultivated (9 types) |
| `loan_amount` | Numerical | Requested loan amount in ₹ |
| `loan_tenure_months` | Numerical | Repayment period in months |
| `previous_loans` | Numerical | Count of prior loans taken |
| `repayment_history` | Categorical | Good / Poor |
| `soil_quality` | Categorical | High / Medium / Low |
| `irrigation_type` | Categorical | Canal / Borewell / Drip / Rainfed |
| `credit_score` | Numerical | CIBIL-style score (300–850) |
| `state` | Categorical | Indian state of the farmer (15 states) |
| `rainfall_mm` | Numerical | Annual rainfall in millimetres |
| `avg_temp_celsius` | Numerical | Average annual temperature in °C |

---

## 📊 Dataset

The project includes a synthetic dataset of 40 labelled records for development and demonstration. For production-grade training, replace `data/loan_data.csv` with one of the following publicly available datasets:

| Dataset | Source | Link | Cost |
|---|---|---|---|
| Loan Prediction Dataset | Kaggle | [kaggle.com/search?q=loan+prediction](https://www.kaggle.com/search?q=loan+prediction) | FREE |
| Indian Agriculture Data | data.gov.in | [data.gov.in](https://data.gov.in) | FREE |
| Farm Income Dataset | Kaggle | [kaggle.com/search?q=farm+income](https://www.kaggle.com/search?q=farm+income) | FREE |

Ensure the replacement dataset includes columns matching the feature schema above, or adapt the encoding section in `train_model.py` accordingly.

---

## 🧠 How the Prediction Works

The application follows an end-to-end ML pipeline across eight steps aligned with the IBM Python for Data Science course structure:

1. **Setup & Data Load** — Project folder initialised, CSV loaded with `pd.read_csv`, null check with `df.isnull().sum()`
2. **EDA** — Distribution plots, default rate by category, and correlation heatmap (`01_EDA.ipynb`)
3. **Data Preprocessing** — Missing values handled with `fillna(median)` and `dropna`; categorical columns encoded with `sklearn LabelEncoder`
4. **Model Training** — `RandomForestClassifier` trained with 80/20 split, evaluated with accuracy, precision, recall, and F1
5. **Feature Importance** — Top 14 features ranked by `model.feature_importances_` and visualised as a horizontal bar chart
6. **Save Model** — Final model persisted with `joblib.dump(model, 'loan_model.pkl')`
7. **Streamlit App** — Interactive UI with sidebar inputs, real-time prediction, explainability chart, and recommendation engine
8. **GitHub Push** — Repository published with README, screenshots, and dataset reference links

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit, custom CSS |
| Machine Learning | scikit-learn (Random Forest) |
| Data Processing | pandas, NumPy |
| Visualisation | Plotly, Matplotlib, Seaborn |
| Model Serialisation | joblib |
| Notebook Environment | Jupyter |

---

## 🚀 Using the App

**Single Prediction** — Fill in the applicant details in the left sidebar (personal, agricultural, loan, and weather inputs), then click **Predict Default Risk**. The app displays a risk verdict, confidence gauge, feature impact chart, and personalised recommendations.

**Batch Prediction** — Navigate to the Batch Prediction tab, download the CSV template, populate it with your applicants, upload it, and click Run Batch Prediction. A scored results file is available for download.

**What-If Simulator** — Navigate to the What-If Simulator tab and adjust any parameter to see how changes in credit score, income, loan amount, or agricultural factors affect the risk score in real time. A delta indicator shows the change relative to the original sidebar inputs.

---

## 👨‍💻 Author

**Rajanithi N**  
B.Tech — AI & Data Science  
Dhanalakshmi Srinivasan University (DSU-SET), Chennai  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/your-username)

---

## 📄 License

This project is licensed under the **MIT License**. See the `LICENSE` file for full details.

---

> *Built as part of the IBM Python for Data Science, AI & Development course — KRONE'26 Hackathon Portfolio, DSU-SET.*
