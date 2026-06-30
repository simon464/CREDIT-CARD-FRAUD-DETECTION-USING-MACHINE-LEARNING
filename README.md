# Credit Card Fraud Detection Using Machine Learning

## Project Overview

This project is a machine-learning-based credit card fraud detection system designed to identify whether a transaction is **legitimate** or **fraudulent**. The system uses trained machine learning models to analyse transaction details such as amount, category, time, customer information, and location-based features, then returns a fraud-risk prediction.

The project also includes a **Streamlit web application** that allows users to make manual predictions, upload CSV files for batch prediction, and view explainable fraud-risk outputs using **SHAP explainability**.

## What the Project Does

The application helps detect suspicious credit card transactions by:

- Loading already-trained `.pkl` machine learning models from the `models/` folder.
- Accepting transaction details from a user through a Streamlit interface.
- Performing feature engineering, including time-based features and geographic distance.
- Predicting whether a transaction is fraudulent or legitimate.
- Supporting multiple trained models such as Logistic Regression, XGBoost, LightGBM, and optionally Random Forest.
- Applying saved decision thresholds to improve fraud detection performance.
- Providing SHAP-based explanations to show why a transaction was considered risky.
- Supporting batch prediction using CSV files.

## Main Features

### 1. Manual Transaction Prediction

Users can enter transaction details manually, including:

- Transaction date and time
- Transaction category
- Transaction amount
- Gender
- City population
- Cardholder date of birth
- Cardholder latitude and longitude
- Merchant latitude and longitude

The app uses these inputs to generate the final model features and predict fraud risk.

### 2. Batch CSV Prediction

Users can upload a CSV file containing multiple transactions. The app processes the file and produces predictions for each transaction.

### 3. Model Loading

The app loads trained model files from the `models/` folder. These models were trained beforehand, so the Streamlit app does **not retrain models**. It only uses the saved models for prediction.

### 4. SHAP Explainability

SHAP is used to explain the model prediction by showing which features increased or reduced the fraud risk. This improves transparency and helps users understand why a transaction was flagged.

### 5. Threshold-Based Prediction

Instead of relying only on the default probability threshold of `0.5`, the project uses optimized thresholds saved from model evaluation. This improves performance on the imbalanced fraud detection problem.

## Dataset

The project used the **Credit Card Transactions Fraud Detection Dataset** from Kaggle. The dataset contains synthetic credit card transaction records with labels showing whether each transaction is fraudulent or legitimate.
(https://www.kaggle.com/datasets/kartik2112/fraud-detection) the link to the dataset used in this project


The dataset includes two main files:

- `fraudTrain.csv`
- `fraudTest.csv`

The original dataset contained raw features such as transaction date, amount, merchant category, cardholder location, merchant location, date of birth, and fraud label.

## Feature Engineering

The final model used ten important features:

| Feature | Description |
|---|---|
| `category` | Merchant transaction category |
| `amt` | Transaction amount |
| `gender` | Cardholder gender |
| `city_pop` | Population of the cardholder's city |
| `hour` | Hour extracted from transaction time |
| `day` | Day of the month |
| `month` | Transaction month |
| `dayofweek` | Day of the week |
| `age` | Cardholder age |
| `distance_km` | Distance between cardholder and merchant |

The `distance_km` feature is engineered from:

- Cardholder latitude
- Cardholder longitude
- Merchant latitude
- Merchant longitude

## Machine Learning Models

The project compares and uses the following models:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

The models are saved as `.pkl` files and placed inside the `models/` folder.

## Model Results

The final model comparison showed that Random Forest and XGBoost performed strongly.

| Model | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|
| Random Forest | 0.9033 | 0.7883 | 0.8419 | 0.9784 |
| XGBoost | 0.9074 | 0.7403 | 0.8154 | 0.9970 |
| Logistic Regression | 0.2919 | 0.4424 | 0.3517 | 0.8498 |
| LightGBM | 0.1179 | 0.8261 | 0.2064 | 0.8997 |

Random Forest achieved the best balance between precision, recall, and F1-score, while XGBoost achieved the highest ROC-AUC.

## Project Folder Structure

```text
CREDIT-CARD-FRAUD-DETECTION-USING-MACHINE-LEARNING/
│
├── app.py
├── requirements.txt
├── README.md
└── models/
    ├── lr_model.pkl
    ├── xgb_model.pkl
    ├── lgbm_model.pkl
    ├── lr_threshold.pkl
    ├── xgb_threshold.pkl
    ├── lgbm_threshold.pkl
    └── thresholds.json
```

Optional files:

```text
rf_model.pkl
rf_threshold.pkl
consensus_package.pkl
```

These optional files may be excluded if they are too large for manual GitHub upload.

## Requirements

The main Python packages used are:

```text
streamlit
pandas
numpy
scikit-learn
scipy
imbalanced-learn
joblib
xgboost
lightgbm
shap
matplotlib
```

Install them using:

```bash
pip install -r requirements.txt
```

## How to Run the App Locally

1. Clone or download this repository.
2. Make sure the trained model files are inside the `models/` folder.
3. Open CMD or terminal inside the project folder.
4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Run the Streamlit app:

```bash
python -m streamlit run app.py
```

## How the Prediction Works

The prediction process follows this flow:

```text
User inputs transaction details
        ↓
Feature engineering
        ↓
Saved machine learning models load from models/
        ↓
Fraud probability is calculated
        ↓
Threshold is applied
        ↓
Transaction is classified as legitimate or fraudulent
        ↓
SHAP explains the decision
```

## Important Notes

- The app does not train models again.
- The trained `.pkl` files must be placed inside the `models/` folder.
- Very large model files may not upload through the GitHub browser upload interface.
- If a model file is missing, the app may still run using the available models.
- This project is intended for academic demonstration and should be tested further before real financial deployment.

## Limitations

- The dataset used is synthetic and may not fully represent real-world fraud patterns.
- The system is not connected to a live banking or payment network.
- Model performance may change when tested on real Kenyan or institutional transaction data.
- Future improvements should include real-time streaming, cloud deployment, probability calibration, and testing on real transaction data.

## Future Improvements

Possible improvements include:

- Deploying the system on cloud infrastructure.
- Adding real-time transaction monitoring using Kafka or cloud streaming tools.
- Testing the model on real Kenyan financial transaction data.
- Improving model calibration.
- Adding stronger authentication and security controls.
- Exploring deep learning models for sequential transaction behaviour.

## Author

**Mugo Simon Ng'ang'a**  
BSc. Data Science and Analytics  
Jomo Kenyatta University of Agriculture and Technology  
2026

## License

This project is for academic and research purposes.
