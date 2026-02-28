# ML Implementing Challenge – Regression Model

## Overview

This project addresses a multivariate regression problem using 20 numerical features to predict a continuous target variable.

The solution includes:
- Model training pipeline
- Hyperparameter tuning
- Batch prediction script
- Model metadata for reproducibility
- Basic validation for production readiness

The final selected model is a tuned XGBoost regressor.

---

## Model Selection Process

During the exploratory phase:

1. Linear Regression was used as a baseline.
2. Random Forest was evaluated to capture nonlinear patterns.
3. XGBoost was tested and showed significantly better performance.
4. Hyperparameter optimization was performed using RandomizedSearchCV with 5-fold cross-validation.

Final cross-validated RMSE: **~1.95**

---

## Project Structure

```
ml_challenge/
│
├── data/
│   ├── training_data.csv
│   └── blind_test_data.csv
│
├── models/
│   ├── xgb_model.pkl
│   └── model_metadata.json
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── notebook_exploration.ipynb
└── README.md
```

---

## How to Run

### 1. Train the model

From the `src` folder:
python train.py


This will:
- Train the XGBoost model
- Save it to `models/xgb_model.pkl`
- Generate metadata in `models/model_metadata.json`

---

### 2. Generate Predictions

python predict.py


This will:
- Load the trained model
- Validate input features
- Generate predictions
- Save `blind_predictions.csv`

---

## Production Considerations

To maintain long-term performance, the following strategies are recommended:

- Monitor feature distribution drift against training statistics.
- Track prediction error once ground truth becomes available.
- Define a retraining policy based on data volume or performance degradation.
- Version models and store metadata for traceability.

This structure allows easy evolution toward API deployment or automated retraining pipelines.