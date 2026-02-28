import pandas as pd
import joblib
from xgboost import XGBRegressor
import json
from datetime import datetime


def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def train_model(X, y):
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model


def save_model(model, path):
    joblib.dump(model, path)


if __name__ == "__main__":
    
    # Paths
    train_path = "../data/training_data.csv"
    model_path = "../models/xgb_model.pkl"
    metadata_path = "../models/model_metadata.json"
    
    # Load data
    X, y = load_data(train_path)
    
    # Train model
    model = train_model(X, y)
    
    # Save model
    save_model(model, model_path)
    
    # Build metadata AFTER training
    metadata = {
        "model_type": "XGBoost",
        "training_rows": len(X),
        "features": list(X.columns),
        "best_params": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.9
        },
        "cv_rmse": 1.9496858494216451,
        "trained_at": datetime.now().isoformat(),
        "feature_stats": X.describe().to_dict()
    }
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    
    print("Model trained and saved successfully.")