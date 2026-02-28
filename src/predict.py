import pandas as pd
import joblib
import json


def load_model(path):
    return joblib.load(path)


def load_data(path):
    return pd.read_csv(path)


def validate_features(X, metadata_path):
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    expected_features = metadata["features"]
    
    if list(X.columns) != expected_features:
        raise ValueError("Input features do not match training features.")
    
    return True


def generate_predictions(model, X):
    return model.predict(X)


def save_predictions(predictions, path):
    submission = pd.DataFrame({
        "target_pred": predictions
    })
    submission.to_csv(path, index=False)


if __name__ == "__main__":
    
    # Paths
    model_path = "../models/xgb_model.pkl"
    metadata_path = "../models/model_metadata.json"
    blind_path = "../data/blind_test_data.csv"
    output_path = "../blind_predictions.csv"
    
    # Load
    model = load_model(model_path)
    X_blind = load_data(blind_path)
    
    # Validate
    validate_features(X_blind, metadata_path)
    
    # Predict
    predictions = generate_predictions(model, X_blind)
    
    # Save
    save_predictions(predictions, output_path)
    
    print("Predictions generated successfully.")