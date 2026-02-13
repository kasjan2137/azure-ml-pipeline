"""
Scoring Script for Model Deployment.

THIS FILE RUNS ON THE DEPLOYED ENDPOINT (not your laptop).

When someone sends a request to your API:
1. init() runs once when the server starts (loads the model)
2. run() runs for EACH request (makes predictions)

EXAMPLE API CALL:
    POST https://your-endpoint.azureml.net/score
    Body: {
        "input_data": {
            "columns": ["age", "sex", "cp", ...],
            "data": [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]
        }
    }
    
    Response: {"predictions": [1], "probabilities": [0.87]}
"""

import os
import json
import joblib
import numpy as np
import pandas as pd


def init():
    """
    Called ONCE when the endpoint starts up.
    
    Loads the model and scaler into memory so they're ready
    for fast predictions.
    """
    global model, scaler, feature_names
    
    # AZUREML_MODEL_DIR is set by Azure ML automatically
    # It points to where the model files are stored
    model_dir = os.getenv("AZUREML_MODEL_DIR", "./model")
    
    model = joblib.load(os.path.join(model_dir, "model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    
    # Load feature names
    features_path = os.path.join(model_dir, "features.txt")
    if os.path.exists(features_path):
        with open(features_path) as f:
            feature_names = f.read().strip().split("\n")
    else:
        feature_names = None
    
    print(f"✅ Model loaded: {type(model).__name__}")


def run(raw_data):
    """
    Called for EACH prediction request.
    
    Args:
        raw_data: JSON string with input data
        
    Returns:
        JSON string with predictions
    """
    try:
        # Parse input
        data = json.loads(raw_data)
        
        # Handle different input formats
        if "input_data" in data:
            input_data = data["input_data"]
            if "columns" in input_data and "data" in input_data:
                df = pd.DataFrame(
                    data=input_data["data"],
                    columns=input_data["columns"]
                )
            else:
                df = pd.DataFrame(input_data)
        elif "data" in data:
            df = pd.DataFrame(data["data"])
        else:
            df = pd.DataFrame(data)
        
        # Scale features
        X_scaled = scaler.transform(df)
        
        # Predict
        predictions = model.predict(X_scaled).tolist()
        probabilities = model.predict_proba(X_scaled)[:, 1].tolist()
        
        result = {
            "predictions": predictions,
            "probabilities": [round(p, 4) for p in probabilities],
        }
        
        return json.dumps(result)
    
    except Exception as e:
        error_result = {"error": str(e)}
        return json.dumps(error_result)
