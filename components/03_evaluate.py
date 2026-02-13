"""
Step 3: Model Evaluation Component.

THIS FILE RUNS INSIDE AZURE ML (not on your laptop).
It's uploaded to Azure and executed on a cloud computer.

WHAT IT DOES:
1. Loads the trained model and scaler
2. Loads the test data
3. Makes predictions
4. Calculates metrics (accuracy, F1, precision, recall, AUC-ROC)
5. Saves evaluation results to JSON
"""

import argparse
import os
import json
import joblib

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import mlflow


def main():
    # ---- Parse arguments ----
    parser = argparse.ArgumentParser(description="Evaluate ML model")
    parser.add_argument(
        "--model_input", type=str, required=True, help="Path to model folder"
    )
    parser.add_argument(
        "--test_data", type=str, required=True, help="Path to test data"
    )
    parser.add_argument(
        "--evaluation_output", type=str, required=True, help="Path to save results"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 3: MODEL EVALUATION")
    print("=" * 60)

    # ---- Load model and scaler ----
    print(f"\n📂 Loading model from: {args.model_input}")

    model_path = os.path.join(args.model_input, "model.pkl")
    scaler_path = os.path.join(args.model_input, "scaler.pkl")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    print(f"   Model type: {type(model).__name__}")

    # ---- Load test data ----
    print(f"\n📂 Loading test data from: {args.test_data}")

    test_files = [f for f in os.listdir(args.test_data) if f.endswith(".csv")]
    if not test_files:
        raise FileNotFoundError(f"No CSV files found in {args.test_data}")

    test_path = os.path.join(args.test_data, test_files[0])
    test_df = pd.read_csv(test_path)

    print(f"   Test samples: {len(test_df)}")

    # ---- Prepare features and target ----
    target_col = "target"
    feature_cols = [c for c in test_df.columns if c != target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # ---- Scale and predict ----
    print(f"\n🔮 Making predictions...")
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # ---- Calculate metrics ----
    metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
        "test_auc_roc": roc_auc_score(y_test, y_proba),
    }

    print(f"\n📊 Test Metrics:")
    for name, value in metrics.items():
        print(f"   {name}: {value:.4f}")
        mlflow.log_metric(name, value)

    # ---- Confusion matrix ----
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n📋 Confusion Matrix:")
    print(f"   True Negatives:  {tn}  |  False Positives: {fp}")
    print(f"   False Negatives: {fn}  |  True Positives:  {tp}")

    # ---- Classification report ----
    report = classification_report(y_test, y_pred)
    print(f"\n📋 Classification Report:")
    print(report)

    # ---- Save results ----
    os.makedirs(args.evaluation_output, exist_ok=True)

    results = {
        "metrics": metrics,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "test_samples": len(test_df),
    }

    results_path = os.path.join(args.evaluation_output, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {results_path}")
    print("\n✅ Evaluation step complete!")


if __name__ == "__main__":
    main()
