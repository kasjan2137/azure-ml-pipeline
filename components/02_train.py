"""
Step 2: Model Training Component.

THIS FILE RUNS INSIDE AZURE ML (not on your laptop).
It's uploaded to Azure and executed on a cloud computer.

WHAT IT DOES:
1. Loads the preprocessed training data
2. Scales features with StandardScaler
3. Trains a Random Forest classifier
4. Logs metrics to MLflow
5. Saves model and scaler as pickle files
"""

import argparse
import os
import joblib

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import mlflow


def main():
    # ---- Parse arguments ----
    parser = argparse.ArgumentParser(description="Train ML model")
    parser.add_argument(
        "--train_data", type=str, required=True, help="Path to training data"
    )
    parser.add_argument(
        "--model_output", type=str, required=True, help="Path to save model"
    )
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees")
    parser.add_argument("--max_depth", type=int, default=10, help="Max tree depth")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 2: MODEL TRAINING")
    print("=" * 60)

    # ---- Load data ----
    print(f"\n📂 Loading training data from: {args.train_data}")

    # Find CSV file in the input folder
    train_files = [f for f in os.listdir(args.train_data) if f.endswith(".csv")]
    if not train_files:
        raise FileNotFoundError(f"No CSV files found in {args.train_data}")

    train_path = os.path.join(args.train_data, train_files[0])
    train_df = pd.read_csv(train_path)

    print(f"   Training samples: {len(train_df)}")

    # ---- Prepare features and target ----
    target_col = "target"
    feature_cols = [c for c in train_df.columns if c != target_col]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    print(f"   Features: {feature_cols}")
    print(f"   Target: '{target_col}' (0 = no disease, 1 = has disease)")

    # ---- Scale features ----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # ---- Train model ----
    print(f"\n🏋️ Training RandomForest model...")
    print(f"   n_estimators: {args.n_estimators}")
    print(f"   max_depth: {args.max_depth}")

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)
    print("   ✅ Training complete!")

    # ---- Evaluate on training data ----
    y_pred = model.predict(X_train_scaled)
    accuracy = accuracy_score(y_train, y_pred)
    f1 = f1_score(y_train, y_pred)

    print(f"\n📊 Training Metrics:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1 Score: {f1:.4f}")

    # ---- Log to MLflow ----
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("n_features", len(feature_cols))
    mlflow.log_metric("train_accuracy", accuracy)
    mlflow.log_metric("train_f1", f1)

    # Feature importances
    importances = dict(zip(feature_cols, model.feature_importances_))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    print(f"\n📈 Feature Importances (top 5):")
    for name, imp in sorted_imp[:5]:
        print(f"   {name}: {imp:.4f}")
        mlflow.log_metric(f"importance_{name}", imp)

    # ---- Save model and artifacts ----
    os.makedirs(args.model_output, exist_ok=True)

    # Save model
    model_path = os.path.join(args.model_output, "model.pkl")
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved to: {model_path}")

    # Save scaler
    scaler_path = os.path.join(args.model_output, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"💾 Scaler saved to: {scaler_path}")

    # Save feature names
    features_path = os.path.join(args.model_output, "features.txt")
    with open(features_path, "w") as f:
        f.write("\n".join(feature_cols))
    print(f"💾 Features saved to: {features_path}")

    print("\n✅ Training step complete!")


if __name__ == "__main__":
    main()
