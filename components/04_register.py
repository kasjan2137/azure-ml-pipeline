"""
Step 4: Model Registration Component.

THIS FILE RUNS INSIDE AZURE ML (not on your laptop).
It's uploaded to Azure and executed on a cloud computer.

WHAT IT DOES:
1. Loads the evaluation results
2. Verifies model artifacts exist
3. Logs completion (model registration happens in deploy step)
"""

import argparse
import os
import json


def main():
    # ---- Parse arguments ----
    parser = argparse.ArgumentParser(description="Register ML model")
    parser.add_argument(
        "--model_input", type=str, required=True, help="Path to model folder"
    )
    parser.add_argument(
        "--evaluation_input", type=str, required=True, help="Path to evaluation results"
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name for registered model"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 4: MODEL REGISTRATION")
    print("=" * 60)

    # ---- Load evaluation metrics ----
    eval_path = os.path.join(args.evaluation_input, "evaluation_results.json")
    with open(eval_path, "r") as f:
        eval_results = json.load(f)

    metrics = eval_results["metrics"]

    print(f"\n📊 Model metrics:")
    for name, value in metrics.items():
        print(f"   {name}: {value:.4f}")

    # ---- Verify model artifacts ----
    print(f"\n📦 Model artifacts in: {args.model_input}")

    model_files = os.listdir(args.model_input)
    print(f"   Files: {model_files}")

    # Verify required files exist
    required_files = ["model.pkl", "scaler.pkl"]
    all_found = True
    for f in required_files:
        fpath = os.path.join(args.model_input, f)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            print(f"   ✅ {f} ({size:,} bytes)")
        else:
            print(f"   ❌ {f} NOT FOUND")
            all_found = False

    if all_found:
        print(f"\n✅ Model '{args.model_name}' verified and ready!")
        print(
            "   Pipeline complete. Run 'python -m src.run_pipeline deploy' to deploy."
        )
    else:
        print(f"\n❌ Model verification failed!")
        raise FileNotFoundError("Required model files not found")


if __name__ == "__main__":
    main()
