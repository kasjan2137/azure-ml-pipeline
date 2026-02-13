"""
Step 1: Data Preparation Component.

THIS FILE RUNS INSIDE AZURE ML (not on your laptop).
It's uploaded to Azure and executed on a cloud computer.

WHAT IT DOES:
1. Reads the raw CSV data
2. Cleans it (handle missing values, remove duplicates)
3. Splits into training set (80%) and test set (20%)
4. Saves both as separate files

WHY SEPARATE FILES?
- Training set: Used to TEACH the model
- Test set: Used to CHECK how good the model is
- You NEVER train on test data (that's cheating!)

THE DATASET:
We use a heart disease prediction dataset.
- Input: Patient features (age, cholesterol, blood pressure, etc.)
- Output: Whether they have heart disease (0 = no, 1 = yes)
- This is a CLASSIFICATION problem (predicting categories)
"""

import argparse
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    # ---- Parse command-line arguments ----
    # Azure ML passes paths as arguments so the component
    # knows where to read input and write output
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="Path to raw data CSV")
    parser.add_argument("--train_data", type=str, help="Path to save training data")
    parser.add_argument("--test_data", type=str, help="Path to save test data")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction for test set")
    args = parser.parse_args()
    
    print("=" * 60)
    print("STEP 1: DATA PREPARATION")
    print("=" * 60)
    
    # ---- Load data ----
    print(f"\n📂 Loading data from: {args.input_data}")
    df = pd.read_csv(args.input_data)
    
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Shape: {df.shape}")
    
    # ---- Clean data ----
    print("\n🧹 Cleaning data...")
    
    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed > 0:
        print(f"   Removed {removed} duplicate rows")
    
    # Handle missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"   Found {missing} missing values")
        # For numeric columns: fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # For categorical columns: fill with mode (most common)
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        print("   ✅ Missing values handled")
    else:
        print("   ✅ No missing values")
    
    # ---- Print data summary ----
    print(f"\n📊 Data Summary:")
    print(f"   Target column: 'target'")
    if "target" in df.columns:
        print(f"   Class distribution:")
        print(f"     0 (No disease):  {(df['target'] == 0).sum()}")
        print(f"     1 (Has disease): {(df['target'] == 1).sum()}")
    
    # ---- Split into train and test ----
    print(f"\n✂️ Splitting data (test_size={args.test_size})...")
    
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=42,        # For reproducibility
        stratify=df["target"],  # Keep same class ratio in both sets
    )
    
    print(f"   Training set: {len(train_df)} rows")
    print(f"   Test set:     {len(test_df)} rows")
    
    # ---- Save outputs ----
    # Create output directories (Azure ML expects this)
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)
    
    train_path = os.path.join(args.train_data, "train.csv")
    test_path = os.path.join(args.test_data, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n💾 Saved:")
    print(f"   Training data: {train_path}")
    print(f"   Test data:     {test_path}")
    print("\n✅ Data preparation complete!")


if __name__ == "__main__":
    main()
