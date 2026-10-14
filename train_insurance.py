"""
Medical Insurance Cost Prediction - Training & Evaluation Script
Course: Machine Learning / Data Science
Author: Sakonlaphat (6630300866)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def main():
    print("=" * 65)
    print("  🏥 Medical Insurance Cost Prediction - Training Pipeline")
    print("=" * 65)

    data_path = os.path.join("data", "insurance.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    # 1. Load Dataset
    df = pd.read_csv(data_path)
    print(f"\n[1] Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"    Features: {df.columns.tolist()[:-1]}")
    print(f"    Target: {df.columns.tolist()[-1]} (Annual charges in USD)")

    # 2. Train-Test Split (80:20)
    X = df.drop(columns=["charges"])
    y = df["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"\n[2] Train/Test Split (80:20):")
    print(f"    - Training Set:   {len(X_train)} samples")
    print(f"    - Test Set:       {len(X_test)} samples (Hidden for evaluation)")

    # 3. Preprocessing Pipeline
    num_features = ["age", "bmi", "children"]
    cat_features = ["sex", "smoker", "region"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_features)
        ]
    )

    # 4. Model Candidates
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
        )
    }

    # 5. Training and Evaluation
    print(f"\n[3] Model Training and Evaluation on Hidden Test Set:")
    print("-" * 65)
    print(f"{'Model':<22} | {'R2 Score':<10} | {'RMSE (USD)':<14} | {'MAE (USD)':<12}")
    print("-" * 65)

    best_name = None
    best_pipe = None
    best_r2 = -float("inf")
    results = {}

    for name, reg in models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", reg)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        results[name] = {"R2": round(r2, 4), "RMSE": round(rmse, 2), "MAE": round(mae, 2)}
        print(f"{name:<22} | {r2:.4f}     | ${rmse:,.2f}      | ${mae:,.2f}")

        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_pipe = pipe

    print("-" * 65)
    print(f"🏆 Best Model: {best_name} (R2 = {best_r2:.4f})")

    # 6. Save Best Model
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "insurance_model.joblib")
    joblib.dump(best_pipe, model_path)
    print(f"\n[4] Saved best pipeline model to: {model_path}")

    # 7. Save Metrics Summary
    metrics_path = os.path.join("models", "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_model": best_name,
            "best_r2": best_r2,
            "all_results": results
        }, f, indent=2)
    print(f"[5] Saved evaluation metrics to: {metrics_path}")
    print("\n✅ Training and evaluation completed successfully!\n")

if __name__ == "__main__":
    main()
