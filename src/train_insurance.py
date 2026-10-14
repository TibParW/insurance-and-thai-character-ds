"""
Medical Insurance Cost Prediction - Training & Evaluation Script (THB Currency)
Course: Machine Learning / Data Science
Author: Sakonlaphat (6630300866) and Group Members
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

EXCHANGE_RATE = 35.0  # อัตราแลกเปลี่ยน 35 บาทต่อ 1 ดอลลาร์สหรัฐ

def main():
    print("=" * 65)
    print("  🏥 Medical Insurance Cost Prediction - Training Pipeline (THB)")
    print("=" * 65)

    data_path = os.path.join("data", "insurance.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    # 1. Load Dataset & Convert Currency to THB (บาท)
    df = pd.read_csv(data_path)
    df["charges_thb"] = (df["charges"] * EXCHANGE_RATE).round(2)
    print(f"\n[1] Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"    Features: {df.columns.tolist()[:6]}")
    print(f"    Target: charges_thb (ค่าเบี้ยประกันสุขภาพรายปี มีหน่วยเป็น บาท THB)")
    print(f"    อัตราแลกเปลี่ยนอ้างอิง: 1 USD = {EXCHANGE_RATE} บาท")

    # 2. Train-Test Split (80:20)
    X = df.drop(columns=["charges", "charges_thb"])
    y = df["charges_thb"]

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
    print("-" * 75)
    print(f"{'Model':<22} | {'R2 Score':<10} | {'RMSE (บาท)':<18} | {'MAE (บาท)':<16}")
    print("-" * 75)

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

        results[name] = {"R2": round(r2, 4), "RMSE_THB": round(rmse, 2), "MAE_THB": round(mae, 2)}
        print(f"{name:<22} | {r2:.4f}     | {rmse:>14,.2f} บาท | {mae:>12,.2f} บาท")

        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_pipe = pipe

    print("-" * 75)
    print(f"🏆 Best Model: {best_name} (R2 = {best_r2:.4f})")

    # 6. Save Best Model
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "insurance_model.joblib")
    joblib.dump(best_pipe, model_path)
    print(f"\n[4] Saved best pipeline model (predicts THB) to: {model_path}")

    # 7. Save Metrics Summary
    metrics_path = os.path.join("models", "insurance_evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "target_currency": "THB",
            "exchange_rate": EXCHANGE_RATE,
            "best_model": best_name,
            "best_r2": best_r2,
            "all_results": results
        }, f, indent=2, ensure_ascii=False)
    print(f"[5] Saved evaluation metrics to: {metrics_path}")
    print("\n✅ Training and evaluation completed successfully!\n")

if __name__ == "__main__":
    main()
