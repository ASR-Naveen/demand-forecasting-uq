import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =====================================================================
# 1. SIMULATED DATASET GENERATION (Mimicking Store/Item Demand)
# =====================================================================
def generate_mock_data():
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", end="2025-12-31", freq="D")
    df = pd.DataFrame({"date": dates})
    
    # Base trend + structural seasonality
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    
    # Base demand equation
    base_demand = 100 + (df["month"] * 2) + (df["day_of_week"] * 5)
    
    # Introduce promotional spikes & holiday anomalies (Heteroscedasticity)
    df["is_promo"] = np.where(np.random.rand(len(df)) > 0.85, 1, 0)
    promo_effect = df["is_promo"] * np.random.normal(50, 15, size=len(df))
    
    # Varied noise profile mimicking market uncertainty
    noise = np.random.normal(0, 10 + (df["is_promo"] * 20), size=len(df))
    
    df["sales"] = np.maximum(0, base_demand + promo_effect + noise).astype(int)
    return df

# =====================================================================
# 2. ADVANCED TIME-BASED FEATURE ENGINEERING
# =====================================================================
def engineer_features(df):
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    
    # Calendar & Cyclical features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    
    # Lag Features (Shifted to prevent target data leakage)
    for lag in:
        df[f"sales_lag_{lag}"] = df["sales"].shift(lag)
        
    # Rolling Statistics
    for window in:
        df[f"rolling_mean_{window}"] = df["sales"].shift(1).rolling(window=window).mean()
        df[f"rolling_std_{window}"] = df["sales"].shift(1).rolling(window=window).std()
        
    # Drop rows with NaNs caused by deep lags/rolling operations
    df = df.dropna().reset_index(drop=True)
    return df

# =====================================================================
# 3. QUANTILE REGRESSION PIPELINE VIA LIGHTGBM
# =====================================================================
def train_quantile_models(train_df, features, target, quantiles=[0.10, 0.50, 0.90]):
    models = {}
    
    for q in quantiles:
        print(f"Training LightGBM model for Quantile: {q}")
        # Configure model specifically for objective evaluation targeting your specific interval
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        model.fit(train_df[features], train_df[target])
        models[q] = model
        
    return models

# =====================================================================
# 4. PERFORMANCE EVALUATION & INTERVAL CALIBRATION
# =====================================================================
def evaluate_forecast(test_df, predictions, alpha_low=0.10, alpha_high=0.90):
    y_true = test_df["sales"]
    y_pred_50 = predictions[0.50]
    
    # Point Accuracy Metrics
    mae = mean_absolute_error(y_true, y_pred_50)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_50))
    
    # Interval Calibration (Empirical Coverage Check)
    # The actual values should ideally fall inside the boundaries exactly (alpha_high - alpha_low) of the time
    in_bounds = (y_true >= predictions[alpha_low]) & (y_true <= predictions[alpha_high])
    empirical_coverage = np.mean(in_bounds)
    expected_coverage = alpha_high - alpha_low
    
    print("\n================ EVALUATION METRICS ================")
    print(f"Point Forecast MAE  : {mae:.2f}")
    print(f"Point Forecast RMSE : {rmse:.2f}")
    print(f"Target Coverage Interval : {expected_coverage * 100:.1f}%")
    print(f"Empirical Coverage Rate : {empirical_coverage * 100:.1f}%")
    
    # Analyze Overconfidence/Underconfidence Breakdown
    if empirical_coverage < expected_coverage:
        print(" -> System Status: Overconfident (Interval is too narrow; missing variations).")
    else:
        print(" -> System Status: Calibrated/Conservative (Interval meets or captures uncertainty safely).")

# =====================================================================
# 5. BUSINESS TRANSLATION: DECISION FRAMEWORK
# =====================================================================
def apply_inventory_logic(test_df, predictions, upper_quantile=0.90):
    decision_df = test_df[["date", "sales", "is_promo"]].copy()
    decision_df["point_forecast"] = predictions[0.50]
    decision_df["upper_bound"] = predictions[upper_quantile]
    
    # Inventory translation rule: Safety Stock = Upper Quantile - Point Forecast
    decision_df["safety_stock"] = np.maximum(0, decision_df["upper_bound"] - decision_df["point_forecast"])
    decision_df["recommended_inventory"] = decision_df["upper_bound"]
    
    # Assess Cost Tradeoffs (Hypothetical Overstock vs Understock Cost factors)
    # Holding excess item inventory costs $2/unit, stockout loss penalty costs $10/unit
    cost_holding = 2.0
    cost_stockout = 10.0
    
    # Strategy A: Ordering exclusively based on standard point forecast (P50)
    shortage_p50 = np.maximum(0, decision_df["sales"] - decision_df["point_forecast"])
    excess_p50 = np.maximum(0, decision_df["point_forecast"] - decision_df["sales"])
    cost_p50 = (shortage_p50 * cost_stockout) + (excess_p50 * cost_holding)
    
    # Strategy B: Ordering using Upper Quantile to buffer risk safely
    shortage_q90 = np.maximum(0, decision_df["sales"] - decision_df["recommended_inventory"])
    excess_q90 = np.maximum(0, decision_df["recommended_inventory"] - decision_df["sales"])
    cost_q90 = (shortage_q90 * cost_stockout) + (excess_q90 * cost_holding)
    
    print("\n=========== FINANCIAL COST-TRADEOFF ANALYSIS ===========")
    print(f"Total Operational Risk Cost via Point Forecast (P50): ${cost_p50.sum():,.2f}")
    print(f"Total Operational Risk Cost via Upper Bound (P90)   : ${cost_q90.sum():,.2f}")
    print(f"Net Financial Savings using Uncertainty Buffer       : ${cost_p50.sum() - cost_q90.sum():,.2f}")
    
    return decision_df

# =====================================================================
# EXECUTION PIPELINE
# =====================================================================
if __name__ == "__main__":
    # Fetch Data
    raw_data = generate_mock_data()
    processed_data = engineer_features(raw_data)
    
    # Define features and dynamic boundaries
    feature_cols = [col for col in processed_data.columns if col not in ["date", "sales"]]
    target_col = "sales"
    
    # Out-of-time Temporal Validation Split (Train: 2024, Test: Last 4 months of 2025)
    split_date = "2025-09-01"
    train_set = processed_data[processed_data["date"] < split_date]
    test_set = processed_data[processed_data["date"] >= split_date]
    
    # Model Training
    quantiles_to_test = [0.10, 0.50, 0.90]
    trained_models = train_quantile_models(train_set, feature_cols, target_col, quantiles_to_test)
    
    # Generate Predictions across the Out-of-time framework
    preds_output = {}
    for q, model in trained_models.items():
        preds_output[q] = model.predict(test_set[feature_cols])
        
    # Evaluate Pipeline Metrics & Coverage
    evaluate_forecast(test_set, preds_output)
    
    # Translate Framework output directly to retail actions
    final_decisions = apply_inventory_logic(test_set, preds_output)
