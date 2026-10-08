# demand-forecasting-uq# Demand Forecasting with Uncertainty Quantification (M5 Walmart Case Study)

## Project Overview
This repository delivers an end-to-end framework for **asymmetric demand forecasting with calibrated uncertainty intervals** using real-world retail transactions from the M5 Walmart dataset. 

Instead of traditional central tendency modeling (MSE/Point Forecasts), this architecture builds distinct multi-quantile pipelines (P₁₀, P₅₀, P₉₀) to dynamically estimate future supply safety boundaries.

## Core Features & Methodology
1. **Feature Engineering:** We extract cross-sectional time configurations including calendar characteristics (`day_of_week`, `month`), promotional trackers (`is_promo`), and 28-day lagged rolling windows (`rolling_mean_28`) to capture non-stationary demand signals without target data leakage.
2. **Asymmetric Optimization:** We use `LightGBM` using a pinball/quantile loss function to directly prioritize regional prediction bands.
3. **Validation Strategy:** Out-of-time temporal division splitting is performed to strictly ensure model integrity across future operational horizons.

## Metrics & Calibration Outcomes
* **Point Performance:** Evaluated via Mean Absolute Error (MAE) at the 50th percentile.
* **Interval Calibration:** Assessed by cross-verifying whether empirical distribution accuracy matches our target bounds (e.g., ensuring actual values fall inside our 10th-90th interval ~80% of the time).
* **Business Translation:** The upper boundary predictions (P₉₀) map directly to dynamic safety stock allocations. By transitioning inventory parameters from the median (P₅₀) to the upper bound (P₉₀), overall financial loss from stockouts is minimized—even when factoring in trailing warehouse carrying costs.
* 
