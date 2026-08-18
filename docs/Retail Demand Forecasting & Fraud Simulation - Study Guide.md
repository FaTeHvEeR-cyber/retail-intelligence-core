Retail Demand Forecasting + Fraud Simulation - Study Guide & Real-World Reference

1. Executive Summary & Project Context

This study guide breaks down the "Retail Demand Forecasting + Fraud Simulation" project assigned by your mentor. It combines time-series forecasting, statistical hypothesis testing, store behavior clustering, and neural network anomaly detection to solve critical retail operations problems.

2. Technical Component Breakdown

Time-Aware EDA & Feature Engineering: Analyzing sales trends, seasonality, day-of-week effects, lag features (7-day/30-day lag), and rolling averages.

Hypothesis Testing: Assessing promotional significance using two-sample t-tests or ANOVA to prove whether marketing campaigns drive statistically significant sales lift.

Demand Forecasting: Baseline Linear Regression and gradient boosting (XGBoost / LightGBM) to forecast future sales across stores/departments.

Store Behavior Clustering: Unsupervised KMeans clustering to group stores by sales volume, promotional response, and return activity.

Anomaly & Fraud Flagging: Artificial Neural Networks (ANN / Multi-Layer Perceptrons or Autoencoders) to detect fraud-like return behaviors (e.g., return abuse, receipt fraud).

3. Real-World Pre-Made Projects & GitHub Benchmarks

Multi-Store Demand Forecasting & Inventory Optimization System: A full-stack pipeline using XGBoost, SHAP explainability, Economic Order Quantity (EOQ), FastAPI, and Streamlit.

Rossmann Sales Prediction Pipeline: Modular XGBoost forecasting pipeline built on the Rossmann dataset with time-aware cross-validation.

Walmart Global Tech - Autoencoders for Store Returns Anomaly Detection: Technical breakdown by Walmart engineers using unsupervised neural networks (Autoencoders / VAE) to flag anomalous store return patterns.

DAFU - Enterprise Fraud Detection Platform: Open-source e-commerce and retail fraud detection suite implementing ML models and sequence analysis.

4. Key Datasets to Practice On

Rossmann Store Sales on Kaggle: Historical daily sales for 1,115 stores with promotional calendar and holiday flags.

Walmart Recruiting Store Sales Forecasting on Kaggle: Sales data across 45 stores with department-level details and markdown events.

Retail Intelligence: Fraud Detection Dataset on Kaggle: 100,000+ retail transaction records with risk indicators and behavioral fraud signals.

5. Real-Time Business Value & Dashboard Architecture

Real-Time Inventory Alerts: Predicting stockouts 2-4 weeks ahead to trigger automated replenishment.

Point-of-Sale Anomaly Scoring: Real-time risk scoring during customer returns to prevent return fraud.

Streamlit / Dash Integration: Interactive charts for demand curves, cluster maps, and flagged fraudulent return logs.

6. Step-by-Step Learning Roadmap

Phase 1: Data Preparation & Time-Aware EDA (Pandas, Datetime, Lag/Rolling Features).

Phase 2: Statistical Hypothesis Testing (Scipy Stats, t-tests, ANOVA).

Phase 3: Demand Forecasting Models (Scikit-Learn, XGBoost, Evaluation metrics like RMSE/MAE).

Phase 4: Unsupervised Clustering (KMeans, Silhouette Score, PCA visualization).

Phase 5: Deep Learning Anomaly Detection (PyTorch/Keras ANN for fraud scoring).

Phase 6: Interactive Dashboard & AI Agent Interface (Streamlit deployment).

7. Codebase Directory Blueprint & Production AI Agent Prompt

#### Visual Project Directory Structure

retail_forecasting_project/

├── 📁 data/

│   ├── 📄 raw/                     <-- Rossmann train.csv and store.csv

│   └── 📄 processed/               <-- Processed datasets with lag & rolling features

├── 📁 src/

│   ├── 📄 __init__.py              <-- Package initialization

│   ├── 📄 data_prep.py             <-- Feature engineering, imputation & time-aware split

│   ├── 📄 train_models.py          <-- Model implementations (Ridge, XGBoost, Prophet/MLP)

│   └── 📄 evaluate.py              <-- Evaluation suite (RMSPE, MAE, RMSE calculation)

├── 📁 models/                      <-- Serialized trained weights (.pkl / .json)

├── 📁 reports/                     <-- comparison_metrics.json & SHAP plots

├── 📄 app.py                       <-- Power BI-style Streamlit Dashboard

├── 📄 requirements.txt             <-- Environment requirements

└── 📄 README.md                    # Project execution guide

#### Enhanced Production Master Prompt for AI Coding Agent

Copy and paste the following prompt into your AI agent environment (VS Code / Anti-Gravity / Cursor):

Role: Senior Machine Learning & Full-Stack AI Engineer.

Task: Construct an end-to-end, production-ready Python codebase for the Rossmann Retail Store Sales Forecasting & Model Benchmarking project.

Project Requirements & Technical Specifications:

1. Directory & Environment Architecture:

   - Create the following modular folder layout: data/raw, data/processed, src/, models/, reports/, app.py, requirements.txt, and README.md.

   - Specify necessary libraries in requirements.txt (pandas, numpy, scikit-learn, xgboost, prophet, streamlit, plotly, shap).

   - Directory Architecture:

retail_forecasting_project/

├── 📁 data/

│   ├── 📄 raw/                     <-- Rossmann train.csv and store.csv

│   └── 📄 processed/               <-- Processed datasets with lag & rolling features

├── 📁 src/

│   ├── 📄 __init__.py              <-- Package initialization

│   ├── 📄 data_prep.py             <-- Feature engineering, imputation & time-aware split

│   ├── 📄 train_models.py          <-- Model implementations (Ridge, XGBoost, Prophet/MLP)

│   └── 📄 evaluate.py              <-- Evaluation suite (RMSPE, MAE, RMSE calculation)

├── 📁 models/                      <-- Serialized trained weights (.pkl / .json)

├── 📁 reports/                     <-- comparison_metrics.json & SHAP plots

├── 📄 app.py                       <-- Power BI-style Streamlit Dashboard

├── 📄 requirements.txt             <-- Environment requirements

└── 📄 README.md                    # Project execution guide

2. Data Engineering Pipeline (src/data_prep.py):

   - Load and merge Rossmann train.csv and store.csv on Store ID.

   - Clean missing values (e.g., fill missing CompetitionDistance with median value).

   - Engineer time-based features: DayOfWeek, Month, Quarter, IsPromo, StateHoliday, and SchoolHoliday.

   - Construct time-series lag features: 7-day sales lag, 14-day sales lag, and 7-day rolling mean sales per store.

   - Implement a strict time-aware split: Use historical data for training, and reserve the last 6 weeks of data as the validation set (preventing future data leakage).

3. Multi-Model Benchmarking Engine (src/train_models.py):

   - Train and save three distinct model architectures:

     a) Baseline Model: Ridge Regression (Linear Regularized Baseline).

     b) Ensemble Model: XGBoost Regressor or LightGBM Regressor (Gradient Boosting).

     c) Time-Series / Neural Net Model: Prophet or MLP Regressor (Neural Net).

   - Save all trained model artifacts into models/.

4. Evaluation Suite (src/evaluate.py):

   - Evaluate all models on the 6-week validation set using the official competition metric:

     RMSPE = sqrt(mean(((y_true - y_pred) / y_true) ** 2))

   - Compute secondary metrics: MAE (Mean Absolute Error) and RMSE.

   - Track training and inference execution times.

   - Save the evaluation summary to reports/comparison_metrics.json.

5. Executive Power BI-Style Interactive Dashboard (app.py):

   - Build a Streamlit application featuring:

     - Top KPI Metrics Banner: Winning Model Name, Best RMSPE Score, Total Projected Revenue, Avg MAE Error.

     - Model Benchmark Visual: Grouped bar chart comparing RMSPE, MAE, and Training Time across Ridge, XGBoost, and Prophet/MLP.

     - Forecast Explorer: Interactive Store ID selector overlaid with Actual Sales vs Predicted Sales time-series curves.

     - Feature Importance Section: Visual summary showing top feature drivers of sales volume.

6. Execution Readme (README.md):

   - Provide clear, step-by-step CLI commands to execute data preparation, model training, evaluation, and launching the Streamlit app.

