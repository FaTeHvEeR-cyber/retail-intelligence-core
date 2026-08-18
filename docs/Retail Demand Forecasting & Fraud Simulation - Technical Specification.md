RETAIL DEMAND FORECASTING & FRAUD SIMULATION SYSTEM

Technical Specification & Architecture Documentation

1. PROJECT OVERVIEW & CORE OBJECTIVES

Project Title: End-to-End Retail Demand Forecasting, Store Segmentation, & Point-of-Sale Anomaly Detection System

Domain: Enterprise Retail Analytics, Time-Series Forecasting, Machine Learning, Deep Learning, MLOps

Core Objective: Design, build, and deploy an end-to-end analytical pipeline that ingests historical multi-store retail data (Rossmann Store Sales dataset), performs rigorous statistical inference and store clustering, benchmarks three distinct forecasting architectures, detects transaction/return anomalies using neural networks, and serves actionable insights via an interactive dashboard.

2. SYSTEM ARCHITECTURE & DIRECTORY STRUCTURE

The codebase follows a modular production layout:retail_forecasting_system/

│

├── data/

│   ├── raw/                  # Original raw datasets (train.csv, store.csv, test.csv)

│   └── processed/            # Cleaned, feature-engineered, and split datasets

│

├── src/

│   ├── __init__.py

│   ├── data_prep.py          # Data ingestion, cleaning, and time-aware feature engineering

│   ├── hypothesis_testing.py # Parametric & non-parametric statistical hypothesis testing

│   ├── clustering.py         # Unsupervised store segmentation (K-Means & PCA)

│   ├── train_models.py       # Multi-model benchmarking engine (Ridge, XGBoost, Prophet/MLP)

│   ├── fraud_detection.py    # POS anomaly/fraud simulation & neural network detection

│   ├── evaluate.py           # Domain evaluation suite (RMSPE, MAE, RMSE, R2)

│   └── utils.py              # Serialization, logging, and plotting utilities

│

├── models/                   # Serialized model artifacts (.pkl, .json)

├── notebooks/                # Exploratory Data Analysis & prototyping notebooks

├── app/

│   └── app.py                # Streamlit interactive business dashboard

│

├── requirements.txt          # Production dependencies

└── README.md                 # Project documentation and run instructions

3. DETAILED COMPONENT BREAKDOWN

MODULE 1: Data Engineering & Preprocessing (src/data_prep.py)

Primary Inputs: Rossmann historical daily sales records (train.csv), store metadata (store.csv).

Data Cleaning & Imputation:

Impute missing CompetitionDistance using median values.

Impute missing promotional intervals (Promo2SinceWeek, Promo2SinceYear) with zero flags.

Filter out closed store records (Open == 0) and zero-sales records where applicable for model training.

Encode categorical store descriptors (StoreType, Assortment, StateHoliday) using one-hot/binary encodings.

Time-Series Feature Engineering:

Temporal Lags: 7-day, 14-day, 21-day, and 30-day lagged sales (y_{t-7}, y_{t-14}, y_{t-21}, y_{t-30}).

Rolling Window Statistics: 7-day, 14-day, and 30-day moving averages and moving standard deviations.

Calendar Features: DayOfWeek, Day, Month, Year, WeekOfYear, IsWeekend, Promo, Promo2, SchoolHoliday, StateHoliday.

Competitive Dynamics: Elapsed months since competition opened, elapsed weeks since promo entry.

Temporal Validation Strategy (Zero Data Leakage):

Strictly enforce a chronological time-aware train-validation split (e.g., training on historical records up to the final 6 weeks; using the final 6 weeks as the validation holdout). Random splits are strictly prohibited.

MODULE 2: Statistical Inference & Hypothesis Testing (src/hypothesis_testing.py)

Objective: Statistically validate business assumptions regarding marketing promotions and store attributes.

Implemented Tests:

Two-Sample Welch’s t-test: Compare mean daily sales during promotional periods (Promo = 1) vs. non-promotional periods (Promo = 0).

One-Way ANOVA: Evaluate variance in sales performance across different store models (StoreType A, B, C, D) and assortment levels (Assortment a, b, c).

Significance Level: Alpha = 0.05. Export t-statistics, F-statistics, and p-values with automated business interpretation.

MODULE 3: Store Clustering & Behavioral Profiling (src/clustering.py)

Objective: Group stores with similar sales patterns, customer volume, and promotional responsiveness.

Engineered Store-Level Aggregations:

Mean daily sales, sales coefficient of variation (volatility), average customer traffic per day, sales lift percentage under promotion.

Unsupervised Pipeline:

Standard scaling (StandardScaler).

K-Means Clustering: Optimal cluster selection via the Elbow Method (Inertia) and Silhouette Score analysis.

Dimensionality Reduction: Principal Component Analysis (PCA) to project store clusters into 2D/3D space for visual profiling.

MODULE 4: Multi-Model Demand Forecasting Engine (src/train_models.py & src/evaluate.py)

Model Benchmark Architectures:

Linear Regularized Baseline: Ridge Regression (penalizes multicollinearity among lagged features).

Gradient Boosted Decision Tree Ensemble: XGBoost Regressor / LightGBM (captures non-linear dynamics, interactions, and feature hierarchies).

Additive Time-Series / Neural Architecture: Prophet (multi-period seasonality and holiday modeling) or MLP Regressor (deep non-linear representations).

Validation & Tuning: TimeSeriesSplit cross-validation across rolling temporal folds.

Evaluation Metrics Suite:

Primary Metric: Root Mean Square Percentage Error (RMSPE).

Secondary Metrics: Mean Absolute Error (MAE), Root Mean Square Error (RMSE), Coefficient of Determination (R^2).

Artifact Serialization: Save trained estimators, preprocessors, and metric logs to models/ using joblib or native model serialization.

MODULE 5: Point-of-Sale Anomaly & Fraud Simulation (src/fraud_detection.py)

Objective: Identify anomalous return spikes, suspicious refund behaviors, and potential point-of-sale policy abuse.

Synthetic Return Log Generation:

Simulate transaction-level return amounts, return frequencies, receipt verification status, and customer tenure.

Neural Anomaly Detection Engine:

Architecture: Deep Artificial Neural Network (ANN) / Autoencoder.

Mechanism: Model standard transaction patterns; compute reconstruction error (MSE) for each transaction.

Thresholding: Flag transactions exceeding the 95th / 99th percentile reconstruction error as high-risk anomalies.

MODULE 6: Interactive Dashboard (app/app.py)

Framework: Streamlit

Dashboard Tabs:

Executive Overview & Exploratory Data Analysis: KPI metrics (total volume, store count, overall promotional lift), historical trends, and correlation heatmaps.

Demand Forecast & Scenario Simulator: Interactive store-level sales forecasting with a "What-If" promotional planner.

Store Segmentation View: Visual cluster maps (PCA scatter plots) with behavioral store profiles.

POS Anomaly Alert Feed: Real-time stream of flagged transactions with severity scores and risk breakdown.

4. STEP-BY-STEP IMPLEMENTATION ROADMAP

Phase 1: Environment & Data Pipeline Setup

Initialize repository structure, install dependencies (pandas, numpy, scipy, scikit-learn, xgboost, streamlit, matplotlib, seaborn).

Implement src/data_prep.py with temporal lag generation and chronological splitting.

Phase 2: Statistical Hypothesis Testing & Store Segmentation

Execute Welch's t-tests and ANOVA in src/hypothesis_testing.py.

Build K-Means clustering and PCA visualization in src/clustering.py.

Phase 3: Model Training & Evaluation Suite

Train Ridge, XGBoost, and Prophet/MLP models in src/train_models.py.

Compute RMSPE, MAE, and RMSE on the holdout set using src/evaluate.py.

Phase 4: POS Anomaly Detection Module

Generate simulated return datasets and train the anomaly detection Autoencoder/ANN in src/fraud_detection.py.

Phase 5: Streamlit Interface & System Integration

Develop app/app.py connecting all serialized models, preprocessors, and visualizations.

Verify end-to-end pipeline execution from raw data ingestion to interactive inference.