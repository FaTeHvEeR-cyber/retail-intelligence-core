Retail Demand Forecasting & Fraud Simulation System

End-to-end analytical pipeline combining time-series demand forecasting, statistical hypothesis testing, unsupervised store segmentation, and neural network anomaly detection for point-of-sale (POS) return fraud — served through an interactive Streamlit dashboard.

1. Project Overview

Capability

Method

Output

Demand Forecasting

Ridge → XGBoost/LightGBM → Prophet/MLP

Store-level sales predictions, RMSPE/MAE/RMSE

Hypothesis Testing

Welch's t-test, One-Way ANOVA

Statistical validation of promo lift & store-type variance

Store Segmentation

StandardScaler → K-Means → PCA

Behavioral store clusters

Fraud/Anomaly Detection

Autoencoder / ANN

Reconstruction-error based return-fraud risk scores

Dashboard

Streamlit

4-tab executive interface tying all modules together

Baseline dataset: Rossmann Store Sales (Kaggle) Secondary dataset: Walmart Recruiting Store Sales Forecasting (Kaggle) Fraud dataset: Retail Intelligence: Fraud Detection Dataset (Kaggle) + synthetic return-log generator

2. Directory Structure

retail_forecasting_project/

├── data/

│   ├── raw/                  <-- train.csv, store.csv (Rossmann)

│   └── processed/            <-- feature-engineered, split datasets

├── src/

│   ├── __init__.py

│   ├── data_prep.py          <-- ingestion, cleaning, lag/rolling features, chrono split

│   ├── hypothesis_testing.py <-- Welch's t-test, ANOVA

│   ├── clustering.py         <-- K-Means, Silhouette Score, PCA

│   ├── train_models.py       <-- Ridge, XGBoost/LightGBM, Prophet/MLP

│   ├── fraud_detection.py    <-- synthetic return generator, Autoencoder/ANN

│   ├── evaluate.py           <-- RMSPE, MAE, RMSE, R²

│   └── utils.py              <-- serialization, logging, plotting helpers

├── models/                   <-- serialized artifacts (.pkl / .pt / .json), versioned

├── notebooks/                <-- EDA & prototyping

├── reports/                  <-- comparison_metrics.json, SHAP plots

├── app/

│   └── app.py                <-- Streamlit dashboard (4 tabs)

├── requirements.txt

└── README.md

3. Non-Negotiable Technical Constraints

Zero data leakage: Never use random train/test splits on time-series data. Training uses historical records; validation uses the final 6 weeks, held out chronologically. Lag/rolling features must only reference t-1, t-7, ... (past only).

Dataset agnosticism: Column mapping is config-driven (YAML/dict), so any retail time-series dataset with equivalent fields (date, store_id, sales, promo, store_type) can be ingested without rewriting core logic.

Missing value policy: CompetitionDistance → median imputation (not zero). Promo2SinceWeek / Promo2SinceYear → zero-flag imputation (absence of promo program is meaningful, unlike missing distance).

Filter before training: Drop Open == 0 records and, where applicable, zero-sales rows that reflect closures rather than genuine demand.

Evaluation standard: Primary metric = RMSPE. Always report MAE and RMSE alongside it. Track training/inference time per model.

Serialization: All fitted scalers, encoders, and model weights saved to models/ with timestamp/version tags — never overwrite silently.

Fraud thresholding: No labeled ground truth assumed. Flag anomalies at the 95th/99th percentile of reconstruction error (MSE), not a fixed cutoff.

4. Setup

python -m venv venv

source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

Place raw Kaggle files (train.csv, store.csv) into data/raw/ before running Phase 1.

5. Execution Guide (Phase by Phase)

Phase 1 — Data Engineering

python -m src.data_prep --raw-dir data/raw --out-dir data/processed

Produces cleaned, feature-engineered, chronologically-split train/validation sets in data/processed/.

Phase 2 — Hypothesis Testing & Clustering

python -m src.hypothesis_testing --data data/processed/train.parquet --out reports/hypothesis_tests.json

python -m src.clustering --data data/processed/train.parquet --out reports/store_clusters.json

Phase 3 — Model Training & Evaluation

python -m src.train_models --data data/processed --models-dir models/

python -m src.evaluate --data data/processed --models-dir models/ --out reports/comparison_metrics.json

Phase 4 — Fraud Simulation & Anomaly Detection

python -m src.fraud_detection --generate-synthetic --n-records 50000 --out data/processed/returns.parquet

python -m src.fraud_detection --train --data data/processed/returns.parquet --models-dir models/

Phase 5 — Launch Dashboard

streamlit run app/app.py

6. Dashboard Tabs

Executive Overview & EDA — KPI banner, historical trends, correlation heatmaps

Demand Forecast & Scenario Simulator — actual vs. predicted curves, what-if promo slider

Store Segmentation — PCA scatter plot of K-Means clusters

POS Anomaly Alert Feed — real-time table of flagged transactions with risk scores

7. Roadmap Status

Phase 1 — Data pipeline

Phase 2 — Hypothesis testing & clustering

Phase 3 — Forecasting models

Phase 4 — Fraud detection

Phase 5 — Dashboard