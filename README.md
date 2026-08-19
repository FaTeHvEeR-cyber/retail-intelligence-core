# Retail Demand Forecasting & Fraud Simulation System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/pytest-29%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, production-ready retail analytics and forecasting engine combining **multi-model time-series demand forecasting**, **statistical hypothesis testing**, **unsupervised store segmentation (K-Means + PCA)**, and **deep learning anomaly detection (PyTorch Autoencoder)** for Point-of-Sale (POS) return fraud — served via a Power BI-style interactive Streamlit dashboard.

---

## 1. System Capabilities & Architecture

| Capability | Methods / Models | Key Outputs |
|---|---|---|
| **Demand Forecasting** | Ridge Regression, XGBoost Regressor, MLP Regressor | Store-level sales forecasts, RMSPE, MAE, RMSE, $R^2$, latency benchmarks |
| **Hypothesis Testing** | Welch's t-test, One-Way ANOVA | Promo lift statistical significance ($p < 10^{-15}$), store type / assortment variance |
| **Store Segmentation** | StandardScaler → K-Means ($K=4$) → PCA 2D Projection | Behavioral store cluster assignments & PCA coordinates |
| **POS Fraud Detection** | Synthetic Return Log Engine + PyTorch Autoencoder | Reconstruction MSE anomaly scoring (95th/99th percentile thresholding) |
| **Executive Interface** | Streamlit + Plotly | 4-tab interactive executive BI dashboard with real-time what-if scenario testing |

---

## 2. Directory Architecture

```text
retail_forecasting_project/
├── data/
│   ├── raw/                  <- Rossmann train.csv, store.csv, test.csv
│   └── processed/            <- Processed parquet datasets with lag & rolling features
├── src/
│   ├── __init__.py           <- Package initialization
│   ├── data_prep.py          <- Feature engineering, imputation & time-aware split
│   ├── hypothesis_testing.py <- Welch's t-test, One-Way ANOVA engine
│   ├── clustering.py         <- Behavioral profiling, K-Means clustering, PCA projection
│   ├── train_models.py       <- Multi-model forecasting engine (Ridge, XGBoost, MLP)
│   ├── evaluate.py           <- Benchmark evaluation suite (RMSPE, MAE, RMSE, R²)
│   └── fraud_detection.py    <- Synthetic POS return generator & PyTorch Autoencoder
├── models/                   <- Serialized pipelines (.joblib) & latest_manifest.json
├── reports/                  <- comparison_metrics.json, store_clusters.csv, hypothesis_tests.json
├── tests/                    <- Complete unit test suite (16 test cases)
├── configs/                  <- Dataset-agnostic column mapping configs (YAML)
├── app.py                    <- Power BI-style Streamlit Dashboard
├── requirements.txt          <- Environment dependencies
└── README.md                 <- Project execution guide & technical reference
```

---

## 3. Non-Negotiable Technical Constraints & Design Principles

1. **Zero Data Leakage**:
   - Strictly time-aware chronological validation split (final 6 weeks held out: `2015-06-19` to `2015-07-31`).
   - Cross-validation uses rolling temporal folds via `TimeSeriesSplit` — no random `KFold` or shuffle.
   - Lag and rolling statistics reference only past timestamps ($t-7, t-14, t-21, t-30$).
   - Categorical encoders and scalers are fit strictly on training splits.
2. **Missing Value Imputation Policy**:
   - `CompetitionDistance`: Imputed with median ($2320.0\,\text{m}$), preserving distance distributions.
   - `Promo2SinceWeek`, `Promo2SinceYear`, `PromoInterval`: Imputed with zero-flags (absence of active promotion program is meaningful signal).
3. **Anomaly & Outlier Isolation**:
   - Records with `Open == 0` (closed stores) are filtered out before feature engineering.
   - Records with `Open == 1` and `Sales == 0` are flagged as `is_zero_sales_anomaly` and excluded from model training to prevent gradient distortion.
4. **Target Leakage Prevention**:
   - `Customers`, `Open`, and `Date` are strictly excluded from forecasting feature matrices.
5. **Primary Metric (RMSPE)**:
   - Root Mean Square Percentage Error:
     $$\text{RMSPE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left(\frac{y_i - \hat{y}_i}{y_i}\right)^2}$$
   - Zero-sales rows are excluded from RMSPE denominator to prevent division-by-zero.

---

## 4. Benchmark Evaluation Results

Evaluated on **40,282** held-out validation samples across all 1,115 stores:

| Rank | Model | Architecture | RMSPE | MAE (€) | RMSE (€) | $R^2$ | Train Time (s) | Inference Latency (s) |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **#1** | **XGBoost Regressor** | Gradient-Boosted Trees | **0.1405** | **627.43** | **904.21** | **0.9124** | 7.86s | 0.0737s |
| **#2** | **MLP Regressor** | Deep Neural Network | **0.1761** | **762.54** | **1,073.60** | **0.8765** | 111.76s | 0.0930s |
| **#3** | **Ridge Regression** | Regularized Linear Baseline | **0.2093** | **876.76** | **1,244.92** | **0.8339** | 3.70s | 0.0712s |

---

## 5. Statistical Validation & Segmentation Insights

### Statistical Hypothesis Testing (`reports/hypothesis_tests.json`)
- **Promotional Lift (Welch's t-test)**:
  - Promo Active Mean Sales: **€8,216.03** vs. Non-Promo Mean Sales: **€5,931.25**
  - Relative Lift: **+38.52%** ($t = 345.77, p < 10^{-15}$, Highly Significant)
- **StoreType Variance (One-Way ANOVA)**:
  - $F = 5,712.89, p < 10^{-15}$ (Type `b` exhibits highest volume, Type `d` highest stability)
- **Assortment Variance (One-Way ANOVA)**:
  - $F = 5,723.84, p < 10^{-15}$ (Extra assortment `c` demonstrates significant lift over basic `a`)

### Store Segmentation (`reports/store_clusters.json`)
- Optimal cluster count determined automatically via Silhouette Score analysis ($K=4$, Score: $0.3230$).
- **Cluster 0**: High-volume, high promo-sensitivity flagship stores.
- **Cluster 1**: Moderate-volume steady suburban neighborhood stores.
- **Cluster 2**: High-traffic compact city stores (Assortment `b`).
- **Cluster 3**: Highly volatile rural stores with low competition proximity.

---

## 6. Installation & Execution Guide

### 1. Environment Setup
```bash
# Clone and enter workspace
cd retail-intelligence-core

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Pipeline Execution Steps

```bash
# Step 1: Feature Engineering & Preprocessing
python -m src.data_prep --raw-dir data/raw --out-dir data/processed

# Step 2: Statistical Hypothesis Testing
python -m src.hypothesis_testing --data data/processed/train_processed.parquet --out reports/hypothesis_tests.json

# Step 3: Unsupervised Store Clustering
python -m src.clustering --data data/processed/train_processed.parquet --out-dir reports

# Step 4: Multi-Model Demand Forecasting Training
python -m src.train_models --data data/processed --models-dir models/

# Step 5: Model Evaluation & Benchmarking
python -m src.evaluate --data data/processed --models-dir models/ --out reports/comparison_metrics.json

# Step 6: Launch Interactive Streamlit Dashboard
streamlit run app.py
```

### 3. Running Unit Tests
```bash
pytest -v tests/
```

---

## 7. Project Roadmap

- [x] **Phase 1**: Data Engineering, Imputation, Lags/Rolling Windows & Chronological Split
- [x] **Phase 2**: Hypothesis Testing (Welch's t-test / ANOVA) & Store Clustering (K-Means / PCA)
- [x] **Phase 3**: Multi-Model Forecasting Benchmark (Ridge, XGBoost, MLP) & Serialization
- [x] **Phase 4**: POS Return Fraud Simulation & PyTorch Autoencoder Anomaly Detection
- [x] **Phase 5**: Executive Streamlit BI Dashboard (5-Tab Power BI layout)
