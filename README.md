# Retail Demand Forecasting & Fraud Simulation System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/pytest-29%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, production-grade retail intelligence and forecasting core combining **multi-model time-series demand forecasting**, **statistical hypothesis testing**, **unsupervised store segmentation (K-Means + PCA)**, and **neural anomaly detection (Bottleneck Autoencoder)** for Point-of-Sale (POS) return fraud — served via a Power BI-style interactive Streamlit dashboard.

**Live Demo:** [View the Streamlit Application Here](https://retail-intelligence-core-866d8wyzj752yio6rrt7g5.streamlit.app/)

---

## 1. System Capabilities & Architecture

| Capability | Methods / Models | Training Cycles / Epochs | Key Outputs |
|---|---|---|---|
| **Demand Forecasting** | Ridge Regression, XGBoost Regressor, MLP Regressor | • Ridge: Closed-form analytical<br>• XGBoost: 150 boosting rounds<br>• MLP: 30 epochs (batch size 256) | Store-level sales forecasts, RMSPE, MAE, RMSE, $R^2$, latency benchmarks |
| **Hypothesis Testing** | Welch's t-test, One-Way ANOVA | N/A (Frequentist Statistical Tests) | Promo lift statistical significance ($p < 10^{-15}$), store type / assortment variance |
| **Store Segmentation** | StandardScaler → K-Means ($K=4$) → PCA 2D Projection | K-Means convergence ($k=4$, silhouette score $0.3230$) | Behavioral store cluster assignments & PCA coordinates |
| **POS Fraud Detection** | Synthetic POS Stream Engine + Symmetric Bottleneck Autoencoder | 49 epochs (early-stopped from max 400 epochs) | Reconstruction MSE anomaly scoring (95th/99th percentile thresholding) |
| **Executive Interface** | Streamlit + Plotly BI Theme | Real-time interactive inference | 4-tab interactive executive BI dashboard with real-time what-if scenario testing |

---

## 2. Directory Architecture

```text
retail-intelligence-core/
├── .streamlit/
│   └── config.toml                           <- Streamlit theme and server configuration
├── configs/
│   └── rossmann_mapping.yaml                 <- Dataset-agnostic column mapping config
├── data/
│   ├── raw/                                  <- Rossmann store and sales raw CSVs
│   └── processed/                            <- Parquet datasets with engineered lag & rolling features
├── docs/
│   ├── Retail Demand Forecasting & Fraud Simulation - Technical Specification.md
│   └── Retail Demand Forecasting & Fraud Simulation - Study Guide.md
├── models/
│   ├── latest_manifest.json                  <- Model registry, metrics, and feature names manifest
│   ├── ridge_v20260818_191414.joblib         <- Serialized Ridge Regression pipeline
│   ├── xgboost_v20260818_191414.joblib       <- Serialized XGBoost Regressor pipeline
│   ├── mlp_v20260818_191414.joblib           <- Serialized MLP Regressor pipeline
│   ├── fraud_autoencoder_latest.joblib       <- Serialized POS Anomaly Autoencoder model
│   └── fraud_scaler_latest.joblib            <- Serialized POS Anomaly feature scaler
├── reports/
│   ├── comparison_metrics.json               <- Benchmark evaluation metrics across all models
│   ├── fraud_alerts.json                     <- Flagged high-risk POS anomaly alerts feed
│   ├── fraud_detection_metrics.json          <- POS detector threshold & diagnostic evaluation metrics
│   ├── hypothesis_tests.json                 <- Welch's t-test & ANOVA statistical test outputs
│   ├── store_clusters.csv                    <- Store cluster assignments table
│   └── store_clusters.json                   <- Store cluster profiles & centroid metrics
├── src/
│   ├── __init__.py                           <- Core package initialization
│   ├── data_prep.py                          <- Feature engineering, imputation & time-aware split
│   ├── hypothesis_testing.py                 <- Welch's t-test, One-Way ANOVA engine
│   ├── clustering.py                         <- Store profiling, K-Means clustering, PCA projection
│   ├── train_models.py                       <- Multi-model forecasting engine (Ridge, XGBoost, MLP)
│   ├── evaluate.py                           <- Benchmark evaluation suite (RMSPE, MAE, RMSE, R²)
│   └── fraud_detection.py                    <- POS return simulator & Bottleneck Autoencoder detector
├── tests/
│   ├── test_app.py                           <- Streamlit UI & prediction pipeline tests
│   ├── test_clustering.py                    <- K-Means & PCA segmentation unit tests
│   ├── test_data_prep.py                     <- Preprocessing & feature engineering unit tests
│   ├── test_evaluate.py                      <- Evaluation metric calculation & benchmark tests
│   ├── test_fraud_detection.py               <- POS anomaly simulation & autoencoder tests
│   ├── test_hypothesis_testing.py            <- Statistical hypothesis testing unit tests
│   └── test_train_models.py                  <- Model training & cross-validation unit tests
├── app.py                                    <- Power BI-style Executive Streamlit Dashboard
├── app_safe_state.py                         <- Session state safe initialization helper
├── clustering.py                             <- Root CLI shortcut to src.clustering
├── data_prep.py                              <- Root CLI shortcut to src.data_prep
├── evaluate.py                               <- Root CLI shortcut to src.evaluate
├── fraud_detection.py                        <- Root CLI shortcut to src.fraud_detection
├── hypothesis_testing.py                     <- Root CLI shortcut to src.hypothesis_testing
├── train_models.py                           <- Root CLI shortcut to src.train_models
├── run_dashboard.bat                         <- Windows quick-launch script
├── pyproject.toml                            <- Project configuration & packaging metadata
├── pytest.ini                                <- Pytest runner configuration
├── requirements.txt                          <- Python environment dependencies
├── runtime.txt                               <- Cloud deployment Python runtime specification
├── LICENSE                                   <- MIT License
└── README.md                                 <- Project execution guide & technical reference
```

---

## 3. Non-Negotiable Technical Constraints & Design Principles

1. **Zero Data Leakage**:
   - Strictly time-aware chronological validation split (final 6 weeks held out: `2015-06-19` to `2015-07-31`).
   - Cross-validation uses rolling temporal folds via `TimeSeriesSplit` (3 folds) — no random `KFold` or shuffle.
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

## 4. Model Training Dynamics & Benchmark Evaluation

### Model Training & Epoch Cycles Breakdown

| Model | Architecture / Method | Epochs / Training Iterations | Training Set Size | Training Time | Early Stopping / Convergence |
|---|---|:---:|:---:|:---:|---|
| **XGBoost Regressor** | Gradient-Boosted Decision Trees | **150 boosting rounds (trees)** | 804,056 rows | 7.86s | `learning_rate=0.08`, `max_depth=8`, `subsample=0.8` |
| **MLP Regressor** | Deep Neural Network (64 $\rightarrow$ 32) | **30 epochs** | 804,056 rows | 111.76s | Reached `max_iter=30`, `batch_size=256`, `activation=relu` |
| **Ridge Regression** | Regularized Linear Model ($\alpha=100.0$) | **Closed-form analytical solution** | 804,056 rows | 3.70s | Exact regularized least-squares matrix inversion |
| **POS Fraud Autoencoder** | Symmetric Bottleneck ANN ($5 \rightarrow 16 \rightarrow 8 \rightarrow 16 \rightarrow 5$) | **49 epochs** | 14,000 transactions | ~1.2s | Early stopped from `max_iter=400` (`n_iter_no_change=15`, $MSE \approx 0.00016$) |

### Benchmark Evaluation Results (`reports/comparison_metrics.json`)

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
git clone https://github.com/FaTeHvEeR-cyber/retail-intelligence-core.git
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

# Step 6: POS Anomaly Simulation & Autoencoder Training
python -m src.fraud_detection --models-dir models/ --reports-dir reports/

# Step 7: Launch Interactive Streamlit Dashboard
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
- [x] **Phase 4**: POS Return Fraud Simulation & Bottleneck Autoencoder Anomaly Detection
- [x] **Phase 5**: Executive Streamlit BI Dashboard (4-Tab Power BI layout with Scenario Simulation)
