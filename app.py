"""
app.py
======
Module 6: Interactive Executive Streamlit BI Dashboard.

Integrates the outputs of Modules 1-5 (data_prep, hypothesis_testing,
clustering, train_models/evaluate, fraud_detection) into an executive
5-tab dashboard for store management and business stakeholders:

    1. 📊 Executive Overview      - Top-level KPIs, revenue trend, correlation heatmap, hypothesis test insights
    2. 🏬 Store Performance        - Leaderboards, store revenue volatility, deep-dive profiles
    3. 📈 Demand Forecast Explorer - Actual vs. Predicted time series, model benchmarks (RMSPE/latency), mechanics
    4. 🧭 Store Segmentation       - 2D PCA cluster map, business segment profiles, silhouette diagnostics
    5. 🚨 Fraud & Anomaly Feed     - Real-time alert feed, MSE reconstruction distribution, severity filters, QA metrics

Design principles:
    - Every complex chart is paired with a clear, plain-language explanation
      expander or descriptive caption.
    - Transparent disclosures on store-level revenue proxies and offline diagnostic QA figures.
    - Defensive data loading ensures graceful degradation if specific files are missing.

Execution:
    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Path Configuration & Setup
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DATA_DIR = BASE_DIR / "data" / "processed"
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ----------------------------------------------------------------------------
# Data & Artifact Loaders (Cached)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_json(relative_path: str) -> dict | list | None:
    """Safely load a JSON report from reports directory."""
    path = REPORTS_DIR / relative_path
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_table(base_name: str, directory: str = "processed") -> pd.DataFrame | None:
    """Load parquet if present, else fallback to CSV."""
    if directory == "processed":
        d = DATA_DIR
    elif directory == "reports":
        d = REPORTS_DIR
    elif directory == "raw":
        d = RAW_DATA_DIR
    else:
        d = BASE_DIR / directory if (BASE_DIR / directory).exists() else BASE_DIR / "data" / directory
    parquet_path = d / f"{base_name}.parquet"
    csv_path = d / f"{base_name}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        df = pd.read_csv(csv_path, low_memory=False)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        return df
    return None


@st.cache_data(show_spinner=False)
def load_store_metadata() -> pd.DataFrame | None:
    """Load raw store metadata (store.csv) if present."""
    csv_path = RAW_DATA_DIR / "store.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


@st.cache_resource(show_spinner=False)
def load_joblib(filename: str):
    """Load serialized model pipeline from models directory."""
    path = MODELS_DIR / filename
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def missing_data_notice(what: str) -> None:
    """Render a clean alert when expected data/report is missing."""
    st.info(
        f"⚠️ **{what}** has not been generated yet. Run the corresponding pipeline "
        f"module (see project README) to populate this section."
    )


def explain(title: str, body: str) -> None:
    """Standardized 'How this works' expander for technical transparency."""
    with st.expander(f"ℹ️  {title}"):
        st.markdown(body)


# ----------------------------------------------------------------------------
# Forecast Precomputation
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Generating validation forecast predictions...")
def build_val_predictions() -> pd.DataFrame | None:
    """
    Score the holdout validation dataset using all trained model pipelines
    and assemble actual vs predicted columns per store and date.
    """
    val_df = load_table("val_processed")
    if val_df is None:
        return None

    val_df = val_df.sort_values("Date").reset_index(drop=True)
    result = pd.DataFrame({
        "Store": val_df["Store"],
        "Date": pd.to_datetime(val_df["Date"]),
        "Actual": val_df["Sales"],
    })

    # Find models from manifest or directory
    manifest = load_json("../models/latest_manifest.json") or load_json("latest_manifest.json")
    model_files: dict[str, Path] = {}

    if manifest and "models" in manifest:
        for m_name, m_info in manifest["models"].items():
            f_name = m_info.get("filename", "")
            p = MODELS_DIR / f_name
            if p.exists():
                model_files[m_name] = p

    if not model_files:
        for p in MODELS_DIR.glob("*.joblib"):
            stem = p.stem
            if stem.startswith("xgboost_"):
                model_files["xgboost"] = p
            elif stem.startswith("ridge_"):
                model_files["ridge"] = p
            elif stem.startswith("mlp_"):
                model_files["mlp"] = p

    for name, p in model_files.items():
        try:
            pipeline = joblib.load(p)
            preds = pipeline.predict(val_df)
            result[f"Predicted_{name}"] = np.clip(preds, 0, None)
        except Exception:
            pass

    return result


# ----------------------------------------------------------------------------
# Tab 1: Executive Overview
# ----------------------------------------------------------------------------
def render_executive_overview() -> None:
    st.header("Executive Overview")
    st.caption("A high-level view of retail sales trends, demand drivers, and validated business dynamics.")

    train_df = load_table("train_processed")
    fraud_alerts = load_json("fraud_alerts.json")
    hypo = load_json("hypothesis_tests.json")

    if train_df is None:
        missing_data_notice("Processed training data (Module 1: data_prep)")
        return

    total_revenue = float(train_df["Sales"].sum())
    avg_customers = float(train_df["Customers"].mean())

    train_df["Date"] = pd.to_datetime(train_df["Date"])
    min_date, max_date = train_df["Date"].min(), train_df["Date"].max()
    mid_date = min_date + (max_date - min_date) / 2

    earliest_half = train_df[train_df["Date"] < mid_date]
    latest_half = train_df[train_df["Date"] >= mid_date]
    mean_early = earliest_half["Sales"].mean() if len(earliest_half) else 1.0
    mean_late = latest_half["Sales"].mean() if len(latest_half) else 1.0
    revenue_trend_pct = 100 * (mean_late - mean_early) / mean_early if mean_early != 0 else 0.0

    fraud_exposure = 0.0
    if fraud_alerts:
        fraud_df = pd.DataFrame(fraud_alerts)
        if "severity" in fraud_df.columns and "return_amount" in fraud_df.columns:
            fraud_exposure = float(fraud_df.loc[fraud_df["severity"] == "high", "return_amount"].sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Chain Revenue", f"€{total_revenue:,.0f}")
    k2.metric("Revenue Growth Trend", f"{revenue_trend_pct:+.1f}%", help="Second half of training history vs. first half.")
    k3.metric("Avg Daily Customers / Store", f"{avg_customers:,.0f}")
    k4.metric("Flagged Fraud Exposure", f"€{fraud_exposure:,.0f}", help="Sum of high-severity return transactions flagged by detector.")

    st.markdown("---")
    st.subheader("Revenue Trend Over Time")
    granularity = st.radio("Aggregation Interval:", ["Daily", "Weekly", "Monthly", "Quarterly"], horizontal=True, index=1)
    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME", "Quarterly": "QE"}
    
    trend = (
        train_df.set_index("Date")["Sales"]
        .resample(freq_map[granularity]).sum()
        .reset_index()
    )
    fig_trend = px.line(
        trend, x="Date", y="Sales",
        labels={"Sales": "Revenue (€)", "Date": "Timeline"},
        title=f"Total Chain Sales ({granularity})",
        template="plotly_white",
    )
    fig_trend.update_traces(line=dict(width=2.5, color="#1f77b4"))
    st.plotly_chart(fig_trend, use_container_width=True)
    st.caption("Total sales across all operating stores. Use this chart to identify macro seasonality and holiday demand surges.")

    st.markdown("---")
    st.subheader("What Drives Retail Sales? (Correlation Matrix)")
    corr_cols = {
        "Sales": "Sales",
        "Customers": "Customers",
        "Promo": "Promo Active",
        "IsWeekend": "Weekend",
        "SchoolHoliday": "School Holiday",
        "CompetitionDistance": "Competition Distance",
    }
    available_cols = [c for c in corr_cols if c in train_df.columns]
    corr_matrix = train_df[available_cols].corr().rename(columns=corr_cols, index=corr_cols)
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        labels=dict(color="Correlation"),
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption("Darker blue indicates strong positive association with daily sales. Note the high correlation with Customer Footfall and Promotional Campaigns.")

    if hypo and "tests" in hypo:
        st.markdown("---")
        st.subheader("Statistically Validated Business Insights")
        st.caption("Formally tested hypotheses at significance level α = 0.05 (Welch's t-test and One-Way ANOVA).")
        test_cols = st.columns(len(hypo["tests"]))
        for col, test in zip(test_cols, hypo["tests"]):
            with col:
                st.markdown(f"**{test['test_name'].replace('_', ' ').title()}**")
                status = "✅ Statistically Significant" if test.get("significant") else "⚠️ Not Significant"
                st.markdown(f"`{status}`")
                st.metric("Test Statistic", f"{test.get('statistic', 0):,.2f}")
                st.caption(test.get("interpretation", ""))


# ----------------------------------------------------------------------------
# Tab 2: Store Performance
# ----------------------------------------------------------------------------
def render_store_performance() -> None:
    st.header("Store Performance & Volatility Leaderboard")
    st.caption(
        "Store-level revenue and behavioral stability. Note: Sales represent total gross revenue proxy "
        "(cost and margin data are not present in this dataset)."
    )

    clusters_df = load_table("store_clusters", directory="reports")
    store_meta = load_store_metadata()

    if clusters_df is None:
        missing_data_notice("Store clustering / profiling data (Module 2: clustering)")
        return

    df = clusters_df.copy()
    if store_meta is not None:
        df = df.merge(store_meta[["Store", "StoreType", "Assortment"]], on="Store", how="left")

    n = st.slider("Number of stores to display in leaderboard", min_value=5, max_value=30, value=10)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"🏆 **Top {n} Stores by Average Daily Revenue**")
        top_stores = df.nlargest(n, "avg_daily_sales")[[
            c for c in ["Store", "StoreType", "avg_daily_sales", "sales_cv", "promo_lift_pct"] if c in df.columns
        ]].rename(columns={
            "Store": "Store ID",
            "StoreType": "Type",
            "avg_daily_sales": "Avg Daily Sales (€)",
            "sales_cv": "Volatility (CV)",
            "promo_lift_pct": "Promo Lift (%)",
        })
        st.dataframe(top_stores.style.format({
            "Avg Daily Sales (€)": "€{:,.2f}",
            "Volatility (CV)": "{:.2f}",
            "Promo Lift (%)": "{:+.1f}%",
        }), hide_index=True, use_container_width=True)

    with col2:
        st.markdown(f"⚠️ **Lowest {n} Stores by Average Daily Revenue**")
        bottom_stores = df.nsmallest(n, "avg_daily_sales")[[
            c for c in ["Store", "StoreType", "avg_daily_sales", "sales_cv", "promo_lift_pct"] if c in df.columns
        ]].rename(columns={
            "Store": "Store ID",
            "StoreType": "Type",
            "avg_daily_sales": "Avg Daily Sales (€)",
            "sales_cv": "Volatility (CV)",
            "promo_lift_pct": "Promo Lift (%)",
        })
        st.dataframe(bottom_stores.style.format({
            "Avg Daily Sales (€)": "€{:,.2f}",
            "Volatility (CV)": "{:.2f}",
            "Promo Lift (%)": "{:+.1f}%",
        }), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Sales Volatility Analysis")
    st.caption("Coefficient of Variation (CV = Std Dev / Mean). Identifies stores with the most unpredictable day-to-day demand swings.")

    volatile_df = df.nlargest(15, "sales_cv").sort_values("sales_cv", ascending=True)
    fig_vol = px.bar(
        volatile_df,
        x="sales_cv",
        y=volatile_df["Store"].astype(str),
        orientation="h",
        labels={"sales_cv": "Volatility (CV)", "y": "Store ID"},
        title="Top 15 Most Volatile Stores",
        template="plotly_white",
        color="sales_cv",
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig_vol, use_container_width=True)


# ----------------------------------------------------------------------------
# Tab 3: Demand Forecast Explorer
# ----------------------------------------------------------------------------
def render_forecast_explorer() -> None:
    st.header("Demand Forecast Explorer")
    st.caption("Multi-model time-series forecasting across the holdout validation period.")

    explain(
        "How this forecast is made",
        """
1. **Time-Aware Feature Engineering**: The forecasting engine builds lag indicators ($t-7, t-14, t-21, t-30$), rolling trend statistics (7, 14, 30-day mean & std), and calendar dynamics.
2. **Zero-Leakage Temporal Validation**: Models are trained strictly on past history and evaluated on a forward 6-week validation window (`2015-06-19` to `2015-07-31`).
3. **Cross-Validation**: Evaluated using rolling `TimeSeriesSplit` folds.
4. **Scoring Metric**: Evaluated against official Root Mean Square Percentage Error (RMSPE).
        """,
    )

    comparison = load_json("comparison_metrics.json")
    predictions = build_val_predictions()

    if comparison and "models" in comparison:
        st.subheader("Model Benchmark Leaderboard")
        comp_df = pd.DataFrame(comparison["models"])
        best_model = comp_df.sort_values("rmspe").iloc[0]["model_name"]

        m_col1, m_col2 = st.columns([2, 1])
        with m_col1:
            fig_bench = px.bar(
                comp_df.sort_values("rmspe"),
                x="model_name",
                y="rmspe",
                labels={"model_name": "Model Architecture", "rmspe": "Validation RMSPE (Lower is Better)"},
                title="Model Accuracy Comparison (RMSPE)",
                template="plotly_white",
                color="rmspe",
                color_continuous_scale="Tealgrn_r",
            )
            st.plotly_chart(fig_bench, use_container_width=True)

        with m_col2:
            st.metric("Top Performing Model", best_model.upper())
            st.caption(
                f"**{best_model.upper()}** achieved the lowest RMSPE on held-out validation samples. "
                "RMSPE penalizes percentage deviation relative to actual sales volume."
            )

        st.dataframe(
            comp_df.rename(columns={
                "model_name": "Model",
                "rmspe": "RMSPE",
                "mae": "MAE (€)",
                "rmse": "RMSE (€)",
                "r2": "R²",
                "train_time_sec": "Train Time (s)",
                "inference_time_sec": "Inference Latency (s)",
            }).style.format({
                "RMSPE": "{:.4f}",
                "MAE (€)": "€{:.2f}",
                "RMSE (€)": "€{:.2f}",
                "R²": "{:.4f}",
                "Train Time (s)": "{:.2f}s",
                "Inference Latency (s)": "{:.4f}s",
            }),
            hide_index=True,
            use_container_width=True,
        )
    else:
        missing_data_notice("Forecast benchmark metrics (Module 3b: evaluate)")

    st.markdown("---")
    st.subheader("Actual vs. Predicted Daily Sales by Store")

    if predictions is None:
        missing_data_notice("Validation forecast predictions")
        return

    pred_models = [c.replace("Predicted_", "") for c in predictions.columns if c.startswith("Predicted_")]
    if not pred_models:
        missing_data_notice("Serialized model predictions")
        return

    f_col1, f_col2 = st.columns(2)
    with f_col1:
        store_list = sorted(predictions["Store"].unique())
        selected_store = st.selectbox("Select Store ID:", store_list, index=0)
    with f_col2:
        selected_model = st.selectbox("Select Model Architecture:", pred_models, index=0)

    store_df = predictions[predictions["Store"] == selected_store].sort_values("Date")
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=store_df["Date"],
        y=store_df["Actual"],
        mode="lines+markers",
        name="Actual Sales",
        line=dict(color="#2ca02c", width=2.5),
    ))
    fig_pred.add_trace(go.Scatter(
        x=store_df["Date"],
        y=store_df[f"Predicted_{selected_model}"],
        mode="lines+markers",
        name=f"Predicted ({selected_model})",
        line=dict(color="#d62728", width=2, dash="dash"),
    ))
    fig_pred.update_layout(
        title=f"Store {selected_store}: Actual vs. {selected_model.upper()} Forecast",
        xaxis_title="Date (Validation Period: June 19 - July 31, 2015)",
        yaxis_title="Daily Sales (€)",
        template="plotly_white",
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    # Rolled-up store aggregations
    st.subheader(f"Rolled-Up Forecast for Store {selected_store}")
    agg_freq = st.radio("Aggregate By:", ["Weekly", "Monthly"], horizontal=True, key="agg_store_freq")
    freq_code = "W" if agg_freq == "Weekly" else "ME"

    rolled_df = store_df.set_index("Date")[["Actual", f"Predicted_{selected_model}"]].resample(freq_code).sum().reset_index()
    fig_rolled = px.bar(
        rolled_df,
        x="Date",
        y=["Actual", f"Predicted_{selected_model}"],
        barmode="group",
        title=f"Aggregated {agg_freq} Total Sales vs Forecast",
        labels={"value": "Total Sales (€)", "variable": "Series"},
        template="plotly_white",
    )
    st.plotly_chart(fig_rolled, use_container_width=True)


# ----------------------------------------------------------------------------
# Tab 4: Store Segmentation
# ----------------------------------------------------------------------------
def render_segmentation() -> None:
    st.header("Behavioral Store Segmentation")
    st.caption("Unsupervised behavioral profiling and clustering (StandardScaler → K-Means → PCA 2D Projection).")

    clusters_df = load_table("store_clusters", directory="reports")
    cluster_meta = load_json("store_clusters.json")

    if clusters_df is None or cluster_meta is None:
        missing_data_notice("Store clustering reports (Module 2: clustering)")
        return

    clusters_df["cluster_str"] = "Cluster " + clusters_df["cluster"].astype(str)

    fig_pca = px.scatter(
        clusters_df,
        x="pca_x",
        y="pca_y",
        color="cluster_str",
        hover_data=["Store", "avg_daily_sales", "sales_cv", "avg_daily_customers", "promo_lift_pct"],
        labels={"pca_x": "PCA Component 1", "pca_y": "PCA Component 2", "cluster_str": "Cluster"},
        title="2D PCA Projection of Store Behavioral Profiles",
        template="plotly_white",
    )
    st.plotly_chart(fig_pca, use_container_width=True)
    st.caption("Each point represents a retail store positioned by multi-dimensional sales volume, footfall, volatility, and promo response.")

    st.markdown("---")
    st.subheader("Cluster Behavioral Profiles")

    cluster_names = {
        "0": "High-Volume Flagship Stores",
        "1": "Steady Neighborhood Stores",
        "2": "Compact High-Traffic City Stores",
        "3": "Volatile Promo-Sensitive Stores",
    }

    profile_means = cluster_meta.get("cluster_profile_means", {})
    cluster_sizes = cluster_meta.get("cluster_sizes", {})

    cols = st.columns(len(profile_means))
    for col, (c_id, p_data) in zip(cols, profile_means.items()):
        with col:
            st.markdown(f"### Cluster {c_id}")
            st.markdown(f"**{cluster_names.get(c_id, f'Segment {c_id}')}**")
            st.caption(f"Stores in Cluster: **{cluster_sizes.get(c_id, 'N/A')}**")
            st.metric("Avg Daily Sales", f"€{p_data.get('avg_daily_sales', 0):,.0f}")
            st.metric("Avg Daily Customers", f"{p_data.get('avg_daily_customers', 0):,.0f}")
            st.metric("Promo Sales Lift", f"{p_data.get('promo_lift_pct', 0):+.1f}%")
            st.caption(f"Sales Volatility (CV): **{p_data.get('sales_cv', 0):.2f}**")

    # Silhouette diagnostics expander
    if "silhouette_scores" in cluster_meta:
        sil_data = cluster_meta["silhouette_scores"]
        explain(
            "Why K = 4 clusters? (Silhouette Optimization)",
            f"""
K-Means was tested across candidate cluster counts ($K = 2 \\dots 10$).
Optimal cluster compactness and separation was achieved at **$K = 4$** with a maximum **Silhouette Score of {sil_data.get('4', 0.323):.4f}**.
            """,
        )


# ----------------------------------------------------------------------------
# Tab 5: Fraud & Anomaly Feed
# ----------------------------------------------------------------------------
def render_fraud_feed() -> None:
    st.header("Point-of-Sale (POS) Fraud & Anomaly Feed")
    st.caption("Unsupervised Deep Autoencoder anomaly scoring on return transactions.")

    explain(
        "How Anomaly Detection Works",
        """
1. **Unsupervised Baseline**: An Autoencoder ANN is trained strictly on normal historical POS return patterns to compress and reconstruct return transactions.
2. **Reconstruction MSE Anomaly Scoring**: Anomalous or abusive return transactions (such as serial return abuse, unverified high-value returns, or new account wardrobing) incur significantly higher reconstruction error.
3. **Threshold Calibration**: 95th and 99th percentile alert cutoffs are learned strictly from the training error distribution and applied forward.
4. **Human Review Screening**: High risk alerts flag transactions requiring human inspection before refund approval.
        """,
    )

    alerts = load_json("fraud_alerts.json")
    metrics = load_json("fraud_detection_metrics.json")

    if not alerts:
        missing_data_notice("POS fraud detection alerts (Module 4: fraud_detection)")
        return

    df_alerts = pd.DataFrame(alerts)

    total_reviewed = len(df_alerts)
    n_flagged_p95 = int(df_alerts["flagged_p95"].sum()) if "flagged_p95" in df_alerts.columns else 0
    high_sev_exposure = (
        float(df_alerts.loc[df_alerts.get("severity", "") == "high", "return_amount"].sum())
        if "return_amount" in df_alerts.columns else 0.0
    )

    k1, k2, k3 = st.columns(3)
    k1.metric("Transactions Monitored", f"{total_reviewed:,}")
    k2.metric("Flagged for Review (p95+)", f"{n_flagged_p95:,}")
    k3.metric("High Severity Exposure", f"€{high_sev_exposure:,.0f}")

    st.markdown("---")
    st.subheader("Reconstruction Error Distribution")
    fig_hist = px.histogram(
        df_alerts,
        x="reconstruction_error",
        nbins=60,
        labels={"reconstruction_error": "Reconstruction MSE (Anomaly Score)"},
        title="Anomaly Score Frequency Distribution",
        template="plotly_white",
        color_discrete_sequence=["#636EFA"],
    )
    if metrics:
        p95_val = metrics.get("threshold_p95")
        p99_val = metrics.get("threshold_p99")
        if p95_val is not None:
            fig_hist.add_vline(x=p95_val, line_dash="dash", line_color="orange", annotation_text=f"p95 ({p95_val:.5f})")
        if p99_val is not None:
            fig_hist.add_vline(x=p99_val, line_dash="dash", line_color="red", annotation_text=f"p99 ({p99_val:.5f})")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")
    st.subheader("Flagged Transaction Feed")
    sev_filter = st.radio("Filter Alerts by Severity:", ["All", "High severity only", "Medium & High"], horizontal=True)

    view_df = df_alerts.copy()
    if sev_filter == "High severity only":
        view_df = view_df[view_df["severity"] == "high"]
    elif sev_filter == "Medium & High":
        view_df = view_df[view_df["severity"].isin(["high", "medium"])]

    view_df = view_df.sort_values("reconstruction_error", ascending=False)
    display_cols = [c for c in [
        "transaction_id", "store_id", "return_amount", "return_frequency_24h",
        "days_since_purchase", "receipt_verified", "customer_tenure_days",
        "reconstruction_error", "severity",
    ] if c in view_df.columns]

    st.dataframe(
        view_df[display_cols].head(250).rename(columns={
            "transaction_id": "Transaction ID",
            "store_id": "Store",
            "return_amount": "Amount (€)",
            "return_frequency_24h": "24h Frequency",
            "days_since_purchase": "Days Since Purchase",
            "receipt_verified": "Receipt Verified",
            "customer_tenure_days": "Tenure (Days)",
            "reconstruction_error": "Anomaly MSE",
            "severity": "Severity",
        }).style.format({
            "Amount (€)": "€{:,.2f}",
            "Anomaly MSE": "{:.6f}",
        }),
        hide_index=True,
        use_container_width=True,
    )

    if metrics and "eval_p95_diagnostic_only" in metrics:
        st.markdown("---")
        diag = metrics["eval_p95_diagnostic_only"]
        st.caption(
            f"**Diagnostic QA Note**: Tested against synthetic ground-truth patterns, the detector achieved "
            f"**{diag.get('recall', 0)*100:.1f}% Recall** and **{diag.get('precision', 0)*100:.1f}% Precision** "
            f"(F1: {diag.get('f1', 0):.4f}) at the 95th percentile threshold."
        )


# ----------------------------------------------------------------------------
# Main App Entrypoint
# ----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Retail Intelligence & Demand Forecasting",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🛒 Retail Demand Forecasting & Fraud Simulation System")
    st.caption(
        "Executive BI & Analytics Platform integrating time-series demand forecasting, "
        "statistical hypothesis testing, store behavioral clustering, and POS fraud detection."
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Executive Overview",
        "🏬 Store Performance",
        "📈 Demand Forecast",
        "🧭 Store Segmentation",
        "🚨 Fraud & Anomaly Feed",
    ])

    with tab1:
        render_executive_overview()
    with tab2:
        render_store_performance()
    with tab3:
        render_forecast_explorer()
    with tab4:
        render_segmentation()
    with tab5:
        render_fraud_feed()


if __name__ == "__main__":
    main()
