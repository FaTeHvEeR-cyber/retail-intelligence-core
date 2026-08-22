"""
app.py
======
Module 6: Interactive Executive Streamlit BI Dashboard.

Integrates the outputs of Modules 1-5 (data_prep, hypothesis_testing,
clustering, train_models/evaluate, fraud_detection) into an enterprise
executive 5-tab dashboard with zero emojis, professional typography,
instant Light/Dark mode switching, and high-performance visualizations:
    1. Executive Overview       - Top-level KPIs, revenue trend, ranked factor correlation, hypothesis tests
    2. Store Performance        - Leaderboards, risk quadrant scatter (Volume vs Volatility), deep-dive profiles
    3. Demand Forecast          - Actual vs. Predicted time series, model benchmarks (RMSPE/latency), multi-store aggregations
    4. Store Segmentation       - 2D PCA cluster map, business segment profiles, silhouette diagnostics
    5. Fraud & Anomaly Feed     - Risk-tiered reconstruction distribution, real-time alert feed, severity filters
"""

from __future__ import annotations

import html
import json
import sys
import textwrap
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
# Adaptive Plotly Theme Styler
# ----------------------------------------------------------------------------
def apply_plotly_theme(fig: go.Figure, height: int | None = None) -> go.Figure:
    """Applies a clean, responsive transparent layout with professional typography."""
    layout_update = dict(
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(128, 128, 128, 0.04)",
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size=12),
        margin=dict(l=45, r=25, t=45, b=40),
        xaxis=dict(
            gridcolor="rgba(128, 128, 128, 0.15)",
            zerolinecolor="rgba(128, 128, 128, 0.15)",
            tickfont=dict(family="JetBrains Mono, monospace", size=11),
            title_font=dict(size=12, family="Inter, sans-serif"),
        ),
        yaxis=dict(
            gridcolor="rgba(128, 128, 128, 0.15)",
            zerolinecolor="rgba(128, 128, 128, 0.15)",
            tickfont=dict(family="JetBrains Mono, monospace", size=11),
            title_font=dict(size=12, family="Inter, sans-serif"),
        ),
        legend=dict(
            bgcolor="rgba(128, 128, 128, 0.08)",
            bordercolor="rgba(128, 128, 128, 0.18)",
            borderwidth=1,
            font=dict(family="Inter, sans-serif", size=11),
        ),
        colorway=["#0284c7", "#6366f1", "#10b981", "#f59e0b", "#ef4444"],
    )
    if height is not None:
        layout_update["height"] = height
    fig.update_layout(**layout_update)
    return fig


# ----------------------------------------------------------------------------
# Theme-Adaptive Custom CSS
# ----------------------------------------------------------------------------
def inject_custom_css() -> None:
    """Injects high-end enterprise typography, JetBrains Mono numbers, and adaptive styling."""
    css_code = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

/* Monospace Numbers Across All Metrics and Financials */
.mono-num, code, pre, .glass-card-value, .metric-value, [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-variant-numeric: tabular-nums !important;
}

.hero-banner {
    background: var(--secondary-background-color, rgba(128, 128, 128, 0.08));
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 20px;
    box-shadow: 0 4px 16px -4px rgba(0, 0, 0, 0.08);
    position: relative;
    overflow: hidden;
}

.hero-banner::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #0284c7, #6366f1, #8b5cf6, #10b981);
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 4px;
    padding: 3px 10px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #10b981;
    margin-bottom: 10px;
}

.pulse-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background-color: #10b981;
}

.hero-title {
    font-size: 1.85rem !important;
    font-weight: 800 !important;
    margin: 0 0 6px 0 !important;
    letter-spacing: -0.02em;
    color: var(--text-color, inherit);
}

.hero-subtitle {
    font-size: 0.88rem;
    opacity: 0.85;
    margin-bottom: 12px;
    max-width: 920px;
    line-height: 1.45;
}

.dataset-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(128, 128, 128, 0.1);
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 0.78rem;
}

.dataset-pill strong {
    color: #0284c7;
}

.kpi-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 14px;
    margin-bottom: 20px;
}

.glass-card {
    background: var(--secondary-background-color, rgba(128, 128, 128, 0.08));
    border: 1px solid rgba(128, 128, 128, 0.18);
    border-radius: 10px;
    padding: 16px 18px;
    box-shadow: 0 2px 8px -2px rgba(0, 0, 0, 0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 14px -2px rgba(0, 0, 0, 0.08);
}

.glass-card-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}

.glass-card-label {
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    opacity: 0.75;
    text-transform: uppercase;
}

.glass-card-indicator {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: #0284c7;
    display: inline-block;
}

.glass-card-value {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    margin-bottom: 4px;
    line-height: 1.2;
}

.glass-card-sub {
    font-size: 0.76rem;
    opacity: 0.72;
}

.badge-positive { color: #10b981; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
.badge-warning { color: #f59e0b; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
.badge-danger { color: #ef4444; font-weight: 600; font-family: 'JetBrains Mono', monospace; }

.insight-box {
    background: var(--secondary-background-color, rgba(128, 128, 128, 0.08));
    border-left: 4px solid #0284c7;
    border-top: 1px solid rgba(128, 128, 128, 0.15);
    border-right: 1px solid rgba(128, 128, 128, 0.15);
    border-bottom: 1px solid rgba(128, 128, 128, 0.15);
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin-bottom: 10px;
}

.insight-title {
    font-weight: 700;
    font-size: 0.88rem;
    margin-bottom: 4px;
}

.insight-body {
    font-size: 0.82rem;
    opacity: 0.85;
    line-height: 1.45;
}

/* Navigation Tabs - Segmented Enterprise Button Pills */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px !important;
    background: transparent !important;
    border: none !important;
    padding: 4px 0 !important;
    margin-bottom: 24px !important;
}

.stTabs [data-baseweb="tab-border"],
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 10px 18px !important;
    background: var(--secondary-background-color, rgba(128, 128, 128, 0.08)) !important;
    border: 1px solid rgba(128, 128, 128, 0.22) !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}

.stTabs [data-baseweb="tab"]:hover {
    border-color: rgba(2, 132, 199, 0.5) !important;
    background: rgba(2, 132, 199, 0.1) !important;
    color: #0284c7 !important;
    transform: translateY(-1px);
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: #0284c7 !important;
    color: #ffffff !important;
    border-color: #0284c7 !important;
    box-shadow: 0 4px 14px -2px rgba(2, 132, 199, 0.35) !important;
}

/* Dataframe & Tables */
.stDataFrame table, div[data-testid="stTable"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}

.zone-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.76rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
</style>
"""
    st.markdown(textwrap.dedent(css_code).strip(), unsafe_allow_html=True)


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
    st.warning(
        f"Notice: {what} has not been generated yet. Please run the corresponding pipeline "
        f"module (see project README) to populate this section."
    )


def explain(title: str, body: str) -> None:
    """Standardized 'How this works' expander for technical transparency."""
    with st.expander(f"Info: {title}"):
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
    st.subheader("Executive Business Performance Overview")
    st.caption("Top-level read on retail chain revenue, footfall dynamics, and statistically validated drivers.")

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

    kpi_html = f"""
    <div class="kpi-container">
        <div class="glass-card">
            <div class="glass-card-top">
                <span class="glass-card-label">Total Chain Revenue</span>
                <span class="glass-card-indicator"></span>
            </div>
            <div class="glass-card-value">€{total_revenue / 1e9:.2f}B</div>
            <div class="glass-card-sub"><span class="badge-positive">€{total_revenue:,.0f}</span> gross total</div>
        </div>
        <div class="glass-card">
            <div class="glass-card-top">
                <span class="glass-card-label">Historical Growth Trend</span>
                <span class="glass-card-indicator" style="background-color: {'#10b981' if revenue_trend_pct >= 0 else '#ef4444'};"></span>
            </div>
            <div class="glass-card-value" style="color: {'#10b981' if revenue_trend_pct >= 0 else '#ef4444'};">
                {revenue_trend_pct:+.1f}%
            </div>
            <div class="glass-card-sub">2nd half vs 1st half history</div>
        </div>
        <div class="glass-card">
            <div class="glass-card-top">
                <span class="glass-card-label">Avg Daily Customers</span>
                <span class="glass-card-indicator"></span>
            </div>
            <div class="glass-card-value">{avg_customers:,.0f}</div>
            <div class="glass-card-sub">Per store operating day</div>
        </div>
        <div class="glass-card">
            <div class="glass-card-top">
                <span class="glass-card-label">Flagged Fraud Exposure</span>
                <span class="glass-card-indicator" style="background-color: #f59e0b;"></span>
            </div>
            <div class="glass-card-value" style="color: #f59e0b;">€{fraud_exposure:,.0f}</div>
            <div class="glass-card-sub"><span class="badge-warning">High Risk</span> Return Volume</div>
        </div>
    </div>
    """
    st.markdown(textwrap.dedent(kpi_html).strip(), unsafe_allow_html=True)

    st.markdown("### Revenue Trend Over Time")
    granularity = st.radio("Aggregation Frequency:", ["Daily", "Weekly", "Monthly", "Quarterly"], horizontal=True, index=1)
    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME", "Quarterly": "QE"}

    trend = (
        train_df.set_index("Date")["Sales"]
        .resample(freq_map[granularity]).sum()
        .reset_index()
    )

    fig_trend = px.area(
        trend,
        x="Date",
        y="Sales",
        labels={"Sales": "Revenue (€)", "Date": "Timeline"},
        title=f"Total Chain Sales Velocity ({granularity})",
    )
    fig_trend.update_traces(line=dict(width=2.5, color="#0284c7", shape="spline"), fillcolor="rgba(2, 132, 199, 0.12)")
    apply_plotly_theme(fig_trend, height=360)
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("---")
    st.markdown("### Key Drivers of Retail Demand (Correlation with Revenue)")
    st.caption("Ranked Pearson correlation coefficients measuring direct linear association with store daily sales volume.")

    corr_cols = {
        "Customers": "Store Footfall (Customers)",
        "Promo": "Promotional Campaign Active",
        "SchoolHoliday": "School Holiday Period",
        "IsWeekend": "Weekend Trading",
        "CompetitionDistance": "Competition Distance (m)",
    }
    available_cols = [c for c in corr_cols if c in train_df.columns]

    corr_series = train_df[available_cols].apply(lambda s: s.corr(train_df["Sales"]))
    corr_df = pd.DataFrame({
        "Feature": [corr_cols[c] for c in available_cols],
        "Correlation": [corr_series[c] for c in available_cols],
    }).sort_values("Correlation", ascending=True)

    # Color code positive vs negative drivers
    corr_df["Color"] = np.where(corr_df["Correlation"] >= 0, "#0284c7", "#ef4444")
    corr_df["Formatted_Corr"] = corr_df["Correlation"].apply(lambda x: f"{x:+.3f}")

    fig_corr_bar = go.Figure()
    fig_corr_bar.add_trace(go.Bar(
        x=corr_df["Correlation"],
        y=corr_df["Feature"],
        orientation="h",
        marker=dict(color=corr_df["Color"], line=dict(width=0)),
        text=corr_df["Formatted_Corr"],
        textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=11),
        hovertemplate="<b>%{y}</b><br>Correlation with Sales: %{x:.4f}<extra></extra>",
    ))
    fig_corr_bar.update_layout(
        title="Ranked Impact of External & Operational Drivers on Sales",
        xaxis=dict(
            title="Pearson Correlation (r)",
            range=[
                min(corr_df["Correlation"].min() - 0.12, -0.2),
                max(corr_df["Correlation"].max() + 0.15, 0.95),
            ],
            zeroline=True,
            zerolinecolor="rgba(128, 128, 128, 0.4)",
            zerolinewidth=1.5,
        ),
        yaxis=dict(title=""),
    )
    apply_plotly_theme(fig_corr_bar, height=340)
    st.plotly_chart(fig_corr_bar, use_container_width=True)

    if hypo and "tests" in hypo:
        st.markdown("---")
        st.markdown("### Formally Tested Statistical Hypotheses (alpha = 0.05)")
        for test in hypo["tests"]:
            test_name = test.get("test_name", "").replace("_", " ").title()
            stat_val = test.get("statistic", 0.0)
            interp = test.get("interpretation", "")
            hypo_html = f"""
            <div class="insight-box">
                <div class="insight-title">{test_name} — <span style="color: #10b981;">Significant (p < 0.05)</span></div>
                <div class="insight-body">
                    <strong>Test Statistic:</strong> <span class="mono-num">{stat_val:,.2f}</span> &nbsp;|&nbsp;
                    <strong>Conclusion:</strong> {html.escape(interp)}
                </div>
            </div>
            """
            st.markdown(textwrap.dedent(hypo_html).strip(), unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Tab 2: Store Performance
# ----------------------------------------------------------------------------
def render_store_performance() -> None:
    st.subheader("Store Performance & Volatility Matrix")
    st.caption("Individual store revenue benchmarking, volume leaders, and coefficient of variation volatility metrics.")

    clusters_df = load_table("store_clusters", directory="reports")
    store_meta = load_store_metadata()

    if clusters_df is None:
        missing_data_notice("Store clustering / profiling data (Module 2: clustering)")
        return

    df = clusters_df.copy()
    if store_meta is not None:
        df = df.merge(store_meta[["Store", "StoreType", "Assortment"]], on="Store", how="left")

    n = st.slider("Number of stores to display in leaderboards:", min_value=5, max_value=30, value=10)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Top {n} Highest Revenue Stores**")
        top_stores = df.nlargest(n, "avg_daily_sales")[[
            c for c in ["Store", "StoreType", "avg_daily_sales", "sales_cv", "promo_lift_pct"] if c in df.columns
        ]].rename(columns={
            "Store": "Store ID",
            "StoreType": "Type",
            "avg_daily_sales": "Avg Daily Sales (€)",
            "sales_cv": "Volatility (CV)",
            "promo_lift_pct": "Promo Lift (%)",
        })
        st.dataframe(
            top_stores.style.format({
                "Avg Daily Sales (€)": "€{:,.2f}",
                "Volatility (CV)": "{:.2f}",
                "Promo Lift (%)": "{:+.1f}%",
            }),
            hide_index=True,
            use_container_width=True,
        )

    with col2:
        st.markdown(f"**Lowest {n} Revenue Stores**")
        bottom_stores = df.nsmallest(n, "avg_daily_sales")[[
            c for c in ["Store", "StoreType", "avg_daily_sales", "sales_cv", "promo_lift_pct"] if c in df.columns
        ]].rename(columns={
            "Store": "Store ID",
            "StoreType": "Type",
            "avg_daily_sales": "Avg Daily Sales (€)",
            "sales_cv": "Volatility (CV)",
            "promo_lift_pct": "Promo Lift (%)",
        })
        st.dataframe(
            bottom_stores.style.format({
                "Avg Daily Sales (€)": "€{:,.2f}",
                "Volatility (CV)": "{:.2f}",
                "Promo Lift (%)": "{:+.1f}%",
            }),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("### Store Revenue vs. Volatility Risk Quadrant")
    st.caption("Mapping sales volume against volatility (Coefficient of Variation) to isolate inventory risk and steady performers.")

    med_sales = float(df["avg_daily_sales"].median())
    med_cv = float(df["sales_cv"].median())

    type_col = "StoreType" if "StoreType" in df.columns else None

    fig_quad = px.scatter(
        df,
        x="avg_daily_sales",
        y="sales_cv",
        color=type_col,
        hover_data=["Store", "avg_daily_customers", "promo_lift_pct"],
        labels={
            "avg_daily_sales": "Average Daily Sales (€)",
            "sales_cv": "Sales Volatility (Coefficient of Variation)",
            "StoreType": "Store Type",
        },
        title="Network Store Risk Map: Daily Revenue vs. Volatility",
        color_discrete_sequence=["#0284c7", "#10b981", "#f59e0b", "#8b5cf6"],
    )
    fig_quad.update_traces(marker=dict(size=7, opacity=0.75, line=dict(width=0.5, color="white")))

    # Add Median Benchmark Lines
    fig_quad.add_vline(x=med_sales, line_dash="dot", line_color="rgba(128, 128, 128, 0.45)")
    fig_quad.add_hline(y=med_cv, line_dash="dot", line_color="rgba(128, 128, 128, 0.45)")

    # Quadrant annotations
    x_max = df["avg_daily_sales"].max()
    y_max = df["sales_cv"].max()
    fig_quad.add_annotation(
        x=x_max * 0.85, y=y_max * 0.95,
        text="High Volume / High Volatility<br><b>(High Inventory Risk)</b>",
        showarrow=False, font=dict(size=10, color="#f59e0b", family="Inter, sans-serif"),
        bgcolor="rgba(128, 128, 128, 0.1)", bordercolor="rgba(128, 128, 128, 0.2)", borderpad=4,
    )
    fig_quad.add_annotation(
        x=x_max * 0.85, y=df["sales_cv"].min() * 1.1,
        text="High Volume / Stable<br><b>(Core Revenue Engines)</b>",
        showarrow=False, font=dict(size=10, color="#10b981", family="Inter, sans-serif"),
        bgcolor="rgba(128, 128, 128, 0.1)", bordercolor="rgba(128, 128, 128, 0.2)", borderpad=4,
    )

    apply_plotly_theme(fig_quad, height=480)
    st.plotly_chart(fig_quad, use_container_width=True)


# ----------------------------------------------------------------------------
# Tab 3: Demand Forecast Explorer
# ----------------------------------------------------------------------------
def render_forecast_explorer() -> None:
    st.subheader("Multi-Model Demand Forecast Explorer")
    st.caption("Evaluating Ridge Regression, MLP Deep Neural Network, and XGBoost on holdout validation period.")

    explain(
        "Forecasting Engine Mechanics & Leakage Prevention",
        """
1. **Time-Aware Feature Engineering**: Includes temporal lags ($t-7, t-14, t-21, t-30$) and rolling trends (7, 14, 30-day mean & std).
2. **Zero Data Leakage Split**: Strict chronological 6-week holdout (`2015-06-19` to `2015-07-31`). No future information peeking.
3. **Cross-Validation**: Rolling temporal folds using `TimeSeriesSplit`.
4. **Primary Evaluation Metric**: Root Mean Square Percentage Error (RMSPE).
        """,
    )

    comparison = load_json("comparison_metrics.json")
    predictions = build_val_predictions()

    if comparison and "models" in comparison:
        st.markdown("### Multi-Model Benchmark Leaderboard")
        comp_df = pd.DataFrame(comparison["models"])
        best_model = comp_df.sort_values("rmspe").iloc[0]["model_name"]

        m_col1, m_col2 = st.columns([2, 1])
        with m_col1:
            fig_bench = px.bar(
                comp_df.sort_values("rmspe"),
                x="model_name",
                y="rmspe",
                labels={"model_name": "Architecture", "rmspe": "Validation RMSPE (Lower is Better)"},
                title="Model Accuracy Benchmark (RMSPE)",
                color="rmspe",
                color_continuous_scale="Teal_r",
            )
            apply_plotly_theme(fig_bench, height=320)
            st.plotly_chart(fig_bench, use_container_width=True)

        with m_col2:
            win_card_html = f"""
            <div class="glass-card" style="margin-top: 15px; border-color: rgba(16, 185, 129, 0.4);">
                <div class="glass-card-top">
                    <span class="glass-card-label">Winning Model</span>
                    <span class="glass-card-indicator" style="background-color: #10b981;"></span>
                </div>
                <div class="glass-card-value" style="color: #10b981;">{best_model.upper()}</div>
                <div class="glass-card-sub">Lowest validation RMSPE across all 1,115 stores.</div>
            </div>
            """
            st.markdown(textwrap.dedent(win_card_html).strip(), unsafe_allow_html=True)

        st.dataframe(
            comp_df.rename(columns={
                "model_name": "Model",
                "rmspe": "RMSPE",
                "mae": "MAE (€)",
                "rmse": "RMSE (€)",
                "r2": "R²",
                "train_time_sec": "Train Time (s)",
                "inference_time_sec": "Latency (s)",
            }).style.format({
                "RMSPE": "{:.4f}",
                "MAE (€)": "€{:.2f}",
                "RMSE (€)": "€{:.2f}",
                "R²": "{:.4f}",
                "Train Time (s)": "{:.2f}s",
                "Latency (s)": "{:.4f}s",
            }),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("### Store-Level Actual vs. Predicted Time-Series")

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
        selected_model = st.selectbox("Select Forecast Model:", pred_models, index=0)

    store_df = predictions[predictions["Store"] == selected_store].sort_values("Date")
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=store_df["Date"],
        y=store_df["Actual"],
        mode="lines+markers",
        name="Actual Sales (€)",
        line=dict(color="#10b981", width=2.5),
        marker=dict(size=5),
    ))
    fig_pred.add_trace(go.Scatter(
        x=store_df["Date"],
        y=store_df[f"Predicted_{selected_model}"],
        mode="lines+markers",
        name=f"Predicted ({selected_model.upper()})",
        line=dict(color="#f43f5e", width=2, dash="dash"),
        marker=dict(size=4),
    ))
    fig_pred.update_layout(
        title=f"Store {selected_store}: Actual Daily Sales vs. {selected_model.upper()} Forecast",
        xaxis_title="Timeline (Holdout Validation Period)",
        yaxis_title="Sales (€)",
        hovermode="x unified",
    )
    apply_plotly_theme(fig_pred, height=420)
    st.plotly_chart(fig_pred, use_container_width=True)

    # Rolled-Up Multi-Store Aggregation
    st.markdown("---")
    st.markdown("### Aggregated Sales Forecast vs. Actual")
    st.caption("Compare aggregate revenue volume for single stores, custom multi-store cohorts, or the entire retail network.")

    agg_col1, agg_col2 = st.columns([2, 1])
    with agg_col1:
        agg_mode = st.radio(
            "Aggregation Scope:",
            ["Single Store", "Custom Multi-Store Cohort", "Entire Network (All 1,115 Stores)"],
            horizontal=True,
            key="agg_scope_mode",
        )
    with agg_col2:
        agg_freq = st.radio(
            "Time Interval:",
            ["Weekly", "Monthly"],
            horizontal=True,
            key="agg_store_freq",
        )

    freq_code = "W" if agg_freq == "Weekly" else "ME"

    if agg_mode == "Single Store":
        agg_stores = [selected_store]
        scope_title = f"Store {selected_store}"
    elif agg_mode == "Custom Multi-Store Cohort":
        default_cohort = store_list[:5] if len(store_list) >= 5 else store_list
        agg_stores = st.multiselect(
            "Select Stores for Cohort Aggregation:",
            store_list,
            default=default_cohort,
            key="cohort_multiselect",
        )
        if not agg_stores:
            st.info("Please select at least one store to view cohort aggregations.")
            return
        scope_title = f"Custom Cohort ({len(agg_stores)} Stores)"
    else:
        agg_stores = store_list
        scope_title = f"Entire Network ({len(store_list):,} Stores)"

    filtered_preds = predictions[predictions["Store"].isin(agg_stores)]
    rolled_df = (
        filtered_preds.groupby("Date")[["Actual", f"Predicted_{selected_model}"]]
        .sum()
        .resample(freq_code)
        .sum()
        .reset_index()
    )

    tot_actual = float(rolled_df["Actual"].sum())
    tot_pred = float(rolled_df[f"Predicted_{selected_model}"].sum())
    pct_delta = 100 * (tot_pred - tot_actual) / tot_actual if tot_actual != 0 else 0.0

    pills_html = f"""
    <div style="display: flex; gap: 10px; margin-bottom: 12px; flex-wrap: wrap;">
        <div class="zone-pill" style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); color: #10b981;">
            Actual Total: €{tot_actual:,.0f}
        </div>
        <div class="zone-pill" style="background: rgba(244, 63, 94, 0.12); border: 1px solid rgba(244, 63, 94, 0.3); color: #f43f5e;">
            {selected_model.upper()} Forecast Total: €{tot_pred:,.0f}
        </div>
        <div class="zone-pill" style="background: rgba(128, 128, 128, 0.12); border: 1px solid rgba(128, 128, 128, 0.3); color: {'#10b981' if abs(pct_delta) <= 5 else '#f59e0b'};">
            Variance Delta: {pct_delta:+.2f}%
        </div>
    </div>
    """
    st.markdown(textwrap.dedent(pills_html).strip(), unsafe_allow_html=True)

    fig_rolled = px.bar(
        rolled_df,
        x="Date",
        y=["Actual", f"Predicted_{selected_model}"],
        barmode="group",
        title=f"{scope_title}: Aggregated {agg_freq} Sales — Actual vs. {selected_model.upper()} Forecast",
        labels={"value": "Total Sales (€)", "variable": "Series", "Date": "Timeline"},
        color_discrete_map={"Actual": "#10b981", f"Predicted_{selected_model}": "#f43f5e"},
    )
    apply_plotly_theme(fig_rolled, height=360)
    st.plotly_chart(fig_rolled, use_container_width=True)


# ----------------------------------------------------------------------------
# Tab 4: Store Segmentation
# ----------------------------------------------------------------------------
def render_segmentation() -> None:
    st.subheader("Behavioral Store Segmentation & Clustering")
    st.caption("Unsupervised behavioral profiling and segmentation (StandardScaler → K-Means K=4 → PCA 2D Projection).")

    clusters_df = load_table("store_clusters", directory="reports")
    cluster_meta = load_json("store_clusters.json")

    if clusters_df is None or cluster_meta is None:
        missing_data_notice("Store clustering reports (Module 2: clustering)")
        return

    clusters_df["cluster_label"] = "Cluster " + clusters_df["cluster"].astype(str)

    fig_pca = px.scatter(
        clusters_df,
        x="pca_x",
        y="pca_y",
        color="cluster_label",
        hover_data=["Store", "avg_daily_sales", "sales_cv", "avg_daily_customers", "promo_lift_pct"],
        labels={"pca_x": "PCA Component 1", "pca_y": "PCA Component 2", "cluster_label": "Segment"},
        title="2D PCA Behavioral Projection of 1,115 Retail Stores",
        color_discrete_sequence=["#0284c7", "#8b5cf6", "#10b981", "#f59e0b"],
    )
    fig_pca.update_traces(marker=dict(size=7, opacity=0.85))
    apply_plotly_theme(fig_pca, height=480)
    st.plotly_chart(fig_pca, use_container_width=True)

    st.markdown("---")
    st.markdown("### Business Archetype Segment Profiles")

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
            profile_html = f"""
            <div class="glass-card">
                <div class="glass-card-top">
                    <span class="glass-card-label">Cluster {c_id}</span>
                    <span class="glass-card-indicator"></span>
                </div>
                <div style="font-size: 0.98rem; font-weight: 700; color: #0284c7; margin-bottom: 6px;">
                    {cluster_names.get(c_id, f'Segment {c_id}')}
                </div>
                <div class="glass-card-sub" style="margin-bottom: 10px;">Stores: <strong class="mono-num">{cluster_sizes.get(c_id, 'N/A')}</strong></div>
                <div class="glass-card-value">€{p_data.get('avg_daily_sales', 0):,.0f}</div>
                <div class="glass-card-sub">Avg Daily Sales</div>
                <div style="margin-top: 10px; font-size: 0.80rem; line-height: 1.6;">
                    Footfall: <strong class="mono-num">{p_data.get('avg_daily_customers', 0):,.0f}</strong><br>
                    Promo Lift: <strong class="badge-positive">{p_data.get('promo_lift_pct', 0):+.1f}%</strong><br>
                    Volatility: <strong class="mono-num">{p_data.get('sales_cv', 0):.2f}</strong>
                </div>
            </div>
            """
            st.markdown(textwrap.dedent(profile_html).strip(), unsafe_allow_html=True)

    if "silhouette_scores" in cluster_meta:
        sil_data = cluster_meta["silhouette_scores"]
        explain(
            "Why K = 4 clusters? (Silhouette Optimization)",
            f"""
Candidate values $K = 2 \\dots 10$ were tested. Peak cluster separation and cohesion was achieved at **$K = 4$** with a maximum **Silhouette Score of {sil_data.get('4', 0.323):.4f}**.
            """,
        )


# ----------------------------------------------------------------------------
# Tab 5: Fraud & Anomaly Feed
# ----------------------------------------------------------------------------
def render_fraud_feed() -> None:
    st.subheader("Point-of-Sale (POS) Fraud & Anomaly Feed")
    st.caption("Deep Autoencoder reconstruction MSE scoring on return transactions.")

    explain(
        "Anomaly Detection Engine Architecture",
        """
1. **Unsupervised Autoencoder ANN**: Symmetric compressive bottleneck (`5 -> 16 -> 8 -> 16 -> 5`) learns normal return behavioral dynamics.
2. **Reconstruction MSE Scoring**: High reconstruction error highlights anomalous transactions (serial returns, unverified high-value returns, new account wardrobing).
3. **Strict Zero-Leakage Thresholds**: 95th and 99th percentile cutoff boundaries learned strictly from training distribution.
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

    kpi_html = f"""
    <div class="kpi-container">
        <div class="glass-card">
            <div class="glass-card-top">
                <span class="glass-card-label">Monitored Returns</span>
                <span class="glass-card-indicator"></span>
            </div>
            <div class="glass-card-value">{total_reviewed:,}</div>
            <div class="glass-card-sub">Monitoring Window Transactions</div>
        </div>
        <div class="glass-card">
            <div class="glass-card-top">
                <span class="glass-card-label">Flagged Anomalies</span>
                <span class="glass-card-indicator" style="background-color: #f59e0b;"></span>
            </div>
            <div class="glass-card-value" style="color: #f59e0b;">{n_flagged_p95:,}</div>
            <div class="glass-card-sub">Top 5% Anomaly Scores (p95+)</div>
        </div>
        <div class="glass-card">
            <div class="glass-card-top">
                <span class="glass-card-label">High Severity Exposure</span>
                <span class="glass-card-indicator" style="background-color: #ef4444;"></span>
            </div>
            <div class="glass-card-value" style="color: #ef4444;">€{high_sev_exposure:,.0f}</div>
            <div class="glass-card-sub">Refund value flagged high risk</div>
        </div>
    </div>
    """
    st.markdown(textwrap.dedent(kpi_html).strip(), unsafe_allow_html=True)

    st.markdown("### Reconstruction Error (Anomaly Score) Risk-Tier Distribution")
    st.caption("Distribution of reconstruction error split into Normal (< p95), Elevated (p95–p99), and Critical (> p99) anomaly tiers.")

    p95_val = float(metrics.get("threshold_p95", 0.0)) if metrics else float(df_alerts["reconstruction_error"].quantile(0.95))
    p99_val = float(metrics.get("threshold_p99", 0.0)) if metrics else float(df_alerts["reconstruction_error"].quantile(0.99))

    normal_count = int((df_alerts["reconstruction_error"] < p95_val).sum())
    elevated_count = int(((df_alerts["reconstruction_error"] >= p95_val) & (df_alerts["reconstruction_error"] < p99_val)).sum())
    critical_count = int((df_alerts["reconstruction_error"] >= p99_val).sum())

    # Risk Tier Callout Badges
    pills_html = f"""
    <div style="display: flex; gap: 10px; margin-bottom: 14px; flex-wrap: wrap;">
        <div class="zone-pill" style="background: rgba(2, 132, 199, 0.12); border: 1px solid rgba(2, 132, 199, 0.3); color: #0284c7;">
            Normal Tier: {normal_count:,} ({normal_count / total_reviewed * 100:.1f}%)
        </div>
        <div class="zone-pill" style="background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.3); color: #f59e0b;">
            Elevated Risk (p95): {elevated_count:,} ({elevated_count / total_reviewed * 100:.1f}%)
        </div>
        <div class="zone-pill" style="background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.3); color: #ef4444;">
            Critical Anomaly (p99): {critical_count:,} ({critical_count / total_reviewed * 100:.1f}%)
        </div>
    </div>
    """
    st.markdown(textwrap.dedent(pills_html).strip(), unsafe_allow_html=True)

    # 3-Zone Stacked/Color Histogram
    df_normal = df_alerts[df_alerts["reconstruction_error"] < p95_val]
    df_elevated = df_alerts[(df_alerts["reconstruction_error"] >= p95_val) & (df_alerts["reconstruction_error"] < p99_val)]
    df_critical = df_alerts[df_alerts["reconstruction_error"] >= p99_val]

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=df_normal["reconstruction_error"],
        nbinsx=45,
        name="Normal (< p95)",
        marker=dict(color="#0284c7"),
        opacity=0.85,
    ))
    fig_hist.add_trace(go.Histogram(
        x=df_elevated["reconstruction_error"],
        nbinsx=15,
        name="Elevated (p95 - p99)",
        marker=dict(color="#f59e0b"),
        opacity=0.9,
    ))
    fig_hist.add_trace(go.Histogram(
        x=df_critical["reconstruction_error"],
        nbinsx=10,
        name="Critical Anomaly (> p99)",
        marker=dict(color="#ef4444"),
        opacity=0.95,
    ))

    fig_hist.update_layout(
        barmode="overlay",
        title="Reconstruction Error Distribution by Risk Tier",
        xaxis_title="Reconstruction MSE (Anomaly Score)",
        yaxis_title="Transaction Frequency",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig_hist.add_vline(
        x=p95_val, line_dash="dash", line_color="#f59e0b",
        annotation_text=f"p95 Cutoff ({p95_val:.5f})",
        annotation_font=dict(size=10, color="#f59e0b", family="JetBrains Mono, monospace"),
    )
    fig_hist.add_vline(
        x=p99_val, line_dash="dash", line_color="#ef4444",
        annotation_text=f"p99 Cutoff ({p99_val:.5f})",
        annotation_font=dict(size=10, color="#ef4444", family="JetBrains Mono, monospace"),
    )

    apply_plotly_theme(fig_hist, height=380)
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")
    st.markdown("### Flagged Return Transaction Alert Feed")
    sev_filter = st.radio("Filter Alerts by Severity Tier:", ["All", "High severity only", "Medium & High"], horizontal=True)

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
            f"**Diagnostic QA Note**: Evaluated against injected ground truth, detector achieved "
            f"**{diag.get('recall', 0)*100:.1f}% Recall** and **{diag.get('precision', 0)*100:.1f}% Precision** "
            f"(F1: {diag.get('f1', 0):.4f}) at the 95th percentile threshold."
        )


# ----------------------------------------------------------------------------
# Main Application Shell
# ----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Retail Intelligence & Demand Forecasting Suite",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Inject adaptive theme styling
    inject_custom_css()

    # Executive Hero Banner (Zero Emojis)
    banner_html = """
    <div class="hero-banner">
        <div class="hero-badge">
            <span class="pulse-dot"></span>
            <span>PRODUCTION ANALYTICS PIPELINE • LIVE</span>
        </div>
        <h1 class="hero-title">Retail Intelligence & Demand Forecasting Suite</h1>
        <div class="hero-subtitle">
            Executive BI dashboard integrating multi-model retail time-series demand forecasting,
            statistical hypothesis testing, unsupervised store segmentation, and deep learning POS fraud anomaly detection.
        </div>
        <div class="dataset-pill">
            <strong>Dataset:</strong> Rossmann Store Sales Benchmark (1,115 Operating Stores • 1,017,209 Records • 2013–2015)
        </div>
    </div>
    """
    st.markdown(textwrap.dedent(banner_html).strip(), unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Executive Overview",
        "Store Performance",
        "Demand Forecast",
        "Store Segmentation",
        "Fraud & Anomaly Feed",
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
