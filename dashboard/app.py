"""
dashboard/app.py
Streamlit dashboard for AirCast.

Sections:
  1. Today's Forecast  — big AQI badge with health category
  2. Predicted vs Actual — 30-day interactive Plotly chart
  3. Model Accuracy    — rolling MAE / MAPE sparkline
  4. Retraining History — events table
  5. Live Reading      — current WAQI API reading
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from datetime import date, timedelta

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import AQI_CATEGORIES, get_aqi_category, PRIMARY_STATION
from pipeline.db import (
    get_actuals,
    get_predictions,
    get_performance_history,
    get_joined_chart_data,
)
from pipeline.fetch_data import fetch_current_aqi

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AirCast — Ahmedabad AQI",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
  .aqi-badge {
    border-radius: 16px;
    padding: 28px 40px;
    text-align: center;
    margin-bottom: 8px;
  }
  .aqi-value   { font-size: 72px; font-weight: 800; line-height: 1; color: #fff; }
  .aqi-label   { font-size: 22px; font-weight: 600; color: #fff; margin-top: 4px; }
  .aqi-sub     { font-size: 13px; color: rgba(255,255,255,0.75); margin-top: 2px; }
  .metric-card {
    background: #1e1e2e;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 12px;
  }
  .stDataFrame { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─── Data loaders (cached) ────────────────────────────────────────────────────

@st.cache_data(ttl=300)   # refresh every 5 minutes
def load_chart_data(days: int = 30) -> pd.DataFrame:
    rows = get_joined_chart_data(days=days, station=PRIMARY_STATION)
    if not rows:
        return pd.DataFrame(columns=["date", "actual_aqi", "predicted"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df


@st.cache_data(ttl=300)
def load_performance(days: int = 30) -> pd.DataFrame:
    rows = get_performance_history(days=days)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["eval_date"] = pd.to_datetime(df["eval_date"])
    return df.sort_values("eval_date")


@st.cache_data(ttl=60)    # refresh live reading every 60 s
def load_live() -> dict | None:
    return fetch_current_aqi(PRIMARY_STATION)


@st.cache_data(ttl=300)
def load_tomorrow_prediction() -> float | None:
    tomorrow = date.today() + timedelta(days=1)
    from pipeline.db import get_prediction_for_date
    row = get_prediction_for_date(target_date=tomorrow, station=PRIMARY_STATION)
    return float(row["predicted"]) if row else None


# ─── Helper: AQI badge HTML ───────────────────────────────────────────────────

def aqi_badge(value: float, label_prefix: str = "") -> str:
    cat = get_aqi_category(value)
    color = cat["color"]
    label = cat["label"]
    return f"""
    <div class="aqi-badge" style="background:{color};">
      <div class="aqi-value">{int(round(value))}</div>
      <div class="aqi-label">{label}</div>
      <div class="aqi-sub">{label_prefix}</div>
    </div>
    """


# ─── Layout ───────────────────────────────────────────────────────────────────

st.title("🌫️ AirCast — Ahmedabad AQI Forecast")
st.caption(f"Station: **{PRIMARY_STATION.title()}**  •  Last refreshed: {date.today().strftime('%d %b %Y')}")

# ── Row 1: three badges ───────────────────────────────────────────────────────

col_live, col_tmr, col_gap = st.columns([1, 1, 2])

live_data = load_live()
tmr_pred  = load_tomorrow_prediction()

with col_live:
    st.subheader("Live Reading")
    if live_data:
        st.markdown(aqi_badge(float(live_data.get("aqi", 0)), "Right now"), unsafe_allow_html=True)
        st.caption(f"Dominant: {live_data.get('dominentpol', 'N/A').upper()}")
    else:
        st.warning("Could not fetch live AQI from WAQI.")

with col_tmr:
    st.subheader("Tomorrow's Forecast")
    if tmr_pred is not None:
        st.markdown(aqi_badge(tmr_pred, "Model prediction"), unsafe_allow_html=True)
    else:
        st.info("No forecast stored yet — run the daily job first.")

with col_gap:
    # AQI legend
    st.subheader("AQI Scale (CPCB India)")
    legend_html = "<div style='display:flex; flex-wrap:wrap; gap:8px; margin-top:6px;'>"
    for cat in AQI_CATEGORIES:
        legend_html += (
            f"<div style='background:{cat['color']};border-radius:8px;"
            f"padding:6px 14px;color:#fff;font-size:13px;font-weight:600;'>"
            f"{cat['label']} ({cat['min']}–{cat['max']})</div>"
        )
    legend_html += "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)

st.divider()

# ── Row 2: Actual vs Predicted chart ─────────────────────────────────────────

st.subheader("📈 Predicted vs Actual AQI (Last 30 Days)")
chart_df = load_chart_data(days=30)

if chart_df.empty:
    st.info("No historical data yet. The chart will populate after the first few daily job runs.")
else:
    fig = go.Figure()

    # AQI background bands
    band_colors = {
        "Good":         ("rgba(0,228,0,0.07)",    0,   50),
        "Satisfactory": ("rgba(146,208,80,0.07)", 51,  100),
        "Moderate":     ("rgba(255,255,0,0.07)",  101, 200),
        "Poor":         ("rgba(255,126,0,0.09)",  201, 300),
        "Very Poor":    ("rgba(255,0,0,0.09)",    301, 400),
        "Severe":       ("rgba(126,0,35,0.10)",   401, 500),
    }
    for name, (rgba, lo, hi) in band_colors.items():
        fig.add_hrect(y0=lo, y1=hi, fillcolor=rgba, line_width=0,
                      annotation_text=name, annotation_position="right",
                      annotation=dict(font_size=10, font_color="gray"))

    if "actual_aqi" in chart_df.columns and chart_df["actual_aqi"].notna().any():
        fig.add_trace(go.Scatter(
            x=chart_df["date"], y=chart_df["actual_aqi"],
            mode="lines+markers", name="Actual AQI",
            line=dict(color="#4c9be8", width=2.5),
            marker=dict(size=5),
        ))

    if "predicted" in chart_df.columns and chart_df["predicted"].notna().any():
        fig.add_trace(go.Scatter(
            x=chart_df["date"], y=chart_df["predicted"],
            mode="lines+markers", name="Predicted AQI",
            line=dict(color="#f87171", width=2, dash="dot"),
            marker=dict(size=5, symbol="diamond"),
        ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="AQI",
        yaxis=dict(range=[0, 500]),
        legend=dict(orientation="h", y=-0.15),
        height=400,
        template="plotly_dark",
        margin=dict(t=10, b=60),
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Row 3: Accuracy metrics ───────────────────────────────────────────────────

st.subheader("📊 Model Accuracy Over Time")
perf_df = load_performance(days=30)

if perf_df.empty:
    st.info("Accuracy metrics will appear here after the first evaluation run.")
else:
    c1, c2, c3 = st.columns(3)
    latest = perf_df.iloc[-1]
    c1.metric("Latest MAE",  f"{latest.get('mae',  0):.2f}", help="Mean Absolute Error")
    c2.metric("Latest RMSE", f"{latest.get('rmse', 0):.2f}", help="Root Mean Squared Error")
    c3.metric("Latest MAPE", f"{latest.get('mape', 0):.2f}%", help="Mean Absolute Percentage Error")

    fig2 = make_subplots(rows=1, cols=2, subplot_titles=("MAE over time", "MAPE (%) over time"))

    fig2.add_trace(go.Scatter(
        x=perf_df["eval_date"], y=perf_df["mae"],
        mode="lines+markers", name="MAE",
        line=dict(color="#f87171", width=2),
    ), row=1, col=1)

    # Retrain threshold line
    fig2.add_hline(y=20, line_dash="dash", line_color="orange",
                   annotation_text="Retrain threshold (MAE=20)", row=1, col=1)

    fig2.add_trace(go.Scatter(
        x=perf_df["eval_date"], y=perf_df["mape"],
        mode="lines+markers", name="MAPE %",
        line=dict(color="#60a5fa", width=2),
    ), row=1, col=2)

    fig2.update_layout(
        height=320, template="plotly_dark",
        showlegend=False, margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Row 4: Retraining history ─────────────────────────────────────────────────

st.subheader("🔄 Retraining History")

if not perf_df.empty:
    retrain_df = perf_df[perf_df["retrain_triggered"] == True].copy()
    if retrain_df.empty:
        st.success("No retraining events yet — model is performing within threshold.")
    else:
        display_cols = ["eval_date", "model_ver", "mae", "new_model_ver", "new_mae",
                        "retrain_reason", "promoted"]
        display_cols = [c for c in display_cols if c in retrain_df.columns]
        st.dataframe(
            retrain_df[display_cols].rename(columns={
                "eval_date":     "Date",
                "model_ver":     "Old Model",
                "mae":           "MAE Before",
                "new_model_ver": "New Model",
                "new_mae":       "MAE After",
                "retrain_reason":"Reason",
                "promoted":      "Promoted",
            }),
            use_container_width=True,
            hide_index=True,
        )
else:
    st.info("No performance data yet.")

st.divider()

# ── Row 5: Raw data expanders ─────────────────────────────────────────────────

with st.expander("📋 Raw predictions data (last 30 days)"):
    pred_rows = get_predictions(days=30, station=PRIMARY_STATION)
    if pred_rows:
        st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No predictions stored yet.")

with st.expander("📋 Raw actuals data (last 30 days)"):
    actual_rows = get_actuals(days=30, station=PRIMARY_STATION)
    if actual_rows:
        st.dataframe(pd.DataFrame(actual_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No actuals stored yet.")

# ── Footer ────────────────────────────────────────────────────────────────────

st.caption(
    "Data source: [WAQI API](https://waqi.info) · "
    "Model: XGBoost (auto-retrained daily) · "
    "Built with Streamlit + Supabase + HuggingFace Hub"
)
