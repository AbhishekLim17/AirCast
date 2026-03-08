"""
dashboard/app.py
AirCast — Light Glassmorphism Dashboard
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
    page_title="AirCast — Air Quality Intelligence",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS — Light Glassmorphism + Responsive ───────────────────────────────────

st.markdown("""
<style>
  /* ── Google Font ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  /* ── Reset & base ── */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
  }

  /* ── Dreamy gradient background ── */
  .stApp {
    background: linear-gradient(135deg,
      #dbeafe 0%,
      #ede9fe 35%,
      #fce7f3 65%,
      #d1fae5 100%);
    background-attachment: fixed;
    min-height: 100vh;
  }

  /* ── Floating orbs (decorative blobs via pseudo elements won't work in Streamlit  */
  /* ── so we layer via the main app background instead) ─────────────────────── */

  /* ── Hide default chrome ── */
  #MainMenu, footer, header { visibility: hidden; }

  /* ── Glass card — frosted white ── */
  .glass-card {
    background: rgba(255, 255, 255, 0.62);
    backdrop-filter: blur(24px) saturate(180%);
    -webkit-backdrop-filter: blur(24px) saturate(180%);
    border: 1px solid rgba(255, 255, 255, 0.88);
    border-top: 1.5px solid rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    padding: 22px 24px 20px 24px;
    margin-bottom: 16px;
    box-shadow:
      0 8px 32px rgba(99, 102, 241, 0.08),
      0 2px 8px rgba(0, 0, 0, 0.04),
      inset 0 1px 0 rgba(255, 255, 255, 0.9);
    transition: box-shadow 0.25s ease, transform 0.2s ease;
  }
  .glass-card:hover {
    box-shadow:
      0 12px 40px rgba(99, 102, 241, 0.14),
      0 4px 12px rgba(0, 0, 0, 0.06),
      inset 0 1px 0 rgba(255, 255, 255, 0.9);
    transform: translateY(-1px);
  }

  /* ── Section label ── */
  .section-title {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 4px;
  }

  /* ── Card title ── */
  .card-title {
    font-size: 16px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0;
  }

  /* ── Stat pill ── */
  .stat-pill {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    background: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-top: 1.5px solid rgba(255, 255, 255, 0.95);
    border-radius: 14px;
    padding: 14px 18px;
    min-width: 86px;
    box-shadow: 0 2px 12px rgba(99, 102, 241, 0.1);
    flex: 1;
  }
  .stat-value {
    font-size: 24px;
    font-weight: 800;
    color: #1e1b4b;
    letter-spacing: -0.5px;
  }
  .stat-label {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6366f1;
    margin-top: 3px;
  }

  /* ── Glow accent line ── */
  .glow-line {
    height: 2px;
    border-radius: 2px;
    margin: 10px 0 16px 0;
    opacity: 0.6;
  }

  /* ── Pulsing status dot ── */
  .dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.85); }
  }

  /* ── Header bar ── */
  .aircast-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
    padding: 14px 20px;
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(24px) saturate(180%);
    -webkit-backdrop-filter: blur(24px) saturate(180%);
    border: 1px solid rgba(255, 255, 255, 0.88);
    border-radius: 18px;
    margin-bottom: 22px;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.08);
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.75) !important;
    backdrop-filter: blur(24px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(24px) saturate(180%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.7) !important;
    box-shadow: 4px 0 24px rgba(99, 102, 241, 0.06) !important;
  }

  /* ── Sidebar text ── */
  [data-testid="stSidebar"] * {
    color: #1e293b !important;
  }
  [data-testid="stSidebar"] .stSlider label {
    color: #475569 !important;
    font-weight: 600 !important;
  }

  /* ── Slider track ── */
  [data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
  }

  /* ── Divider ── */
  hr {
    border-color: rgba(99, 102, 241, 0.15) !important;
    margin: 12px 0 !important;
  }

  /* ── Data table ── */
  .stDataFrame {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid rgba(99, 102, 241, 0.12) !important;
  }

  /* ── Metric ── */
  [data-testid="metric-container"] label {
    color: #6366f1 !important;
    font-weight: 600 !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-weight: 800 !important;
  }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: rgba(99,102,241,0.04); }
  ::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.25); border-radius: 4px; }

  /* ─────────────────────────────────────────────────────────────────────────
     RESPONSIVE — Tablet (≤900px)
  ───────────────────────────────────────────────────────────────────────── */
  @media (max-width: 900px) {
    .glass-card {
      padding: 16px 16px 14px 16px;
      border-radius: 16px;
    }
    .stat-pill {
      padding: 10px 12px;
      min-width: unset;
    }
    .stat-value { font-size: 20px; }
    .stat-label { font-size: 8px; }
  }

  /* ─────────────────────────────────────────────────────────────────────────
     RESPONSIVE — Mobile (≤600px)
  ───────────────────────────────────────────────────────────────────────── */
  @media (max-width: 600px) {
    /* Stack columns — Streamlit columns become block on mobile via CSS */
    [data-testid="stHorizontalBlock"] > div {
      min-width: 100% !important;
      flex: 0 0 100% !important;
    }
    .glass-card {
      padding: 14px 14px 12px 14px;
      border-radius: 14px;
      margin-bottom: 12px;
    }
    .aircast-header {
      padding: 10px 14px;
      border-radius: 14px;
    }
    .section-title { font-size: 9px; letter-spacing: 2px; }
    .card-title { font-size: 14px; }
    .stat-pill { padding: 10px 10px; min-width: 72px; }
    .stat-value { font-size: 18px; }
    .stat-label { font-size: 8px; letter-spacing: 1px; }
    .glow-line { margin: 8px 0 12px 0; }
    /* Make sidebar overlay friendly */
    [data-testid="stSidebar"] {
      min-width: 260px !important;
      max-width: 80vw !important;
    }
  }

  /* ─────────────────────────────────────────────────────────────────────────
     RESPONSIVE — Small mobile (≤400px)
  ───────────────────────────────────────────────────────────────────────── */
  @media (max-width: 400px) {
    .glass-card { padding: 12px 10px; border-radius: 12px; }
    .stat-value { font-size: 16px; }
    .aircast-header { border-radius: 12px; }
  }
</style>
""", unsafe_allow_html=True)

# ─── Data loaders ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_chart_data(days: int = 30) -> pd.DataFrame:
    rows = get_joined_chart_data(days=days, station=PRIMARY_STATION)
    if not rows:
        return pd.DataFrame(columns=["date", "actual_aqi", "predicted"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")

@st.cache_data(ttl=300)
def load_performance(days: int = 30) -> pd.DataFrame:
    rows = get_performance_history(days=days)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["eval_date"] = pd.to_datetime(df["eval_date"])
    return df.sort_values("eval_date")

@st.cache_data(ttl=60)
def load_live() -> dict | None:
    return fetch_current_aqi(PRIMARY_STATION)

@st.cache_data(ttl=300)
def load_tomorrow_prediction() -> float | None:
    tomorrow = date.today() + timedelta(days=1)
    from pipeline.db import get_prediction_for_date
    row = get_prediction_for_date(target_date=tomorrow, station=PRIMARY_STATION)
    return float(row["predicted"]) if row else None

# ─── Gauge chart ──────────────────────────────────────────────────────────────

def make_gauge(value: float, title: str) -> go.Figure:
    cat   = get_aqi_category(value)
    color = cat["color"]
    steps = [
        {"range": [0,   50],  "color": "rgba(16,185,129,0.12)"},
        {"range": [51,  100], "color": "rgba(132,204,22,0.12)"},
        {"range": [101, 200], "color": "rgba(234,179,8,0.12)"},
        {"range": [201, 300], "color": "rgba(249,115,22,0.12)"},
        {"range": [301, 400], "color": "rgba(239,68,68,0.12)"},
        {"range": [401, 500], "color": "rgba(127,29,29,0.12)"},
    ]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"size": 46, "color": color, "family": "Inter, sans-serif"},
                "suffix": ""},
        title={
            "text": (
                f"<b style='color:#0f172a'>{title}</b>"
                f"<br><span style='font-size:12px;color:{color};font-weight:600'>"
                f"{cat['label']}</span>"
            ),
            "font": {"size": 14, "family": "Inter, sans-serif"},
        },
        gauge={
            "axis": {
                "range": [0, 500],
                "tickwidth": 1,
                "tickcolor": "rgba(100,116,139,0.3)",
                "tickfont": {"color": "#94a3b8", "size": 10},
            },
            "bar": {"color": color, "thickness": 0.22},
            "bgcolor": "rgba(255,255,255,0)",
            "borderwidth": 0,
            "steps": steps,
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": value,
            },
        },
    ))
    fig.update_layout(
        height=240,
        margin=dict(t=44, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#0f172a",
    )
    return fig

# ─── Forecast chart — light mode ─────────────────────────────────────────────

def make_forecast_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    bands = [
        (0,   50,  "rgba(16,185,129,0.05)"),
        (51,  100, "rgba(132,204,22,0.05)"),
        (101, 200, "rgba(234,179,8,0.05)"),
        (201, 300, "rgba(249,115,22,0.06)"),
        (301, 400, "rgba(239,68,68,0.06)"),
        (401, 500, "rgba(127,29,29,0.06)"),
    ]
    for lo, hi, rgba in bands:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=rgba, line_width=0, layer="below")

    has_actual = "actual_aqi" in df.columns and df["actual_aqi"].notna().any()
    has_pred   = "predicted"  in df.columns and df["predicted"].notna().any()

    if has_actual:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["actual_aqi"],
            mode="lines+markers", name="Actual AQI",
            line=dict(color="#0ea5e9", width=2.5, shape="spline"),
            marker=dict(size=6, color="#0ea5e9",
                        line=dict(color="white", width=1.5)),
            fill="tozeroy",
            fillcolor="rgba(14,165,233,0.08)",
        ))
    if has_pred:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["predicted"],
            mode="lines+markers", name="Predicted",
            line=dict(color="#8b5cf6", width=2, dash="dot", shape="spline"),
            marker=dict(size=6, symbol="diamond", color="#8b5cf6",
                        line=dict(color="white", width=1.5)),
        ))

    fig.update_layout(
        height=320,
        template="none",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=True, gridcolor="rgba(148,163,184,0.2)",
            zeroline=False, title="",
            tickfont=dict(color="#64748b", size=11, family="Inter"),
            tickformat="%d %b",
            showline=True, linecolor="rgba(148,163,184,0.3)",
        ),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(148,163,184,0.2)",
            zeroline=False, range=[0, 500], title="AQI",
            tickfont=dict(color="#64748b", size=11, family="Inter"),
            title_font=dict(color="#94a3b8", size=11),
            showline=True, linecolor="rgba(148,163,184,0.3)",
        ),
        legend=dict(
            orientation="h", y=1.1, x=0,
            font=dict(color="#334155", size=12, family="Inter"),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(t=20, b=20, l=10, r=10),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.95)",
            font_color="#0f172a",
            bordercolor="rgba(99,102,241,0.3)",
            font_family="Inter",
        ),
    )
    return fig

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:18px 0 10px 0;'>
      <div style='font-size:36px;filter:drop-shadow(0 2px 8px rgba(99,102,241,0.4));'>🌤️</div>
      <div style='font-size:20px;font-weight:800;color:#1e1b4b;
      letter-spacing:-0.5px;margin-top:6px;font-family:Inter,sans-serif;'>AirCast</div>
      <div style='font-size:10px;color:#6366f1;letter-spacing:3px;
      text-transform:uppercase;font-weight:600;margin-top:2px;'>Air Quality Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    days_range = st.select_slider(
        "📅 History window",
        options=[7, 14, 30, 60, 90],
        value=30,
    )
    st.divider()

    st.markdown(
        "<div style='font-size:10px;color:#6366f1;letter-spacing:2px;font-weight:700;"
        "text-transform:uppercase;margin-bottom:10px;font-family:Inter,sans-serif;'>"
        "AQI Scale · CPCB India</div>",
        unsafe_allow_html=True,
    )
    for cat in AQI_CATEGORIES:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:7px;"
            f"background:rgba(255,255,255,0.6);backdrop-filter:blur(8px);"
            f"border:1px solid rgba(255,255,255,0.85);"
            f"border-radius:10px;padding:7px 10px;'>"
            f"<div style='width:10px;height:10px;border-radius:50%;flex-shrink:0;"
            f"background:{cat['color']};box-shadow:0 0 8px {cat['color']}88;'></div>"
            f"<div style='font-size:12px;color:#1e293b;font-weight:600;"
            f"font-family:Inter,sans-serif;'>{cat['label']} "
            f"<span style='color:#64748b;font-weight:400;font-size:11px;'>"
            f"({cat['min']}–{cat['max']})</span></div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.divider()
    st.markdown(
        f"<div style='font-size:10px;color:#64748b;font-family:Inter,sans-serif;'>"
        f"📍 Station: <span style='color:#6366f1;font-weight:600;'>"
        f"{PRIMARY_STATION.upper()}</span><br>"
        f"🗓️ Updated: <span style='color:#475569;'>"
        f"{date.today().strftime('%d %b %Y')}</span></div>",
        unsafe_allow_html=True,
    )

# ─── Load data ────────────────────────────────────────────────────────────────

live_data = load_live()
tmr_pred  = load_tomorrow_prediction()
chart_df  = load_chart_data(days=days_range)
perf_df   = load_performance(days=days_range)

live_aqi = float(live_data.get("aqi", 0)) if live_data else None
live_cat = get_aqi_category(live_aqi) if live_aqi is not None else None

# ─── Header bar ───────────────────────────────────────────────────────────────

dot_color  = live_cat["color"] if live_cat else "#6366f1"
cat_label  = live_cat["label"] if live_cat else "Loading..."
live_value = f"{live_aqi:.0f}" if live_aqi else "—"

st.markdown(
    f"<div class='aircast-header'>"
    f"<div style='display:flex;align-items:center;gap:12px;'>"
    f"<div style='font-size:28px;filter:drop-shadow(0 2px 8px rgba(99,102,241,0.4));'>🌤️</div>"
    f"<div>"
    f"<div style='font-size:22px;font-weight:800;color:#0f172a;"
    f"letter-spacing:-0.5px;line-height:1.1;font-family:Inter,sans-serif;'>AirCast</div>"
    f"<div style='font-size:12px;color:#6366f1;font-weight:500;"
    f"font-family:Inter,sans-serif;'>Ahmedabad · Air Quality Intelligence</div>"
    f"</div></div>"
    f"<div style='display:flex;align-items:center;gap:10px;flex-wrap:wrap;'>"
    f"<div style='font-size:13px;color:#475569;background:rgba(255,255,255,0.8);"
    f"border:1px solid rgba(99,102,241,0.2);border-radius:20px;padding:5px 14px;"
    f"font-family:Inter,sans-serif;font-weight:500;'>"
    f"<span class='dot' style='background:{dot_color};'></span>Live monitoring</div>"
    f"<div style='font-size:13px;font-weight:700;color:white;"
    f"background:{dot_color};border-radius:20px;padding:5px 16px;"
    f"font-family:Inter,sans-serif;box-shadow:0 2px 12px {dot_color}66;'>"
    f"AQI {live_value} · {cat_label}</div>"
    f"</div></div>",
    unsafe_allow_html=True,
)

# ─── Row 1: Two gauges + stats ────────────────────────────────────────────────

col_g1, col_g2, col_stats = st.columns([1, 1, 1.4])

with col_g1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title'>🟢 Live Reading</div>"
        "<div class='card-title'>Current AQI</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='glow-line' style='background:linear-gradient(90deg,#0ea5e9,#38bdf8,transparent);'></div>",
        unsafe_allow_html=True,
    )
    if live_aqi is not None:
        st.plotly_chart(
            make_gauge(live_aqi, "Now"),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        dominant = live_data.get("dominentpol", "N/A").upper()
        st.markdown(
            f"<div style='text-align:center;font-size:12px;color:#64748b;"
            f"font-family:Inter,sans-serif;margin-top:-8px;'>"
            f"Dominant pollutant: "
            f"<b style='color:#0f172a;'>{dominant}</b></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='color:#94a3b8;padding:60px 0;text-align:center;"
            "font-size:13px;font-family:Inter,sans-serif;'>WAQI data unavailable</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with col_g2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title'>🔮 Tomorrow's Forecast</div>"
        "<div class='card-title'>Predicted AQI</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='glow-line' style='background:linear-gradient(90deg,#8b5cf6,#a78bfa,transparent);'></div>",
        unsafe_allow_html=True,
    )
    if tmr_pred is not None:
        st.plotly_chart(
            make_gauge(tmr_pred, "Tomorrow"),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        tmr_date = (date.today() + timedelta(days=1)).strftime("%A, %d %b %Y")
        st.markdown(
            f"<div style='text-align:center;font-size:12px;color:#64748b;"
            f"font-family:Inter,sans-serif;margin-top:-8px;'>{tmr_date}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='color:#94a3b8;padding:60px 0;text-align:center;"
            "font-size:13px;font-family:Inter,sans-serif;'>No prediction yet<br>"
            "<span style='font-size:11px;color:#cbd5e0;'>Run daily job first</span></div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with col_stats:
    st.markdown("<div class='glass-card' style='height:100%;'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title'>📊 Model Performance</div>"
        "<div class='card-title'>Accuracy Metrics</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='glow-line' style='background:linear-gradient(90deg,#6366f1,#8b5cf6,#ec4899,transparent);'></div>",
        unsafe_allow_html=True,
    )

    if not perf_df.empty:
        latest = perf_df.iloc[-1]
        mae  = latest.get("mae",  None)
        rmse = latest.get("rmse", None)
        mape = latest.get("mape", None)

        pills = [
            (f"{mae:.1f}"   if mae  is not None else "—", "MAE"),
            (f"{rmse:.1f}"  if rmse is not None else "—", "RMSE"),
            (f"{mape:.1f}%" if mape is not None else "—", "MAPE"),
        ]
        pill_html = (
            "<div style='display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px;'>"
        )
        for val, lbl in pills:
            pill_html += (
                f"<div class='stat-pill'>"
                f"<div class='stat-value'>{val}</div>"
                f"<div class='stat-label'>{lbl}</div>"
                f"</div>"
            )
        pill_html += "</div>"
        st.markdown(pill_html, unsafe_allow_html=True)

        # Sparkline — MAE trend
        fig_spark = go.Figure()
        fig_spark.add_trace(go.Scatter(
            x=perf_df["eval_date"], y=perf_df["mae"],
            mode="lines", fill="tozeroy",
            line=dict(color="#6366f1", width=2),
            fillcolor="rgba(99,102,241,0.08)",
        ))
        fig_spark.add_hline(
            y=20, line_dash="dot", line_color="rgba(234,179,8,0.7)",
            annotation_text="threshold",
            annotation_font_size=9,
            annotation_font_color="rgba(161,130,8,0.8)",
        )
        fig_spark.update_layout(
            height=100, margin=dict(t=4, b=4, l=0, r=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            showlegend=False,
        )
        st.plotly_chart(fig_spark, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            "<div style='font-size:10px;color:#94a3b8;text-align:right;"
            "margin-top:-6px;font-family:Inter,sans-serif;'>MAE trend over time</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='color:#94a3b8;font-size:13px;padding:20px 0;"
            "font-family:Inter,sans-serif;'>"
            "Stats appear after the first evaluation run.</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ─── Row 2: Main forecast chart ───────────────────────────────────────────────

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown(
    f"<div style='display:flex;align-items:flex-start;justify-content:space-between;"
    f"flex-wrap:wrap;gap:8px;margin-bottom:4px;'>"
    f"<div>"
    f"<div class='section-title'>📈 Forecast vs Reality</div>"
    f"<div class='card-title'>Predicted · Actual AQI — Last {days_range} days</div>"
    f"</div>"
    f"<div style='font-size:11px;color:#6366f1;font-weight:600;"
    f"background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);"
    f"border-radius:8px;padding:4px 12px;font-family:Inter,sans-serif;white-space:nowrap;'>"
    f"Ahmedabad · CPCB India</div>"
    f"</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='glow-line' style='background:linear-gradient(90deg,#0ea5e9,#8b5cf6,#ec4899,transparent);'></div>",
    unsafe_allow_html=True,
)

if chart_df.empty:
    st.markdown(
        "<div style='color:#94a3b8;font-size:13px;padding:30px 0;text-align:center;"
        "font-family:Inter,sans-serif;'>"
        "Chart populates after the first daily job runs.</div>",
        unsafe_allow_html=True,
    )
else:
    st.plotly_chart(
        make_forecast_chart(chart_df),
        use_container_width=True,
        config={"displayModeBar": False},
    )
st.markdown("</div>", unsafe_allow_html=True)

# ─── Row 3: Accuracy history + Retrain log ────────────────────────────────────

col_acc, col_ret = st.columns([1.4, 1])

with col_acc:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title'>📉 Accuracy History</div>"
        "<div class='card-title'>MAE & MAPE over time</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='glow-line' style='background:linear-gradient(90deg,#8b5cf6,#6366f1,transparent);'></div>",
        unsafe_allow_html=True,
    )

    if perf_df.empty:
        st.markdown(
            "<div style='color:#94a3b8;font-size:13px;padding:20px 0;"
            "font-family:Inter,sans-serif;'>No data yet.</div>",
            unsafe_allow_html=True,
        )
    else:
        fig3 = make_subplots(
            rows=1, cols=2,
            subplot_titles=["MAE over time", "MAPE (%) over time"],
        )
        fig3.add_trace(go.Scatter(
            x=perf_df["eval_date"], y=perf_df["mae"],
            mode="lines+markers", name="MAE",
            line=dict(color="#6366f1", width=2.5, shape="spline"),
            marker=dict(size=5, color="#6366f1", line=dict(color="white", width=1.5)),
            fill="tozeroy", fillcolor="rgba(99,102,241,0.07)",
        ), row=1, col=1)
        fig3.add_hline(
            y=20, line_dash="dash", line_color="rgba(234,179,8,0.7)",
            annotation_text="retrain threshold",
            annotation_font_size=9,
            annotation_font_color="rgba(161,130,8,0.8)",
            row=1, col=1,
        )
        fig3.add_trace(go.Scatter(
            x=perf_df["eval_date"], y=perf_df["mape"],
            mode="lines+markers", name="MAPE %",
            line=dict(color="#0ea5e9", width=2.5, shape="spline"),
            marker=dict(size=5, color="#0ea5e9", line=dict(color="white", width=1.5)),
            fill="tozeroy", fillcolor="rgba(14,165,233,0.07)",
        ), row=1, col=2)
        fig3.update_layout(
            height=250, template="none",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, margin=dict(t=28, b=8, l=0, r=0),
            font=dict(color="#64748b", family="Inter, sans-serif"),
        )
        for anno in fig3.layout.annotations:
            anno.font.color = "#64748b"
            anno.font.size  = 11
        fig3.update_xaxes(
            showgrid=True, gridcolor="rgba(148,163,184,0.2)",
            tickfont=dict(size=10, color="#94a3b8"),
            showline=True, linecolor="rgba(148,163,184,0.3)",
        )
        fig3.update_yaxes(
            showgrid=True, gridcolor="rgba(148,163,184,0.2)",
            tickfont=dict(size=10, color="#94a3b8"),
            showline=True, linecolor="rgba(148,163,184,0.3)",
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

with col_ret:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title'>🔄 Retraining Log</div>"
        "<div class='card-title'>Auto-retrain events</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='glow-line' style='background:linear-gradient(90deg,#f59e0b,#ef4444,transparent);'></div>",
        unsafe_allow_html=True,
    )

    if perf_df.empty:
        st.markdown(
            "<div style='color:#94a3b8;font-size:13px;padding:20px 0;"
            "font-family:Inter,sans-serif;'>No data yet.</div>",
            unsafe_allow_html=True,
        )
    else:
        retrain_df = perf_df[perf_df.get("retrain_triggered", pd.Series(dtype=bool)) == True].copy()
        if retrain_df.empty:
            st.markdown(
                "<div style='display:flex;align-items:center;gap:12px;padding:18px 0;'>"
                "<div style='font-size:32px;'>✅</div>"
                "<div style='font-size:13px;color:#475569;font-family:Inter,sans-serif;'>"
                "<b style='color:#0f172a;'>All good!</b><br>"
                "Model is within threshold.<br>No retraining needed.</div></div>",
                unsafe_allow_html=True,
            )
        else:
            dcols = ["eval_date", "mae", "new_mae", "promoted"]
            dcols = [c for c in dcols if c in retrain_df.columns]
            st.dataframe(
                retrain_df[dcols].rename(columns={
                    "eval_date": "Date",
                    "mae":       "MAE Before",
                    "new_mae":   "MAE After",
                    "promoted":  "Promoted",
                }).reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )
        n = len(retrain_df)
        st.markdown(
            f"<div style='margin-top:12px;font-size:11px;color:#94a3b8;"
            f"font-family:Inter,sans-serif;'>"
            f"Total retrains: "
            f"<b style='color:#0f172a;font-weight:700;'>{n}</b></div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
