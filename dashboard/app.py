"""
dashboard/app.py
AirCast — Mission Control Dashboard (Dark Glassmorphism)
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
    page_title="AirCast — Mission Control",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Mission Control CSS ──────────────────────────────────────────────────────

st.markdown("""
<style>
  /* ── Global background — deep navy, not pure black ── */
  .stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1a1f35 50%, #0f172a 100%);
  }

  /* ── Hide default streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }

  /* ── Glass card — visible dark slate with blue tint ── */
  .glass-card {
    background: linear-gradient(145deg, #1e3347 0%, #172840 100%);
    border: 1px solid rgba(99,179,237,0.35);
    border-top: 1px solid rgba(99,179,237,0.55);
    border-radius: 16px;
    padding: 22px 26px;
    margin-bottom: 16px;
    box-shadow: 0 4px 28px rgba(0,0,0,0.6), 0 0 0 1px rgba(99,179,237,0.05), inset 0 1px 0 rgba(255,255,255,0.08);
  }

  /* ── Section header ── */
  .section-title {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #63b3ed;
    margin-bottom: 4px;
  }

  /* ── Card title ── */
  .card-title {
    font-size: 16px;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 0;
  }

  /* ── Stat pill ── */
  .stat-pill {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    background: linear-gradient(145deg, #1a3a5c 0%, #122540 100%);
    border: 1px solid rgba(99,179,237,0.3);
    border-radius: 12px;
    padding: 14px 20px;
    min-width: 90px;
  }
  .stat-value { font-size: 26px; font-weight: 800; color: #e2e8f0; }
  .stat-label { font-size: 10px; font-weight: 600; letter-spacing: 1.5px;
                text-transform: uppercase; color: #63b3ed; margin-top: 3px; }

  /* ── Glow accent line ── */
  .glow-line {
    height: 2px;
    border-radius: 2px;
    margin: 12px 0 18px 0;
  }

  /* ── Status dot ── */
  .dot { display:inline-block; width:8px; height:8px; border-radius:50%;
         margin-right:6px; animation: pulse 2s infinite; }
  @keyframes pulse {
    0%,100% { opacity:1; } 50% { opacity:0.3; }
  }

  /* ── Sidebar styling ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1829 0%, #111827 100%) !important;
    border-right: 1px solid rgba(99,179,237,0.15) !important;
  }

  /* ── Sidebar text ── */
  [data-testid="stSidebar"] * { color: #cbd5e1 !important; }

  /* ── Slider track ── */
  [data-testid="stSlider"] > div > div > div > div {
    background: #3b82f6 !important;
  }

  /* ── Data table ── */
  .stDataFrame { border-radius: 12px; overflow: hidden; }

  /* ── Metric sub-text ── */
  [data-testid="metric-container"] label { color: #93c5fd !important; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
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
    cat = get_aqi_category(value)
    color = cat["color"]
    steps = [
        {"range": [0,   50],  "color": "rgba(0,228,0,0.15)"},
        {"range": [51,  100], "color": "rgba(146,208,80,0.15)"},
        {"range": [101, 200], "color": "rgba(255,255,0,0.15)"},
        {"range": [201, 300], "color": "rgba(255,126,0,0.15)"},
        {"range": [301, 400], "color": "rgba(255,0,0,0.15)"},
        {"range": [401, 500], "color": "rgba(126,0,35,0.15)"},
    ]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"size": 48, "color": color, "family": "monospace"}},
        title={
            "text": f"<b>{title}</b><br><span style='font-size:13px;color:{color}'>{cat['label']}</span>",
            "font": {"size": 14, "color": "rgba(255,255,255,0.6)"},
        },
        gauge={
            "axis": {
                "range": [0, 500], "tickwidth": 1,
                "tickcolor": "rgba(255,255,255,0.2)",
                "tickfont": {"color": "rgba(255,255,255,0.3)", "size": 10},
            },
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
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
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
    )
    return fig

# ─── Glowing forecast chart ───────────────────────────────────────────────────

def make_forecast_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    bands = [
        (0,   50,  "rgba(0,228,0,0.04)"),
        (51,  100, "rgba(146,208,80,0.04)"),
        (101, 200, "rgba(255,255,0,0.04)"),
        (201, 300, "rgba(255,126,0,0.05)"),
        (301, 400, "rgba(255,0,0,0.05)"),
        (401, 500, "rgba(126,0,35,0.06)"),
    ]
    for lo, hi, rgba in bands:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=rgba, line_width=0)

    has_actual = "actual_aqi" in df.columns and df["actual_aqi"].notna().any()
    has_pred   = "predicted"  in df.columns and df["predicted"].notna().any()

    if has_actual:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["actual_aqi"],
            mode="lines+markers", name="Actual",
            line=dict(color="#38bdf8", width=2.5),
            marker=dict(size=6, color="#38bdf8", line=dict(color="#0ea5e9", width=1)),
            fill="tozeroy",
            fillcolor="rgba(56,189,248,0.05)",
        ))

    if has_pred:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["predicted"],
            mode="lines+markers", name="Predicted",
            line=dict(color="#f472b6", width=2, dash="dot"),
            marker=dict(size=6, symbol="diamond", color="#f472b6",
                        line=dict(color="#ec4899", width=1)),
        ))

    fig.update_layout(
        height=340,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=True, gridcolor="rgba(255,255,255,0.04)",
            zeroline=False, title="",
            tickfont=dict(color="rgba(255,255,255,0.4)", size=11),
        ),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(255,255,255,0.04)",
            zeroline=False, range=[0, 500], title="AQI",
            tickfont=dict(color="rgba(255,255,255,0.4)", size=11),
            title_font=dict(color="rgba(255,255,255,0.3)", size=11),
        ),
        legend=dict(
            orientation="h", y=1.08, x=0,
            font=dict(color="rgba(255,255,255,0.6)", size=12),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(t=20, b=20, l=10, r=10),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1e293b", font_color="white", bordercolor="#334155"),
    )
    return fig

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
      <div style='font-size:32px'>🛰️</div>
      <div style='font-size:18px; font-weight:800; color:#e2e8f0; letter-spacing:1px;'>AirCast</div>
      <div style='font-size:11px; color:rgba(255,255,255,0.3); letter-spacing:2px;
      text-transform:uppercase;'>Mission Control</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    days_range = st.select_slider(
        "History window",
        options=[7, 14, 30, 60, 90],
        value=30,
    )
    st.divider()

    st.markdown(
        "<div style='font-size:11px; color:#63b3ed; letter-spacing:1px;"
        "text-transform:uppercase; margin-bottom:8px;font-weight:600;'>AQI Scale (CPCB India)</div>",
        unsafe_allow_html=True,
    )
    for cat in AQI_CATEGORIES:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px;"
            f"background:rgba(255,255,255,0.04);border-radius:8px;padding:6px 10px;'>"
            f"<div style='width:10px;height:10px;border-radius:50%;"
            f"background:{cat['color']};flex-shrink:0;box-shadow:0 0 6px {cat['color']};'></div>"
            f"<div style='font-size:12px;color:#cbd5e1;font-weight:500;'>{cat['label']} "
            f"<span style='color:#64748b;'>({cat['min']}–{cat['max']})</span></div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.divider()
    st.markdown(
        f"<div style='font-size:10px;color:#64748b;'>"
        f"Station: <span style='color:#93c5fd;'>{PRIMARY_STATION.upper()}</span><br>"
        f"Updated: {date.today().strftime('%d %b %Y')}</div>",
        unsafe_allow_html=True,
    )

# ─── Load data ────────────────────────────────────────────────────────────────

live_data = load_live()
tmr_pred  = load_tomorrow_prediction()
chart_df  = load_chart_data(days=days_range)
perf_df   = load_performance(days=days_range)

live_aqi  = float(live_data.get("aqi", 0)) if live_data else None
live_cat  = get_aqi_category(live_aqi) if live_aqi is not None else None

# ─── Header bar ───────────────────────────────────────────────────────────────

dot_color = live_cat["color"] if live_cat else "#64748b"
st.markdown(
    f"<div style='display:flex;align-items:center;justify-content:space-between;"
    f"padding:12px 4px 20px 4px;border-bottom:2px solid rgba(99,179,237,0.2);margin-bottom:24px;'>"
    f"<div style='font-size:24px;font-weight:800;color:#f1f5f9;letter-spacing:-0.5px;'>"
    f"🛰️ AirCast "
    f"<span style='font-size:14px;font-weight:400;color:#63b3ed;'>/ Ahmedabad</span></div>"
    f"<div style='font-size:13px;color:#94a3b8;background:rgba(99,179,237,0.1);"
    f"border:1px solid rgba(99,179,237,0.2);border-radius:20px;padding:4px 14px;'>"
    f"<span class='dot' style='background:{dot_color}'></span>Live monitoring</div>"
    f"</div>",
    unsafe_allow_html=True,
)

# ─── Row 1: Two gauges + stats ────────────────────────────────────────────────

col_g1, col_g2, col_stats = st.columns([1, 1, 1.4])

with col_g1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🟢 Live Reading</div>", unsafe_allow_html=True)
    if live_aqi is not None:
        st.plotly_chart(
            make_gauge(live_aqi, "Now"),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown(
            f"<div style='text-align:center;font-size:12px;color:rgba(255,255,255,0.35);margin-top:-8px;'>"
            f"Dominant: <b style='color:rgba(255,255,255,0.6)'>"
            f"{live_data.get('dominentpol','N/A').upper()}</b></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='color:rgba(255,255,255,0.35);padding:60px 0;text-align:center;"
            "font-size:13px;'>WAQI unavailable</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with col_g2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🔮 Tomorrow's Forecast</div>", unsafe_allow_html=True)
    if tmr_pred is not None:
        st.plotly_chart(
            make_gauge(tmr_pred, "Tomorrow"),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        tmr_date = (date.today() + timedelta(days=1)).strftime("%a, %d %b")
        st.markdown(
            f"<div style='text-align:center;font-size:12px;color:rgba(255,255,255,0.35);margin-top:-8px;'>"
            f"{tmr_date}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='color:rgba(255,255,255,0.35);padding:60px 0;text-align:center;"
            "font-size:13px;'>No prediction yet<br>"
            "<span style='font-size:11px;'>Run daily job first</span></div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with col_stats:
    st.markdown("<div class='glass-card' style='height:100%;'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📊 Model Performance</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='glow-line' style='background:linear-gradient(90deg,#3b82f6,#8b5cf6,transparent);'></div>",
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
        pill_html = "<div style='display:flex;gap:12px;flex-wrap:wrap;margin-bottom:18px;'>"
        for val, lbl in pills:
            pill_html += (
                f"<div class='stat-pill'>"
                f"<div class='stat-value'>{val}</div>"
                f"<div class='stat-label'>{lbl}</div>"
                f"</div>"
            )
        pill_html += "</div>"
        st.markdown(pill_html, unsafe_allow_html=True)

        fig_spark = go.Figure()
        fig_spark.add_trace(go.Scatter(
            x=perf_df["eval_date"], y=perf_df["mae"],
            mode="lines", fill="tozeroy",
            line=dict(color="#f87171", width=1.5),
            fillcolor="rgba(248,113,113,0.08)",
        ))
        fig_spark.add_hline(
            y=20, line_dash="dot", line_color="rgba(251,191,36,0.5)",
            annotation_text="threshold",
            annotation_font_size=9,
            annotation_font_color="rgba(251,191,36,0.5)",
        )
        fig_spark.update_layout(
            height=110, margin=dict(t=4, b=4, l=0, r=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            showlegend=False,
        )
        st.plotly_chart(fig_spark, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            "<div style='font-size:10px;color:rgba(255,255,255,0.25);text-align:right;margin-top:-8px;'>"
            "MAE over time</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='color:rgba(255,255,255,0.25);font-size:13px;padding:20px 0;'>"
            "Stats appear after the first evaluation run.</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ─── Row 2: Main forecast chart ───────────────────────────────────────────────

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown(
    f"<div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;'>"
    f"<div><div class='section-title'>📈 Forecast vs Reality</div>"
    f"<div style='font-size:16px;font-weight:700;color:#e2e8f0;letter-spacing:-0.3px;'>"
    f"Predicted · Actual AQI — Last {days_range} days</div></div>"
    f"<div style='font-size:11px;color:#64748b;background:rgba(99,179,237,0.08);"
    f"border:1px solid rgba(99,179,237,0.15);border-radius:8px;padding:3px 10px;'>Ahmedabad · CPCB India</div>"
    f"</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='glow-line' style='background:linear-gradient(90deg,#38bdf8,#f472b6,transparent);'></div>",
    unsafe_allow_html=True,
)

if chart_df.empty:
    st.markdown(
        "<div style='color:#64748b;font-size:13px;padding:30px 0;text-align:center;'>"
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

# ─── Row 3: Accuracy detail + Retrain log ─────────────────────────────────────

col_acc, col_ret = st.columns([1.4, 1])

with col_acc:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📉 Accuracy History</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='glow-line' style='background:linear-gradient(90deg,#8b5cf6,#3b82f6,transparent);'></div>",
        unsafe_allow_html=True,
    )

    if perf_df.empty:
        st.markdown(
            "<div style='color:rgba(255,255,255,0.25);font-size:13px;padding:20px 0;'>No data yet.</div>",
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
            line=dict(color="#f87171", width=2),
            marker=dict(size=5),
            fill="tozeroy", fillcolor="rgba(248,113,113,0.07)",
        ), row=1, col=1)
        fig3.add_hline(
            y=20, line_dash="dash", line_color="rgba(251,191,36,0.6)",
            annotation_text="retrain threshold",
            annotation_font_size=9,
            annotation_font_color="rgba(251,191,36,0.5)",
            row=1, col=1,
        )
        fig3.add_trace(go.Scatter(
            x=perf_df["eval_date"], y=perf_df["mape"],
            mode="lines+markers", name="MAPE %",
            line=dict(color="#818cf8", width=2),
            marker=dict(size=5),
            fill="tozeroy", fillcolor="rgba(129,140,248,0.07)",
        ), row=1, col=2)
        fig3.update_layout(
            height=260, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, margin=dict(t=30, b=10, l=0, r=0),
            font_color="rgba(255,255,255,0.5)",
        )
        fig3.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.04)", tickfont_size=10)
        fig3.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.04)", tickfont_size=10)
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

with col_ret:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🔄 Retraining Log</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='glow-line' style='background:linear-gradient(90deg,#f59e0b,#ef4444,transparent);'></div>",
        unsafe_allow_html=True,
    )

    if perf_df.empty:
        st.markdown(
            "<div style='color:rgba(255,255,255,0.25);font-size:13px;padding:20px 0;'>No data yet.</div>",
            unsafe_allow_html=True,
        )
    else:
        retrain_df = perf_df[perf_df["retrain_triggered"] == True].copy()
        if retrain_df.empty:
            st.markdown(
                "<div style='display:flex;align-items:center;gap:10px;padding:16px 0;'>"
                "<div style='font-size:24px;'>✅</div>"
                "<div style='font-size:13px;color:rgba(255,255,255,0.45);'>"
                "Model within threshold.<br>No retraining needed.</div></div>",
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
            f"<div style='margin-top:12px;font-size:11px;color:rgba(255,255,255,0.25);'>"
            f"Total retrains: <b style='color:rgba(255,255,255,0.5);'>{n}</b></div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
