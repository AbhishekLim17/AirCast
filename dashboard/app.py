"""
dashboard/app.py
AirCast — Air Quality Intelligence Dashboard
"""

import sys
import html as html_lib
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from datetime import date, timedelta

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import AQI_CATEGORIES, get_aqi_category, PRIMARY_STATION, RETRAIN_MAE_THRESHOLD
from pipeline.db import (
    get_actuals,
    get_predictions,
    get_performance_history,
    get_joined_chart_data,
)
from pipeline.fetch_data import fetch_current_aqi

# ─── Helpers ──────────────────────────────────────────────────────────────────

def badge_text_color(hex_color: str) -> str:
    """Return #fff or #1a1a1a based on relative luminance of the background."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#ffffff" if luminance < 0.55 else "#1a1a1a"

# ─── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AirCast — Air Quality Intelligence",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Theme definitions ────────────────────────────────────────────────────────

THEMES = {
    "Light": {
        "bg_gradient":        "linear-gradient(135deg,#e0e7ff 0%,#ede9fe 30%,#e0f2fe 65%,#f0fdf4 100%)",
        "glass":              "rgba(255,255,255,0.78)",
        "glass_border":       "rgba(255,255,255,0.90)",
        "glass_top_border":   "rgba(255,255,255,0.97)",
        "text":               "#0f172a",
        "muted":              "#64748b",
        "accent":             "#4f46e5",
        "sidebar_bg":         "rgba(255,255,255,0.82)",
        "sidebar_text":       "#1e293b",
        "monitor_bg":         "rgba(240,253,244,0.9)",
        "monitor_border":     "rgba(34,197,94,0.35)",
        "monitor_text":       "#166534",
        "input_bg":           "rgba(255,255,255,0.6)",
        "chart_grid":         "rgba(100,116,139,0.15)",
        "chart_tick":         "#64748b",
        "card_text":          "#0f172a",
        "empty_text":         "#94a3b8",
        "info_bg":            "rgba(239,246,255,0.9)",
        "info_border":        "rgba(99,102,241,0.25)",
        "info_text":          "#3730a3",
        "retrain_total_text": "#0f172a",
    },
    "Dark": {
        "bg_gradient":        "linear-gradient(135deg,#0f172a 0%,#111827 35%,#1f2937 70%,#0b1120 100%)",
        "glass":              "rgba(15,23,42,0.80)",
        "glass_border":       "rgba(148,163,184,0.18)",
        "glass_top_border":   "rgba(148,163,184,0.30)",
        "text":               "#e2e8f0",
        "muted":              "#94a3b8",
        "accent":             "#818cf8",
        "sidebar_bg":         "rgba(2,6,23,0.88)",
        "sidebar_text":       "#e2e8f0",
        "monitor_bg":         "rgba(20,83,45,0.45)",
        "monitor_border":     "rgba(74,222,128,0.5)",
        "monitor_text":       "#bbf7d0",
        "input_bg":           "rgba(30,41,59,0.7)",
        "chart_grid":         "rgba(148,163,184,0.12)",
        "chart_tick":         "#94a3b8",
        "card_text":          "#e2e8f0",
        "empty_text":         "#64748b",
        "info_bg":            "rgba(30,27,75,0.7)",
        "info_border":        "rgba(129,140,248,0.3)",
        "info_text":          "#a5b4fc",
        "retrain_total_text": "#e2e8f0",
    },
}

# ─── Session state init ───────────────────────────────────────────────────────

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Light"

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:2px 0 4px 0;'>
      <div style='font-size:30px;filter:drop-shadow(0 2px 8px rgba(99,102,241,0.35));'>🌤️</div>
      <div style='font-size:18px;font-weight:800;letter-spacing:-0.5px;
      margin-top:3px;font-family:Inter,sans-serif;'>AirCast</div>
      <div style='font-size:9px;color:#6366f1;letter-spacing:3px;text-transform:uppercase;
      font-weight:600;margin-top:1px;font-family:Inter,sans-serif;'>Air Quality Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── Theme toggle (switch instead of radio) ─────────────────────────────────
    is_dark_toggle = st.toggle(
        "🌙 Dark Mode",
        value=(st.session_state.theme_mode == "Dark"),
        key="sidebar_theme_toggle",
    )
    chosen = "Dark" if is_dark_toggle else "Light"
    if chosen != st.session_state.theme_mode:
        st.session_state.theme_mode = chosen
        st.rerun()

    st.divider()

    days_range = st.select_slider(
        "📅 History window",
        options=[7, 14, 30, 60, 90],
        value=30,
    )
    st.divider()

    # ── AQI scale legend ───────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:10px;color:#6366f1;letter-spacing:2px;font-weight:700;"
        "text-transform:uppercase;margin-bottom:10px;font-family:Inter,sans-serif;'>"
        "AQI Scale · CPCB India</div>",
        unsafe_allow_html=True,
    )
    for cat in AQI_CATEGORIES:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:6px;"
            f"background:rgba(255,255,255,0.55);backdrop-filter:blur(8px);"
            f"border:1px solid rgba(255,255,255,0.8);border-radius:10px;padding:6px 10px;'>"
            f"<div style='width:10px;height:10px;border-radius:50%;flex-shrink:0;"
            f"background:{cat['color']};box-shadow:0 0 8px {cat['color']}88;'></div>"
            f"<div style='font-size:12px;font-weight:600;font-family:Inter,sans-serif;'>"
            f"{cat['label']} "
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

theme = THEMES[st.session_state.theme_mode]
is_dark = st.session_state.theme_mode == "Dark"

# ─── CSS injection ────────────────────────────────────────────────────────────

# Build CSS with Python f-string (avoids Template $ conflicts with CSS variables)
css = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; }}

/* ── Remove default Streamlit top chrome & excess padding ──────────────────── */
[data-testid="stHeader"]      {{ display: none !important; }}
[data-testid="stDecoration"]  {{ display: none !important; }}
[data-testid="stToolbar"]     {{ display: none !important; }}
/* tighten the main block top padding (was 4-6 rem by default) */
.block-container {{
    padding-top: 0.6rem !important;
    padding-bottom: 1rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
}}

/* ── Background ────────────────────────────────────────────────────────────── */
.stApp {{
    background: {theme["bg_gradient"]};
    background-attachment: fixed;
}}
#MainMenu, footer {{ visibility: hidden; }}

/* ── Glass cards ───────────────────────────────────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {theme["glass"]} !important;
    backdrop-filter: blur(24px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(24px) saturate(180%) !important;
    border: 1px solid {theme["glass_border"]} !important;
    border-top: 1.5px solid {theme["glass_top_border"]} !important;
    border-radius: 20px !important;
    box-shadow:
        0 8px 32px rgba(99,102,241,0.09),
        0 2px 8px rgba(0,0,0,0.04),
        inset 0 1px 0 rgba(255,255,255,0.9) !important;
    transition: box-shadow 0.25s ease, transform 0.2s ease !important;
}}
[data-testid="stVerticalBlockBorderWrapper"]:hover {{
    box-shadow:
        0 14px 44px rgba(99,102,241,0.14),
        0 4px 12px rgba(0,0,0,0.06),
        inset 0 1px 0 rgba(255,255,255,0.9) !important;
    transform: translateY(-1px);
}}

/* ── Typography helpers ────────────────────────────────────────────────────── */
.section-title {{
    font-size: 10px; font-weight: 700; letter-spacing: 2.5px;
    text-transform: uppercase; color: {theme["accent"]};
    margin-bottom: 2px; font-family: 'Inter', sans-serif;
}}
.card-title {{
    font-size: 16px; font-weight: 700; color: {theme["card_text"]};
    margin-bottom: 0; font-family: 'Inter', sans-serif;
}}
.glow-line {{ height: 2px; border-radius: 2px; margin: 8px 0 14px 0; opacity: 0.65; }}

/* ── Blinking live dot ─────────────────────────────────────────────────────── */
.dot {{
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    margin-right: 6px; animation: blink 2s ease-in-out infinite;
}}
@keyframes blink {{
    0%,100% {{ opacity:1; transform:scale(1); }}
    50%      {{ opacity:0.35; transform:scale(0.8); }}
}}

/* ── Header bar ────────────────────────────────────────────────────────────── */
.aircast-header {{
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 10px; padding: 8px 16px;
    background: {theme["glass"]};
    backdrop-filter: blur(24px) saturate(180%);
    -webkit-backdrop-filter: blur(24px) saturate(180%);
    border: 1px solid {theme["glass_border"]};
    border-top: 1.5px solid {theme["glass_top_border"]};
    border-radius: 16px; margin-bottom: 12px;
    box-shadow: 0 4px 20px rgba(99,102,241,0.08);
}}

/* ── Stat pills ────────────────────────────────────────────────────────────── */
.stat-pill {{
    display: inline-flex; flex-direction: column; align-items: center;
    background: linear-gradient(145deg,rgba(238,242,255,0.25),rgba(255,255,255,0.2));
    border: 1px solid rgba(79,70,229,0.18);
    border-top: 1.5px solid rgba(255,255,255,0.97);
    border-radius: 14px; padding: 12px 16px; min-width: 80px;
    box-shadow: 0 2px 14px rgba(79,70,229,0.12); flex: 1;
}}
.stat-value {{ font-size: 24px; font-weight: 800; color: {theme["text"]}; letter-spacing: -0.5px; }}
.stat-label {{ font-size: 9px; font-weight: 700; letter-spacing: 2px;
               text-transform: uppercase; color: {theme["accent"]}; margin-top: 3px; }}

/* ── Info box override for dark mode ───────────────────────────────────────── */
[data-testid="stAlert"] {{
    background: {theme["info_bg"]} !important;
    border-color: {theme["info_border"]} !important;
    color: {theme["info_text"]} !important;
    border-radius: 12px !important;
}}

/* ── Sidebar ────────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background: {theme["sidebar_bg"]} !important;
    backdrop-filter: blur(28px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(180%) !important;
    border-right: 1px solid rgba(255,255,255,0.5) !important;
    box-shadow: 4px 0 28px rgba(99,102,241,0.07) !important;
}}
[data-testid="stSidebar"] * {{ color: {theme["sidebar_text"]} !important; }}
/* Remove extra padding at the top of sidebar content */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
    padding-top: 0.3rem !important;
}}
[data-testid="stSidebar"] .block-container {{
    padding-top: 0 !important;
}}

/* ── Sidebar re-open button — MUST always remain visible & clickable ─────── */
/* Handles both expanded (collapse arrow) and collapsed (re-open arrow) */
button[data-testid="stSidebarNavCollapseButton"],
button[data-testid="stSidebarNavExpandButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {{
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    pointer-events: all !important;
    z-index: 999999 !important;
    position: relative !important;
}}
/* Style the expand button when sidebar is collapsed */
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button {{
    background: {theme["glass"]} !important;
    backdrop-filter: blur(20px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
    border: 1.5px solid {theme["accent"]} !important;
    border-radius: 0 14px 14px 0 !important;
    box-shadow: 4px 0 24px rgba(99,102,241,0.25) !important;
    padding: 18px 10px !important;
    color: {theme["accent"]} !important;
    transition: all 0.2s ease !important;
    height: auto !important;
    min-height: 52px !important;
    min-width: 32px !important;
}}
[data-testid="stSidebarCollapsedControl"] button:hover,
[data-testid="collapsedControl"] button:hover {{
    background: rgba(99,102,241,0.15) !important;
    box-shadow: 4px 0 32px rgba(99,102,241,0.35) !important;
    transform: scale(1.05);
}}

/* ── Radio / slider inputs ─────────────────────────────────────────────────── */
[data-testid="stSlider"] > div > div > div > div {{
    background: linear-gradient(90deg,#4f46e5,#7c3aed) !important;
}}
div[data-baseweb="radio"] label {{ color: {theme["text"]} !important; }}

/* ── Buttons ───────────────────────────────────────────────────────────────── */
div[data-testid="stButton"] button {{
    background: {theme["glass"]} !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid {theme["glass_border"]} !important;
    border-radius: 20px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    color: {theme["text"]} !important;
    padding: 5px 16px !important;
    font-size: 13px !important;
    box-shadow: 0 2px 10px rgba(99,102,241,0.10) !important;
    transition: all 0.2s ease !important;
}}
div[data-testid="stButton"] button:hover {{
    box-shadow: 0 4px 18px rgba(99,102,241,0.22) !important;
    transform: translateY(-1px) !important;
}}

/* ── Data table ──────────────────────────────────────────────────────────────── */
.stDataFrame {{
    border-radius: 14px !important;
    overflow: hidden !important;
    {'filter: invert(0.92) hue-rotate(180deg);' if is_dark else ''}
}}

/* ── Scrollbar ───────────────────────────────────────────────────────────────── */
hr {{ border-color: rgba(99,102,241,0.15) !important; margin: 10px 0 !important; }}
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: rgba(99,102,241,0.04); }}
::-webkit-scrollbar-thumb {{ background: rgba(99,102,241,0.25); border-radius: 4px; }}

/* ── Responsive ──────────────────────────────────────────────────────────────── */
@media (max-width: 900px) {{
    [data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 16px !important; }}
    .stat-value {{ font-size: 20px; }}
    .stat-pill  {{ padding: 10px 12px; min-width: unset; }}
}}
@media (max-width: 600px) {{
    [data-testid="stHorizontalBlock"] > div {{
        min-width: 100% !important; flex: 0 0 100% !important;
    }}
    [data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 14px !important; }}
    .aircast-header {{ padding: 8px 12px; border-radius: 12px; }}
    .card-title     {{ font-size: 14px; }}
    .stat-pill      {{ padding: 10px 10px; min-width: 68px; }}
    .stat-value     {{ font-size: 18px; }}
    [data-testid="stSidebar"] {{ min-width: 260px !important; max-width: 80vw !important; }}
}}
"""

st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

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
    # Vivid colour bands so each AQI zone is clearly visible on the gauge
    steps = [
        {"range": [0,   50],  "color": "rgba(22,163,74,0.30)"},     # Green  — Good
        {"range": [51,  100], "color": "rgba(132,204,22,0.28)"},    # Lime   — Satisfactory
        {"range": [101, 200], "color": "rgba(234,179,8,0.28)"},     # Amber  — Moderate
        {"range": [201, 300], "color": "rgba(249,115,22,0.30)"},    # Orange — Poor
        {"range": [301, 400], "color": "rgba(239,68,68,0.30)"},     # Red    — Very Poor
        {"range": [401, 500], "color": "rgba(127,29,29,0.35)"},     # Maroon — Severe
    ]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"size": 46, "color": color, "family": "Inter, sans-serif"}},
        title={
            "text": (
                f"<b style='color:{theme['text']}'>{title}</b>"
                f"<br><span style='font-size:12px;color:{color};font-weight:700'>"
                f"{cat['label']}</span>"
            ),
            "font": {"size": 14, "family": "Inter, sans-serif"},
        },
        gauge={
            "axis": {
                "range": [0, 500], "tickwidth": 1,
                "tickcolor": theme["chart_tick"],
                "tickfont": {"color": theme["chart_tick"], "size": 10},
            },
            "bar": {"color": color, "thickness": 0.22},
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
        margin=dict(t=44, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=theme["text"],
    )
    return fig

# ─── Forecast chart ───────────────────────────────────────────────────────────

def make_forecast_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    # AQI band backgrounds
    for lo, hi, rgba in [
        (0,   50,  "rgba(16,185,129,0.06)"),
        (51,  100, "rgba(132,204,22,0.06)"),
        (101, 200, "rgba(234,179,8,0.06)"),
        (201, 300, "rgba(249,115,22,0.07)"),
        (301, 400, "rgba(239,68,68,0.07)"),
        (401, 500, "rgba(127,29,29,0.07)"),
    ]:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=rgba, line_width=0, layer="below")

    if "actual_aqi" in df.columns and df["actual_aqi"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["actual_aqi"],
            mode="lines+markers", name="Actual AQI",
            line=dict(color="#0ea5e9", width=2.5, shape="spline"),
            marker=dict(size=6, color="#0ea5e9", line=dict(color="white", width=1.5)),
            fill="tozeroy", fillcolor="rgba(14,165,233,0.07)",
            hovertemplate="<b>Actual AQI</b>: %{y:.0f}<br>%{x|%d %b %Y}<extra></extra>",
        ))
    if "predicted" in df.columns and df["predicted"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["predicted"],
            mode="lines+markers", name="Predicted AQI",
            line=dict(color="#8b5cf6", width=2, dash="dot", shape="spline"),
            marker=dict(size=6, symbol="diamond", color="#8b5cf6",
                        line=dict(color="white", width=1.5)),
            hovertemplate="<b>Predicted AQI</b>: %{y:.0f}<br>%{x|%d %b %Y}<extra></extra>",
        ))

    n_ticks = min(len(df), 8)
    fig.update_layout(
        height=340, template="none",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        # Disable all zooming / panning interactions
        dragmode=False,
        xaxis=dict(
            title=dict(text="Date", font=dict(color=theme["muted"], size=12, family="Inter")),
            showgrid=True, gridcolor=theme["chart_grid"], zeroline=False,
            tickfont=dict(color=theme["chart_tick"], size=11, family="Inter"),
            tickformat="%d %b",
            nticks=n_ticks,
            tickangle=-30,
            showline=True, linecolor=theme["chart_grid"],
            automargin=True,
            # Disable zoom on this axis
            fixedrange=True,
        ),
        yaxis=dict(
            title=dict(text="AQI", font=dict(color=theme["muted"], size=12, family="Inter")),
            showgrid=True, gridcolor=theme["chart_grid"], zeroline=False,
            range=[0, 520],
            dtick=100,
            tickfont=dict(color=theme["chart_tick"], size=11, family="Inter"),
            showline=True, linecolor=theme["chart_grid"],
            automargin=True,
            fixedrange=True,
        ),
        legend=dict(
            orientation="h", y=-0.22, x=0.5, xanchor="center",
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=theme["text"], size=12, family="Inter"),
        ),
        margin=dict(t=16, b=80, l=64, r=20),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.96)" if not is_dark else "rgba(15,23,42,0.96)",
            font_color=theme["text"],
            bordercolor="rgba(99,102,241,0.3)", font_family="Inter",
        ),
    )
    return fig

# ─── Accuracy chart ───────────────────────────────────────────────────────────

def make_accuracy_chart(perf_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["MAE over time", "MAPE (%) over time"],
        horizontal_spacing=0.14,
    )
    fig.add_trace(go.Scatter(
        x=perf_df["eval_date"], y=perf_df["mae"],
        mode="lines+markers", name="MAE",
        line=dict(color="#6366f1", width=2.5, shape="spline"),
        marker=dict(size=5, color="#6366f1", line=dict(color="white", width=1.5)),
        fill="tozeroy", fillcolor="rgba(99,102,241,0.07)",
        hovertemplate="<b>MAE</b>: %{y:.2f}<br>%{x|%d %b}<extra></extra>",
    ), row=1, col=1)

    fig.add_hline(
        y=RETRAIN_MAE_THRESHOLD, line_dash="dash", line_color="rgba(249,115,22,0.8)",
        annotation_text=f"Retrain if MAE > {RETRAIN_MAE_THRESHOLD}",
        annotation_font_size=9, annotation_font_color="#c2410c",
        annotation_position="top left",
        row=1, col=1,
    )

    if "mape" in perf_df.columns and perf_df["mape"].notna().any():
        fig.add_trace(go.Scatter(
            x=perf_df["eval_date"], y=perf_df["mape"],
            mode="lines+markers", name="MAPE %",
            line=dict(color="#0ea5e9", width=2.5, shape="spline"),
            marker=dict(size=5, color="#0ea5e9", line=dict(color="white", width=1.5)),
            fill="tozeroy", fillcolor="rgba(14,165,233,0.07)",
            hovertemplate="<b>MAPE</b>: %{y:.1f}%<br>%{x|%d %b}<extra></extra>",
        ), row=1, col=2)

    fig.update_layout(
        height=280, template="none",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(t=40, b=72, l=64, r=20),
        dragmode=False,
        font=dict(color=theme["muted"], family="Inter, sans-serif"),
    )
    for anno in fig.layout.annotations:
        anno.font.color = theme["muted"]
        anno.font.size  = 11

    axis_style = dict(
        showgrid=True, gridcolor=theme["chart_grid"],
        tickfont=dict(size=10, color=theme["chart_tick"]),
        showline=True, linecolor=theme["chart_grid"],
        automargin=True, fixedrange=True,
        tickangle=-30, nticks=6,
        title_font=dict(color=theme["muted"], size=11),
    )
    fig.update_xaxes(**axis_style, tickformat="%d %b",
                     title_text="Date")
    fig.update_yaxes(
        showgrid=True, gridcolor=theme["chart_grid"],
        tickfont=dict(size=10, color=theme["chart_tick"]),
        showline=True, linecolor=theme["chart_grid"],
        automargin=True, fixedrange=True,
        title_font=dict(color=theme["muted"], size=11),
    )
    fig.update_yaxes(title_text="MAE (AQI points)", row=1, col=1)
    fig.update_yaxes(title_text="MAPE (%)", row=1, col=2)
    return fig

# ─── Load data ────────────────────────────────────────────────────────────────

live_data = load_live()
tmr_pred  = load_tomorrow_prediction()
chart_df  = load_chart_data(days=days_range)
perf_df   = load_performance(days=days_range)

live_aqi = float(live_data.get("aqi")) if live_data and "aqi" in live_data else None
live_cat = get_aqi_category(live_aqi) if live_aqi is not None else None

# ─── Header bar ───────────────────────────────────────────────────────────────

# AQI category colour drives both the badge and the live dot for this reading
dot_color  = live_cat["color"] if live_cat else "#22c55e"   # green fallback = "Good"
cat_label  = live_cat["label"] if live_cat else "Loading…"
live_value = f"{live_aqi:.0f}" if live_aqi is not None else "—"
badge_fg   = badge_text_color(dot_color)

st.markdown(
    f"<div class='aircast-header'>"
    f"<div style='display:flex;align-items:center;gap:12px;'>"
    f"<div style='font-size:28px;filter:drop-shadow(0 2px 8px rgba(99,102,241,0.35));'>🌤️</div>"
    f"<div>"
    f"<div style='font-size:20px;font-weight:800;color:{theme['text']};letter-spacing:-0.5px;"
    f"line-height:1.15;font-family:Inter,sans-serif;'>AirCast</div>"
    f"<div style='font-size:11px;color:#6366f1;font-weight:500;"
    f"font-family:Inter,sans-serif;'>Ahmedabad · Air Quality Intelligence</div>"
    f"</div></div>"
    f"<div style='display:flex;align-items:center;gap:10px;flex-wrap:wrap;'>"
    # Live monitoring pill — always green (system is live)
    f"<div style='font-size:12px;color:{theme['monitor_text']};background:{theme['monitor_bg']};"
    f"border:1px solid {theme['monitor_border']};border-radius:20px;padding:4px 12px;"
    f"font-family:Inter,sans-serif;font-weight:500;'>"
    f"<span class='dot' style='background:#22c55e;'></span>Live</div>"
    # AQI badge — colour matches the current AQI category
    f"<div style='font-size:12px;font-weight:700;color:{badge_fg};"
    f"background:{dot_color};border-radius:20px;padding:4px 14px;"
    f"font-family:Inter,sans-serif;box-shadow:0 2px 12px {dot_color}55;'>"
    f"AQI {live_value} · {cat_label}</div>"
    f"</div></div>",
    unsafe_allow_html=True,
)

# ─── Row 1: Two gauges + stats ────────────────────────────────────────────────

col_g1, col_g2, col_stats = st.columns([1, 1, 1.4])

with col_g1:
    with st.container(border=True):
        st.markdown(
            "<div class='section-title'>🟢 Live Reading</div>"
            "<div class='card-title'>Current AQI</div>"
            "<div class='glow-line' style='background:linear-gradient(90deg,#0ea5e9,#38bdf8,transparent);'></div>",
            unsafe_allow_html=True,
        )
        if live_aqi is not None:
            st.plotly_chart(
                make_gauge(live_aqi, "Now"),
                use_container_width=True,
                config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False, "staticPlot": True},
            )
            dominant = html_lib.escape(
                (live_data.get("dominant_pollutant") or "N/A").upper()
            )
            st.markdown(
                f"<div style='text-align:center;font-size:12px;color:{theme['muted']};"
                f"font-family:Inter,sans-serif;margin-top:-8px;'>"
                f"Main pollutant: <b style='color:{theme['text']};'>{dominant}</b></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='color:{theme['empty_text']};padding:60px 0;text-align:center;"
                f"font-size:13px;font-family:Inter,sans-serif;'>"
                f"⚠️ <b>Live air quality is unavailable right now.</b><br>"
                f"<span style='font-size:12px;'>"
                f"This usually means the WAQI API is temporarily down or your internet "
                f"is disconnected. Try refreshing the page in a few minutes.</span></div>",
                unsafe_allow_html=True,
            )

with col_g2:
    with st.container(border=True):
        st.markdown(
            "<div class='section-title'>🔮 Tomorrow's Forecast</div>"
            "<div class='card-title'>Predicted AQI</div>"
            "<div class='glow-line' style='background:linear-gradient(90deg,#8b5cf6,#a78bfa,transparent);'></div>",
            unsafe_allow_html=True,
        )
        if tmr_pred is not None:
            st.plotly_chart(
                make_gauge(tmr_pred, "Tomorrow"),
                use_container_width=True,
                config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False, "staticPlot": True},
            )
            tmr_date = (date.today() + timedelta(days=1)).strftime("%A, %d %b %Y")
            st.markdown(
                f"<div style='text-align:center;font-size:12px;color:{theme['muted']};"
                f"font-family:Inter,sans-serif;margin-top:-8px;'>{tmr_date}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='color:{theme['empty_text']};padding:60px 0;text-align:center;"
                f"font-size:13px;font-family:Inter,sans-serif;'>"
                f"🔮 <b>Tomorrow's forecast isn't ready yet.</b><br>"
                f"<span style='font-size:11px;'>"
                f"The AI model generates a new prediction every morning. "
                f"Check back later today or after the next daily update runs.</span></div>",
                unsafe_allow_html=True,
            )

with col_stats:
    with st.container(border=True):
        st.markdown(
            "<div class='section-title'>📊 Model Accuracy</div>"
            "<div class='card-title'>How well is the AI predicting?</div>"
            "<div class='glow-line' style='background:linear-gradient(90deg,#6366f1,#8b5cf6,#ec4899,transparent);'></div>",
            unsafe_allow_html=True,
        )
        if not perf_df.empty:
            latest = perf_df.iloc[-1]
            mae  = latest.get("mae",  None)
            rmse = latest.get("rmse", None)
            mape = latest.get("mape", None)

            # Friendly interpretations
            mae_tip  = f"{mae:.1f} AQI points average error"   if mae  is not None else "—"
            rmse_tip = f"{rmse:.1f} (penalises big misses)"    if rmse is not None else "—"
            mape_tip = f"{mape:.1f}% off on average"           if mape is not None else "—"

            pills = [
                (f"{mae:.1f}"   if mae  is not None else "—", "MAE",  mae_tip),
                (f"{rmse:.1f}"  if rmse is not None else "—", "RMSE", rmse_tip),
                (f"{mape:.1f}%" if mape is not None else "—", "MAPE", mape_tip),
            ]
            pill_html = "<div style='display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px;'>"
            for val, lbl, tip in pills:
                pill_html += (
                    f"<div class='stat-pill' title='{tip}'>"
                    f"<div class='stat-value'>{val}</div>"
                    f"<div class='stat-label'>{lbl}</div>"
                    f"</div>"
                )
            pill_html += "</div>"
            st.markdown(pill_html, unsafe_allow_html=True)

            # Plain-English guide
            st.markdown(
                f"<div style='background:{theme['info_bg']};border:1px solid {theme['info_border']};"
                f"border-radius:10px;padding:10px 12px;font-size:12px;color:{theme['info_text']};"
                f"font-family:Inter,sans-serif;line-height:1.6;'>"
                f"<b>How to read this:</b><br>"
                f"• <b>MAE</b> = average number of AQI points the prediction is off by <i>(lower = better)</i><br>"
                f"• <b>RMSE</b> = similar to MAE but punishes large errors more heavily<br>"
                f"• <b>MAPE</b> = error as a percentage of the real value <i>(lower = better)</i>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Mini MAE sparkline
            fig_spark = go.Figure()
            fig_spark.add_trace(go.Scatter(
                x=perf_df["eval_date"], y=perf_df["mae"],
                mode="lines", fill="tozeroy",
                line=dict(color="#6366f1", width=2),
                fillcolor="rgba(99,102,241,0.08)",
                hovertemplate="MAE: %{y:.2f}<br>%{x|%d %b}<extra></extra>",
            ))
            fig_spark.add_hline(
                y=RETRAIN_MAE_THRESHOLD, line_dash="dot", line_color="rgba(249,115,22,0.8)",
                annotation_text="retrain limit",
                annotation_font_size=8,
                annotation_font_color="#c2410c",
            )
            fig_spark.update_layout(
                height=90, margin=dict(t=4, b=4, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                dragmode=False,
                xaxis=dict(visible=False, fixedrange=True),
                yaxis=dict(visible=False, fixedrange=True),
                showlegend=False,
            )
            st.plotly_chart(fig_spark, use_container_width=True,
                            config={"displayModeBar": False, "staticPlot": True})
            st.markdown(
                f"<div style='font-size:10px;color:{theme['muted']};text-align:right;"
                f"margin-top:-6px;font-family:Inter,sans-serif;'>MAE trend ↑ bad · ↓ good</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='color:{theme['empty_text']};font-size:13px;padding:20px 0;"
                f"font-family:Inter,sans-serif;'>"
                f"📊 <b>No accuracy data yet.</b><br>"
                f"<span style='font-size:12px;'>"
                f"Once the model has made at least one prediction and the next day's real AQI "
                f"has been collected, accuracy stats will automatically appear here.</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

# ─── Row 2: Forecast vs reality chart ────────────────────────────────────────

with st.container(border=True):
    st.markdown(
        f"<div style='display:flex;align-items:flex-start;justify-content:space-between;"
        f"flex-wrap:wrap;gap:8px;margin-bottom:4px;'>"
        f"<div>"
        f"<div class='section-title'>📈 Forecast vs Reality</div>"
        f"<div class='card-title'>Predicted vs Actual AQI — Last {days_range} days</div>"
        f"</div>"
        f"<div style='font-size:11px;color:#6366f1;font-weight:600;"
        f"background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);"
        f"border-radius:8px;padding:4px 10px;font-family:Inter,sans-serif;white-space:nowrap;'>"
        f"Ahmedabad · CPCB India</div></div>"
        f"<div class='glow-line' style='background:linear-gradient(90deg,#0ea5e9,#8b5cf6,#ec4899,transparent);'></div>",
        unsafe_allow_html=True,
    )
    if chart_df.empty:
        st.markdown(
            f"<div style='color:{theme['empty_text']};font-size:13px;padding:30px 0;text-align:center;"
            f"font-family:Inter,sans-serif;'>"
            f"📈 <b>No chart data yet.</b><br>"
            f"<span style='font-size:12px;'>The chart will fill in automatically each day as the "
            f"AI makes predictions and real AQI readings are collected from the sensor network."
            f"</span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.plotly_chart(
            make_forecast_chart(chart_df),
            use_container_width=True,
            config={"displayModeBar": False, "scrollZoom": False,
                    "doubleClick": False, "staticPlot": False},
        )

# ─── Row 3: Accuracy history + Retrain log ────────────────────────────────────

col_acc, col_ret = st.columns([1.4, 1])

with col_acc:
    with st.container(border=True):
        st.markdown(
            "<div class='section-title'>📉 Accuracy History</div>"
            "<div class='card-title'>MAE & MAPE over time</div>"
            "<div class='glow-line' style='background:linear-gradient(90deg,#8b5cf6,#6366f1,transparent);'></div>",
            unsafe_allow_html=True,
        )
        if perf_df.empty:
            st.markdown(
                f"<div style='color:{theme['empty_text']};font-size:13px;padding:20px 0;"
                f"font-family:Inter,sans-serif;'>"
                f"📉 <b>Accuracy history will appear here</b> after the model has run its "
                f"first evaluation (the day after the first prediction is made).</div>",
                unsafe_allow_html=True,
            )
        else:
            st.plotly_chart(
                make_accuracy_chart(perf_df),
                use_container_width=True,
                config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
            )

with col_ret:
    with st.container(border=True):
        st.markdown(
            "<div class='section-title'>🔄 Auto-Retrain Log</div>"
            "<div class='card-title'>When did the AI re-learn?</div>"
            "<div class='glow-line' style='background:linear-gradient(90deg,#f59e0b,#ef4444,transparent);'></div>",
            unsafe_allow_html=True,
        )
        if perf_df.empty:
            st.markdown(
                f"<div style='color:{theme['empty_text']};font-size:13px;padding:20px 0;"
                f"font-family:Inter,sans-serif;'>"
                f"🔄 <b>No retraining events yet.</b><br>"
                f"<span style='font-size:12px;'>When the model's prediction error gets too high "
                f"(MAE > {RETRAIN_MAE_THRESHOLD}), the AI automatically re-trains itself on fresh "
                f"data. Each event is logged here.</span></div>",
                unsafe_allow_html=True,
            )
        else:
            retrain_mask = perf_df.get("retrain_triggered", pd.Series(dtype=bool))
            if len(retrain_mask) != len(perf_df):
                retrain_mask = pd.Series([False] * len(perf_df))
            retrain_df = perf_df[retrain_mask == True].copy()

            if retrain_df.empty:
                st.markdown(
                    "<div style='display:flex;align-items:center;gap:12px;padding:18px 0;'>"
                    "<div style='font-size:32px;'>✅</div>"
                    f"<div style='font-size:13px;color:{theme['muted']};font-family:Inter,sans-serif;'>"
                    f"<b style='color:{theme['text']};'>All good!</b><br>"
                    f"The model's error is within the acceptable limit "
                    f"(MAE ≤ {RETRAIN_MAE_THRESHOLD} AQI points).<br>"
                    f"No retraining has been needed so far.</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                dcols = [c for c in ["eval_date", "mae", "new_mae", "promoted"] if c in retrain_df.columns]
                st.dataframe(
                    retrain_df[dcols].rename(columns={
                        "eval_date": "Date",
                        "mae":       "Error Before",
                        "new_mae":   "Error After",
                        "promoted":  "Improved?",
                    }).reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True,
                )

            n = len(retrain_df)
            st.markdown(
                f"<div style='margin-top:12px;font-size:11px;color:{theme['muted']};"
                f"font-family:Inter,sans-serif;'>Total re-trains triggered: "
                f"<b style='color:{theme['retrain_total_text']};font-weight:700;'>{n}</b></div>",
                unsafe_allow_html=True,
            )

# ─── Footer spacer ────────────────────────────────────────────────────────────

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
