# AirCast

**Live Dashboard → [aircast-abad.streamlit.app](https://aircast-abad.streamlit.app)**

A self-correcting AQI (Air Quality Index) forecasting system for Ahmedabad, India.  
Every day it fetches real AQI data, compares it against what it predicted the day before,  
logs the error, and retrains itself automatically if accuracy drops below threshold.

---

## Architecture

```
GitHub Actions (cron 23:30 UTC daily = 05:00 IST)
    │
    ▼
scheduler/daily_job.py
    ├── 1. Fetch yesterday's actual AQI  (WAQI API → Supabase: actuals)
    ├── 2. Load yesterday's prediction   (Supabase: predictions)
    ├── 3. Compute MAE / RMSE / MAPE     (Supabase: model_performance)
    ├── 4. Backfill missing performance rows from prediction/actual overlaps
    ├── 5. If rolling MAE (7d) > threshold → retrain XGBoost
    ├── 6. Push retrained model to Hugging Face Hub
    └── 7. Generate tomorrow's prediction (calibration + adaptive blending)

Streamlit Dashboard (Streamlit Community Cloud)
    ├── Today's AQI prediction + health category
    ├── Predicted vs Actual chart (last 30 days)
    ├── MAE / RMSE / MAPE accuracy panel
    └── Retraining history table
```

---

## Tech Stack

| Layer | Tool | Cost |
|---|---|---|
| Language | Python 3.11+ | Free |
| ML Model | XGBoost + Optuna tuning | Free |
| Database | Supabase (PostgreSQL) | Free (500MB) |
| Model Storage | Hugging Face Hub | Free |
| Scheduler | GitHub Actions | Free (2000 min/month) |
| Dashboard | Streamlit Community Cloud | Free |
| Data Source | WAQI API | Free (1000 req/day) |

---

## Project Structure

```
AirCast/
├── data/
│   ├── raw/                    # Downloaded Kaggle CSVs (gitignored)
│   └── processed/              # Feature-engineered dataset (gitignored)
├── pipeline/
│   ├── fetch_data.py           # WAQI API client
│   ├── db.py                   # Supabase helper functions
│   ├── preprocess.py           # Feature engineering
│   ├── train.py                # XGBoost training + Optuna
│   ├── evaluate.py             # MAE, RMSE, MAPE metrics
│   └── model_store.py          # Hugging Face Hub push/pull
├── scheduler/
│   └── daily_job.py            # Main daily orchestrator
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── database/
│   └── schema.sql              # Supabase table definitions
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   └── 02_xgboost.ipynb        # Model training walkthrough
├── .github/
│   └── workflows/
│       └── daily_job.yml       # GitHub Actions cron job
├── config.py                   # All constants & env loading
├── requirements.txt            # Streamlit Cloud deps
├── requirements-actions.txt    # GitHub Actions deps
├── .env.example
└── README.md
```

---

## Setup Guide

### 1. Clone & install

```bash
git clone https://github.com/AbhishekLim17/AirCast.git
cd AirCast
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in your keys in .env (never commit this file)
```

Required keys:
- `WAQI_API_TOKEN` — free at [aqicn.org/api](https://aqicn.org/api/)
- `SUPABASE_URL` + `SUPABASE_KEY` — from your [Supabase](https://supabase.com) project
- `HF_TOKEN` + `HF_USERNAME` — from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- `HF_REPO_NAME` — target Hugging Face model repo name
- `RETRAIN_MAE_THRESHOLD` — rolling MAE trigger for automatic retraining

### 3. Set up database

In your Supabase project → SQL Editor → New Query → paste and run `database/schema.sql`.

### 4. Add GitHub Secrets

In your GitHub repo → Settings → Secrets and variables → Actions, add the same keys from `.env`.

### 5. Run manually

```bash
# Test API fetch
python pipeline/fetch_data.py

# Run full daily job locally
python -m scheduler.daily_job
```

### 6. Launch dashboard locally

```bash
streamlit run dashboard/app.py
```

---

## AQI Health Categories (CPCB India)

| AQI Range | Category | Color |
|---|---|---|
| 0 – 50 | Good | Green |
| 51 – 100 | Satisfactory | Light Green |
| 101 – 200 | Moderate | Amber |
| 201 – 300 | Poor | Orange |
| 301 – 400 | Very Poor | Red |
| 401 – 500 | Severe | Dark Red |

---

## Model Performance

Metrics are live and change daily because the system self-corrects and retrains automatically.

- See real-time MAE / RMSE / MAPE on the dashboard: [aircast-abad.streamlit.app](https://aircast-abad.streamlit.app)
- For current values, treat dashboard numbers as the source of truth

---

## Development Phases

- [x] Phase 1 — Foundation & Environment Setup
- [x] Phase 2 — Data Layer (Fetch + Store)
- [x] Phase 3 — Feature Engineering & Historical Data
- [x] Phase 4 — Model Training (XGBoost)
- [x] Phase 5 — Self-Correction Daily Job
- [x] Phase 6 — Streamlit Dashboard
- [x] Phase 7 — Integration Testing
- [x] Final Phase — Cleanup & Release

---

## License

MIT
