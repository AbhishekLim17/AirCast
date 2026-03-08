# AirCast

A self-correcting AQI (Air Quality Index) forecasting system for Ahmedabad, India.  
Every day it fetches real AQI data, compares it against what it predicted the day before,  
logs the error, and retrains itself automatically if accuracy drops below threshold.

---

## Architecture

```
GitHub Actions (cron 06:00 UTC daily)
    │
    ▼
scheduler/daily_job.py
    ├── 1. Fetch yesterday's actual AQI  (WAQI API → Supabase: actuals)
    ├── 2. Load yesterday's prediction   (Supabase: predictions)
    ├── 3. Compute MAE / RMSE / MAPE     (Supabase: model_performance)
    ├── 4. If MAE > threshold → retrain XGBoost on last 90 days
    ├── 5. If new model is better → push to Hugging Face Hub
    └── 6. Generate tomorrow's prediction → store in Supabase

Streamlit Dashboard (Streamlit Community Cloud)
    ├── Today's AQI prediction + health category
    ├── Predicted vs Actual chart (last 30 days)
    ├── MAE / MAPE accuracy panel
    └── Retraining history table
```

---

## Tech Stack

| Layer | Tool | Cost |
|---|---|---|
| Language | Python 3.11 | Free |
| ML Model | XGBoost + Optuna tuning | Free |
| Database | Supabase (PostgreSQL) | Free (500MB) |
| Model Storage | Hugging Face Hub | Free |
| Scheduler | GitHub Actions | Free (2000 min/month) |
| Dashboard | Streamlit Community Cloud | Free |
| Data Source | WAQI API | Free (1000 req/day) |

---

## Project Structure

```
AQI/
├── data/
│   ├── raw/                    # Downloaded Kaggle CSVs (gitignored)
│   └── processed/              # Feature-engineered dataset
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
├── tests/
│   ├── test_pipeline.py
│   ├── test_db.py
│   └── test_daily_job.py
├── .github/
│   └── workflows/
│       └── daily_job.yml       # GitHub Actions cron job
├── config.py                   # All constants & env loading
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup Guide

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/aqi-prediction.git
cd aqi-prediction
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

### 3. Set up database

In your Supabase project → SQL Editor → New Query → paste and run `database/schema.sql`.

### 4. Add GitHub Secrets

In your GitHub repo → Settings → Secrets and variables → Actions, add the same keys from `.env`.

### 5. Run manually

```bash
# Test API fetch
python pipeline/fetch_data.py

# Run full daily job locally
python scheduler/daily_job.py
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
| 101 – 200 | Moderate | Yellow |
| 201 – 300 | Poor | Orange |
| 301 – 400 | Very Poor | Red |
| 401 – 500 | Severe | Dark Red |

---

## Development Phases

- [x] Phase 1 — Foundation & Environment Setup
- [ ] Phase 2 — Data Layer (Fetch + Store)
- [ ] Phase 3 — Feature Engineering & Historical Data
- [ ] Phase 4 — Model Training (XGBoost)
- [ ] Phase 5 — Self-Correction Daily Job
- [ ] Phase 6 — Streamlit Dashboard
- [ ] Phase 7 — Integration Testing
- [ ] Final Phase — Cleanup & Release

---

## License

MIT
