# AirCast — Code Review & Suggestions
> Reviewed on: March 12, 2026  
> Scope: Full codebase review covering bugs, security, UX, ML improvements, and a major feature proposal.

---

## Part 1 — Existing Codebase Fixes (16 Suggestions)

---

### 1. Python Version Mismatch in CI
**File:** `.github/workflows/daily_job.yml`  
**Priority:** High | **Effort:** Trivial

**Problem:**  
GitHub Actions is configured to use Python **3.11**, but `requirements.txt` is pinned for Python **3.13** and `config.py` uses type hints like `float | None` (valid in 3.10+). This inconsistency can cause silent dependency behaviour differences or future breakage.

**Fix:**
```yaml
# Change this:
python-version: "3.11"

# To:
python-version: "3.13"
```

---

### 2. Pickle is a Security Vulnerability for Model Storage
**File:** `pipeline/model_store.py`  
**Priority:** High | **Effort:** Low

**Problem:**  
The trained XGBoost model is serialized with `pickle.dump()` and later loaded with `pickle.load()`. Pickle can execute **arbitrary code** during deserialization. Since the Hugging Face repo is set to `private=False` (public), anyone can overwrite or tamper with the model file. If a compromised pickle is loaded, it can run malicious code on the server or in GitHub Actions.

This is an OWASP Top 10 vulnerability — *A08: Software and Data Integrity Failures*.

**Fix:**
```python
# Replace pickle with joblib (already in requirements.txt):
import joblib

# Saving:
joblib.dump(model, tmp_path)

# Loading:
model = joblib.load(cached_path)

# Save metadata separately as JSON — not bundled inside the model file:
import json
with open("model_meta.json", "w") as f:
    json.dump({"feature_cols": feature_cols, "metrics": metrics}, f)
```

---

### 3. Hugging Face Model Repo is Public
**File:** `pipeline/model_store.py`  
**Priority:** Medium | **Effort:** Trivial

**Problem:**  
```python
api.create_repo(..., private=False)
```
The trained model is publicly downloadable. It encodes patterns learned from Ahmedabad AQI data, and direct public write access (if token is ever leaked) puts the pipeline at risk.

**Fix:**
```python
api.create_repo(..., private=True)
```
The existing `HF_TOKEN` already handles authenticated reads. No other change needed.

---

### 4. No Retry Logic When WAQI API Fails
**File:** `scheduler/daily_job.py`  
**Priority:** High | **Effort:** Low

**Problem:**  
If WAQI returns a network error or 5xx response, the job immediately aborts with no retry. That entire day's data is permanently lost — there is no backfill mechanism.

**Fix:**
```python
import time

def fetch_with_retry(station: str, retries: int = 3, delay: int = 5):
    for attempt in range(retries):
        data = fetch_current_aqi(station)
        if data is not None:
            return data
        logger.warning("Attempt %d failed, retrying in %ds...", attempt + 1, delay)
        time.sleep(delay)
    return None
```

---

### 5. Point-in-Time MAE is Statistically Meaningless
**File:** `scheduler/daily_job.py` — `step_evaluate()`  
**Priority:** Medium | **Effort:** Low

**Problem:**  
`step_evaluate()` computes MAE, RMSE, and MAPE on a **single data point** — one day's prediction vs actual. A single-point MAE is meaningless; it's just the absolute error of one guess. Yet this value is stored in the `model_performance` table and displayed on the dashboard as "accuracy", which misleads users.

The `step_decide_retrain()` function does this correctly (rolling 7-day window). `step_evaluate()` should do the same.

**Fix:**  
Change `step_evaluate()` to compute metrics over the last 7–30 days from `get_actuals()` and `get_predictions()` instead of a single pair of values.

---

### 6. Dashboard Cache TTL is Too Short
**File:** `dashboard/app.py`  
**Priority:** Low | **Effort:** Trivial

**Problem:**  
```python
@st.cache_data(ttl=300)  # 5 minutes
```
The data only ever changes **once per day** when the GitHub Actions job runs at 06:00 UTC. Caching for 5 minutes means Supabase receives hundreds of unnecessary read requests per day, burning through the free 500MB limit faster.

**Fix:**
```python
@st.cache_data(ttl=3600)  # 1 hour is plenty
```
Optionally add a manual "Refresh" button so users can force a fresh pull without waiting for TTL expiry.

---

### 7. No Graceful Degradation When Database is Down
**File:** `dashboard/app.py`  
**Priority:** Medium | **Effort:** Low

**Problem:**  
If Supabase is unreachable (maintenance, quota exceeded, network issue), the dashboard renders a blank/broken state with no message. Users see an empty chart with no explanation.

**Fix:**
```python
try:
    data = load_chart_data(days=days)
except Exception:
    st.warning("Data is temporarily unavailable. Please check back in a few minutes.")
    st.stop()
```

---

### 8. No "Last Updated" Timestamp on Dashboard
**File:** `dashboard/app.py`  
**Priority:** Low | **Effort:** Trivial

**Problem:**  
There is no indication of when the data was last refreshed. A user cannot tell if they are looking at today's data or data from 3 days ago. An AQI dashboard without a freshness indicator loses credibility.

**Fix:**  
Store and display the `inserted_at` / `updated_at` timestamp from Supabase in the dashboard header.

```python
st.caption(f"Last updated: {latest_row['date']} at 06:00 UTC")
```

---

### 9. Only One Station — Single Point of Failure
**File:** `config.py`  
**Priority:** Medium | **Effort:** Medium

**Problem:**  
```python
AHMEDABAD_STATIONS = ["ahmedabad"]  # only one
```
The code already supports a list of stations but only one is configured. If the `"ahmedabad"` WAQI station goes offline (which happens), the entire pipeline fails silently.

**Fix:**  
Add 1–2 fallback AUDA sub-stations. In `fetch_data.py`, iterate through stations and average valid readings, falling back to the next if one fails.

---

### 10. No Data Validation on Fetched AQI Values
**File:** `scheduler/daily_job.py`  
**Priority:** High | **Effort:** Low

**Problem:**  
```python
actual_aqi = float(data.get("aqi") or data.get("value", 0))
```
If WAQI returns `0`, `-1`, or `999` (common API error sentinels), this value is stored directly into the database and used for model training. Over time, corrupted values degrade model accuracy silently.

**Fix:**
```python
AQI_MIN, AQI_MAX = 1, 500

if not (AQI_MIN <= actual_aqi <= AQI_MAX):
    logger.warning("AQI value %.1f is out of valid range [%d, %d] — skipping.", actual_aqi, AQI_MIN, AQI_MAX)
    return None
```

---

### 11. No Seasonal / Calendar Features in the Model
**File:** `pipeline/preprocess.py`  
**Priority:** Medium | **Effort:** Medium

**Problem:**  
The model uses lag-day values and rolling averages but no calendar-based features. Ahmedabad has massive, predictable seasonal AQI patterns:
- **Diwali (Oct/Nov):** AQI spikes dramatically from fireworks
- **Winter (Dec–Feb):** Temperature inversions trap pollutants
- **Monsoon (Jun–Sep):** Rain cleans the air, AQI is lowest

Ignoring these patterns forces the model to "rediscover" them from lags alone, wasting predictive power.

**Fix:**  
In `preprocess.py`, add:
```python
df["month"]       = df.index.month
df["day_of_year"] = df.index.dayofyear
df["is_winter"]   = df.index.month.isin([12, 1, 2]).astype(int)
df["is_monsoon"]  = df.index.month.isin([6, 7, 8, 9]).astype(int)
```
These are free features with high signal — simple to add, meaningful improvement expected.

---

### 12. No Multi-Day Forecast
**Priority:** High | **Effort:** High

**Problem:**  
The system only predicts tomorrow's AQI. A user planning an outdoor event 3 days from now cannot use this app. Every weather and AQI app users compare to shows 7-day forecasts.

**Fix:**  
Run the model iteratively — predict day+1, feed that prediction back in as the lag-1 feature, predict day+2, and so on.

```python
predictions = []
seed_features = today_features.copy()

for step in range(3):  # 3-day forecast
    pred = model.predict(seed_features)[0]
    predictions.append(pred)
    # Shift lag features forward
    seed_features["lag_1"] = pred
    seed_features["lag_7"] = seed_features.get("lag_6", pred)
```

Show a 3-bar forecast on the dashboard for Day+1, Day+2, Day+3.

---

### 13. No Alerting When the Daily Job Fails
**File:** `.github/workflows/daily_job.yml`  
**Priority:** Medium | **Effort:** Low

**Problem:**  
If the GitHub Actions job fails, logs are uploaded as artifacts but no one is notified. The database silently has a gap for that day. You might not notice for days.

**Fix:**  
Add a failure notification step using a Telegram webhook (or GitHub's built-in email notification):
```yaml
- name: Notify on failure
  if: failure()
  run: |
    curl -s -X POST "https://api.telegram.org/bot${{ secrets.TELEGRAM_TOKEN }}/sendMessage" \
    -d "chat_id=${{ secrets.TELEGRAM_CHAT_ID }}" \
    -d "text=AirCast daily job FAILED on $(date)"
```

---

### 14. No Health Recommendation Text on Dashboard
**File:** `dashboard/app.py`  
**Priority:** Low | **Effort:** Low

**Problem:**  
The dashboard shows the AQI category with a color badge (Good / Moderate / Hazardous) but no actionable advice. Most users don't know what AQI 156 means for their health.

**Fix:**  
Add CPCB-standard recommendations below the AQI badge:

| Category | Advice |
|---|---|
| Good (0–50) | Air quality is satisfactory. Enjoy outdoor activities. |
| Satisfactory (51–100) | Unusually sensitive groups should consider limiting prolonged outdoor exertion. |
| Moderate (101–200) | Sensitive groups (children, elderly, patients) should avoid outdoor exertion. |
| Poor (201–300) | Everyone should limit outdoor activities. Sensitive groups stay indoors. |
| Very Poor (301–400) | Everyone should avoid outdoor exertion. Keep windows closed. |
| Severe (401–500) | Everyone should stay indoors. Avoid all outdoor activity. |

---

### 15. Model Feature Columns Stored Inside the Serialized File
**File:** `pipeline/model_store.py`  
**Priority:** Medium | **Effort:** Low

**Problem:**  
```python
bundle = {"model": model, "feature_cols": feature_cols, "metrics": metrics}
pickle.dump(bundle, tmp)
```
Feature column names are bundled inside the serialized file. If feature engineering changes (e.g., adding seasonal features from Suggestion #11), old cached models on Hugging Face will have a stale feature list. This mismatch only surfaces at prediction time as a silent wrong prediction or a crash.

**Fix:**  
Save `feature_cols` and `metrics` as a separate `model_meta.json` alongside the model:
```json
{
  "feature_cols": ["lag_1", "lag_7", "rolling_7", "month", "..."],
  "metrics": {"mae": 18.4, "rmse": 22.1, "mape": 12.3},
  "trained_at": "2026-03-12",
  "training_days": 90
}
```

---

### 16. No Data Export / Download Option on Dashboard
**File:** `dashboard/app.py`  
**Priority:** Low | **Effort:** Trivial

**Problem:**  
Researchers, journalists, and power users who want the raw data have no way to get it from the dashboard.

**Fix:**
```python
csv = df.to_csv(index=False)
st.download_button(
    label="Download as CSV",
    data=csv,
    file_name=f"ahmedabad_aqi_{date.today()}.csv",
    mime="text/csv",
)
```
One line of code in Streamlit. Huge UX improvement for technical users.

---
---

## Part 2 — Major Feature Proposal: Multi-City / Global AQI Support

This is a significant architectural enhancement that would transform AirCast from a single-city tool into a genuinely useful global AQI forecasting platform.

---

### The Problem with the Current Design

The entire pipeline is hardcoded for Ahmedabad:
- `PRIMARY_STATION = "ahmedabad"` in `config.py`
- Training data is Ahmedabad-only from Kaggle
- Dashboard shows no way to select another city

This means the app is **only useful to people in Ahmedabad**. Any other visitor gets data irrelevant to them.

---

### Proposed Feature: City Picker + On-Demand Training

#### Part A — Location Selection UI

**Option 1: City dropdown (simpler, recommended first)**
```python
SUPPORTED_CITIES = ["Ahmedabad", "Mumbai", "Delhi", "Bangalore", "Chennai", "Pune", ...]
city = st.selectbox("Select your city", SUPPORTED_CITIES)
```

**Option 2: Interactive map click (requires `streamlit-folium` package)**
```python
import folium
from streamlit_folium import st_folium

m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)  # India centered
result = st_folium(m, height=400)
lat = result["last_clicked"]["lat"]
lng = result["last_clicked"]["lng"]
```

WAQI supports coordinate-based queries natively:
```
GET https://api.waqi.info/feed/geo:{lat};{lng}/?token=TOKEN
```
Clicking anywhere on the map instantly returns the nearest monitoring station's AQI.

---

#### Part B — Historical Data Source

**Problem:** WAQI free API does NOT provide historical data (only current readings).

**Solution: OpenAQ API** — free, open-source, global, no authentication required for basic use.

```python
import requests

def fetch_openaq_history(city: str, days: int = 730) -> pd.DataFrame:
    """Fetch up to 2 years of historical AQI data from OpenAQ."""
    url = "https://api.openaq.org/v3/measurements"
    params = {
        "city": city,
        "parameter": ["pm25", "pm10", "no2", "co", "o3"],
        "dateFrom": (date.today() - timedelta(days=days)).isoformat(),
        "dateTo": date.today().isoformat(),
        "limit": 10000,
    }
    response = requests.get(url, params=params, timeout=30)
    return pd.DataFrame(response.json()["results"])
```

OpenAQ covers **100+ countries** and has PM2.5, PM10, NO2, CO, O3 — the exact features the model needs.

**Caveat:** Not every city has 2 years of data. Small towns, especially rural India, have gaps. The CSV upload fallback (Part D) handles this.

---

#### Part C — The Training Time Problem & Solution

XGBoost with 50 Optuna trials takes **2–5 minutes** per city. You cannot make users wait that long inside a web app. Here is the optimized 3-stage approach:

```
User selects a city
        |
Stage 1: Check Supabase — is a cached model available for this city?
        |-- YES --> Load model instantly --> Show predictions
        |-- NO  --> Continue to Stage 2
                        |
                Stage 2: Fetch historical data from OpenAQ
                        |-- Got 90+ days --> Go to Stage 3
                        |-- Not enough   --> Show CSV upload widget (Part D)
                                |
                        Stage 3: Quick-train with progress bar
                                - Reduce Optuna trials: 50 --> 10
                                - Training time: ~30-45 seconds (acceptable)
                                - Show progress: "Fetching data... Training... Done!"
                                - Cache trained model to Supabase + Hugging Face
                                        |
                                Show predictions
```

**Key benefit:** The second user from the same city gets instant results because the model is already cached.

---

#### Part D — CSV Upload Fallback

For cities where OpenAQ has no data:

```python
st.info("No historical data found for this city via OpenAQ.")
st.markdown("Download the CSV template and upload your city's AQI data.")

uploaded = st.file_uploader("Upload AQI CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)

    required = ["date", "aqi", "pm25", "pm10"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        with st.spinner("Training model for your city..."):
            model, metrics = run_retrain(df)
            cache_model(city_name, model, metrics)
        st.success(f"Model trained! MAE: {metrics['mae']:.1f}")
```

The existing Kaggle dataset (`city_day.csv`) already has data for 26 Indian cities — Delhi, Mumbai, Bangalore, Chennai, Pune, Hyderabad, etc. These can be pre-trained and bundled so most Indian cities work out of the box without any wait.

---

#### Part E — Per-City Model Caching

New Supabase table needed:

```sql
CREATE TABLE city_models (
    city         TEXT PRIMARY KEY,
    trained_at   TIMESTAMPTZ DEFAULT NOW(),
    mae          FLOAT,
    rmse         FLOAT,
    model_path   TEXT,
    days_of_data INTEGER,
    query_count  INTEGER DEFAULT 1
);
```

Hugging Face Hub structure:
```
{username}/aircast-models/
    models/
        ahmedabad/
            xgb_model.joblib
            model_meta.json
        delhi/
            xgb_model.joblib
            model_meta.json
        mumbai/
            xgb_model.joblib
            model_meta.json
```

---

#### Part F — Daily Retraining at Scale

Current design retrains one city (Ahmedabad) daily. You cannot run a job per city for hundreds of cities.

**Solution:** Only retrain the **top N most-queried cities** in the daily job, using `query_count` from the `city_models` table.

```python
# In daily_job.py:
top_cities = get_top_queried_cities(limit=10)
for city in top_cities:
    run_retrain_for_city(city)
```

Less popular cities only retrain when a user explicitly clicks a "Refresh Model" button on the dashboard.

---

### Summary: What Changes vs What Stays the Same

| Component | Current | After This Feature |
|---|---|---|
| Location selector | None (hardcoded Ahmedabad) | City dropdown + optional map |
| Training data source | Kaggle CSV (static, Ahmedabad only) | OpenAQ API (dynamic, global) |
| Model storage | Single file on Hugging Face | Per-city files keyed by city name |
| Training trigger | Daily cron only | On-demand for new cities + daily for top cities |
| Supabase tables | `actuals`, `predictions`, `model_performance` | Same + new `city_models` table |
| Optuna trials | 50 (slow, fine for daily) | 10 for new cities (fast), 50 for daily |
| Fallback | None | CSV upload widget |
| Dashboard | Ahmedabad-only | Any city, cached for repeat visitors |

---

### What is NOT Feasible (Honest Assessment)

- **Sub-5-second training:** XGBoost cannot train that fast on 2 years of data. The 30–45 second approach with a progress bar is the practical minimum.
- **100% global coverage:** OpenAQ does not have every city. Rural and small-town users will hit the CSV fallback.
- **Unlimited free training at scale:** GitHub Actions has 2000 min/month free. Training 100 new cities would exhaust that budget. Fine for a personal/portfolio project.

---

### Recommended Implementation Order

1. **Phase 1 (Low effort, high impact):** Add city dropdown using Kaggle's existing 26-city dataset. Pre-train all 26 models. No new APIs needed.
2. **Phase 2:** Add OpenAQ integration for cities outside the Kaggle dataset.
3. **Phase 3:** Add the interactive map picker using `streamlit-folium`.
4. **Phase 4:** Add CSV upload fallback for unsupported cities.
5. **Phase 5:** Add `city_models` Supabase table and per-city model caching for instant repeat visits.

---

## Quick Priority Reference

| # | Suggestion | Priority | Effort |
|---|---|---|---|
| 2 | Replace pickle with joblib (security) | **Critical** | Low |
| 10 | Validate AQI values before storing | High | Low |
| 4 | Add retry logic for WAQI API failures | High | Low |
| 1 | Fix Python version mismatch in CI | High | Trivial |
| 3 | Make Hugging Face repo private | Medium | Trivial |
| 5 | Fix point-in-time MAE evaluation | Medium | Low |
| 9 | Add fallback stations | Medium | Medium |
| 11 | Add seasonal calendar features | Medium | Medium |
| 13 | Add failure alerting | Medium | Low |
| 15 | Separate feature_cols from model file | Medium | Low |
| 7 | Graceful DB failure UX | Medium | Low |
| 12 | Multi-day (3-day) forecast | High impact | High |
| 6 | Increase cache TTL | Low | Trivial |
| 8 | Add last-updated timestamp | Low | Trivial |
| 14 | Add health recommendation text | Low | Low |
| 16 | Add CSV download button | Low | Trivial |
| — | **Multi-city / global support** | **Game-changing** | High |

---

*This document was prepared as a peer code review. All suggestions are optional and prioritized — start with the Critical and High items before tackling the larger feature proposal.*
