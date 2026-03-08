-- ─────────────────────────────────────────────────────────────────────────────
-- AirCast — Database Schema
-- Run this once against your Supabase project (SQL Editor → New Query → Run)
-- ─────────────────────────────────────────────────────────────────────────────

-- Enable UUID generation (Supabase has this by default, kept for portability)
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ─── Table 1: predictions ─────────────────────────────────────────────────────
-- Stores what the model predicted for each date BEFORE that day arrives.
CREATE TABLE IF NOT EXISTS predictions (
    id           BIGSERIAL    PRIMARY KEY,
    date         DATE         NOT NULL,
    station      VARCHAR(60)  NOT NULL DEFAULT 'ahmedabad',
    predicted    NUMERIC(6,2) NOT NULL,         -- predicted AQI value
    model_ver    VARCHAR(30)  NOT NULL,          -- e.g. "v1.0.0" or HF commit SHA
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    CONSTRAINT predictions_date_station_unique UNIQUE (date, station)
);

COMMENT ON TABLE predictions IS
    'Model predictions logged one day ahead. Each (date, station) pair is unique.';

-- ─── Table 2: actuals ─────────────────────────────────────────────────────────
-- Stores the real AQI fetched the following morning from WAQI API.
CREATE TABLE IF NOT EXISTS actuals (
    id           BIGSERIAL    PRIMARY KEY,
    date         DATE         NOT NULL,
    station      VARCHAR(60)  NOT NULL DEFAULT 'ahmedabad',
    actual_aqi   NUMERIC(6,2) NOT NULL,          -- observed AQI value
    dominant_pollutant VARCHAR(20),              -- e.g. "pm25", "pm10"
    fetched_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    CONSTRAINT actuals_date_station_unique UNIQUE (date, station)
);

COMMENT ON TABLE actuals IS
    'Real AQI observations fetched from WAQI API the morning after each forecast day.';

-- ─── Table 3: model_performance ───────────────────────────────────────────────
-- One row per evaluation run. Tracks daily error and retraining events.
CREATE TABLE IF NOT EXISTS model_performance (
    id              BIGSERIAL    PRIMARY KEY,
    eval_date       DATE         NOT NULL,       -- the date being evaluated
    model_ver       VARCHAR(30)  NOT NULL,        -- model version that made prediction
    mae             NUMERIC(8,4),                -- Mean Absolute Error
    rmse            NUMERIC(8,4),                -- Root Mean Squared Error
    mape            NUMERIC(8,4),                -- Mean Absolute Percentage Error (%)
    retrain_triggered   BOOLEAN  NOT NULL DEFAULT FALSE,
    retrain_reason  TEXT,                        -- e.g. "MAE 24.3 exceeded threshold 20"
    new_model_ver   VARCHAR(30),                 -- set if retraining produced a new model
    new_mae         NUMERIC(8,4),                -- post-retrain MAE on hold-out set
    promoted        BOOLEAN      NOT NULL DEFAULT FALSE,  -- TRUE if new model replaced old
    logged_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    CONSTRAINT model_performance_eval_date_unique UNIQUE (eval_date)
);

COMMENT ON TABLE model_performance IS
    'Daily evaluation log. Records error metrics and whether retraining was triggered and successful.';

-- ─── Indexes ──────────────────────────────────────────────────────────────────
-- Support fast dashboard queries: "give me last 30 days of predictions + actuals"
CREATE INDEX IF NOT EXISTS idx_predictions_date  ON predictions  (date DESC);
CREATE INDEX IF NOT EXISTS idx_actuals_date      ON actuals      (date DESC);
CREATE INDEX IF NOT EXISTS idx_model_perf_date   ON model_performance (eval_date DESC);

-- ─── Row-Level Security (Supabase) ────────────────────────────────────────────
-- All writes go through the service-role key (server-side only).
-- The dashboard reads via the anon key — read-only is sufficient.
ALTER TABLE predictions      ENABLE ROW LEVEL SECURITY;
ALTER TABLE actuals           ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_performance ENABLE ROW LEVEL SECURITY;

-- Anon users: SELECT only
CREATE POLICY "anon_read_predictions"
    ON predictions FOR SELECT USING (true);

CREATE POLICY "anon_read_actuals"
    ON actuals FOR SELECT USING (true);

CREATE POLICY "anon_read_model_performance"
    ON model_performance FOR SELECT USING (true);

-- Service role bypasses RLS by default in Supabase — no insert policies needed here.
