-- =============================================================================
-- QUANT INTELLIGENCE PLATFORM — MASTER DATABASE MIGRATION 001
-- Target Engine: PostgreSQL 18.6 + TimescaleDB 2.29.2 + pgvector 0.8.x
-- Precision Mandate: 5-Decimal Sub-Penny Support ($0.00001 to $10,000.00+)
-- Philosophy Mandate: 100% Raw Market Physics (Zero Lagging Human Indicators)
-- =============================================================================

BEGIN;

-- 1. Enable Required Institutional Extensions
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS vector CASCADE;
CREATE EXTENSION IF NOT EXISTS pg_trgm CASCADE;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp" CASCADE;

-- =============================================================================
-- 2. STREAM 1: RAW CHART & TAPE PHYSICS HYPERTABLE (`raw_bars`)
-- Multi-scale grids: Tick, 1m, 3m, 5m, 15m, 1H, 4H, 1D, 1W, 1M
-- =============================================================================
CREATE TABLE IF NOT EXISTS raw_bars (
    ts                  TIMESTAMPTZ      NOT NULL,
    timestamp_ns        BIGINT           NOT NULL,
    symbol              TEXT             NOT NULL,
    timeframe           TEXT             NOT NULL,
    open_price          NUMERIC(18, 5)   NOT NULL CHECK (open_price >= 0.00001),
    high_price          NUMERIC(18, 5)   NOT NULL CHECK (high_price >= open_price OR high_price >= 0.00001),
    low_price           NUMERIC(18, 5)   NOT NULL CHECK (low_price >= 0.00001 AND low_price <= high_price),
    close_price         NUMERIC(18, 5)   NOT NULL CHECK (close_price >= 0.00001),
    volume              BIGINT           NOT NULL CHECK (volume >= 0),
    vwap                NUMERIC(18, 5)   NOT NULL CHECK (vwap >= 0.00001),
    dollar_turnover     NUMERIC(22, 5)   NOT NULL CHECK (dollar_turnover >= 0),
    trade_count         INTEGER          NOT NULL CHECK (trade_count >= 0),
    bid_ask_spread      NUMERIC(18, 5)   NOT NULL DEFAULT 0.00001,
    time_delta_ms       BIGINT           NOT NULL DEFAULT 60000,
    PRIMARY KEY (ts, symbol, timeframe)
);

SELECT create_hypertable(
    'raw_bars',
    by_range('ts', INTERVAL '1 day'),
    if_not_exists => TRUE
);

SELECT add_dimension(
    'raw_bars',
    by_hash('symbol', 16),
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_raw_bars_symbol_tf_ts
    ON raw_bars (symbol, timeframe, ts DESC);

ALTER TABLE raw_bars SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol, timeframe',
    timescaledb.compress_orderby = 'ts DESC, timestamp_ns DESC'
);

SELECT add_compression_policy('raw_bars', INTERVAL '7 days', if_not_exists => TRUE);

-- =============================================================================
-- 3. STREAM 2: RAW OPTIONS & DEALER POSITIONING HYPERTABLE (`raw_options_flow`)
-- Strike-by-strike Gamma / Vanna Squeeze Physics
-- =============================================================================
CREATE TABLE IF NOT EXISTS raw_options_flow (
    ts                  TIMESTAMPTZ      NOT NULL,
    timestamp_ns        BIGINT           NOT NULL,
    underlying          TEXT             NOT NULL,
    expiration_date     DATE             NOT NULL,
    dte                 INTEGER          NOT NULL CHECK (dte >= 0),
    strike              NUMERIC(18, 5)   NOT NULL CHECK (strike > 0),
    option_right        CHAR(1)          NOT NULL CHECK (option_right IN ('C', 'P')),
    bid                 NUMERIC(18, 5)   NOT NULL DEFAULT 0,
    ask                 NUMERIC(18, 5)   NOT NULL DEFAULT 0,
    last_price          NUMERIC(18, 5)   NOT NULL DEFAULT 0,
    call_volume         BIGINT           NOT NULL DEFAULT 0,
    put_volume          BIGINT           NOT NULL DEFAULT 0,
    open_interest       BIGINT           NOT NULL DEFAULT 0,
    ask_fill_ratio      DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    bid_fill_ratio      DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    implied_volatility  DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    delta               DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    gamma               DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    vanna               DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    theta               DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    vega                DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    PRIMARY KEY (ts, underlying, expiration_date, strike, option_right)
);

SELECT create_hypertable(
    'raw_options_flow',
    by_range('ts', INTERVAL '1 day'),
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_raw_options_underlying_ts
    ON raw_options_flow (underlying, ts DESC);

-- =============================================================================
-- 4. STREAM 3: RAW SHORT & FLOAT MECHANICS HYPERTABLE (`raw_short_float`)
-- Short Squeeze & Micro-Cap Dilution Physics
-- =============================================================================
CREATE TABLE IF NOT EXISTS raw_short_float (
    ts                          TIMESTAMPTZ      NOT NULL,
    timestamp_ns                BIGINT           NOT NULL,
    symbol                      TEXT             NOT NULL,
    public_float_shares         BIGINT           NOT NULL CHECK (public_float_shares > 0),
    shares_outstanding          BIGINT           NOT NULL CHECK (shares_outstanding >= public_float_shares),
    short_interest_shares       BIGINT           NOT NULL DEFAULT 0,
    days_to_cover               DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    intraday_short_vol_ratio    DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    cost_to_borrow_annual_pct   DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    PRIMARY KEY (ts, symbol)
);

SELECT create_hypertable(
    'raw_short_float',
    by_range('ts', INTERVAL '7 days'),
    if_not_exists => TRUE
);

-- =============================================================================
-- 5. STREAM 4: RAW MARKET INTERNALS HYPERTABLE (`raw_market_internals`)
-- Synchronized Macro Heartbeat: $TICK, $ADD, $VOLD, VIX Term Structure & Sectors
-- =============================================================================
CREATE TABLE IF NOT EXISTS raw_market_internals (
    ts                  TIMESTAMPTZ      NOT NULL,
    timestamp_ns        BIGINT           NOT NULL,
    nyse_tick           INTEGER          NOT NULL,
    nasdaq_tick         INTEGER          NOT NULL,
    add_breadth         INTEGER          NOT NULL,
    vold_up_down_ratio  DOUBLE PRECISION NOT NULL,
    vix_spot            DOUBLE PRECISION NOT NULL,
    vix_front_month     DOUBLE PRECISION NOT NULL,
    sector_flows_json   JSONB            NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (ts)
);

SELECT create_hypertable(
    'raw_market_internals',
    by_range('ts', INTERVAL '1 day'),
    if_not_exists => TRUE
);

-- =============================================================================
-- 6. PGVECTOR 0.8 HISTORICAL TRACE EMBEDDINGS (`trace_embeddings`)
-- Sub-5ms HNSW Cosine Similarity Lookup for Historical Precedent Matching
-- =============================================================================
CREATE TABLE IF NOT EXISTS trace_embeddings (
    trace_id            UUID             PRIMARY KEY DEFAULT uuid_generate_v4(),
    brain_id            TEXT             NOT NULL CHECK (brain_id IN ('BRAIN-01', 'BRAIN-02', 'BRAIN-03', 'BRAIN-04', 'BRAIN-05')),
    symbol              TEXT             NOT NULL,
    timeframe           TEXT             NOT NULL,
    inflection_ts       TIMESTAMPTZ      NOT NULL,
    timestamp_ns        BIGINT           NOT NULL,
    direction           TEXT             NOT NULL CHECK (direction IN ('LONG', 'SHORT')),
    entry_price         NUMERIC(18, 5)   NOT NULL CHECK (entry_price >= 0.00001),
    realized_r_multiple DOUBLE PRECISION NOT NULL,
    mfe_pct             DOUBLE PRECISION NOT NULL,
    mae_pct             DOUBLE PRECISION NOT NULL,
    hold_duration_ms    BIGINT           NOT NULL,
    was_winner          BOOLEAN          NOT NULL,
    primary_driver      TEXT             NOT NULL,
    stream_weights_json JSONB            NOT NULL DEFAULT '{}'::jsonb,
    embedding           vector(128)      NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trace_embeddings_hnsw_cosine
    ON trace_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 24, ef_construction = 128);

CREATE INDEX IF NOT EXISTS idx_trace_embeddings_brain_driver
    ON trace_embeddings (brain_id, primary_driver, was_winner);

-- =============================================================================
-- 7. DEDICATED LIVE "POSITIONS OPENED" DECK (`positions_opened`)
-- Every live signal lands here instantaneously with ticking telemetry
-- =============================================================================
CREATE TABLE IF NOT EXISTS positions_opened (
    position_id                 UUID             PRIMARY KEY DEFAULT uuid_generate_v4(),
    opened_at                   TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    opened_timestamp_ns         BIGINT           NOT NULL,
    brain_id                    TEXT             NOT NULL CHECK (brain_id IN ('BRAIN-01', 'BRAIN-02', 'BRAIN-03', 'BRAIN-04', 'BRAIN-05')),
    brain_name                  TEXT             NOT NULL,
    symbol                      TEXT             NOT NULL,
    exchange_index              TEXT             NOT NULL DEFAULT 'US-EQUITY',
    direction                   TEXT             NOT NULL CHECK (direction IN ('LONG', 'SHORT')),
    -- Stock Execution & Live Ticking Levels (5-decimal $0.00001 precision)
    entry_price                 NUMERIC(18, 5)   NOT NULL CHECK (entry_price >= 0.00001),
    live_price                  NUMERIC(18, 5)   NOT NULL CHECK (live_price >= 0.00001),
    structural_stop_price       NUMERIC(18, 5)   NOT NULL CHECK (structural_stop_price >= 0.00001),
    target_1_price              NUMERIC(18, 5)   NOT NULL CHECK (target_1_price >= 0.00001),
    target_2_price              NUMERIC(18, 5)   NOT NULL CHECK (target_2_price >= 0.00001),
    quantity_shares             NUMERIC(18, 4)   NOT NULL DEFAULT 100.0,
    -- Matched Options Execution & Live Ticking Premiums
    option_structure            TEXT             NOT NULL DEFAULT 'NONE',
    matched_option_contract     TEXT             NOT NULL DEFAULT 'SHARES_ONLY',
    option_entry_premium        NUMERIC(18, 5)   NOT NULL DEFAULT 0.00000,
    option_live_premium         NUMERIC(18, 5)   NOT NULL DEFAULT 0.00000,
    option_stop_premium         NUMERIC(18, 5)   NOT NULL DEFAULT 0.00000,
    option_target_premium       NUMERIC(18, 5)   NOT NULL DEFAULT 0.00000,
    option_contracts_count      INTEGER          NOT NULL DEFAULT 0,
    -- Live Unrealized Telemetry
    unrealized_stock_pnl_usd    NUMERIC(18, 5)   NOT NULL DEFAULT 0.00000,
    unrealized_stock_pnl_pct    DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    unrealized_option_pnl_usd   NUMERIC(18, 5)   NOT NULL DEFAULT 0.00000,
    unrealized_option_pnl_pct   DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    current_r_multiple          DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    mae_pct                     DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    mfe_pct                     DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    -- AI Raw Fingerprint & Trace Health Telemetry
    ai_confidence_pct           DOUBLE PRECISION NOT NULL CHECK (ai_confidence_pct BETWEEN 0.0 AND 100.0),
    primary_driver              TEXT             NOT NULL,
    trace_health_badge          TEXT             NOT NULL DEFAULT 'TRACE_ACCELERATING',
    precedent_matches_count     INTEGER          NOT NULL DEFAULT 0,
    precedent_win_rate_pct      DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    stream_weights_json         JSONB            NOT NULL DEFAULT '{}'::jsonb,
    price_sparkline_json        JSONB            NOT NULL DEFAULT '[]'::jsonb,
    updated_at                  TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_positions_opened_brain_symbol
    ON positions_opened (brain_id, symbol, opened_at DESC);

CREATE INDEX IF NOT EXISTS idx_positions_opened_driver_conf
    ON positions_opened (primary_driver, ai_confidence_pct DESC);

-- =============================================================================
-- 8. HISTORICAL CLOSED TRADE LEDGER (`closed_trade_ledger`)
-- Immutable audit log of every completed trade with Hyper-Filter indexes
-- =============================================================================
CREATE TABLE IF NOT EXISTS closed_trade_ledger (
    trade_id                    UUID             PRIMARY KEY DEFAULT uuid_generate_v4(),
    position_id                 UUID             NOT NULL UNIQUE,
    outcome_badge               TEXT             NOT NULL CHECK (outcome_badge IN (
                                    'CLOSED - TARGET HIT',
                                    'CLOSED - AI DYNAMIC EXIT',
                                    'CLOSED - TRAILING STOP',
                                    'CLOSED - STOPPED OUT'
                                )),
    brain_id                    TEXT             NOT NULL CHECK (brain_id IN ('BRAIN-01', 'BRAIN-02', 'BRAIN-03', 'BRAIN-04', 'BRAIN-05')),
    brain_name                  TEXT             NOT NULL,
    symbol                      TEXT             NOT NULL,
    exchange_index              TEXT             NOT NULL,
    direction                   TEXT             NOT NULL CHECK (direction IN ('LONG', 'SHORT')),
    -- Full Millisecond Timestamps & Hold Duration
    opened_at                   TIMESTAMPTZ      NOT NULL,
    closed_at                   TIMESTAMPTZ      NOT NULL,
    opened_timestamp_ns         BIGINT           NOT NULL,
    closed_timestamp_ns         BIGINT           NOT NULL,
    hold_duration_ms            BIGINT           NOT NULL CHECK (hold_duration_ms >= 0),
    -- Stock Execution Levels (5-decimal $0.00001 precision)
    entry_price                 NUMERIC(18, 5)   NOT NULL CHECK (entry_price >= 0.00001),
    initial_structural_stop     NUMERIC(18, 5)   NOT NULL CHECK (initial_structural_stop >= 0.00001),
    target_1_price              NUMERIC(18, 5)   NOT NULL CHECK (target_1_price >= 0.00001),
    target_2_price              NUMERIC(18, 5)   NOT NULL CHECK (target_2_price >= 0.00001),
    actual_exit_price           NUMERIC(18, 5)   NOT NULL CHECK (actual_exit_price >= 0.00001),
    quantity_shares             NUMERIC(18, 4)   NOT NULL DEFAULT 100.0,
    -- Matched Options Execution Levels
    option_structure            TEXT             NOT NULL DEFAULT 'NONE',
    matched_option_contract     TEXT             NOT NULL DEFAULT 'SHARES_ONLY',
    option_entry_premium        NUMERIC(18, 5)   NOT NULL DEFAULT 0.00000,
    option_stop_premium         NUMERIC(18, 5)   NOT NULL DEFAULT 0.00000,
    option_exit_premium         NUMERIC(18, 5)   NOT NULL DEFAULT 0.00000,
    option_contracts_count      INTEGER          NOT NULL DEFAULT 0,
    -- Realized Performance & Running Equity
    gross_pnl_usd               NUMERIC(18, 5)   NOT NULL,
    slippage_commissions_usd    NUMERIC(18, 5)   NOT NULL DEFAULT 0.00000,
    net_pnl_usd                 NUMERIC(18, 5)   NOT NULL,
    option_net_pnl_usd          NUMERIC(18, 5)   NOT NULL DEFAULT 0.00000,
    stock_return_pct            DOUBLE PRECISION NOT NULL,
    option_return_pct           DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    realized_r_multiple         DOUBLE PRECISION NOT NULL,
    mae_pct                     DOUBLE PRECISION NOT NULL,
    mfe_pct                     DOUBLE PRECISION NOT NULL,
    is_winner                   BOOLEAN          NOT NULL,
    cumulative_net_profit_usd   NUMERIC(20, 5)   NOT NULL,
    -- AI Fingerprint & Historical Precedent Snapshot
    ai_confidence_pct           DOUBLE PRECISION NOT NULL,
    primary_driver              TEXT             NOT NULL,
    precedent_matches_count     INTEGER          NOT NULL DEFAULT 0,
    precedent_win_rate_pct      DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    fingerprint_snapshot_json   JSONB            NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_closed_ledger_closed_at
    ON closed_trade_ledger (closed_at DESC);

CREATE INDEX IF NOT EXISTS idx_closed_ledger_hyper_filter
    ON closed_trade_ledger (brain_id, is_winner, outcome_badge, exchange_index, primary_driver, closed_at DESC);

CREATE INDEX IF NOT EXISTS idx_closed_ledger_symbol_trgm
    ON closed_trade_ledger USING gin (symbol gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_closed_ledger_price_hold
    ON closed_trade_ledger (entry_price, hold_duration_ms, realized_r_multiple);

-- =============================================================================
-- 9. 3-SPEED CONTINUOUS SELF-LEARNING & EVOLUTION LOG (`brain_evolution_log`)
-- Tracks Speed 1 (Intraday), Speed 2 (4:15 PM Autopsy), Speed 3 (Weekend Tournament)
-- =============================================================================
CREATE TABLE IF NOT EXISTS brain_evolution_log (
    evolution_id                UUID             PRIMARY KEY DEFAULT uuid_generate_v4(),
    executed_at                 TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    brain_id                    TEXT             NOT NULL CHECK (brain_id IN ('BRAIN-01', 'BRAIN-02', 'BRAIN-03', 'BRAIN-04', 'BRAIN-05')),
    evolution_speed             TEXT             NOT NULL CHECK (evolution_speed IN (
                                    'SPEED_1_INTRADAY_ADAPTATION',
                                    'SPEED_2_NIGHTLY_AUTOPSY',
                                    'SPEED_3_WEEKEND_TOURNAMENT'
                                )),
    champion_model_version      TEXT             NOT NULL,
    challenger_model_version    TEXT             NOT NULL,
    signals_graded_count        INTEGER          NOT NULL DEFAULT 0,
    hard_negatives_mined        INTEGER          NOT NULL DEFAULT 0,
    missed_moves_discovered     INTEGER          NOT NULL DEFAULT 0,
    champion_net_profit_usd     NUMERIC(18, 5)   NOT NULL DEFAULT 0.00000,
    challenger_net_profit_usd   NUMERIC(18, 5)   NOT NULL DEFAULT 0.00000,
    champion_win_rate_pct       DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    challenger_win_rate_pct     DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    champion_profit_factor      DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    challenger_profit_factor    DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    champion_max_drawdown_pct   DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    challenger_max_drawdown_pct DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    promoted_to_production      BOOLEAN          NOT NULL DEFAULT FALSE,
    onnx_artifact_path          TEXT             NOT NULL,
    telemetry_report_json       JSONB            NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_brain_evolution_brain_ts
    ON brain_evolution_log (brain_id, executed_at DESC);

COMMIT;

