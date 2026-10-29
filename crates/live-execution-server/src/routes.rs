//! Axum 0.8.9 REST + Telemetry Server Routes (`/{id}` curly-brace path syntax).

use std::collections::HashSet;
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;

use axum::Json;
use axum::Router;
use axum::extract::{Path as AxumPath, Query, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use market_data_bus::{
    AlpacaFreeAdapter, FreeBulkBackfillAdapter, HistoricalReplayAdapter,
    InstitutionalExpansionAdapter, MultiFeedRouter, ProviderTelemetry, ReplaySpeed,
    SchwabDirectAdapter, Timeframe, TwoStageMarketFunnel,
};
use raw_fingerprint_core::{BrainHorizonId, PrimaryFingerprintDriver, RawBarRecord, SubPennyPrice};
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;
use uuid::Uuid;

use crate::exporter::{ExportScope, export_to_csv_bytes, export_to_xlsx_bytes};
use crate::ledger::{
    ClosedTradeRow, FilteredLedgerSnapshot, HyperFilterQuery, OpenPositionRow, RealTimeLedgerEngine,
};
use crate::onnx_engine::{LiveBrainInferenceOutput, OnnxMultiBrainEngine};
use crate::precedent_knn::{HistoricalTraceRecord, PrecedentKnnIndex, PrecedentMatchSummary};

/// Shared thread-safe server state across all Axum 0.8 routes.
#[derive(Clone)]
pub struct AppState {
    pub data_bus: Arc<MultiFeedRouter>,
    pub scanner_funnel: Arc<TwoStageMarketFunnel>,
    pub onnx_engine: Arc<OnnxMultiBrainEngine>,
    pub precedent_knn: Arc<PrecedentKnnIndex>,
    pub ledger: Arc<RealTimeLedgerEngine>,
}

impl AppState {
    /// Initializes the complete server state with all 5 Market Data Bus adapter slots,
    /// ONNX 5-Brain inference engine, `pgvector` k-NN precedent store, and Real-Time Ledger.
    pub fn new(models_dir: Option<&Path>) -> Self {
        let data_bus = Arc::new(MultiFeedRouter::new(4096));
        data_bus.register_adapter(
            Arc::new(HistoricalReplayAdapter::new(ReplaySpeed::MaxSpeed, 500)),
            true,
        );
        data_bus.register_adapter(Arc::new(AlpacaFreeAdapter::new(None)), false);
        data_bus.register_adapter(Arc::new(SchwabDirectAdapter::new()), false);
        data_bus.register_adapter(Arc::new(FreeBulkBackfillAdapter::new()), false);
        data_bus.register_adapter(
            Arc::new(InstitutionalExpansionAdapter::new("POLYGON_DATABENTO_IBKR")),
            false,
        );

        Self {
            data_bus,
            scanner_funnel: Arc::new(TwoStageMarketFunnel::new()),
            onnx_engine: Arc::new(OnnxMultiBrainEngine::new(models_dir)),
            precedent_knn: Arc::new(PrecedentKnnIndex::new_preseeded()),
            ledger: Arc::new(RealTimeLedgerEngine::new()),
        }
    }

    /// Runs an automated end-to-end `HistoricalReplayAdapter` multi-timeframe simulation:
    /// 1. Evaluates tickers across all 5 Brains (`BRAIN-01` `$0.00003` sub-penny `ZNTX` through `BRAIN-05` `GS`).
    /// 2. Opens live positions in `positions_opened`.
    /// 3. Streams synthetic multi-bar price trajectories (Target Hit, Stopped Out, AI Dynamic Exit, and Active Open).
    pub async fn run_historical_replay_simulation(&self) -> Result<FilteredLedgerSnapshot, String> {
        let simulation_plan = [
            (
                "ZNTX",
                BrainHorizonId::Brain01PennyAnomaly,
                "OTC_PENNY",
                false,
                "TARGET_HIT",
            ),
            (
                "KULR",
                BrainHorizonId::Brain01PennyAnomaly,
                "RUSSELL_2000",
                true,
                "OPEN",
            ),
            (
                "NVDA",
                BrainHorizonId::Brain02IntradayDayTrade,
                "NASDAQ_100",
                true,
                "TARGET_HIT",
            ),
            (
                "TSLA",
                BrainHorizonId::Brain02IntradayDayTrade,
                "NASDAQ_100",
                true,
                "DYNAMIC_EXIT",
            ),
            (
                "PLTR",
                BrainHorizonId::Brain03ShortTermSwing,
                "SP_500",
                true,
                "TARGET_HIT",
            ),
            (
                "IONQ",
                BrainHorizonId::Brain03ShortTermSwing,
                "RUSSELL_2000",
                true,
                "STOPPED_OUT",
            ),
            (
                "AMD",
                BrainHorizonId::Brain04CoreSwing,
                "NASDAQ_100",
                true,
                "OPEN",
            ),
            (
                "GS",
                BrainHorizonId::Brain05LongTermWealth,
                "DOW_30",
                true,
                "TARGET_HIT",
            ),
        ];

        let internals_unified = self
            .data_bus
            .fetch_market_internals_with_failover()
            .await
            .map_err(|e| e.to_string())?;
        let raw_internals = internals_unified.to_raw_internals_snapshot("SMH");

        for (symbol, brain_id, exchange_idx, is_opt, scenario) in simulation_plan {
            let unified_bars = self
                .data_bus
                .fetch_historical_bars_with_failover(symbol, Timeframe::M5, 32)
                .await
                .map_err(|e| e.to_string())?;
            let raw_bars: Vec<RawBarRecord> =
                unified_bars.iter().map(|b| b.to_raw_bar_record()).collect();

            let opt_chain = self
                .data_bus
                .fetch_option_chain_with_failover(symbol)
                .await
                .map_err(|e| e.to_string())?;
            let raw_opts = opt_chain.to_raw_options_snapshot();

            let snaps = self
                .data_bus
                .fetch_universe_snapshots_with_failover(Some(&[symbol.to_string()]))
                .await
                .map_err(|e| e.to_string())?;
            let raw_sf = snaps
                .first()
                .map(|s| s.to_raw_short_float_snapshot())
                .unwrap_or_default();

            let inference = self
                .onnx_engine
                .evaluate_live_signal(
                    symbol,
                    brain_id,
                    is_opt,
                    &raw_bars,
                    &raw_opts,
                    &raw_sf,
                    &raw_internals,
                )
                .map_err(|e| e.to_string())?;

            let precedents = self.precedent_knn.query_precedents(
                brain_id,
                &inference.signal.latent_embedding_128,
                25,
            );

            let open_pos =
                self.ledger
                    .open_position(exchange_idx, &inference, &precedents, 10_000.0);

            // Stream subsequent bars according to the simulation scenario
            let mut stream_bars = raw_bars.clone();
            let last_ts = stream_bars.last().map(|b| b.timestamp_ns).unwrap_or(0);

            match scenario {
                "TARGET_HIT" => {
                    // Intermediate bar reaching Target 1, then final bar hitting Target 2
                    let mut bar_t1 = *stream_bars.last().unwrap();
                    bar_t1.timestamp_ns = last_ts + 300_000_000_000;
                    bar_t1.close = open_pos.target_1_price;
                    bar_t1.high = open_pos.target_1_price.max(open_pos.entry_price);
                    bar_t1.low = open_pos.target_1_price.min(open_pos.entry_price);
                    stream_bars.push(bar_t1);
                    let _ = self.ledger.process_live_bar_for_symbol(
                        symbol,
                        &stream_bars,
                        &raw_internals,
                    );

                    let mut bar_t2 = bar_t1;
                    bar_t2.timestamp_ns = last_ts + 900_000_000_000;
                    bar_t2.close = open_pos.target_2_price;
                    bar_t2.high = open_pos.target_2_price.max(open_pos.entry_price);
                    bar_t2.low = open_pos.target_2_price.min(open_pos.entry_price);
                    stream_bars.push(bar_t2);
                    let closed = self.ledger.process_live_bar_for_symbol(
                        symbol,
                        &stream_bars,
                        &raw_internals,
                    );

                    for c in closed {
                        self.precedent_knn.insert_trace(HistoricalTraceRecord {
                            trace_id: c.trade_id,
                            brain_id: c.brain_id,
                            symbol: c.symbol,
                            timeframe: "5m".to_string(),
                            timestamp_ns: c.closed_timestamp_ns,
                            realized_r_multiple: c.realized_r_multiple,
                            mfe_pct: c.mfe_pct,
                            mae_pct: c.mae_pct,
                            hold_duration_ms: c.hold_duration_ms,
                            was_winner: c.is_winner,
                            primary_driver: c.primary_driver,
                            embedding_128: inference.signal.latent_embedding_128.clone(),
                        });
                    }
                }
                "DYNAMIC_EXIT" => {
                    // Move into profit (at Target 1), then print an opposing absorption wick before Target 2 on 6x volume surge
                    let mut bar_exhaust = *stream_bars.last().unwrap();
                    bar_exhaust.timestamp_ns = last_ts + 600_000_000_000;
                    bar_exhaust.open = open_pos.target_1_price;
                    bar_exhaust.close = open_pos.target_1_price;
                    let mid_t1_t2_u = (open_pos.target_1_price.scaled_units() * 2
                        + open_pos.target_2_price.scaled_units() * 8)
                        / 10;
                    let wick_extreme = SubPennyPrice::from_scaled_units(mid_t1_t2_u.max(1))
                        .unwrap_or(open_pos.target_1_price);
                    let mut headwind_internals = raw_internals;
                    match open_pos.direction {
                        raw_fingerprint_core::TradeDirection::Long => {
                            bar_exhaust.high = wick_extreme.max(open_pos.target_1_price);
                            bar_exhaust.low = open_pos.entry_price.min(open_pos.target_1_price);
                            headwind_internals.nyse_tick = -1150;
                        }
                        raw_fingerprint_core::TradeDirection::Short => {
                            bar_exhaust.low = wick_extreme.min(open_pos.target_1_price);
                            bar_exhaust.high = open_pos.entry_price.max(open_pos.target_1_price);
                            headwind_internals.nyse_tick = 1150;
                        }
                    }
                    bar_exhaust.volume *= 6;
                    stream_bars.push(bar_exhaust);
                    let _ = self.ledger.process_live_bar_for_symbol(
                        symbol,
                        &stream_bars,
                        &headwind_internals,
                    );
                }
                "STOPPED_OUT" => {
                    let mut bar_stop = *stream_bars.last().unwrap();
                    bar_stop.timestamp_ns = last_ts + 420_000_000_000;
                    bar_stop.close = open_pos.structural_stop_price;
                    bar_stop.low = open_pos.structural_stop_price.min(open_pos.entry_price);
                    bar_stop.high = open_pos.structural_stop_price.max(open_pos.entry_price);
                    stream_bars.push(bar_stop);
                    let _ = self.ledger.process_live_bar_for_symbol(
                        symbol,
                        &stream_bars,
                        &raw_internals,
                    );
                }
                _ => {
                    // Keep position open in `POSITIONS OPENED` with live ticking gain toward T1
                    let mut bar_live = *stream_bars.last().unwrap();
                    bar_live.timestamp_ns = last_ts + 240_000_000_000;
                    let mid_u = (open_pos.entry_price.scaled_units()
                        + open_pos.target_1_price.scaled_units())
                        / 2;
                    let mid_p = SubPennyPrice::from_scaled_units(mid_u.max(1))
                        .unwrap_or(open_pos.entry_price);
                    bar_live.close = mid_p;
                    bar_live.high = mid_p.max(open_pos.entry_price);
                    bar_live.low = mid_p.min(open_pos.entry_price);
                    stream_bars.push(bar_live);
                    let _ = self.ledger.process_live_bar_for_symbol(
                        symbol,
                        &stream_bars,
                        &raw_internals,
                    );
                }
            }
        }

        Ok(self
            .ledger
            .query_filtered_snapshot(&HyperFilterQuery::default()))
    }
}

/// Builds the complete Axum 0.8.9 Router with curly-brace `/{param}` routes.
pub fn build_axum_router(state: AppState) -> Router {
    Router::new()
        .route("/api/v1/health", get(health_handler))
        .route("/api/v1/providers", get(providers_telemetry_handler))
        .route(
            "/api/v1/providers/hotswap/{provider_id}",
            post(hotswap_provider_handler),
        )
        .route("/api/v1/ledger/snapshot", get(ledger_snapshot_handler))
        .route("/api/v1/positions/{id}", get(get_open_position_handler))
        .route("/api/v1/trades/{id}", get(get_closed_trade_handler))
        .route(
            "/api/v1/brains/{brain_id}/evaluate/{symbol}",
            post(evaluate_symbol_handler),
        )
        .route("/api/v1/replay/simulate", post(replay_simulate_handler))
        .route("/api/v1/export/csv", get(export_csv_handler))
        .route("/api/v1/export/xlsx", get(export_xlsx_handler))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub server_version: String,
    pub active_brains_count: usize,
    pub open_positions_count: usize,
    pub closed_trades_count: usize,
}

async fn health_handler(State(state): State<AppState>) -> Json<HealthResponse> {
    let snap = state
        .ledger
        .query_filtered_snapshot(&HyperFilterQuery::default());
    Json(HealthResponse {
        status: "OK".to_string(),
        server_version: "argon-live-execution-server-v0.1.0".to_string(),
        active_brains_count: 5,
        open_positions_count: snap.positions_opened.len(),
        closed_trades_count: snap.closed_trades.len(),
    })
}

async fn providers_telemetry_handler(
    State(state): State<AppState>,
) -> Json<Vec<ProviderTelemetry>> {
    Json(state.data_bus.telemetry_snapshot())
}

async fn hotswap_provider_handler(
    State(state): State<AppState>,
    AxumPath(provider_id): AxumPath<String>,
) -> Result<Json<Vec<ProviderTelemetry>>, (StatusCode, String)> {
    state
        .data_bus
        .hot_swap_primary(&provider_id)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    Ok(Json(state.data_bus.telemetry_snapshot()))
}

#[derive(Debug, Default, Deserialize)]
pub struct LedgerFilterParams {
    pub view_mode: Option<String>,
    pub brain_id: Option<String>,
    pub outcome_filter: Option<String>,
    pub exchange_index: Option<String>,
    pub symbols: Option<String>,
    pub min_price: Option<String>,
    pub max_price: Option<String>,
    pub min_hold_ms: Option<i64>,
    pub max_hold_ms: Option<i64>,
    pub min_confidence_pct: Option<f64>,
    pub primary_driver: Option<String>,
    pub scope: Option<String>,
}

impl LedgerFilterParams {
    pub fn into_hyper_filter(&self) -> HyperFilterQuery {
        let brain_id = self.brain_id.as_deref().and_then(parse_brain_id);
        let primary_driver = self.primary_driver.as_deref().and_then(parse_driver);
        let symbols = self.symbols.as_ref().map(|s| {
            s.split(',')
                .map(|tok| tok.trim().to_uppercase())
                .filter(|tok| !tok.is_empty())
                .collect::<HashSet<String>>()
        });
        let min_entry_price = self
            .min_price
            .as_deref()
            .and_then(|s| SubPennyPrice::from_str(s).ok());
        let max_entry_price = self
            .max_price
            .as_deref()
            .and_then(|s| SubPennyPrice::from_str(s).ok());

        HyperFilterQuery {
            view_mode: self.view_mode.clone(),
            brain_id,
            outcome_filter: self.outcome_filter.clone(),
            exchange_index: self.exchange_index.clone(),
            symbols,
            min_entry_price,
            max_entry_price,
            min_hold_duration_ms: self.min_hold_ms,
            max_hold_duration_ms: self.max_hold_ms,
            min_confidence_pct: self.min_confidence_pct,
            primary_driver,
        }
    }
}

fn parse_brain_id(code: &str) -> Option<BrainHorizonId> {
    match code.to_uppercase().as_str() {
        "BRAIN-01" | "PENNY" => Some(BrainHorizonId::Brain01PennyAnomaly),
        "BRAIN-02" | "DAY_TRADE" => Some(BrainHorizonId::Brain02IntradayDayTrade),
        "BRAIN-03" | "SHORT_SWING" => Some(BrainHorizonId::Brain03ShortTermSwing),
        "BRAIN-04" | "CORE_SWING" => Some(BrainHorizonId::Brain04CoreSwing),
        "BRAIN-05" | "LONG_TERM" => Some(BrainHorizonId::Brain05LongTermWealth),
        _ => None,
    }
}

fn parse_driver(code: &str) -> Option<PrimaryFingerprintDriver> {
    match code.to_uppercase().as_str() {
        "VOLUME_ABSORPTION" => Some(PrimaryFingerprintDriver::VolumeAbsorption),
        "GAMMA_SQUEEZE" => Some(PrimaryFingerprintDriver::GammaSqueeze),
        "SHORT_SQUEEZE" => Some(PrimaryFingerprintDriver::ShortSqueeze),
        "INTERNALS_ALIGNMENT" => Some(PrimaryFingerprintDriver::InternalsAlignment),
        _ => None,
    }
}

async fn ledger_snapshot_handler(
    State(state): State<AppState>,
    Query(params): Query<LedgerFilterParams>,
) -> Json<FilteredLedgerSnapshot> {
    let query = params.into_hyper_filter();
    Json(state.ledger.query_filtered_snapshot(&query))
}

async fn get_open_position_handler(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<Uuid>,
) -> Result<Json<OpenPositionRow>, StatusCode> {
    state
        .ledger
        .find_open_position(id)
        .map(Json)
        .ok_or(StatusCode::NOT_FOUND)
}

async fn get_closed_trade_handler(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<Uuid>,
) -> Result<Json<ClosedTradeRow>, StatusCode> {
    state
        .ledger
        .find_closed_trade(id)
        .map(Json)
        .ok_or(StatusCode::NOT_FOUND)
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EvaluateSymbolResponse {
    pub inference: LiveBrainInferenceOutput,
    pub precedents: PrecedentMatchSummary,
    pub opened_position: OpenPositionRow,
}

async fn evaluate_symbol_handler(
    State(state): State<AppState>,
    AxumPath((brain_str, symbol)): AxumPath<(String, String)>,
) -> Result<Json<EvaluateSymbolResponse>, (StatusCode, String)> {
    let brain_id = parse_brain_id(&brain_str).unwrap_or(BrainHorizonId::Brain02IntradayDayTrade);

    let unified_bars = state
        .data_bus
        .fetch_historical_bars_with_failover(&symbol, Timeframe::M5, 32)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let raw_bars: Vec<RawBarRecord> = unified_bars.iter().map(|b| b.to_raw_bar_record()).collect();

    let opt_chain = state
        .data_bus
        .fetch_option_chain_with_failover(&symbol)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let raw_opts = opt_chain.to_raw_options_snapshot();

    let snaps = state
        .data_bus
        .fetch_universe_snapshots_with_failover(Some(std::slice::from_ref(&symbol)))
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let first_snap = snaps.first();
    let raw_sf = first_snap
        .map(|s| s.to_raw_short_float_snapshot())
        .unwrap_or_default();
    let is_opt = first_snap.map(|s| s.is_optionable).unwrap_or(true);
    let exchange_idx = first_snap
        .map(|s| s.exchange_index.as_str())
        .unwrap_or("SP_500");

    let internals = state
        .data_bus
        .fetch_market_internals_with_failover()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .to_raw_internals_snapshot("SMH");

    let inference = state
        .onnx_engine
        .evaluate_live_signal(
            &symbol, brain_id, is_opt, &raw_bars, &raw_opts, &raw_sf, &internals,
        )
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let precedents =
        state
            .precedent_knn
            .query_precedents(brain_id, &inference.signal.latent_embedding_128, 25);
    let opened_position =
        state
            .ledger
            .open_position(exchange_idx, &inference, &precedents, 10_000.0);

    Ok(Json(EvaluateSymbolResponse {
        inference,
        precedents,
        opened_position,
    }))
}

async fn replay_simulate_handler(
    State(state): State<AppState>,
) -> Result<Json<FilteredLedgerSnapshot>, (StatusCode, String)> {
    let snap = state
        .run_historical_replay_simulation()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(snap))
}

async fn export_csv_handler(
    State(state): State<AppState>,
    Query(params): Query<LedgerFilterParams>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let scope = ExportScope::from_query_str(params.scope.as_deref());
    let filter = match scope {
        ExportScope::FullMasterDatabase => HyperFilterQuery::default(),
        ExportScope::PositionsOpenedOnly => HyperFilterQuery {
            view_mode: Some("POSITIONS_OPENED".to_string()),
            ..params.into_hyper_filter()
        },
        ExportScope::ActiveFilteredSlice => params.into_hyper_filter(),
    };

    let snapshot = state.ledger.query_filtered_snapshot(&filter);
    let csv_bytes = export_to_csv_bytes(&snapshot, scope)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/csv; charset=utf-8"),
    );
    headers.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_static("attachment; filename=\"argon_quant_ledger_export.csv\""),
    );
    Ok((headers, csv_bytes))
}

async fn export_xlsx_handler(
    State(state): State<AppState>,
    Query(params): Query<LedgerFilterParams>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let scope = ExportScope::from_query_str(params.scope.as_deref());
    let filter = match scope {
        ExportScope::FullMasterDatabase => HyperFilterQuery::default(),
        ExportScope::PositionsOpenedOnly => HyperFilterQuery {
            view_mode: Some("POSITIONS_OPENED".to_string()),
            ..params.into_hyper_filter()
        },
        ExportScope::ActiveFilteredSlice => params.into_hyper_filter(),
    };

    let snapshot = state.ledger.query_filtered_snapshot(&filter);
    let xlsx_bytes = export_to_xlsx_bytes(&snapshot)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
    );
    headers.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_static("attachment; filename=\"argon_quant_master_workbook.xlsx\""),
    );
    Ok((headers, xlsx_bytes))
}
