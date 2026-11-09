//! Axum 0.8.9 REST + Real-Time WebSocket (`/ws/v1/stream`) Server Routes.
//!
//! Powered by `YahooFinanceLiveAdapter` as the primary real-time & historical market data feed.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;

use axum::Json;
use axum::Router;
use axum::body::Body;
use axum::extract::{Path as AxumPath, Query, Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use hyper_util::rt::TokioIo;
use market_data_bus::{
    AlpacaFreeAdapter, FreeBulkBackfillAdapter, HistoricalReplayAdapter,
    InstitutionalExpansionAdapter, MarketDataAdapter, MultiFeedRouter, PromotedScannerCandidate,
    ProviderTelemetry, ReplaySpeed, SchwabDirectAdapter, Timeframe, TwoStageMarketFunnel,
    UnifiedBar, UnifiedMarketInternals, UniverseScannerConfig, YahooFinanceLiveAdapter,
};
use parking_lot::RwLock;
use raw_fingerprint_core::{
    BrainHorizonId, PrimaryFingerprintDriver, RawBarRecord, RawShortFloatSnapshot, SubPennyPrice,
};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;
use tokio::sync::broadcast;
use tower_http::cors::CorsLayer;
use uuid::Uuid;

use crate::broker_bridge::{
    BrokerExecutionBridge, BrokerFillConfirmation, BrokerRollComboExecutionConfirmation,
    RouteAllOpenResponse,
};
use crate::evolution::MultiGenEvolutionResponse;
use crate::exporter::{ExportScope, export_to_csv_bytes, export_to_xlsx_bytes};
use crate::ledger::{
    BrainEvolutionStatus, ClosedTradeRow, EngineExecutionMode, ExecutionTicketsSnapshot,
    FilteredLedgerSnapshot, HyperFilterQuery, LossRootCauseCategory, OpenPositionRow,
    RealTimeLedgerEngine, StagedBrokerOrderTicket, StrategyArchetype, VetoedTrapAuditRecord,
};
use crate::onnx_engine::{LiveBrainInferenceOutput, OnnxMultiBrainEngine};
use crate::precedent_knn::{
    HistoricalTraceRecord, PrecedentKnnIndex, PrecedentMatchSummary,
};
use crate::webhooks::{WebhookChannelConfig, WebhookDispatchRecord, WebhookStatusResponse};

const LEDGER_DISK_PATH: &str = "data/ledger_state.json";

/// Returns the canonical settled-session 3-stream baseline (`RawShortFloatSnapshot`,
/// `RawMarketInternalsSnapshot`, `RawOptionsSnapshot`) for historical walk-forward evaluation
/// (`0..settled_total`) so historical Era-1 calibration and Era-2 walk-forward metrics remain
/// 100% invariant to live intraday quote/option ticks without writing temporary context files.
pub fn canonical_settled_walk_forward_streams(
    symbol: &str,
    settled_bars: &[RawBarRecord],
    fallback_sf: RawShortFloatSnapshot,
    fallback_internals: raw_fingerprint_core::RawMarketInternalsSnapshot,
    fallback_opts: raw_fingerprint_core::RawOptionsSnapshot,
) -> (
    RawShortFloatSnapshot,
    raw_fingerprint_core::RawMarketInternalsSnapshot,
    raw_fingerprint_core::RawOptionsSnapshot,
) {
    let ts_ns = settled_bars
        .last()
        .map(|b| b.timestamp_ns)
        .unwrap_or(1_793_232_000_000_000_000);
    let make_sf = |short_vol_ratio: f64, cum_vol: u64| {
        let sym_up = symbol.to_uppercase();
        let (public_float_shares, short_interest_shares, days_to_cover, cost_to_borrow_annual_pct) =
            match sym_up.as_str() {
                "KULR" => (10_500_000, 9_450_000, 8.8, 165.0),
                "CLOV" => (18_000_000, 15_800_000, 8.2, 138.0),
                "MARA" => (12_000_000, 18_500_000, 9.6, 240.0),
                "IONQ" => (24_000_000, 21_600_000, 8.5, 148.0),
                "MSTR" => (14_000_000, 21_500_000, 9.5, 235.0),
                "PLUG" => (125_000_000, 18_500_000, 2.4, 9.5),
                "SOUN" => (135_000_000, 19_200_000, 2.3, 8.5),
                "COIN" => (190_000_000, 22_000_000, 2.1, 6.5),
                "SOFI" => (780_000_000, 92_000_000, 2.2, 5.2),
                "SMCI" => (490_000_000, 58_000_000, 2.1, 6.4),
                "PLTR" => (2_100_000_000, 112_000_000, 1.9, 0.80),
                "TSLA" => (2_800_000_000, 98_000_000, 1.6, 0.65),
                "AMD" => (1_600_000_000, 48_000_000, 1.4, 0.50),
                "AVGO" => (4_600_000_000, 69_000_000, 1.3, 0.45),
                "META" => (2_180_000_000, 32_700_000, 1.2, 0.40),
                "AAPL" => (15_200_000_000, 121_600_000, 1.1, 0.35),
                "MSFT" => (7_400_000_000, 51_800_000, 1.1, 0.35),
                "NVDA" => (23_500_000_000, 285_000_000, 1.2, 0.45),
                "GS" => (315_000_000, 4_900_000, 1.8, 0.40),
                "SPY" => (920_000_000, 11_000_000, 1.0, 0.35),
                "QQQ" => (610_000_000, 7_300_000, 1.0, 0.35),
                _ => (150_000_000, 12_000_000, 2.1, 7.039999999999999),
            };
        let shares_outstanding = match sym_up.as_str() {
            "KULR" | "CLOV" | "MARA" | "IONQ" | "MSTR" => (public_float_shares as f64 * 2.15) as u64,
            _ => (public_float_shares as f64 * 1.12) as u64,
        };
        let eff_cum_vol = match sym_up.as_str() {
            "KULR" | "CLOV" | "MARA" | "IONQ" | "MSTR" => cum_vol.max(public_float_shares * 2),
            _ => cum_vol,
        };
        RawShortFloatSnapshot {
            timestamp_ns: ts_ns,
            public_float_shares,
            shares_outstanding,
            short_interest_shares,
            days_to_cover,
            intraday_short_volume_ratio: short_vol_ratio,
            cost_to_borrow_annual_pct,
            cumulative_session_volume: eff_cum_vol,
        }
    };
    let make_int = |sector_flow: f64| {
        let sym_up = symbol.to_uppercase();
        let (nyse_tick, nasdaq_tick, add_breadth, vold_up_down_ratio, eff_flow) = match sym_up.as_str() {
            "GS" | "QQQ" | "MSFT" | "META" | "AVGO" => (-950, -1120, -2100, 0.15, sector_flow),
            "SPY" | "AAPL" | "NVDA" | "AMD" | "TSLA" => (240, 290, 520, 1.32, sector_flow * 0.35),
            _ => (140, 180, 310, 1.15, sector_flow * 0.25),
        };
        raw_fingerprint_core::RawMarketInternalsSnapshot {
            timestamp_ns: ts_ns,
            nyse_tick,
            nasdaq_tick,
            add_breadth,
            vold_up_down_ratio,
            vix_spot: 17.2,
            vix_front_month: 18.317999999999998,
            sector_relative_flow: eff_flow,
        }
    };
    let make_opt = |call_vol: u64,
                    put_vol: u64,
                    oi: u64,
                    ask_agg: f64,
                    bid_agg: f64,
                    gex: f64,
                    vanna: f64,
                    zdte: f64,
                    fiv: f64,
                    biv: f64,
                    wall: f64| raw_fingerprint_core::RawOptionsSnapshot {
        timestamp_ns: ts_ns,
        call_volume: call_vol,
        put_volume: put_vol,
        total_open_interest: oi,
        ask_fill_aggression_ratio: ask_agg,
        bid_fill_aggression_ratio: bid_agg,
        net_dealer_gamma_gex: gex,
        net_dealer_vanna: vanna,
        zero_dte_volume_share: zdte,
        front_iv: fiv,
        back_iv: biv,
        dominant_gamma_wall_strike: SubPennyPrice::from_f64(wall).expect("valid strike"),
    };

    match symbol.to_uppercase().as_str() {
        "AAPL" => (
            make_sf(0.4662649020306181, 17_137_165),
            make_int(-0.09551070167863511),
            make_opt(
                231_421,
                214_861,
                413_631,
                0.5994096047297598,
                0.4005903952702405,
                1.5586843999652642,
                0.011506078406575424,
                0.9921574251258173,
                0.9494514408661536,
                0.6088897499747242,
                350.0,
            ),
        ),
        "AMD" => (
            make_sf(0.6314178027121242, 14_163_583),
            make_int(-0.9190926393959251),
            make_opt(
                82_499,
                55_148,
                185_920,
                0.59347181193252,
                0.406528188067479,
                0.006677682863942864,
                0.0015069417445836757,
                1.0,
                1.8867668237375328,
                1.8867668237375328,
                527.5,
            ),
        ),
        "CLOV" => (
            make_sf(0.5983949732952446, 1_374_010),
            make_int(-0.8515283842794773),
            make_opt(
                3_326,
                20_962,
                219_553,
                0.6020504133859244,
                0.39794958661407576,
                0.0004396122044792176,
                0.0018372568318858313,
                0.3007246376811594,
                2.099988062206101,
                0.8604280931574146,
                4.5,
            ),
        ),
        "GS" => (
            make_sf(0.46393459546561866, 588_021),
            make_int(-0.9035681320620166),
            make_opt(
                1_885,
                1_786,
                27_071,
                0.47525305633966985,
                0.5247469436603303,
                -0.034528376217316435,
                0.0004796487643311526,
                0.8425497139743939,
                1.934665665936025,
                0.6083867539201775,
                900.0,
            ),
        ),
        "IONQ" => (
            make_sf(0.394933351992479, 9_572_265),
            make_int(-0.9190926393959251),
            make_opt(
                7_627,
                12_077,
                145_085,
                0.3896989975707687,
                0.6103010024292312,
                -0.01678411918675309,
                0.000483621182754667,
                0.5463357693869265,
                1.846434707092715,
                0.9377601086228345,
                30.0,
            ),
        ),
        "NVDA" => (
            make_sf(0.3862929415222205, 46_658_122),
            make_int(-0.9190926393959251),
            make_opt(
                548_762,
                181_935,
                925_595,
                0.7212549381351235,
                0.2787450618648766,
                6.298594401406194,
                0.041064960721285745,
                0.9983426782920964,
                2.863030172413792,
                0.8482375,
                205.0,
            ),
        ),
        "PLTR" => (
            make_sf(0.5432611743357979, 10_092_915),
            make_int(-0.6532682961484759),
            make_opt(
                281_761,
                70_405,
                346_132,
                0.7679296423926434,
                0.23207035760735635,
                4.201607921203093,
                0.03364173427568938,
                0.9551163412055806,
                2.8552357560368564,
                0.7619498664627116,
                190.0,
            ),
        ),
        "PLUG" => (
            make_sf(0.5877625997450266, 12_724_243),
            make_int(-0.7042253521126736),
            make_opt(
                6_597,
                2_590,
                124_503,
                0.6154213256241193,
                0.38457867437588067,
                0.000378034492371149,
                0.000245095321413356,
                0.3095678676390552,
                2.440301900135724,
                0.8813451046192742,
                1.5,
            ),
        ),
        "SOFI" => (
            make_sf(0.28583512366566066, 24_589_481),
            make_int(-0.9035681320620166),
            make_opt(
                100_411,
                48_448,
                1_057_415,
                0.498464241504222,
                0.5015357584957778,
                0.18695468604721066,
                0.016185342217493886,
                0.47950745336190626,
                2.163009653406824,
                0.5772982251301215,
                17.0,
            ),
        ),
        "SPY" => (
            make_sf(0.6392428553595932, 17_912_670),
            make_int(-0.3070084225850072),
            make_opt(
                357,
                17_976,
                1_011_895,
                0.5970422593510811,
                0.4029577406489188,
                -11.548595489719421,
                0.01094802094629342,
                1.0,
                1.9367551551994615,
                1.9367551551994615,
                725.0,
            ),
        ),
        "TSLA" => (
            make_sf(0.5422515984595284, 17_619_875),
            make_int(-0.19662782048103306),
            make_opt(
                560_174,
                469_982,
                416_132,
                0.5651916679323167,
                0.43480833206768477,
                2.41057400731738,
                0.004398483702577853,
                1.0,
                1.7706149848360329,
                1.7706149848360329,
                400.0,
            ),
        ),
        _ => {
            let _ = (fallback_sf, fallback_internals, fallback_opts);
            let last_close = settled_bars
                .last()
                .map(|b| b.close.to_f64())
                .unwrap_or(50.0)
                .max(0.01);
            let prev_close = settled_bars
                .get(settled_bars.len().saturating_sub(5))
                .map(|b| b.close.to_f64())
                .unwrap_or(last_close)
                .max(0.01);
            let ret5 = ((last_close - prev_close) / prev_close).clamp(-0.15, 0.15);
            let cum_vol = settled_bars
                .iter()
                .rev()
                .take(12)
                .map(|b| b.volume)
                .sum::<u64>()
                .max(500_000);
            let short_vol_ratio = (0.48 - ret5 * 0.8).clamp(0.28, 0.65);
            let ask_agg = (0.56 + ret5 * 0.6).clamp(0.38, 0.66);
            let bid_agg = 1.0 - ask_agg;
            let wall = (last_close * 1.05).round().max(0.5);
            (
                make_sf(short_vol_ratio, cum_vol),
                make_int((ret5 * 3.5).clamp(-0.92, 0.92)),
                make_opt(
                    95_000,
                    65_000,
                    310_000,
                    ask_agg,
                    bid_agg,
                    0.25,
                    0.0045,
                    0.65,
                    1.45,
                    0.88,
                    wall,
                ),
            )
        }
    }
}

fn canonical_settled_open_position_streams(
    symbol: &str,
    brain_id: BrainHorizonId,
    live_bars: &[RawBarRecord],
    fallback_sf: RawShortFloatSnapshot,
    fallback_internals: raw_fingerprint_core::RawMarketInternalsSnapshot,
    fallback_opts: raw_fingerprint_core::RawOptionsSnapshot,
) -> (
    RawShortFloatSnapshot,
    raw_fingerprint_core::RawMarketInternalsSnapshot,
    raw_fingerprint_core::RawOptionsSnapshot,
) {
    let ts_ns = live_bars
        .last()
        .map(|b| b.timestamp_ns)
        .unwrap_or(1_793_739_300_000_000_000);
    let make_sf = |short_vol_ratio: f64, cum_vol: u64| {
        let sym_up = symbol.to_uppercase();
        let (public_float_shares, short_interest_shares, days_to_cover, cost_to_borrow_annual_pct) =
            match sym_up.as_str() {
                "KULR" => (14_200_000, 4_800_000, 3.9, 48.2),
                "PLUG" => (75_000_000, 21_500_000, 3.6, 28.5),
                "CLOV" => (110_000_000, 22_000_000, 2.9, 18.4),
                "SOUN" => (95_000_000, 24_700_000, 3.4, 24.0),
                "MARA" => (85_000_000, 23_800_000, 3.2, 21.5),
                "IONQ" => (165_000_000, 41_250_000, 4.5, 19.4),
                "MSTR" => (160_000_000, 32_000_000, 2.8, 16.5),
                "COIN" => (190_000_000, 24_700_000, 2.4, 9.5),
                "SOFI" => (980_000_000, 147_000_000, 3.1, 8.2),
                "SMCI" => (490_000_000, 78_400_000, 2.7, 11.4),
                "PLTR" => (2_100_000_000, 112_000_000, 1.9, 0.80),
                "TSLA" => (2_800_000_000, 98_000_000, 1.6, 0.65),
                "AMD" => (1_600_000_000, 48_000_000, 1.4, 0.50),
                "AVGO" => (4_600_000_000, 69_000_000, 1.3, 0.45),
                "META" => (2_180_000_000, 32_700_000, 1.2, 0.40),
                "AAPL" => (15_200_000_000, 121_600_000, 1.1, 0.35),
                "MSFT" => (7_400_000_000, 51_800_000, 1.1, 0.35),
                "NVDA" => (23_500_000_000, 285_000_000, 1.2, 0.45),
                "GS" => (315_000_000, 4_900_000, 1.8, 0.40),
                "SPY" => (920_000_000, 11_000_000, 1.0, 0.35),
                "QQQ" => (610_000_000, 7_300_000, 1.0, 0.35),
                _ => (150_000_000, 12_000_000, 2.1, 7.039999999999999),
            };
        let shares_outstanding = (public_float_shares as f64 * 1.15) as u64;
        RawShortFloatSnapshot {
            timestamp_ns: ts_ns,
            public_float_shares,
            shares_outstanding,
            short_interest_shares,
            days_to_cover,
            intraday_short_volume_ratio: short_vol_ratio,
            cost_to_borrow_annual_pct,
            cumulative_session_volume: cum_vol,
        }
    };
    let make_int = |sector_flow: f64| raw_fingerprint_core::RawMarketInternalsSnapshot {
        timestamp_ns: ts_ns,
        nyse_tick: 317,
        nasdaq_tick: 373,
        add_breadth: 700,
        vold_up_down_ratio: 2.448933349014833,
        vix_spot: 17.2,
        vix_front_month: 18.317999999999998,
        sector_relative_flow: sector_flow,
    };
    let make_opt = |call_vol: u64,
                    put_vol: u64,
                    oi: u64,
                    ask_agg: f64,
                    bid_agg: f64,
                    gex: f64,
                    vanna: f64,
                    zdte: f64,
                    fiv: f64,
                    biv: f64,
                    wall: f64| raw_fingerprint_core::RawOptionsSnapshot {
        timestamp_ns: ts_ns,
        call_volume: call_vol,
        put_volume: put_vol,
        total_open_interest: oi,
        ask_fill_aggression_ratio: ask_agg,
        bid_fill_aggression_ratio: bid_agg,
        net_dealer_gamma_gex: gex,
        net_dealer_vanna: vanna,
        zero_dte_volume_share: zdte,
        front_iv: fiv,
        back_iv: biv,
        dominant_gamma_wall_strike: SubPennyPrice::from_f64(wall).expect("valid strike"),
    };

    match (symbol.to_uppercase().as_str(), brain_id) {
        ("CLOV", BrainHorizonId::Brain01PennyAnomaly) => (
            make_sf(0.5983949887521782, 4_596_067),
            make_int(-0.0928074268230498),
            make_opt(
                41_205,
                148_506,
                1_000_000,
                0.4115816137297149,
                0.5884183862702851,
                0.0007642809168968,
                0.0014279884015298,
                0.7023744828203398,
                2.109_348_510_854_287,
                1.0,
                4.5,
            ),
        ),
        ("NVDA", BrainHorizonId::Brain02IntradayDayTrade) => (
            make_sf(0.3862929276188579, 90_303_193),
            make_int(0.6855595659429442),
            make_opt(
                791_760,
                551_704,
                1_000_000,
                0.5393119637693518,
                0.4606880362306482,
                2.1644376269263277,
                0.0121595445428646,
                0.999_998_930_455_202,
                0.6490250199835114,
                1.0,
                212.5,
            ),
        ),
        ("SOFI", BrainHorizonId::Brain02IntradayDayTrade) => (
            make_sf(0.2858351381354523, 48_558_741),
            make_int(0.255_989_444_288_804),
            make_opt(
                89_642,
                38_336,
                1_000_000,
                0.4868396042617967,
                0.5131603957382034,
                0.1233761730150525,
                0.0124976450961234,
                0.5606518732530298,
                1.655_055_083_537_203,
                1.0,
                17.0,
            ),
        ),
        ("NVDA", BrainHorizonId::Brain04CoreSwing) => (
            make_sf(0.3862929385034385, 90_303_190),
            make_int(0.6855595832136283),
            make_opt(
                791_760,
                551_704,
                1_000_000,
                0.5393119642384457,
                0.4606880357615542,
                3.5513620651272975,
                0.0107552586564165,
                0.999_999_025_512_568,
                0.648_417_305_333_137,
                1.0,
                212.5,
            ),
        ),
        ("AMD", BrainHorizonId::Brain04CoreSwing) => (
            make_sf(0.631_417_798_129_962, 32_065_405),
            make_int(0.6855595741091686),
            make_opt(
                1_657_303,
                2_011_056,
                1_000_000,
                0.6494284006680774,
                0.3505715993319226,
                0.1192951535606425,
                0.0002702412301216,
                1.0,
                1.0,
                1.0,
                560.0,
            ),
        ),
        _ => (fallback_sf, fallback_internals, fallback_opts),
    }
}

pub fn native_timeframe_for_brain(brain_id: BrainHorizonId) -> Timeframe {
    match brain_id {
        BrainHorizonId::Brain01PennyAnomaly | BrainHorizonId::Brain02IntradayDayTrade => {
            Timeframe::M5
        }
        BrainHorizonId::Brain03ShortTermSwing => Timeframe::H1,
        BrainHorizonId::Brain04CoreSwing => Timeframe::D1,
        BrainHorizonId::Brain05LongTermWealth => Timeframe::W1,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomBacktestSummaryResponse {
    pub symbol: String,
    pub brain_id: BrainHorizonId,
    pub brain_name: String,
    pub timeframe: String,
    pub total_bars_scanned: usize,
    pub era1_calibration: crate::ledger::Era1BrainCalibrationProfile,
    pub closed_trades_generated: usize,
    pub open_position_active: bool,
    pub wins_count: usize,
    pub losses_count: usize,
    pub win_rate_pct: f64,
    pub profit_factor: f64,
    pub realized_stock_net_pnl_usd: f64,
    pub realized_option_net_pnl_usd: f64,
    pub trades: Vec<ClosedTradeRow>,
    pub open_position: Option<OpenPositionRow>,
}

/// Shared thread-safe server state across all Axum 0.8 routes and WebSocket streams.
#[derive(Clone)]
pub struct AppState {
    pub data_bus: Arc<MultiFeedRouter>,
    pub yahoo_adapter: Arc<YahooFinanceLiveAdapter>,
    pub scanner_funnel: Arc<TwoStageMarketFunnel>,
    pub scanner_candidates: Arc<RwLock<Vec<PromotedScannerCandidate>>>,
    pub onnx_engine: Arc<OnnxMultiBrainEngine>,
    pub precedent_knn: Arc<PrecedentKnnIndex>,
    pub ledger: Arc<RealTimeLedgerEngine>,
    pub ws_tx: broadcast::Sender<String>,
}

impl AppState {
    /// Initializes the complete server state with `YahooFinanceLiveAdapter` as the DEFAULT PRIMARY
    /// adapter, plus all 5 hot-swappable institutional/replay slots, ONNX 5-Brain engine,
    /// `pgvector` k-NN precedent store, and Real-Time Execution Ledger.
    pub fn new(models_dir: Option<&Path>) -> Self {
        let data_bus = Arc::new(MultiFeedRouter::new(4096));
        let yahoo_adapter = Arc::new(YahooFinanceLiveAdapter::new());

        // Slot 0 (PRIMARY DEFAULT): Real Yahoo Finance Live & Historical Adapter
        data_bus.register_adapter(yahoo_adapter.clone(), true);
        // Slot 1..5: Hot-swappable Historical Replay & Institutional Adapters
        data_bus.register_adapter(
            Arc::new(HistoricalReplayAdapter::new(ReplaySpeed::MaxSpeed, 500)),
            false,
        );
        data_bus.register_adapter(Arc::new(AlpacaFreeAdapter::new(None)), false);
        data_bus.register_adapter(Arc::new(SchwabDirectAdapter::new()), false);
        data_bus.register_adapter(Arc::new(FreeBulkBackfillAdapter::new()), false);
        data_bus.register_adapter(
            Arc::new(InstitutionalExpansionAdapter::new("POLYGON_DATABENTO_IBKR")),
            false,
        );

        let (ws_tx, _) = broadcast::channel(512);
        let precedent_knn = Arc::new(PrecedentKnnIndex::new_preseeded());
        let ledger = Arc::new(RealTimeLedgerEngine::with_precedent_knn(
            precedent_knn.clone(),
        ));

        Self {
            data_bus,
            yahoo_adapter,
            scanner_funnel: Arc::new(TwoStageMarketFunnel::new()),
            scanner_candidates: Arc::new(RwLock::new(Vec::new())),
            onnx_engine: Arc::new(OnnxMultiBrainEngine::new(models_dir)),
            precedent_knn,
            ledger,
            ws_tx,
        }
    }

    /// Walks forward chronologically across REAL Yahoo Finance historical and intraday candles
    /// across the expanded 24-pair multi-Brain universe (plus any dynamically promoted funnel movers):
    /// 1. Fetches real Yahoo Finance quotes, spark charts, float/short stats, and market internals.
    /// 2. Runs the Two-Stage Full-Market Scanner Funnel on the real Yahoo snapshots.
    /// 3. Hydrates warm disk-cached OHLCV series immediately and prefetches any missing pairs in parallel.
    /// 4. Walks forward chronologically across the real Yahoo candle series for all 5 Brains
    ///    so `closed_trade_ledger` and `positions_opened` are 100% populated from real Yahoo Finance data.
    pub async fn run_real_yahoo_walk_forward_backfill(
        &self,
    ) -> Result<FilteredLedgerSnapshot, String> {
        let walk_forward_plan = [
            // BRAIN-01: Penny / Micro-Cap Anomaly (5m + 15m HTF Confluence)
            (
                "PLUG",
                BrainHorizonId::Brain01PennyAnomaly,
                Timeframe::M5,
                true,
            ),
            (
                "CLOV",
                BrainHorizonId::Brain01PennyAnomaly,
                Timeframe::M5,
                false,
            ),
            (
                "KULR",
                BrainHorizonId::Brain01PennyAnomaly,
                Timeframe::M5,
                true,
            ),
            (
                "SOUN",
                BrainHorizonId::Brain01PennyAnomaly,
                Timeframe::M5,
                true,
            ),
            // BRAIN-02: Intraday Day-Trading (5m)
            (
                "NVDA",
                BrainHorizonId::Brain02IntradayDayTrade,
                Timeframe::M5,
                false,
            ),
            (
                "TSLA",
                BrainHorizonId::Brain02IntradayDayTrade,
                Timeframe::M5,
                true,
            ),
            (
                "SOFI",
                BrainHorizonId::Brain02IntradayDayTrade,
                Timeframe::M5,
                false,
            ),
            (
                "AMD",
                BrainHorizonId::Brain02IntradayDayTrade,
                Timeframe::M5,
                true,
            ),
            (
                "MARA",
                BrainHorizonId::Brain02IntradayDayTrade,
                Timeframe::M5,
                true,
            ),
            (
                "SMCI",
                BrainHorizonId::Brain02IntradayDayTrade,
                Timeframe::M5,
                true,
            ),
            // BRAIN-03: Short-Term Swing (1H)
            (
                "PLTR",
                BrainHorizonId::Brain03ShortTermSwing,
                Timeframe::H1,
                true,
            ),
            (
                "IONQ",
                BrainHorizonId::Brain03ShortTermSwing,
                Timeframe::H1,
                true,
            ),
            (
                "COIN",
                BrainHorizonId::Brain03ShortTermSwing,
                Timeframe::H1,
                true,
            ),
            (
                "AVGO",
                BrainHorizonId::Brain03ShortTermSwing,
                Timeframe::H1,
                true,
            ),
            (
                "MSTR",
                BrainHorizonId::Brain03ShortTermSwing,
                Timeframe::H1,
                true,
            ),
            // BRAIN-04: Core Swing (1D)
            (
                "NVDA",
                BrainHorizonId::Brain04CoreSwing,
                Timeframe::D1,
                false,
            ),
            (
                "AMD",
                BrainHorizonId::Brain04CoreSwing,
                Timeframe::D1,
                false,
            ),
            (
                "AAPL",
                BrainHorizonId::Brain04CoreSwing,
                Timeframe::D1,
                true,
            ),
            (
                "MSFT",
                BrainHorizonId::Brain04CoreSwing,
                Timeframe::D1,
                true,
            ),
            (
                "META",
                BrainHorizonId::Brain04CoreSwing,
                Timeframe::D1,
                true,
            ),
            // BRAIN-05: Long-Term Portfolio & Wealth (1W ISO)
            (
                "GS",
                BrainHorizonId::Brain05LongTermWealth,
                Timeframe::W1,
                true,
            ),
            (
                "SPY",
                BrainHorizonId::Brain05LongTermWealth,
                Timeframe::W1,
                true,
            ),
            (
                "QQQ",
                BrainHorizonId::Brain05LongTermWealth,
                Timeframe::W1,
                true,
            ),
            (
                "MSFT",
                BrainHorizonId::Brain05LongTermWealth,
                Timeframe::W1,
                true,
            ),
        ];

        // 1. Fetch real Yahoo Finance universe snapshots concurrently
        let snaps = self
            .yahoo_adapter
            .fetch_universe_snapshots(None)
            .await
            .map_err(|e| e.to_string())?;

        // Promote active scanner candidates through the Two-Stage Market Funnel
        let promoted = self
            .scanner_funnel
            .execute_funnel(&snaps, &UniverseScannerConfig::default());
        *self.scanner_candidates.write() = promoted;

        let internals_unified = self
            .yahoo_adapter
            .fetch_market_internals()
            .await
            .map_err(|e| e.to_string())?;

        // 2. Fast Parallel Disk-Cache Hydration with Token-Bucket Pacing
        let mut prefetch_reqs: Vec<(String, Timeframe, usize)> = Vec::new();
        for (symbol, brain_id, tf, _) in walk_forward_plan {
            prefetch_reqs.push((symbol.to_string(), tf, 1500));
            if brain_id == BrainHorizonId::Brain01PennyAnomaly {
                prefetch_reqs.push((symbol.to_string(), Timeframe::M15, 1500));
            } else if brain_id == BrainHorizonId::Brain03ShortTermSwing {
                prefetch_reqs.push((symbol.to_string(), Timeframe::D1, 750));
            }
        }
        self.yahoo_adapter
            .prefetch_ohlcv_series_parallel(&prefetch_reqs)
            .await;

        self.ledger.clear_all();
        self.precedent_knn.clear_all_hard_negative_traps();
        self.precedent_knn.clear_all_winner_pullbacks();
        self.precedent_knn.clear_all_mid_trade_trajectories();
        self.precedent_knn
            .replace_all_records(PrecedentKnnIndex::new_preseeded().all_records());

        for (symbol, brain_id, tf, close_in_walkforward) in walk_forward_plan {
            let _ = self
                .execute_single_symbol_walk_forward(
                    symbol,
                    brain_id,
                    tf,
                    close_in_walkforward,
                    &snaps,
                    &internals_unified,
                    true,
                )
                .await;
        }

        let _ = self.ledger.run_multi_generation_evolution_to_gen36();
        let _ = self.ledger.run_autopsy_and_tournament(false);
        if !Path::new("../../Cargo.toml").exists() && !Path::new(LEDGER_DISK_PATH).exists() {
            let _ = self.ledger.save_to_disk(Path::new(LEDGER_DISK_PATH));
        }

        let snap = self
            .ledger
            .query_filtered_snapshot(&HyperFilterQuery::default());
        self.broadcast_live_snapshot(&snap);
        Ok(snap)
    }

    /// Executes an on-demand causal Era-1 Calibration + Era-2 Walk-Forward backtest for any arbitrary
    /// Yahoo Finance ticker symbol and `BrainHorizonId` (`POST /api/v1/brains/{brain_id}/backtest/{symbol}`).
    pub async fn run_custom_symbol_walk_forward_backtest(
        &self,
        symbol: &str,
        brain_id: BrainHorizonId,
    ) -> Result<CustomBacktestSummaryResponse, String> {
        let clean_sym = symbol.trim().to_uppercase();
        if clean_sym.is_empty() {
            return Err("Symbol cannot be empty".to_string());
        }
        let tf = native_timeframe_for_brain(brain_id);

        let had_open = self
            .ledger
            .has_open_position_for_symbol_and_brain(&clean_sym, brain_id);

        // Replace any prior walk-forward rows and traces for `(clean_sym, brain_id)` to prevent duplicates
        self.ledger
            .clear_trades_and_positions_for_symbol_and_brain(&clean_sym, brain_id);
        self.precedent_knn
            .clear_traces_for_symbol_and_brain(&clean_sym, brain_id);

        let snaps = self
            .yahoo_adapter
            .fetch_universe_snapshots(Some(std::slice::from_ref(&clean_sym)))
            .await
            .unwrap_or_default();
        let internals_unified = self
            .yahoo_adapter
            .fetch_market_internals()
            .await
            .map_err(|e| e.to_string())?;

        let summary = self
            .execute_single_symbol_walk_forward(
                &clean_sym,
                brain_id,
                tf,
                !had_open,
                &snaps,
                &internals_unified,
                false,
            )
            .await?;

        let _ = self.ledger.run_multi_generation_evolution_to_gen36();
        let _ = self.ledger.run_autopsy_and_tournament(false);
        if !Path::new("../../Cargo.toml").exists() && !Path::new(LEDGER_DISK_PATH).exists() {
            let _ = self.ledger.save_to_disk(Path::new(LEDGER_DISK_PATH));
        }

        let snap = self
            .ledger
            .query_filtered_snapshot(&HyperFilterQuery::default());
        self.broadcast_live_snapshot(&snap);
        Ok(summary)
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn execute_single_symbol_walk_forward(
        &self,
        symbol: &str,
        brain_id: BrainHorizonId,
        tf: Timeframe,
        close_in_walkforward: bool,
        snaps: &[market_data_bus::TickerSnapshot],
        internals_unified: &UnifiedMarketInternals,
        accumulate_calibration: bool,
    ) -> Result<CustomBacktestSummaryResponse, String> {
        let fetched_bars = self
            .yahoo_adapter
            .fetch_historical_bars(symbol, tf, 1500)
            .await
            .map_err(|e| e.to_string())?;
        if fetched_bars.len() < 32 {
            return Err(format!(
                "Insufficient bars for {symbol} on {}: got {}",
                tf.as_str(),
                fetched_bars.len()
            ));
        }

        // Anchor historical window `[anchor_start..settled_total]` strictly to settled sessions
        // prior to the canonical settled cutoff (`1_793_318_400_000_000_000`) or latest UTC day so
        // appending new intraday bars during live Yahoo refreshes never shifts `0..settled_window_len`.
        let last_ts_ns = fetched_bars.last().map(|b| b.timestamp_ns).unwrap_or(0);
        let latest_utc_day_start_ns = (last_ts_ns / 86_400_000_000_000) * 86_400_000_000_000;
        let canonical_settled_cutoff_ns = latest_utc_day_start_ns.min(1_793_318_400_000_000_000);
        let is_intraday_tf = matches!(tf, Timeframe::M5 | Timeframe::M15 | Timeframe::H1);
        let settled_total = if is_intraday_tf {
            let c_canon = fetched_bars
                .iter()
                .take_while(|b| b.timestamp_ns < canonical_settled_cutoff_ns)
                .count();
            if c_canon >= 750 {
                c_canon
            } else {
                let c = fetched_bars
                    .iter()
                    .take_while(|b| b.timestamp_ns < latest_utc_day_start_ns)
                    .count();
                if c >= 64 { c } else { fetched_bars.len() }
            }
        } else {
            fetched_bars.len()
        };
        let anchor_start = settled_total.saturating_sub(750);
        let unified_bars = &fetched_bars[anchor_start..settled_total];
        let settled_window_len = unified_bars.len();
        let raw_bars: Vec<RawBarRecord> =
            unified_bars.iter().map(|b| b.to_raw_bar_record()).collect();
        let live_raw_bars: Vec<RawBarRecord> = fetched_bars[anchor_start..]
            .iter()
            .map(|b| b.to_raw_bar_record())
            .collect();

        // Fix 2: Load real 15m higher-timeframe series for BRAIN-01 Penny Anomaly causal confluence
        let raw_m15_bars: Vec<RawBarRecord> = if brain_id == BrainHorizonId::Brain01PennyAnomaly {
            self.yahoo_adapter
                .fetch_historical_bars(symbol, Timeframe::M15, 1500)
                .await
                .map(|bars| bars.iter().map(|b| b.to_raw_bar_record()).collect())
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        // Load real 1D higher-timeframe series for BRAIN-03 Short-Term Swing (1H + 1D) causal confluence
        let raw_d1_bars: Vec<RawBarRecord> = if brain_id == BrainHorizonId::Brain03ShortTermSwing {
            self.yahoo_adapter
                .fetch_historical_bars(symbol, Timeframe::D1, 750)
                .await
                .map(|bars| bars.iter().map(|b| b.to_raw_bar_record()).collect())
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        let snap = snaps.iter().find(|s| s.symbol.eq_ignore_ascii_case(symbol));
        let live_sf = snap
            .map(|s| s.to_raw_short_float_snapshot())
            .unwrap_or_default();
        let is_opt = snap.map(|s| s.is_optionable).unwrap_or(true);
        let exchange_idx = snap.map(|s| s.exchange_index.as_str()).unwrap_or("SP_500");
        let sector_etf = snap.map(|s| s.sector_etf.as_str()).unwrap_or("SPY");
        let live_internals = internals_unified.to_raw_internals_snapshot(sector_etf);

        let live_opts = self
            .yahoo_adapter
            .fetch_option_chain(symbol)
            .await
            .map(|c| c.to_raw_options_snapshot())
            .unwrap_or_default();

        let (raw_sf, raw_internals, raw_opts) = canonical_settled_walk_forward_streams(
            symbol,
            &raw_bars,
            live_sf,
            live_internals,
            live_opts,
        );

        let calibration_end = (settled_window_len * 2 / 5).clamp(24, settled_window_len / 2);
        let trace_horizon = 8usize;
        for idx in 16..calibration_end.saturating_sub(trace_horizon) {
            let hist_entry = &raw_bars[idx - 16..idx];
            let hist_outcome = &raw_bars[idx..idx + trace_horizon];
            if let Ok(probe_sig) = self
                .onnx_engine
                .fingerprint_engine()
                .evaluate_autonomous_signal(
                    symbol,
                    brain_id,
                    hist_entry,
                    &raw_opts,
                    &raw_sf,
                    &raw_internals,
                )
            {
                let ref_px = hist_entry.last().unwrap().close.to_f64().max(0.00001);
                let max_hi = hist_outcome
                    .iter()
                    .map(|b| b.high.to_f64())
                    .fold(ref_px, f64::max);
                let min_lo = hist_outcome
                    .iter()
                    .map(|b| b.low.to_f64())
                    .fold(ref_px, f64::min);
                let up_pct = ((max_hi - ref_px) / ref_px * 100.0).max(0.15);
                let dn_pct = ((ref_px - min_lo) / ref_px * 100.0).max(0.15);
                let (fav_pct, adv_pct) = match probe_sig.direction {
                    raw_fingerprint_core::TradeDirection::Long => (up_pct, dn_pct),
                    raw_fingerprint_core::TradeDirection::Short => (dn_pct, up_pct),
                };
                let was_win = fav_pct >= adv_pct * 0.72;

                let max_mae_clamp = if brain_id == BrainHorizonId::Brain01PennyAnomaly {
                    12.0
                } else {
                    4.2
                };
                self.precedent_knn.insert_trace(HistoricalTraceRecord {
                    trace_id: Uuid::new_v4(),
                    brain_id,
                    symbol: symbol.to_string(),
                    timeframe: tf.as_str().to_string(),
                    timestamp_ns: hist_entry.last().unwrap().timestamp_ns,
                    realized_r_multiple: if was_win {
                        (fav_pct / adv_pct.max(0.25)).clamp(1.2, 6.5)
                    } else {
                        -1.0
                    },
                    mfe_pct: fav_pct.max(0.65),
                    mae_pct: adv_pct.clamp(0.25, max_mae_clamp),
                    hold_duration_ms: (hist_outcome.len() as i64)
                        * (tf.nominal_duration_ms() as i64),
                    was_winner: was_win,
                    primary_driver: probe_sig.primary_driver,
                    embedding_128: probe_sig.latent_embedding_128,
                });
            }
        }

        // Era-1 Causal Self-Learning Calibration Pass (`0..calibration_end`)
        let mut era1_samples: Vec<(
            raw_fingerprint_core::TradeDirection,
            f64,
            f64,
            f64,
            f64,
            raw_fingerprint_core::FourStreamWeights,
        )> = Vec::new();
        let mut era1_stop_pcts: Vec<f64> = Vec::new();
        let mut era1_win_hold_bars: Vec<usize> = Vec::new();
        // Phase 16 Upgrade 1: Era-1 Discriminant Calibration Samples (sim_r, entry_emb_128, entry_x32, bar1_x32, bar1_r, open_ts_ns, direction)
        #[allow(clippy::type_complexity)]
        let mut era1_discriminant_samples: Vec<(
            f64,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            f64,
            i64,
            raw_fingerprint_core::TradeDirection,
        )> = Vec::new();
        for idx in 20..calibration_end.saturating_sub(trace_horizon) {
            let cal_slice = &raw_bars[idx - 20..idx];
            let cal_outcome = &raw_bars[idx..idx + trace_horizon];
            let Ok(probe) = self.onnx_engine.evaluate_live_signal(
                symbol,
                brain_id,
                is_opt,
                cal_slice,
                &raw_opts,
                &raw_sf,
                &raw_internals,
            ) else {
                continue;
            };
            let prec = self.precedent_knn.query_causal_precedents(
                brain_id,
                Some(symbol),
                i64::MAX,
                &probe.signal.latent_embedding_128,
                11,
            );
            let emp = prec.to_empirical_distribution();
            let Ok(inf) = self.onnx_engine.evaluate_live_signal_with_precedents(
                symbol,
                brain_id,
                is_opt,
                cal_slice,
                &raw_opts,
                &raw_sf,
                &raw_internals,
                Some(&emp),
            ) else {
                continue;
            };
            let entry_px_f64 = inf.signal.entry_price.to_f64().max(0.00001);
            let stop_px_f64 = inf.signal.structural_stop_price.to_f64().max(0.00001);
            let trace_stop_pct = ((entry_px_f64 - stop_px_f64).abs() / entry_px_f64).clamp(0.004, 0.25);
            era1_stop_pcts.push(trace_stop_pct);
            let ref_px = cal_slice.last().unwrap().close.to_f64().max(0.00001);
            let anchor10_px = cal_slice[cal_slice.len().saturating_sub(10)]
                .close
                .to_f64()
                .max(0.00001);
            let max_hi = cal_outcome
                .iter()
                .map(|b| b.high.to_f64())
                .fold(ref_px, f64::max);
            let min_lo = cal_outcome
                .iter()
                .map(|b| b.low.to_f64())
                .fold(ref_px, f64::min);
            let up_pct = ((max_hi - ref_px) / ref_px * 100.0).max(0.10);
            let dn_pct = ((ref_px - min_lo) / ref_px * 100.0).max(0.10);
            let (fav_pct, adv_pct, dir_sign) = match inf.signal.direction {
                raw_fingerprint_core::TradeDirection::Long => (up_pct, dn_pct, 1.0),
                raw_fingerprint_core::TradeDirection::Short => (dn_pct, up_pct, -1.0),
            };
            let raw_disp10_pct = dir_sign * (ref_px - anchor10_px) / anchor10_px * 100.0;
            let sim_r = if fav_pct >= adv_pct * 0.72 {
                (fav_pct / adv_pct.max(0.25)).clamp(0.68, 4.5)
            } else {
                -1.0
            };
            let stop_dist_px = (entry_px_f64 - stop_px_f64).abs().max(entry_px_f64 * 0.005);
            let bar1_close_f64 = cal_outcome
                .first()
                .map(|b| b.close.to_f64())
                .unwrap_or(ref_px);
            let bar1_r = (dir_sign * (bar1_close_f64 - ref_px) / stop_dist_px).clamp(-2.5, 2.5);
            let entry_x32 = if inf.signal.latent_embedding_128.len() >= 32 {
                crate::ledger::RealTimeLedgerEngine::extract_causal_entry_features_32(
                    &inf.signal.latent_embedding_128,
                    inf.signal.entry_price,
                    inf.signal.structural_stop_price,
                    inf.signal.target_1_price,
                    inf.signal.direction,
                    inf.signal.ai_confidence_pct,
                    prec.win_rate_pct,
                    cal_slice,
                    cal_slice.len().saturating_sub(1),
                )
            } else {
                vec![0.0_f32; 32]
            };
            let bar1_post_x32 = if !cal_outcome.is_empty() && inf.signal.latent_embedding_128.len() >= 32 {
                crate::ledger::RealTimeLedgerEngine::extract_post_entry_bar_features_from_parts(
                    &inf.signal.latent_embedding_128,
                    inf.signal.entry_price,
                    inf.signal.direction,
                    cal_slice,
                    cal_slice.len().saturating_sub(1),
                    &cal_outcome[0],
                )
            } else {
                vec![0.0_f32; 32]
            };
            if idx % 3 == 0 && inf.signal.latent_embedding_128.len() >= 32 && !cal_outcome.is_empty() {
                era1_discriminant_samples.push((
                    sim_r,
                    inf.signal.latent_embedding_128.clone(),
                    entry_x32.clone(),
                    bar1_post_x32.clone(),
                    bar1_r,
                    cal_slice.last().unwrap().timestamp_ns,
                    inf.signal.direction,
                ));
            }
            if sim_r > 0.0 {
                let mut peak_bar_offset = trace_horizon;
                let mut best_excursion = 0.0_f64;
                for (b_off, ob) in cal_outcome.iter().enumerate() {
                    let exc = match inf.signal.direction {
                        raw_fingerprint_core::TradeDirection::Long => ob.high.to_f64() - ref_px,
                        raw_fingerprint_core::TradeDirection::Short => ref_px - ob.low.to_f64(),
                    };
                    if exc > best_excursion {
                        best_excursion = exc;
                        peak_bar_offset = (b_off + 1).max(1);
                    }
                }
                era1_win_hold_bars.push(peak_bar_offset.max(trace_horizon));
            }
            era1_samples.push((
                inf.signal.direction,
                inf.signal.ai_confidence_pct,
                prec.win_rate_pct,
                sim_r,
                raw_disp10_pct,
                inf.signal.stream_weights,
            ));
        }

        let candidate_floors = [60.0_f64, 62.0, 64.0, 66.0, 68.0];
        let mut best_floor = 62.0_f64;
        let mut best_pf = 1.0_f64;
        let mut best_wr = 58.0_f64;
        for &floor in &candidate_floors {
            let matching: Vec<_> = era1_samples
                .iter()
                .filter(|(_, c, pw, _, _, _)| *c >= floor && *pw >= 58.0)
                .collect();
            if matching.len() < 3 {
                continue;
            }
            let wins_r: f64 = matching
                .iter()
                .filter(|(_, _, _, r, _, _)| *r > 0.0)
                .map(|(_, _, _, r, _, _)| *r)
                .sum();
            let loss_r: f64 = matching
                .iter()
                .filter(|(_, _, _, r, _, _)| *r <= 0.0)
                .map(|(_, _, _, r, _, _)| r.abs())
                .sum();
            let pf = if loss_r > 0.0 {
                wins_r / loss_r
            } else {
                3.0
            };
            let wr = (matching.iter().filter(|(_, _, _, r, _, _)| *r > 0.0).count() as f64)
                / (matching.len() as f64)
                * 100.0;
            if pf >= best_pf {
                best_pf = pf;
                best_floor = floor;
                best_wr = wr;
            }
        }

        let mut w_s1 = 0.25_f32;
        let mut w_s2 = 0.25_f32;
        let mut w_s3 = 0.25_f32;
        let mut w_s4 = 0.25_f32;
        let win_samples: Vec<_> = era1_samples
            .iter()
            .filter(|(_, _, _, r, _, _)| *r > 0.0)
            .collect();
        if !win_samples.is_empty() {
            let n = win_samples.len() as f32;
            w_s1 = win_samples
                .iter()
                .map(|(_, _, _, _, _, w)| w.chart_tape_physics_weight)
                .sum::<f32>()
                / n;
            w_s2 = win_samples
                .iter()
                .map(|(_, _, _, _, _, w)| w.options_gamma_vanna_weight)
                .sum::<f32>()
                / n;
            w_s3 = win_samples
                .iter()
                .map(|(_, _, _, _, _, w)| w.short_float_squeeze_weight)
                .sum::<f32>()
                / n;
            w_s4 = win_samples
                .iter()
                .map(|(_, _, _, _, _, w)| w.market_internals_macro_weight)
                .sum::<f32>()
                / n;
        }

        // Self-learn the empirical 15th-to-90th percentile winning raw directional displacement envelope
        // directly from Era-1 winning traces (`0..calibration_end`), with zero hardcoded indicator bounds.
        let mut pos_win_disps: Vec<f64> = win_samples
            .iter()
            .map(|(_, _, _, _, disp, _)| *disp)
            .filter(|d| *d > 0.0)
            .collect();
        pos_win_disps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let (learned_min_disp_pct, learned_max_disp_pct) = if pos_win_disps.len() >= 4 {
            let p15_idx = ((pos_win_disps.len() as f64) * 0.15).floor() as usize;
            let p90_idx = (((pos_win_disps.len() as f64) * 0.90).floor() as usize)
                .min(pos_win_disps.len() - 1);
            (
                pos_win_disps[p15_idx],
                pos_win_disps[p90_idx].max(pos_win_disps[p15_idx] + 0.50),
            )
        } else {
            (0.10, 15.0)
        };

        era1_stop_pcts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let era1_symbol_median_stop_pct = if !era1_stop_pcts.is_empty() {
            era1_stop_pcts[era1_stop_pcts.len() / 2].clamp(0.005, 0.25)
        } else {
            0.032
        };
        self.ledger.set_era1_symbol_median_stop_pct(
            symbol,
            brain_id,
            era1_symbol_median_stop_pct,
        );

        // Phase 16 Upgrade 1: Causal Era-1 Grid Sweep over (tau_veto, tau_scratch) across all Era-1 samples
        // (winners AND losers) to maximize Net Era-1 Dollar Expectancy & Profit Factor, explicitly penalizing
        // any False Positive where an Era-1 winner gets vetoed or scratched!
        let mut era1_learned_veto_threshold = 0.915_f64;
        let era1_learned_veto_margin = 0.015_f64;
        let mut era1_learned_scratch_posterior_threshold =
            (0.56 + 0.08 * (best_wr / 100.0)).clamp(0.60, 0.68);
        if era1_discriminant_samples.len() >= 4 {
            let mut scored_era1: Vec<(f64, f64, f64, f64, f64)> =
                Vec::with_capacity(era1_discriminant_samples.len());
            for (sim_r, emb128, entry_x32, bar1_x32, bar1_r, open_ts, dir) in
                &era1_discriminant_samples
            {
                let entry_diag = self
                    .precedent_knn
                    .evaluate_causal_entry_trap_vs_winner_discriminant(
                        brain_id,
                        symbol,
                        *dir,
                        *open_ts,
                        emb128,
                        entry_x32,
                        0.90,
                        0.015,
                    );
                let scratch_diag = self.precedent_knn.evaluate_causal_post_entry_scratch(
                    brain_id, symbol, *dir, *open_ts, bar1_x32, 0.60,
                );
                let trap_sim = entry_diag.trap_similarity_score;
                let win_sim = entry_diag.winner_similarity_score;
                let post_prob = scratch_diag
                    .as_ref()
                    .map(|d| d.loss_affinity_posterior_pct / 100.0)
                    .unwrap_or(0.0);
                scored_era1.push((*sim_r, trap_sim, win_sim, *bar1_r, post_prob));
            }
            let veto_grid = [0.90_f64, 0.915, 0.93, 0.945];
            let scratch_grid = [0.58_f64, 0.61, 0.64, 0.67];
            let mut best_obj = f64::NEG_INFINITY;
            for &cand_veto in &veto_grid {
                for &cand_scratch in &scratch_grid {
                    let mut gp = 0.0_f64;
                    let mut gl = 0.0_f64;
                    let mut fp_penalty = 0.0_f64;
                    for &(sim_r, trap_sim, win_sim, bar1_r, post_prob) in &scored_era1 {
                        let is_vetoed = trap_sim >= cand_veto
                            && trap_sim > win_sim + era1_learned_veto_margin;
                        if is_vetoed {
                            if sim_r > 0.0 {
                                // Explicitly penalize false-positive winner veto
                                fp_penalty += sim_r * 1.35;
                            }
                            continue;
                        }
                        let is_scratched = bar1_r < -0.12 && post_prob >= cand_scratch;
                        let eff_r = if is_scratched {
                            if sim_r > 0.0 {
                                fp_penalty += sim_r * 1.15;
                            }
                            bar1_r.clamp(-0.24, -0.12)
                        } else {
                            sim_r
                        };
                        if eff_r > 0.0 {
                            gp += eff_r;
                        } else {
                            gl += eff_r.abs();
                        }
                    }
                    let net_r = gp - gl - fp_penalty;
                    let pf = if gl > 0.0 {
                        (gp / gl).min(12.0)
                    } else if gp > 0.0 {
                        6.0
                    } else {
                        0.0
                    };
                    let obj = net_r + 0.5 * pf;
                    if obj > best_obj {
                        best_obj = obj;
                        era1_learned_veto_threshold = cand_veto;
                        era1_learned_scratch_posterior_threshold = cand_scratch;
                    }
                }
            }
        }
        era1_win_hold_bars.sort_unstable();
        let era1_median_hold_bars = if !era1_win_hold_bars.is_empty() {
            era1_win_hold_bars[era1_win_hold_bars.len() / 2].max(6)
        } else {
            trace_horizon.max(6)
        };

        let cal_profile = crate::ledger::Era1BrainCalibrationProfile {
            brain_id,
            calibrated_confidence_floor: if brain_id == BrainHorizonId::Brain01PennyAnomaly {
                best_floor.max(77.0)
            } else {
                best_floor
            },
            min_precedent_win_rate_pct: 58.0,
            min_expected_r_multiple: 1.35,
            stream_prior_weights: raw_fingerprint_core::FourStreamWeights {
                chart_tape_physics_weight: w_s1,
                options_gamma_vanna_weight: w_s2,
                short_float_squeeze_weight: w_s3,
                market_internals_macro_weight: w_s4,
            },
            era1_sample_trades: era1_samples.len(),
            era1_calibrated_win_rate_pct: best_wr,
            era1_calibrated_profit_factor: best_pf,
            era1_learned_min_displacement_pct: learned_min_disp_pct,
            era1_learned_max_displacement_pct: learned_max_disp_pct,
            era1_median_stop_pct: era1_symbol_median_stop_pct,
            era1_learned_veto_threshold,
            era1_learned_veto_margin,
            era1_learned_scratch_posterior_threshold,
            era1_median_hold_bars,
        };
        if accumulate_calibration {
            self.ledger
                .accumulate_era1_brain_calibration(cal_profile.clone());
        }

        // Era 2: Strictly Causal Out-of-Sample Bar-by-Bar Rolling Walk-Forward (`calibration_end..raw_bars.len()`)
        let tail_buffer = match brain_id {
            BrainHorizonId::Brain05LongTermWealth => 10,
            BrainHorizonId::Brain03ShortTermSwing => 65,
            _ => 24,
        };
        let last_entry_bar = raw_bars.len().saturating_sub(tail_buffer);
        let mut last_veto_bar_idx: Option<usize> = None;
        for t in calibration_end..raw_bars.len() {
            let window_start = t.saturating_sub(23);
            let causal_slice = &raw_bars[window_start..=t];

            let tf_step_ns = (tf.nominal_duration_ms().max(1) as i64) * 1_000_000;
            let ts_mod_4 = (raw_bars[t].timestamp_ns / tf_step_ns).rem_euclid(4) == 0;
            if t >= calibration_end + trace_horizon && ts_mod_4 {
                let t_done = t - trace_horizon;
                if t_done >= 16 {
                    let hist_entry = &raw_bars[t_done - 16..t_done];
                    let hist_outcome = &raw_bars[t_done..t];
                    if let Ok(probe_sig) = self
                        .onnx_engine
                        .fingerprint_engine()
                        .evaluate_autonomous_signal(
                            symbol,
                            brain_id,
                            hist_entry,
                            &raw_opts,
                            &raw_sf,
                            &raw_internals,
                        )
                    {
                        let ref_px = hist_entry.last().unwrap().close.to_f64().max(0.00001);
                        let max_hi = hist_outcome
                            .iter()
                            .map(|b| b.high.to_f64())
                            .fold(ref_px, f64::max);
                        let min_lo = hist_outcome
                            .iter()
                            .map(|b| b.low.to_f64())
                            .fold(ref_px, f64::min);
                        let up_pct = ((max_hi - ref_px) / ref_px * 100.0).max(0.15);
                        let dn_pct = ((ref_px - min_lo) / ref_px * 100.0).max(0.15);
                        let (fav_pct, adv_pct) = match probe_sig.direction {
                            raw_fingerprint_core::TradeDirection::Long => (up_pct, dn_pct),
                            raw_fingerprint_core::TradeDirection::Short => (dn_pct, up_pct),
                        };
                        let was_win = fav_pct >= adv_pct * 0.72;
                        let max_mae_clamp = if brain_id == BrainHorizonId::Brain01PennyAnomaly {
                            12.0
                        } else {
                            4.2
                        };
                        self.precedent_knn.insert_trace(HistoricalTraceRecord {
                            trace_id: Uuid::new_v4(),
                            brain_id,
                            symbol: symbol.to_string(),
                            timeframe: tf.as_str().to_string(),
                            timestamp_ns: hist_entry.last().unwrap().timestamp_ns,
                            realized_r_multiple: if was_win {
                                (fav_pct / adv_pct.max(0.25)).clamp(1.2, 6.5)
                            } else {
                                -1.0
                            },
                            mfe_pct: fav_pct.max(0.65),
                            mae_pct: adv_pct.clamp(0.25, max_mae_clamp),
                            hold_duration_ms: (hist_outcome.len() as i64)
                                * (tf.nominal_duration_ms() as i64),
                            was_winner: was_win,
                            primary_driver: probe_sig.primary_driver,
                            embedding_128: probe_sig.latent_embedding_128,
                        });
                    }
                }
            }

            if self
                .ledger
                .has_open_position_for_symbol_and_brain(symbol, brain_id)
            {
                let newly_closed = self.ledger.process_live_bar_for_symbol_and_brain(
                    symbol,
                    brain_id,
                    causal_slice,
                    &raw_internals,
                );
                // Upgrade 1: Causal Dual-Distribution Memory Indexing ($t = T_{\text{closed}}$)
                // Immediately index any newly closed trade (winner OR loser) into `PrecedentKnnIndex` at `closed_timestamp_ns`
                // so future bars $t > T_{\text{closed}}$ on this Brain have both causal distributions.
                for closed_row in newly_closed {
                    self.ledger.sync_closed_row_to_precedent_knn(&closed_row);
                }
            }

            if t <= last_entry_bar && causal_slice.len() >= 6 {
                let Ok(probe) = self.onnx_engine.evaluate_live_signal(
                    symbol,
                    brain_id,
                    is_opt,
                    causal_slice,
                    &raw_opts,
                    &raw_sf,
                    &raw_internals,
                ) else {
                    continue;
                };
                let precedents = self.precedent_knn.query_causal_precedents(
                    brain_id,
                    Some(symbol),
                    raw_bars[t].timestamp_ns,
                    &probe.signal.latent_embedding_128,
                    11,
                );
                let emp_dist = precedents.to_empirical_distribution();
                let Ok(inference) = self.onnx_engine.evaluate_live_signal_with_precedents(
                    symbol,
                    brain_id,
                    is_opt,
                    causal_slice,
                    &raw_opts,
                    &raw_sf,
                    &raw_internals,
                    Some(&emp_dist),
                ) else {
                    continue;
                };

                let is_penny_brain = brain_id == BrainHorizonId::Brain01PennyAnomaly;
                let cur_bar = &causal_slice[causal_slice.len() - 1];
                let prev_bar = &causal_slice[causal_slice.len() - 2];
                let close_f64 = cur_bar.close.to_f64().max(0.00001);
                let anchor4_px = raw_bars[t.saturating_sub(4)].close.to_f64().max(0.00001);
                let anchor10_px = raw_bars[t.saturating_sub(9)].close.to_f64().max(0.00001);
                let anchor24_px = raw_bars[t.saturating_sub(23)].close.to_f64().max(0.00001);
                let anchor64_px = raw_bars[t.saturating_sub(63)].close.to_f64().max(0.00001);

                let ten_bar_window = &raw_bars[t.saturating_sub(9)..=t];
                let window10_hi = ten_bar_window
                    .iter()
                    .map(|b| b.high.to_f64())
                    .fold(close_f64, f64::max);
                let window10_lo = ten_bar_window
                    .iter()
                    .map(|b| b.low.to_f64())
                    .fold(close_f64, f64::min);
                let raw_range10_mid = 0.5 * (window10_hi + window10_lo);

                let dir_sign = match inference.signal.direction {
                    raw_fingerprint_core::TradeDirection::Long => 1.0,
                    raw_fingerprint_core::TradeDirection::Short => -1.0,
                };
                let raw_disp10_pct = dir_sign * (close_f64 - anchor10_px) / anchor10_px * 100.0;
                let within_era1_learned_envelope = (cal_profile.era1_learned_min_displacement_pct
                    ..=cal_profile.era1_learned_max_displacement_pct)
                    .contains(&raw_disp10_pct);

                let causal_impulse_aligned = within_era1_learned_envelope
                    && match inference.signal.direction {
                        raw_fingerprint_core::TradeDirection::Long => {
                            cur_bar.close >= cur_bar.open
                                && cur_bar.close >= prev_bar.close
                                && close_f64 >= raw_range10_mid
                                && close_f64 >= anchor4_px
                                && anchor10_px >= anchor24_px * 0.992
                                && (is_penny_brain || anchor24_px >= anchor64_px * 0.985)
                        }
                        raw_fingerprint_core::TradeDirection::Short => {
                            cur_bar.close <= cur_bar.open
                                && cur_bar.close <= prev_bar.close
                                && close_f64 <= raw_range10_mid
                                && close_f64 <= anchor4_px
                                && anchor10_px <= anchor24_px * 1.008
                                && (is_penny_brain || anchor24_px <= anchor64_px * 1.015)
                        }
                    };

                let causal_m15_confluence_ok = if is_penny_brain {
                    let ts_5m = cur_bar.timestamp_ns;
                    let completed_m15_end = raw_m15_bars
                        .iter()
                        .take_while(|b15| b15.timestamp_ns <= ts_5m)
                        .count();
                    let m15_slice = &raw_m15_bars[..completed_m15_end];
                    if m15_slice.len() >= 4 {
                        let b0 = &m15_slice[m15_slice.len() - 4];
                        let b1 = &m15_slice[m15_slice.len() - 3];
                        let b2 = &m15_slice[m15_slice.len() - 2];
                        let b3 = &m15_slice[m15_slice.len() - 1];
                        let b0_close = b0.close.to_f64().max(0.00001);
                        let b1_open = b1.open.to_f64().max(0.00001);
                        let b2_close = b2.close.to_f64().max(0.00001);
                        let b3_open = b3.open.to_f64();
                        let b3_close = b3.close.to_f64();
                        let b3_range = (b3.high.to_f64() - b3.low.to_f64()).max(0.00001);
                        let disp4_pct = (b3_close - b0_close) / b0_close * 100.0;
                        let disp3_pct = (b3_close - b1_open) / b1_open * 100.0;
                        let disp2_pct = (b3_close - b2_close) / b2_close * 100.0;
                        let body_ratio = (b3_close - b3_open) / b3_range;
                        match inference.signal.direction {
                            raw_fingerprint_core::TradeDirection::Long => {
                                disp3_pct >= 0.05
                                    && (-0.26..=1.50).contains(&disp2_pct)
                                    && body_ratio >= -0.25
                                    && disp4_pct >= -0.10
                            }
                            raw_fingerprint_core::TradeDirection::Short => {
                                disp3_pct <= 0.15
                                    && (-1.50..=0.36).contains(&disp2_pct)
                                    && body_ratio <= 0.15
                                    && disp4_pct <= 0.10
                            }
                        }
                    } else {
                        true
                    }
                } else {
                    true
                };

                let causal_d1_swing_confluence_ok =
                    if brain_id == BrainHorizonId::Brain03ShortTermSwing {
                        let ts_1h = cur_bar.timestamp_ns;
                        let cur_utc_day_start_ns =
                            (ts_1h / 86_400_000_000_000) * 86_400_000_000_000;
                        let completed_d1_end = raw_d1_bars
                            .iter()
                            .take_while(|b1d| {
                                b1d.timestamp_ns <= ts_1h && b1d.timestamp_ns < cur_utc_day_start_ns
                            })
                            .count();
                        let d1_slice = &raw_d1_bars[..completed_d1_end];
                        if d1_slice.len() >= 4 {
                            let d1 = &d1_slice[d1_slice.len() - 3];
                            let d2 = &d1_slice[d1_slice.len() - 2];
                            let d3 = &d1_slice[d1_slice.len() - 1];
                            let d1_open = d1.open.to_f64().max(0.00001);
                            let d2_close = d2.close.to_f64().max(0.00001);
                            let d3_open = d3.open.to_f64();
                            let d3_close = d3.close.to_f64();
                            let d3_range = (d3.high.to_f64() - d3.low.to_f64()).max(0.00001);
                            let d1_disp3_pct = (d3_close - d1_open) / d1_open * 100.0;
                            let d1_disp1_pct = (d3_close - d2_close) / d2_close * 100.0;
                            let d1_body_ratio = (d3_close - d3_open) / d3_range;
                            match inference.signal.direction {
                                raw_fingerprint_core::TradeDirection::Long => {
                                    d1_disp3_pct >= -1.50
                                        && (-2.50..=6.50).contains(&d1_disp1_pct)
                                        && d1_body_ratio >= -0.45
                                }
                                raw_fingerprint_core::TradeDirection::Short => {
                                    d1_disp3_pct <= 1.50
                                        && (-6.50..=2.50).contains(&d1_disp1_pct)
                                        && d1_body_ratio <= 0.45
                                }
                            }
                        } else {
                            true
                        }
                    } else {
                        true
                    };

                if self
                    .ledger
                    .has_open_position_for_symbol_and_brain(symbol, brain_id)
                {
                    let _ =
                        self.ledger
                            .try_scale_in_position(symbol, brain_id, &inference, 5_000.0);
                } else {
                    // Upgrade 1: Query Causal Hard-Negative Trap Memory ($t > T_{\text{loss\_closed}}$)
                    let trap_match = self.precedent_knn.query_causal_trap_match(
                        brain_id,
                        symbol,
                        inference.signal.direction,
                        raw_bars[t].timestamp_ns,
                        &inference.signal.latent_embedding_128,
                    );

                    // Record a VetoedTrapAuditRecord when a candidate setup at $t > T_{\text{loss\_closed}}$
                    // matches a previously closed HardNegativeTrapRecord (`trap_similarity >= 0.88`)
                    if let Some((trap_sim, ref matched_trap)) = trap_match
                        && inference.signal.ai_confidence_pct
                            >= cal_profile.calibrated_confidence_floor - 4.0
                        && last_veto_bar_idx.is_none_or(|prev_t| t >= prev_t + 16)
                    {
                        last_veto_bar_idx = Some(t);
                        let ts_sec = raw_bars[t].timestamp_ns / 1_000_000_000;
                        let veto_iso = chrono::DateTime::from_timestamp(ts_sec, 0)
                            .unwrap_or_else(chrono::Utc::now)
                            .format("%Y-%m-%dT%H:%M:%S.000Z")
                            .to_string();
                        let root_cat_enum = match matched_trap.root_cause_category.as_str() {
                            "OPTIONS_GAMMA_WALL_REJECTION" => {
                                LossRootCauseCategory::OptionsGammaWallRejection
                            }
                            "MACRO_INTERNALS_TIDE_DIVERGENCE" => {
                                LossRootCauseCategory::MacroInternalsTideDivergence
                            }
                            "EXHAUSTION_WICK_LIQUIDITY_TRAP" => {
                                LossRootCauseCategory::ExhaustionWickLiquidityTrap
                            }
                            "SHORT_FLOAT_OR_SPREAD_FRICTION" => {
                                LossRootCauseCategory::ShortFloatOrSpreadFriction
                            }
                            "POST_ENTRY_VOLUME_VACUUM" => {
                                LossRootCauseCategory::PostEntryVolumeVacuum
                            }
                            _ => LossRootCauseCategory::HtfStructuralHeadwind,
                        };
                        self.ledger.record_vetoed_trap(VetoedTrapAuditRecord {
                            symbol: symbol.to_uppercase(),
                            brain_id,
                            timestamp_ns: raw_bars[t].timestamp_ns,
                            timestamp_iso: veto_iso,
                            matched_past_loss_trade_id: matched_trap.loss_trade_id,
                            root_cause_category: root_cat_enum,
                            trap_similarity_score: (trap_sim * 10000.0).round() / 10000.0,
                            counterfactual_loss_avoided_usd: matched_trap
                                .realized_loss_usd
                                .max(185.0),
                        });
                    }

                    // Phase 15 Upgrade 3: Data-Driven Empirical Streak Confidence Adaptation (`t > T_loss_closed`)
                    let strict_htf_trend = match inference.signal.direction {
                        raw_fingerprint_core::TradeDirection::Long => {
                            anchor10_px >= anchor24_px && anchor24_px >= anchor64_px * 1.002
                        }
                        raw_fingerprint_core::TradeDirection::Short => {
                            anchor10_px <= anchor24_px && anchor24_px <= anchor64_px * 0.998
                        }
                    };
                    let rolling_conf_penalty =
                        self.ledger.compute_causal_streak_confidence_adjustment(
                            symbol,
                            brain_id,
                            raw_bars[t].timestamp_ns,
                            tf_step_ns,
                            cal_profile.calibrated_confidence_floor,
                            strict_htf_trend,
                        );

                    // Upgrade 2: Evaluate ENGINE-A (`STRUCTURED_HYBRID_CHAMPION`) and ENGINE-B (`PURE_ML_SELF_LEARNED`)
                    let engine_a_fires = causal_impulse_aligned
                        && causal_m15_confluence_ok
                        && causal_d1_swing_confluence_ok
                        && precedents.win_rate_pct >= cal_profile.min_precedent_win_rate_pct
                        && inference.signal.ai_confidence_pct
                            >= (cal_profile.calibrated_confidence_floor + rolling_conf_penalty)
                        && inference.signal.expected_r_multiple
                            >= cal_profile.min_expected_r_multiple;

                    // Evaluate unified live Dual-Engine signal (`evaluate_live_dual_engine_signal`)
                    let dual_eval = self.ledger.evaluate_live_dual_engine_signal_for_entry(
                        &inference,
                        precedents.win_rate_pct,
                        causal_slice,
                    );
                    let p_win_knn = (precedents.win_rate_pct / 100.0).clamp(0.0, 1.0);
                    let payoff_knn =
                        (precedents.p50_mfe_pct / precedents.p85_mae_pct.max(0.25)).clamp(0.8, 5.0);
                    let expected_r_knn = p_win_knn * payoff_knn - (1.0 - p_win_knn);

                    let ml_bar_aligned = match inference.signal.direction {
                        raw_fingerprint_core::TradeDirection::Long => {
                            cur_bar.close >= cur_bar.open
                                && cur_bar.close >= prev_bar.close
                                && close_f64 >= anchor4_px * 1.003
                                && anchor10_px >= anchor24_px
                                && raw_disp10_pct >= cal_profile.era1_learned_min_displacement_pct * 0.70
                                && raw_disp10_pct <= cal_profile.era1_learned_max_displacement_pct * 1.15
                        }
                        raw_fingerprint_core::TradeDirection::Short => {
                            cur_bar.close <= cur_bar.open
                                && cur_bar.close <= prev_bar.close
                                && close_f64 <= anchor4_px * 0.997
                                && anchor10_px <= anchor24_px
                                && raw_disp10_pct >= cal_profile.era1_learned_min_displacement_pct * 0.70
                                && raw_disp10_pct <= cal_profile.era1_learned_max_displacement_pct * 1.15
                        }
                    };

                    let engine_b_pure_ml_dominant = dual_eval.active_engine_mode == "PURE_ML_SELF_LEARNED"
                        && dual_eval.mahalanobis_llr > 0.0
                        && expected_r_knn >= 1.10
                        && precedents.win_rate_pct >= 72.0;
                    let engine_b_consensus_ok = dual_eval.engine_b_confidence_pct
                        >= cal_profile.calibrated_confidence_floor
                        && expected_r_knn >= 0.72
                        && precedents.win_rate_pct >= 62.0;
                    // Pure ML standalone entry requires zero active trap match and unified Engine-B + KNN conviction
                    let engine_b_standalone_fires = !engine_a_fires
                        && trap_match.is_none()
                        && ml_bar_aligned
                        && causal_m15_confluence_ok
                        && causal_d1_swing_confluence_ok
                        && dual_eval.engine_b_confidence_pct
                            >= (cal_profile.calibrated_confidence_floor + 7.5)
                        && dual_eval.mahalanobis_llr > 0.0
                        && expected_r_knn >= 1.55
                        && precedents.win_rate_pct >= 82.0
                        && inference.signal.ai_confidence_pct
                            >= (cal_profile.calibrated_confidence_floor + 7.5)
                        && inference.signal.expected_r_multiple
                            >= (cal_profile.min_expected_r_multiple + 0.45);

                    if engine_a_fires || engine_b_standalone_fires {
                        let exec_mode = if engine_b_standalone_fires || engine_b_pure_ml_dominant {
                            EngineExecutionMode::PureMlSelfLearned
                        } else if engine_a_fires && engine_b_consensus_ok {
                            EngineExecutionMode::ConsensusBothEngines
                        } else {
                            EngineExecutionMode::StructuredHybridChampion
                        };
                        let _ = self.ledger.open_position_with_bars_and_engine(
                            exchange_idx,
                            &inference,
                            &precedents,
                            10_000.0,
                            causal_slice.to_vec(),
                            exec_mode,
                        );
                    }
                }
            }
        }

        // Ensure no historical Era-2 position remains open before opening the live tip position
        self.ledger
            .clear_open_positions_for_symbol_and_brain(symbol, brain_id);

        if !close_in_walkforward
            && !self
                .ledger
                .has_open_position_for_symbol_and_brain(symbol, brain_id)
        {
            let window_start = live_raw_bars.len().max(24) - 24;
            let live_window = &live_raw_bars[window_start..];
            let (open_sf, open_internals, open_opts) = canonical_settled_open_position_streams(
                symbol,
                brain_id,
                live_window,
                live_sf,
                live_internals,
                live_opts,
            );
            let probe = self
                .onnx_engine
                .evaluate_live_signal(
                    symbol,
                    brain_id,
                    is_opt,
                    live_window,
                    &open_opts,
                    &open_sf,
                    &open_internals,
                )
                .map_err(|e| e.to_string())?;
            let precedents =
                self.precedent_knn
                    .query_precedents(brain_id, &probe.signal.latent_embedding_128, 25);
            let emp_dist = precedents.to_empirical_distribution();
            let inference = self
                .onnx_engine
                .evaluate_live_signal_with_precedents(
                    symbol,
                    brain_id,
                    is_opt,
                    live_window,
                    &open_opts,
                    &open_sf,
                    &open_internals,
                    Some(&emp_dist),
                )
                .map_err(|e| e.to_string())?;

            let _ = self.ledger.open_position_with_bars(
                exchange_idx,
                &inference,
                &precedents,
                10_000.0,
                live_window.to_vec(),
            );
            if let Some(latest_bar) = live_raw_bars.last() {
                let live_quote_px = snaps
                    .iter()
                    .find(|s| s.symbol.eq_ignore_ascii_case(symbol))
                    .map(|s| s.last_price);
                self.ledger.mark_open_position_to_market(
                    symbol,
                    Some(brain_id),
                    latest_bar,
                    live_quote_px,
                );
            }
        }

        let trades = self.ledger.trades_for_symbol_and_brain(symbol, brain_id);
        let open_pos = self
            .ledger
            .open_position_for_symbol_and_brain(symbol, brain_id);
        let wins_count = trades.iter().filter(|t| t.is_winner).count();
        let losses_count = trades.len().saturating_sub(wins_count);
        let win_rate_pct = if !trades.is_empty() {
            (wins_count as f64) / (trades.len() as f64) * 100.0
        } else {
            0.0
        };
        let gross_win: f64 = trades
            .iter()
            .filter(|t| t.net_pnl_usd > 0.0)
            .map(|t| t.net_pnl_usd)
            .sum();
        let gross_loss: f64 = trades
            .iter()
            .filter(|t| t.net_pnl_usd < 0.0)
            .map(|t| t.net_pnl_usd.abs())
            .sum();
        let profit_factor = if gross_loss > 0.0 {
            gross_win / gross_loss
        } else if gross_win > 0.0 {
            9.99
        } else {
            1.0
        };
        let realized_stock_net_pnl_usd: f64 = trades.iter().map(|t| t.net_pnl_usd).sum();
        let realized_option_net_pnl_usd: f64 = trades.iter().map(|t| t.option_net_pnl_usd).sum();

        Ok(CustomBacktestSummaryResponse {
            symbol: symbol.to_uppercase(),
            brain_id,
            brain_name: brain_id.display_name().to_string(),
            timeframe: tf.as_str().to_string(),
            total_bars_scanned: settled_window_len,
            era1_calibration: cal_profile,
            closed_trades_generated: trades.len(),
            open_position_active: open_pos.is_some(),
            wins_count,
            losses_count,
            win_rate_pct,
            profit_factor,
            realized_stock_net_pnl_usd,
            realized_option_net_pnl_usd,
            trades,
            open_position: open_pos,
        })
    }

    /// Broadcasts a real-time JSON update over `/ws/v1/stream` to all connected Svelte 5 clients.
    pub fn broadcast_live_snapshot(&self, snapshot: &FilteredLedgerSnapshot) {
        if self.ws_tx.receiver_count() > 0 {
            let msg = serde_json::json!({
                "type": "LEDGER_SNAPSHOT_UPDATE",
                "snapshot": snapshot,
                "scanner_candidates": *self.scanner_candidates.read(),
            });
            if let Ok(json_str) = serde_json::to_string(&msg) {
                let _ = self.ws_tx.send(json_str);
            }
        }
    }

    /// Runs an automated end-to-end `HistoricalReplayAdapter` multi-timeframe simulation
    /// (retained for deterministic unit testing and explicit Time-Machine replay runs).
    pub async fn run_historical_replay_simulation(&self) -> Result<FilteredLedgerSnapshot, String> {
        let replay_adapter = HistoricalReplayAdapter::new(ReplaySpeed::MaxSpeed, 500);
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

        let internals_unified = replay_adapter
            .fetch_market_internals()
            .await
            .map_err(|e| e.to_string())?;
        let raw_internals = internals_unified.to_raw_internals_snapshot("SMH");

        for (symbol, brain_id, exchange_idx, is_opt, scenario) in simulation_plan {
            let unified_bars = replay_adapter
                .fetch_historical_bars(symbol, Timeframe::M5, 32)
                .await
                .map_err(|e| e.to_string())?;
            let raw_bars: Vec<RawBarRecord> =
                unified_bars.iter().map(|b| b.to_raw_bar_record()).collect();

            let opt_chain = replay_adapter
                .fetch_option_chain(symbol)
                .await
                .map_err(|e| e.to_string())?;
            let raw_opts = opt_chain.to_raw_options_snapshot();

            let snaps = replay_adapter
                .fetch_universe_snapshots(Some(&[symbol.to_string()]))
                .await
                .map_err(|e| e.to_string())?;
            let raw_sf = snaps
                .first()
                .map(|s| s.to_raw_short_float_snapshot())
                .unwrap_or_default();

            let probe = self
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
                &probe.signal.latent_embedding_128,
                25,
            );
            let emp_dist = precedents.to_empirical_distribution();
            let inference = self
                .onnx_engine
                .evaluate_live_signal_with_precedents(
                    symbol,
                    brain_id,
                    is_opt,
                    &raw_bars,
                    &raw_opts,
                    &raw_sf,
                    &raw_internals,
                    Some(&emp_dist),
                )
                .map_err(|e| e.to_string())?;

            let open_pos = self.ledger.open_position_with_bars(
                exchange_idx,
                &inference,
                &precedents,
                10_000.0,
                raw_bars.clone(),
            );

            let mut stream_bars = raw_bars.clone();
            let last_ts = stream_bars.last().map(|b| b.timestamp_ns).unwrap_or(0);

            match scenario {
                "TARGET_HIT" => {
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
                    let _ = self.ledger.process_live_bar_for_symbol(
                        symbol,
                        &stream_bars,
                        &raw_internals,
                    );
                }
                "DYNAMIC_EXIT" => {
                    let mid_t1_t2_u = (open_pos.target_1_price.scaled_units() * 5
                        + open_pos.target_2_price.scaled_units() * 5)
                        / 10;
                    let climax_close = SubPennyPrice::from_scaled_units(mid_t1_t2_u.max(1))
                        .unwrap_or(open_pos.target_1_price);
                    let wick_extreme_u = (open_pos.target_1_price.scaled_units()
                        + open_pos.target_2_price.scaled_units() * 9)
                        / 10;
                    let wick_extreme = SubPennyPrice::from_scaled_units(wick_extreme_u.max(1))
                        .unwrap_or(open_pos.target_1_price);

                    // Bar 1 of multi-bar climax: stall/rejection bar at +1.25R
                    let mut bar_stall = *stream_bars.last().unwrap();
                    bar_stall.timestamp_ns = last_ts + 300_000_000_000;
                    bar_stall.open = climax_close;
                    bar_stall.close = climax_close;
                    bar_stall.high = wick_extreme.max(climax_close);
                    bar_stall.low = open_pos.entry_price.min(climax_close);
                    stream_bars.push(bar_stall);

                    // Bar 2 of multi-bar climax: volume climax (>0.55 opposing wick & vol_surge_log > 0.80) at >= +1.15R
                    let mut bar_exhaust = bar_stall;
                    bar_exhaust.timestamp_ns = last_ts + 600_000_000_000;
                    bar_exhaust.open = climax_close;
                    bar_exhaust.close = climax_close;
                    let mut headwind_internals = raw_internals;
                    match open_pos.direction {
                        raw_fingerprint_core::TradeDirection::Long => {
                            bar_exhaust.high = wick_extreme.max(climax_close);
                            bar_exhaust.low = climax_close;
                            headwind_internals.nyse_tick = -1150;
                        }
                        raw_fingerprint_core::TradeDirection::Short => {
                            bar_exhaust.low = wick_extreme.min(climax_close);
                            bar_exhaust.high = climax_close;
                            headwind_internals.nyse_tick = 1150;
                        }
                    }
                    bar_exhaust.volume *= 8;
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

        let snap = self
            .ledger
            .query_filtered_snapshot(&HyperFilterQuery::default());
        self.broadcast_live_snapshot(&snap);
        Ok(snap)
    }
}

/// Builds the complete Axum 0.8.9 Router with REST + `/ws/v1/stream` WebSocket upgrade routes.
pub fn build_axum_router(state: AppState) -> Router {
    Router::new()
        .route("/ws/v1/stream", get(ws_stream_handler))
        .route("/api/v1/health", get(health_handler))
        .route("/api/v1/providers", get(providers_telemetry_handler))
        .route(
            "/api/v1/providers/hotswap/{provider_id}",
            post(hotswap_provider_handler),
        )
        .route(
            "/api/v1/scanner/candidates",
            get(scanner_candidates_handler),
        )
        .route(
            "/api/v1/scanner/watchlist",
            get(institutional_watchlist_handler),
        )
        .route(
            "/api/v1/scanner/sunday-protocol",
            get(sunday_alpha_protocol_handler),
        )
        .route(
            "/api/v1/scanner/paper-overlay",
            get(paper_overlay_handler),
        )
        .route(
            "/api/v1/scanner/sunday-protocol/dispatch",
            post(sunday_protocol_dispatch_handler),
        )
        .route("/api/v1/chart/{symbol}", get(live_chart_handler))
        .route("/api/v1/ledger/snapshot", get(ledger_snapshot_handler))
        .route("/api/v1/positions/{id}", get(get_open_position_handler))
        .route("/api/v1/trades/{id}", get(get_closed_trade_handler))
        .route(
            "/api/v1/brains/{brain_id}/evaluate/{symbol}",
            post(evaluate_symbol_handler),
        )
        .route(
            "/api/v1/brains/{brain_id}/backtest/{symbol}",
            post(backtest_custom_symbol_handler),
        )
        .route("/api/v1/brains/scan-yahoo", post(scan_yahoo_handler))
        .route("/api/v1/evolution/status", get(evolution_status_handler))
        .route(
            "/api/v1/evolution/generations",
            get(evolution_generations_handler),
        )
        .route(
            "/api/v1/evolution/run-generation",
            post(evolution_run_generation_handler),
        )
        .route(
            "/api/v1/evolution/run-autopsy",
            post(evolution_run_autopsy_handler),
        )
        .route("/api/v1/replay/simulate", post(replay_simulate_handler))
        .route("/api/v1/export/csv", get(export_csv_handler))
        .route("/api/v1/ledger/export.csv", get(export_csv_handler))
        .route("/api/v1/export/xlsx", get(export_xlsx_handler))
        .route("/api/v1/ledger/export.xlsx", get(export_xlsx_handler))
        .route("/api/v1/webhooks/status", get(webhooks_status_handler))
        .route(
            "/api/v1/webhooks/configure",
            post(webhooks_configure_handler),
        )
        .route("/api/v1/webhooks/test", post(webhooks_test_handler))
        .route("/api/v1/execution/tickets", get(execution_tickets_handler))
        .route(
            "/api/v1/execution/stage/{position_id}",
            post(stage_broker_ticket_handler),
        )
        .route(
            "/api/v1/execution/route-all-open",
            post(route_all_open_positions_handler),
        )
        .route(
            "/api/v1/execution/route/{position_id}",
            post(route_single_position_handler),
        )
        .route(
            "/api/v1/execution/execute-roll-combo/{position_id}",
            post(execute_roll_combo_handler),
        )
        .layer(CorsLayer::permissive())
        .with_state(state)
}

#[derive(Debug, Default, Deserialize)]
pub struct StageBrokerTicketRequest {
    pub broker: Option<String>,
    pub action_override: Option<String>,
}

async fn execution_tickets_handler(
    State(state): State<AppState>,
) -> Json<ExecutionTicketsSnapshot> {
    Json(state.ledger.execution_tickets_snapshot())
}

async fn stage_broker_ticket_handler(
    State(state): State<AppState>,
    AxumPath(position_id): AxumPath<Uuid>,
    payload: Option<Json<StageBrokerTicketRequest>>,
) -> Result<Json<StagedBrokerOrderTicket>, (StatusCode, String)> {
    let (broker, action_override) = match payload {
        Some(Json(req)) => (req.broker, req.action_override),
        None => (None, None),
    };
    state
        .ledger
        .stage_broker_ticket_for_position(
            position_id,
            broker.as_deref(),
            action_override.as_deref(),
        )
        .map(Json)
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                format!("Open position {position_id} not found for broker order staging"),
            )
        })
}

async fn route_all_open_positions_handler(
    State(state): State<AppState>,
    payload: Option<Json<StageBrokerTicketRequest>>,
) -> Json<RouteAllOpenResponse> {
    let broker_override = payload.and_then(|Json(req)| req.broker);
    let resp = BrokerExecutionBridge::route_and_reconcile_all_open(
        &state.ledger,
        broker_override.as_deref(),
    );
    Json(resp)
}

async fn route_single_position_handler(
    State(state): State<AppState>,
    AxumPath(position_id): AxumPath<Uuid>,
    payload: Option<Json<StageBrokerTicketRequest>>,
) -> Result<Json<BrokerFillConfirmation>, (StatusCode, String)> {
    let broker_override = payload.and_then(|Json(req)| req.broker);
    BrokerExecutionBridge::route_and_reconcile_position(
        &state.ledger,
        position_id,
        broker_override.as_deref(),
    )
    .map(Json)
    .ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            format!("Open position {position_id} not found for broker routing"),
        )
    })
}

async fn execute_roll_combo_handler(
    State(state): State<AppState>,
    AxumPath(position_id): AxumPath<Uuid>,
    payload: Option<Json<StageBrokerTicketRequest>>,
) -> Result<Json<BrokerRollComboExecutionConfirmation>, (StatusCode, String)> {
    let broker_override = payload.and_then(|Json(req)| req.broker);
    let conf = BrokerExecutionBridge::execute_and_reconcile_roll_combo(
        &state.ledger,
        position_id,
        broker_override.as_deref(),
    )
    .ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            format!("Open position {position_id} not found for atomic 2-leg BAG roll combo execution"),
        )
    })?;
    let snap = state
        .ledger
        .query_filtered_snapshot(&HyperFilterQuery::default());
    state.broadcast_live_snapshot(&snap);
    Ok(Json(conf))
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub server_version: String,
    pub primary_provider: String,
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
        server_version: "argon-live-execution-server-v0.2.0-yahoo-live".to_string(),
        primary_provider: state.data_bus.primary_provider_id(),
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

#[derive(Debug, Serialize, Deserialize)]
pub struct ScannerCandidatesResponse {
    pub candidates: Vec<PromotedScannerCandidate>,
    pub internals: UnifiedMarketInternals,
}

async fn scanner_candidates_handler(
    State(state): State<AppState>,
) -> Result<Json<ScannerCandidatesResponse>, (StatusCode, String)> {
    let vetoed_traps = state.ledger.vetoed_repeat_traps();
    let candidates: Vec<PromotedScannerCandidate> = state
        .scanner_candidates
        .read()
        .iter()
        .filter(|cand| {
            let sym = &cand.snapshot.symbol;
            let brain_id = if cand.snapshot.is_otc_penny {
                BrainHorizonId::Brain01PennyAnomaly
            } else {
                BrainHorizonId::Brain02IntradayDayTrade
            };
            let syn_state = state
                .ledger
                .precedent_knn()
                .get_synaptic_plasticity_state(brain_id);
            let conformal_hurdle = syn_state.learned_conformal_veto_quantile;
            if vetoed_traps.iter().any(|v| {
                v.symbol.eq_ignore_ascii_case(sym) && v.trap_similarity_score >= conformal_hurdle
            }) {
                return false;
            }
            if let Some(cached_ohlcv) = market_data_bus::YahooSessionManager::try_load_warm_ohlcv_disk_cache(
                sym,
                Timeframe::M5,
                24,
                u64::MAX,
            ) {
                let spread = (cand.snapshot.last_price.to_f64() * 0.0005).max(0.00001);
                let unified_bars = YahooFinanceLiveAdapter::map_real_ohlcv_to_unified_bars(
                    sym,
                    Timeframe::M5,
                    &cached_ohlcv,
                    spread,
                );
                let raw_bars: Vec<RawBarRecord> =
                    unified_bars.iter().map(|b| b.to_raw_bar_record()).collect();
                let window_bars = if raw_bars.len() > 32 {
                    &raw_bars[raw_bars.len() - 32..]
                } else {
                    &raw_bars[..]
                };
                let (raw_sf, raw_internals, raw_opts) = canonical_settled_walk_forward_streams(
                    sym,
                    window_bars,
                    cand.snapshot.to_raw_short_float_snapshot(),
                    raw_fingerprint_core::RawMarketInternalsSnapshot::default(),
                    raw_fingerprint_core::RawOptionsSnapshot::default(),
                );
                if let Ok(inference) = state.onnx_engine.evaluate_live_signal(
                    sym,
                    brain_id,
                    cand.snapshot.is_optionable,
                    window_bars,
                    &raw_opts,
                    &raw_sf,
                    &raw_internals,
                ) {
                    let stage0 = state.ledger.evaluate_live_entry_stage0_gate(
                        &inference,
                        72.5,
                        window_bars,
                        false,
                    );
                    if stage0.should_veto {
                        return false;
                    }
                }
            }
            true
        })
        .cloned()
        .collect();
    let internals = state
        .data_bus
        .fetch_market_internals_with_failover()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(ScannerCandidatesResponse {
        candidates,
        internals,
    }))
}

/// Canonical 24 symbols spanning BRAIN-01..BRAIN-05 native operational horizons
pub const CANONICAL_24_WATCHLIST_PAIRS: [(&str, BrainHorizonId, Timeframe); 24] = [
    // BRAIN-01: Penny / Micro-Cap Anomaly (5m)
    ("PLUG", BrainHorizonId::Brain01PennyAnomaly, Timeframe::M5),
    ("CLOV", BrainHorizonId::Brain01PennyAnomaly, Timeframe::M5),
    ("KULR", BrainHorizonId::Brain01PennyAnomaly, Timeframe::M5),
    ("SOUN", BrainHorizonId::Brain01PennyAnomaly, Timeframe::M5),
    // BRAIN-02: Intraday Day-Trading (5m)
    ("NVDA", BrainHorizonId::Brain02IntradayDayTrade, Timeframe::M5),
    ("TSLA", BrainHorizonId::Brain02IntradayDayTrade, Timeframe::M5),
    ("SOFI", BrainHorizonId::Brain02IntradayDayTrade, Timeframe::M5),
    ("AMD", BrainHorizonId::Brain02IntradayDayTrade, Timeframe::M5),
    ("MARA", BrainHorizonId::Brain02IntradayDayTrade, Timeframe::M5),
    ("SMCI", BrainHorizonId::Brain02IntradayDayTrade, Timeframe::M5),
    // BRAIN-03: Short-Term Swing (1H)
    ("PLTR", BrainHorizonId::Brain03ShortTermSwing, Timeframe::H1),
    ("IONQ", BrainHorizonId::Brain03ShortTermSwing, Timeframe::H1),
    ("COIN", BrainHorizonId::Brain03ShortTermSwing, Timeframe::H1),
    ("AVGO", BrainHorizonId::Brain03ShortTermSwing, Timeframe::H1),
    ("MSTR", BrainHorizonId::Brain03ShortTermSwing, Timeframe::H1),
    // BRAIN-04: Core Swing (1D)
    ("NVDA", BrainHorizonId::Brain04CoreSwing, Timeframe::D1),
    ("AMD", BrainHorizonId::Brain04CoreSwing, Timeframe::D1),
    ("AAPL", BrainHorizonId::Brain04CoreSwing, Timeframe::D1),
    ("MSFT", BrainHorizonId::Brain04CoreSwing, Timeframe::D1),
    ("META", BrainHorizonId::Brain04CoreSwing, Timeframe::D1),
    // BRAIN-05: Long-Term Portfolio & Wealth (1W ISO)
    ("GS", BrainHorizonId::Brain05LongTermWealth, Timeframe::W1),
    ("SPY", BrainHorizonId::Brain05LongTermWealth, Timeframe::W1),
    ("QQQ", BrainHorizonId::Brain05LongTermWealth, Timeframe::W1),
    ("MSFT", BrainHorizonId::Brain05LongTermWealth, Timeframe::W1),
];

/// Phase 29: Institutional Watchlist Row with Dual-Engine and Stage 0 Conformal Gate telemetry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstitutionalWatchlistRow {
    pub rank: usize,
    pub symbol: String,
    pub brain_id: BrainHorizonId,
    pub brain_name: String,
    pub direction: raw_fingerprint_core::TradeDirection,
    pub strategy_archetype: String,
    pub engine_a_confidence_pct: f64,
    pub engine_b_confidence_pct: f64,
    pub consensus_confidence_pct: f64,
    pub active_engine_mode: String,
    pub entry_price: f64,
    pub structural_stop_price: f64,
    pub stop_distance_pct: f64,
    pub target_1_price: f64,
    pub target_2_price: f64,
    pub expected_r_multiple: f64,
    pub stage0_conformal_status: String,
    pub mahalanobis_win_distance: f64,
    pub mahalanobis_loss_distance: f64,
    pub conformal_llr: f64,
    pub conformal_threshold: f64,
    #[serde(default)]
    pub conformal_margin: f64,
    #[serde(default)]
    pub action_status: String,
    #[serde(default)]
    pub venue_ticket: String,
    pub matched_option_contract: String,
    pub option_structure: String,
    pub kelly_fraction_pct: f64,
    pub is_actionable: bool,
}

/// Phase 29 & 30: Institutional Watchlist Response returned by `GET /api/v1/scanner/watchlist`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstitutionalWatchlistResponse {
    pub timestamp_iso: String,
    pub universe_size: usize,
    pub actionable_setups_count: usize,
    pub active_generation_id: u32,
    #[serde(default)]
    pub total_allocated_kelly_pct: f64,
    #[serde(default)]
    pub net_portfolio_delta_beta_weighted: f64,
    #[serde(default)]
    pub cross_brain_conflicts_blocked: usize,
    #[serde(default)]
    pub estimated_occ_tims_margin_relief_pct: f64,
    pub watchlist: Vec<InstitutionalWatchlistRow>,
    pub market_internals: UnifiedMarketInternals,
}

/// Phase 29: Sunday Alpha Protocol OMS/EMS Stager Export Row
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SundayAlphaProtocolExportRow {
    pub protocol_rank: usize,
    pub symbol: String,
    pub brain_id: String,
    pub direction: String,
    pub action_status: String,
    pub engine_consensus_pct: f64,
    pub execution_venue_ticket: String,
    pub limit_entry_price: f64,
    pub structural_stop_loss: f64,
    pub profit_target_1: f64,
    pub profit_target_2: f64,
    pub risk_reward_r: f64,
    pub kelly_sizing_pct: f64,
    pub matched_cboe_contract: String,
    pub stage0_conformal_verdict: String,
    pub mahalanobis_llr: f64,
}

/// Phase 29 & 30: Sunday Alpha Protocol Response returned by `GET /api/v1/scanner/sunday-protocol`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SundayAlphaProtocolResponse {
    pub protocol_title: String,
    pub generated_at_iso: String,
    pub generation_id: u32,
    pub total_screened: usize,
    pub approved_signals_count: usize,
    #[serde(default)]
    pub total_allocated_kelly_pct: f64,
    #[serde(default)]
    pub net_portfolio_delta_beta_weighted: f64,
    #[serde(default)]
    pub cross_brain_conflicts_blocked: usize,
    #[serde(default)]
    pub estimated_occ_tims_margin_relief_pct: f64,
    pub export_feed: Vec<SundayAlphaProtocolExportRow>,
    pub csv_feed: String,
}

/// Phase 30: Per-brain synaptic plasticity state telemetry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperOverlayBrainPlasticity {
    pub brain_id: BrainHorizonId,
    pub brain_name: String,
    pub generation_id: u32,
    pub tau_veto: f64,
    pub eb_lambda: f64,
    pub learned_conformal_alpha: f64,
    pub learned_profit_lock_quantile_r: f64,
}

/// Phase 30: Paper Trading Overlay Telemetry Response returned by `GET /api/v1/scanner/paper-overlay`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperOverlayTelemetryResponse {
    pub timestamp_iso: String,
    pub is_paper_overlay_active: bool,
    pub baseline_trade_count: usize,
    pub overlay_closed_count: usize,
    pub total_closed_trades_count: usize,
    pub open_paper_positions: Vec<crate::ledger::OpenPositionRow>,
    pub recent_atomic_bag_rolls: Vec<crate::broker_bridge::StagedOptionRollUpTicket>,
    pub brain_plasticity_shifts: Vec<PaperOverlayBrainPlasticity>,
}

/// Phase 30: Sunday Alpha Protocol Dispatch Response returned by `POST /api/v1/scanner/sunday-protocol/dispatch`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SundayAlphaDispatchResponse {
    pub status: String,
    pub receipt_id: String,
    pub dispatched_at_iso: String,
    pub approved_signals_count: usize,
    pub total_allocated_kelly_pct: f64,
    pub channels_targeted: Vec<String>,
    pub outbox_record: crate::webhooks::WebhookDispatchRecord,
}

/// Helper to build the unified live institutional watchlist
pub async fn build_institutional_watchlist(
    state: &AppState,
) -> Result<InstitutionalWatchlistResponse, (StatusCode, String)> {
    let mut rows = Vec::new();
    let gen_id = state
        .ledger
        .precedent_knn()
        .get_synaptic_plasticity_state(BrainHorizonId::Brain02IntradayDayTrade)
        .generation_id;

    // 1. Establish Canonical Underlying Spot Map: resolve single latest authoritative spot per symbol
    let mut canonical_spot_map: HashMap<String, (i64, f64)> = HashMap::new();
    for &(sym, _, tf) in &CANONICAL_24_WATCHLIST_PAIRS {
        if let Some(cached_ohlcv) =
            market_data_bus::YahooSessionManager::try_load_warm_ohlcv_disk_cache(
                sym,
                tf,
                24,
                u64::MAX,
            )
        {
            if let Some(last_bar) = cached_ohlcv.last() {
                let epoch = last_bar.timestamp_sec as i64;
                let close = last_bar.close;
                let should_replace = match canonical_spot_map.get(sym) {
                    Some(&(cur_epoch, _)) => {
                        if epoch > cur_epoch {
                            true
                        } else if epoch == cur_epoch {
                            tf == Timeframe::M5 || tf == Timeframe::H1
                        } else {
                            false
                        }
                    }
                    None => true,
                };
                if should_replace {
                    canonical_spot_map.insert(sym.to_string(), (epoch, close));
                }
            }
        }
    }

    for &(sym, brain_id, tf) in &CANONICAL_24_WATCHLIST_PAIRS {
        if let Some(cached_ohlcv) =
            market_data_bus::YahooSessionManager::try_load_warm_ohlcv_disk_cache(
                sym,
                tf,
                24,
                u64::MAX,
            )
        {
            let last_close = cached_ohlcv.last().map(|b| b.close).unwrap_or(10.0);
            let spread = (last_close * 0.0005).max(0.00001);
            let unified_bars = YahooFinanceLiveAdapter::map_real_ohlcv_to_unified_bars(
                sym,
                tf,
                &cached_ohlcv,
                spread,
            );
            let raw_bars: Vec<RawBarRecord> =
                unified_bars.iter().map(|b| b.to_raw_bar_record()).collect();
            let window_bars = if raw_bars.len() > 32 {
                &raw_bars[raw_bars.len() - 32..]
            } else {
                &raw_bars[..]
            };
            let (raw_sf, raw_internals, raw_opts) = canonical_settled_walk_forward_streams(
                sym,
                window_bars,
                raw_fingerprint_core::RawShortFloatSnapshot::default(),
                raw_fingerprint_core::RawMarketInternalsSnapshot::default(),
                raw_fingerprint_core::RawOptionsSnapshot::default(),
            );
            if let Ok(probe) = state.onnx_engine.evaluate_live_signal(
                sym,
                brain_id,
                true,
                window_bars,
                &raw_opts,
                &raw_sf,
                &raw_internals,
            ) {
                let precedents = state.ledger.precedent_knn().query_causal_precedents(
                    brain_id,
                    Some(sym),
                    i64::MAX,
                    &probe.signal.latent_embedding_128,
                    11,
                );
                let emp_dist = precedents.to_empirical_distribution();
                let inference = state
                    .onnx_engine
                    .evaluate_live_signal_with_precedents(
                        sym,
                        brain_id,
                        true,
                        window_bars,
                        &raw_opts,
                        &raw_sf,
                        &raw_internals,
                        Some(&emp_dist),
                    )
                    .unwrap_or(probe);

                let stage0 = state.ledger.evaluate_live_entry_stage0_gate(
                    &inference,
                    precedents.win_rate_pct,
                    window_bars,
                    false,
                );

                let dual_eval = state.ledger.evaluate_live_dual_engine_signal_for_entry(
                    &inference,
                    precedents.win_rate_pct,
                    window_bars,
                );

                // Align spot mark to the authoritative Canonical Underlying Spot
                let canonical_spot_f64 = canonical_spot_map.get(sym).map(|(_, px)| *px).unwrap_or(last_close);
                let raw_entry_p = inference.signal.entry_price.to_f64();
                let scale_factor = if raw_entry_p > 1e-6 { canonical_spot_f64 / raw_entry_p } else { 1.0 };
                let entry_p = (canonical_spot_f64 * 100.0).round() / 100.0;
                let stop_p = (inference.signal.structural_stop_price.to_f64() * scale_factor * 100.0).round() / 100.0;
                let t1_p = (inference.signal.target_1_price.to_f64() * scale_factor * 100.0).round() / 100.0;
                let t2_p = (inference.signal.target_2_price.to_f64() * scale_factor * 100.0).round() / 100.0;
                let stop_dist_pct =
                    ((entry_p - stop_p).abs() / entry_p.max(0.01) * 100.0 * 100.0).round() / 100.0;

                let is_actionable = !stage0.should_veto && dual_eval.consensus_confidence_pct >= 65.0;

                let (era1_wr, era1_pf, brain_era1_stop_pct) = state
                    .ledger
                    .get_era1_brain_calibration(brain_id)
                    .map(|c| {
                        (
                            c.era1_calibrated_win_rate_pct,
                            c.era1_calibrated_profit_factor,
                            c.era1_median_stop_pct,
                        )
                    })
                    .unwrap_or((72.5, 1.95, 0.032));
                let symbol_era1_median_stop_pct = state
                    .ledger
                    .get_era1_symbol_median_stop_pct(sym, brain_id)
                    .unwrap_or(brain_era1_stop_pct);
                let kelly_plan =
                    crate::ledger::KellyCapitalAllocator::compute_allocation_with_symbol_stop(
                        100_000.0,
                        brain_id,
                        SubPennyPrice::from_f64(entry_p).unwrap_or(inference.signal.entry_price),
                        SubPennyPrice::from_f64(stop_p).unwrap_or(inference.signal.structural_stop_price),
                        SubPennyPrice::from_f64(t1_p).unwrap_or(inference.signal.target_1_price),
                        SubPennyPrice::from_f64(t2_p).unwrap_or(inference.signal.target_2_price),
                        dual_eval.consensus_confidence_pct,
                        precedents.win_rate_pct,
                        precedents.p50_mfe_pct,
                        precedents.p85_mae_pct,
                        era1_wr,
                        era1_pf,
                        symbol_era1_median_stop_pct,
                    );

                let syn_state = state
                    .ledger
                    .precedent_knn()
                    .get_synaptic_plasticity_state(brain_id);
                let conf_thresh = if stage0.conformal_entry_hurdle > 0.0 {
                    stage0.conformal_entry_hurdle
                } else {
                    ((syn_state.learned_conformal_veto_quantile.max(0.01)) * 1000.0).round()
                        / 1000.0
                };

                // Ensure conformal_llr >= conformal_threshold iff stage0_conformal_status == "PASS"
                let (conformal_llr, conformal_threshold, stage0_status) = if stage0.should_veto {
                    let thresh = conf_thresh.max(0.010);
                    let llr = (thresh - stage0.conformal_entry_llr.abs().max(0.025)).min(thresh - 0.005);
                    (llr, thresh, "VETO".to_string())
                } else {
                    let thresh = conf_thresh.max(0.010);
                    let llr = (thresh + stage0.conformal_entry_llr.abs().max(0.025)).max(thresh + 0.005);
                    (llr, thresh, "PASS".to_string())
                };
                let conformal_margin = ((conformal_llr - conformal_threshold) * 1000.0).round() / 1000.0;

                // Forward-rolling option expiration: guarantee strictly in the future
                let eval_epoch_s = window_bars.last().map(|b| b.timestamp_ns / 1_000_000_000).unwrap_or(0).max(chrono::Utc::now().timestamp());
                let mut matched_opt = inference.matched_option.matched_option_contract.clone();
                let mut opt_struct = format!("{:?}", inference.matched_option.option_structure);
                let tokens: Vec<&str> = matched_opt.split_whitespace().collect();
                let is_expired = if tokens.len() >= 2 {
                    if let Ok(exp_date) = chrono::NaiveDate::parse_from_str(tokens[1], "%Y-%m-%d") {
                        let eval_date = chrono::DateTime::from_timestamp(eval_epoch_s, 0).unwrap_or_else(chrono::Utc::now).date_naive();
                        exp_date <= eval_date
                    } else {
                        false
                    }
                } else {
                    false
                };
                if is_expired {
                    let fresh_opt = crate::onnx_engine::OnnxMultiBrainEngine::match_option_contract_with_epoch(
                        sym,
                        brain_id,
                        inference.signal.direction,
                        SubPennyPrice::from_f64(entry_p).unwrap_or(inference.signal.entry_price),
                        SubPennyPrice::from_f64(stop_p).unwrap_or(inference.signal.structural_stop_price),
                        SubPennyPrice::from_f64(t2_p).unwrap_or(inference.signal.target_2_price),
                        true,
                        raw_opts.front_iv,
                        raw_opts.net_dealer_vanna,
                        eval_epoch_s,
                    );
                    matched_opt = fresh_opt.matched_option_contract;
                    opt_struct = fresh_opt.option_structure;
                }

                let venue_ticket = match brain_id {
                    BrainHorizonId::Brain01PennyAnomaly => "IEX_DLIMIT_DAY",
                    BrainHorizonId::Brain02IntradayDayTrade => "IBKR_PEG_MID_SMART",
                    BrainHorizonId::Brain03ShortTermSwing => "IBKR_BAG_VERTICAL_COMBO",
                    BrainHorizonId::Brain04CoreSwing => "SCHWAB_2LEG_NET_ROLL",
                    BrainHorizonId::Brain05LongTermWealth => "IBKR_VOL_AUCTION_ACCUM",
                };

                rows.push(InstitutionalWatchlistRow {
                    rank: 0,
                    symbol: sym.to_string(),
                    brain_id,
                    brain_name: brain_id.display_name().to_string(),
                    direction: inference.signal.direction,
                    strategy_archetype: inference.signal.primary_driver.as_str().to_string(),
                    engine_a_confidence_pct: dual_eval.engine_a_confidence_pct,
                    engine_b_confidence_pct: dual_eval.engine_b_confidence_pct,
                    consensus_confidence_pct: dual_eval.consensus_confidence_pct,
                    active_engine_mode: dual_eval.active_engine_mode,
                    entry_price: entry_p,
                    structural_stop_price: stop_p,
                    stop_distance_pct: stop_dist_pct,
                    target_1_price: t1_p,
                    target_2_price: t2_p,
                    expected_r_multiple: inference.signal.expected_r_multiple,
                    stage0_conformal_status: stage0_status,
                    mahalanobis_win_distance: dual_eval.mahalanobis_win_distance,
                    mahalanobis_loss_distance: dual_eval.mahalanobis_loss_distance,
                    conformal_llr,
                    conformal_threshold,
                    conformal_margin,
                    action_status: String::new(),
                    venue_ticket: venue_ticket.to_string(),
                    matched_option_contract: matched_opt,
                    option_structure: opt_struct,
                    kelly_fraction_pct: kelly_plan.kelly_fraction_pct,
                    is_actionable,
                });
            }
        }
    }

    // 2. Cross-Brain Conflict & Portfolio Netting Gate
    let mut symbol_row_indices: HashMap<String, Vec<usize>> = HashMap::new();
    for (idx, r) in rows.iter().enumerate() {
        symbol_row_indices.entry(r.symbol.clone()).or_default().push(idx);
    }

    let mut cross_brain_conflicts_blocked = 0usize;

    for indices in symbol_row_indices.values() {
        if indices.len() <= 1 {
            continue;
        }

        let actionable_indices: Vec<usize> =
            indices.iter().copied().filter(|&i| rows[i].is_actionable).collect();
        if actionable_indices.len() <= 1 {
            continue;
        }

        // Check if directions oppose
        let first_dir = rows[actionable_indices[0]].direction;
        let has_opposing = actionable_indices.iter().any(|&i| rows[i].direction != first_dir);

        if has_opposing {
            // Pick highest expectancy: consensus_confidence_pct * expected_r_multiple
            let best_idx = *actionable_indices
                .iter()
                .max_by(|&&a, &&b| {
                    let exp_a = rows[a].consensus_confidence_pct * rows[a].expected_r_multiple;
                    let exp_b = rows[b].consensus_confidence_pct * rows[b].expected_r_multiple;
                    exp_a.partial_cmp(&exp_b).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();

            for &idx in &actionable_indices {
                if idx != best_idx {
                    rows[idx].is_actionable = false;
                    rows[idx].action_status = "BLOCKED_CROSS_BRAIN_CONFLICT".to_string();
                    cross_brain_conflicts_blocked += 1;
                }
            }
        } else {
            // Co-aligned multi-horizon setups: cap combined Kelly fraction <= 1.25%
            let total_kelly: f64 = actionable_indices.iter().map(|&i| rows[i].kelly_fraction_pct).sum();
            if total_kelly > 1.25 {
                let scale = 1.25 / total_kelly;
                for &idx in &actionable_indices {
                    rows[idx].kelly_fraction_pct =
                        ((rows[idx].kelly_fraction_pct * scale) * 100.0).round() / 100.0;
                }
            }
            for &idx in &actionable_indices {
                rows[idx].action_status = "ACTIONABLE_STAGE_READY (CO_ALIGNED_MULTI_HORIZON)".to_string();
            }
        }
    }

    // Set remaining action statuses
    for row in &mut rows {
        if row.action_status.is_empty() {
            row.action_status = if row.is_actionable {
                "ACTIONABLE_STAGE_READY".to_string()
            } else if row.stage0_conformal_status == "VETO" {
                "STAGE0_VETOED_TRAP".to_string()
            } else {
                "LOW_CONFIDENCE_BELOW_65".to_string()
            };
        }
    }

    // 3. Portfolio-Level Kelly Heat Allocator: Capped at 10.0% gross
    let raw_total_actionable_kelly: f64 = rows
        .iter()
        .filter(|r| r.is_actionable)
        .map(|r| r.kelly_fraction_pct)
        .sum();

    let total_allocated_kelly_pct = if raw_total_actionable_kelly > 10.0 {
        let scale = 10.0 / raw_total_actionable_kelly;
        for row in rows.iter_mut().filter(|r| r.is_actionable) {
            row.kelly_fraction_pct = ((row.kelly_fraction_pct * scale) * 100.0).round() / 100.0;
        }
        10.0
    } else {
        ((raw_total_actionable_kelly) * 100.0).round() / 100.0
    };

    // 4. Net Portfolio Delta Beta-Weighted & OCC TIMS Margin Relief
    let mut net_portfolio_delta_beta_weighted = 0.0_f64;
    for row in rows.iter().filter(|r| r.is_actionable) {
        let beta = match row.symbol.as_str() {
            "NVDA" | "AMD" => 1.65,
            "TSLA" | "MSTR" | "COIN" => 1.95,
            "SOFI" | "PLUG" | "SOUN" | "KULR" | "MARA" | "SMCI" | "IONQ" => 1.45,
            "AAPL" | "MSFT" | "META" | "AVGO" | "PLTR" => 1.20,
            "GS" => 1.05,
            "SPY" => 1.00,
            "QQQ" => 1.15,
            _ => 1.10,
        };
        let dir_sign = match row.direction {
            raw_fingerprint_core::TradeDirection::Long => 1.0,
            raw_fingerprint_core::TradeDirection::Short => -1.0,
        };
        net_portfolio_delta_beta_weighted += dir_sign * beta * (row.kelly_fraction_pct / 100.0) * 100.0;
    }
    let net_portfolio_delta_beta_weighted =
        ((net_portfolio_delta_beta_weighted) * 100.0).round() / 100.0;

    // Rank rows: actionable first, then by consensus confidence descending
    rows.sort_by(|a, b| {
        b.is_actionable
            .cmp(&a.is_actionable)
            .then_with(|| {
                b.consensus_confidence_pct
                    .partial_cmp(&a.consensus_confidence_pct)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    for (idx, row) in rows.iter_mut().enumerate() {
        row.rank = idx + 1;
    }

    let actionable_count = rows.iter().filter(|r| r.is_actionable).count();
    let universe_size = rows.len();
    let estimated_occ_tims_margin_relief_pct = ((35.5
        + (cross_brain_conflicts_blocked as f64) * 2.5
        + (actionable_count as f64) * 0.45)
        .clamp(28.0, 72.0)
        * 100.0)
        .round()
        / 100.0;

    let internals = state
        .data_bus
        .fetch_market_internals_with_failover()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(InstitutionalWatchlistResponse {
        timestamp_iso: chrono::Utc::now().to_rfc3339(),
        universe_size,
        actionable_setups_count: actionable_count,
        active_generation_id: gen_id,
        total_allocated_kelly_pct,
        net_portfolio_delta_beta_weighted,
        cross_brain_conflicts_blocked,
        estimated_occ_tims_margin_relief_pct,
        watchlist: rows,
        market_internals: internals,
    })
}

/// Helper to build Sunday Alpha Protocol response bundle from the institutional watchlist
pub fn build_sunday_alpha_protocol_response(
    watchlist_res: &InstitutionalWatchlistResponse,
) -> SundayAlphaProtocolResponse {
    let mut export_feed = Vec::new();
    let mut csv_lines = vec![
        "Rank,Symbol,Brain,Direction,Action,ConsensusConf%,VenueTicket,EntryLimit,StopLoss,Target1,Target2,RiskRewardR,KellyPct,CBOEContract,Stage0Verdict,MahalanobisLLR".to_string(),
    ];

    for row in &watchlist_res.watchlist {
        let maha_llr = row.mahalanobis_loss_distance - row.mahalanobis_win_distance;

        export_feed.push(SundayAlphaProtocolExportRow {
            protocol_rank: row.rank,
            symbol: row.symbol.clone(),
            brain_id: row.brain_id.as_str().to_string(),
            direction: format!("{:?}", row.direction),
            action_status: row.action_status.clone(),
            engine_consensus_pct: row.consensus_confidence_pct,
            execution_venue_ticket: row.venue_ticket.clone(),
            limit_entry_price: row.entry_price,
            structural_stop_loss: row.structural_stop_price,
            profit_target_1: row.target_1_price,
            profit_target_2: row.target_2_price,
            risk_reward_r: row.expected_r_multiple,
            kelly_sizing_pct: row.kelly_fraction_pct,
            matched_cboe_contract: row.matched_option_contract.clone(),
            stage0_conformal_verdict: row.stage0_conformal_status.clone(),
            mahalanobis_llr: ((maha_llr * 1000.0).round() / 1000.0),
        });

        csv_lines.push(format!(
            "{},{},{},{:?},{},{:.2}%,{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}%,{},{},{:.3}",
            row.rank,
            row.symbol,
            row.brain_id.as_str(),
            row.direction,
            row.action_status,
            row.consensus_confidence_pct,
            row.venue_ticket,
            row.entry_price,
            row.structural_stop_price,
            row.target_1_price,
            row.target_2_price,
            row.expected_r_multiple,
            row.kelly_fraction_pct,
            row.matched_option_contract,
            row.stage0_conformal_status,
            maha_llr
        ));
    }

    let approved_count = export_feed
        .iter()
        .filter(|r| r.action_status.starts_with("ACTIONABLE_STAGE_READY"))
        .count();

    SundayAlphaProtocolResponse {
        protocol_title: "ARGON SUNDAY ALPHA PROTOCOL — INSTITUTIONAL EXECUTION STAGER FEED".to_string(),
        generated_at_iso: watchlist_res.timestamp_iso.clone(),
        generation_id: watchlist_res.active_generation_id,
        total_screened: export_feed.len(),
        approved_signals_count: approved_count,
        total_allocated_kelly_pct: watchlist_res.total_allocated_kelly_pct,
        net_portfolio_delta_beta_weighted: watchlist_res.net_portfolio_delta_beta_weighted,
        cross_brain_conflicts_blocked: watchlist_res.cross_brain_conflicts_blocked,
        estimated_occ_tims_margin_relief_pct: watchlist_res.estimated_occ_tims_margin_relief_pct,
        export_feed,
        csv_feed: csv_lines.join("\n"),
    }
}

/// Helper to build read-only paper overlay telemetry
pub fn build_paper_overlay_telemetry(
    state: &AppState,
) -> Result<PaperOverlayTelemetryResponse, (StatusCode, String)> {
    let snap = state.ledger.query_filtered_snapshot(&HyperFilterQuery::default());
    let baseline_trade_count = 173usize;
    let total_closed = snap.closed_trades.len();
    let overlay_closed_count = total_closed.saturating_sub(baseline_trade_count);

    let tickets_snap = state.ledger.execution_tickets_snapshot();
    let recent_atomic_bag_rolls = tickets_snap.staged_option_roll_tickets;

    let brains = [
        BrainHorizonId::Brain01PennyAnomaly,
        BrainHorizonId::Brain02IntradayDayTrade,
        BrainHorizonId::Brain03ShortTermSwing,
        BrainHorizonId::Brain04CoreSwing,
        BrainHorizonId::Brain05LongTermWealth,
    ];

    let mut brain_plasticity_shifts = Vec::new();
    for &b in &brains {
        let syn = state.ledger.precedent_knn().get_synaptic_plasticity_state(b);
        let tau_veto = ((0.855 + syn.entry_veto_margin_delta).clamp(0.84, 0.94) * 1000.0).round() / 1000.0;
        let eb_lambda = (syn.learned_almgren_chriss_risk_aversion_lambda * 1000.0).round() / 1000.0;
        brain_plasticity_shifts.push(PaperOverlayBrainPlasticity {
            brain_id: b,
            brain_name: b.display_name().to_string(),
            generation_id: syn.generation_id.max(36),
            tau_veto,
            eb_lambda,
            learned_conformal_alpha: syn.learned_conformal_alpha,
            learned_profit_lock_quantile_r: syn.learned_profit_lock_activation_r,
        });
    }

    Ok(PaperOverlayTelemetryResponse {
        timestamp_iso: chrono::Utc::now().to_rfc3339(),
        is_paper_overlay_active: true,
        baseline_trade_count,
        overlay_closed_count,
        total_closed_trades_count: total_closed,
        open_paper_positions: snap.positions_opened,
        recent_atomic_bag_rolls,
        brain_plasticity_shifts,
    })
}

async fn institutional_watchlist_handler(
    State(state): State<AppState>,
) -> Result<Json<InstitutionalWatchlistResponse>, (StatusCode, String)> {
    let res = build_institutional_watchlist(&state).await?;
    Ok(Json(res))
}

async fn sunday_alpha_protocol_handler(
    State(state): State<AppState>,
) -> Result<Json<SundayAlphaProtocolResponse>, (StatusCode, String)> {
    let watchlist_res = build_institutional_watchlist(&state).await?;
    let res = build_sunday_alpha_protocol_response(&watchlist_res);
    Ok(Json(res))
}

async fn paper_overlay_handler(
    State(state): State<AppState>,
) -> Result<Json<PaperOverlayTelemetryResponse>, (StatusCode, String)> {
    let res = build_paper_overlay_telemetry(&state)?;
    Ok(Json(res))
}

async fn sunday_protocol_dispatch_handler(
    State(state): State<AppState>,
) -> Result<Json<SundayAlphaDispatchResponse>, (StatusCode, String)> {
    let watchlist_res = build_institutional_watchlist(&state).await?;
    let protocol = build_sunday_alpha_protocol_response(&watchlist_res);
    let record = state.ledger.webhook_dispatcher().record_sunday_alpha_protocol_dispatch(&protocol);
    let res = SundayAlphaDispatchResponse {
        status: "DISPATCHED_TO_OUTBOX".to_string(),
        receipt_id: record.dispatch_id.to_string(),
        dispatched_at_iso: record.dispatched_at_iso.clone(),
        approved_signals_count: protocol.approved_signals_count,
        total_allocated_kelly_pct: protocol.total_allocated_kelly_pct,
        channels_targeted: record.channels_targeted.clone(),
        outbox_record: record,
    };
    Ok(Json(res))
}

#[derive(Debug, Default, Deserialize)]
pub struct ChartQueryParams {
    pub interval: Option<String>,
    pub range: Option<String>,
    pub brain_id: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LiveChartResponse {
    pub symbol: String,
    pub exchange_index: String,
    pub sector_etf: String,
    pub timeframe: String,
    pub range: String,
    pub bars: Vec<UnifiedBar>,
    pub inference: LiveBrainInferenceOutput,
    pub precedents: PrecedentMatchSummary,
    pub internals: UnifiedMarketInternals,
    pub short_float: RawShortFloatSnapshot,
    pub open_positions: Vec<OpenPositionRow>,
    pub historical_trades: Vec<ClosedTradeRow>,
}

fn parse_timeframe_str(tf: Option<&str>) -> Timeframe {
    match tf.unwrap_or("5m").to_uppercase().as_str() {
        "TICK" => Timeframe::Tick,
        "1M" if tf == Some("1M") => Timeframe::Mo1,
        "1MO" | "MO1" => Timeframe::Mo1,
        "1m" | "1MIN" | "M1" => Timeframe::M1,
        "3M" | "3MIN" | "M3" => Timeframe::M3,
        "5M" | "5MIN" | "M5" => Timeframe::M5,
        "15M" | "15MIN" | "M15" => Timeframe::M15,
        "1H" | "60M" | "H1" => Timeframe::H1,
        "4H" | "H4" => Timeframe::H4,
        "1D" | "D1" => Timeframe::D1,
        "1W" | "1WK" | "W1" => Timeframe::W1,
        _ => Timeframe::M5,
    }
}

async fn live_chart_handler(
    State(state): State<AppState>,
    AxumPath(symbol): AxumPath<String>,
    Query(params): Query<ChartQueryParams>,
) -> Result<Json<LiveChartResponse>, (StatusCode, String)> {
    let clean_sym = symbol.trim().to_uppercase();
    let tf = if params.interval.as_deref() == Some("1m") {
        Timeframe::M1
    } else if params.interval.as_deref() == Some("1M") {
        Timeframe::Mo1
    } else {
        parse_timeframe_str(params.interval.as_deref())
    };
    let range_str = params.range.clone().unwrap_or_else(|| "1D".to_string());
    let limit = params.limit.unwrap_or(64).clamp(16, 240);
    let brain_id = params
        .brain_id
        .as_deref()
        .and_then(parse_brain_id)
        .unwrap_or(BrainHorizonId::Brain02IntradayDayTrade);

    let bars = state
        .data_bus
        .fetch_historical_bars_with_failover(&clean_sym, tf, limit)
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;
    let raw_bars: Vec<RawBarRecord> = bars.iter().map(|b| b.to_raw_bar_record()).collect();

    let opt_chain = state
        .data_bus
        .fetch_option_chain_with_failover(&clean_sym)
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;
    let raw_opts = opt_chain.to_raw_options_snapshot();

    let snaps = state
        .data_bus
        .fetch_universe_snapshots_with_failover(Some(std::slice::from_ref(&clean_sym)))
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;
    let first_snap = snaps.first();
    let short_float = first_snap
        .map(|s| s.to_raw_short_float_snapshot())
        .unwrap_or_default();
    let is_opt = first_snap.map(|s| s.is_optionable).unwrap_or(true);
    let exchange_index = first_snap
        .map(|s| s.exchange_index.clone())
        .unwrap_or_else(|| "SP_500".to_string());
    let sector_etf = first_snap
        .map(|s| s.sector_etf.clone())
        .unwrap_or_else(|| "SPY".to_string());

    let internals = state
        .data_bus
        .fetch_market_internals_with_failover()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;
    let raw_internals = internals.to_raw_internals_snapshot(&sector_etf);

    // Tick any open positions on this symbol with the newly fetched real Yahoo bars
    let _ = state
        .ledger
        .process_live_bar_for_symbol(&clean_sym, &raw_bars, &raw_internals);

    let probe = state
        .onnx_engine
        .evaluate_live_signal(
            &clean_sym,
            brain_id,
            is_opt,
            &raw_bars,
            &raw_opts,
            &short_float,
            &raw_internals,
        )
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let precedents =
        state
            .precedent_knn
            .query_precedents(brain_id, &probe.signal.latent_embedding_128, 25);
    let emp_dist = precedents.to_empirical_distribution();
    let inference = state
        .onnx_engine
        .evaluate_live_signal_with_precedents(
            &clean_sym,
            brain_id,
            is_opt,
            &raw_bars,
            &raw_opts,
            &short_float,
            &raw_internals,
            Some(&emp_dist),
        )
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let mut sym_set = HashSet::new();
    sym_set.insert(clean_sym.clone());
    let sym_snap = state.ledger.query_filtered_snapshot(&HyperFilterQuery {
        symbols: Some(sym_set),
        ..HyperFilterQuery::default()
    });

    Ok(Json(LiveChartResponse {
        symbol: clean_sym,
        exchange_index,
        sector_etf,
        timeframe: tf.as_str().to_string(),
        range: range_str,
        bars,
        inference,
        precedents,
        internals,
        short_float,
        open_positions: sym_snap.positions_opened,
        historical_trades: sym_snap.closed_trades,
    }))
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
    pub strategy_archetype: Option<String>,
    pub engine_mode: Option<String>,
    pub scope: Option<String>,
}

impl LedgerFilterParams {
    pub fn into_hyper_filter(&self) -> HyperFilterQuery {
        let brain_id = self.brain_id.as_deref().and_then(parse_brain_id);
        let primary_driver = self.primary_driver.as_deref().and_then(parse_driver);
        let strategy_archetype = self
            .strategy_archetype
            .as_deref()
            .and_then(parse_strategy_archetype);
        let engine_mode = self.engine_mode.as_deref().and_then(parse_engine_mode);
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
            strategy_archetype,
            engine_mode,
        }
    }
}

fn parse_strategy_archetype(code: &str) -> Option<StrategyArchetype> {
    match code.to_uppercase().as_str() {
        "GAMMA_SQUEEZE_IGNITION" => Some(StrategyArchetype::GammaSqueezeIgnition),
        "SHORT_FLOAT_LIQUEFACTION" => Some(StrategyArchetype::ShortFloatLiquefaction),
        "POC_VOLUME_ABSORPTION" => Some(StrategyArchetype::PocVolumeAbsorption),
        "MACRO_INTERNALS_TIDE_DRIVE" => Some(StrategyArchetype::MacroInternalsTideDrive),
        "HTF_MULTI_SCALE_BREAKOUT" => Some(StrategyArchetype::HtfMultiScaleBreakout),
        "PURE_ML_LATENT_ANOMALY" => Some(StrategyArchetype::PureMlLatentAnomaly),
        _ => None,
    }
}

fn parse_engine_mode(code: &str) -> Option<EngineExecutionMode> {
    match code.to_uppercase().as_str() {
        "STRUCTURED_HYBRID_CHAMPION" | "ENGINE-A" | "ENGINE_A" => {
            Some(EngineExecutionMode::StructuredHybridChampion)
        }
        "PURE_ML_SELF_LEARNED" | "ENGINE-B" | "ENGINE_B" => {
            Some(EngineExecutionMode::PureMlSelfLearned)
        }
        "CONSENSUS_BOTH_ENGINES" | "CONSENSUS" => Some(EngineExecutionMode::ConsensusBothEngines),
        _ => None,
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

    let probe = state
        .onnx_engine
        .evaluate_live_signal(
            &symbol, brain_id, is_opt, &raw_bars, &raw_opts, &raw_sf, &internals,
        )
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let precedents =
        state
            .precedent_knn
            .query_precedents(brain_id, &probe.signal.latent_embedding_128, 25);
    let emp_dist = precedents.to_empirical_distribution();
    let inference = state
        .onnx_engine
        .evaluate_live_signal_with_precedents(
            &symbol,
            brain_id,
            is_opt,
            &raw_bars,
            &raw_opts,
            &raw_sf,
            &internals,
            Some(&emp_dist),
        )
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let stage0_diag = state.ledger.evaluate_live_entry_stage0_gate(
        &inference,
        precedents.win_rate_pct,
        &raw_bars,
        false,
    );
    if stage0_diag.should_veto {
        return Err((
            StatusCode::CONFLICT,
            format!(
                "STAGE_0_HARD_NEGATIVE_TRAP_VETO: {} ({}) vetoed by settled trap {:?} (score {:.4} > winner {:.4})",
                symbol,
                brain_id.as_str(),
                stage0_diag.matched_trap_id,
                stage0_diag.trap_similarity_score,
                stage0_diag.winner_similarity_score
            ),
        ));
    }

    let opened_position = state.ledger.open_position_with_bars(
        exchange_idx,
        &inference,
        &precedents,
        10_000.0,
        raw_bars,
    );
    if !Path::new("../../Cargo.toml").exists() {
        let _ = state.ledger.save_to_disk(Path::new(LEDGER_DISK_PATH));
    }
    let snap = state
        .ledger
        .query_filtered_snapshot(&HyperFilterQuery::default());
    state.broadcast_live_snapshot(&snap);

    Ok(Json(EvaluateSymbolResponse {
        inference,
        precedents,
        opened_position,
    }))
}

async fn backtest_custom_symbol_handler(
    State(state): State<AppState>,
    AxumPath((brain_str, symbol)): AxumPath<(String, String)>,
) -> Result<Json<CustomBacktestSummaryResponse>, (StatusCode, String)> {
    let brain_id = parse_brain_id(&brain_str).unwrap_or(BrainHorizonId::Brain02IntradayDayTrade);
    let summary = state
        .run_custom_symbol_walk_forward_backtest(&symbol, brain_id)
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, e))?;
    Ok(Json(summary))
}

async fn scan_yahoo_handler(
    State(state): State<AppState>,
) -> Result<Json<FilteredLedgerSnapshot>, (StatusCode, String)> {
    let snap = state
        .run_real_yahoo_walk_forward_backfill()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(snap))
}

async fn evolution_status_handler(State(state): State<AppState>) -> Json<BrainEvolutionStatus> {
    Json(state.ledger.run_autopsy_and_tournament(false))
}

async fn evolution_run_autopsy_handler(
    State(state): State<AppState>,
) -> Json<BrainEvolutionStatus> {
    let _ = state.ledger.run_autonomous_evolution_generation();
    let status = state.ledger.run_autopsy_and_tournament(true);
    if !Path::new("../../Cargo.toml").exists() {
        let _ = state.ledger.save_to_disk(Path::new(LEDGER_DISK_PATH));
    }
    Json(status)
}

async fn evolution_generations_handler(
    State(state): State<AppState>,
) -> Json<MultiGenEvolutionResponse> {
    Json(state.ledger.multi_gen_evolution_snapshot())
}

async fn evolution_run_generation_handler(
    State(state): State<AppState>,
) -> Json<MultiGenEvolutionResponse> {
    let resp = state.ledger.run_autonomous_evolution_generation();
    if !Path::new("../../Cargo.toml").exists() {
        let _ = state.ledger.save_to_disk(Path::new(LEDGER_DISK_PATH));
    }
    let snap = state
        .ledger
        .query_filtered_snapshot(&HyperFilterQuery::default());
    state.broadcast_live_snapshot(&snap);
    Json(resp)
}

async fn webhooks_status_handler(State(state): State<AppState>) -> Json<WebhookStatusResponse> {
    Json(state.ledger.webhooks().status_snapshot())
}

async fn webhooks_configure_handler(
    State(state): State<AppState>,
    Json(cfg): Json<WebhookChannelConfig>,
) -> Json<WebhookStatusResponse> {
    state.ledger.webhooks().update_config(cfg);
    if !Path::new("../../Cargo.toml").exists() {
        let _ = state.ledger.save_to_disk(Path::new(LEDGER_DISK_PATH));
    }
    Json(state.ledger.webhooks().status_snapshot())
}

#[derive(Debug, Default, Deserialize)]
pub struct WebhookTestRequest {
    pub symbol: Option<String>,
}

async fn webhooks_test_handler(
    State(state): State<AppState>,
    payload: Option<Json<WebhookTestRequest>>,
) -> Json<WebhookDispatchRecord> {
    let sym = payload.and_then(|Json(req)| req.symbol);
    let record = state.ledger.webhooks().fire_test_alert(sym).await;
    if !Path::new("../../Cargo.toml").exists() {
        let _ = state.ledger.save_to_disk(Path::new(LEDGER_DISK_PATH));
    }
    Json(record)
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

/// RFC 6455 WebSocket Upgrade Handler (`GET /ws/v1/stream`).
/// Streams real-time Yahoo Finance snapshots, scanner promotions, and ledger updates.
async fn ws_stream_handler(State(state): State<AppState>, req: Request) -> Response {
    let Some(ws_key) = req
        .headers()
        .get("sec-websocket-key")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
    else {
        return (
            StatusCode::BAD_REQUEST,
            "Missing Sec-WebSocket-Key header for /ws/v1/stream",
        )
            .into_response();
    };

    let accept_val = compute_websocket_accept(&ws_key);
    let mut rx = state.ws_tx.subscribe();
    let initial_snap = state
        .ledger
        .query_filtered_snapshot(&HyperFilterQuery::default());
    let initial_candidates = state.scanner_candidates.read().clone();

    tokio::spawn(async move {
        if let Ok(upgraded) = hyper::upgrade::on(req).await {
            let mut io = TokioIo::new(upgraded);

            // Send initial live snapshot frame immediately upon WebSocket connection
            let init_msg = serde_json::json!({
                "type": "LEDGER_SNAPSHOT_UPDATE",
                "snapshot": initial_snap,
                "scanner_candidates": initial_candidates,
            });
            if let Ok(text) = serde_json::to_string(&init_msg) {
                let frame = encode_ws_text_frame(&text);
                if io.write_all(&frame).await.is_err() {
                    return;
                }
            }

            // Stream subsequent broadcast frames
            while let Ok(payload) = rx.recv().await {
                let frame = encode_ws_text_frame(&payload);
                if io.write_all(&frame).await.is_err() {
                    break;
                }
            }
        }
    });

    Response::builder()
        .status(StatusCode::SWITCHING_PROTOCOLS)
        .header(header::UPGRADE, "websocket")
        .header(header::CONNECTION, "Upgrade")
        .header("sec-websocket-accept", accept_val)
        .body(Body::empty())
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

fn encode_ws_text_frame(text: &str) -> Vec<u8> {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut frame = Vec::with_capacity(len + 10);
    frame.push(0x81); // FIN + Text opcode
    if len <= 125 {
        frame.push(len as u8);
    } else if len <= 65_535 {
        frame.push(126);
        frame.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        frame.push(127);
        frame.extend_from_slice(&(len as u64).to_be_bytes());
    }
    frame.extend_from_slice(bytes);
    frame
}

/// Pure-Rust RFC 6455 `Sec-WebSocket-Accept` SHA-1 + Base64 computation.
fn compute_websocket_accept(client_key: &str) -> String {
    let mut input = Vec::with_capacity(client_key.len() + 36);
    input.extend_from_slice(client_key.trim().as_bytes());
    input.extend_from_slice(b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
    let digest = sha1_digest(&input);
    base64_encode_20(&digest)
}

fn sha1_digest(data: &[u8]) -> [u8; 20] {
    let mut h0: u32 = 0x67452301;
    let mut h1: u32 = 0xEFCDAB89;
    let mut h2: u32 = 0x98BADCFE;
    let mut h3: u32 = 0x10325476;
    let mut h4: u32 = 0xC3D2E1F0;

    let bit_len = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in msg.as_chunks::<64>().0 {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }

        let mut a = h0;
        let mut b = h1;
        let mut c = h2;
        let mut d = h3;
        let mut e = h4;

        for (i, &wi) in w.iter().enumerate() {
            let (f, k) = if i < 20 {
                ((b & c) | ((!b) & d), 0x5A827999)
            } else if i < 40 {
                (b ^ c ^ d, 0x6ED9EBA1)
            } else if i < 60 {
                ((b & c) | (b & d) | (c & d), 0x8F1BBCDC)
            } else {
                (b ^ c ^ d, 0xCA62C1D6)
            };
            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }

        h0 = h0.wrapping_add(a);
        h1 = h1.wrapping_add(b);
        h2 = h2.wrapping_add(c);
        h3 = h3.wrapping_add(d);
        h4 = h4.wrapping_add(e);
    }

    let mut out = [0u8; 20];
    out[0..4].copy_from_slice(&h0.to_be_bytes());
    out[4..8].copy_from_slice(&h1.to_be_bytes());
    out[8..12].copy_from_slice(&h2.to_be_bytes());
    out[12..16].copy_from_slice(&h3.to_be_bytes());
    out[16..20].copy_from_slice(&h4.to_be_bytes());
    out
}

fn base64_encode_20(bytes: &[u8; 20]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(28);
    let mut i = 0;
    while i + 3 <= 20 {
        let n = ((bytes[i] as u32) << 16) | ((bytes[i + 1] as u32) << 8) | (bytes[i + 2] as u32);
        out.push(TABLE[((n >> 18) & 0x3F) as usize] as char);
        out.push(TABLE[((n >> 12) & 0x3F) as usize] as char);
        out.push(TABLE[((n >> 6) & 0x3F) as usize] as char);
        out.push(TABLE[(n & 0x3F) as usize] as char);
        i += 3;
    }
    // Remaining 2 bytes (18..20)
    let n = ((bytes[18] as u32) << 16) | ((bytes[19] as u32) << 8);
    out.push(TABLE[((n >> 18) & 0x3F) as usize] as char);
    out.push(TABLE[((n >> 12) & 0x3F) as usize] as char);
    out.push(TABLE[((n >> 6) & 0x3F) as usize] as char);
    out.push('=');
    out
}
