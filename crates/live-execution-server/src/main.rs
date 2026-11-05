//! Binary entry point for `live-execution-server` (`http://0.0.0.0:8090`, configurable via `ARGON_PORT`).

use std::net::SocketAddr;
use std::path::Path;
use std::time::Duration;

use live_execution_server::{AppState, HyperFilterQuery, build_axum_router};
use market_data_bus::{MarketDataAdapter, Timeframe};
use raw_fingerprint_core::RawBarRecord;
use tokio::net::TcpListener;
use tracing::{info, warn};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "live_execution_server=info,tower_http=info".into()),
        )
        .init();

    let models_dir = Path::new("models");
    let state = AppState::new(if models_dir.exists() {
        Some(models_dir)
    } else {
        None
    });

    // Load canonical persisted ledger state from data/ledger_state.json first; fallback to Yahoo walk-forward backfill if missing
    if state.ledger.load_from_disk(Path::new("data/ledger_state.json")) {
        let snap = state
            .ledger
            .query_filtered_snapshot(&HyperFilterQuery::default());
        info!(
            "Loaded canonical ledger state from data/ledger_state.json: {} open positions, {} closed trades, realized net P&L: ${:.2}, Kelly net P&L: ${:.2}",
            snap.positions_opened.len(),
            snap.closed_trades.len(),
            snap.scoreboard.realized_net_profit_usd,
            snap.scoreboard.kelly_realized_net_profit_usd
        );
    } else {
        match state.run_real_yahoo_walk_forward_backfill().await {
            Ok(snap) => {
                info!(
                    "Real Yahoo Finance walk-forward backfill complete: {} open positions, {} closed trades, realized net P&L: ${:.2}",
                    snap.positions_opened.len(),
                    snap.closed_trades.len(),
                    snap.scoreboard.realized_net_profit_usd
                );
            }
            Err(e) => {
                warn!("Yahoo live backfill returned warning ({e}); running historical replay fallback");
                let _ = state.run_historical_replay_simulation().await;
            }
        }
    }

    // Spawn background Tokio live Yahoo Finance polling & WebSocket broadcast loop
    let bg_state = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(25));
        interval.tick().await; // consume immediate 0ms tick so boot snapshot is untouched
        loop {
            interval.tick().await;
            let snap = bg_state
                .ledger
                .query_filtered_snapshot(&HyperFilterQuery::default());
            let open_targets: Vec<(String, raw_fingerprint_core::BrainHorizonId, i64)> = snap
                .positions_opened
                .iter()
                .map(|p| (p.symbol.clone(), p.brain_id, p.opened_timestamp_ns))
                .collect();

            if !open_targets.is_empty()
                && let Ok(internals_u) = bg_state.yahoo_adapter.fetch_market_internals().await
            {
                let raw_internals = internals_u.to_raw_internals_snapshot("SPY");
                for (sym, brain_id, opened_ts_ns) in &open_targets {
                    let native_tf = match brain_id {
                        raw_fingerprint_core::BrainHorizonId::Brain01PennyAnomaly
                        | raw_fingerprint_core::BrainHorizonId::Brain02IntradayDayTrade => {
                            Timeframe::M5
                        }
                        raw_fingerprint_core::BrainHorizonId::Brain03ShortTermSwing => {
                            Timeframe::H1
                        }
                        raw_fingerprint_core::BrainHorizonId::Brain04CoreSwing => Timeframe::D1,
                        raw_fingerprint_core::BrainHorizonId::Brain05LongTermWealth => {
                            Timeframe::W1
                        }
                    };
                    if let Ok(bars) = bg_state
                        .yahoo_adapter
                        .fetch_historical_bars(sym, native_tf, 24)
                        .await
                    {
                        let raw_bars: Vec<RawBarRecord> =
                            bars.iter().map(|b| b.to_raw_bar_record()).collect();
                        if raw_bars
                            .last()
                            .is_some_and(|b| b.timestamp_ns > *opened_ts_ns)
                        {
                            let settled = bg_state.ledger.process_live_bar_for_symbol_and_brain(
                                sym,
                                *brain_id,
                                &raw_bars,
                                &raw_internals,
                            );
                            for closed_row in &settled {
                                let _ = bg_state
                                    .ledger
                                    .on_live_trade_settled_continual_plasticity(closed_row);
                            }
                        } else if let Some(latest_bar) = raw_bars.last() {
                            bg_state.ledger.mark_open_position_to_market(
                                sym,
                                Some(*brain_id),
                                latest_bar,
                                None,
                            );
                        }
                    }
                }
            }

            let updated_snap = bg_state
                .ledger
                .query_filtered_snapshot(&HyperFilterQuery::default());
            bg_state.broadcast_live_snapshot(&updated_snap);
        }
    });

    let app = build_axum_router(state);
    let port: u16 = std::env::var("ARGON_PORT")
        .ok()
        .and_then(|v| v.trim().parse::<u16>().ok())
        .unwrap_or(8090);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!(
        "Argon Quant Live Execution Server (Yahoo Finance Live Primary) listening on http://{addr}"
    );

    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
