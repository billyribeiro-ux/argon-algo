//! Binary entry point for `live-execution-server` (`http://0.0.0.0:8080`).

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

    // Seed initial real Yahoo Finance walk-forward ledger & active open positions on boot
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
            warn!("Yahoo live backfill returned warning ({e}); checking persisted disk state");
            if !state
                .ledger
                .load_from_disk(Path::new("data/ledger_state.json"))
            {
                let _ = state.run_historical_replay_simulation().await;
            }
        }
    }

    // Spawn background Tokio live Yahoo Finance polling & WebSocket broadcast loop
    let bg_state = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(25));
        loop {
            interval.tick().await;
            let snap = bg_state
                .ledger
                .query_filtered_snapshot(&HyperFilterQuery::default());
            let open_targets: Vec<(String, raw_fingerprint_core::BrainHorizonId)> = snap
                .positions_opened
                .iter()
                .map(|p| (p.symbol.clone(), p.brain_id))
                .collect();

            if !open_targets.is_empty()
                && let Ok(internals_u) = bg_state.yahoo_adapter.fetch_market_internals().await
            {
                let raw_internals = internals_u.to_raw_internals_snapshot("SPY");
                for (sym, brain_id) in &open_targets {
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
                        let _ = bg_state.ledger.process_live_bar_for_symbol_and_brain(
                            sym,
                            *brain_id,
                            &raw_bars,
                            &raw_internals,
                        );
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
    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    info!(
        "Argon Quant Live Execution Server (Yahoo Finance Live Primary) listening on http://{addr}"
    );

    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
