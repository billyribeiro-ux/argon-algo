//! Binary entry point for `live-execution-server` (`http://0.0.0.0:8090`, configurable via `ARGON_PORT`).

use std::net::SocketAddr;
use std::path::Path;
use std::time::Duration;

use live_execution_server::{AppState, HyperFilterQuery, build_axum_router};
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

    // Load live runtime state from data/live_runtime_state.json first; fallback read-only to data/ledger_state.json
    let load_src = state.ledger.load_runtime_or_seed_benchmark(
        Path::new(live_execution_server::ledger::LIVE_RUNTIME_STATE_PATH),
        Path::new(live_execution_server::ledger::BENCHMARK_LEDGER_FIXTURE_PATH),
    );
    if load_src != "EMPTY_UNINITIALIZED" {
        let snap = state
            .ledger
            .query_filtered_snapshot(&HyperFilterQuery::default());
        info!(
            "Loaded {} state: {} open positions, {} closed trades, realized net P&L: ${:.2}, Kelly net P&L: ${:.2}",
            load_src,
            snap.positions_opened.len(),
            snap.closed_trades.len(),
            snap.scoreboard.realized_net_profit_usd,
            snap.scoreboard.kelly_realized_net_profit_usd
        );
    } else {
        match state.run_real_yahoo_walk_forward_backfill().await {
            Ok(snap) => {
                let _ = state.ledger.save_runtime_state_atomic(Path::new(
                    live_execution_server::ledger::LIVE_RUNTIME_STATE_PATH,
                ));
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
                let _ = state.ledger.save_runtime_state_atomic(Path::new(
                    live_execution_server::ledger::LIVE_RUNTIME_STATE_PATH,
                ));
            }
        }
    }

    let rl_brain_count = state.ledger.snapshot_pure_rl_brains().len();
    info!("PureAutonomousRlBrain ensembles active in live daemon: {rl_brain_count}");

    // Spawn unattended multi-timeframe candle-boundary scanner + atomic persistence + WS broadcast loop
    let bg_state = state.clone();
    tokio::spawn(async move {
        let respect_market_hours = std::env::var("ARGON_SCANNER_RESPECT_MARKET_HOURS")
            .ok()
            .is_none_or(|v| !matches!(v.trim(), "0" | "false" | "FALSE"));

        let now0 = chrono::Utc::now().timestamp();
        let mut last_m1_bucket = now0.div_euclid(60);
        let mut last_m5_bucket = now0.div_euclid(300);
        let mut last_h1_bucket = now0.div_euclid(3_600);
        let mut last_d1_bucket = (now0 - 5 * 3_600).div_euclid(86_400);

        let mut interval = tokio::time::interval(Duration::from_secs(10));
        interval.tick().await; // consume immediate 0ms tick so boot snapshot is untouched
        loop {
            interval.tick().await;
            let now_sec = chrono::Utc::now().timestamp();
            let b_m1 = now_sec.div_euclid(60);
            let b_m5 = now_sec.div_euclid(300);
            let b_h1 = now_sec.div_euclid(3_600);
            let b_d1 = (now_sec - 5 * 3_600).div_euclid(86_400);

            let scan_m1 = b_m1 > last_m1_bucket;
            let scan_m5 = b_m5 > last_m5_bucket;
            let scan_h1 = b_h1 > last_h1_bucket;
            let scan_d1 = b_d1 > last_d1_bucket;
            last_m1_bucket = b_m1;
            last_m5_bucket = b_m5;
            last_h1_bucket = b_h1;
            last_d1_bucket = b_d1;

            let digest_before = bg_state.ledger.runtime_mutation_digest();
            let _ = bg_state
                .run_unattended_scanner_cycle(
                    now_sec,
                    scan_m1,
                    scan_m5,
                    scan_h1,
                    scan_d1,
                    respect_market_hours,
                )
                .await;
            let digest_after = bg_state.ledger.runtime_mutation_digest();
            if digest_after != digest_before {
                let _ = bg_state.ledger.save_runtime_state_atomic(Path::new(
                    live_execution_server::ledger::LIVE_RUNTIME_STATE_PATH,
                ));
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
