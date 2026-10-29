//! Binary entry point for `live-execution-server` (`http://0.0.0.0:8080`).

use std::net::SocketAddr;
use std::path::Path;

use live_execution_server::{AppState, build_axum_router};
use tokio::net::TcpListener;
use tracing::info;

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

    // Seed initial historical replay simulation so the terminal connects to live & closed rows immediately
    let _ = state.run_historical_replay_simulation().await;

    let app = build_axum_router(state);
    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    info!("Argon Quant Live Execution Server listening on http://{addr}");

    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

