//! `ingest_30yr.rs` — CLI Binary for Phase 1 30+ Year Market Data Ingestion & Verification.
//!
//! Usage:
//!   cargo run --release -p market-data-bus --bin ingest_30yr -- --ingest [--pilot | --full]
//!   cargo run --release -p market-data-bus --bin ingest_30yr -- --verify
//!   cargo run --release -p market-data-bus --bin ingest_30yr -- --ingest --verify

use market_data_bus::pipeline::IngestionPipeline;
use std::path::PathBuf;
use tracing::{error, info};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();

    let args: Vec<String> = std::env::args().collect();

    let mut do_ingest = false;
    let mut do_verify = false;
    let mut is_pilot = true;

    for arg in &args[1..] {
        match arg.as_str() {
            "--ingest" => do_ingest = true,
            "--verify" => do_verify = true,
            "--pilot" => is_pilot = true,
            "--full" => is_pilot = false,
            "--help" | "-h" => {
                println!("Usage: ingest_30yr [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --ingest     Run data ingestion (Step A: Corp Actions, Step B: Macro/VIX, Step C: Equities & Rollups)");
                println!("  --verify     Run 7-point verification audit and output markdown report");
                println!("  --pilot      Ingest core pilot curriculum symbols (SPY, QQQ, IWM, NVDA, AAPL, etc.)");
                println!("  --full       Ingest full ~250-symbol multi-regime curriculum universe");
                println!("  -h, --help   Print help");
                return Ok(());
            }
            unknown => {
                eprintln!("Unknown argument: {unknown}");
                std::process::exit(1);
            }
        }
    }

    if !do_ingest && !do_verify {
        do_ingest = true;
        do_verify = true;
    }

    let raw_dir = PathBuf::from("data/raw_downloads");
    let shards_dir = PathBuf::from("data/shards");
    let ohlcv_cache_dir = PathBuf::from("data/ohlcv_cache");

    let mut pipeline = IngestionPipeline::new(&raw_dir, &shards_dir, &ohlcv_cache_dir)?;

    if do_ingest {
        info!("============================================================");
        info!("Starting Phase 1 30-Year Data Ingestion Pipeline");
        info!("Pilot Mode: {}", is_pilot);
        info!("============================================================");

        pipeline.ingest_corporate_actions().await?;
        pipeline.ingest_macro_rates().await?;
        pipeline.ingest_universe_bars(is_pilot).await?;
        info!("Ingestion completed successfully.");
    }

    if do_verify {
        info!("============================================================");
        info!("Starting Phase 1 Verification & Invariant Audit");
        info!("============================================================");

        match pipeline.verify_shards() {
            Ok(report) => {
                let md = report.format_markdown();
                println!("\n{md}\n");
                info!("Verification Audit PASSED with 0 errors. Exiting 0.");
                std::process::exit(0);
            }
            Err(e) => {
                error!("Verification Audit FAILED: {e}");
                std::process::exit(1);
            }
        }
    }

    Ok(())
}
