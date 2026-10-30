//! Agnostic `MarketDataAdapter` Trait & Multi-Feed Hot-Swap / Failover Router.

use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::broadcast;

use crate::types::{
    MarketStreamEvent, TickerSnapshot, Timeframe, UnifiedBar, UnifiedMarketInternals,
    UnifiedOptionChain,
};

#[derive(Debug, Error)]
pub enum DataBusError {
    #[error("Provider '{0}' is not registered in the Market Data Bus")]
    ProviderNotFound(String),
    #[error("All registered market data providers failed; last error: {0}")]
    AllProvidersFailed(String),
    #[error("Adapter connection error: {0}")]
    ConnectionError(String),
    #[error("Invalid request parameter: {0}")]
    InvalidRequest(String),
}

/// Provider health and latency telemetry reported to the Command Terminal HUD.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderTelemetry {
    pub provider_id: String,
    pub display_name: String,
    pub connected: bool,
    pub is_primary: bool,
    pub last_latency_micros: u64,
    pub messages_processed: u64,
    pub failover_count: u32,
}

/// Standardized interface implemented by all free and institutional market data feeds.
#[allow(clippy::double_must_use)] #[async_trait]
pub trait MarketDataAdapter: Send + Sync {
    /// Unique slot identifier (e.g., `"historical_replay"`, `"alpaca_free"`, `"schwab_direct"`).
    fn provider_id(&self) -> &'static str;

    /// Human-readable adapter name.
    fn display_name(&self) -> &'static str;

    /// Establishes WebSocket / REST session.
    async fn connect(&self) -> Result<(), DataBusError>;

    /// Gracefully disconnects active streams.
    async fn disconnect(&self) -> Result<(), DataBusError>;

    /// Returns `true` if the adapter is healthy and ready to serve requests.
    fn is_connected(&self) -> bool;

    /// Fetches historical multi-timeframe bars for a symbol.
    async fn fetch_historical_bars(
        &self,
        symbol: &str,
        timeframe: Timeframe,
        limit: usize,
    ) -> Result<Vec<UnifiedBar>, DataBusError>;

    /// Fetches wide-net market snapshots across the requested universe or full market.
    async fn fetch_universe_snapshots(
        &self,
        symbols: Option<&[String]>,
    ) -> Result<Vec<TickerSnapshot>, DataBusError>;

    /// Fetches the live options chain with Greeks for an underlying symbol (`Stream 2`).
    async fn fetch_option_chain(
        &self,
        underlying: &str,
    ) -> Result<UnifiedOptionChain, DataBusError>;

    /// Fetches the current synchronized market internals heartbeat (`Stream 4`).
    async fn fetch_market_internals(&self) -> Result<UnifiedMarketInternals, DataBusError>;
}

/// Multi-Feed Router & Hot-Swap Manager.
///
/// Runs multiple `MarketDataAdapter` slots concurrently, supports zero-downtime hot-swapping,
/// and automatically fails over in milliseconds if the primary provider errors or disconnects.
pub struct MultiFeedRouter {
    adapters: DashMap<String, Arc<dyn MarketDataAdapter>>,
    priority_order: RwLock<Vec<String>>,
    telemetry: DashMap<String, ProviderTelemetry>,
    event_tx: broadcast::Sender<MarketStreamEvent>,
}

impl MultiFeedRouter {
    pub fn new(channel_capacity: usize) -> Self {
        let (event_tx, _) = broadcast::channel(channel_capacity.max(1024));
        Self {
            adapters: DashMap::new(),
            priority_order: RwLock::new(Vec::new()),
            telemetry: DashMap::new(),
            event_tx,
        }
    }

    /// Registers a plug-and-play `MarketDataAdapter` slot.
    pub fn register_adapter(&self, adapter: Arc<dyn MarketDataAdapter>, make_primary: bool) {
        let id = adapter.provider_id().to_string();
        let name = adapter.display_name().to_string();
        let connected = adapter.is_connected();

        self.adapters.insert(id.clone(), adapter);

        let mut order = self.priority_order.write();
        order.retain(|existing| existing != &id);
        if make_primary {
            order.insert(0, id.clone());
        } else {
            order.push(id.clone());
        }
        let primary_id = order.first().cloned().unwrap_or_default();
        drop(order);

        self.telemetry.insert(
            id.clone(),
            ProviderTelemetry {
                provider_id: id,
                display_name: name,
                connected,
                is_primary: make_primary,
                last_latency_micros: 0,
                messages_processed: 0,
                failover_count: 0,
            },
        );
        self.refresh_primary_flags(&primary_id);
    }

    /// Hot-swaps the active primary provider at runtime in `<1ms` without dropping open streams.
    pub fn hot_swap_primary(&self, provider_id: &str) -> Result<(), DataBusError> {
        if !self.adapters.contains_key(provider_id) {
            return Err(DataBusError::ProviderNotFound(provider_id.to_string()));
        }
        {
            let mut order = self.priority_order.write();
            order.retain(|id| id != provider_id);
            order.insert(0, provider_id.to_string());
        }
        self.refresh_primary_flags(provider_id);
        Ok(())
    }

    fn refresh_primary_flags(&self, primary_id: &str) {
        for mut entry in self.telemetry.iter_mut() {
            entry.is_primary = entry.provider_id == primary_id;
        }
    }

    /// Subscribes to the unified real-time broadcast bus (`Bar`, `Tick`, `OptionChain`, `Internals`).
    #[inline]
    pub fn subscribe_stream(&self) -> broadcast::Receiver<MarketStreamEvent> {
        self.event_tx.subscribe()
    }

    /// Publishes a live event onto the unified broadcast bus.
    #[inline]
    pub fn publish_event(&self, event: MarketStreamEvent) {
        let _ = self.event_tx.send(event);
    }

    /// Returns the provider ID of the current primary adapter slot.
    pub fn primary_provider_id(&self) -> String {
        self.priority_order
            .read()
            .first()
            .cloned()
            .unwrap_or_else(|| "yahoo_finance_live".to_string())
    }

    /// Returns live status and latency telemetry for all registered adapter slots.
    pub fn telemetry_snapshot(&self) -> Vec<ProviderTelemetry> {
        let order = self.priority_order.read().clone();
        let mut out = Vec::with_capacity(order.len());
        for id in order {
            if let Some(t) = self.telemetry.get(&id) {
                out.push(t.clone());
            }
        }
        out
    }

    /// Executes `fetch_historical_bars` with automatic millisecond failover across registered adapters.
    pub async fn fetch_historical_bars_with_failover(
        &self,
        symbol: &str,
        timeframe: Timeframe,
        limit: usize,
    ) -> Result<Vec<UnifiedBar>, DataBusError> {
        let order = self.priority_order.read().clone();
        let mut last_err = String::from("no providers registered");

        for (idx, provider_id) in order.iter().enumerate() {
            if let Some(adapter) = self.adapters.get(provider_id) {
                let start = Instant::now();
                match adapter
                    .fetch_historical_bars(symbol, timeframe, limit)
                    .await
                {
                    Ok(bars) => {
                        let elapsed_us = start.elapsed().as_micros() as u64;
                        if let Some(mut t) = self.telemetry.get_mut(provider_id) {
                            t.connected = true;
                            t.last_latency_micros = elapsed_us;
                            t.messages_processed += bars.len() as u64;
                            if idx > 0 {
                                t.failover_count += 1;
                            }
                        }
                        return Ok(bars);
                    }
                    Err(e) => {
                        last_err = e.to_string();
                        if let Some(mut t) = self.telemetry.get_mut(provider_id) {
                            t.connected = false;
                        }
                    }
                }
            }
        }
        Err(DataBusError::AllProvidersFailed(last_err))
    }

    /// Executes `fetch_universe_snapshots` with automatic failover.
    pub async fn fetch_universe_snapshots_with_failover(
        &self,
        symbols: Option<&[String]>,
    ) -> Result<Vec<TickerSnapshot>, DataBusError> {
        let order = self.priority_order.read().clone();
        let mut last_err = String::from("no providers registered");

        for provider_id in &order {
            if let Some(adapter) = self.adapters.get(provider_id) {
                let start = Instant::now();
                match adapter.fetch_universe_snapshots(symbols).await {
                    Ok(snaps) => {
                        let elapsed_us = start.elapsed().as_micros() as u64;
                        if let Some(mut t) = self.telemetry.get_mut(provider_id) {
                            t.connected = true;
                            t.last_latency_micros = elapsed_us;
                            t.messages_processed += snaps.len() as u64;
                        }
                        return Ok(snaps);
                    }
                    Err(e) => {
                        last_err = e.to_string();
                    }
                }
            }
        }
        Err(DataBusError::AllProvidersFailed(last_err))
    }

    /// Executes `fetch_option_chain` with automatic failover.
    pub async fn fetch_option_chain_with_failover(
        &self,
        underlying: &str,
    ) -> Result<UnifiedOptionChain, DataBusError> {
        let order = self.priority_order.read().clone();
        let mut last_err = String::from("no providers registered");

        for provider_id in &order {
            if let Some(adapter) = self.adapters.get(provider_id) {
                let start = Instant::now();
                match adapter.fetch_option_chain(underlying).await {
                    Ok(chain) => {
                        let elapsed_us = start.elapsed().as_micros() as u64;
                        if let Some(mut t) = self.telemetry.get_mut(provider_id) {
                            t.connected = true;
                            t.last_latency_micros = elapsed_us;
                            t.messages_processed += chain.contracts.len() as u64;
                        }
                        return Ok(chain);
                    }
                    Err(e) => {
                        last_err = e.to_string();
                    }
                }
            }
        }
        Err(DataBusError::AllProvidersFailed(last_err))
    }

    /// Executes `fetch_market_internals` with automatic failover.
    pub async fn fetch_market_internals_with_failover(
        &self,
    ) -> Result<UnifiedMarketInternals, DataBusError> {
        let order = self.priority_order.read().clone();
        let mut last_err = String::from("no providers registered");

        for provider_id in &order {
            if let Some(adapter) = self.adapters.get(provider_id) {
                match adapter.fetch_market_internals().await {
                    Ok(internals) => return Ok(internals),
                    Err(e) => {
                        last_err = e.to_string();
                    }
                }
            }
        }
        Err(DataBusError::AllProvidersFailed(last_err))
    }
}
