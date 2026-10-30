//! `market-data-bus` — Agnostic Plug-and-Play Market Data Bus, Multi-Feed Hot-Swap Router,
//! and Two-Stage 11,000+ Ticker Full-Market Scanner Funnel.

pub mod adapter;
pub mod adapters;
pub mod scanner;
pub mod types;
pub mod yahoo;

pub use adapter::{DataBusError, MarketDataAdapter, MultiFeedRouter, ProviderTelemetry};
pub use adapters::{
    AlpacaFreeAdapter, FreeBulkBackfillAdapter, HistoricalReplayAdapter,
    InstitutionalExpansionAdapter, ReplaySpeed, SchwabDirectAdapter,
};
pub use scanner::{
    AssetCapabilityToggle, IndexBasketFilter, PromotedScannerCandidate, TwoStageMarketFunnel,
    UniverseScannerConfig,
};
pub use types::{
    MarketStreamEvent, OptionRight, TickerSnapshot, Timeframe, UnifiedBar, UnifiedMarketInternals,
    UnifiedOptionChain, UnifiedOptionContract, UnifiedTick,
};
pub use yahoo::{
    REAL_YAHOO_CORE_UNIVERSE, RealUniverseTickerMeta, RealYahooSymbolData, YahooFinanceLiveAdapter,
    YahooSessionManager,
};
