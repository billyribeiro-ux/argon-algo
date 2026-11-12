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
    FinraShortVolumeRecord, MACRO_BREADTH_ETFS, REAL_YAHOO_CORE_UNIVERSE, RealExchangeOhlcvBar,
    RealUniverseTickerMeta, RealYahooSymbolData, VIX_TERM_SYMBOLS, YahooFinanceLiveAdapter,
    YahooSessionManager, build_causal_universe_snapshots_at_timestamp,
    compute_causal_internals_at_timestamp, lookup_bar_le_timestamp,
};
