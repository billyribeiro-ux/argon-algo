# 03 — Agnostic Plug-and-Play Market Data Bus Specification
**Version:** 1.0 (Living Blueprint)  
**Core Mandate:** Zero Vendor Lock-In. Build on Free Feeds First, Hot-Swap or Multi-Stream Any Provider Later.

---

## 1. The Provider Adapter Architecture
The AI Brains, Database, and Command Terminal **never talk directly to an external API**. All external feeds connect through a standardized `MarketDataAdapter` interface that normalizes incoming data into four strict internal Rust/Python structs:
* `UnifiedBar` (`symbol`, `timestamp_ns`, `timeframe`, `open`, `high`, `low`, `close`, `volume`, `vwap`, `trade_count`)
* `UnifiedTick` (`symbol`, `timestamp_ns`, `price`, `size`, `exchange`, `conditions`, `bid`, `ask`)
* `UnifiedOptionChain` (`underlying`, `timestamp_ns`, `expiration`, `strike`, `right`, `bid`, `ask`, `last`, `volume`, `open_interest`, `iv`, `delta`, `gamma`, `theta`, `vega`)
* `UnifiedMarketInternals` (`timestamp_ns`, `nyse_tick`, `nasdaq_tick`, `add`, `vold`, `vix_spot`, `vix_front_month`)

---

## 2. Multi-Feed Routing & Hot-Swap Slots
Users can enable a single provider or run **Multiple Providers Simultaneously** (e.g., Provider A for deep historical backfills + Provider B for live options chains + Provider C for OTC penny data) with automatic millisecond failover:

### Phase 1 Starter Slots (100% Free to Build & Test Right Now)
1. **`HistoricalReplayAdapter` (Built-In Time Machine):**
   - Streams saved historical market sessions at `1x`, `5x`, `10x`, or `Max` speed as if the market were open live. Essential for testing live signals, UI updates, and P&L tracking at night and on weekends.
2. **`AlpacaFreeAdapter` (Zero-Cost Starter):**
   - Provides 7+ years of historical multi-timeframe bars (`1m` to `1D`), real-time WebSocket streaming, bulk market snapshots, and free options chain snapshots.
3. **`SchwabDirectAdapter` (Free with Existing Charles Schwab Account):**
   - Provides institutional real-time Level 1 / Level 2 WebSockets, full live Options Chains with Greeks, and real-time Market Internals (`$TICK`, `$ADD`, `$VOLD`).
4. **`FreeBulkBackfillAdapter` (`Tiingo` / `Finnhub` / `yfinance`):**
   - Zero-cost deep daily/weekly/intraday historical backfill for training the Long-Term and Swing Brains across 10+ years of data.

### Phase 2+ Expansion Slots (Drop-In Ready Whenever Needed)
5. **`InteractiveBrokersAdapter` (IBKR TWS / Client Portal API):** Full global equities, options, OTC pennies, and sub-second execution routing.
6. **`PolygonIoAdapter` / `DatabentoAdapter`:** Full OPRA options tick firehose and full SIP/OTC tick-by-tick historical archives.

---

## 3. Future Additions / Brainstorm Notes
* [ ] *Add future broker or data feed adapters here...*
