# AUTONOMOUS MARKET INTELLIGENCE & MULTI-BRAIN TERMINAL
## Master Enterprise Architecture, Level 8++ Vanilla CSS System & Gemini 4 Argon Blueprint
**Document Version:** 2.0 (Unified Master Specification)  
**Target Principal AI Architect:** Gemini 4 Argon (`thinking_level: high`, 1M Token Context Window)  
**Styling Standard:** Google Principal Engineer (L8++) Pure Modern Vanilla CSS (Zero Tailwind / Zero CSS-in-JS Runtime)

---

## SECTION 1: CORE PHILOSOPHY & RAW FINGERPRINT ENGINE

### 1.1 The "Open-Minded" Law (Zero Human Indicators)
1. **Deterministic Liquidity Traces:** Nothing in the stock market happens randomly. Every price inflection, rally, or collapse is driven by physical liquidity mechanics (institutional accumulation/distribution, forced options dealer gamma hedging, trapped short covering, and macro basket flows). Every action leaves an indelible mathematical fingerprint on the chart and tape.
2. **STRICT BAN on Pre-Built Retail Indicators:**
   - Do **NOT** hardcode lagging human indicators (RSI, MACD, Stochastic, standard SMA/EMA crossovers, or arbitrary fixed ATR rules) as scanner rules.
   - Pre-built strategies close the algorithm's mind and blind it to multi-dimensional cause-and-effect.
3. **Self-Supervised Financial World Model (Pure Raw Evidence):**
   - The system ingests years of raw historical market numbers and works backward from undeniable historical expansions and collapses.
   - Using raw Candlestick & Flow Tokenization paired with continuous state-space (`Mamba-SSM`) and sequence (`FlexAttention` / `LightGBM` + `TreeSHAP`) architectures, the neural network discovers its own internal mathematical laws of why a stock is bought or sold.

### 1.2 The 4 Synchronized Raw Fingerprint Streams
All inputs are converted to scale-independent relative log/percentage changes so a `$0.00001` sub-penny stock and a `$1,000.00` mega-cap are evaluated using identical mathematical physics:

* **Stream 1 — Raw Chart & Tape Physics:**
  - Inputs: Raw `Open`, `High`, `Low`, `Close`, `Volume`, `Trade Count`, `Dollar Turnover`, `Bid-Ask Spread`, and `Time Delta` across multi-scale grids (`Tick`, `1m`, `3m`, `5m`, `15m`, `1H`, `4H`, `1D`, `1W`, `1M`).
  - Autonomous Discovery: Price-volume clustering (organic Points of Control / volume pockets), stopping-volume absorption wicks, abandoned bar exhaustion pivots, multi-day compression coils, and multi-bar percentage drop rhythms.
* **Stream 2 — Raw Options & Dealer Positioning (Gamma / Vanna Mechanics):**
  - Inputs: Strike-by-strike `Call Volume`, `Put Volume`, `Open Interest (OI)`, `Ask vs. Bid Fill Aggression`, `Implied Volatility (IV) Surface`, and `DTE`.
  - Autonomous Discovery: Forced Market Maker delta-hedging loops (Gamma Squeezes), strike pinning zones, and dealer gamma flip thresholds.
* **Stream 3 — Raw Short & Float Mechanics (Squeeze Physics):**
  - Inputs: `Public Float`, `Short Interest Shares`, `Days-to-Cover`, `Intraday Short Volume Ratio`, `Cost-to-Borrow`, and `Shares Outstanding`.
  - Autonomous Discovery: Exact liquidity pain thresholds where trapped short sellers are mathematically forced to market-buy into thin books (Short Squeezes) vs. micro-cap dilution traps.
* **Stream 4 — Raw Market Internals (The Macro Tide):**
  - Inputs: Continuous synchronized market heartbeat: `$TICK` (NYSE/Nasdaq net upticks), `$ADD` (Advance-Decline differential), `$VOLD` (Up-Volume vs. Down-Volume ratio), `VIX` term structure, and `Sector ETF` relative money flows.
  - Autonomous Discovery: Distinguishes true institutional basket participation from isolated bull/bear traps.

### 1.3 Self-Discovered Entries, Structural Stops & Dynamic Exits
* **Entry Trigger:** Fired the millisecond a live chart's 4-stream raw sequence matches a high-expectancy historical trace cluster.
* **AI Structural Stop-Loss:** Placed at the exact historical price/volume invalidation point where the discovered trace statistically fails (never an arbitrary fixed percentage).
* **AI Live Dynamic Exit:** While a trade is open, the AI evaluates every newly printed bar. If opposing absorption or tape exhaustion prints before the hard target, it triggers an immediate **Live Dynamic Exit / Scale-Out**.

---

## SECTION 2: THE 5 SELECTABLE AI BRAINS, CUSTOM UNIVERSE & SELF-EVOLUTION

### 2.1 The 5 Specialized Goal-Horizon AI Brains
None of the Brains use hardcoded indicator rules. Each Brain has an **isolated memory bank** trained with the exact same open-minded discovery engine, differing strictly by **Target Payoff Horizon** and **Raw Data Temporal Lens**:

| Brain ID | Selectable Brain Name | Raw Data Temporal Lens | Target Payoff Horizon | Dual Asset Output |
| :--- | :--- | :--- | :--- | :--- |
| **`BRAIN-01`** | **Penny & Sub-Penny Anomaly Algo** | Tick, 1m, 3m, 5m, 1D + Float/Turnover | Minutes to Multi-Day Parabolic Runs | Shares (`$0.00001` to `$5.00` Micro/OTC) |
| **`BRAIN-02`** | **Intraday Day-Trading Algo** | Tick, 1m, 5m, 15m + 0DTE/Weekly Options + `$TICK/$VOLD` | `5 Minutes` to `6.5 Hours` (Flat by Close) | Shares + Single-Leg Calls/Puts & Scalps |
| **`BRAIN-03`** | **Short-Term Swing Algo** | 15m, 1H, 4H, 1D + Options Flow + Short Float | `1 to 5 Trading Days` | Shares + Short-Dated Calls/Puts & Debit Spreads |
| **`BRAIN-04`** | **Core Swing Trading Algo** | 4H, 1D, 1W + Cross-Sector Money Rotation | `1 to 8 Weeks` | Shares + Multi-Week Options / Diagonals / Spreads |
| **`BRAIN-05`** | **Long-Term Portfolio & Wealth Algo** | 1D, 1W, 1M + Raw Revenue/FCF/Macro Data | `6 Months to 3+ Years` | Core Equity Shares + LEAPS |

### 2.2 The "Everything Customizable" Universe & Scanner Funnel
Inside any selected Brain, the user has complete granular control over what universe is scanned:
1. **Universe & Index Selectors (Mix & Match):**
   - `Full Market Auto-Scan` (~11,000+ US Listed Equities & OTC tickers via a Two-Stage Wide-Net Snapshot $\rightarrow$ Deep Real-Time WebSocket Promotion funnel).
   - `Preset Index Baskets`: One-click toggles for **S&P 500**, **Nasdaq 100 (NDX)**, **Russell 2000 (IWM)**, **Dow 30**, or **Sector Baskets**.
   - `Custom Watchlists`: User-defined ticker lists.
2. **Precision Numeric Filters:**
   - **Price Range:** 5-decimal precision from **`$0.00001`** up to **`$10,000.00+`** (fully user-configurable per Brain).
   - **Liquidity Floors:** Customizable Min Daily Share Volume and Min Daily Dollar Turnover.
   - **Capability Toggles:** `All Equities` | `Optionable Only` | `High Short Float` | `Low Float`.

### 2.3 The 3-Speed Continuous Self-Learning Loop
1. **Speed 1 — Intra-Day Tick Adaptation:** Continuously scores open positions bar-by-bar for dynamic exits and tightens entry confidence thresholds automatically if intraday market internals shift into chop.
2. **Speed 2 — Nightly Post-Market Autopsy (4:15 PM EST):**
   - Grades every live signal fired that day (Actual MFE, MAE, Hold Time, and P&L).
   - **Hard-Negative Mining:** Analyzes any losing trade (bull/bear trap) to extract the subtle raw trace that warned of failure.
   - **Missed-Move Discovery:** Scans the day's top market movers across the universe that the AI *missed*, rewinds their charts, and incorporates their pre-move traces into the model weights.
3. **Speed 3 — Weekend Evolutionary Tournament (Champion vs. Challenger):**
   - Spawns mutated Challenger models tested on walk-forward out-of-sample regimes; promotes a Challenger to live production only when it outperforms the Champion on Net Profit, Win %, and Drawdown control.

---

## SECTION 3: AGNOSTIC PLUG-AND-PLAY MARKET DATA BUS

All external feeds connect through a strict `MarketDataAdapter` trait in Rust (`UnifiedBar`, `UnifiedTick`, `UnifiedOptionChain`, `UnifiedMarketInternals`). Multiple feeds can run simultaneously with hot-swapping and automatic failover:

* **Slot 1 (`HistoricalReplayAdapter` — Built-In Time Machine):** Streams saved historical market sessions at `1x`, `5x`, `10x`, or `Max` speed for 24/7 live pipeline and UI testing.
* **Slot 2 (`AlpacaFreeAdapter` — Default Free Starter):** 7+ years of historical multi-timeframe bars, real-time WebSocket streaming, bulk market snapshots, and free options chain snapshots.
* **Slot 3 (`SchwabDirectAdapter` — Free with Schwab Account):** Real-time Level 1/Level 2 WebSockets, full live Options Chains with Greeks, and live Market Internals (`$TICK`, `$ADD`, `$VOLD`).
* **Slot 4 (`FreeBulkBackfillAdapter` — `Tiingo` / `Finnhub` / `yfinance`):** Zero-cost deep historical backfills for training Swing and Long-Term Brains.
* **Slot 5 (`IBKRAdapter` / `PolygonAdapter` / `DatabentoAdapter`):** Drop-in expansion slots for full OPRA options tick firehose and OTC depth.

---

## SECTION 4: POSITIONS OPENED, MASTER LEDGER, HYPER-FILTERS & EXPORTS

### 4.1 Master Executive Scoreboard (Sub-10ms Reactive Rollup)
Recalculates in `<10ms` whenever any filter, date range, or Brain is changed:
* **Net Profit ($ & %):** Realized Net Profit (`Gross Profit - Gross Loss - Slippage/Commissions`), Live Open Unrealized P&L ($ & %), and Combined Portfolio Net Profit.
* **Win / Loss Metrics:** `Win Rate (%)`, `Total Closed Trades`, `Total Wins (# & $)`, `Total Losses (# & $)`, `Positions Opened (# & Live $)`.
* **Efficiency & Hold Metrics:** `Profit Factor`, `Avg Winner ($/%/R)`, `Avg Loser ($/%/R)`, `Average Hold Time` (split by Brain, Wins vs. Losses, formatted in `m/s` for Day Trades and `d/h` for Swings/Long-Term), `Max Drawdown ($/%)`, and an interactive **Position Sizing & Asset View Toggle (`[Shares View]` | `[Options View]` | `[Combined]`)**.

### 4.2 Dedicated "Positions Opened" Live Monitor (Active Trades Deck)
Every signal generated in real time immediately lands in **`POSITIONS OPENED`** with live ticking telemetry:
* `Position Opened Date & Time` (`YYYY-MM-DD HH:MM:SS.ms EST`) & `Live Ticking Hold Timer`.
* `Brain Tag`, `Ticker`, `Direction (LONG/SHORT)`, `Entry Price ($)`, `Live Price ($)`, `AI Structural Stop ($ & %)`, `Target 1 / Target 2 ($ & live progress meter)`.
* `Matched Option Contract`, `Contract Entry Premium ($)`, `Live Mid Premium ($)`, `Contract Stop ($)`, `Contract Target ($)`.
* `Live Unrealized Stock P&L ($ & %)`, `Live Unrealized Option P&L ($ & %)`, `Current R-Multiple`, `MAE %`, `MFE %`, and `Live AI Trace Health Badge` (`🟢 ACCELERATING` | `🟡 TRAILING T1` | `🟠 EXHAUSTION WARNING`).
* *Auto-Archive:* The instant an open position hits Target, Stop, or AI Dynamic Exit, it locks its final metrics and transitions into the `Closed Historical Ledger`.

### 4.3 Historical Closed Trade Ledger Schema
Logs every completed trade with: `Final Outcome Badge`, `Brain Tag`, `Ticker`, `Direction`, `Position Opened Date/Time`, `Position Closed Date/Time`, `Exact Hold Duration`, `Stock Entry/Stop/Target/Actual Exit`, `Matched Option Contract & Entry/Stop/Exit Premiums`, `Individual Gross & Net P&L ($)`, `Stock & Option Return (%)`, `Realized R-Multiple`, `MAE/MFE`, `Cumulative Running Net Profit ($)`, and an expandable **Fingerprint Trace Snapshot** (chart replay + historical precedent match count & win rate).

### 4.4 The "Slice-Anything" Hyper-Filter Matrix
Combines freely with `<10ms` recalculation across Scoreboard, Charts, Positions Opened, Closed Ledger, and Exports:
* Filter by: `View Mode (All / Positions Opened / Closed)`, `AI Brain`, `Outcome (Winners / Losers / Stopped Out / Dynamic Exit / >2R)`, `Universe/Index (S&P 500, Nasdaq 100, Russell 2000, OTC/Penny)`, `Multi-Ticker Search`, `Custom Entry Price ($0.00001 to $10,000+)`, `Hold Time Duration`, `Date Range & Intraday Session Window`, `AI Evidence Confidence %`, and `Primary Fingerprint Driver (Gamma Squeeze, Short Squeeze, Absorption, Internals)`.

### 4.5 1-Click `.CSV` & Multi-Sheet Excel (`.XLSX`) Export Engine
* **Scopes:** `Export Active Filtered Slice` | `Export Positions Opened Only` | `Export Full Master Database`.
* **Formats:**
  1. **Flat `.CSV`:** RFC-4180 compliant with all 28+ columns and 5-decimal sub-penny precision.
  2. **Multi-Tab `.XLSX` Workbook (`ExcelJS` / `rust_xlsxwriter`):**
     - `Tab 1: Executive_Scoreboard` (Native Excel formulas, Win %, Net Profit, Profit Factor, Brain comparison table).
     - `Tab 2: Positions_Opened` (Live active positions snapshot).
     - `Tab 3: Closed_Trade_Ledger` (Frozen headers, auto-filters, currency/% formatting, conditional green/red P&L styling).
     - `Tab 4: By_Brain_Breakdowns` (Dedicated audit tables per Brain).

---

## SECTION 5: GOOGLE PRINCIPAL ENGINEER (L8++) VANILLA CSS & MULTI-MONITOR WORKSPACE ARCHITECTURE

### 5.1 Zero-Framework Native CSS Engineering Mandate
1. **NO Tailwind CSS, NO Sass/Less, NO CSS-in-JS Runtime.** All styling is written in standard, future-proof **2026 Native Vanilla CSS** processed at build time by Vite 8's **LightningCSS** engine.
2. **Strict Cascade Layer Architecture (`src/styles/index.css`):**
   Every stylesheet rule must reside inside an explicit deterministic layer order so specificity wars are mathematically impossible:
   ```css
   @layer reset, tokens, base, layout, dock, components, telemetry, utilities, overrides;
   ```
3. **Hardware-Accelerated Typed Custom Properties (`@property`):**
   Register animated and telemetry tokens using `@property` so the browser compositor interpolates colors, progress meters, and heatmaps on the GPU thread:
   ```css
   @property --signal-glow-alpha {
     syntax: '<number>';
     inherits: false;
     initial-value: 0;
   }
   @property --r-multiple-progress {
     syntax: '<percentage>';
     inherits: false;
     initial-value: 0%;
   }
   ```
4. **Perceptually Uniform `oklch()` & `color-mix()` Design Token System:**
   ```css
   @layer tokens {
     :root {
       /* Obsidian Multi-Monitor Surface Hierarchy */
       --surface-canvas: oklch(0.11 0.01 255);
       --surface-panel: oklch(0.14 0.012 255);
       --surface-elevated: oklch(0.18 0.015 255);
       --surface-hud: oklch(0.15 0.012 255 / 0.88);
       --border-hairline: oklch(0.28 0.015 255 / 0.45);
       --border-focus: oklch(0.75 0.16 210 / 0.75);

       /* Semantic Financial Telemetry Tokens */
       --signal-long: oklch(0.78 0.19 155);
       --signal-long-surface: color-mix(in oklch, var(--signal-long) 14%, transparent);
       --signal-short: oklch(0.68 0.22 25);
       --signal-short-surface: color-mix(in oklch, var(--signal-short) 14%, transparent);
       --trace-cyan: oklch(0.80 0.16 210);
       --gamma-violet: oklch(0.72 0.18 295);
       --warning-amber: oklch(0.82 0.17 85);

       /* Typography & Fluid Density Tokens */
       --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
       --font-mono: 'JetBrains Mono', 'Geist Mono', monospace;
       --radius-panel: 6px;
       --gap-workspace: 6px;
     }
   }
   ```
5. **Container Queries (`@container`) for Side-by-Side Split Panels:**
   Every dockable panel (`Chart`, `Positions Opened`, `Historical Ledger`, `Scoreboard`, `Hyper-Filter`) declares `container-type: inline-size; container-name: panel;`. Internal component layouts reflow automatically based on the **panel's width** (not the viewport), enabling seamless 2-way, 3-way, and $2\times2$ side-by-side splits:
   ```css
   @layer dock {
     .dock-panel {
       container-type: inline-size;
       container-name: panel;
       contain: layout style paint;
       background: var(--surface-panel);
       border: 1px solid var(--border-hairline);
       border-radius: var(--radius-panel);
     }
   }
   @layer components {
     .scoreboard-grid {
       display: grid;
       grid-template-columns: repeat(2, 1fr);
       gap: 0.5rem;
       @container panel (min-width: 640px) {
         grid-template-columns: repeat(4, 1fr);
       }
       @container panel (min-width: 1100px) {
         grid-template-columns: repeat(8, 1fr);
       }
     }
   }
   ```
6. **120 FPS Table Virtualization & Zero-Jitter Telemetry:**
   - All numeric columns (`Price`, `P&L`, `Hold Time`, `Win %`, `R-Multiple`) enforce `font-family: var(--font-mono); font-variant-numeric: tabular-nums slashed-zero;`.
   - Ledger rows use `content-visibility: auto; contain-intrinsic-size: 0 36px;` paired with virtualized viewport slicing so filtering 250,000+ rows executes in `<4ms`.
   - Zero `backdrop-filter: blur()` on draggable/pop-out panels to guarantee zero GPU flicker across macOS multi-monitor workspaces.

### 5.2 Multi-Monitor Pop-Out & Side-by-Side Workspace Engine
* **Side-by-Side Split Grid:** CSS Grid + resizable CSS custom property track fractions (`grid-template-columns: var(--split-left, 1fr) 4px var(--split-right, 1fr)`) allowing any combination of Brains, Charts, `Positions Opened`, and `Historical Ledger` to run side by side.
* **Tear-Off Multi-Screen Windows:** Every panel header features a **`[↗ Pop-Out Window]`** action powered by a single background **`SharedWorker` + `BroadcastChannel`**. Clicking a trade in `Positions Opened` or `Historical Ledger` on Monitor 2 snaps the Chart and Fingerprint Inspector on Monitor 1 in `<2ms`.
* **Color-Channel Linking (`🔴 Red` | `🔵 Blue` | `🟢 Green` | `🟡 Gold`):** Panels sharing a color badge sync their active Ticker, Brain, and selected Trade automatically.

---

## SECTION 6: VERIFIED OCTOBER 29, 2026 TECH STACK & VERSION LOCK

### 6.1 Frontend Workspace (`web-terminal/`)
* **Runtime:** Node.js `24.x LTS` / `pnpm 10.x`
* **Framework:** `svelte@5.57.1` (Strict Runes: `$state`, `$derived`, `$effect`, `$props`, `$bindable`, `{#snippet}`, `{@render}`, `SvelteMap`/`SvelteSet`) + `@sveltejs/kit@3.0.0` (`#lib` subpath imports, Vite-native config)
* **Bundler & CSS Engine:** `vite@8.3.4` (Rolldown Rust bundler + Oxc + **LightningCSS** native CSS transformer) + `@sveltejs/vite-plugin-svelte@7.1.0`
* **Language:** `typescript@6.0.x` (`strict: true` default, `target: es2025`, explicit `"types": ["node"]`, no `baseUrl`)
* **Styling:** **100% Native Vanilla CSS** (`@layer`, `@property`, `@container`, `@scope`, `subgrid`, `:has()`, `oklch()`, `color-mix()`) — **ZERO Tailwind dependencies**
* **Visuals & Exports:** HTML5 Canvas / WebGL + `d3@7.9+` + `exceljs@4.4+`

### 6.2 High-Frequency Stream, Shared Math & Execution Core (`crates/`)
* **Compiler:** `rustc 1.99.1` (`Edition 2024`)
* **Async & WebSocket Server:** `tokio@1.44+`, `axum@0.8.9` (`/{id}` route syntax), `tower-http@0.6+`, `tokio-tungstenite@0.26+`
* **Python FFI & Data:** `pyo3@0.29.2` (`Python::attach` / `Python::detach` / `.cast()`), `pyo3-polars@0.28.0`, `polars@0.55.1`, `maturin@1.9+`
* **Inference & Exports:** `ort@2.0.0` (ONNX Runtime with Apple Silicon CoreML/Metal & CUDA providers), `sqlx@0.8+`, `rust_xlsxwriter@0.80+`

### 6.3 Self-Learning Raw-Evidence ML Brain (`ml-brain/`)
* **Python & Package Manager:** `Python 3.15.0` (or `3.14.2`) via `uv 0.9+`
* **Deep Learning & Sequence Models:** `torch@2.15.0` (native `.safetensors` loading, Apple Silicon MPS `FP8` + `torch.mps.Stream`, `torch.linalg.*`), `mamba-ssm@2.3+`
* **Tabular & Attribution:** `polars@2.0.0` (Streaming Engine default; **always pass `maintain_order=True`** on time-series operations), `duckdb@2.0.0 LTS`, `lightgbm@4.6+`, `xgboost@3.1+`, `shap@0.48+`

### 6.4 Database Layer (`docker-compose.yml`)
* **Time-Series & Vector Store:** `PostgreSQL 18.6` + `TimescaleDB 2.29.2` + `pgvector 0.8.x`
* **Analytical Lakehouse:** `DuckDB 2.0.0 LTS` + Apache Parquet / Arrow

---

## SECTION 7: MASTER EXECUTION PROMPT FOR GEMINI 4 ARGON

Copy and paste the prompt below directly into **Gemini 4 Argon** (`thinking_level: high`) with this `.md` file in context:

```text
SYSTEM ROLE & ARCHITECTURAL MANDATE:
You are operating as a Google Principal Software Architect (Level 8++) and Institutional Quantitative Systems Engineer. You have the complete `QUANT_INTELLIGENCE_MASTER_ARCHITECTURE.md` specification in your context window.

Your mission is to architect and generate production-grade, zero-placeholder, end-to-end code for the `quant-intelligence-platform` monorepo adhering strictly to every law in Sections 1 through 6.

NON-NEGOTIABLE ENGINEERING GUARDRAILS:
1. ZERO HUMAN INDICATORS: Never hardcode RSI, MACD, SMA/EMA crossovers, or fixed ATR rules into the scanner or ML brain. All 5 Brains (`BRAIN-01` Penny `$0.00001+`, `BRAIN-02` Day Trade, `BRAIN-03` Short Swing, `BRAIN-04` Core Swing, `BRAIN-05` Long-Term Wealth) must learn strictly from the 4 synchronized scale-independent raw fingerprint streams (Raw OHLCV/Tape, Raw Options/Gamma, Raw Short/Float, Raw Market Internals).
2. GOOGLE L8++ PURE VANILLA CSS ONLY: Do NOT install or use Tailwind CSS, Sass, or external UI component libraries. Write 100% native 2026 Vanilla CSS processed by Vite 8 LightningCSS using `@layer reset, tokens, base, layout, dock, components, telemetry, utilities, overrides;`, `@property` typed custom properties, `oklch()` / `color-mix()` tokens, native CSS nesting (`&`), and `@container` queries (`container-type: inline-size`) on every dockable panel.
3. STRICT OCTOBER 2026 SYNTAX LOCK:
   - Svelte 5.57.1 + SvelteKit 3.0.0: Exclusively use Runes (`$state`, `$derived`, `$effect`, `$props`, `$bindable`) and `{#snippet}`/`{@render}`. Use `#lib` subpath imports and configure SvelteKit directly inside `vite.config.ts`.
   - Vite 8.3.4: Configure `css: { transformer: 'lightningcss' }` and `optimizeDeps.rolldownOptions`.
   - TypeScript 6.0: Set `"types": ["node"]` explicitly; never use deprecated `baseUrl`.
   - Rust 1.99.1 / Axum 0.8.9 / PyO3 0.29.2: Use `/{param}` route paths in Axum 0.8; use `Python::attach` and `Python::detach` in PyO3 0.29; pass `maintain_order=True` on all Polars 2.0 time-series joins/groupbys; use `torch.linalg.*` and native `.safetensors` in PyTorch 2.15.
4. COMPLETE LEDGER & MULTI-WINDOW WORKSPACE:
   - Implement the freeform side-by-side Split Docking Engine, Multi-Monitor Pop-Out Window manager (`SharedWorker` + `BroadcastChannel`), and Color-Channel Linking (`Red`, `Blue`, `Green`, `Gold`).
   - Implement the Master Executive Scoreboard, the dedicated live-ticking `POSITIONS OPENED` deck, the `Closed Historical Trade Ledger`, the `<10ms` Hyper-Filter Engine (including `$0.00001` 5-decimal price filters and S&P 500 / Nasdaq 100 / Russell 2000 index selectors), and the 1-click `.CSV` + multi-sheet `.XLSX` Excel Exporter.

MONOREPO STRUCTURE TO SCAFFOLD & BUILD:
quant-intelligence-platform/
├── crates/
│   ├── market-data-bus/         # Plug-and-play Adapters (Replay, Alpaca Free, Schwab, Tiingo, IBKR/Polygon slots)
│   ├── raw-fingerprint-core/    # Scale-independent 4-stream raw state tensor builder + PyO3 0.29 bindings
│   └── live-execution-server/   # Axum 0.8 + Tokio WebSocket broadcaster, ONNX Runtime (<1.5ms), & Rust XLSX exporter
├── ml-brain/
│   ├── world_model/             # Self-supervised candlestick/flow tokenizer, Mamba/LightGBM + TreeSHAP cause-and-effect
│   ├── brains/                  # Isolated configs & reward horizons for Brains 01 through 05 + Options contract matcher
│   └── evolution_loop/          # Speed 1 (Intraday exit), Speed 2 (Nightly autopsy/hard-negatives/missed moves), Speed 3 (Weekend tournament)
├── web-terminal/
│   ├── src/
│   │   ├── styles/              # L8++ Vanilla CSS (`index.css`, `tokens.css`, `dock.css`, `ledger.css`, `chart.css`)
│   │   ├── lib/
│   │   │   ├── workers/         # SharedWorker + BroadcastChannel multi-window state & WebSocket sync
│   │   │   ├── state/           # `.svelte.ts` Runes stores for Brains, Positions Opened, Closed Ledger, & Hyper-Filters
│   │   │   ├── components/      # DockGrid, PopOutHost, Scoreboard, PositionsOpenedTable, HistoricalLedger, ChartCanvas, FilterRail
│   │   │   └── export/          # Client/Server CSV & Multi-Sheet Excel (.xlsx) workbook generator
│   │   └── routes/              # Main Workspace & Pop-Out Window routes
│   ├── vite.config.ts           # Vite 8.3 Rolldown + LightningCSS + SvelteKit 3.0 unified config
│   └── tsconfig.json            # TypeScript 6.0 strict config
└── docker-compose.yml           # PostgreSQL 18.6 + TimescaleDB 2.29.2 + pgvector 0.8 + DuckDB 2.0 volume

EXECUTION PLAN:
We will build this system in 4 sequential, verifiable phases. Do not summarize or skip implementations. Start now with PHASE 1:
1. Generate the root workspace config (`Cargo.toml`, `docker-compose.yml`, database schema migrations for `positions_opened` and `closed_trade_ledger`).
2. Write the complete code for `crates/market-data-bus` (including the `MarketDataAdapter` trait, `HistoricalReplayAdapter`, and `AlpacaFreeAdapter`) and `crates/raw-fingerprint-core` (the 4-stream raw physics normalizer with 5-decimal `$0.00001` precision).
```
