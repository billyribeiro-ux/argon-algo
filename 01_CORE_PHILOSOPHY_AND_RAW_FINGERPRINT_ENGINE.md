# 01 — Core Philosophy & Raw Fingerprint Engine Specification
**Version:** 1.0 (Living Blueprint)  
**Target Architect Model:** Gemini 4 Argon (`thinking_level: high`, 1M Context Window)  
**Core Mandate:** Zero Human Indicators. 100% Self-Supervised Raw Market Evidence Discovery.

---

## 1. Philosophical Mandate (The "Open-Minded" Law)
1. **Nothing Happens Randomly:** Every movement in the financial market is caused by physical liquidity mechanics (institutional accumulation/distribution, forced dealer gamma hedging, trapped short liquefaction, and macro basket flows). Every action leaves an indelible mathematical fingerprint on the chart and tape.
2. **STRICT BAN on Pre-Built Human Indicators:** 
   - Do **NOT** hardcode lagging retail indicators (RSI, MACD, Stochastic, standard SMA/EMA crossovers, or arbitrary ATR rules) as scanner filters.
   - Pre-built strategies close the algorithm's mind and blind it to multi-dimensional cause-and-effect.
3. **Pure Raw-Evidence Discovery (Financial World Model):**
   - The system ingests raw historical market physics across years of data and works backward from undeniable historical price expansions and collapses.
   - Using K-Line / Candlestick Tokenization (inspired by domain-specific financial foundation models like *Kronos* and continuous flow-matching *KiT* / *Mamba-3* state-space models), the neural network discovers its own latent mathematical representations of why a stock is bought or sold.

---

## 2. The 4 Synchronized Raw Fingerprint Streams
All inputs are normalized to scale-independent relative log/percentage changes so a `$0.0001` sub-penny stock and a `$1,000` mega-cap are evaluated using the exact same mathematical physics.

### Stream 1: Raw Chart & Tape Physics (OHLCV + Microstructure)
* Raw `Open`, `High`, `Low`, `Close`, `Volume`, `Trade Count`, `VWAP/Turnover`, and `Time Delta` across multi-scale grids (`Tick`, `1m`, `5m`, `15m`, `1H`, `4H`, `1D`, `1W`).
* **What the AI Discovers Autonomously:** Price-volume clustering (organic Points of Control / volume nodes), stopping-volume absorption wicks, abandoned bar exhaustion gaps, multi-day volatility compression coils, and multi-bar percentage drop rhythms.

### Stream 2: Raw Options & Dealer Positioning (Gamma / Vanna Mechanics)
* Raw Strike-by-Strike `Call Volume`, `Put Volume`, `Open Interest (OI)`, `Bid vs. Ask Fill Aggression`, `Implied Volatility (IV) Term Structure`, and `Time to Expiration`.
* **What the AI Discovers Autonomously:** Forced Market Maker delta-hedging loops (Gamma Squeezes), strike pinning zones, and dealer gamma flip thresholds.

### Stream 3: Raw Short & Float Mechanics (Squeeze Physics)
* Raw `Public Float`, `Short Interest Shares`, `Days-to-Cover`, `Intraday Short Volume Ratio`, `Cost-to-Borrow`, and `Shares Outstanding`.
* **What the AI Discovers Autonomously:** Exact liquidity exhaustion points where trapped short sellers are mathematically forced to cover into thin order books (Short Squeezes) vs. dilution traps.

### Stream 4: Raw Market Internals (The Macro Tide)
* Continuous synchronized market heartbeat: `$TICK` (NYSE/Nasdaq net upticks), `$ADD` (Advance-Decline differential), `$VOLD` (Up-Volume vs. Down-Volume ratio), `VIX` spot vs. futures curve, and `Sector ETF` relative money flows.
* **What the AI Discovers Autonomously:** Whether an individual stock breakout is backed by broad institutional basket buying or is an isolated bull trap fighting a negative macro tide.

---

## 3. Reverse-Trace Self-Supervised Training Protocol
1. **Ground-Truth Pinning:** Scan years of historical bars across the entire market universe. Tag every structural displacement (explosive rally or flush) and every failed fakeout.
2. **Multi-Horizon Rewind:** Rewind the raw 4-stream tensor `5`, `50`, `500`, and `2,000` bars prior to the inflection point.
3. **Latent Cause-and-Effect Extraction:** Train the state-space / transformer backbone to predict the full future payoff distribution (`Expected Move %`, `Time-to-Peak Hold Duration`, `Max Adverse Excursion [MAE]`, `Max Favorable Excursion [MFE]`).
4. **Structural Stop & Exit Discovery:**
   - **Stop-Loss:** Placed at the exact historical price/volume invalidation point where the discovered trace statistically fails (never an arbitrary fixed percentage).
   - **Dynamic Exit:** Triggered live when real-time bars print the learned mathematical signature of buying exhaustion or institutional distribution.

---

## 4. Future Additions / Brainstorm Notes
* [ ] *Add future notes or custom raw data streams here...*
