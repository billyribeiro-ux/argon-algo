# 05 — State-of-the-Art UI/UX, Side-by-Side Docking & Multi-Monitor Spec
**Version:** 1.0 (Living Blueprint)  
**Frontend Stack:** Svelte 5 (Runes: `$state`, `$derived`, `$effect`), SvelteKit, TypeScript, Tailwind CSS, HTML5 Canvas / WebGL + D3.js  
**Core Mandate:** Institutional Multi-Monitor Freedom, Side-by-Side Brain/View Splitting, 120 FPS Performance, and Ultra-High-End OKLCH Aesthetics.

---

## 1. Multi-Monitor & Side-by-Side Docking Engine
The UI is built on a **Modular Workspace & Window Manager** where no panel is permanently trapped in a static position.

### A. Freeform Split-Screen & Side-by-Side Docking
* **Drag-and-Drop Split Zones:** Every module (`Live Chart`, `Execution Ledger`, `Executive Scoreboard`, `Hyper-Filter Scanner`, `AI Fingerprint Inspector`, `Equity Curve`, `AI Autopsy Log`) is a dockable panel.
* **Side-by-Side Modes:**
  1. **Side-by-Side Brains:** Run 2, 3, or 4 different AI Brains side by side in split columns or a $2 \times 2$ quad grid (e.g., *Day-Trading Brain* on the left half, *Short-Term Swing Brain* on the right half), each with its own live scanner and filtered ledger.
  2. **Side-by-Side Multi-Timeframe / Stock + Options Charts:** View the underlying stock chart and the matched Options contract chart (or 1m + 5m + 15m + Daily charts) side by side with synchronized crosshairs.
  3. **Side-by-Side Ledger & Chart:** Dock the Master Execution Ledger either **below** the chart (Horizontal Split) or **side-by-side** next to the chart (Vertical Split) for ultrawide/dual-monitor setups.

### B. Multi-Screen "Tear-Off" Pop-Out Windows
* Every panel header includes a **`[↗ Pop-Out Window]`** control using the Multi-Screen Window Placement API (`window.getScreenDetails()`).
* **Zero-Latency Cross-Window Sync (`SharedWorker` + `BroadcastChannel`):**
  - A single `SharedWorker` manages the live WebSocket stream and filtered state store across all open browser/desktop windows.
  - Clicking a trade in a popped-out Ledger on **Monitor 2** instantaneously updates the Chart and AI Fingerprint panel on **Monitor 1** (`<2ms` IPC latency).
* **One-Click "Restore All Windows":** Saving a workspace remembers the exact monitor coordinates (`screenX`, `screenY`, `width`, `height`) of every popped-out window and restores them across all screens with one click.

### C. Color-Channel Linking (`🔴 Red` | `🔵 Blue` | `🟢 Green` | `🟡 Gold` | `⚪ Unlinked`)
* Every panel features a color-link selector.
* Panels assigned to the same color channel synchronize their active Ticker, Trade Selection, and Brain automatically, allowing multiple independent workflows to run side by side without interfering with each other.

---

## 2. Visual Design System & Institutional Aesthetics

### A. OKLCH Color Architecture & Surface Hierarchy
* **Base Obsidian Canvas:** `oklch(0.12 0.01 255)` (`#090c10`)
* **Docked Panel Surface:** `oklch(0.15 0.012 255)` (`#0f1419`)
* **Elevated Header / Floating HUD:** `rgba(15, 20, 25, 0.85)` (Zero `backdrop-filter` flicker during macOS multi-monitor window dragging; crisp `1px solid rgba(255, 255, 255, 0.08)` hairline borders).
* **Semantic Signal Tokens:**
  - `Signal Long / Positive Net P&L`: `oklch(0.78 0.19 155)` (Electric Emerald)
  - `Signal Short / Stop-Loss / Negative P&L`: `oklch(0.68 0.22 25)` (Laser Crimson)
  - `AI Attention Trace / Gamma Hotspot`: `oklch(0.80 0.16 210)` (Cyber Cyan)
  - `Active In-Trade Pulse`: `oklch(0.82 0.17 85)` (Institutional Amber)

### B. Typography & Micro-Telemetry
* **UI & Headers:** `Inter` / `Geist Sans` (tight tracking `-0.02em`, high-contrast legibility).
* **Numeric & Ledger Data:** `JetBrains Mono` / `Geist Mono` with mandatory `font-variant-numeric: tabular-nums` and `slashed-zero` so live ticking prices, hold times, and P&L columns never shift horizontally.
* **Inline Visual Cells:** Every trade row in the Ledger renders a 60px inline Canvas sparkline of the trade's price path, an `R-Multiple` bar meter, and a color-coded `Hold Time` badge.

---

## 3. Interactive Chart & AI Trace Canvas Specification
1. **Hardware-Accelerated Candlesticks & Volume Profile:** 120 FPS zoom/pan canvas supporting sub-penny (`$0.00001`, 5-decimal precision) to `$10,000+` price scales automatically.
2. **Automated Trade Lifecycle Plotting:**
   - Plots exact **ENTRY arrow + price tag**, **AI Structural STOP line (red dashed)**, **TARGET 1/2 lines (emerald dashed)**, and **FINAL EXIT marker**.
   - Draws a semi-transparent shaded duration box connecting Entry to Exit displaying `+R Multiple`, `Net P&L ($/%)`, and `Exact Hold Time` directly on the canvas.
3. **Neural Attention Heatmap Toggle:**
   - Optional overlay that illuminates the exact historical candles and volume bars that triggered the AI Brain's raw-evidence pattern match.

---

## 4. Workspace Presets & Command Palette (`Cmd + K`)
* **Saved Layout Slots:** Top bar features 1-click preset buttons (`Layout 1`, `Layout 2`, `Layout 3`, `+ Save Current Layout`) storing full split-grid ratios and multi-monitor window placements.
* **Global Command Palette (`Cmd + K`):** Instant keyboard-driven control to switch Brains, filter the Ledger (e.g., type `> filter brain:swing outcome:win`), jump to any ticker, or pop out a panel.

---

## 5. Future Additions / Brainstorm Notes
* [ ] *Add custom theme tokens, sound/voice alert synthesis, or custom panel widgets here...*

