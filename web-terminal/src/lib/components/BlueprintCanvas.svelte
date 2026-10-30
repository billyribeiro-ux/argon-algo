<script lang="ts">
  import {
    terminalStore,
    formatSubPennyPrice,
    formatSignedUsd,
    type ScannerSignalCandidate,
    type ChartTimeframeId,
    type ChartRangeId,
    type RawBarCandle
  } from '$lib/state/terminal.svelte';

  const TIMEFRAMES: ChartTimeframeId[] = ['1m', '3m', '5m', '15m', '1H', '4H', '1D', '1W', '1M'];
  const RANGES: ChartRangeId[] = ['1D', '5D', '1M', '6M', '1Y', '5Y'];

  let showVerificationPopover = $state(false);
  let customSymbolInput = $state('');
  let hoveredBarIndex = $state<number | null>(null);
  let hoverMouseX = $state<number>(0);
  let hoverMouseY = $state<number>(0);

  // Interactive Zoom & Pan state
  let visibleBarCount = $state<number>(64);
  let panOffsetBars = $state<number>(0);
  let isDragging = $state<boolean>(false);
  let dragStartX = $state<number>(0);
  let dragStartPanOffset = $state<number>(0);

  const blueprint = $derived<ScannerSignalCandidate>(terminalStore.activeBlueprint);
  const symbolClosedTrades = $derived(terminalStore.selectedSymbolClosedTrades);
  const symbolOpenPositions = $derived(
    terminalStore.openPositions.filter(
      (p) => p.symbol.toUpperCase() === blueprint.symbol.toUpperCase()
    )
  );

  const windowedBars = $derived.by((): RawBarCandle[] => {
    const total = blueprint.bars.length;
    if (total === 0) return [];
    const count = Math.max(8, Math.min(total, visibleBarCount));
    const maxOffset = Math.max(0, total - count);
    const clampedOffset = Math.max(0, Math.min(maxOffset, panOffsetBars));
    const endIdx = total - clampedOffset;
    const startIdx = Math.max(0, endIdx - count);
    return blueprint.bars.slice(startIdx, endIdx);
  });

  const hoveredBar = $derived.by(() => {
    if (hoveredBarIndex === null) return null;
    return windowedBars[hoveredBarIndex] ?? null;
  });

  const hoveredBarLogReturnPct = $derived.by(() => {
    if (!hoveredBar || hoveredBar.open <= 0) return 0;
    return Math.log(hoveredBar.close / hoveredBar.open) * 100;
  });

  const chartGeometry = $derived.by(() => {
    const bars = windowedBars;
    const allPrices = [
      blueprint.entryPrice,
      blueprint.structuralStopPrice,
      blueprint.target1Price,
      blueprint.target2Price,
      ...bars.flatMap((b) => [b.high, b.low]),
      ...symbolClosedTrades.flatMap((t) => [t.entryPrice, t.exitPrice]),
      ...symbolOpenPositions.flatMap((p) => [p.entryPrice, p.livePrice])
    ].filter((p) => Number.isFinite(p) && p > 0);

    const safePrices = allPrices.length > 0 ? allPrices : [100, 105];
    const minP = Math.min(...safePrices) * 0.996;
    const maxP = Math.max(...safePrices) * 1.004;
    const rangeP = Math.max(maxP - minP, 0.00001);
    const maxVol = Math.max(1, ...bars.map((b) => b.volume));

    const width = 760;
    const height = 350;
    const padTop = 22;
    const padBottom = 64;
    const plotHeight = height - padTop - padBottom;

    const priceToY = (p: number) =>
      padTop + plotHeight - ((p - minP) / rangeP) * plotHeight;

    return {
      width,
      height,
      minP,
      maxP,
      maxVol,
      entryY: priceToY(blueprint.entryPrice || safePrices[0]!),
      stopY: priceToY(blueprint.structuralStopPrice || safePrices[0]!),
      t1Y: priceToY(blueprint.target1Price || safePrices[0]!),
      t2Y: priceToY(blueprint.target2Price || safePrices[0]!),
      priceToY
    };
  });

  function handleSymbolSubmit(e: SubmitEvent): void {
    e.preventDefault();
    const sym = customSymbolInput.trim().toUpperCase();
    if (!sym) return;
    panOffsetBars = 0;
    terminalStore.selectTicker(sym);
  }

  function handleWheelZoom(e: WheelEvent): void {
    e.preventDefault();
    const total = Math.max(16, blueprint.bars.length);
    if (e.deltaY < 0) {
      visibleBarCount = Math.max(10, Math.floor(visibleBarCount * 0.85));
    } else {
      visibleBarCount = Math.min(total, Math.ceil(visibleBarCount * 1.15));
    }
  }

  function handlePointerDown(e: PointerEvent): void {
    isDragging = true;
    dragStartX = e.clientX;
    dragStartPanOffset = panOffsetBars;
  }

  function handlePointerMove(e: PointerEvent): void {
    const svg = e.currentTarget as SVGSVGElement;
    const rect = svg.getBoundingClientRect();
    const relX = ((e.clientX - rect.left) / Math.max(1, rect.width)) * chartGeometry.width;
    const relY = ((e.clientY - rect.top) / Math.max(1, rect.height)) * chartGeometry.height;
    hoverMouseX = relX;
    hoverMouseY = relY;

    if (windowedBars.length > 0) {
      const stepX = (chartGeometry.width - 140) / Math.max(1, windowedBars.length);
      const rawIdx = Math.round((relX - 60) / stepX);
      if (rawIdx >= 0 && rawIdx < windowedBars.length) {
        hoveredBarIndex = rawIdx;
      } else {
        hoveredBarIndex = null;
      }
    }

    if (isDragging) {
      const dx = e.clientX - dragStartX;
      const barsDelta = Math.round(dx / 8);
      const maxOffset = Math.max(0, blueprint.bars.length - visibleBarCount);
      panOffsetBars = Math.max(0, Math.min(maxOffset, dragStartPanOffset + barsDelta));
    }
  }

  function handlePointerUp(): void {
    isDragging = false;
  }

  function resetZoomAndPan(): void {
    visibleBarCount = Math.max(32, blueprint.bars.length);
    panOffsetBars = 0;
  }
</script>

<section class="glass-panel panel-blueprint-canvas" aria-label="Raw Price & AI Trade Blueprint Canvas">
  <header class="panel-header" style="flex-wrap: wrap; gap: 8px;">
    <div class="panel-title" style="flex-wrap: wrap;">
      <span class="badge {blueprint.direction === 'LONG' ? 'badge-long' : 'badge-short'}">
        {blueprint.direction}
      </span>
      <span>{blueprint.symbol}</span>
      <span class="tabular-num" style="color: var(--signal-ai-core);">
        {formatSubPennyPrice(blueprint.livePrice)}
      </span>
      <span class="badge badge-ai-core">{blueprint.brainId}</span>
      <span class="badge badge-gamma">{blueprint.exchangeIndex}</span>
      {#if terminalStore.chartLoading}
        <span class="badge badge-squeeze">SYNCING YAHOO...</span>
      {:else}
        <span class="badge badge-long">YAHOO LIVE ({blueprint.bars.length} BARS)</span>
      {/if}
    </div>

    <div style="display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
      <!-- Any-Ticker Live Yahoo Finance Search -->
      <form onsubmit={handleSymbolSubmit} style="display: flex; align-items: center; gap: 4px;">
        <input
          type="text"
          class="input-control"
          placeholder="Ticker (NVDA, PLTR, BTC-USD...)"
          bind:value={customSymbolInput}
          style="inline-size: 155px; font-size: 0.72rem;"
          aria-label="Search Any Yahoo Finance Ticker"
        />
        <button type="submit" class="action-btn">Load Yahoo</button>
      </form>

      <button
        type="button"
        class="action-btn signal-anchor-trigger"
        onclick={() => (showVerificationPopover = !showVerificationPopover)}
      >
        AI Blueprint Proof ({blueprint.aiConfidencePct.toFixed(1)}%)
      </button>
      <button
        type="button"
        class="action-btn action-btn-primary"
        onclick={() => terminalStore.openSignalAsPosition(blueprint)}
      >
        + Deploy to POSITIONS OPENED
      </button>
    </div>
  </header>

  <!-- Multi-Timeframe & Range Switcher Strip -->
  <div
    style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; padding: 6px 12px; background: var(--surface-1); border-block-end: 1px solid var(--border-subtle); font-size: 0.7rem;"
  >
    <div style="display: flex; align-items: center; gap: 4px; flex-wrap: wrap;">
      <span style="color: var(--text-secondary); font-weight: 700; margin-inline-end: 4px;">TF:</span>
      {#each TIMEFRAMES as tf (tf)}
        <button
          type="button"
          class="filter-pill"
          data-selected={terminalStore.activeTimeframe === tf}
          onclick={() => terminalStore.setTimeframe(tf)}
          style="padding: 2px 7px; font-size: 0.68rem;"
        >
          {tf}
        </button>
      {/each}
    </div>

    <div style="display: flex; align-items: center; gap: 4px; flex-wrap: wrap;">
      <span style="color: var(--text-secondary); font-weight: 700; margin-inline-end: 4px;">RANGE:</span>
      {#each RANGES as rg (rg)}
        <button
          type="button"
          class="filter-pill"
          data-selected={terminalStore.activeRange === rg}
          onclick={() => terminalStore.setRange(rg)}
          style="padding: 2px 7px; font-size: 0.68rem;"
        >
          {rg}
        </button>
      {/each}
      <button
        type="button"
        class="action-btn"
        onclick={resetZoomAndPan}
        style="padding: 2px 7px; font-size: 0.66rem;"
      >
        Reset Zoom
      </button>
    </div>
  </div>

  {#if showVerificationPopover}
    <div class="signal-verification-popover" role="dialog" aria-label="AI Signal Verification">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-block-end: 8px;">
        <strong style="font-size: 0.8rem; text-transform: uppercase;">
          Zero-Skew Raw Sequence Verification ({blueprint.symbol})
        </strong>
        <button type="button" class="action-btn" onclick={() => (showVerificationPopover = false)}>
          Close
        </button>
      </div>
      <p style="font-size: 0.74rem; color: var(--text-secondary); margin-block-end: 8px;">
        Computed directly by <code>raw-fingerprint-core</code> + <code>PrecedentKnnIndex</code> using
        real Yahoo Finance OHLCV bars, p85 MAE structural invalidation, p50/p80 MFE targets, and
        Black-Scholes OPRA Greeks.
      </p>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 0.73rem;">
        <div>Entry Zone: <strong class="tabular-num">{formatSubPennyPrice(blueprint.entryPrice)}</strong></div>
        <div>Structural Stop (p85 MAE): <strong class="tabular-num pnl-negative">{formatSubPennyPrice(blueprint.structuralStopPrice)}</strong></div>
        <div>Target 1 (p50 MFE): <strong class="tabular-num pnl-positive">{formatSubPennyPrice(blueprint.target1Price)}</strong></div>
        <div>Target 2 (p80 MFE): <strong class="tabular-num pnl-positive">{formatSubPennyPrice(blueprint.target2Price)}</strong></div>
        <div>Expected Payoff: <strong class="tabular-num">+{blueprint.expectedRMultiple.toFixed(2)}R</strong></div>
        <div>Hold Horizon: <strong class="tabular-num">{blueprint.predictedHoldWindow}</strong></div>
      </div>
    </div>
  {/if}

  <!-- Blueprint Telemetry Strip -->
  <div
    style="display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px; padding: 10px 12px; background: var(--surface-0); border-block-end: 1px solid var(--border-subtle); font-size: 0.73rem;"
  >
    <div>
      <div style="color: var(--text-secondary); font-size: 0.66rem;">AI ENTRY ZONE</div>
      <div class="tabular-num" style="font-weight: 700;">
        {formatSubPennyPrice(blueprint.entryPrice)}
      </div>
    </div>
    <div>
      <div style="color: var(--text-secondary); font-size: 0.66rem;">STRUCTURAL STOP (p85 MAE)</div>
      <div class="tabular-num pnl-negative" style="font-weight: 700;">
        {formatSubPennyPrice(blueprint.structuralStopPrice)}
      </div>
    </div>
    <div>
      <div style="color: var(--text-secondary); font-size: 0.66rem;">TARGET 1 (p50 MFE)</div>
      <div class="tabular-num pnl-positive" style="font-weight: 700;">
        {formatSubPennyPrice(blueprint.target1Price)}
      </div>
    </div>
    <div>
      <div style="color: var(--text-secondary); font-size: 0.66rem;">TARGET 2 (p80 MFE)</div>
      <div class="tabular-num pnl-positive" style="font-weight: 700;">
        {formatSubPennyPrice(blueprint.target2Price)} ({blueprint.expectedRMultiple.toFixed(2)}R)
      </div>
    </div>
    <div>
      <div style="color: var(--text-secondary); font-size: 0.66rem;">PREDICTED WINDOW</div>
      <div class="tabular-num" style="font-weight: 700; color: var(--signal-ai-core);">
        {blueprint.predictedHoldWindow}
      </div>
    </div>
  </div>

  <!-- Interactive Crosshair Hover Telemetry Bar -->
  {#if hoveredBar}
    <div
      class="tabular-num"
      style="display: flex; flex-wrap: wrap; gap: 12px; padding: 5px 12px; background: oklch(0.16 0.02 255); border-block-end: 1px solid var(--border-subtle); font-size: 0.69rem;"
    >
      <span>
        TIME: <strong>{new Date(hoveredBar.timestampMs).toISOString().replace('T', ' ').slice(0, 19)} UTC</strong>
      </span>
      <span>O: <strong>{formatSubPennyPrice(hoveredBar.open)}</strong></span>
      <span>H: <strong class="pnl-positive">{formatSubPennyPrice(hoveredBar.high)}</strong></span>
      <span>L: <strong class="pnl-negative">{formatSubPennyPrice(hoveredBar.low)}</strong></span>
      <span>C: <strong>{formatSubPennyPrice(hoveredBar.close)}</strong></span>
      <span>VOL: <strong>{hoveredBar.volume.toLocaleString('en-US')}</strong></span>
      <span class={hoveredBarLogReturnPct >= 0 ? 'pnl-positive' : 'pnl-negative'}>
        LOG RET: <strong>{hoveredBarLogReturnPct >= 0 ? '+' : ''}{hoveredBarLogReturnPct.toFixed(3)}%</strong>
      </span>
      <span style="color: var(--signal-ai-core);">
        ATTN WEIGHT: <strong>{(hoveredBar.attentionWeight * 100).toFixed(1)}%</strong>
      </span>
    </div>
  {/if}

  <!-- Pure Raw Price & Volume Canvas + AI Attention Heatmap Overlay + Trade Entry/Exit Markers -->
  <div style="padding: 10px 12px; background: var(--surface-0); position: relative;">
    <svg
      viewBox="0 0 {chartGeometry.width} {chartGeometry.height}"
      style="inline-size: 100%; block-size: auto; display: block; border-radius: var(--radius-sm); background: oklch(0.12 0.014 255); cursor: {isDragging ? 'grabbing' : 'crosshair'};"
      role="img"
      aria-label="Raw Yahoo Finance OHLCV Sequence and AI Structural Blueprint Overlay"
      onwheel={handleWheelZoom}
      onpointerdown={handlePointerDown}
      onpointermove={handlePointerMove}
      onpointerup={handlePointerUp}
      onpointerleave={() => {
        isDragging = false;
        hoveredBarIndex = null;
      }}
    >
      <!-- Horizontal Structural Blueprint Zones -->
      <line
        x1="0"
        y1={chartGeometry.t2Y}
        x2={chartGeometry.width}
        y2={chartGeometry.t2Y}
        stroke="oklch(0.76 0.19 152)"
        stroke-dasharray="5 4"
        stroke-width="1.5"
      />
      <text
        x="10"
        y={Math.max(14, chartGeometry.t2Y - 5)}
        fill="oklch(0.76 0.19 152)"
        font-size="10"
        font-family="monospace"
      >
        TARGET 2 (p80 MFE): {formatSubPennyPrice(blueprint.target2Price)} (+{blueprint.expectedRMultiple.toFixed(2)}R)
      </text>

      <line
        x1="0"
        y1={chartGeometry.t1Y}
        x2={chartGeometry.width}
        y2={chartGeometry.t1Y}
        stroke="oklch(0.76 0.19 152 / 0.75)"
        stroke-dasharray="3 3"
        stroke-width="1.2"
      />
      <text
        x="10"
        y={Math.max(14, chartGeometry.t1Y - 5)}
        fill="oklch(0.76 0.19 152)"
        font-size="10"
        font-family="monospace"
      >
        TARGET 1 (p50 MFE): {formatSubPennyPrice(blueprint.target1Price)}
      </text>

      <line
        x1="0"
        y1={chartGeometry.entryY}
        x2={chartGeometry.width}
        y2={chartGeometry.entryY}
        stroke="oklch(0.74 0.16 235)"
        stroke-width="1.5"
      />
      <text
        x="10"
        y={Math.max(14, chartGeometry.entryY - 5)}
        fill="oklch(0.74 0.16 235)"
        font-size="10"
        font-family="monospace"
      >
        ENTRY ZONE: {formatSubPennyPrice(blueprint.entryPrice)}
      </text>

      <line
        x1="0"
        y1={chartGeometry.stopY}
        x2={chartGeometry.width}
        y2={chartGeometry.stopY}
        stroke="oklch(0.68 0.22 22)"
        stroke-dasharray="4 3"
        stroke-width="1.5"
      />
      <text
        x="10"
        y={Math.min(chartGeometry.height - 54, chartGeometry.stopY + 12)}
        fill="oklch(0.68 0.22 22)"
        font-size="10"
        font-family="monospace"
      >
        AI STRUCTURAL STOP (p85 MAE): {formatSubPennyPrice(blueprint.structuralStopPrice)}
      </text>

      <!-- Raw Yahoo Finance Candles + Transformer Attention Glow + Raw Volume Profile -->
      {#each windowedBars as bar, idx (bar.timestampMs + '-' + idx)}
        {@const stepX = (chartGeometry.width - 140) / Math.max(1, windowedBars.length)}
        {@const xCenter = 60 + idx * stepX}
        {@const openY = chartGeometry.priceToY(bar.open)}
        {@const closeY = chartGeometry.priceToY(bar.close)}
        {@const highY = chartGeometry.priceToY(bar.high)}
        {@const lowY = chartGeometry.priceToY(bar.low)}
        {@const isBull = bar.close >= bar.open}
        {@const color = isBull ? 'oklch(0.76 0.19 152)' : 'oklch(0.68 0.22 22)'}
        {@const volBarH = (bar.volume / chartGeometry.maxVol) * 44}

        <!-- Per-Bar AI Attention Heatmap Highlight for Anomaly Candles -->
        {#if bar.attentionWeight > 0.72}
          <rect
            x={xCenter - stepX * 0.44}
            y={highY - 4}
            width={stepX * 0.88}
            height={Math.max(12, lowY - highY + 8)}
            fill="oklch(0.78 0.17 295 / {bar.attentionWeight * 0.22})"
            rx="2"
          />
        {/if}

        <!-- Raw High-Low Wick -->
        <line x1={xCenter} y1={highY} x2={xCenter} y2={lowY} stroke={color} stroke-width="1.4" />

        <!-- Raw Open-Close Body -->
        <rect
          x={xCenter - stepX * 0.32}
          y={Math.min(openY, closeY)}
          width={Math.max(2.5, stepX * 0.64)}
          height={Math.max(2, Math.abs(closeY - openY))}
          fill={color}
          rx="1"
        />

        <!-- Raw Volume Bar -->
        <rect
          x={xCenter - stepX * 0.3}
          y={chartGeometry.height - 6 - volBarH}
          width={Math.max(2.5, stepX * 0.6)}
          height={volBarH}
          fill={bar.attentionWeight > 0.72 ? 'oklch(0.78 0.17 295 / 0.78)' : color}
          opacity="0.55"
        />
      {/each}

      <!-- Historical Closed Trade Entry & Exit Markers on Real Yahoo Chart -->
      {#each symbolClosedTrades.slice(0, 4) as trade, tIdx (trade.tradeId)}
        {@const entryMarkerX = Math.max(85, chartGeometry.width - 280 - tIdx * 95)}
        {@const exitMarkerX = Math.min(chartGeometry.width - 95, entryMarkerX + 75)}
        {@const entryMarkerY = chartGeometry.priceToY(trade.entryPrice)}
        {@const exitMarkerY = chartGeometry.priceToY(trade.exitPrice)}
        {@const tradeColor = trade.isWinner ? 'oklch(0.80 0.20 152)' : 'oklch(0.68 0.22 22)'}

        <line
          x1={entryMarkerX}
          y1={entryMarkerY}
          x2={exitMarkerX}
          y2={exitMarkerY}
          stroke={tradeColor}
          stroke-dasharray="2 2"
          stroke-width="1.5"
        />
        <circle cx={entryMarkerX} cy={entryMarkerY} r="4.5" fill="oklch(0.74 0.16 235)" />
        <text
          x={entryMarkerX - 24}
          y={Math.max(14, entryMarkerY - 7)}
          fill="oklch(0.85 0.12 235)"
          font-size="9"
          font-family="monospace"
        >
          IN {formatSubPennyPrice(trade.entryPrice)}
        </text>

        <circle cx={exitMarkerX} cy={exitMarkerY} r="5" fill={tradeColor} />
        <text
          x={exitMarkerX - 30}
          y={Math.max(14, exitMarkerY - 8)}
          fill={tradeColor}
          font-size="9"
          font-weight="bold"
          font-family="monospace"
        >
          {trade.outcomeBadge} ({formatSignedUsd(trade.combinedTotalPnlUsd)} | {trade.realizedRMultiple >= 0 ? '+' : ''}{trade.realizedRMultiple.toFixed(1)}R)
        </text>
      {/each}

      <!-- Interactive Crosshair Lines -->
      {#if hoveredBarIndex !== null}
        <line
          x1={hoverMouseX}
          y1="0"
          x2={hoverMouseX}
          y2={chartGeometry.height}
          stroke="oklch(0.85 0.05 250 / 0.45)"
          stroke-dasharray="3 3"
          stroke-width="1"
        />
        <line
          x1="0"
          y1={hoverMouseY}
          x2={chartGeometry.width}
          y2={hoverMouseY}
          stroke="oklch(0.85 0.05 250 / 0.45)"
          stroke-dasharray="3 3"
          stroke-width="1"
        />
      {/if}

      <!-- Predicted Future Expansion Corridor -->
      <polygon
        points="{chartGeometry.width - 80},{chartGeometry.entryY} {chartGeometry.width - 8},{chartGeometry.t2Y} {chartGeometry.width - 8},{chartGeometry.t1Y}"
        fill="oklch(0.76 0.19 152 / 0.18)"
        stroke="oklch(0.76 0.19 152 / 0.65)"
        stroke-width="1"
      />
    </svg>
  </div>
</section>
