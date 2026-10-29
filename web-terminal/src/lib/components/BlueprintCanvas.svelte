<script lang="ts">
  import {
    terminalStore,
    formatSubPennyPrice,
    type ScannerSignalCandidate
  } from '$lib/state/terminal.svelte';

  let showVerificationPopover = $state(false);

  const blueprint = $derived<ScannerSignalCandidate>(terminalStore.activeBlueprint);

  const chartGeometry = $derived.by(() => {
    const bars = blueprint.bars;
    const allPrices = [
      blueprint.entryPrice,
      blueprint.structuralStopPrice,
      blueprint.target1Price,
      blueprint.target2Price,
      ...bars.flatMap((b) => [b.high, b.low])
    ];
    const minP = Math.min(...allPrices) * 0.996;
    const maxP = Math.max(...allPrices) * 1.004;
    const rangeP = Math.max(maxP - minP, 0.00001);
    const maxVol = Math.max(1, ...bars.map((b) => b.volume));

    const width = 760;
    const height = 340;
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
      entryY: priceToY(blueprint.entryPrice),
      stopY: priceToY(blueprint.structuralStopPrice),
      t1Y: priceToY(blueprint.target1Price),
      t2Y: priceToY(blueprint.target2Price),
      priceToY
    };
  });
</script>

<section class="glass-panel panel-blueprint-canvas" aria-label="Raw Price & AI Trade Blueprint Canvas">
  <header class="panel-header">
    <div class="panel-title">
      <span class="badge {blueprint.direction === 'LONG' ? 'badge-long' : 'badge-short'}">
        {blueprint.direction}
      </span>
      <span>{blueprint.symbol}</span>
      <span class="tabular-num" style="color: var(--signal-ai-core);">
        {formatSubPennyPrice(blueprint.livePrice)}
      </span>
      <span class="badge badge-ai-core">{blueprint.brainId}</span>
      <span class="badge badge-gamma">{blueprint.exchangeIndex}</span>
    </div>

    <div style="display: flex; align-items: center; gap: 8px;">
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
        Computed directly by <code>raw-fingerprint-core</code> using 100% scale-independent log/relative
        price-volume physics, OPRA Gamma/Vanna flow, and FINRA short-squeeze pressure.
      </p>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 0.73rem;">
        <div>Entry Zone: <strong class="tabular-num">{formatSubPennyPrice(blueprint.entryPrice)}</strong></div>
        <div>Structural Stop: <strong class="tabular-num pnl-negative">{formatSubPennyPrice(blueprint.structuralStopPrice)}</strong></div>
        <div>Target 1 (Scale): <strong class="tabular-num pnl-positive">{formatSubPennyPrice(blueprint.target1Price)}</strong></div>
        <div>Target 2 (Full): <strong class="tabular-num pnl-positive">{formatSubPennyPrice(blueprint.target2Price)}</strong></div>
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
      <div style="color: var(--text-secondary); font-size: 0.66rem;">STRUCTURAL STOP</div>
      <div class="tabular-num pnl-negative" style="font-weight: 700;">
        {formatSubPennyPrice(blueprint.structuralStopPrice)}
      </div>
    </div>
    <div>
      <div style="color: var(--text-secondary); font-size: 0.66rem;">TARGET 1 (PARTIAL)</div>
      <div class="tabular-num pnl-positive" style="font-weight: 700;">
        {formatSubPennyPrice(blueprint.target1Price)}
      </div>
    </div>
    <div>
      <div style="color: var(--text-secondary); font-size: 0.66rem;">TARGET 2 (EXPANSION)</div>
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

  <!-- Pure Raw Price & Volume Canvas + AI Attention Heatmap Overlay (Zero Lagging Indicators) -->
  <div style="padding: 10px 12px; background: var(--surface-0);">
    <svg
      viewBox="0 0 {chartGeometry.width} {chartGeometry.height}"
      style="inline-size: 100%; block-size: auto; display: block; border-radius: var(--radius-sm); background: oklch(0.12 0.014 255);"
      role="img"
      aria-label="Raw OHLCV Sequence and AI Structural Blueprint Overlay"
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
        TARGET 2: {formatSubPennyPrice(blueprint.target2Price)} (+{blueprint.expectedRMultiple.toFixed(2)}R)
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
        TARGET 1: {formatSubPennyPrice(blueprint.target1Price)}
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
        AI STRUCTURAL STOP: {formatSubPennyPrice(blueprint.structuralStopPrice)}
      </text>

      <!-- Raw Candles + Transformer Attention Glow + Raw Volume Profile -->
      {#each blueprint.bars as bar, idx (bar.timestampMs)}
        {@const stepX = (chartGeometry.width - 140) / Math.max(1, blueprint.bars.length)}
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
          width={Math.max(3, stepX * 0.64)}
          height={Math.max(2, Math.abs(closeY - openY))}
          fill={color}
          rx="1"
        />

        <!-- Raw Volume Bar -->
        <rect
          x={xCenter - stepX * 0.3}
          y={chartGeometry.height - 6 - volBarH}
          width={Math.max(3, stepX * 0.6)}
          height={volBarH}
          fill={bar.attentionWeight > 0.72 ? 'oklch(0.78 0.17 295 / 0.78)' : color}
          opacity="0.55"
        />
      {/each}

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

