<script lang="ts">
  import '../styles/reset.css';
  import '../styles/tokens.css';
  import '../styles/layout.css';
  import '../styles/components.css';
  import '../styles/telemetry.css';

  import {
    terminalStore,
    formatSignedUsd,
    type BrainId
  } from '$lib/state/terminal.svelte';
  import type { Snippet } from 'svelte';

  let { children }: { children: Snippet } = $props();

  const BRAIN_TABS: Array<{ id: BrainId; label: string; horizon: string }> = [
    { id: 'ALL_5_BRAINS', label: 'ALL 5 BRAINS', horizon: '1m - 12m' },
    { id: 'BRAIN-01', label: 'BRAIN-01 SCALP', horizon: '1m - 60m' },
    { id: 'BRAIN-02', label: 'BRAIN-02 DAY TRADE', horizon: '1h - 6.5h' },
    { id: 'BRAIN-03', label: 'BRAIN-03 SWING', horizon: '1d - 10d' },
    { id: 'BRAIN-04', label: 'BRAIN-04 MULTI-WEEK', horizon: '2w - 8w' },
    { id: 'BRAIN-05', label: 'BRAIN-05 WEALTH', horizon: '2m - 12m+' }
  ];

  const internals = $derived(terminalStore.macroInternals);
  const openSummary = $derived(terminalStore.openPositionsSummary);
  const kpis = $derived(terminalStore.scorecardKpis);
</script>

<div class="terminal-shell">
  <!-- Top Omni-Command Bar -->
  <header class="top-omni-bar">
    <div style="display: flex; align-items: center; gap: 12px;">
      <a href="/" style="display: flex; align-items: center; gap: 8px; font-weight: 800; font-size: 0.88rem; letter-spacing: 0.04em;">
        <span class="badge badge-ai-core">ARGON-4</span>
        <span>QUANT INTELLIGENCE TERMINAL</span>
      </a>

      <!-- 5-Brain Horizon Selector -->
      <nav class="brain-selector-bar" aria-label="Multi-Brain Horizon Selector">
        {#each BRAIN_TABS as b (b.id)}
          <button
            type="button"
            class="brain-pill"
            data-active={terminalStore.activeBrain === b.id}
            onclick={() => terminalStore.selectBrain(b.id)}
          >
            <span>{b.label}</span>
            <span class="tabular-num" style="font-size: 0.64rem; opacity: 0.78;">({b.horizon})</span>
          </button>
        {/each}
      </nav>
    </div>

    <!-- Universe Price Tier & Multi-Monitor Workspace Detach Controls -->
    <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
      <select class="input-control" bind:value={terminalStore.universeTierMode} aria-label="Universe Price Tier">
        <option value="ALL">UNIVERSE: ALL ($0.00001 - $10,000+)</option>
        <option value="SUB_PENNY_ONLY">SUB-PENNY ($0.00001 - $5.00)</option>
        <option value="INSTITUTIONAL_ONLY">INSTITUTIONAL ($5.00+)</option>
      </select>

      <select class="input-control" bind:value={terminalStore.activeAssetClass} aria-label="Asset Class Mode">
        <option value="ALL">STOCKS + OPTIONS</option>
        <option value="OPTIONS">OPTIONS MATCHED</option>
        <option value="STOCKS">EQUITY ONLY</option>
      </select>

      <button
        type="button"
        class="action-btn"
        onclick={() => terminalStore.syncBus.detachPanelWindow('/blueprint', 'monitor_2_blueprint', 1)}
      >
        Detach Blueprint
      </button>
      <button
        type="button"
        class="action-btn"
        onclick={() => terminalStore.syncBus.detachPanelWindow('/positions-and-ledger', 'monitor_3_ledger', 2)}
      >
        Detach Ledger
      </button>
      <button
        type="button"
        class="action-btn action-btn-export"
        onclick={() => terminalStore.syncBus.launchMultiMonitorPreset()}
      >
        Launch 4-Monitor Preset
      </button>
    </div>
  </header>

  <!-- Live Macro Tide & Plug-and-Play Bus Status Strip -->
  <div class="macro-tide-strip" role="status" aria-label="Live Macro Tide and Data Bus Telemetry">
    <div style="display: flex; align-items: center; gap: 14px; flex-wrap: wrap;">
      <span class="tabular-num">
        NYSE $TICK:
        <strong class={internals.nyseTick >= 0 ? 'pnl-positive' : 'pnl-negative'}>
          {internals.nyseTick >= 0 ? '+' : ''}{internals.nyseTick}
        </strong>
      </span>
      <span class="tabular-num">
        $ADD: <strong class="pnl-positive">+{internals.nyseAdd}</strong>
      </span>
      <span class="tabular-num">
        $VOLD: <strong class="pnl-positive">{internals.voldRatio.toFixed(2)}x</strong>
      </span>
      <span class="tabular-num">
        SOXX / SMH:
        <strong class="pnl-positive">+{internals.soxxPct.toFixed(2)}% / +{internals.smhPct.toFixed(2)}%</strong>
      </span>
      <span class="tabular-num">
        VIX Slope: <strong>{internals.vixTermSlope.toFixed(3)}</strong>
      </span>
    </div>

    <div style="display: flex; align-items: center; gap: 14px; flex-wrap: wrap;">
      <span class="tabular-num">
        OPEN P&amp;L ({openSummary.count}):
        <strong class={openSummary.combinedUnrealizedPnlUsd >= 0 ? 'pnl-positive' : 'pnl-negative'}>
          {formatSignedUsd(openSummary.combinedUnrealizedPnlUsd)}
        </strong>
      </span>
      <span class="tabular-num">
        CLOSED REALIZED ({kpis.totalWins}W/{kpis.totalLosses}L):
        <strong class={kpis.netCombinedPnlUsd >= 0 ? 'pnl-positive' : 'pnl-negative'}>
          {formatSignedUsd(kpis.netCombinedPnlUsd)}
        </strong>
      </span>
      <a href="/macro-and-adapters" class="badge badge-long">
        BUS: DATABENTO + OPRA + FINRA (0.8ms)
      </a>
    </div>
  </div>

  <!-- Main Workspace Viewport -->
  <main>
    {@render children()}
  </main>
</div>

