<script lang="ts">
  import '../styles/reset.css';
  import '../styles/tokens.css';
  import '../styles/layout.css';
  import '../styles/components.css';
  import '../styles/telemetry.css';

  import {
    terminalStore,
    formatSignedUsd,
    type BrainId,
    type LinkChannelColor,
    type WorkspaceLayoutMode
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

  const LAYOUT_MODES: Array<{ id: WorkspaceLayoutMode; label: string }> = [
    { id: 'SINGLE', label: 'Full Bento' },
    { id: 'SPLIT_50_50', label: '50/50 Split' },
    { id: 'QUAD_2X2', label: '2x2 Quad' }
  ];

  const LINK_CHANNELS: Array<{ id: LinkChannelColor; label: string; colorCss: string }> = [
    { id: 'RED', label: 'RED', colorCss: 'oklch(0.68 0.22 22)' },
    { id: 'BLUE', label: 'BLUE', colorCss: 'oklch(0.72 0.17 240)' },
    { id: 'GREEN', label: 'GREEN', colorCss: 'oklch(0.76 0.19 152)' },
    { id: 'GOLD', label: 'GOLD', colorCss: 'oklch(0.82 0.17 85)' },
    { id: 'UNLINKED', label: 'UNLINKED', colorCss: 'oklch(0.55 0.02 255)' }
  ];

  const internals = $derived(terminalStore.macroInternals);
  const openSummary = $derived(terminalStore.openPositionsSummary);
  const kpis = $derived(terminalStore.scorecardKpis);
  const primaryProvider = $derived(
    terminalStore.providers.find((p) => p.role === 'PRIMARY')?.providerId ?? 'yahoo_finance_live'
  );
</script>

<div class="terminal-shell" data-workspace-layout={terminalStore.workspaceLayout}>
  <!-- Top Omni-Command Bar -->
  <header class="top-omni-bar">
    <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
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

    <!-- Universe Price Tier, Layout Switcher, Color-Channel Link & Multi-Monitor Detach Controls -->
    <div style="display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
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

      <!-- Non-Destructive Capital Sizing Mode Toggle ([ Fixed $10K Baseline ] vs [ Fractional-Kelly Sizing ]) -->
      <div
        style="display: flex; align-items: center; gap: 3px; padding: 2px 6px; background: var(--surface-0); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);"
        role="group"
        aria-label="Capital Sizing Mode"
      >
        <button
          type="button"
          class="filter-pill"
          data-selected={terminalStore.sizingMode === 'FIXED_10K_BASELINE'}
          onclick={() => terminalStore.setSizingMode('FIXED_10K_BASELINE')}
          style="padding: 2px 6px; font-size: 0.66rem; font-weight: 700;"
          title="Phase 11.1 Non-Destructive Fixed $10K Notional Baseline"
        >
          Fixed $10K Baseline
        </button>
        <button
          type="button"
          class="filter-pill"
          data-selected={terminalStore.sizingMode === 'FRACTIONAL_KELLY'}
          onclick={() => terminalStore.setSizingMode('FRACTIONAL_KELLY')}
          style="padding: 2px 6px; font-size: 0.66rem; font-weight: 700;"
          title="Phase 12 Causal Fractional-Kelly Risk Sizing (0.35x Kelly)"
        >
          Fractional-Kelly Sizing
        </button>
      </div>

      <!-- Side-by-Side Workspace Layout Mode Selector -->
      <div
        style="display: flex; align-items: center; gap: 3px; padding: 2px 6px; background: var(--surface-0); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);"
        role="group"
        aria-label="Workspace Layout Mode"
      >
        {#each LAYOUT_MODES as lm (lm.id)}
          <button
            type="button"
            class="filter-pill"
            data-selected={terminalStore.workspaceLayout === lm.id}
            onclick={() => terminalStore.setWorkspaceLayout(lm.id)}
            style="padding: 2px 6px; font-size: 0.66rem;"
          >
            {lm.label}
          </button>
        {/each}
      </div>

      <!-- Color-Channel Link Badge Selector -->
      <div
        style="display: flex; align-items: center; gap: 4px; padding: 2px 6px; background: var(--surface-0); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);"
        role="group"
        aria-label="Window Color Link Channel"
      >
        <span style="font-size: 0.64rem; color: var(--text-secondary); font-weight: 700;">CH:</span>
        {#each LINK_CHANNELS as ch (ch.id)}
          <button
            type="button"
            onclick={() => terminalStore.setLinkChannel(ch.id, true)}
            title="Link Window to {ch.label} Channel"
            style="inline-size: 14px; block-size: 14px; border-radius: 50%; background: {ch.colorCss}; border: {terminalStore.activeLinkChannel === ch.id ? '2px solid white' : '1px solid transparent'}; cursor: pointer;"
            aria-label="Channel {ch.label}"
          ></button>
        {/each}
      </div>

      <button
        type="button"
        class="action-btn"
        onclick={() =>
          terminalStore.syncBus.detachPanelWindow(
            '/blueprint',
            'monitor_2_blueprint',
            1,
            terminalStore.activeLinkChannel
          )}
      >
        Detach Blueprint
      </button>
      <button
        type="button"
        class="action-btn"
        onclick={() =>
          terminalStore.syncBus.detachPanelWindow(
            '/positions-and-ledger',
            'monitor_3_ledger',
            2,
            terminalStore.activeLinkChannel
          )}
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

      <!-- Multi-Monitor Web Audio & Speech Synthesis Controls -->
      <div
        style="display: flex; align-items: center; gap: 4px; padding: 2px 6px; background: var(--surface-0); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);"
        role="group"
        aria-label="Execution Audio Alert Controls"
      >
        <button
          type="button"
          class="filter-pill"
          data-selected={terminalStore.audioMode !== 'MUTED'}
          onclick={() => terminalStore.cycleAudioMode()}
          title="Cycle Execution Audio Alert Mode (Chimes + Voice / Chimes Only / Muted)"
          style="padding: 2px 7px; font-size: 0.66rem; font-weight: 700;"
        >
          {#if terminalStore.audioMode === 'VOICE_AND_CHIMES'}
            🔊 Audio: Chimes + Voice
          {:else if terminalStore.audioMode === 'CHIMES_ONLY'}
            🔔 Audio: Chimes Only
          {:else}
            🔇 Audio: Muted
          {/if}
        </button>
        <button
          type="button"
          class="action-btn"
          onclick={() => terminalStore.testExecutionAudio()}
          style="padding: 2px 6px; font-size: 0.65rem;"
          title="Play Web Audio API Chime & Speech Synthesis Verification"
        >
          Test Sound
        </button>
      </div>
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
        $ADD:
        <strong class={internals.nyseAdd >= 0 ? 'pnl-positive' : 'pnl-negative'}>
          {internals.nyseAdd >= 0 ? '+' : ''}{internals.nyseAdd}
        </strong>
      </span>
      <span class="tabular-num">
        $VOLD: <strong class="pnl-positive">{internals.voldRatio.toFixed(2)}x</strong>
      </span>
      <span class="tabular-num">
        VIX Spot: <strong>{internals.vixSpot.toFixed(2)}</strong>
      </span>
      <span class="tabular-num">
        SOXX / SMH:
        <strong class={internals.smhPct >= 0 ? 'pnl-positive' : 'pnl-negative'}>
          {internals.soxxPct >= 0 ? '+' : ''}{internals.soxxPct.toFixed(2)}% /
          {internals.smhPct >= 0 ? '+' : ''}{internals.smhPct.toFixed(2)}%
        </strong>
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
      <a href="/macro-and-adapters" class="badge {terminalStore.backendConnected ? 'badge-long' : 'badge-squeeze'}">
        BUS: {primaryProvider.toUpperCase()} ({terminalStore.backendConnected ? 'LIVE WS' : 'RECONNECTING'})
      </a>
    </div>
  </div>

  <!-- Main Workspace Viewport -->
  <main>
    {@render children()}
  </main>
</div>
