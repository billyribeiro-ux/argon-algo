<script lang="ts">
  import { terminalStore } from '$lib/state/terminal.svelte';
  import BrainAutopsyLab from '$lib/components/BrainAutopsyLab.svelte';

  const internals = $derived(terminalStore.macroInternals);
  const providers = $derived(terminalStore.providers);
  const REPLAY_SPEEDS: Array<'1x' | '5x' | '10x' | '50x' | 'MAX'> = ['1x', '5x', '10x', '50x', 'MAX'];

  let scanningYahoo = $state(false);
  let swappingProviderId = $state<string | null>(null);

  async function handleYahooScan(): Promise<void> {
    scanningYahoo = true;
    await terminalStore.triggerYahooUniverseScan();
    scanningYahoo = false;
  }

  async function handleHotSwap(providerId: string): Promise<void> {
    swappingProviderId = providerId;
    await terminalStore.hotSwapProvider(providerId);
    swappingProviderId = null;
  }
</script>

<div class="command-bento-grid">
  <section class="glass-panel" style="grid-column: span 6;">
    <header class="panel-header" style="flex-wrap: wrap; gap: 8px;">
      <div class="panel-title">
        <span>Plug-and-Play Market Data Bus &amp; Zero-Downtime Failover Matrix</span>
      </div>

      <div style="display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
        <div style="display: flex; align-items: center; gap: 3px;">
          <span style="font-size: 0.66rem; color: var(--text-secondary);">REPLAY:</span>
          {#each REPLAY_SPEEDS as spd (spd)}
            <button
              type="button"
              class="filter-pill"
              data-selected={terminalStore.replaySpeed === spd}
              onclick={() => (terminalStore.replaySpeed = spd)}
              style="padding: 2px 6px; font-size: 0.65rem;"
            >
              {spd}
            </button>
          {/each}
        </div>

        <button
          type="button"
          class="action-btn action-btn-primary"
          onclick={handleYahooScan}
          disabled={scanningYahoo}
        >
          {scanningYahoo
            ? 'Scanning Live Yahoo Finance...'
            : '[ Scan Real Yahoo Finance Universe Now ]'}
        </button>
      </div>
    </header>

    <div class="data-table-container">
      <table class="telemetry-table">
        <thead>
          <tr>
            <th>Provider Adapter</th>
            <th>Stream Category</th>
            <th>Failover Tier</th>
            <th>Health Status</th>
            <th data-numeric="true">Wire Latency</th>
            <th data-numeric="true">Throughput</th>
            <th>Hot-Swap Control</th>
          </tr>
        </thead>
        <tbody>
          {#each providers as p (p.providerId)}
            <tr>
              <td>
                <strong>{p.providerId}</strong>
                <div style="font-size: 0.64rem; color: var(--text-secondary);">{p.displayName}</div>
              </td>
              <td><span class="badge badge-ai-core">{p.category}</span></td>
              <td><span class="badge badge-gamma">{p.role}</span></td>
              <td>
                <span class="badge {p.healthy ? 'badge-long' : 'badge-short'}">
                  {p.healthy ? 'ONLINE (ACTIVE)' : 'FAILOVER STANDBY'}
                </span>
              </td>
              <td data-numeric="true">{p.latencyMs.toFixed(2)} ms</td>
              <td data-numeric="true">{p.messagesPerSec.toLocaleString('en-US')} /s</td>
              <td>
                {#if p.role === 'PRIMARY'}
                  <span class="badge badge-long">ACTIVE PRIMARY</span>
                {:else}
                  <button
                    type="button"
                    class="action-btn"
                    onclick={() => handleHotSwap(p.providerId)}
                    disabled={swappingProviderId === p.providerId}
                  >
                    {swappingProviderId === p.providerId ? 'Swapping...' : 'Hot-Swap Primary'}
                  </button>
                {/if}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </section>

  <section class="glass-panel" style="grid-column: span 6;">
    <header class="panel-header">
      <div class="panel-title">
        <span>Stream 4: Live Yahoo Finance Macro Tide &amp; Cross-Asset Internals</span>
      </div>
    </header>
    <div class="scorecard-kpi-ribbon">
      <div class="kpi-card">
        <span class="kpi-label">NYSE $TICK Impulse</span>
        <span class="kpi-metric-value {internals.nyseTick >= 0 ? 'pnl-positive' : 'pnl-negative'}">
          {internals.nyseTick >= 0 ? '+' : ''}{internals.nyseTick}
        </span>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">Advance/Decline ($ADD)</span>
        <span class="kpi-metric-value {internals.nyseAdd >= 0 ? 'pnl-positive' : 'pnl-negative'}">
          {internals.nyseAdd >= 0 ? '+' : ''}{internals.nyseAdd}
        </span>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">Up/Down Vol ($VOLD)</span>
        <span class="kpi-metric-value pnl-positive">{internals.voldRatio.toFixed(2)}x</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">Semiconductor Tide (SOXX/SMH)</span>
        <span class="kpi-metric-value {internals.smhPct >= 0 ? 'pnl-positive' : 'pnl-negative'}">
          {internals.soxxPct >= 0 ? '+' : ''}{internals.soxxPct.toFixed(2)}% /
          {internals.smhPct >= 0 ? '+' : ''}{internals.smhPct.toFixed(2)}%
        </span>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">VIX Spot / Front Month</span>
        <span class="kpi-metric-value">
          {internals.vixSpot.toFixed(2)} / {internals.vixFrontMonth.toFixed(2)}
        </span>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">VIX Term Slope (VX1/VX2)</span>
        <span class="kpi-metric-value">{internals.vixTermSlope.toFixed(3)}</span>
      </div>
    </div>
  </section>

  <!-- 3-Speed Continuous Self-Learning & Nightly Autopsy Lab -->
  <BrainAutopsyLab />
</div>
