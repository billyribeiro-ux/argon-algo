<script lang="ts">
  import { terminalStore } from '$lib/state/terminal.svelte';

  const internals = $derived(terminalStore.macroInternals);
  const providers = $derived(terminalStore.providers);
</script>

<div class="command-bento-grid">
  <section class="glass-panel" style="grid-column: span 6;">
    <header class="panel-header">
      <div class="panel-title">
        <span>Plug-and-Play Market Data Bus &amp; Zero-Downtime Failover Matrix</span>
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
            <th data-numeric="true">Throughput (msg/s)</th>
          </tr>
        </thead>
        <tbody>
          {#each providers as p (p.providerId)}
            <tr>
              <td><strong>{p.providerId}</strong></td>
              <td><span class="badge badge-ai-core">{p.category}</span></td>
              <td><span class="badge badge-gamma">{p.role}</span></td>
              <td>
                <span class="badge {p.healthy ? 'badge-long' : 'badge-short'}">
                  {p.healthy ? 'ONLINE (ACTIVE)' : 'FAILOVER STANDBY'}
                </span>
              </td>
              <td data-numeric="true">{p.latencyMs.toFixed(1)} ms</td>
              <td data-numeric="true">{p.messagesPerSec.toLocaleString('en-US')} /s</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </section>

  <section class="glass-panel" style="grid-column: span 6;">
    <header class="panel-header">
      <div class="panel-title">
        <span>Stream 4: Live Macro Tide &amp; Cross-Asset Internals</span>
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
        <span class="kpi-metric-value pnl-positive">+{internals.nyseAdd}</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">Up/Down Vol ($VOLD)</span>
        <span class="kpi-metric-value pnl-positive">{internals.voldRatio.toFixed(2)}x</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">Semiconductor Tide (SOXX/SMH)</span>
        <span class="kpi-metric-value pnl-positive">
          +{internals.soxxPct.toFixed(2)}% / +{internals.smhPct.toFixed(2)}%
        </span>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">Credit Spread (HYG/LQD)</span>
        <span class="kpi-metric-value pnl-positive">{internals.hygLqdSpreadBps.toFixed(1)} bps</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">VIX Term Slope (VX1/VX2)</span>
        <span class="kpi-metric-value">{internals.vixTermSlope.toFixed(3)}</span>
      </div>
    </div>
  </section>
</div>

