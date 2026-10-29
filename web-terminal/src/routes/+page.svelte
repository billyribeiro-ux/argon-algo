<script lang="ts">
  import {
    terminalStore,
    formatSubPennyPrice
  } from '$lib/state/terminal.svelte';
  import BlueprintCanvas from '$lib/components/BlueprintCanvas.svelte';
  import RawEvidencePanel from '$lib/components/RawEvidencePanel.svelte';
  import PositionsOpenedPanel from '$lib/components/PositionsOpenedPanel.svelte';
  import ClosedTradeLedgerPanel from '$lib/components/ClosedTradeLedgerPanel.svelte';

  const candidates = $derived(terminalStore.filteredScannerCandidates);
</script>

<div class="command-bento-grid">
  <!-- Left Panel: Two-Stage 11,000+ Ticker Universe Scanner -->
  <section class="glass-panel panel-scanner" aria-label="11,000+ Ticker Universe Scanner">
    <header class="panel-header">
      <div class="panel-title">
        <span>11,000+ Universe Scanner</span>
        <span class="badge badge-ai-core">{candidates.length} TOP SIGNALS</span>
      </div>
      <span style="font-size: 0.68rem; color: var(--text-secondary);" class="tabular-num">
        Stage-1: 11,420 → Stage-2: 150
      </span>
    </header>

    <div class="data-table-container">
      <table class="telemetry-table">
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Brain</th>
            <th>Dir</th>
            <th data-numeric="true">Live Price</th>
            <th data-numeric="true">Target 2 (R)</th>
            <th data-numeric="true" class="scanner-compact-hide">AI Conf</th>
            <th class="scanner-compact-hide">Primary Driver</th>
          </tr>
        </thead>
        <tbody>
          {#each candidates as item (item.symbol + item.brainId)}
            <tr
              data-selected={terminalStore.selectedTicker === item.symbol}
              onclick={() => terminalStore.selectTicker(item.symbol)}
              style="cursor: pointer;"
            >
              <td>
                <strong>{item.symbol}</strong>
                <div style="font-size: 0.64rem; color: var(--text-secondary);">
                  {item.exchangeIndex}
                </div>
              </td>
              <td><span class="badge badge-ai-core">{item.brainId}</span></td>
              <td>
                <span class="badge {item.direction === 'LONG' ? 'badge-long' : 'badge-short'}">
                  {item.direction}
                </span>
              </td>
              <td data-numeric="true" style="font-weight: 700;">
                {formatSubPennyPrice(item.livePrice)}
              </td>
              <td data-numeric="true" class="pnl-positive">
                {formatSubPennyPrice(item.target2Price)} (+{item.expectedRMultiple.toFixed(1)}R)
              </td>
              <td data-numeric="true" class="scanner-compact-hide">
                {item.aiConfidencePct.toFixed(1)}%
              </td>
              <td class="scanner-compact-hide">
                <span class="badge badge-gamma">{item.primaryDriver}</span>
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </section>

  <!-- Center Panel: Raw Price & Volume Canvas + AI Trade Blueprint Overlay -->
  <BlueprintCanvas />

  <!-- Right Panel: 4-Stream Raw Evidence Hub, Options Contract Matcher & Precedent k-NN -->
  <RawEvidencePanel />

  <!-- Full-Width Panel 1: POSITIONS OPENED Real-Time Command Tracker -->
  <PositionsOpenedPanel />

  <!-- Full-Width Panel 2: CLOSED TRADE LEDGER, 6-Dimension Hyper-Filter & 1-Click CSV/XLSX Exporters -->
  <ClosedTradeLedgerPanel />
</div>

