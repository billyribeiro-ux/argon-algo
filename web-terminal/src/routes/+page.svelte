<script lang="ts">
  import {
    terminalStore,
    formatSubPennyPrice,
    formatSignedUsd
  } from '$lib/state/terminal.svelte';
  import BlueprintCanvas from '$lib/components/BlueprintCanvas.svelte';
  import RawEvidencePanel from '$lib/components/RawEvidencePanel.svelte';
  import PositionsOpenedPanel from '$lib/components/PositionsOpenedPanel.svelte';
  import ClosedTradeLedgerPanel from '$lib/components/ClosedTradeLedgerPanel.svelte';
  import BrainAutopsyLab from '$lib/components/BrainAutopsyLab.svelte';

  const candidates = $derived(terminalStore.filteredScannerCandidates);
  let scanningYahoo = $state(false);

  async function handleYahooScan(): Promise<void> {
    scanningYahoo = true;
    await terminalStore.triggerYahooUniverseScan();
    scanningYahoo = false;
  }
</script>

{#if terminalStore.workspaceLayout === 'SPLIT_50_50'}
  <!-- 50/50 DUAL COMPARISON WORKSPACE (LEFT = RED CHANNEL, RIGHT = BLUE CHANNEL) -->
  <section
    aria-label="50/50 Dual Comparison Workspace"
    style="display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; padding: 10px 12px 0 12px;"
  >
    {#each terminalStore.splitPanes as pane (pane.paneId)}
      {@const paneClosed = terminalStore.closedLedger.filter(
        (t) => t.brainId === pane.brainId || t.symbol.toUpperCase() === pane.symbol.toUpperCase()
      )}
      {@const paneOpen = terminalStore.openPositions.filter(
        (p) => p.brainId === pane.brainId || p.symbol.toUpperCase() === pane.symbol.toUpperCase()
      )}
      {@const paneNetPnl = paneClosed.reduce((acc, r) => acc + r.combinedTotalPnlUsd, 0)}
      <div style="display: flex; flex-direction: column; gap: 8px; min-inline-size: 0;">
        <BlueprintCanvas
          paneId={pane.paneId}
          paneTitle={pane.title}
          paneLinkChannel={pane.linkChannel}
          paneBrainId={pane.brainId}
          paneTimeframe={pane.timeframe}
          paneRange={pane.range}
          paneLoading={pane.loading}
          customBlueprint={pane.blueprint}
        />

        <!-- Independent Comparison Ledger Slice for this Pane -->
        <section class="glass-panel" aria-label="{pane.title} Filtered Ledger Slice">
          <header class="panel-header">
            <div class="panel-title">
              <span>{pane.title} — {pane.brainId} / {pane.symbol} LEDGER SLICE</span>
              <span class="badge badge-ai-core">{paneOpen.length} OPEN</span>
              <span class="badge badge-gamma">{paneClosed.length} CLOSED</span>
            </div>
            <div class="tabular-num {paneNetPnl >= 0 ? 'pnl-positive' : 'pnl-negative'}" style="font-size: 0.75rem; font-weight: 700;">
              SLICE NET P&L: {formatSignedUsd(paneNetPnl)}
            </div>
          </header>
          <div class="data-table-container" style="max-block-size: 185px;">
            <table class="telemetry-table">
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Brain</th>
                  <th>Dir</th>
                  <th>Outcome</th>
                  <th data-numeric="true">Entry → Exit</th>
                  <th data-numeric="true">Realized R</th>
                  <th data-numeric="true">Net P&L ($)</th>
                </tr>
              </thead>
              <tbody>
                {#each paneClosed.slice(0, 12) as tr (tr.tradeId)}
                  <tr
                    onclick={() =>
                      terminalStore.updateViewportPane(pane.paneId, {
                        symbol: tr.symbol,
                        brainId: tr.brainId
                      })}
                    style="cursor: pointer;"
                  >
                    <td><strong>{tr.symbol}</strong></td>
                    <td><span class="badge badge-ai-core">{tr.brainId}</span></td>
                    <td>
                      <span class="badge {tr.direction === 'LONG' ? 'badge-long' : 'badge-short'}">
                        {tr.direction}
                      </span>
                    </td>
                    <td>
                      <span class="badge {tr.isWinner ? 'badge-long' : 'badge-short'}">
                        {tr.outcomeBadge}
                      </span>
                    </td>
                    <td data-numeric="true">
                      {formatSubPennyPrice(tr.entryPrice)} → {formatSubPennyPrice(tr.exitPrice)}
                    </td>
                    <td data-numeric="true" class={tr.realizedRMultiple >= 0 ? 'pnl-positive' : 'pnl-negative'}>
                      {tr.realizedRMultiple >= 0 ? '+' : ''}{tr.realizedRMultiple.toFixed(2)}R
                    </td>
                    <td data-numeric="true" class={tr.combinedTotalPnlUsd >= 0 ? 'pnl-positive' : 'pnl-negative'}>
                      {formatSignedUsd(tr.combinedTotalPnlUsd)}
                    </td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    {/each}
  </section>
{:else if terminalStore.workspaceLayout === 'QUAD_2X2'}
  <!-- 2x2 QUAD MATRIX: 4 SIMULTANEOUS BRAIN VIEWPORTS TICKING SIDE-BY-SIDE -->
  <section
    aria-label="2x2 Quad Brain Matrix Viewports"
    style="display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; padding: 10px 12px 0 12px;"
  >
    {#each terminalStore.quadViewports as vp (vp.paneId)}
      <div style="min-inline-size: 0;">
        <BlueprintCanvas
          paneId={vp.paneId}
          paneTitle={vp.title}
          paneLinkChannel={vp.linkChannel}
          paneBrainId={vp.brainId}
          paneTimeframe={vp.timeframe}
          paneRange={vp.range}
          paneLoading={vp.loading}
          customBlueprint={vp.blueprint}
        />
      </div>
    {/each}
  </section>
{/if}

<div class="command-bento-grid">
  {#if terminalStore.workspaceLayout === 'SINGLE'}
    <!-- Left Panel: Two-Stage 11,000+ Ticker Universe Scanner -->
    <section class="glass-panel panel-scanner" aria-label="11,000+ Ticker Universe Scanner">
      <header class="panel-header" style="flex-wrap: wrap; gap: 6px;">
        <div class="panel-title">
          <span>Yahoo Finance Live Scanner</span>
          <span class="badge badge-ai-core">{candidates.length} SIGNALS</span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
          <button
            type="button"
            class="action-btn"
            onclick={handleYahooScan}
            disabled={scanningYahoo}
            style="padding: 2px 7px; font-size: 0.67rem;"
          >
            {scanningYahoo ? 'Scanning Yahoo...' : 'Scan Yahoo Now'}
          </button>
        </div>
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
  {/if}

  <!-- Full-Width Panel 1: POSITIONS OPENED Real-Time Command Tracker -->
  <PositionsOpenedPanel />

  <!-- Full-Width Panel 2: CLOSED TRADE LEDGER, 6-Dimension Hyper-Filter & 1-Click CSV/XLSX Exporters -->
  <ClosedTradeLedgerPanel />

  <!-- Full-Width Panel 3: 3-Speed Continuous Self-Learning & Nightly Autopsy Lab -->
  <BrainAutopsyLab />
</div>
