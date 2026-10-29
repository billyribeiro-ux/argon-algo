<script lang="ts">
  import {
    terminalStore,
    formatSubPennyPrice,
    formatSignedUsd,
    formatSignedPct,
    formatHoldDuration
  } from '$lib/state/terminal.svelte';

  const summary = $derived(terminalStore.openPositionsSummary);
</script>

<section class="glass-panel panel-positions-opened" aria-label="Positions Opened Real-Time Command Panel">
  <header class="panel-header">
    <div class="panel-title">
      <span>POSITIONS OPENED</span>
      <span class="badge badge-ai-core">{summary.count} ACTIVE</span>
      <span
        class="tabular-num {summary.combinedUnrealizedPnlUsd >= 0 ? 'pnl-positive' : 'pnl-negative'}"
        style="font-size: 0.78rem; margin-inline-start: 8px;"
      >
        Unrealized Total: {formatSignedUsd(summary.combinedUnrealizedPnlUsd)}
        (Stock: {formatSignedUsd(summary.unrealizedStockPnlUsd)} | Options: {formatSignedUsd(summary.unrealizedOptionPnlUsd)})
      </span>
    </div>

    <div style="display: flex; align-items: center; gap: 8px;">
      <button type="button" class="action-btn" onclick={() => terminalStore.stepLiveSimulation()}>
        Step Live Bar Tick
      </button>
    </div>
  </header>

  <div class="data-table-container positions-grid-wrapper">
    <table class="telemetry-table">
      <thead>
        <tr>
          <th>Trade ID</th>
          <th>Symbol</th>
          <th>Brain</th>
          <th>Dir</th>
          <th data-numeric="true">Entry Price</th>
          <th data-numeric="true">Live Price</th>
          <th data-numeric="true">Structural Stop</th>
          <th data-numeric="true">Target 1 / Target 2</th>
          <th data-numeric="true">Stock P&amp;L ($ / %)</th>
          <th>Options Contract</th>
          <th data-numeric="true">Opt Entry / Live</th>
          <th data-numeric="true">Option P&amp;L ($ / %)</th>
          <th data-numeric="true">Greeks (Δ / Γ / IV)</th>
          <th data-numeric="true">R-Mult &amp; Excursion</th>
          <th>Hold Window</th>
          <th>Speed-1 AI Trace Health</th>
          <th>Archive / Exit</th>
        </tr>
      </thead>
      <tbody>
        {#each terminalStore.openPositions as pos (pos.tradeId)}
          <tr
            data-selected={terminalStore.selectedTicker === pos.symbol}
            onclick={() => terminalStore.selectTicker(pos.symbol)}
          >
            <td class="tabular-num">{pos.tradeId}</td>
            <td>
              <strong>{pos.symbol}</strong>
              <span style="font-size: 0.65rem; color: var(--text-secondary); margin-inline-start: 4px;">
                {pos.exchangeIndex}
              </span>
            </td>
            <td><span class="badge badge-ai-core">{pos.brainId}</span></td>
            <td>
              <span class="badge {pos.direction === 'LONG' ? 'badge-long' : 'badge-short'}">
                {pos.direction}
              </span>
            </td>
            <td data-numeric="true">{formatSubPennyPrice(pos.entryPrice)}</td>
            <td data-numeric="true" style="font-weight: 700;">{formatSubPennyPrice(pos.livePrice)}</td>
            <td data-numeric="true" class="pnl-negative">
              {formatSubPennyPrice(pos.structuralStopPrice)}
            </td>
            <td data-numeric="true" class="pnl-positive">
              {formatSubPennyPrice(pos.target1Price)} / {formatSubPennyPrice(pos.target2Price)}
            </td>
            <td
              data-numeric="true"
              class={pos.unrealizedStockPnlUsd >= 0 ? 'pnl-positive' : 'pnl-negative'}
            >
              {formatSignedUsd(pos.unrealizedStockPnlUsd)} ({formatSignedPct(pos.unrealizedStockPnlPct)})
            </td>
            <td class="tabular-num">{pos.optionContractLabel}</td>
            <td data-numeric="true">
              {#if pos.optionContractsCount > 0}
                {formatSubPennyPrice(pos.optionEntryPremium)} → {formatSubPennyPrice(pos.optionLivePremium)}
              {:else}
                N/A
              {/if}
            </td>
            <td
              data-numeric="true"
              class={pos.unrealizedOptionPnlUsd >= 0 ? 'pnl-positive' : 'pnl-negative'}
            >
              {#if pos.optionContractsCount > 0}
                {formatSignedUsd(pos.unrealizedOptionPnlUsd)} ({formatSignedPct(pos.unrealizedOptionPnlPct)})
              {:else}
                $0.00
              {/if}
            </td>
            <td data-numeric="true">
              {#if pos.optionContractsCount > 0}
                Δ {pos.delta.toFixed(2)} | Γ {pos.gamma.toFixed(3)} | {pos.ivPct.toFixed(1)}%
              {:else}
                Equity Direct
              {/if}
            </td>
            <td data-numeric="true">
              <span class={pos.currentRMultiple >= 0 ? 'pnl-positive' : 'pnl-negative'}>
                {pos.currentRMultiple >= 0 ? '+' : ''}{pos.currentRMultiple.toFixed(2)}R
              </span>
              <span style="color: var(--text-secondary); font-size: 0.68rem;">
                (+{pos.mfePct.toFixed(1)}% / -{pos.maePct.toFixed(1)}%)
              </span>
            </td>
            <td class="tabular-num">
              {formatHoldDuration(pos.elapsedHoldMs)} / {formatHoldDuration(pos.predictedHoldMs)}
            </td>
            <td>
              <span
                class="badge {pos.traceHealthBadge === 'OPTIMAL'
                  ? 'badge-long'
                  : pos.traceHealthBadge === 'EXHAUSTION WARNING'
                    ? 'badge-squeeze'
                    : 'badge-short'}"
              >
                {pos.traceHealthBadge}
              </span>
            </td>
            <td>
              <div style="display: flex; gap: 4px;">
                <button
                  type="button"
                  class="action-btn action-btn-primary"
                  onclick={(e) => {
                    e.stopPropagation();
                    terminalStore.closeOpenPosition(pos.tradeId, 'TARGET HIT');
                  }}
                >
                  Target Hit
                </button>
                <button
                  type="button"
                  class="action-btn"
                  onclick={(e) => {
                    e.stopPropagation();
                    terminalStore.closeOpenPosition(pos.tradeId, 'AI DYNAMIC EXIT');
                  }}
                >
                  AI Dynamic Exit
                </button>
              </div>
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
</section>

