<script lang="ts">
  import {
    terminalStore,
    formatSubPennyPrice,
    formatSignedUsd,
    formatSignedPct,
    formatHoldDuration
  } from '$lib/state/terminal.svelte';

  const summary = $derived(terminalStore.openPositionsSummary);
  const heat = $derived(terminalStore.portfolioHeatTelemetry);
  const stagedTicket = $derived(terminalStore.activeStagedTicketModal);
  let activeBrokerTab = $state<'IBKR_TWS' | 'CHARLES_SCHWAB' | 'ALPACA_API'>('IBKR_TWS');
  let activeTicketAction = $state<'BUY_TO_OPEN' | 'SELL_TO_CLOSE' | 'SCRATCH_EXIT'>('BUY_TO_OPEN');
  let copiedPayloadFeedback = $state<boolean>(false);

  async function handleStageBrokerTicket(
    positionId: string,
    broker: 'IBKR_TWS' | 'CHARLES_SCHWAB' | 'ALPACA_API' = activeBrokerTab,
    action?: 'BUY_TO_OPEN' | 'SELL_TO_CLOSE' | 'SCRATCH_EXIT'
  ) {
    activeBrokerTab = broker;
    const ticket = await terminalStore.stageBrokerOrderTicket(positionId, broker, action);
    if (ticket) {
      activeTicketAction = (ticket.action as 'BUY_TO_OPEN' | 'SELL_TO_CLOSE' | 'SCRATCH_EXIT') ?? 'BUY_TO_OPEN';
    }
  }

  async function copySelectedBrokerPayload(payloadText: string) {
    if (typeof navigator !== 'undefined' && navigator.clipboard) {
      try {
        await navigator.clipboard.writeText(payloadText);
        copiedPayloadFeedback = true;
        setTimeout(() => {
          copiedPayloadFeedback = false;
        }, 1800);
      } catch {
        // Ignore clipboard permission errors
      }
    }
  }
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

  <!-- Cross-Brain Portfolio Delta-Gamma Heat & Ledoit-Wolf Correlation Ribbon -->
  <div
    style="padding: 8px 12px; background: var(--surface-0); border-block-end: 1px solid var(--border-subtle); display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; font-size: 0.70rem;"
  >
    <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
      <span style="font-weight: 800; color: var(--text-secondary);">
        PORTFOLIO DELTA-GAMMA HEAT &amp; CORRELATION CAP:
      </span>
      <span
        class="badge {heat.aggregateOpenRiskPct <= 4.5
          ? 'badge-long'
          : heat.aggregateOpenRiskPct <= 7.0
            ? 'badge-squeeze'
            : 'badge-short'}"
      >
        {heat.heatRegimeBadge}
      </span>
      <span class="tabular-num">
        Open Risk: <strong>{heat.aggregateOpenRiskPct.toFixed(2)}%</strong> / {heat.portfolioHeatCapPct.toFixed(2)}% Cap
        (${heat.aggregateOpenRiskUsd.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })})
      </span>
      <span class="tabular-num {heat.netPortfolioBetaDeltaUsd >= 0 ? 'pnl-positive' : 'pnl-negative'}">
        Net Beta-Δ: <strong>{formatSignedUsd(heat.netPortfolioBetaDeltaUsd)}</strong>
      </span>
      <span class="tabular-num" style="color: var(--text-secondary);">
        1% Γ: {formatSignedUsd(heat.netPortfolioGamma1pctUsd)} | Vanna: {formatSignedUsd(heat.netPortfolioVanna1pctUsd)}
      </span>
      <span class="tabular-num">
        Ledoit-Wolf ρ (Mean / Max): <strong>{heat.meanPairwiseCorrelation.toFixed(2)}</strong> / <strong>{heat.maxPairwiseCorrelation.toFixed(2)}</strong>
      </span>
      <span class="badge badge-ai-core tabular-num">
        Kelly Scale: {heat.correlationHeatMultiplier.toFixed(2)}x
      </span>
    </div>
    <div style="display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
      <span style="font-size: 0.65rem; color: var(--text-secondary);">
        {heat.overlappingClustersSummary}
      </span>
      {#each heat.pairwiseCorrelations.slice(0, 3) as cell (cell.posALabel + cell.posBLabel)}
        <span
          class="badge tabular-num"
          style="font-size: 0.60rem; background: var(--surface-2); color: var(--text-primary);"
        >
          {cell.posALabel} ↔ {cell.posBLabel}: ρ={cell.ledoitWolfSimilarity.toFixed(2)}
        </span>
      {/each}
    </div>
  </div>

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
          <th data-numeric="true">Kelly % / 1R Risk ($) / Notional ($)</th>
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
            <td data-numeric="true" class="tabular-num">
              <span class="badge badge-ai-core">{pos.kellyFractionPct.toFixed(2)}%</span>
              <span style="margin-inline-start: 4px; color: var(--text-secondary);">
                ${pos.riskBudgetUsd.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </span>
              /
              <strong>
                ${pos.notionalAllocatedUsd.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
              </strong>
              {#if pos.trancheCount > 1}
                <span class="badge badge-long" style="margin-inline-start: 4px;">{pos.trancheCount}x Tranche</span>
              {/if}
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
              <div style="display: flex; gap: 4px; flex-wrap: wrap;">
                <button
                  type="button"
                  class="action-btn action-btn-export"
                  onclick={(e) => {
                    e.stopPropagation();
                    void handleStageBrokerTicket(pos.tradeId, activeBrokerTab);
                  }}
                >
                  ⚡ Stage Broker Ticket
                </button>
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

  {#if stagedTicket}
    {@const activePayload =
      activeBrokerTab === 'IBKR_TWS'
        ? stagedTicket.ibkrTwsPayloadJson
        : activeBrokerTab === 'CHARLES_SCHWAB'
          ? stagedTicket.schwabTraderPayloadJson
          : stagedTicket.alpacaApiPayloadJson}
    <div
      style="padding: 12px 14px; background: var(--surface-1); border-block-start: 1px solid var(--border-subtle); display: flex; flex-direction: column; gap: 10px;"
      role="region"
      aria-label="Staged Institutional Broker Order Ticket"
    >
      <div style="display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap;">
        <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
          <span class="badge badge-ai-core">⚡ STAGED BROKER ORDER TICKET</span>
          <strong>{stagedTicket.symbol} ({stagedTicket.brainId})</strong>
          <span class="badge {stagedTicket.direction === 'LONG' ? 'badge-long' : 'badge-short'}">
            {stagedTicket.direction} | {stagedTicket.action}
          </span>
          <span class="badge badge-long">{stagedTicket.orderStatus}</span>
          <span class="tabular-num" style="font-size: 0.72rem; color: var(--text-secondary);">
            OCC: <strong>{stagedTicket.occOptionSymbol}</strong> | Qty: <strong>{stagedTicket.equityQuantityShares} sh</strong> + <strong>{stagedTicket.optionContractsCount}x {stagedTicket.optionStructure}</strong>
          </span>
        </div>

        <div style="display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
          {#each (['IBKR_TWS', 'CHARLES_SCHWAB', 'ALPACA_API'] as const) as brokerId (brokerId)}
            <button
              type="button"
              class="filter-pill"
              data-selected={activeBrokerTab === brokerId}
              onclick={() => void handleStageBrokerTicket(stagedTicket.positionId, brokerId, activeTicketAction)}
              style="padding: 3px 8px; font-size: 0.66rem;"
            >
              {brokerId === 'IBKR_TWS'
                ? 'IBKR TWS API'
                : brokerId === 'CHARLES_SCHWAB'
                  ? 'Charles Schwab'
                  : 'Alpaca API v2'}
            </button>
          {/each}
          {#each (['BUY_TO_OPEN', 'SELL_TO_CLOSE', 'SCRATCH_EXIT'] as const) as act (act)}
            <button
              type="button"
              class="filter-pill"
              data-selected={activeTicketAction === act}
              onclick={() => {
                activeTicketAction = act;
                void handleStageBrokerTicket(stagedTicket.positionId, activeBrokerTab, act);
              }}
              style="padding: 3px 8px; font-size: 0.64rem;"
            >
              {act}
            </button>
          {/each}
          <button
            type="button"
            class="action-btn action-btn-primary"
            onclick={() => void copySelectedBrokerPayload(activePayload)}
          >
            {copiedPayloadFeedback ? 'Copied JSON!' : 'Copy Broker JSON'}
          </button>
          <button
            type="button"
            class="action-btn"
            onclick={() => terminalStore.closeStagedTicketModal()}
          >
            Close Ticket
          </button>
        </div>
      </div>

      <div
        style="display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: 8px; font-size: 0.70rem;"
      >
        <div class="kpi-card" style="padding: 6px 8px;">
          <span class="kpi-label">Limit / Live Mark Price</span>
          <span class="kpi-metric-value tabular-num">
            {formatSubPennyPrice(stagedTicket.limitPrice)} / {formatSubPennyPrice(stagedTicket.liveMarkPrice)}
          </span>
        </div>
        <div class="kpi-card" style="padding: 6px 8px;">
          <span class="kpi-label">Structural Stop / Target 1</span>
          <span class="kpi-metric-value tabular-num">
            <span class="pnl-negative">{formatSubPennyPrice(stagedTicket.stopLossTriggerPrice)}</span>
            /
            <span class="pnl-positive">{formatSubPennyPrice(stagedTicket.target1PartialLimitPrice)}</span>
          </span>
        </div>
        <div class="kpi-card" style="padding: 6px 8px;">
          <span class="kpi-label">Option Limit / Stop / T1 Prem</span>
          <span class="kpi-metric-value tabular-num">
            {formatSubPennyPrice(stagedTicket.optionEntryLimitPremium)} / {formatSubPennyPrice(stagedTicket.optionStopTriggerPremium)} / {formatSubPennyPrice(stagedTicket.optionTarget1LimitPremium)}
          </span>
        </div>
        <div class="kpi-card" style="padding: 6px 8px;">
          <span class="kpi-label">Kelly Risk Budget &amp; Heat Scale</span>
          <span class="kpi-metric-value tabular-num">
            ${stagedTicket.kellyRiskBudgetUsd.toFixed(2)} ({stagedTicket.kellyFractionPct.toFixed(2)}% | {stagedTicket.correlationHeatMultiplier.toFixed(2)}x)
          </span>
        </div>
      </div>

      <pre
        style="margin: 0; padding: 8px 10px; background: var(--surface-0); border: 1px solid var(--border-subtle); border-radius: 4px; font-size: 0.66rem; max-block-size: 160px; overflow: auto; color: var(--text-primary);"
      >{activePayload}</pre>
    </div>
  {/if}
</section>

