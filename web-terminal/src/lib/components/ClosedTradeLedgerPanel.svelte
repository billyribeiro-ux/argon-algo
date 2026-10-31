<script lang="ts">
  import {
    terminalStore,
    formatSubPennyPrice,
    formatSignedUsd,
    formatSignedPct,
    formatHoldDuration,
    type QuickFilterPillId
  } from '$lib/state/terminal.svelte';

  const kpis = $derived(terminalStore.scorecardKpis);
  const rows = $derived(terminalStore.filteredLedger);
  const curve = $derived(terminalStore.equityCurve);
  const strategyBoard = $derived(terminalStore.byStrategyLeaderboard);
  const engineBoard = $derived(terminalStore.byEngineBreakdown);
  let expandedAutopsyTradeId = $state<string | null>(null);

  const QUICK_PILLS: Array<{ id: QuickFilterPillId; label: string }> = [
    { id: 'ALL', label: 'ALL CLOSED TRADES' },
    { id: 'WINS_ONLY', label: 'WINS ONLY' },
    { id: 'LOSSES_ONLY', label: 'LOSSES ONLY' },
    { id: 'TARGET_HITS', label: 'TARGET HITS' },
    { id: 'DYNAMIC_EXITS', label: 'AI DYNAMIC EXITS' },
    { id: 'OPTIONS_ONLY', label: 'OPTIONS ONLY' },
    { id: 'SUB_PENNY', label: 'SUB-PENNY ($0.00001–$5)' },
    { id: 'MEGA_CAP', label: 'MEGA-CAP & ETFS' },
    { id: 'HIGH_CONFIDENCE_90', label: '90%+ AI CONFIDENCE' }
  ];
</script>

<section class="glass-panel panel-closed-ledger" aria-label="Closed Trade Ledger, Hyper-Filter & Export Suite">
  <header class="panel-header">
    <div class="panel-title">
      <span>CLOSED TRADE LEDGER &amp; HYPER-FILTER SUITE</span>
      <span class="badge badge-long">WINS: {kpis.totalWins} ({kpis.winRatePct.toFixed(1)}%)</span>
      <span class="badge badge-short">LOSSES: {kpis.totalLosses} ({kpis.lossRatePct.toFixed(1)}%)</span>
    </div>

    <!-- 1-Click Institutional Export Buttons -->
    <div style="display: flex; align-items: center; gap: 8px;">
      <button type="button" class="action-btn" onclick={() => terminalStore.resetAllLedgerFilters()}>
        Reset Filters
      </button>
      <button
        type="button"
        class="action-btn action-btn-export"
        onclick={() => terminalStore.downloadFilteredCsv()}
      >
        [ EXPORT FILTERED (.CSV) ]
      </button>
      <button
        type="button"
        class="action-btn action-btn-primary"
        onclick={() => terminalStore.downloadFilteredXlsx()}
      >
        [ EXPORT EXCEL WORKBOOK (.XLSX) ]
      </button>
    </div>
  </header>

  <!-- Dynamic Executive Scorecard KPI Ribbon -->
  <div class="scorecard-kpi-ribbon">
    <div class="kpi-card">
      <span class="kpi-label">Filtered Trades (Win / Loss)</span>
      <span class="kpi-metric-value">
        {kpis.totalClosed}
        <span class="pnl-positive">({kpis.totalWins}W</span> /
        <span class="pnl-negative">{kpis.totalLosses}L)</span>
      </span>
    </div>
    <div class="kpi-card">
      <span class="kpi-label">Win Rate / Profit Factor</span>
      <span class="kpi-metric-value pnl-positive">
        {kpis.winRatePct.toFixed(1)}% | PF {kpis.profitFactor.toFixed(2)}
      </span>
    </div>
    <div class="kpi-card">
      <span class="kpi-label">Expectancy (Per Trade)</span>
      <span class="kpi-metric-value {kpis.expectancyR >= 0 ? 'pnl-positive' : 'pnl-negative'}">
        {kpis.expectancyR >= 0 ? '+' : ''}{kpis.expectancyR.toFixed(2)}R ({formatSignedUsd(kpis.expectancyUsd)})
      </span>
    </div>
    <div class="kpi-card">
      <span class="kpi-label">Net Stock P&amp;L</span>
      <span class="kpi-metric-value {kpis.netRealizedStockPnlUsd >= 0 ? 'pnl-positive' : 'pnl-negative'}">
        {formatSignedUsd(kpis.netRealizedStockPnlUsd)}
      </span>
    </div>
    <div class="kpi-card">
      <span class="kpi-label">Net Option P&amp;L</span>
      <span class="kpi-metric-value {kpis.netRealizedOptionPnlUsd >= 0 ? 'pnl-positive' : 'pnl-negative'}">
        {formatSignedUsd(kpis.netRealizedOptionPnlUsd)}
      </span>
    </div>
    <div class="kpi-card">
      <span class="kpi-label">Combined Realized P&amp;L</span>
      <span class="kpi-metric-value {kpis.netCombinedPnlUsd >= 0 ? 'pnl-positive' : 'pnl-negative'}">
        {formatSignedUsd(kpis.netCombinedPnlUsd)}
      </span>
    </div>
  </div>

  <!-- Interactive Strategy Archetype & Parallel Dual-Engine Win-Rate Leaderboard -->
  {#if strategyBoard.length > 0 || engineBoard.length > 0}
    <div
      style="padding: 8px 12px; background: var(--surface-0); border-block-end: 1px solid var(--border-subtle); display: flex; flex-direction: column; gap: 8px;"
    >
      {#if engineBoard.length > 0}
        <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
          <span style="font-size: 0.66rem; font-weight: 800; color: var(--text-secondary);">
            PARALLEL ENGINE LEADERBOARD (CLICK TO FILTER):
          </span>
          {#each engineBoard as eng (eng.engineMode)}
            <button
              type="button"
              class="filter-pill"
              data-selected={terminalStore.engineFilter === eng.engineMode}
              onclick={() => terminalStore.setEngineFilter(eng.engineMode)}
              style="padding: 3px 8px; font-size: 0.66rem; display: flex; align-items: center; gap: 6px;"
            >
              <strong>{eng.displayName}</strong>
              <span class="tabular-num pnl-positive">{eng.winRatePct.toFixed(1)}% WR</span>
              <span class="tabular-num">({eng.wins}W/{eng.losses}L | PF {eng.profitFactor.toFixed(2)})</span>
              <span class="tabular-num {eng.realizedStockNetPnlUsd >= 0 ? 'pnl-positive' : 'pnl-negative'}">
                {formatSignedUsd(
                  terminalStore.sizingMode === 'FRACTIONAL_KELLY'
                    ? eng.kellyRealizedNetPnlUsd
                    : eng.realizedStockNetPnlUsd
                )}
              </span>
            </button>
          {/each}
        </div>
      {/if}

      {#if strategyBoard.length > 0}
        <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
          <span style="font-size: 0.66rem; font-weight: 800; color: var(--text-secondary);">
            STRATEGY ARCHETYPE LEADERBOARD (CLICK TO FILTER):
          </span>
          {#each strategyBoard as strat (strat.strategyArchetype)}
            <button
              type="button"
              class="filter-pill"
              data-selected={terminalStore.strategyFilter === strat.strategyArchetype}
              onclick={() => terminalStore.setStrategyFilter(strat.strategyArchetype)}
              style="padding: 3px 8px; font-size: 0.66rem; display: flex; align-items: center; gap: 6px;"
            >
              <strong>{strat.displayName}</strong>
              <span class="tabular-num pnl-positive">{strat.winRatePct.toFixed(1)}% WR</span>
              <span class="tabular-num">({strat.wins}W/{strat.losses}L | PF {strat.profitFactor.toFixed(2)})</span>
              <span class="tabular-num" style="opacity: 0.85;">Best: {strat.bestSymbol}</span>
            </button>
          {/each}
        </div>
      {/if}
    </div>
  {/if}

  <!-- 6-Dimension Hyper-Filter Command Bar -->
  <div
    style="padding: 10px 12px; background: var(--surface-1); border-block-end: 1px solid var(--border-subtle); display: flex; flex-direction: column; gap: 10px;"
  >
    <!-- Dimension 1: Quick-Toggle Pills -->
    <div class="filter-pill-group" role="group" aria-label="Quick Ledger Filter Pills">
      {#each QUICK_PILLS as pill (pill.id)}
        <button
          type="button"
          class="filter-pill"
          data-selected={terminalStore.quickPill === pill.id}
          onclick={() => terminalStore.setQuickFilterPill(pill.id)}
        >
          {pill.label}
        </button>
      {/each}
    </div>

    <!-- Dimensions 2 - 6: Compounding Filter Controls -->
    <div
      style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; align-items: center;"
    >
      <label style="display: flex; flex-direction: column; gap: 3px; font-size: 0.68rem; color: var(--text-secondary);">
        <span>TICKER SEARCH</span>
        <input
          type="text"
          class="input-control"
          placeholder="e.g. NVDA, HYMC, GS..."
          bind:value={terminalStore.searchSymbol}
        />
      </label>

      <label style="display: flex; flex-direction: column; gap: 3px; font-size: 0.68rem; color: var(--text-secondary);">
        <span>BRAIN HORIZON</span>
        <select class="input-control" bind:value={terminalStore.brainFilter}>
          <option value="ALL_5_BRAINS">ALL 5 BRAINS</option>
          <option value="BRAIN-01">BRAIN-01 (Scalp)</option>
          <option value="BRAIN-02">BRAIN-02 (Day Trade)</option>
          <option value="BRAIN-03">BRAIN-03 (Multi-Day Swing)</option>
          <option value="BRAIN-04">BRAIN-04 (Multi-Week)</option>
          <option value="BRAIN-05">BRAIN-05 (Long-Term Wealth)</option>
        </select>
      </label>

      <label style="display: flex; flex-direction: column; gap: 3px; font-size: 0.68rem; color: var(--text-secondary);">
        <span>DIRECTION</span>
        <select class="input-control" bind:value={terminalStore.directionFilter}>
          <option value="ALL">LONG + SHORT</option>
          <option value="LONG">LONG ONLY</option>
          <option value="SHORT">SHORT ONLY</option>
        </select>
      </label>

      <label style="display: flex; flex-direction: column; gap: 3px; font-size: 0.68rem; color: var(--text-secondary);">
        <span>OUTCOME STATUS</span>
        <select class="input-control" bind:value={terminalStore.outcomeFilter}>
          <option value="ALL">ALL OUTCOMES</option>
          <option value="TARGET HIT">TARGET HIT</option>
          <option value="AI DYNAMIC EXIT">AI DYNAMIC EXIT</option>
          <option value="TRAILING STOP">TRAILING STOP</option>
          <option value="STOPPED OUT">STOPPED OUT</option>
          <option value="TIME EXPIRED">TIME EXPIRED</option>
        </select>
      </label>

      <label style="display: flex; flex-direction: column; gap: 3px; font-size: 0.68rem; color: var(--text-secondary);">
        <span>PRIMARY STREAM DRIVER</span>
        <select class="input-control" bind:value={terminalStore.driverFilter}>
          <option value="ALL">ALL 4 RAW STREAMS</option>
          <option value="PRICE_VOLUME_ANOMALY">PRICE &amp; VOLUME ANOMALY</option>
          <option value="OPTIONS_GAMMA_VANNA">OPTIONS GAMMA / VANNA</option>
          <option value="SHORT_FLOAT_SQUEEZE">SHORT &amp; FLOAT SQUEEZE</option>
          <option value="MARKET_INTERNALS_TIDE">MARKET INTERNALS TIDE</option>
        </select>
      </label>

      <label style="display: flex; flex-direction: column; gap: 3px; font-size: 0.68rem; color: var(--text-secondary);">
        <span>STRATEGY ARCHETYPE</span>
        <select class="input-control" bind:value={terminalStore.strategyFilter}>
          <option value="ALL">ALL 6 ARCHETYPES</option>
          <option value="GAMMA_SQUEEZE_IGNITION">GAMMA SQUEEZE IGNITION</option>
          <option value="SHORT_FLOAT_LIQUEFACTION">SHORT FLOAT LIQUEFACTION</option>
          <option value="POC_VOLUME_ABSORPTION">POC VOLUME ABSORPTION</option>
          <option value="MACRO_INTERNALS_TIDE_DRIVE">MACRO INTERNALS TIDE DRIVE</option>
          <option value="HTF_MULTI_SCALE_BREAKOUT">HTF MULTI-SCALE BREAKOUT</option>
          <option value="PURE_ML_LATENT_ANOMALY">PURE ML LATENT ANOMALY</option>
        </select>
      </label>

      <label style="display: flex; flex-direction: column; gap: 3px; font-size: 0.68rem; color: var(--text-secondary);">
        <span>EXECUTION ENGINE</span>
        <select class="input-control" bind:value={terminalStore.engineFilter}>
          <option value="ALL">ALL ENGINES</option>
          <option value="STRUCTURED_HYBRID_CHAMPION">ENGINE-A (HYBRID CHAMPION)</option>
          <option value="CONSENSUS_BOTH_ENGINES">CONSENSUS (ENGINE-A + ENGINE-B)</option>
          <option value="PURE_ML_SELF_LEARNED">ENGINE-B (PURE ML SELF-LEARNED)</option>
        </select>
      </label>

      <label style="display: flex; flex-direction: column; gap: 3px; font-size: 0.68rem; color: var(--text-secondary);">
        <span>MIN AI CONFIDENCE: {terminalStore.minConfidencePct}%</span>
        <input
          type="range"
          min="50"
          max="98"
          step="1"
          bind:value={terminalStore.minConfidencePct}
        />
      </label>

      <label style="display: flex; flex-direction: column; gap: 3px; font-size: 0.68rem; color: var(--text-secondary);">
        <span>MIN REALIZED R: {terminalStore.minRMultiple}R</span>
        <input
          type="range"
          min="-5"
          max="6"
          step="0.5"
          bind:value={terminalStore.minRMultiple}
        />
      </label>
    </div>
  </div>

  <!-- Closed Trade Ledger Table -->
  <div class="data-table-container">
    <table class="telemetry-table">
      <thead>
        <tr>
          <th>Trade ID</th>
          <th>Symbol</th>
          <th>Brain &amp; Engine</th>
          <th>Dir</th>
          <th>Outcome Badge</th>
          <th data-numeric="true">Entry → Exit Price</th>
          <th data-numeric="true">Stop / T1 / T2</th>
          <th data-numeric="true">Kelly % / 1R Risk ($) / Notional ($)</th>
          <th data-numeric="true">Stock P&amp;L ($ / %)</th>
          <th>Option Contract</th>
          <th data-numeric="true">Opt Entry → Exit</th>
          <th data-numeric="true">Option P&amp;L ($ / %)</th>
          <th data-numeric="true">Combined Total P&amp;L</th>
          <th data-numeric="true">Realized R</th>
          <th data-numeric="true">MFE / MAE</th>
          <th data-numeric="true">AI Conf</th>
          <th>Strategy Archetype &amp; Loss Forensic Autopsy</th>
          <th>Hold Duration</th>
          <th>Bar-by-Bar Scrubber</th>
        </tr>
      </thead>
      <tbody>
        {#each rows as row (row.tradeId)}
          {@const rowStockPnl =
            terminalStore.sizingMode === 'FRACTIONAL_KELLY'
              ? row.kellyRealizedStockPnlUsd
              : row.realizedStockPnlUsd}
          {@const rowCombinedPnl = rowStockPnl + row.realizedOptionPnlUsd}
          <tr
            data-selected={terminalStore.activeReplayTrade?.tradeId === row.tradeId || terminalStore.selectedTicker === row.symbol}
            onclick={() => terminalStore.selectClosedTradeForReplay(row)}
            style="cursor: pointer;"
            title="Click to launch Full-Lifecycle Bar-by-Bar Trade Replay Scrubber for {row.symbol} ({row.tradeId})"
          >
            <td class="tabular-num">{row.tradeId}</td>
            <td>
              <strong>{row.symbol}</strong>
              <span style="font-size: 0.65rem; color: var(--text-secondary); margin-inline-start: 4px;">
                {row.exchangeIndex}
              </span>
            </td>
            <td>
              <span class="badge badge-ai-core">{row.brainId}</span>
              <span class="badge {row.engineMode === 'CONSENSUS_BOTH_ENGINES' ? 'badge-long' : 'badge-gamma'}" style="margin-inline-start: 3px; font-size: 0.62rem;">
                {row.engineMode === 'CONSENSUS_BOTH_ENGINES'
                  ? 'A+B CONSENSUS'
                  : row.engineMode === 'PURE_ML_SELF_LEARNED'
                    ? 'ENGINE-B ML'
                    : 'ENGINE-A'}
              </span>
            </td>
            <td>
              <span class="badge {row.direction === 'LONG' ? 'badge-long' : 'badge-short'}">
                {row.direction}
              </span>
            </td>
            <td>
              <span class="badge {row.isWinner ? 'badge-long' : 'badge-short'}">
                CLOSED - {row.outcomeBadge}
              </span>
            </td>
            <td data-numeric="true">
              {formatSubPennyPrice(row.entryPrice)} → <strong>{formatSubPennyPrice(row.exitPrice)}</strong>
            </td>
            <td data-numeric="true" style="font-size: 0.68rem; color: var(--text-secondary);">
              {formatSubPennyPrice(row.structuralStopAtEntry)} / {formatSubPennyPrice(row.target1AtEntry)} / {formatSubPennyPrice(row.target2AtEntry)}
            </td>
            <td data-numeric="true" class="tabular-num">
              <span class="badge badge-ai-core">{row.kellyFractionPct.toFixed(2)}%</span>
              <span style="margin-inline-start: 4px; color: var(--text-secondary);">
                ${row.riskBudgetUsd.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </span>
              /
              <strong>
                ${row.notionalAllocatedUsd.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
              </strong>
              {#if row.trancheCount > 1}
                <span class="badge badge-long" style="margin-inline-start: 4px;">{row.trancheCount}x</span>
              {/if}
            </td>
            <td data-numeric="true" class={rowStockPnl >= 0 ? 'pnl-positive' : 'pnl-negative'}>
              {formatSignedUsd(rowStockPnl)} ({formatSignedPct(row.realizedStockPnlPct)})
            </td>
            <td class="tabular-num">
              {#if row.optionContractsCount > 0}
                {row.optionExpiration} ${row.optionStrike.toFixed(2)} {row.optionType} ({row.optionContractsCount}x)
              {:else}
                STOCK ONLY
              {/if}
            </td>
            <td data-numeric="true">
              {#if row.optionContractsCount > 0}
                {formatSubPennyPrice(row.optionEntryPremium)} → {formatSubPennyPrice(row.optionExitPremium)}
              {:else}
                N/A
              {/if}
            </td>
            <td data-numeric="true" class={row.realizedOptionPnlUsd >= 0 ? 'pnl-positive' : 'pnl-negative'}>
              {#if row.optionContractsCount > 0}
                {formatSignedUsd(row.realizedOptionPnlUsd)} ({formatSignedPct(row.realizedOptionPnlPct)})
              {:else}
                $0.00
              {/if}
            </td>
            <td
              data-numeric="true"
              style="font-weight: 700;"
              class={rowCombinedPnl >= 0 ? 'pnl-positive' : 'pnl-negative'}
            >
              {formatSignedUsd(rowCombinedPnl)}
            </td>
            <td data-numeric="true" class={row.realizedRMultiple >= 0 ? 'pnl-positive' : 'pnl-negative'}>
              {row.realizedRMultiple >= 0 ? '+' : ''}{row.realizedRMultiple.toFixed(2)}R
            </td>
            <td data-numeric="true">
              +{row.mfePct.toFixed(2)}% / -{row.maePct.toFixed(2)}%
            </td>
            <td data-numeric="true">{row.aiConfidenceAtEntryPct.toFixed(1)}%</td>
            <td>
              <span class="badge badge-gamma">{row.strategyArchetype}</span>
              {#if row.lossForensicAutopsy}
                <button
                  type="button"
                  class="badge badge-short"
                  style="margin-inline-start: 4px; cursor: pointer; border: 1px solid oklch(0.68 0.22 22);"
                  onclick={(e) => {
                    e.stopPropagation();
                    expandedAutopsyTradeId =
                      expandedAutopsyTradeId === row.tradeId ? null : row.tradeId;
                  }}
                >
                  🔬 {row.lossForensicAutopsy.rootCauseCategory}
                </button>
              {:else}
                <span style="font-size: 0.65rem; color: var(--text-secondary); margin-inline-start: 4px;">
                  {row.autopsyRootCauseCode}
                </span>
              {/if}
            </td>
            <td class="tabular-num">{formatHoldDuration(row.holdDurationMs)}</td>
            <td>
              <button
                type="button"
                class="action-btn {terminalStore.activeReplayTrade?.tradeId === row.tradeId ? 'action-btn-primary' : ''}"
                onclick={(e) => {
                  e.stopPropagation();
                  terminalStore.selectClosedTradeForReplay(row);
                }}
              >
                [ 🎬 Replay ({row.entryBars.length}b) ]
              </button>
            </td>
          </tr>
          {#if row.lossForensicAutopsy && expandedAutopsyTradeId === row.tradeId}
            <tr style="background: var(--surface-0);">
              <td colspan="19" style="padding: 10px 14px; font-size: 0.72rem; line-height: 1.45;">
                <div style="display: flex; flex-direction: column; gap: 6px;">
                  <div>
                    <strong class="pnl-negative">
                      LOSS FORENSIC AUTOPSY ({row.tradeId} — {row.symbol} {row.brainId}):
                    </strong>
                    <span> {row.lossForensicAutopsy.whatWentWrongSummary}</span>
                  </div>
                  <div>
                    <strong class="pnl-positive">SYNTHESIZED ADAPTATION RULE:</strong>
                    <span> {row.lossForensicAutopsy.learnedAdaptationRule}</span>
                    <span class="badge badge-ai-core" style="margin-inline-start: 8px;">
                      Counterfactual Early Exit: {row.lossForensicAutopsy.counterfactualEarlyExitR.toFixed(2)}R
                    </span>
                  </div>
                  <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                    {#each row.lossForensicAutopsy.topCulpritFeatures as cf (cf.featureIndex)}
                      <span class="badge badge-short">
                        #{cf.featureIndex} {cf.featureName} ({cf.streamName}): z={cf.anomalyZScore.toFixed(2)}σ (Loss={cf.lossValue.toFixed(3)} vs WinCentroid={cf.winnerCentroidValue.toFixed(3)})
                      </span>
                    {/each}
                  </div>
                </div>
              </td>
            </tr>
          {/if}
        {/each}
      </tbody>
    </table>
  </div>

  <!-- Cumulative Realized R & Equity Strip -->
  {#if curve.length > 0}
    <div
      style="padding: 8px 12px; background: var(--surface-0); border-block-start: 1px solid var(--border-subtle); display: flex; flex-wrap: wrap; gap: 14px; font-size: 0.72rem; align-items: center;"
    >
      <span style="color: var(--text-secondary); font-weight: 700;">EQUITY CURVE PROGRESSION:</span>
      {#each curve as pt (pt.tradeId)}
        <span class="tabular-num">
          #{pt.step} {pt.symbol}:
          <strong class={pt.cumulativePnlUsd >= 0 ? 'pnl-positive' : 'pnl-negative'}>
            {formatSignedUsd(pt.cumulativePnlUsd)} ({pt.cumulativeR >= 0 ? '+' : ''}{pt.cumulativeR.toFixed(2)}R)
          </strong>
        </span>
      {/each}
    </div>
  {/if}
</section>

