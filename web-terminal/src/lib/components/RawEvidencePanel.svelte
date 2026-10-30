<script lang="ts">
  import {
    terminalStore,
    formatSubPennyPrice,
    type ScannerSignalCandidate
  } from '$lib/state/terminal.svelte';

  const blueprint = $derived<ScannerSignalCandidate>(terminalStore.activeBlueprint);
</script>

<aside class="glass-panel panel-evidence-hub" aria-label="4-Stream Raw Evidence & Options Matcher">
  <header class="panel-header">
    <div class="panel-title">
      <span>4-Stream Raw Evidence Hub</span>
    </div>
    <span class="badge badge-ai-core">{blueprint.aiConfidencePct.toFixed(1)}% CONF</span>
  </header>

  <div style="padding: 12px; display: flex; flex-direction: column; gap: 14px;">
    <!-- 4-Stream Contribution Meters -->
    <div style="display: flex; flex-direction: column; gap: 9px;">
      <div style="font-size: 0.7rem; font-weight: 700; text-transform: uppercase; color: var(--text-secondary);">
        Causal Stream Attribution ({blueprint.primaryDriver})
      </div>

      <div>
        <div style="display: flex; justify-content: space-between; font-size: 0.72rem; margin-block-end: 3px;">
          <span>1. Price &amp; Volume Anomaly</span>
          <span class="tabular-num">{(blueprint.streamWeights.priceVolumeAnomaly * 100).toFixed(1)}%</span>
        </div>
        <div class="stream-meter-track">
          <div
            class="stream-meter-fill"
            style="inline-size: {blueprint.streamWeights.priceVolumeAnomaly * 100}%; background: var(--signal-long);"
          ></div>
        </div>
      </div>

      <div>
        <div style="display: flex; justify-content: space-between; font-size: 0.72rem; margin-block-end: 3px;">
          <span>2. Options Gamma / Vanna Flow</span>
          <span class="tabular-num">{(blueprint.streamWeights.optionsGammaVanna * 100).toFixed(1)}%</span>
        </div>
        <div class="stream-meter-track">
          <div
            class="stream-meter-fill"
            style="inline-size: {blueprint.streamWeights.optionsGammaVanna * 100}%; background: var(--signal-gamma);"
          ></div>
        </div>
      </div>

      <div>
        <div style="display: flex; justify-content: space-between; font-size: 0.72rem; margin-block-end: 3px;">
          <span>3. Short &amp; Float Squeeze Physics</span>
          <span class="tabular-num">{(blueprint.streamWeights.shortFloatSqueeze * 100).toFixed(1)}%</span>
        </div>
        <div class="stream-meter-track">
          <div
            class="stream-meter-fill"
            style="inline-size: {blueprint.streamWeights.shortFloatSqueeze * 100}%; background: var(--signal-squeeze);"
          ></div>
        </div>
      </div>

      <div>
        <div style="display: flex; justify-content: space-between; font-size: 0.72rem; margin-block-end: 3px;">
          <span>4. Market Internals &amp; Sector Tide</span>
          <span class="tabular-num">{(blueprint.streamWeights.marketInternalsTide * 100).toFixed(1)}%</span>
        </div>
        <div class="stream-meter-track">
          <div
            class="stream-meter-fill"
            style="inline-size: {blueprint.streamWeights.marketInternalsTide * 100}%; background: var(--signal-ai-core);"
          ></div>
        </div>
      </div>
    </div>

    <!-- pgvector Historical Precedent k-NN Card -->
    <div
      style="padding: 10px; background: var(--surface-0); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);"
    >
      <div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; color: var(--signal-ai-core); margin-block-end: 4px;">
        pgvector Historical Precedent k-NN
      </div>
      <div style="font-size: 0.78rem; font-weight: 700; margin-block-end: 6px;">
        Matches {blueprint.precedentMatchSummary.matchingWins}/{blueprint.precedentMatchSummary.totalK} Past Expansion Traces
      </div>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 0.72rem;">
        <div>
          Cluster Win Rate:
          <strong class="tabular-num pnl-positive">
            {blueprint.precedentMatchSummary.winRatePct.toFixed(1)}%
          </strong>
        </div>
        <div>
          Avg Realized R:
          <strong class="tabular-num pnl-positive">
            +{blueprint.precedentMatchSummary.avgRealizedR.toFixed(2)}R
          </strong>
        </div>
        <div>
          p80 MFE Excursion:
          <strong class="tabular-num pnl-positive">+{blueprint.precedentMatchSummary.avgMfePct.toFixed(2)}%</strong>
        </div>
        <div>
          p85 MAE Invalidation:
          <strong class="tabular-num pnl-negative">
            -{blueprint.precedentMatchSummary.avgMaePct.toFixed(2)}%
          </strong>
        </div>
      </div>
    </div>

    <!-- Optimal Options Contract & All 6 Analytical Greeks Card -->
    <div
      style="padding: 10px; background: var(--surface-0); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);"
    >
      <div style="display: flex; justify-content: space-between; align-items: center; margin-block-end: 6px;">
        <span style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; color: var(--signal-gamma);">
          AI Optimal Options Matcher
        </span>
        {#if blueprint.optionsRecommendation}
          <span class="badge badge-ai-core">{blueprint.optionsRecommendation.optionStructure}</span>
        {:else}
          <span class="badge badge-long">SHARES_ONLY</span>
        {/if}
      </div>

      {#if blueprint.optionsRecommendation}
        {@const opt = blueprint.optionsRecommendation}
        <div style="display: flex; justify-content: space-between; align-items: center; margin-block-end: 6px;">
          <strong class="tabular-num" style="font-size: 0.75rem;">
            {opt.matchedContractLabel}
          </strong>
          <span class="badge badge-gamma">{opt.dte} DTE</span>
        </div>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 5px; font-size: 0.71rem;">
          <div>Entry Premium: <strong class="tabular-num">{formatSubPennyPrice(opt.entryPremium)}</strong></div>
          <div>Target Premium: <strong class="tabular-num pnl-positive">{formatSubPennyPrice(opt.targetPremium)}</strong></div>
          <div>Stop Premium: <strong class="tabular-num pnl-negative">{formatSubPennyPrice(opt.stopPremium)}</strong></div>
          <div>Implied Vol (IV): <strong class="tabular-num">{opt.ivPct.toFixed(1)}%</strong></div>
          <div>Delta (Δ): <strong class="tabular-num">{opt.delta.toFixed(2)}</strong></div>
          <div>Gamma (Γ): <strong class="tabular-num">{opt.gamma.toFixed(3)}</strong></div>
          <div>Vanna (V): <strong class="tabular-num">{opt.vanna.toFixed(3)}</strong></div>
          <div>Theta (Θ): <strong class="tabular-num pnl-negative">{opt.theta.toFixed(2)}</strong></div>
          <div>Vega (ν): <strong class="tabular-num">{opt.vega.toFixed(2)}</strong></div>
          <div>Convexity: <strong class="tabular-num pnl-positive">OPTIMAL</strong></div>
        </div>
      {:else}
        <div style="font-size: 0.73rem; color: var(--text-secondary);">
          Pure Equity / Sub-Penny Execution ({formatSubPennyPrice(blueprint.livePrice)}) — 100% Direct Share Accumulation
        </div>
      {/if}
    </div>

    <!-- Live TreeSHAP Causal Feature Attribution -->
    {#if blueprint.shapDrivers.length > 0}
      <div
        style="padding: 10px; background: var(--surface-0); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);"
      >
        <div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; color: var(--text-secondary); margin-block-end: 6px;">
          TreeSHAP Raw Feature Attribution
        </div>
        <div style="display: flex; flex-direction: column; gap: 4px; font-size: 0.7rem;">
          {#each blueprint.shapDrivers.slice(0, 4) as drv (drv.feature)}
            <div style="display: flex; justify-content: space-between; align-items: center;">
              <code style="font-size: 0.67rem; color: var(--text-primary);">{drv.feature}</code>
              <strong class="tabular-num {drv.impact >= 0 ? 'pnl-positive' : 'pnl-negative'}">
                {drv.impact >= 0 ? '+' : ''}{drv.impact.toFixed(3)}
              </strong>
            </div>
          {/each}
        </div>
      </div>
    {/if}
  </div>
</aside>
