<script lang="ts">
  import { terminalStore } from '$lib/state/terminal.svelte';

  const evo = $derived(terminalStore.evolutionStatus);
  let isRunningAutopsy = $state(false);

  async function handleRunAutopsy(): Promise<void> {
    isRunningAutopsy = true;
    await terminalStore.runNightlyAutopsyNow();
    isRunningAutopsy = false;
  }
</script>

<section
  class="glass-panel"
  style="grid-column: span 12;"
  aria-label="3-Speed Self-Learning, Nightly Autopsy & Challenger Tournament Lab"
>
  <header class="panel-header" style="flex-wrap: wrap; gap: 8px;">
    <div class="panel-title" style="flex-wrap: wrap;">
      <span>3-SPEED CONTINUOUS SELF-LEARNING &amp; NIGHTLY AUTOPSY LAB</span>
      {#if evo}
        <span class="badge badge-ai-core">RUNS: {evo.totalAutopsyRuns}</span>
        <span class="badge {evo.chopThrottleActive ? 'badge-short' : 'badge-long'}">
          CHOP THROTTLE: {evo.chopThrottleActive ? 'ACTIVE' : 'NOMINAL'} (FLOOR {evo.chopConfidenceFloorPct.toFixed(1)}%)
        </span>
      {/if}
    </div>

    <div style="display: flex; align-items: center; gap: 8px;">
      {#if evo}
        <span class="tabular-num" style="font-size: 0.69rem; color: var(--text-secondary);">
          Last Autopsy: {evo.lastAutopsyIso.replace('T', ' ').slice(0, 19)} UTC
        </span>
      {/if}
      <button
        type="button"
        class="action-btn action-btn-primary"
        onclick={handleRunAutopsy}
        disabled={isRunningAutopsy}
      >
        {isRunningAutopsy
          ? 'Running Autopsy & Walk-Forward Tournament...'
          : '[ Run Nightly Autopsy & Challenger Tournament Now ]'}
      </button>
    </div>
  </header>

  {#if evo}
    <div class="scorecard-kpi-ribbon">
      <div class="kpi-card">
        <span class="kpi-label">Autopsy Cycles Executed</span>
        <span class="kpi-metric-value">{evo.totalAutopsyRuns}</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">Hard-Negatives Mined (3x Loss Weight)</span>
        <span class="kpi-metric-value pnl-negative">{evo.hardNegativesMined}</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">Missed 3R+ Expansion Moves Indexed</span>
        <span class="kpi-metric-value pnl-positive">{evo.missedMovesIndexed}</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">Speed-1 Chop Confidence Floor</span>
        <span class="kpi-metric-value" style="color: var(--signal-ai-core);">
          {evo.chopConfidenceFloorPct.toFixed(1)}%
        </span>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">Graded Closed Trades</span>
        <span class="kpi-metric-value">{evo.tradeGrades.length}</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">Promoted Challenger Brains</span>
        <span class="kpi-metric-value pnl-positive">
          {evo.tournamentResults.filter((t) => t.challengerPromoted).length} / {evo.tournamentResults.length}
        </span>
      </div>
    </div>

    {#if evo.era1Calibrations && evo.era1Calibrations.length > 0}
      <div style="padding: 12px 12px 0 12px;">
        <div class="glass-panel" style="background: var(--surface-0);">
          <header class="panel-header">
            <div class="panel-title">
              <span>Era-1 Causal Calibration &amp; 4-Stream Learned Priors (Zero Lookahead Out-of-Sample Gate)</span>
            </div>
          </header>
          <div class="data-table-container">
            <table class="telemetry-table">
              <thead>
                <tr>
                  <th>Brain</th>
                  <th data-numeric="true">Calibrated Conf Floor</th>
                  <th data-numeric="true">Min Precedent WR</th>
                  <th data-numeric="true">Min Expected R</th>
                  <th data-numeric="true">Era-1 Samples</th>
                  <th data-numeric="true">Era-1 Calibrated WR</th>
                  <th data-numeric="true">Era-1 Calibrated PF</th>
                  <th data-numeric="true">Learned Disp Envelope (p15..p90)</th>
                  <th data-numeric="true">Chart/Tape %</th>
                  <th data-numeric="true">Options/Gamma %</th>
                  <th data-numeric="true">Short/Float %</th>
                  <th data-numeric="true">Internals/Macro %</th>
                </tr>
              </thead>
              <tbody>
                {#each evo.era1Calibrations as cal (cal.brainId)}
                  <tr>
                    <td><span class="badge badge-ai-core">{cal.brainId}</span></td>
                    <td data-numeric="true" class="tabular-num">
                      <strong>{cal.calibratedConfidenceFloor.toFixed(1)}%</strong>
                    </td>
                    <td data-numeric="true" class="tabular-num">
                      {cal.minPrecedentWinRatePct.toFixed(1)}%
                    </td>
                    <td data-numeric="true" class="tabular-num">
                      +{cal.minExpectedRMultiple.toFixed(2)}R
                    </td>
                    <td data-numeric="true" class="tabular-num">
                      {cal.era1SampleTrades}
                    </td>
                    <td data-numeric="true" class="tabular-num pnl-positive">
                      {cal.era1CalibratedWinRatePct.toFixed(1)}%
                    </td>
                    <td data-numeric="true" class="tabular-num pnl-positive">
                      <strong>{cal.era1CalibratedProfitFactor.toFixed(2)}</strong>
                    </td>
                    <td data-numeric="true" class="tabular-num">
                      +{cal.era1LearnedMinDisplacementPct.toFixed(2)}% .. +{cal.era1LearnedMaxDisplacementPct.toFixed(2)}%
                    </td>
                    <td data-numeric="true" class="tabular-num">
                      {(cal.chartTapePhysicsWeight * 100).toFixed(1)}%
                    </td>
                    <td data-numeric="true" class="tabular-num">
                      {(cal.optionsGammaVannaWeight * 100).toFixed(1)}%
                    </td>
                    <td data-numeric="true" class="tabular-num">
                      {(cal.shortFloatSqueezeWeight * 100).toFixed(1)}%
                    </td>
                    <td data-numeric="true" class="tabular-num">
                      {(cal.marketInternalsMacroWeight * 100).toFixed(1)}%
                    </td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    {/if}

    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 12px; padding: 12px;">
      <!-- Speed 3: Champion vs Challenger Walk-Forward Tournament -->
      <div class="glass-panel" style="background: var(--surface-0);">
        <header class="panel-header">
          <div class="panel-title">
            <span>Speed 3: Champion vs. Challenger Walk-Forward Tournament</span>
          </div>
        </header>
        <div class="data-table-container">
          <table class="telemetry-table">
            <thead>
              <tr>
                <th>Brain</th>
                <th>Champion → Challenger</th>
                <th data-numeric="true">PF (Champ → Chal)</th>
                <th data-numeric="true">Max DD %</th>
                <th>Status</th>
                <th>Promotion Verdict</th>
              </tr>
            </thead>
            <tbody>
              {#each evo.tournamentResults as tr (tr.brainId)}
                <tr>
                  <td>
                    <span class="badge badge-ai-core">{tr.brainId}</span>
                    <div style="font-size: 0.64rem; color: var(--text-secondary);">{tr.brainName}</div>
                  </td>
                  <td class="tabular-num" style="font-size: 0.7rem;">
                    {tr.championVersion} → <strong>{tr.challengerVersion}</strong>
                  </td>
                  <td data-numeric="true">
                    {tr.championProfitFactor.toFixed(2)} →
                    <strong class="pnl-positive">{tr.challengerProfitFactor.toFixed(2)}</strong>
                  </td>
                  <td data-numeric="true">
                    {tr.championMaxDrawdownPct.toFixed(2)}% →
                    <strong class="pnl-positive">{tr.challengerMaxDrawdownPct.toFixed(2)}%</strong>
                  </td>
                  <td>
                    <span class="badge {tr.challengerPromoted ? 'badge-long' : 'badge-squeeze'}">
                      {tr.challengerPromoted ? 'PROMOTED TO LIVE ONNX' : 'RETAINED CHAMPION'}
                    </span>
                  </td>
                  <td style="font-size: 0.68rem; color: var(--text-secondary);">
                    {tr.promotionReason}
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      </div>

      <!-- Speed 2: Nightly Autopsy Trade Grading & Hard-Negative Mining -->
      <div class="glass-panel" style="background: var(--surface-0);">
        <header class="panel-header">
          <div class="panel-title">
            <span>Speed 2: Post-Market Trade Autopsy &amp; Hard-Negative Mining</span>
          </div>
        </header>
        <div class="data-table-container">
          <table class="telemetry-table">
            <thead>
              <tr>
                <th>Trade ID</th>
                <th>Symbol</th>
                <th>Brain</th>
                <th>Grade</th>
                <th data-numeric="true">Realized R</th>
                <th data-numeric="true">MFE / MAE</th>
                <th>Hard-Neg</th>
                <th>Causal Diagnosis &amp; Weight Update</th>
              </tr>
            </thead>
            <tbody>
              {#each evo.tradeGrades.slice(0, 12) as g (g.tradeId)}
                <tr
                  onclick={() => terminalStore.selectTicker(g.symbol)}
                  style="cursor: pointer;"
                >
                  <td class="tabular-num">{g.tradeId}</td>
                  <td><strong>{g.symbol}</strong></td>
                  <td><span class="badge badge-ai-core">{g.brainId}</span></td>
                  <td>
                    <span
                      class="badge {g.grade.startsWith('A')
                        ? 'badge-long'
                        : g.grade === 'B'
                          ? 'badge-gamma'
                          : g.grade === 'C'
                            ? 'badge-squeeze'
                            : 'badge-short'}"
                    >
                      GRADE {g.grade}
                    </span>
                  </td>
                  <td
                    data-numeric="true"
                    class={g.realizedRMultiple >= 0 ? 'pnl-positive' : 'pnl-negative'}
                  >
                    {g.realizedRMultiple >= 0 ? '+' : ''}{g.realizedRMultiple.toFixed(2)}R
                  </td>
                  <td data-numeric="true">
                    +{g.mfePct.toFixed(2)}% / -{g.maePct.toFixed(2)}%
                  </td>
                  <td>
                    {#if g.isHardNegative}
                      <span class="badge badge-short">3X LOSS WEIGHT</span>
                    {:else}
                      <span class="badge badge-long">CLEAN</span>
                    {/if}
                  </td>
                  <td style="font-size: 0.67rem; color: var(--text-secondary);">
                    <div><strong>{g.diagnosis}</strong></div>
                    <div>{g.weightAdjustmentSummary}</div>
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- Phase 12 Upgrade 1: Automatic Per-Loss Forensic Autopsy & Causal Adaptive Trap-Immunity Audit Log -->
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 12px; padding: 0 12px 12px 12px;">
      <!-- Automatic Per-Loss Forensic Dissection (32-Feature Culprit Z-Scores & Synthesized Adaptation Rules) -->
      <div class="glass-panel" style="background: var(--surface-0);">
        <header class="panel-header">
          <div class="panel-title">
            <span>Automatic Per-Loss Forensic Autopsy ({evo.lossAutopsies?.length ?? 0} Dissected Losses)</span>
          </div>
        </header>
        <div class="data-table-container">
          <table class="telemetry-table">
            <thead>
              <tr>
                <th>Loss Trade</th>
                <th>Root Cause Category</th>
                <th>Top 32-Feature Culprits (Z-Score)</th>
                <th data-numeric="true">Early Exit R</th>
                <th>Forensic Diagnosis &amp; Synthesized Rule</th>
              </tr>
            </thead>
            <tbody>
              {#each evo.lossAutopsies ?? [] as la (la.tradeId)}
                <tr onclick={() => terminalStore.selectTicker(la.symbol)} style="cursor: pointer;">
                  <td>
                    <div class="tabular-num"><strong>{la.tradeId}</strong></div>
                    <div>
                      <strong>{la.symbol}</strong>
                      <span class="badge badge-ai-core" style="margin-inline-start: 4px;">{la.brainId}</span>
                    </div>
                  </td>
                  <td>
                    <span class="badge badge-short">{la.rootCauseCategory}</span>
                  </td>
                  <td style="font-size: 0.65rem;">
                    {#each la.topCulpritFeatures as cf (cf.featureIndex)}
                      <div class="tabular-num">
                        <strong>#{cf.featureIndex} {cf.featureName}:</strong>
                        <span class="pnl-negative">z={cf.anomalyZScore.toFixed(2)}σ</span>
                        ({cf.lossValue.toFixed(2)} vs {cf.winnerCentroidValue.toFixed(2)})
                      </div>
                    {/each}
                  </td>
                  <td data-numeric="true" class="tabular-num pnl-positive">
                    {la.counterfactualEarlyExitR.toFixed(2)}R
                  </td>
                  <td style="font-size: 0.66rem; color: var(--text-secondary);">
                    <div style="color: var(--text-primary);">{la.whatWentWrongSummary}</div>
                    <div style="margin-block-start: 3px;" class="pnl-positive">
                      <strong>Rule:</strong> {la.learnedAdaptationRule}
                    </div>
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      </div>

      <!-- Causal Adaptive Trap-Immunity Veto Audit Log (t > T_loss_closed) -->
      <div class="glass-panel" style="background: var(--surface-0);">
        <header class="panel-header">
          <div class="panel-title">
            <span>Causal Adaptive Trap-Immunity Veto Audit Log ({evo.vetoedRepeatTraps?.length ?? 0} Vetoed Repeat Traps)</span>
          </div>
        </header>
        <div class="data-table-container">
          <table class="telemetry-table">
            <thead>
              <tr>
                <th>Candidate Bar Time</th>
                <th>Symbol &amp; Brain</th>
                <th>Matched Past Loss</th>
                <th>Trap Category</th>
                <th data-numeric="true">Cosine Sim</th>
                <th data-numeric="true">Loss Avoided ($)</th>
              </tr>
            </thead>
            <tbody>
              {#each evo.vetoedRepeatTraps ?? [] as vt, idx (`${vt.symbol}-${vt.timestampNs}-${idx}`)}
                <tr onclick={() => terminalStore.selectTicker(vt.symbol)} style="cursor: pointer;">
                  <td class="tabular-num" style="font-size: 0.67rem;">
                    {vt.timestampIso.replace('T', ' ').slice(0, 19)}
                  </td>
                  <td>
                    <strong>{vt.symbol}</strong>
                    <span class="badge badge-ai-core" style="margin-inline-start: 4px;">{vt.brainId}</span>
                  </td>
                  <td class="tabular-num">
                    <span class="badge badge-short">{vt.matchedPastLossTradeId}</span>
                  </td>
                  <td>
                    <span class="badge badge-squeeze">{vt.rootCauseCategory}</span>
                  </td>
                  <td data-numeric="true" class="tabular-num pnl-negative">
                    <strong>{(vt.trapSimilarityScore * 100).toFixed(1)}%</strong>
                  </td>
                  <td data-numeric="true" class="tabular-num pnl-positive">
                    <strong>+${vt.counterfactualLossAvoidedUsd.toFixed(2)}</strong>
                  </td>
                </tr>
              {:else}
                <tr>
                  <td colspan="6" style="padding: 12px; color: var(--text-secondary);">
                    Zero repeat-trap setups encountered after causal loss indexing (t &gt; T_loss_closed).
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  {:else}
    <div style="padding: 16px; font-size: 0.78rem; color: var(--text-secondary);">
      Click <strong>[ Run Nightly Autopsy &amp; Challenger Tournament Now ]</strong> to execute Speed-2
      MAE/MFE Trade Grading and Speed-3 Champion-vs-Challenger Walk-Forward Validation across all 5 Brains.
    </div>
  {/if}
</section>

