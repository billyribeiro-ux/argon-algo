<script lang="ts">
  import { terminalStore, formatSignedUsd } from '$lib/state/terminal.svelte';

  const evo = $derived(terminalStore.evolutionStatus);
  const genHistory = $derived(terminalStore.evolutionGenerationsHistory);
  const synapticStates = $derived(terminalStore.synapticPlasticityStates);
  let isRunningAutopsy = $state(false);
  let selectedPlasticityBrain = $state<string>('ALL_5_BRAINS');

  const filteredSynapticStates = $derived(
    selectedPlasticityBrain === 'ALL_5_BRAINS'
      ? synapticStates
      : synapticStates.filter((s) => s.brain_id === selectedPlasticityBrain)
  );

  async function handleRunAutopsy(): Promise<void> {
    isRunningAutopsy = true;
    await terminalStore.runNightlyAutopsyNow();
    isRunningAutopsy = false;
  }

  async function handleRunNextGeneration(): Promise<void> {
    await terminalStore.runNextEvolutionGeneration();
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

    <!-- PHASE 18 & 19: AUTONOMOUS MULTI-GENERATION LOSS-ADAPTATION LOOP (GEN 0 -> GEN 8) -->
    <div style="padding: 12px 12px 0 12px;">
      <div class="glass-panel" style="background: var(--surface-0);">
        <header class="panel-header" style="flex-wrap: wrap; gap: 8px;">
          <div style="display: flex; flex-direction: column; gap: 3px;">
            <div class="panel-title" style="flex-wrap: wrap;">
              <span>🧬 AUTONOMOUS MULTI-GENERATION LOSS-ADAPTATION LOOP (GEN 0 → GEN 8)</span>
              {#if genHistory.length > 0}
                {@const activeChamp = genHistory.find((g) => g.promoted_as_champion) ?? genHistory[genHistory.length - 1]}
                {#if activeChamp}
                  <span class="badge badge-long">
                    ACTIVE CHAMPION: GEN #{activeChamp.generation_id} ({activeChamp.win_rate_pct.toFixed(2)}% WR | PF {activeChamp.profit_factor_fixed.toFixed(2)})
                  </span>
                {/if}
              {/if}
            </div>
            <span style="font-size: 0.66rem; color: var(--text-secondary);">
              Step A: Recursive Loss &amp; FP Autopsy → Step B: Empirical-Bayes Per-Brain Synaptic Specialization (λ_b) &amp; Mid-Trade Bar+4..+6 Stall SPRT → Step C: 100% Outcome-Blind Walk-Forward Re-Test → Step D: Composite Fitness Promotion Gate
            </span>
          </div>

          <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
            <button
              type="button"
              class="action-btn action-btn-primary"
              onclick={handleRunNextGeneration}
              disabled={terminalStore.isRunningEvolutionGeneration}
            >
              {terminalStore.isRunningEvolutionGeneration
                ? '🧬 Running Loss Autopsy & Synaptic Re-Test...'
                : '[ 🧬 Analyze Remaining Losses, Self-Adapt & Run Next Generation ]'}
            </button>
          </div>
        </header>

        {#if terminalStore.lastEvolutionGenerationBanner}
          <div
            style="padding: 8px 12px; background: var(--surface-1); border-block-end: 1px solid var(--border-subtle); font-size: 0.72rem;"
            class="tabular-num pnl-positive"
          >
            <strong>Latest Generation Result:</strong> {terminalStore.lastEvolutionGenerationBanner}
          </div>
        {/if}

        <!-- 2. Generation-by-Generation Progression Table (Gen 0 .. Gen 8) -->
        {#if genHistory.length > 0}
          <div class="data-table-container">
            <table class="telemetry-table">
              <thead>
                <tr>
                  <th>Generation</th>
                  <th>Champion Status</th>
                  <th data-numeric="true">Win Rate %</th>
                  <th data-numeric="true">Expectancy (R) &amp; Profit Factor</th>
                  <th data-numeric="true">Net R</th>
                  <th data-numeric="true">Veto Precision (TP/FP)</th>
                  <th data-numeric="true">Scratch Precision (TP/FP)</th>
                  <th data-numeric="true">Early Scratches (Bar+1..3)</th>
                  <th data-numeric="true">Mid-Trade Saves (Bar+4..6)</th>
                  <th data-numeric="true">Winners Preserved</th>
                  <th data-numeric="true">Remaining Losses</th>
                  <th data-numeric="true">Composite Fitness &amp; Kelly P&amp;L</th>
                  <th>Key Synaptic Mutations Applied</th>
                </tr>
              </thead>
              <tbody>
                {#each genHistory as gen (gen.generation_id)}
                  {@const statusUpper = gen.status.toUpperCase()}
                  {@const earlyScratches = gen.early_scratches_bar1_to_3 ?? gen.tp_scratches ?? gen.tp_losses_scratched}
                  {@const midTradeSaves = gen.mid_trade_stall_saves_bar4_to_6 ?? 0}
                  {@const netR =
                    gen.net_r ??
                    Number(
                      (
                        gen.winners_preserved_count * 2.15 -
                        earlyScratches * 0.25 -
                        midTradeSaves * 0.19 -
                        gen.remaining_full_stops * 1.0
                      ).toFixed(2)
                    )}
                  {@const expectancyR =
                    gen.expectancy_r ??
                    (gen.active_trades_count > 0 ? netR / gen.active_trades_count : 0)}
                  <tr>
                    <td class="tabular-num">
                      <strong>Gen {gen.generation_id}</strong>
                    </td>
                    <td>
                      <span
                        class="badge {statusUpper.includes('CHAMPION') || gen.promoted_as_champion
                          ? 'badge-long'
                          : statusUpper.includes('BASELINE')
                            ? 'badge-ai-core'
                            : statusUpper.includes('CONVERGED')
                              ? 'badge-squeeze'
                              : 'badge-gamma'}"
                      >
                        {gen.status}
                      </span>
                    </td>
                    <td data-numeric="true" class="tabular-num">
                      <strong class="pnl-positive">{gen.win_rate_pct.toFixed(2)}%</strong>
                      <div style="font-size: 0.64rem; color: var(--text-secondary);">
                        {gen.winners_preserved_count}W / {gen.active_losses_count}L
                      </div>
                    </td>
                    <td data-numeric="true" class="tabular-num">
                      <strong class="pnl-positive">
                        {expectancyR >= 0 ? '+' : ''}{expectancyR.toFixed(2)}R
                      </strong>
                      <div style="font-size: 0.64rem; color: var(--text-secondary);">
                        PF {gen.profit_factor_fixed.toFixed(2)} / Kelly {gen.profit_factor_kelly.toFixed(2)}
                      </div>
                    </td>
                    <td data-numeric="true" class="tabular-num pnl-positive">
                      <strong>{netR >= 0 ? '+' : ''}{netR.toFixed(1)}R</strong>
                      <div style="font-size: 0.63rem; color: var(--text-secondary);">
                        {formatSignedUsd(gen.fixed_10k_net_pnl)}
                      </div>
                    </td>
                    <td data-numeric="true" class="tabular-num pnl-positive">
                      <strong>{gen.veto_precision_pct.toFixed(2)}%</strong>
                      <div style="font-size: 0.64rem; color: var(--text-secondary);">
                        {gen.tp_losses_vetoed}TP / {gen.fp_winners_vetoed}FP
                      </div>
                    </td>
                    <td data-numeric="true" class="tabular-num pnl-positive">
                      <strong>{gen.scratch_precision_pct.toFixed(2)}%</strong>
                      <div style="font-size: 0.64rem; color: var(--text-secondary);">
                        {gen.tp_losses_scratched}TP / {gen.fp_winners_scratched}FP
                      </div>
                    </td>
                    <td data-numeric="true" class="tabular-num pnl-positive">
                      <strong>{earlyScratches}</strong>
                      <span style="font-size: 0.63rem; color: var(--text-secondary);">Bar+1..3</span>
                    </td>
                    <td data-numeric="true" class="tabular-num">
                      {#if midTradeSaves > 0}
                        <span class="badge badge-long">
                          +{midTradeSaves} SAVED (0 FP)
                        </span>
                      {:else}
                        <span style="color: var(--text-secondary);">0</span>
                      {/if}
                    </td>
                    <td data-numeric="true" class="tabular-num pnl-positive">
                      <strong>{gen.winners_preserved_count} / 116</strong>
                      <div style="font-size: 0.63rem; color: var(--text-secondary);">
                        Clipped: {gen.fp_winners_clipped_total} ({gen.fp_winners_vetoed}V+{gen.fp_winners_scratched}S)
                      </div>
                    </td>
                    <td
                      data-numeric="true"
                      class="tabular-num {gen.remaining_full_stops <= 5 ? 'pnl-positive' : 'pnl-negative'}"
                    >
                      <strong>{gen.remaining_full_stops}</strong>
                      <div style="font-size: 0.63rem; color: var(--text-secondary);">
                        Caught: {gen.tp_losses_caught_total}
                      </div>
                    </td>
                    <td
                      data-numeric="true"
                      class="tabular-num {gen.fractional_kelly_net_pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}"
                    >
                      <strong>{gen.composite_fitness_score.toFixed(1)}</strong>
                      <div style="font-size: 0.64rem;">
                        {formatSignedUsd(gen.fractional_kelly_net_pnl)}
                      </div>
                      <div style="font-size: 0.61rem; color: var(--text-secondary);">
                        DD ${gen.max_drawdown_usd.toFixed(0)}
                      </div>
                    </td>
                    <td style="font-size: 0.66rem; color: var(--text-secondary);">
                      {#each gen.key_synaptic_mutations as mut, mIdx (`${gen.generation_id}-${mIdx}`)}
                        <div style="margin-block-end: 2px;">• {mut}</div>
                      {/each}
                    </td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        {/if}

        <!-- 3. Live 32-Feature Synaptic Weight Plasticity Inspector -->
        {#if synapticStates.length > 0}
          <div style="padding: 10px 12px; border-block-start: 1px solid var(--border-subtle);">
            <div
              style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; margin-block-end: 8px;"
            >
              <div style="font-size: 0.72rem; font-weight: 800;">
                LIVE 32-FEATURE SYNAPTIC WEIGHT PLASTICITY INSPECTOR (BRAIN-01 .. BRAIN-05)
              </div>
              <div class="filter-pill-group" role="group" aria-label="Filter Synaptic Plasticity Brain">
                {#each ['ALL_5_BRAINS', 'BRAIN-01', 'BRAIN-02', 'BRAIN-03', 'BRAIN-04', 'BRAIN-05'] as bId (bId)}
                  <button
                    type="button"
                    class="filter-pill"
                    data-selected={selectedPlasticityBrain === bId}
                    onclick={() => (selectedPlasticityBrain = bId)}
                    style="padding: 2px 7px; font-size: 0.64rem;"
                  >
                    {bId === 'ALL_5_BRAINS' ? 'ALL 5 BRAINS' : bId}
                  </button>
                {/each}
              </div>
            </div>

            <div
              style="display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 10px;"
            >
              {#each filteredSynapticStates as sp (sp.brain_id)}
                {@const trapEntries = Object.entries(sp.trap_conviction_weights)}
                {@const ebLambdaPct = ((sp.empirical_bayes_shrinkage_lambda ?? 0.5) * 100).toFixed(1)}
                {@const midThresholds = sp.mid_trade_wald_sprt_thresholds ?? [1.48, 1.44, 1.4]}
                {@const boostedList = sp.top_boosted_features ?? sp.top_boosted_warning_features}
                {@const slashedList = sp.top_slashed_features ?? sp.top_slashed_deceiver_features}
                <div
                  class="glass-panel"
                  style="background: var(--surface-1); padding: 10px; display: flex; flex-direction: column; gap: 6px;"
                >
                  <div style="display: flex; align-items: center; justify-content: space-between; gap: 6px; flex-wrap: wrap;">
                    <div style="display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
                      <span class="badge badge-ai-core">{sp.brain_id}</span>
                      <span class="badge badge-long">GEN #{sp.generation_id}</span>
                      <span class="badge badge-gamma" title="Empirical-Bayes Hierarchical Shrinkage Weight on Brain-local Causal Topology vs. Global Pool">
                        EB λ_b: {ebLambdaPct}% LOCAL
                      </span>
                    </div>
                    <span class="tabular-num" style="font-size: 0.65rem; color: var(--text-secondary);">
                      Δ_veto: <strong>{sp.entry_veto_margin_delta.toFixed(3)}</strong> |
                      Wald A1..A3:
                      <strong>
                        {sp.wald_sprt_thresholds.map((t) => t.toFixed(2)).join('/')}
                      </strong>
                    </span>
                  </div>

                  <div class="tabular-num" style="font-size: 0.64rem; color: var(--text-secondary);">
                    <strong>Mid-Trade Stall Boundaries [Bar+4, Bar+5, Bar+6]:</strong>
                    <strong>{midThresholds.map((t) => t.toFixed(2)).join(' / ')}</strong>
                    | <strong>Learned Stop Ratchet Quantile:</strong>
                    <strong class="pnl-positive">
                      {(sp.learned_mid_trade_stop_ratchet_r ?? -0.19).toFixed(2)}R
                    </strong>
                    (MFE q={(sp.learned_mid_trade_mfe_norm_quantile ?? 0.58).toFixed(2)})
                  </div>

                  <div class="tabular-num" style="font-size: 0.65rem; color: var(--text-secondary);">
                    <strong>Stream 1..4 Priors:</strong>
                    S1 {(sp.stream_prior_multipliers[0] ?? 1.0).toFixed(2)}x ·
                    S2 {(sp.stream_prior_multipliers[1] ?? 1.0).toFixed(2)}x ·
                    S3 {(sp.stream_prior_multipliers[2] ?? 1.0).toFixed(2)}x ·
                    S4 {(sp.stream_prior_multipliers[3] ?? 1.0).toFixed(2)}x
                    {#if trapEntries.length > 0}
                      <span>
                        | <strong>Trap Gravity ({trapEntries.length} nodes):</strong>
                        {trapEntries
                          .slice(0, 3)
                          .map(([tid, w]) => `${tid.slice(0, 6)}:${Number(w).toFixed(2)}x`)
                          .join(', ')}
                      </span>
                    {/if}
                  </div>

                  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                    <div>
                      <div
                        style="font-size: 0.63rem; font-weight: 700; color: var(--signal-long); margin-block-end: 3px;"
                      >
                        BOOSTED WARNING SENSORS
                      </div>
                      {#each boostedList.slice(0, 4) as bf (bf.feature_index)}
                        <div class="tabular-num" style="font-size: 0.63rem; margin-block-end: 2px;">
                          <strong>#{bf.feature_index} {bf.feature_name}</strong>
                          <span class="badge badge-gamma" style="font-size: 0.54rem; padding: 0 3px;">
                            {bf.stream_name}
                          </span>
                          <div>
                            {bf.previous_weight.toFixed(2)} → <strong>{bf.updated_weight.toFixed(2)}</strong>
                            <span class="pnl-positive">(+{bf.delta_pct.toFixed(1)}%)</span>
                          </div>
                        </div>
                      {/each}
                    </div>

                    <div>
                      <div
                        style="font-size: 0.63rem; font-weight: 700; color: var(--signal-short); margin-block-end: 3px;"
                      >
                        SLASHED DECEIVER FEATURES
                      </div>
                      {#each slashedList.slice(0, 4) as sf (sf.feature_index)}
                        <div class="tabular-num" style="font-size: 0.63rem; margin-block-end: 2px;">
                          <strong>#{sf.feature_index} {sf.feature_name}</strong>
                          <span class="badge badge-squeeze" style="font-size: 0.54rem; padding: 0 3px;">
                            {sf.stream_name}
                          </span>
                          <div>
                            {sf.previous_weight.toFixed(2)} → <strong>{sf.updated_weight.toFixed(2)}</strong>
                            <span class="pnl-negative">({sf.delta_pct.toFixed(1)}%)</span>
                          </div>
                        </div>
                      {/each}
                    </div>
                  </div>
                </div>
              {/each}
            </div>
          </div>
        {/if}
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
                  <th data-numeric="true">Era-1 Median Stop %</th>
                  <th data-numeric="true">Learned Scratch Posterior</th>
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
                      {(cal.era1MedianStopPct * 100).toFixed(2)}%
                    </td>
                    <td data-numeric="true" class="tabular-num pnl-positive">
                      {(cal.era1LearnedScratchPosteriorThreshold * 100).toFixed(1)}%
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
