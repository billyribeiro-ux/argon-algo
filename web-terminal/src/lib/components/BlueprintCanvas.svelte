<script lang="ts">
  import { onDestroy } from 'svelte';
  import {
    terminalStore,
    formatSubPennyPrice,
    formatSignedUsd,
    type ScannerSignalCandidate,
    type ChartTimeframeId,
    type ChartRangeId,
    type RawBarCandle,
    type BrainId,
    type LinkChannelColor,
    type ClosedTradeLedgerItem
  } from '$lib/state/terminal.svelte';
  import { executionAudio } from '$lib/audio/execution-audio';

  const TIMEFRAMES: ChartTimeframeId[] = ['1m', '3m', '5m', '15m', '1H', '4H', '1D', '1W', '1M'];
  const RANGES: ChartRangeId[] = ['1D', '5D', '1M', '6M', '1Y', '5Y'];
  const BRAIN_OPTIONS: Array<Exclude<BrainId, 'ALL_5_BRAINS'>> = [
    'BRAIN-01',
    'BRAIN-02',
    'BRAIN-03',
    'BRAIN-04',
    'BRAIN-05'
  ];

  let {
    customBlueprint = undefined,
    paneId = undefined,
    paneTitle = undefined,
    paneTimeframe = undefined,
    paneRange = undefined,
    paneBrainId = undefined,
    paneLinkChannel = undefined,
    paneLoading = false
  }: {
    customBlueprint?: ScannerSignalCandidate | null;
    paneId?: string;
    paneTitle?: string;
    paneTimeframe?: ChartTimeframeId;
    paneRange?: ChartRangeId;
    paneBrainId?: Exclude<BrainId, 'ALL_5_BRAINS'>;
    paneLinkChannel?: LinkChannelColor;
    paneLoading?: boolean;
  } = $props();

  let showVerificationPopover = $state(false);
  let customSymbolInput = $state('');
  let hoveredBarIndex = $state<number | null>(null);
  let hoverMouseX = $state<number>(0);
  let hoverMouseY = $state<number>(0);

  // Interactive Zoom & Pan state
  let visibleBarCount = $state<number>(64);
  let panOffsetBars = $state<number>(0);
  let isDragging = $state<boolean>(false);
  let dragStartX = $state<number>(0);
  let dragStartPanOffset = $state<number>(0);

  // Institutional Bar-by-Bar Trade Replay Scrubber state (Maximization 2)
  let currentReplayBarIdx = $state<number>(0);
  let isReplayPlaying = $state<boolean>(false);
  let replaySpeedMultiplier = $state<1 | 4 | 10>(1);
  let replayTimerId: ReturnType<typeof setInterval> | null = null;
  let lastReplayTradeId = $state<string | null>(null);

  const blueprint = $derived<ScannerSignalCandidate>(
    customBlueprint ?? terminalStore.activeBlueprint
  );
  const effectiveTimeframe = $derived<ChartTimeframeId>(
    paneTimeframe ?? terminalStore.activeTimeframe
  );
  const effectiveRange = $derived<ChartRangeId>(paneRange ?? terminalStore.activeRange);
  const isChartLoading = $derived<boolean>(
    paneId ? paneLoading : terminalStore.chartLoading
  );
  const symbolClosedTrades = $derived(
    terminalStore.closedLedger.filter(
      (t) => t.symbol.toUpperCase() === blueprint.symbol.toUpperCase()
    )
  );
  const symbolOpenPositions = $derived(
    terminalStore.openPositions.filter(
      (p) => p.symbol.toUpperCase() === blueprint.symbol.toUpperCase()
    )
  );

  const activeReplayTrade = $derived<ClosedTradeLedgerItem | null>(
    !paneId ? terminalStore.activeReplayTrade : null
  );

  const replayLifecycle = $derived.by(() => {
    const tr = activeReplayTrade;
    if (!tr || !tr.entryBars || tr.entryBars.length === 0) return null;
    const bars = tr.entryBars;
    const total = bars.length;
    const entryIdx = Math.max(
      0,
      Math.min(
        total - 1,
        tr.entryBarIdx > 0 && tr.entryBarIdx < total
          ? tr.entryBarIdx
          : Math.min(24, Math.max(0, total - 2))
      )
    );
    const exitIdx = Math.max(
      entryIdx,
      Math.min(total - 1, tr.exitBarIdx > entryIdx ? tr.exitBarIdx : total - 1)
    );
    const isLong = tr.direction === 'LONG';
    const entryPrice = tr.entryPrice;
    const initialStop = tr.structuralStopAtEntry;
    const risk1R = Math.max(0.00001, Math.abs(entryPrice - initialStop));
    const t1Price = tr.target1AtEntry;
    const t2Price = tr.target2AtEntry;
    const runnerLockR = tr.brainId === 'BRAIN-01' ? 0.6 : 0.45;

    let t1HitBarIdx: number | null = null;
    let scaleInBarIdx: number | null = null;
    for (let k = entryIdx + 1; k <= exitIdx; k++) {
      const b = bars[k]!;
      const favExcursion = isLong ? b.high - entryPrice : entryPrice - b.low;
      if (
        scaleInBarIdx === null &&
        tr.trancheCount > 1 &&
        favExcursion >= 0.32 * risk1R
      ) {
        scaleInBarIdx = k;
      }
      const reachedT1 = isLong ? b.high >= t1Price : b.low <= t1Price;
      if (t1HitBarIdx === null && reachedT1) {
        t1HitBarIdx = k;
      }
    }

    const clampedCursor = Math.max(0, Math.min(total - 1, currentReplayBarIdx));
    let activeStopPrice = initialStop;
    let stopRatchetStage = 'INITIAL STOP (p85 MAE)';
    let partialBankedAtCursor = false;
    let activeTranchesAtCursor = clampedCursor >= entryIdx ? 1 : 0;

    for (let k = entryIdx; k <= Math.min(clampedCursor, exitIdx); k++) {
      const b = bars[k]!;
      const favR = (isLong ? b.high - entryPrice : entryPrice - b.low) / risk1R;
      if (scaleInBarIdx !== null && k >= scaleInBarIdx) {
        activeTranchesAtCursor = tr.trancheCount;
        const scaleStop = isLong ? entryPrice + 0.3 * risk1R : entryPrice - 0.3 * risk1R;
        if ((isLong && scaleStop > activeStopPrice) || (!isLong && scaleStop < activeStopPrice)) {
          activeStopPrice = scaleStop;
          stopRatchetStage = 'SCALE-IN FLOOR (+0.30R)';
        }
      }
      if (t1HitBarIdx !== null && k >= t1HitBarIdx) {
        partialBankedAtCursor = true;
        const runnerStop = isLong
          ? entryPrice + runnerLockR * risk1R
          : entryPrice - runnerLockR * risk1R;
        if ((isLong && runnerStop > activeStopPrice) || (!isLong && runnerStop < activeStopPrice)) {
          activeStopPrice = runnerStop;
          stopRatchetStage = `RUNNER RATCHET (+${runnerLockR.toFixed(2)}R)`;
        }
      }
      if (favR >= 1.7) {
        const lock1R = isLong ? entryPrice + 1.0 * risk1R : entryPrice - 1.0 * risk1R;
        if ((isLong && lock1R > activeStopPrice) || (!isLong && lock1R < activeStopPrice)) {
          activeStopPrice = lock1R;
          stopRatchetStage = 'RUNNER RATCHET (+1.00R LOCK)';
        }
      }
    }

    const curBar = bars[clampedCursor]!;
    let simulatedR = 0;
    let simulatedPnlUsd = 0;
    let phaseBadge = 'PRE-ENTRY CONTEXT';

    if (clampedCursor < entryIdx) {
      phaseBadge = 'PRE-ENTRY CONTEXT';
      simulatedR = 0;
      simulatedPnlUsd = 0;
    } else if (clampedCursor >= exitIdx) {
      phaseBadge = `FINAL EXIT — ${tr.outcomeBadge}`;
      simulatedR = tr.realizedRMultiple;
      simulatedPnlUsd = tr.combinedTotalPnlUsd;
    } else {
      phaseBadge = partialBankedAtCursor ? 'RUNNER HOLD (BANKED 50% @ T1)' : 'ACTIVE HOLD';
      const signedMove = isLong ? curBar.close - entryPrice : entryPrice - curBar.close;
      const markR = signedMove / risk1R;
      const t1R = Math.abs(t1Price - entryPrice) / risk1R;
      simulatedR = partialBankedAtCursor ? 0.5 * t1R + 0.5 * markR : markR;
      const effectiveRiskUsd = Math.max(75, tr.riskBudgetUsd || 150);
      simulatedPnlUsd = simulatedR * effectiveRiskUsd;
    }

    return {
      trade: tr,
      bars,
      total,
      entryIdx,
      exitIdx,
      t1HitBarIdx,
      scaleInBarIdx,
      clampedCursor,
      activeStopPrice,
      stopRatchetStage,
      partialBankedAtCursor,
      activeTranchesAtCursor,
      simulatedR,
      simulatedPnlUsd,
      phaseBadge,
      t1Price,
      t2Price
    };
  });

  $effect(() => {
    const tr = activeReplayTrade;
    if (tr && tr.tradeId !== lastReplayTradeId) {
      lastReplayTradeId = tr.tradeId;
      stopReplayTimer();
      const total = tr.entryBars.length;
      const entryIdx =
        tr.entryBarIdx > 0 && tr.entryBarIdx < total
          ? tr.entryBarIdx
          : Math.min(24, Math.max(0, total - 2));
      currentReplayBarIdx = entryIdx;
      panOffsetBars = 0;
      visibleBarCount = Math.max(48, total);
    } else if (!tr && lastReplayTradeId !== null) {
      lastReplayTradeId = null;
      stopReplayTimer();
    }
  });

  function stopReplayTimer(): void {
    isReplayPlaying = false;
    if (replayTimerId !== null) {
      clearInterval(replayTimerId);
      replayTimerId = null;
    }
  }

  onDestroy(() => {
    stopReplayTimer();
  });

  function emitReplayAudioIfMilestone(nextBarIdx: number): void {
    if (!replayLifecycle) return;
    const { trade, entryIdx, t1HitBarIdx, scaleInBarIdx, exitIdx } = replayLifecycle;
    if (nextBarIdx === entryIdx) {
      executionAudio.playSignalEntryChime(trade.direction);
    } else if (t1HitBarIdx !== null && nextBarIdx === t1HitBarIdx) {
      executionAudio.playTarget1BankedChime();
    } else if (scaleInBarIdx !== null && nextBarIdx === scaleInBarIdx) {
      executionAudio.playScaleInChime();
    } else if (nextBarIdx === exitIdx) {
      executionAudio.playPositionClosedChime(trade.isWinner);
    }
  }

  function stepReplayBy(delta: number): void {
    if (!replayLifecycle) return;
    const next = Math.max(0, Math.min(replayLifecycle.total - 1, currentReplayBarIdx + delta));
    currentReplayBarIdx = next;
    emitReplayAudioIfMilestone(next);
    if (next >= replayLifecycle.total - 1) {
      stopReplayTimer();
    }
  }

  function jumpReplayToEntry(): void {
    if (!replayLifecycle) return;
    stopReplayTimer();
    currentReplayBarIdx = replayLifecycle.entryIdx;
    emitReplayAudioIfMilestone(replayLifecycle.entryIdx);
  }

  function jumpReplayToFullOutcome(): void {
    if (!replayLifecycle) return;
    stopReplayTimer();
    currentReplayBarIdx = replayLifecycle.exitIdx;
    emitReplayAudioIfMilestone(replayLifecycle.exitIdx);
  }

  function startReplayInterval(): void {
    if (replayTimerId !== null) {
      clearInterval(replayTimerId);
    }
    const intervalMs = replaySpeedMultiplier === 10 ? 75 : replaySpeedMultiplier === 4 ? 180 : 480;
    replayTimerId = setInterval(() => {
      if (!replayLifecycle) {
        stopReplayTimer();
        return;
      }
      if (currentReplayBarIdx >= replayLifecycle.total - 1) {
        stopReplayTimer();
        return;
      }
      stepReplayBy(1);
    }, intervalMs);
  }

  function toggleReplayPlayPause(): void {
    if (!replayLifecycle) return;
    if (isReplayPlaying) {
      stopReplayTimer();
      return;
    }
    if (currentReplayBarIdx >= replayLifecycle.total - 1) {
      currentReplayBarIdx = replayLifecycle.entryIdx;
    }
    isReplayPlaying = true;
    startReplayInterval();
  }

  function setReplaySpeed(mult: 1 | 4 | 10): void {
    replaySpeedMultiplier = mult;
    if (isReplayPlaying) {
      startReplayInterval();
    }
  }

  function toggleReplayModeButton(): void {
    if (activeReplayTrade) {
      stopReplayTimer();
      terminalStore.clearReplayTrade();
      return;
    }
    const candidate =
      symbolClosedTrades[0] ?? terminalStore.closedLedger[0] ?? null;
    if (candidate) {
      terminalStore.selectClosedTradeForReplay(candidate);
    }
  }

  const windowedBars = $derived.by((): RawBarCandle[] => {
    if (replayLifecycle) {
      return replayLifecycle.bars.slice(0, replayLifecycle.clampedCursor + 1);
    }
    const total = blueprint.bars.length;
    if (total === 0) return [];
    const count = Math.max(8, Math.min(total, visibleBarCount));
    const maxOffset = Math.max(0, total - count);
    const clampedOffset = Math.max(0, Math.min(maxOffset, panOffsetBars));
    const endIdx = total - clampedOffset;
    const startIdx = Math.max(0, endIdx - count);
    return blueprint.bars.slice(startIdx, endIdx);
  });

  const hoveredBar = $derived.by(() => {
    if (hoveredBarIndex === null) return null;
    return windowedBars[hoveredBarIndex] ?? null;
  });

  const activeInspectBar = $derived.by(() => {
    if (hoveredBar) return hoveredBar;
    return windowedBars.length > 0 ? windowedBars[windowedBars.length - 1]! : null;
  });

  const hoveredBarLogReturnPct = $derived.by(() => {
    if (!activeInspectBar || activeInspectBar.open <= 0) return 0;
    return Math.log(activeInspectBar.close / activeInspectBar.open) * 100;
  });

  const chartGeometry = $derived.by(() => {
    const bars = replayLifecycle ? replayLifecycle.bars : windowedBars;
    const barPrices = bars
      .flatMap((b) => [b.high, b.low])
      .filter((p) => Number.isFinite(p) && p > 0);
    const refMid =
      barPrices.length > 0
        ? (Math.min(...barPrices) + Math.max(...barPrices)) * 0.5
        : blueprint.entryPrice || 100;
    const levelCandidates = [
      blueprint.entryPrice,
      blueprint.structuralStopPrice,
      blueprint.target1Price,
      blueprint.target2Price,
      replayLifecycle?.activeStopPrice ?? 0
    ].filter(
      (p) => Number.isFinite(p) && p > 0 && Math.abs(p - refMid) / Math.max(refMid, 0.00001) <= 0.25
    );
    const allPrices = [...barPrices, ...levelCandidates];

    const safePrices = allPrices.length > 0 ? allPrices : [100, 105];
    const minP = Math.min(...safePrices) * 0.996;
    const maxP = Math.max(...safePrices) * 1.004;
    const rangeP = Math.max(maxP - minP, 0.00001);
    const maxVol = Math.max(1, ...bars.map((b) => b.volume));

    const width = 760;
    const height = 350;
    const padTop = 22;
    const padBottom = 64;
    const plotHeight = height - padTop - padBottom;

    const priceToY = (p: number) =>
      Math.max(
        padTop,
        Math.min(padTop + plotHeight, padTop + plotHeight - ((p - minP) / rangeP) * plotHeight)
      );

    return {
      width,
      height,
      minP,
      maxP,
      maxVol,
      entryY: priceToY(blueprint.entryPrice || safePrices[0]!),
      stopY: priceToY(blueprint.structuralStopPrice || safePrices[0]!),
      activeReplayStopY: priceToY(
        replayLifecycle?.activeStopPrice ?? blueprint.structuralStopPrice ?? safePrices[0]!
      ),
      t1Y: priceToY(blueprint.target1Price || safePrices[0]!),
      t2Y: priceToY(blueprint.target2Price || safePrices[0]!),
      priceToY
    };
  });

  function handleSymbolSubmit(e: SubmitEvent): void {
    e.preventDefault();
    const sym = customSymbolInput.trim().toUpperCase();
    if (!sym) return;
    panOffsetBars = 0;
    if (paneId) {
      void terminalStore.updateViewportPane(paneId, { symbol: sym });
    } else {
      terminalStore.selectTicker(sym);
    }
  }

  function handleTimeframeSelect(tf: ChartTimeframeId): void {
    if (paneId) {
      void terminalStore.updateViewportPane(paneId, { timeframe: tf });
    } else {
      terminalStore.setTimeframe(tf);
    }
  }

  function handleRangeSelect(rg: ChartRangeId): void {
    if (paneId) {
      void terminalStore.updateViewportPane(paneId, { range: rg });
    } else {
      terminalStore.setRange(rg);
    }
  }

  function handleBrainSelect(e: Event): void {
    const val = (e.currentTarget as HTMLSelectElement).value as Exclude<BrainId, 'ALL_5_BRAINS'>;
    if (paneId) {
      void terminalStore.updateViewportPane(paneId, { brainId: val });
    } else {
      terminalStore.selectBrain(val);
    }
  }

  function handlePopoutPane(): void {
    void terminalStore.syncBus.detachPanelWindow(
      '/blueprint',
      `${blueprint.symbol} — ${paneBrainId ?? blueprint.brainId}`,
      1,
      paneLinkChannel ?? terminalStore.activeLinkChannel
    );
  }

  function handleWheelZoom(e: WheelEvent): void {
    e.preventDefault();
    const total = Math.max(16, blueprint.bars.length);
    if (e.deltaY < 0) {
      visibleBarCount = Math.max(10, Math.floor(visibleBarCount * 0.85));
    } else {
      visibleBarCount = Math.min(total, Math.ceil(visibleBarCount * 1.15));
    }
  }

  function handlePointerDown(e: PointerEvent): void {
    isDragging = true;
    dragStartX = e.clientX;
    dragStartPanOffset = panOffsetBars;
  }

  function handlePointerMove(e: PointerEvent): void {
    const svg = e.currentTarget as SVGSVGElement;
    const rect = svg.getBoundingClientRect();
    const relX = ((e.clientX - rect.left) / Math.max(1, rect.width)) * chartGeometry.width;
    const relY = ((e.clientY - rect.top) / Math.max(1, rect.height)) * chartGeometry.height;
    hoverMouseX = relX;
    hoverMouseY = relY;

    if (windowedBars.length > 0) {
      const stepX = (chartGeometry.width - 140) / Math.max(1, windowedBars.length);
      const rawIdx = Math.round((relX - 60) / stepX);
      if (rawIdx >= 0 && rawIdx < windowedBars.length) {
        hoveredBarIndex = rawIdx;
      } else {
        hoveredBarIndex = null;
      }
    }

    if (isDragging) {
      const dx = e.clientX - dragStartX;
      const barsDelta = Math.round(dx / 8);
      const maxOffset = Math.max(0, blueprint.bars.length - visibleBarCount);
      panOffsetBars = Math.max(0, Math.min(maxOffset, dragStartPanOffset + barsDelta));
    }
  }

  function handlePointerUp(): void {
    isDragging = false;
  }

  function resetZoomAndPan(): void {
    visibleBarCount = Math.max(32, blueprint.bars.length);
    panOffsetBars = 0;
  }
</script>

<section class="glass-panel panel-blueprint-canvas" aria-label="Raw Price & AI Trade Blueprint Canvas">
  <header class="panel-header" style="flex-wrap: wrap; gap: 8px;">
    <div class="panel-title" style="flex-wrap: wrap;">
      {#if paneLinkChannel}
        <span class="badge badge-ai-core">
          {paneLinkChannel === 'RED'
            ? '🔴 RED'
            : paneLinkChannel === 'BLUE'
              ? '🔵 BLUE'
              : paneLinkChannel === 'GREEN'
                ? '🟢 GREEN'
                : '🟡 GOLD'}
        </span>
      {/if}
      {#if paneTitle}
        <span style="font-size: 0.72rem; color: var(--text-secondary);">{paneTitle}:</span>
      {/if}
      <span class="badge {blueprint.direction === 'LONG' ? 'badge-long' : 'badge-short'}">
        {blueprint.direction}
      </span>
      <span>{blueprint.symbol}</span>
      <span class="tabular-num" style="color: var(--signal-ai-core);">
        {formatSubPennyPrice(blueprint.livePrice)}
      </span>
      {#if paneId}
        <select
          class="select-control"
          value={paneBrainId ?? blueprint.brainId}
          onchange={handleBrainSelect}
          aria-label="Viewport Brain Horizon Selector"
          style="font-size: 0.68rem; padding: 2px 6px;"
        >
          {#each BRAIN_OPTIONS as bOpt (bOpt)}
            <option value={bOpt}>{bOpt}</option>
          {/each}
        </select>
      {:else}
        <span class="badge badge-ai-core">{blueprint.brainId}</span>
      {/if}
      <span class="badge badge-gamma">{blueprint.exchangeIndex}</span>
      {#if isChartLoading}
        <span class="badge badge-squeeze">SYNCING YAHOO...</span>
      {:else}
        <span class="badge badge-long">YAHOO LIVE ({blueprint.bars.length} BARS)</span>
      {/if}
    </div>

    <div style="display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
      <!-- Any-Ticker Live Yahoo Finance Search -->
      <form onsubmit={handleSymbolSubmit} style="display: flex; align-items: center; gap: 4px;">
        <input
          type="text"
          class="input-control"
          placeholder="Ticker (NVDA, PLTR, BTC-USD...)"
          bind:value={customSymbolInput}
          style="inline-size: 135px; font-size: 0.72rem;"
          aria-label="Search Any Yahoo Finance Ticker"
        />
        <button type="submit" class="action-btn">Load Yahoo</button>
      </form>

      <button
        type="button"
        class="action-btn"
        onclick={() =>
          terminalStore.runCustomSymbolBacktest(
            blueprint.symbol,
            paneBrainId ?? blueprint.brainId
          )}
        disabled={terminalStore.backtestLoading}
        title="Run 100% Causal Walk-Forward Backtest (Era-1 Calibration + Era-2 Out-of-Sample Execution) on this ticker"
      >
        {terminalStore.backtestLoading
          ? `⚡ Backtesting ${blueprint.symbol}...`
          : `⚡ Deep Backtest ${blueprint.symbol} on ${paneBrainId ?? blueprint.brainId}`}
      </button>

      {#if !paneId}
        <button
          type="button"
          class="action-btn {activeReplayTrade ? 'action-btn-primary' : ''}"
          onclick={toggleReplayModeButton}
          title="Toggle Full-Lifecycle (PRE-ENTRY + HOLD + EXIT) Bar-by-Bar Trade Replay Scrubber"
        >
          {activeReplayTrade ? '🎬 Exit Replay Mode' : '🎬 Replay Mode'}
        </button>
      {/if}

      {#if paneId}
        <button
          type="button"
          class="action-btn"
          onclick={handlePopoutPane}
          title="Pop out this viewport into a dedicated OS window"
        >
          ↗ Pop-Out
        </button>
      {/if}

      <button
        type="button"
        class="action-btn signal-anchor-trigger"
        onclick={() => (showVerificationPopover = !showVerificationPopover)}
      >
        AI Blueprint Proof ({blueprint.aiConfidencePct.toFixed(1)}%)
      </button>
      <button
        type="button"
        class="action-btn action-btn-primary"
        onclick={() => terminalStore.openSignalAsPosition(blueprint)}
      >
        + Deploy to POSITIONS OPENED
      </button>
    </div>
  </header>

  {#if terminalStore.lastBacktestSummary}
    {@const bt = terminalStore.lastBacktestSummary}
    {@const totalNetPnl = bt.realizedStockNetPnlUsd + bt.realizedOptionNetPnlUsd}
    <div
      role="status"
      aria-label="Causal Walk-Forward Backtest Telemetry Banner"
      style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; padding: 7px 12px; background: oklch(0.2 0.04 250 / 0.85); border-block-end: 1px solid oklch(0.65 0.16 250 / 0.5); font-size: 0.72rem;"
    >
      <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
        <span class="badge badge-ai-core">⚡ CAUSAL WALK-FORWARD BACKTEST COMPLETE</span>
        <strong>{bt.symbol} · {bt.brainId} ({bt.timeframe})</strong>
        <span class="tabular-num" style="color: var(--text-secondary);">
          Bars Scanned: <strong style="color: var(--text-primary);">{bt.totalBarsScanned}</strong>
          (Era-1 Floor: {bt.era1Calibration.calibratedConfidenceFloor.toFixed(1)}%)
        </span>
        <span class="tabular-num">
          Trades: <strong>{bt.closedTradesGenerated}</strong>
          {#if bt.openPositionActive}
            (+1 OPEN)
          {/if}
        </span>
        <span class="tabular-num">
          Win Rate:
          <strong class={bt.winRatePct >= 50 ? 'pnl-positive' : 'pnl-negative'}>
            {bt.winRatePct.toFixed(1)}%
          </strong>
        </span>
        <span class="tabular-num">
          Profit Factor:
          <strong class={bt.profitFactor >= 1.25 ? 'pnl-positive' : 'pnl-negative'}>
            {bt.profitFactor.toFixed(2)}
          </strong>
        </span>
        <span class="tabular-num">
          Net P&amp;L:
          <strong class={totalNetPnl >= 0 ? 'pnl-positive' : 'pnl-negative'}>
            {formatSignedUsd(totalNetPnl)}
          </strong>
        </span>
      </div>
      <button
        type="button"
        class="action-btn"
        onclick={() => terminalStore.dismissBacktestSummary()}
        style="padding: 2px 7px; font-size: 0.65rem;"
      >
        Dismiss ✕
      </button>
    </div>
  {/if}

  {#if replayLifecycle}
    <div
      role="region"
      aria-label="Institutional Trade Replay Control Deck"
      style="display: flex; flex-direction: column; gap: 8px; padding: 9px 12px; background: oklch(0.18 0.04 275 / 0.92); border-block-end: 1px solid oklch(0.65 0.16 275 / 0.55); font-size: 0.72rem;"
    >
      <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
        <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
          <span class="badge badge-ai-core">🎬 FULL-LIFECYCLE TRADE REPLAY</span>
          <strong>
            {replayLifecycle.trade.tradeId} · {replayLifecycle.trade.symbol} ({replayLifecycle.trade.brainId})
          </strong>
          <span class="badge {replayLifecycle.simulatedPnlUsd >= 0 ? 'badge-long' : 'badge-short'}">
            {replayLifecycle.phaseBadge}
          </span>
          <span class="tabular-num">
            Simulated P&amp;L:
            <strong class={replayLifecycle.simulatedPnlUsd >= 0 ? 'pnl-positive' : 'pnl-negative'}>
              {formatSignedUsd(replayLifecycle.simulatedPnlUsd)} ({replayLifecycle.simulatedR >= 0 ? '+' : ''}{replayLifecycle.simulatedR.toFixed(2)}R)
            </strong>
          </span>
          <span class="tabular-num" style="color: var(--text-secondary);">
            Stop Ratchet:
            <strong style="color: oklch(0.82 0.16 80);">
              {formatSubPennyPrice(replayLifecycle.activeStopPrice)} ({replayLifecycle.stopRatchetStage})
            </strong>
          </span>
          <span class="tabular-num">
            Tranches: <strong>{replayLifecycle.activeTranchesAtCursor}/{replayLifecycle.trade.trancheCount}</strong>
          </span>
          {#if replayLifecycle.partialBankedAtCursor}
            <span class="badge badge-long">BANKED 50% @ T1</span>
          {/if}
          <span class="tabular-num" style="color: var(--text-secondary);">
            Kelly: <strong>{replayLifecycle.trade.kellyFractionPct.toFixed(2)}%</strong>
            (1R: ${replayLifecycle.trade.riskBudgetUsd.toFixed(0)} | Notional: ${replayLifecycle.trade.notionalAllocatedUsd.toFixed(0)})
          </span>
        </div>

        <button
          type="button"
          class="action-btn"
          onclick={() => {
            stopReplayTimer();
            terminalStore.clearReplayTrade();
          }}
          style="padding: 2px 8px; font-size: 0.66rem;"
        >
          ✕ Close Replay
        </button>
      </div>

      <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
        <div style="display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
          <button type="button" class="action-btn" onclick={jumpReplayToEntry}>
            [ ⏮ Jump to Entry ]
          </button>
          <button
            type="button"
            class="action-btn"
            onclick={() => stepReplayBy(-1)}
            disabled={replayLifecycle.clampedCursor <= 0}
          >
            [ ◀ -1 Bar ]
          </button>
          <button
            type="button"
            class="action-btn action-btn-primary"
            onclick={toggleReplayPlayPause}
          >
            {isReplayPlaying ? '[ ⏸ Pause ]' : '[ ▶ Play ]'}
          </button>
          <button
            type="button"
            class="action-btn"
            onclick={() => stepReplayBy(1)}
            disabled={replayLifecycle.clampedCursor >= replayLifecycle.total - 1}
          >
            [ +1 Bar ▶ ]
          </button>
          <button type="button" class="action-btn" onclick={jumpReplayToFullOutcome}>
            [ ⏭ Full Outcome ]
          </button>

          <div style="display: inline-flex; align-items: center; gap: 3px; margin-inline-start: 6px;">
            <span style="color: var(--text-secondary); font-size: 0.66rem;">Speed:</span>
            {#each [1, 4, 10] as spd (spd)}
              <button
                type="button"
                class="filter-pill"
                data-selected={replaySpeedMultiplier === spd}
                onclick={() => setReplaySpeed(spd as 1 | 4 | 10)}
                style="padding: 1px 6px; font-size: 0.65rem;"
              >
                {spd}x
              </button>
            {/each}
          </div>
        </div>

        <div style="display: flex; align-items: center; gap: 8px; flex: 1; min-inline-size: 220px;">
          <span class="tabular-num" style="font-size: 0.68rem; white-space: nowrap;">
            Bar <strong>{replayLifecycle.clampedCursor + 1}</strong> of <strong>{replayLifecycle.total}</strong>
            (Entry #{replayLifecycle.entryIdx + 1} → Exit #{replayLifecycle.exitIdx + 1})
          </span>
          <input
            type="range"
            min="0"
            max={Math.max(1, replayLifecycle.total - 1)}
            step="1"
            bind:value={currentReplayBarIdx}
            aria-label="Trade Replay Bar Scrubber"
            style="flex: 1;"
          />
        </div>
      </div>
    </div>
  {/if}

  <!-- Multi-Timeframe & Range Switcher Strip -->
  <div
    style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; padding: 6px 12px; background: var(--surface-1); border-block-end: 1px solid var(--border-subtle); font-size: 0.7rem;"
  >
    <div style="display: flex; align-items: center; gap: 4px; flex-wrap: wrap;">
      <span style="color: var(--text-secondary); font-weight: 700; margin-inline-end: 4px;">TF:</span>
      {#each TIMEFRAMES as tf (tf)}
        <button
          type="button"
          class="filter-pill"
          data-selected={effectiveTimeframe === tf}
          onclick={() => handleTimeframeSelect(tf)}
          style="padding: 2px 7px; font-size: 0.68rem;"
        >
          {tf}
        </button>
      {/each}
    </div>

    <div style="display: flex; align-items: center; gap: 4px; flex-wrap: wrap;">
      <span style="color: var(--text-secondary); font-weight: 700; margin-inline-end: 4px;">RANGE:</span>
      {#each RANGES as rg (rg)}
        <button
          type="button"
          class="filter-pill"
          data-selected={effectiveRange === rg}
          onclick={() => handleRangeSelect(rg)}
          style="padding: 2px 7px; font-size: 0.68rem;"
        >
          {rg}
        </button>
      {/each}
      <button
        type="button"
        class="action-btn"
        onclick={resetZoomAndPan}
        style="padding: 2px 7px; font-size: 0.66rem;"
      >
        Reset Zoom
      </button>
    </div>
  </div>

  {#if showVerificationPopover}
    <div class="signal-verification-popover" role="dialog" aria-label="AI Signal Verification">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-block-end: 8px;">
        <strong style="font-size: 0.8rem; text-transform: uppercase;">
          Zero-Skew Raw Sequence Verification ({blueprint.symbol})
        </strong>
        <button type="button" class="action-btn" onclick={() => (showVerificationPopover = false)}>
          Close
        </button>
      </div>
      <p style="font-size: 0.74rem; color: var(--text-secondary); margin-block-end: 8px;">
        Computed directly by <code>raw-fingerprint-core</code> + <code>PrecedentKnnIndex</code> using
        real Yahoo Finance OHLCV bars, p85 MAE structural invalidation, p50/p80 MFE targets, and
        Black-Scholes OPRA Greeks.
      </p>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 0.73rem;">
        <div>Entry Zone: <strong class="tabular-num">{formatSubPennyPrice(blueprint.entryPrice)}</strong></div>
        <div>Structural Stop (p85 MAE): <strong class="tabular-num pnl-negative">{formatSubPennyPrice(blueprint.structuralStopPrice)}</strong></div>
        <div>Target 1 (p50 MFE): <strong class="tabular-num pnl-positive">{formatSubPennyPrice(blueprint.target1Price)}</strong></div>
        <div>Target 2 (p80 MFE): <strong class="tabular-num pnl-positive">{formatSubPennyPrice(blueprint.target2Price)}</strong></div>
        <div>Expected Payoff: <strong class="tabular-num">+{blueprint.expectedRMultiple.toFixed(2)}R</strong></div>
        <div>Hold Horizon: <strong class="tabular-num">{blueprint.predictedHoldWindow}</strong></div>
      </div>
    </div>
  {/if}

  <!-- Blueprint Telemetry Strip -->
  <div
    style="display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px; padding: 10px 12px; background: var(--surface-0); border-block-end: 1px solid var(--border-subtle); font-size: 0.73rem;"
  >
    <div>
      <div style="color: var(--text-secondary); font-size: 0.66rem;">AI ENTRY ZONE</div>
      <div class="tabular-num" style="font-weight: 700;">
        {formatSubPennyPrice(blueprint.entryPrice)}
      </div>
    </div>
    <div>
      <div style="color: var(--text-secondary); font-size: 0.66rem;">STRUCTURAL STOP (p85 MAE)</div>
      <div class="tabular-num pnl-negative" style="font-weight: 700;">
        {formatSubPennyPrice(blueprint.structuralStopPrice)}
      </div>
    </div>
    <div>
      <div style="color: var(--text-secondary); font-size: 0.66rem;">TARGET 1 (p50 MFE)</div>
      <div class="tabular-num pnl-positive" style="font-weight: 700;">
        {formatSubPennyPrice(blueprint.target1Price)}
      </div>
    </div>
    <div>
      <div style="color: var(--text-secondary); font-size: 0.66rem;">TARGET 2 (p80 MFE)</div>
      <div class="tabular-num pnl-positive" style="font-weight: 700;">
        {formatSubPennyPrice(blueprint.target2Price)} ({blueprint.expectedRMultiple.toFixed(2)}R)
      </div>
    </div>
    <div>
      <div style="color: var(--text-secondary); font-size: 0.66rem;">PREDICTED WINDOW</div>
      <div class="tabular-num" style="font-weight: 700; color: var(--signal-ai-core);">
        {blueprint.predictedHoldWindow}
      </div>
    </div>
  </div>

  <!-- Zero-CLS Always-Mounted Crosshair / Latest-Bar Telemetry Bar -->
  <div
    class="tabular-num"
    style="display: flex; flex-wrap: wrap; gap: 12px; padding: 5px 12px; min-block-size: 27px; align-items: center; background: oklch(0.16 0.02 255); border-block-end: 1px solid var(--border-subtle); font-size: 0.69rem;"
  >
    {#if activeInspectBar}
      <span>
        {hoveredBar ? 'CROSSHAIR' : 'LATEST BAR'}:
        <strong>{new Date(activeInspectBar.timestampMs).toISOString().replace('T', ' ').slice(0, 19)} UTC</strong>
      </span>
      <span>O: <strong>{formatSubPennyPrice(activeInspectBar.open)}</strong></span>
      <span>H: <strong class="pnl-positive">{formatSubPennyPrice(activeInspectBar.high)}</strong></span>
      <span>L: <strong class="pnl-negative">{formatSubPennyPrice(activeInspectBar.low)}</strong></span>
      <span>C: <strong>{formatSubPennyPrice(activeInspectBar.close)}</strong></span>
      <span>VOL: <strong>{activeInspectBar.volume.toLocaleString('en-US')}</strong></span>
      <span class={hoveredBarLogReturnPct >= 0 ? 'pnl-positive' : 'pnl-negative'}>
        LOG RET: <strong>{hoveredBarLogReturnPct >= 0 ? '+' : ''}{hoveredBarLogReturnPct.toFixed(3)}%</strong>
      </span>
      <span style="color: var(--signal-ai-core);">
        ATTN WEIGHT: <strong>{(activeInspectBar.attentionWeight * 100).toFixed(1)}%</strong>
      </span>
    {:else}
      <span style="color: var(--text-secondary);">Hover chart candlesticks to inspect sub-millisecond OHLCV &amp; attention telemetry</span>
    {/if}
  </div>

  <!-- Pure Raw Price & Volume Canvas + AI Attention Heatmap Overlay + Trade Entry/Exit Markers -->
  <div style="padding: 10px 12px; background: var(--surface-0); position: relative;">
    <svg
      viewBox="0 0 {chartGeometry.width} {chartGeometry.height}"
      style="inline-size: 100%; block-size: auto; display: block; border-radius: var(--radius-sm); background: oklch(0.12 0.014 255); cursor: {isDragging ? 'grabbing' : 'crosshair'};"
      role="img"
      aria-label="Raw Yahoo Finance OHLCV Sequence and AI Structural Blueprint Overlay"
      onwheel={handleWheelZoom}
      onpointerdown={handlePointerDown}
      onpointermove={handlePointerMove}
      onpointerup={handlePointerUp}
      onpointerleave={() => {
        isDragging = false;
        hoveredBarIndex = null;
      }}
    >
      <!-- Horizontal Structural Blueprint Zones -->
      <line
        x1="0"
        y1={chartGeometry.t2Y}
        x2={chartGeometry.width}
        y2={chartGeometry.t2Y}
        stroke="oklch(0.76 0.19 152)"
        stroke-dasharray="5 4"
        stroke-width="1.5"
      />
      <text
        x="10"
        y={Math.max(14, chartGeometry.t2Y - 5)}
        fill="oklch(0.76 0.19 152)"
        font-size="10"
        font-family="monospace"
      >
        TARGET 2 (p80 MFE): {formatSubPennyPrice(blueprint.target2Price)} (+{blueprint.expectedRMultiple.toFixed(2)}R)
      </text>

      <line
        x1="0"
        y1={chartGeometry.t1Y}
        x2={chartGeometry.width}
        y2={chartGeometry.t1Y}
        stroke="oklch(0.76 0.19 152 / 0.75)"
        stroke-dasharray="3 3"
        stroke-width="1.2"
      />
      <text
        x="10"
        y={Math.max(14, chartGeometry.t1Y - 5)}
        fill="oklch(0.76 0.19 152)"
        font-size="10"
        font-family="monospace"
      >
        TARGET 1 (p50 MFE): {formatSubPennyPrice(blueprint.target1Price)}
      </text>

      <line
        x1="0"
        y1={chartGeometry.entryY}
        x2={chartGeometry.width}
        y2={chartGeometry.entryY}
        stroke="oklch(0.74 0.16 235)"
        stroke-width="1.5"
      />
      <text
        x="10"
        y={Math.max(14, chartGeometry.entryY - 5)}
        fill="oklch(0.74 0.16 235)"
        font-size="10"
        font-family="monospace"
      >
        ENTRY ZONE: {formatSubPennyPrice(blueprint.entryPrice)}
      </text>

      <line
        x1="0"
        y1={chartGeometry.stopY}
        x2={chartGeometry.width}
        y2={chartGeometry.stopY}
        stroke="oklch(0.68 0.22 22)"
        stroke-dasharray="4 3"
        stroke-width="1.5"
      />
      <text
        x="10"
        y={Math.min(chartGeometry.height - 54, chartGeometry.stopY + 12)}
        fill="oklch(0.68 0.22 22)"
        font-size="10"
        font-family="monospace"
      >
        AI STRUCTURAL STOP (p85 MAE): {formatSubPennyPrice(blueprint.structuralStopPrice)}
      </text>

      {#if replayLifecycle}
        <line
          x1="0"
          y1={chartGeometry.activeReplayStopY}
          x2={chartGeometry.width}
          y2={chartGeometry.activeReplayStopY}
          stroke="oklch(0.84 0.17 85)"
          stroke-dasharray="6 2"
          stroke-width="2"
        />
        <text
          x={chartGeometry.width - 340}
          y={Math.max(14, chartGeometry.activeReplayStopY - 6)}
          fill="oklch(0.84 0.17 85)"
          font-size="10"
          font-weight="bold"
          font-family="monospace"
        >
          DYNAMIC STOP [{replayLifecycle.stopRatchetStage}]: {formatSubPennyPrice(replayLifecycle.activeStopPrice)}
        </text>
      {/if}

      <!-- Raw Yahoo Finance Candles + Transformer Attention Glow + Raw Volume Profile -->
      {#each windowedBars as bar, idx (bar.timestampMs + '-' + idx)}
        {@const stepX = (chartGeometry.width - 140) / Math.max(1, windowedBars.length)}
        {@const xCenter = 60 + idx * stepX}
        {@const openY = chartGeometry.priceToY(bar.open)}
        {@const closeY = chartGeometry.priceToY(bar.close)}
        {@const highY = chartGeometry.priceToY(bar.high)}
        {@const lowY = chartGeometry.priceToY(bar.low)}
        {@const isBull = bar.close >= bar.open}
        {@const color = isBull ? 'oklch(0.76 0.19 152)' : 'oklch(0.68 0.22 22)'}
        {@const volBarH = (bar.volume / chartGeometry.maxVol) * 44}

        <!-- Per-Bar AI Attention Heatmap Highlight for Anomaly Candles -->
        {#if bar.attentionWeight > 0.72}
          <rect
            x={xCenter - stepX * 0.44}
            y={highY - 4}
            width={stepX * 0.88}
            height={Math.max(12, lowY - highY + 8)}
            fill="oklch(0.78 0.17 295 / {bar.attentionWeight * 0.22})"
            rx="2"
          />
        {/if}

        <!-- Raw High-Low Wick -->
        <line x1={xCenter} y1={highY} x2={xCenter} y2={lowY} stroke={color} stroke-width="1.4" />

        <!-- Raw Open-Close Body -->
        <rect
          x={xCenter - stepX * 0.32}
          y={Math.min(openY, closeY)}
          width={Math.max(2.5, stepX * 0.64)}
          height={Math.max(2, Math.abs(closeY - openY))}
          fill={color}
          rx="1"
        />

        <!-- Raw Volume Bar -->
        <rect
          x={xCenter - stepX * 0.3}
          y={chartGeometry.height - 6 - volBarH}
          width={Math.max(2.5, stepX * 0.6)}
          height={volBarH}
          fill={bar.attentionWeight > 0.72 ? 'oklch(0.78 0.17 295 / 0.78)' : color}
          opacity="0.55"
        />

        {#if replayLifecycle}
          {#if idx === replayLifecycle.entryIdx}
            <line
              x1={xCenter}
              y1="20"
              x2={xCenter}
              y2={chartGeometry.height - 55}
              stroke="oklch(0.74 0.16 235 / 0.75)"
              stroke-dasharray="2 2"
              stroke-width="1.3"
            />
            <circle cx={xCenter} cy={chartGeometry.entryY} r="4.5" fill="oklch(0.74 0.16 235)" />
            <text
              x={Math.max(8, xCenter - 26)}
              y="16"
              fill="oklch(0.84 0.14 235)"
              font-size="9"
              font-weight="bold"
              font-family="monospace"
            >
              ENTRY #{idx + 1}
            </text>
          {/if}
          {#if replayLifecycle.scaleInBarIdx !== null && idx === replayLifecycle.scaleInBarIdx}
            <circle cx={xCenter} cy={highY - 8} r="4" fill="oklch(0.82 0.17 295)" />
            <text
              x={Math.max(8, xCenter - 28)}
              y={Math.max(26, highY - 13)}
              fill="oklch(0.86 0.15 295)"
              font-size="8.5"
              font-weight="bold"
              font-family="monospace"
            >
              SCALE-IN ({replayLifecycle.trade.trancheCount}x)
            </text>
          {/if}
          {#if replayLifecycle.t1HitBarIdx !== null && idx === replayLifecycle.t1HitBarIdx}
            <circle cx={xCenter} cy={chartGeometry.t1Y} r="4.5" fill="oklch(0.84 0.18 85)" />
            <text
              x={Math.max(8, xCenter - 34)}
              y={Math.max(26, chartGeometry.t1Y - 8)}
              fill="oklch(0.86 0.18 85)"
              font-size="8.5"
              font-weight="bold"
              font-family="monospace"
            >
              BANKED 50% @ T1
            </text>
          {/if}
          {#if idx === replayLifecycle.exitIdx}
            <circle
              cx={xCenter}
              cy={closeY}
              r="5"
              fill={replayLifecycle.trade.isWinner ? 'oklch(0.80 0.20 152)' : 'oklch(0.68 0.22 22)'}
            />
            <text
              x={Math.max(8, Math.min(chartGeometry.width - 140, xCenter - 38))}
              y={Math.max(28, closeY - 9)}
              fill={replayLifecycle.trade.isWinner ? 'oklch(0.80 0.20 152)' : 'oklch(0.68 0.22 22)'}
              font-size="9"
              font-weight="bold"
              font-family="monospace"
            >
              EXIT ({replayLifecycle.trade.outcomeBadge})
            </text>
          {/if}
        {/if}
      {/each}

      <!-- Historical Closed Trade Entry & Exit Markers on Real Yahoo Chart -->
      {#each symbolClosedTrades.slice(0, 4) as trade, tIdx (trade.tradeId)}
        {@const entryMarkerX = Math.max(85, chartGeometry.width - 280 - tIdx * 95)}
        {@const exitMarkerX = Math.min(chartGeometry.width - 95, entryMarkerX + 75)}
        {@const entryMarkerY = chartGeometry.priceToY(trade.entryPrice)}
        {@const exitMarkerY = chartGeometry.priceToY(trade.exitPrice)}
        {@const tradeColor = trade.isWinner ? 'oklch(0.80 0.20 152)' : 'oklch(0.68 0.22 22)'}

        <line
          x1={entryMarkerX}
          y1={entryMarkerY}
          x2={exitMarkerX}
          y2={exitMarkerY}
          stroke={tradeColor}
          stroke-dasharray="2 2"
          stroke-width="1.5"
        />
        <circle cx={entryMarkerX} cy={entryMarkerY} r="4.5" fill="oklch(0.74 0.16 235)" />
        <text
          x={entryMarkerX - 24}
          y={Math.max(14, entryMarkerY - 7)}
          fill="oklch(0.85 0.12 235)"
          font-size="9"
          font-family="monospace"
        >
          IN {formatSubPennyPrice(trade.entryPrice)}
        </text>

        <circle cx={exitMarkerX} cy={exitMarkerY} r="5" fill={tradeColor} />
        <text
          x={exitMarkerX - 30}
          y={Math.max(14, exitMarkerY - 8)}
          fill={tradeColor}
          font-size="9"
          font-weight="bold"
          font-family="monospace"
        >
          {trade.outcomeBadge} ({formatSignedUsd(trade.combinedTotalPnlUsd)} | {trade.realizedRMultiple >= 0 ? '+' : ''}{trade.realizedRMultiple.toFixed(1)}R)
        </text>
      {/each}

      <!-- Interactive Crosshair Lines -->
      {#if hoveredBarIndex !== null}
        <line
          x1={hoverMouseX}
          y1="0"
          x2={hoverMouseX}
          y2={chartGeometry.height}
          stroke="oklch(0.85 0.05 250 / 0.45)"
          stroke-dasharray="3 3"
          stroke-width="1"
        />
        <line
          x1="0"
          y1={hoverMouseY}
          x2={chartGeometry.width}
          y2={hoverMouseY}
          stroke="oklch(0.85 0.05 250 / 0.45)"
          stroke-dasharray="3 3"
          stroke-width="1"
        />
      {/if}

      <!-- Predicted Future Expansion Corridor -->
      <polygon
        points="{chartGeometry.width - 80},{chartGeometry.entryY} {chartGeometry.width - 8},{chartGeometry.t2Y} {chartGeometry.width - 8},{chartGeometry.t1Y}"
        fill="oklch(0.76 0.19 152 / 0.18)"
        stroke="oklch(0.76 0.19 152 / 0.65)"
        stroke-width="1"
      />
    </svg>
  </div>
</section>
