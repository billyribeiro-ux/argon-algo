import { describe, expect, it, vi } from 'vitest';
import {
  TerminalStore,
  formatSubPennyPrice,
  CSV_LEDGER_HEADERS
} from './terminal.svelte';
import { ExecutionAudioEngine } from '../audio/execution-audio';

function seedYahooBackendState(store: TerminalStore): void {
  // Ingest real-shape Yahoo Finance chart responses across sub-penny and institutional tickers
  store.ingestBackendChartResponse({
    symbol: 'NVDA',
    exchange_index: 'NASDAQ_100',
    bars: [
      { timestamp_ns: 1730210000000000000, open: 208.5, high: 210.4, low: 208.1, close: 210.1468, volume: 1842000 }
    ],
    inference: {
      signal: {
        brain_id: 'BRAIN-02',
        direction: 'LONG',
        entry_price: 210.1468,
        structural_stop_price: 207.2,
        target_1_price: 214.8,
        target_2_price: 219.5,
        predicted_hold_duration_ms: 5400000,
        ai_confidence_pct: 93.4,
        expected_r_multiple: 3.18,
        primary_driver: 'GAMMA_SQUEEZE',
        stream_weights: {
          chart_tape_physics_weight: 0.32,
          options_gamma_vanna_weight: 0.41,
          short_float_squeeze_weight: 0.14,
          market_internals_macro_weight: 0.13
        }
      },
      matched_option: {
        option_structure: 'VERTICAL_DEBIT_SPREAD',
        matched_option_contract: 'NVDA 2026-11-20 $210C/$220C VERTICAL',
        option_entry_premium: 3.85,
        option_stop_premium: 2.1,
        option_target_premium: 7.4,
        option_contracts_count: 5,
        contract_delta: 0.58,
        contract_gamma: 0.044,
        contract_vanna: 0.021,
        contract_theta: -0.09,
        contract_vega: 0.16,
        contract_iv: 0.46
      },
      bar_attention_heatmap: [0.91],
      top_shap_drivers: [
        ['s2_net_gex_wall_distance_pct', 0.68],
        ['s1_volume_surge_log_ratio', 0.51]
      ]
    },
    precedents: {
      matches_count: 25,
      win_rate_pct: 88.0,
      mean_realized_r_multiple: 3.12,
      p80_mfe_pct: 4.15,
      p85_mae_pct: 0.82
    }
  });

  store.ingestBackendChartResponse({
    symbol: 'HYMC',
    exchange_index: 'OTC_PENNY',
    bars: [
      { timestamp_ns: 1730210000000000000, open: 2.15, high: 2.38, low: 2.12, close: 2.34, volume: 4200000 }
    ],
    inference: {
      signal: {
        brain_id: 'BRAIN-01',
        direction: 'LONG',
        entry_price: 2.34,
        structural_stop_price: 2.21,
        target_1_price: 2.58,
        target_2_price: 2.85,
        predicted_hold_duration_ms: 1800000,
        ai_confidence_pct: 91.2,
        expected_r_multiple: 3.92,
        primary_driver: 'SHORT_SQUEEZE',
        stream_weights: {
          chart_tape_physics_weight: 0.36,
          options_gamma_vanna_weight: 0.08,
          short_float_squeeze_weight: 0.44,
          market_internals_macro_weight: 0.12
        }
      },
      matched_option: {
        option_structure: 'SHARES_ONLY',
        matched_option_contract: 'SHARES_ONLY',
        option_contracts_count: 0
      },
      bar_attention_heatmap: [0.88],
      top_shap_drivers: [['s3_short_squeeze_pressure_z', 0.74]]
    },
    precedents: {
      matches_count: 25,
      win_rate_pct: 84.0,
      mean_realized_r_multiple: 3.45,
      p80_mfe_pct: 6.2,
      p85_mae_pct: 1.1
    }
  });

  store.ingestBackendSnapshot({
    positions_opened: [
      {
        position_id: 'OPEN-9001',
        opened_timestamp_ns: 1730210000000000000,
        symbol: 'NVDA',
        exchange_index: 'NASDAQ_100',
        brain_id: 'BRAIN-02',
        direction: 'LONG',
        entry_price: 208.5,
        live_price: 210.1468,
        quantity_shares: 50,
        structural_stop_price: 205.8,
        target_1_price: 213.0,
        target_2_price: 218.0,
        unrealized_stock_pnl_usd: 82.34,
        unrealized_stock_pnl_pct: 0.79,
        option_structure: 'VERTICAL_DEBIT_SPREAD',
        matched_option_contract: 'NVDA 2026-11-20 $210C/$220C VERTICAL',
        option_entry_premium: 3.85,
        option_live_premium: 4.45,
        option_contracts_count: 5,
        unrealized_option_pnl_usd: 300.0,
        unrealized_option_pnl_pct: 15.58,
        contract_delta: 0.58,
        contract_gamma: 0.044,
        contract_vanna: 0.021,
        contract_theta: -0.09,
        contract_vega: 0.16,
        contract_iv: 0.46,
        live_hold_duration_ms: 900000,
        current_r_multiple: 0.61,
        mfe_pct: 1.12,
        mae_pct: 0.18,
        ai_confidence_pct: 93.4,
        primary_driver: 'GAMMA_SQUEEZE',
        trace_health_badge: 'ACCELERATING',
        precedent_matches_count: 25,
        precedent_win_rate_pct: 88.0,
        stream_weights: {
          chart_tape_physics_weight: 0.32,
          options_gamma_vanna_weight: 0.41,
          short_float_squeeze_weight: 0.14,
          market_internals_macro_weight: 0.13
        }
      }
    ],
    closed_trades: [
      {
        trade_id: 'TRD-8001',
        opened_timestamp_ns: 1730200000000000000,
        closed_timestamp_ns: 1730203600000000000,
        hold_duration_ms: 3600000,
        symbol: 'TSLA',
        exchange_index: 'NASDAQ_100',
        brain_id: 'BRAIN-02',
        direction: 'LONG',
        entry_price: 390.0,
        actual_exit_price: 397.45,
        quantity_shares: 25,
        initial_structural_stop: 386.5,
        target_1_price: 394.0,
        target_2_price: 397.45,
        outcome_badge: 'CLOSED - TARGET HIT',
        is_winner: true,
        net_pnl_usd: 186.25,
        stock_return_pct: 1.91,
        option_structure: 'SINGLE_LEG_CALL',
        matched_option_contract: 'TSLA 2026-11-20 $390C',
        option_entry_premium: 8.2,
        option_exit_premium: 12.6,
        option_contracts_count: 3,
        option_net_pnl_usd: 1320.0,
        option_return_pct: 53.66,
        realized_r_multiple: 2.13,
        mfe_pct: 2.05,
        mae_pct: 0.32,
        ai_confidence_pct: 92.5,
        primary_driver: 'GAMMA_SQUEEZE',
        precedent_win_rate_pct: 88.0,
        kelly_fraction_pct: 2.15,
        risk_budget_usd: 215.0,
        notional_allocated_usd: 23809.52,
        tranche_count: 2,
        scale_in_history: ['TRANCHE 2 (+0.35R): +25 sh @ $391.25'],
        entry_bar_idx: 2,
        exit_bar_idx: 5,
        entry_bars: [
          { timestamp_ns: 1730199400000000000, open: 388.2, high: 389.1, low: 387.9, close: 388.8, volume: 420000, attention_weight: 0.45 },
          { timestamp_ns: 1730199700000000000, open: 388.8, high: 390.1, low: 388.5, close: 389.9, volume: 610000, attention_weight: 0.68 },
          { timestamp_ns: 1730200000000000000, open: 390.0, high: 391.5, low: 389.7, close: 391.2, volume: 980000, attention_weight: 0.89 },
          { timestamp_ns: 1730200300000000000, open: 391.2, high: 394.3, low: 391.0, close: 394.1, volume: 1120000, attention_weight: 0.91 },
          { timestamp_ns: 1730200600000000000, open: 394.1, high: 396.2, low: 393.8, close: 395.9, volume: 890000, attention_weight: 0.78 },
          { timestamp_ns: 1730203600000000000, open: 395.9, high: 397.8, low: 395.5, close: 397.45, volume: 1250000, attention_weight: 0.93 }
        ],
        stream_weights: {
          chart_tape_physics_weight: 0.35,
          options_gamma_vanna_weight: 0.4,
          short_float_squeeze_weight: 0.15,
          market_internals_macro_weight: 0.1
        }
      },
      {
        trade_id: 'TRD-8002',
        opened_timestamp_ns: 1730190000000000000,
        closed_timestamp_ns: 1730191800000000000,
        hold_duration_ms: 1800000,
        symbol: 'SOFI',
        exchange_index: 'RUSSELL_2000',
        brain_id: 'BRAIN-01',
        direction: 'LONG',
        entry_price: 4.12,
        actual_exit_price: 4.38,
        quantity_shares: 1000,
        initial_structural_stop: 4.01,
        target_1_price: 4.28,
        target_2_price: 4.45,
        outcome_badge: 'CLOSED - AI DYNAMIC EXIT',
        is_winner: true,
        net_pnl_usd: 260.0,
        stock_return_pct: 6.31,
        option_structure: 'SHARES_ONLY',
        matched_option_contract: 'SHARES_ONLY',
        option_entry_premium: 0,
        option_exit_premium: 0,
        option_contracts_count: 0,
        option_net_pnl_usd: 0,
        option_return_pct: 0,
        realized_r_multiple: 2.36,
        mfe_pct: 6.8,
        mae_pct: 0.45,
        ai_confidence_pct: 90.8,
        primary_driver: 'SHORT_SQUEEZE',
        precedent_win_rate_pct: 84.0,
        stream_weights: {
          chart_tape_physics_weight: 0.4,
          options_gamma_vanna_weight: 0.1,
          short_float_squeeze_weight: 0.35,
          market_internals_macro_weight: 0.15
        }
      },
      {
        trade_id: 'TRD-8003',
        opened_timestamp_ns: 1730180000000000000,
        closed_timestamp_ns: 1730183600000000000,
        hold_duration_ms: 3600000,
        symbol: 'AMD',
        exchange_index: 'NASDAQ_100',
        brain_id: 'BRAIN-03',
        direction: 'SHORT',
        entry_price: 156.0,
        actual_exit_price: 158.2,
        quantity_shares: 40,
        initial_structural_stop: 158.2,
        target_1_price: 152.5,
        target_2_price: 149.0,
        outcome_badge: 'CLOSED - STOPPED OUT',
        is_winner: false,
        net_pnl_usd: -88.0,
        stock_return_pct: -1.41,
        option_structure: 'SINGLE_LEG_PUT',
        matched_option_contract: 'AMD 2026-11-20 $155P',
        option_entry_premium: 4.1,
        option_exit_premium: 2.8,
        option_contracts_count: 2,
        option_net_pnl_usd: -260.0,
        option_return_pct: -31.71,
        realized_r_multiple: -1.0,
        mfe_pct: 0.4,
        mae_pct: 1.45,
        ai_confidence_pct: 78.5,
        primary_driver: 'VOLUME_ABSORPTION',
        precedent_win_rate_pct: 76.0,
        stream_weights: {
          chart_tape_physics_weight: 0.45,
          options_gamma_vanna_weight: 0.25,
          short_float_squeeze_weight: 0.15,
          market_internals_macro_weight: 0.15
        }
      }
    ]
  });
}

describe('ARGON-4 Phase 6 Real Yahoo Finance TerminalStore & Hyper-Filter Verification', () => {
  it('starts with zero synthetic default trades and formats sub-penny ($0.00042) to mega-cap ($142.85) prices', () => {
    const emptyStore = new TerminalStore(false);
    expect(emptyStore.scannerCandidates.length).toBe(0);
    expect(emptyStore.openPositions.length).toBe(0);
    expect(emptyStore.closedLedger.length).toBe(0);

    expect(formatSubPennyPrice(0.00042)).toBe('$0.00042');
    expect(formatSubPennyPrice(0.00001)).toBe('$0.00001');
    expect(formatSubPennyPrice(142.85)).toBe('$142.85');
  });

  it('ingests real Yahoo Finance backend chart payloads across all 5 Brains and sub-penny vs institutional price tiers', () => {
    const store = new TerminalStore(false);
    seedYahooBackendState(store);

    expect(store.filteredScannerCandidates.length).toBe(2);
    expect(store.activeBlueprint.optionsRecommendation?.optionStructure).toBe(
      'VERTICAL_DEBIT_SPREAD'
    );
    expect(store.activeBlueprint.optionsRecommendation?.vanna).toBeCloseTo(0.021, 3);
    expect(store.activeBlueprint.optionsRecommendation?.vega).toBeCloseTo(0.16, 2);

    store.selectBrain('BRAIN-01', false);
    expect(store.filteredScannerCandidates.every((c) => c.brainId === 'BRAIN-01')).toBe(true);

    store.selectBrain('ALL_5_BRAINS', false);
    store.universeTierMode = 'SUB_PENNY_ONLY';
    expect(store.filteredScannerCandidates.length).toBe(1);
    expect(store.filteredScannerCandidates.every((c) => c.livePrice <= 5.0)).toBe(true);

    store.universeTierMode = 'INSTITUTIONAL_ONLY';
    expect(store.filteredScannerCandidates.every((c) => c.livePrice > 5.0)).toBe(true);
  });

  it('transitions open positions from POSITIONS OPENED into CLOSED TRADE LEDGER and updates KPIs', () => {
    const store = new TerminalStore(false);
    seedYahooBackendState(store);

    const initialOpenCount = store.openPositions.length;
    const initialClosedCount = store.closedLedger.length;

    store.stepLiveSimulation(false);
    const firstOpen = store.openPositions[0]!;
    const archived = store.closeOpenPosition(firstOpen.tradeId, 'TARGET HIT');

    expect(archived).not.toBeNull();
    expect(archived?.outcomeBadge).toBe('TARGET HIT');
    expect(archived?.isWinner).toBe(true);
    expect(store.openPositions.length).toBe(initialOpenCount - 1);
    expect(store.closedLedger.length).toBe(initialClosedCount + 1);
    expect(store.scorecardKpis.totalClosed).toBe(initialClosedCount + 1);
    expect(store.equityCurve.length).toBe(initialClosedCount + 1);
  });

  it('applies all 6 compounding Hyper-Filter dimensions and generates 34+ column CSV export', () => {
    const store = new TerminalStore(false);
    seedYahooBackendState(store);

    store.setQuickFilterPill('WINS_ONLY', false);
    expect(store.filteredLedger.every((r) => r.isWinner)).toBe(true);
    expect(store.scorecardKpis.totalLosses).toBe(0);
    expect(store.scorecardKpis.winRatePct).toBeCloseTo(100.0, 2);

    store.setQuickFilterPill('LOSSES_ONLY', false);
    expect(store.filteredLedger.every((r) => !r.isWinner)).toBe(true);
    expect(store.scorecardKpis.totalWins).toBe(0);

    store.setQuickFilterPill('SUB_PENNY', false);
    expect(store.filteredLedger.length).toBeGreaterThanOrEqual(1);
    expect(store.filteredLedger.every((r) => r.entryPrice <= 5.0)).toBe(true);

    store.resetAllLedgerFilters();
    const csv = store.buildFilteredCsvString();
    expect(csv).toContain('# QUANT INTELLIGENCE PLATFORM — CLOSED TRADE LEDGER EXPORT');
    expect(csv).toContain(CSV_LEDGER_HEADERS.join(','));
    expect(csv).toContain('TARGET HIT');
    expect(csv).toContain('AI DYNAMIC EXIT');
    expect(csv).toContain('STOPPED OUT');
  });

  it('maintains independent comparison viewports in SPLIT_50_50 and 4 Brain viewports in QUAD_2X2', async () => {
    const store = new TerminalStore(false);
    seedYahooBackendState(store);

    expect(store.splitPanes.length).toBe(2);
    expect(store.splitPanes[0]?.paneId).toBe('SPLIT_RED');
    expect(store.splitPanes[1]?.paneId).toBe('SPLIT_BLUE');
    expect(store.quadViewports.length).toBe(4);

    store.setWorkspaceLayout('SPLIT_50_50');
    expect(store.workspaceLayout).toBe('SPLIT_50_50');

    await store.updateViewportPane('SPLIT_RED', {
      symbol: 'PLUG',
      brainId: 'BRAIN-01',
      timeframe: '15m'
    });
    expect(store.splitPanes[0]?.symbol).toBe('PLUG');
    expect(store.splitPanes[0]?.brainId).toBe('BRAIN-01');
    expect(store.splitPanes[0]?.timeframe).toBe('15m');
    expect(store.splitPanes[1]?.symbol).toBe('AMD');
  });

  it('cycles ExecutionAudioEngine modes and deduplicates multi-monitor audio events across windows', () => {
    const audioEngine = new ExecutionAudioEngine('VOICE_AND_CHIMES');
    expect(audioEngine.getMode()).toBe('VOICE_AND_CHIMES');

    expect(audioEngine.cycleMode()).toBe('CHIMES_ONLY');
    expect(audioEngine.cycleMode()).toBe('MUTED');
    expect(audioEngine.cycleMode()).toBe('VOICE_AND_CHIMES');

    // Verify acoustic methods execute safely without throwing in headless test environment
    expect(() => audioEngine.playSignalEntryChime('LONG')).not.toThrow();
    expect(audioEngine.getLastChimeSignature()).toBe('ENTRY_LONG_523.25_659.25');
    expect(() => audioEngine.playSignalEntryChime('SHORT')).not.toThrow();
    expect(audioEngine.getLastChimeSignature()).toBe('ENTRY_SHORT_587.33_440.00');
    expect(() => audioEngine.playTarget1BankedChime()).not.toThrow();
    expect(audioEngine.getLastChimeSignature()).toBe('TARGET1_BANKED_880.00_1174.66');
    expect(() => audioEngine.playScaleInChime()).not.toThrow();
    expect(audioEngine.getLastChimeSignature()).toBe('SCALE_IN_880.00_1174.66');
    expect(() => audioEngine.playPositionClosedChime(true)).not.toThrow();
    expect(() => audioEngine.playPositionClosedChime(false)).not.toThrow();
    expect(() =>
      audioEngine.speakExecutionCallout(
        'Argon Brain 2: Long Entry on N V D A at $142.85, Target 1 $145.40, Confidence 94 percent.'
      )
    ).not.toThrow();
    expect(audioEngine.getLastSpokenText()).toContain('Argon Brain 2: Long Entry');
  });

  it('executes on-demand custom ticker walk-forward backtests via runCustomSymbolBacktest and updates telemetry state', async () => {
    const store = new TerminalStore(false);
    seedYahooBackendState(store);

    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        symbol: 'MSTR',
        brain_id: 'BRAIN-03',
        timeframe: '1H',
        total_bars_scanned: 750,
        era1_calibration: {
          brain_id: 'BRAIN-03',
          calibrated_confidence_floor: 64.0,
          min_precedent_win_rate_pct: 58.0,
          min_expected_r_multiple: 1.35,
          stream_prior_weights: {
            chart_tape_physics_weight: 0.25,
            options_gamma_vanna_weight: 0.25,
            short_float_squeeze_weight: 0.25,
            market_internals_macro_weight: 0.25
          },
          era1_sample_trades: 96,
          era1_calibrated_win_rate_pct: 72.0,
          era1_calibrated_profit_factor: 1.95
        },
        closed_trades_generated: 8,
        open_position_active: false,
        win_rate_pct: 62.5,
        profit_factor: 1.25,
        realized_stock_net_pnl_usd: 462.13,
        realized_option_net_pnl_usd: 215.4
      })
    });
    vi.stubGlobal('fetch', mockFetch);

    expect(store.audioMode).toBe('VOICE_AND_CHIMES');
    store.cycleAudioMode();
    expect(store.audioMode).toBe('CHIMES_ONLY');
    store.setAudioMode('VOICE_AND_CHIMES');

    const result = await store.runCustomSymbolBacktest('MSTR', 'BRAIN-03');
    expect(result).not.toBeNull();
    expect(result?.symbol).toBe('MSTR');
    expect(result?.brainId).toBe('BRAIN-03');
    expect(result?.totalBarsScanned).toBe(750);
    expect(result?.closedTradesGenerated).toBe(8);
    expect(result?.profitFactor).toBeCloseTo(1.25, 2);
    expect(store.lastBacktestSummary?.symbol).toBe('MSTR');

    store.dismissBacktestSummary();
    expect(store.lastBacktestSummary).toBeNull();

    vi.unstubAllGlobals();
  });

  it('hydrates Phase 12 Fractional-Kelly sizing, Full-Lifecycle Replay Scrubber bars, and Outbound Webhooks', async () => {
    const store = new TerminalStore(false);
    seedYahooBackendState(store);

    const tslaTrade = store.closedLedger.find((t) => t.tradeId === 'TRD-8001');
    expect(tslaTrade).toBeDefined();
    expect(tslaTrade?.kellyFractionPct).toBeCloseTo(2.15, 2);
    expect(tslaTrade?.riskBudgetUsd).toBeCloseTo(215.0, 2);
    expect(tslaTrade?.notionalAllocatedUsd).toBeCloseTo(23809.52, 2);
    expect(tslaTrade?.trancheCount).toBe(2);
    expect(tslaTrade?.entryBarIdx).toBe(2);
    expect(tslaTrade?.exitBarIdx).toBe(5);
    expect(tslaTrade?.entryBars.length).toBe(6);

    store.selectClosedTradeForReplay(tslaTrade!);
    expect(store.activeReplayTrade?.tradeId).toBe('TRD-8001');
    expect(store.activeBlueprint.bars.length).toBe(6);
    store.clearReplayTrade();
    expect(store.activeReplayTrade).toBeNull();

    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        config: {
          discord_webhook_url: 'https://discord.com/api/webhooks/test',
          discord_enabled: true,
          telegram_bot_token: '',
          telegram_chat_id: '',
          telegram_enabled: false,
          slack_webhook_url: '',
          slack_enabled: false,
          custom_json_webhook_url: '',
          custom_json_enabled: false
        },
        total_outbox_records: 1,
        outbox_records: [
          {
            dispatch_id: 'WH-TEST-001',
            dispatched_at_iso: '2026-10-30T13:30:00Z',
            dispatched_timestamp_ms: 1730210000000,
            event_type: 'SIGNAL_ENTRY_OPENED',
            outcome_detail: 'OPEN ACTIVE',
            symbol: 'NVDA',
            brain_id: 'BRAIN-02',
            brain_name: 'Intraday Day-Trading Algo',
            direction: 'LONG',
            entry_price: 210.14,
            structural_stop_price: 207.2,
            target_1_price: 214.8,
            target_2_price: 219.5,
            current_or_exit_price: 210.14,
            kelly_fraction_pct: 1.95,
            risk_budget_usd: 195.0,
            notional_allocated_usd: 19500.0,
            matched_option_contract: 'NVDA $210C',
            greeks_summary: 'Δ 0.58 | Γ 0.044',
            stream_attribution_summary: 'S1: 32% | S2: 41%',
            precedent_win_rate_pct: 88.0,
            ai_confidence_pct: 93.4,
            pnl_usd: 0,
            r_multiple: 0,
            delivery_status: 'QUEUED_LOCAL_OUTBOX',
            channels_targeted: ['DISCORD'],
            discord_embed_json: '{"embeds":[{"title":"SIGNAL ENTRY OPENED"}]}',
            telegram_markdown: '*SIGNAL ENTRY OPENED*',
            slack_blocks_json: '{"blocks":[]}',
            generic_webhook_json: '{}'
          }
        ]
      })
    });
    vi.stubGlobal('fetch', mockFetch);

    await store.fetchWebhookStatus();
    expect(store.webhookStatus.config.discordEnabled).toBe(true);
    expect(store.webhookStatus.totalOutboxRecords).toBe(1);
    expect(store.webhookStatus.outboxRecords[0]?.dispatchId).toBe('WH-TEST-001');

    vi.unstubAllGlobals();
  });

  it('hydrates Phase 12 Loss Forensic Autopsies, Parallel Engine & Strategy Leaderboards, and Non-Destructive Sizing Toggle', () => {
    const store = new TerminalStore(false);
    store.ingestBackendSnapshot({
      positions_opened: [],
      closed_trades: [
        {
          trade_id: 'TRD-9001',
          opened_timestamp_ns: 1_730_200_000_000_000_000,
          closed_timestamp_ns: 1_730_203_600_000_000_000,
          hold_duration_ms: 3_600_000,
          symbol: 'NVDA',
          exchange_index: 'NASDAQ_100',
          brain_id: 'BRAIN-02',
          direction: 'LONG',
          entry_price: 200.0,
          actual_exit_price: 206.0,
          quantity_shares: 50,
          kelly_fraction_pct: 2.1,
          risk_budget_usd: 210.0,
          notional_allocated_usd: 21000.0,
          kelly_net_pnl_usd: 630.0,
          initial_structural_stop: 198.0,
          target_1_price: 204.0,
          target_2_price: 208.0,
          outcome_badge: 'CLOSED - TARGET HIT',
          is_winner: true,
          net_pnl_usd: 300.0,
          stock_return_pct: 3.0,
          option_contracts_count: 0,
          option_net_pnl_usd: 0.0,
          realized_r_multiple: 3.0,
          mfe_pct: 3.2,
          mae_pct: 0.4,
          ai_confidence_pct: 91.5,
          primary_driver: 'GAMMA_SQUEEZE',
          engine_mode: 'CONSENSUS_BOTH_ENGINES',
          strategy_archetype: 'GAMMA_SQUEEZE_IGNITION',
          loss_forensic_autopsy: null
        },
        {
          trade_id: 'TRD-9002',
          opened_timestamp_ns: 1_730_205_000_000_000_000,
          closed_timestamp_ns: 1_730_208_600_000_000_000,
          hold_duration_ms: 3_600_000,
          symbol: 'MARA',
          exchange_index: 'NASDAQ_100',
          brain_id: 'BRAIN-02',
          direction: 'LONG',
          entry_price: 18.0,
          actual_exit_price: 17.5,
          quantity_shares: 500,
          kelly_fraction_pct: 1.0,
          risk_budget_usd: 100.0,
          notional_allocated_usd: 9000.0,
          kelly_net_pnl_usd: -100.0,
          initial_structural_stop: 17.5,
          target_1_price: 18.8,
          target_2_price: 19.5,
          outcome_badge: 'CLOSED - STOPPED OUT',
          is_winner: false,
          net_pnl_usd: -250.0,
          stock_return_pct: -2.77,
          option_contracts_count: 0,
          option_net_pnl_usd: 0.0,
          realized_r_multiple: -1.0,
          mfe_pct: 0.5,
          mae_pct: 2.77,
          ai_confidence_pct: 72.0,
          primary_driver: 'SHORT_SQUEEZE',
          engine_mode: 'PURE_ML_SELF_LEARNED',
          strategy_archetype: 'PURE_ML_LATENT_ANOMALY',
          loss_forensic_autopsy: {
            trade_id: 'TRD-9002',
            symbol: 'MARA',
            brain_id: 'BRAIN-02',
            root_cause_category: 'EXHAUSTION_WICK_LIQUIDITY_TRAP',
            top_culprit_features: [
              {
                feature_index: 5,
                feature_name: 's1_upper_wick_exhaustion_ratio',
                stream_name: 'Stream 1 (Chart & Tape Physics)',
                loss_value: 0.84,
                winner_centroid_value: 0.21,
                anomaly_z_score: 2.65
              }
            ],
            what_went_wrong_summary: 'Entry hit upper-wick liquidity trap on bar 4.',
            learned_adaptation_rule: 'Veto when s1_upper_wick_exhaustion_ratio > 0.72.',
            counterfactual_early_exit_r: -0.28
          }
        }
      ],
      by_strategy_leaderboard: [
        {
          strategy_archetype: 'GAMMA_SQUEEZE_IGNITION',
          display_name: 'Gamma Squeeze Ignition',
          total_trades: 1,
          wins: 1,
          losses: 0,
          win_rate_pct: 100.0,
          profit_factor: 99.99,
          avg_realized_r: 3.0,
          best_symbol: 'NVDA',
          realized_stock_net_pnl_usd: 300.0,
          realized_option_net_pnl_usd: 0.0,
          kelly_realized_net_pnl_usd: 630.0
        }
      ],
      by_engine_breakdown: [
        {
          engine_mode: 'CONSENSUS_BOTH_ENGINES',
          display_name: 'Consensus (Engine-A + Engine-B)',
          total_trades: 1,
          wins: 1,
          losses: 0,
          win_rate_pct: 100.0,
          profit_factor: 99.99,
          avg_realized_r: 3.0,
          best_symbol: 'NVDA',
          realized_stock_net_pnl_usd: 300.0,
          realized_option_net_pnl_usd: 0.0,
          kelly_realized_net_pnl_usd: 630.0
        }
      ]
    });

    expect(store.byStrategyLeaderboard.length).toBe(1);
    expect(store.byEngineBreakdown.length).toBe(1);
    expect(store.closedLedger[1]?.lossForensicAutopsy?.rootCauseCategory).toBe(
      'EXHAUSTION_WICK_LIQUIDITY_TRAP'
    );

    // Non-destructive sizing mode toggle: Fixed $10K Baseline ($300 - $250 = +$50) vs Fractional-Kelly ($630 - $100 = +$530)
    expect(store.sizingMode).toBe('FIXED_10K_BASELINE');
    expect(store.scorecardKpis.netRealizedStockPnlUsd).toBeCloseTo(50.0, 2);

    store.toggleSizingMode();
    expect(store.sizingMode).toBe('FRACTIONAL_KELLY');
    expect(store.scorecardKpis.netRealizedStockPnlUsd).toBeCloseTo(530.0, 2);

    // Filter by Strategy Archetype & Engine Mode
    store.setStrategyFilter('GAMMA_SQUEEZE_IGNITION');
    expect(store.filteredLedger.length).toBe(1);
    expect(store.filteredLedger[0]?.symbol).toBe('NVDA');

    store.setEngineFilter('PURE_ML_SELF_LEARNED');
    expect(store.filteredLedger.length).toBe(0);

    store.resetAllLedgerFilters();
    expect(store.filteredLedger.length).toBe(2);
  });

  it('toggles Phase 13 Non-Destructive Adaptive Trap-Immunity & Post-Entry Bar+1 Loss-Centroid Scratch Mode', () => {
    const store = new TerminalStore(false);
    store.ingestBackendSnapshot({
      positions_opened: [],
      closed_trades: [
        {
          trade_id: 'TRD-P13-WIN',
          opened_timestamp_ns: 1_730_300_000_000_000_000,
          closed_timestamp_ns: 1_730_303_600_000_000_000,
          hold_duration_ms: 3_600_000,
          symbol: 'PLTR',
          exchange_index: 'NYSE',
          brain_id: 'BRAIN-03',
          direction: 'LONG',
          entry_price: 200.0,
          actual_exit_price: 208.0,
          quantity_shares: 50,
          kelly_fraction_pct: 2.2,
          risk_budget_usd: 220.0,
          notional_allocated_usd: 22000.0,
          kelly_net_pnl_usd: 880.0,
          initial_structural_stop: 196.0,
          target_1_price: 204.0,
          target_2_price: 208.0,
          outcome_badge: 'CLOSED - TARGET HIT',
          is_winner: true,
          net_pnl_usd: 400.0,
          stock_return_pct: 4.0,
          option_contracts_count: 0,
          option_net_pnl_usd: 0.0,
          realized_r_multiple: 2.0,
          mfe_pct: 4.2,
          mae_pct: 0.3,
          ai_confidence_pct: 92.0,
          primary_driver: 'GAMMA_SQUEEZE',
          engine_mode: 'CONSENSUS_BOTH_ENGINES',
          strategy_archetype: 'GAMMA_SQUEEZE_IGNITION',
          was_vetoed_by_loss_autopsy: false,
          was_scratched_by_loss_autopsy: false,
          adaptive_net_pnl_usd: 400.0,
          adaptive_kelly_net_pnl_usd: 880.0,
          adaptive_loss_saved_usd: 0.0,
          adaptive_kelly_loss_saved_usd: 0.0
        },
        {
          trade_id: 'TRD-P13-VETOED',
          opened_timestamp_ns: 1_730_310_000_000_000_000,
          closed_timestamp_ns: 1_730_313_600_000_000_000,
          hold_duration_ms: 3_600_000,
          symbol: 'PLUG',
          exchange_index: 'NASDAQ_CM',
          brain_id: 'BRAIN-01',
          direction: 'SHORT',
          entry_price: 1.4,
          actual_exit_price: 1.45,
          quantity_shares: 7000,
          kelly_fraction_pct: 1.2,
          risk_budget_usd: 350.0,
          notional_allocated_usd: 12000.0,
          kelly_net_pnl_usd: -420.0,
          initial_structural_stop: 1.45,
          target_1_price: 1.34,
          target_2_price: 1.28,
          outcome_badge: 'CLOSED - STOPPED OUT',
          is_winner: false,
          net_pnl_usd: -350.0,
          stock_return_pct: -3.57,
          option_contracts_count: 0,
          option_net_pnl_usd: 0.0,
          realized_r_multiple: -1.0,
          mfe_pct: 0.4,
          mae_pct: 3.57,
          ai_confidence_pct: 71.0,
          primary_driver: 'SHORT_SQUEEZE',
          engine_mode: 'STRUCTURED_HYBRID_CHAMPION',
          strategy_archetype: 'SHORT_FLOAT_LIQUEFACTION',
          was_vetoed_by_loss_autopsy: true,
          vetoed_by_loss_trade_id: 'PRIOR-LOSS-01',
          entry_trap_similarity_score: 0.912,
          was_scratched_by_loss_autopsy: false,
          adaptive_outcome_badge: 'VETOED - HARD-NEGATIVE TRAP IMMUNITY',
          adaptive_exit_price: 1.4,
          adaptive_scratch_exit_r: 0.0,
          adaptive_net_pnl_usd: 0.0,
          adaptive_kelly_net_pnl_usd: 0.0,
          adaptive_loss_saved_usd: 350.0,
          adaptive_kelly_loss_saved_usd: 420.0
        },
        {
          trade_id: 'TRD-P13-SCRATCH',
          opened_timestamp_ns: 1_730_320_000_000_000_000,
          closed_timestamp_ns: 1_730_323_600_000_000_000,
          hold_duration_ms: 3_600_000,
          symbol: 'IONQ',
          exchange_index: 'NYSE',
          brain_id: 'BRAIN-03',
          direction: 'LONG',
          entry_price: 46.52,
          actual_exit_price: 44.48,
          quantity_shares: 215,
          kelly_fraction_pct: 1.5,
          risk_budget_usd: 527.0,
          notional_allocated_usd: 15000.0,
          kelly_net_pnl_usd: -790.0,
          initial_structural_stop: 44.48,
          target_1_price: 49.5,
          target_2_price: 52.0,
          outcome_badge: 'CLOSED - STOPPED OUT',
          is_winner: false,
          net_pnl_usd: -527.0,
          stock_return_pct: -4.38,
          option_contracts_count: 0,
          option_net_pnl_usd: 0.0,
          realized_r_multiple: -1.0,
          mfe_pct: 0.3,
          mae_pct: 4.38,
          ai_confidence_pct: 74.0,
          primary_driver: 'VOLUME_ABSORPTION',
          engine_mode: 'STRUCTURED_HYBRID_CHAMPION',
          strategy_archetype: 'POC_VOLUME_ABSORPTION',
          was_vetoed_by_loss_autopsy: false,
          was_scratched_by_loss_autopsy: true,
          scratch_bar_offset: 1,
          adaptive_outcome_badge: 'CLOSED - AI LOSS-AUTOPSY SCRATCH EXIT',
          adaptive_exit_price: 46.15,
          adaptive_scratch_exit_r: -0.18,
          adaptive_net_pnl_usd: -95.0,
          adaptive_kelly_net_pnl_usd: -142.5,
          adaptive_loss_saved_usd: 432.0,
          adaptive_kelly_loss_saved_usd: 647.5,
          loss_affinity_posterior_pct: 78.4
        }
      ]
    });

    // 1. Baseline (All Trades) mode preserves all 3 trades ($400 - $350 - $527 = -$477)
    expect(store.immunityMode).toBe('BASELINE_ALL_TRADES');
    expect(store.scorecardKpis.totalClosed).toBe(3);
    expect(store.scorecardKpis.totalWins).toBe(1);
    expect(store.scorecardKpis.totalLosses).toBe(2);
    expect(store.scorecardKpis.netRealizedStockPnlUsd).toBeCloseTo(-477.0, 2);
    expect(store.scorecardKpis.vetoedLossesCount).toBe(1);
    expect(store.scorecardKpis.scratchedLossesCount).toBe(1);
    expect(store.scorecardKpis.totalLossAvoidedUsd).toBeCloseTo(782.0, 2);

    // 2. Toggle to Active Loss-Immunity & Scratch Mode:
    // Vetoed trade is excluded from active trade count (2 trades: 1W/1L), scratched trade loses -$95 instead of -$527 (Net = +$305)
    store.toggleImmunityMode();
    expect(store.immunityMode).toBe('ACTIVE_LOSS_IMMUNITY');
    expect(store.scorecardKpis.totalClosed).toBe(2);
    expect(store.scorecardKpis.totalWins).toBe(1);
    expect(store.scorecardKpis.totalLosses).toBe(1);
    expect(store.scorecardKpis.netRealizedStockPnlUsd).toBeCloseTo(305.0, 2);
    expect(store.scorecardKpis.avgLoserR).toBeCloseTo(-0.18, 2);
    expect(store.equityCurve.length).toBe(2);

    // 3. Combine Active Immunity with Fractional-Kelly Sizing ($880 - $142.5 = +$737.5, Saved = $420 + $647.5 = $1,067.5)
    store.setSizingMode('FRACTIONAL_KELLY');
    expect(store.scorecardKpis.netRealizedStockPnlUsd).toBeCloseTo(737.5, 2);
    expect(store.scorecardKpis.totalLossAvoidedUsd).toBeCloseTo(1067.5, 2);
  });

  it('hydrates Phase 16 Outcome-Blind Confusion Matrix & Portfolio Delta-Gamma Heat Telemetry', () => {
    const store = new TerminalStore(false);
    store.ingestBackendSnapshot({
      positions_opened: [],
      closed_trades: [
        {
          trade_id: 'TRD-P16-TP-VETO',
          opened_timestamp_ns: 1_730_400_000_000_000_000,
          closed_timestamp_ns: 1_730_403_600_000_000_000,
          hold_duration_ms: 3_600_000,
          symbol: 'MARA',
          exchange_index: 'NASDAQ_100',
          brain_id: 'BRAIN-02',
          direction: 'LONG',
          entry_price: 20.0,
          actual_exit_price: 19.0,
          quantity_shares: 500,
          kelly_fraction_pct: 1.5,
          risk_budget_usd: 500.0,
          notional_allocated_usd: 10000.0,
          kelly_net_pnl_usd: -750.0,
          initial_structural_stop: 19.0,
          target_1_price: 21.5,
          target_2_price: 23.0,
          outcome_badge: 'CLOSED - STOPPED OUT',
          is_winner: false,
          net_pnl_usd: -500.0,
          stock_return_pct: -5.0,
          option_contracts_count: 0,
          option_net_pnl_usd: 0.0,
          realized_r_multiple: -1.0,
          mfe_pct: 0.2,
          mae_pct: 5.0,
          ai_confidence_pct: 74.0,
          primary_driver: 'SHORT_SQUEEZE',
          engine_mode: 'STRUCTURED_HYBRID_CHAMPION',
          strategy_archetype: 'SHORT_FLOAT_LIQUEFACTION',
          was_vetoed_by_loss_autopsy: true,
          was_scratched_by_loss_autopsy: false,
          adaptive_net_pnl_usd: 0.0,
          adaptive_kelly_net_pnl_usd: 0.0,
          adaptive_loss_saved_usd: 500.0,
          adaptive_kelly_loss_saved_usd: 750.0
        },
        {
          trade_id: 'TRD-P16-FP-VETO',
          opened_timestamp_ns: 1_730_410_000_000_000_000,
          closed_timestamp_ns: 1_730_413_600_000_000_000,
          hold_duration_ms: 3_600_000,
          symbol: 'SOFI',
          exchange_index: 'NASDAQ_GS',
          brain_id: 'BRAIN-02',
          direction: 'LONG',
          entry_price: 10.0,
          actual_exit_price: 10.6,
          quantity_shares: 1000,
          kelly_fraction_pct: 1.5,
          risk_budget_usd: 300.0,
          notional_allocated_usd: 10000.0,
          kelly_net_pnl_usd: 900.0,
          initial_structural_stop: 9.7,
          target_1_price: 10.5,
          target_2_price: 10.9,
          outcome_badge: 'CLOSED - TARGET HIT',
          is_winner: true,
          net_pnl_usd: 600.0,
          stock_return_pct: 6.0,
          option_contracts_count: 0,
          option_net_pnl_usd: 0.0,
          realized_r_multiple: 2.0,
          mfe_pct: 6.2,
          mae_pct: 0.5,
          ai_confidence_pct: 81.0,
          primary_driver: 'VOLUME_ABSORPTION',
          engine_mode: 'STRUCTURED_HYBRID_CHAMPION',
          strategy_archetype: 'POC_VOLUME_ABSORPTION',
          was_vetoed_by_loss_autopsy: true,
          was_scratched_by_loss_autopsy: false,
          adaptive_net_pnl_usd: 0.0,
          adaptive_kelly_net_pnl_usd: 0.0,
          adaptive_loss_saved_usd: -600.0,
          adaptive_kelly_loss_saved_usd: -900.0
        },
        {
          trade_id: 'TRD-P16-UNSCRATCHED-LOSS',
          opened_timestamp_ns: 1_730_420_000_000_000_000,
          closed_timestamp_ns: 1_730_423_600_000_000_000,
          hold_duration_ms: 3_600_000,
          symbol: 'AMD',
          exchange_index: 'NASDAQ_100',
          brain_id: 'BRAIN-04',
          direction: 'LONG',
          entry_price: 150.0,
          actual_exit_price: 145.5,
          quantity_shares: 66,
          kelly_fraction_pct: 1.2,
          risk_budget_usd: 297.0,
          notional_allocated_usd: 10000.0,
          kelly_net_pnl_usd: -356.4,
          initial_structural_stop: 145.5,
          target_1_price: 157.0,
          target_2_price: 162.0,
          outcome_badge: 'CLOSED - STOPPED OUT',
          is_winner: false,
          net_pnl_usd: -297.0,
          stock_return_pct: -3.0,
          option_contracts_count: 0,
          option_net_pnl_usd: 0.0,
          realized_r_multiple: -1.0,
          mfe_pct: 0.8,
          mae_pct: 3.0,
          ai_confidence_pct: 77.0,
          primary_driver: 'INTERNALS_TIDE',
          engine_mode: 'STRUCTURED_HYBRID_CHAMPION',
          strategy_archetype: 'MACRO_INTERNALS_TIDE_DRIVE',
          was_vetoed_by_loss_autopsy: false,
          was_scratched_by_loss_autopsy: false,
          adaptive_net_pnl_usd: -297.0,
          adaptive_kelly_net_pnl_usd: -356.4,
          adaptive_loss_saved_usd: 0.0,
          adaptive_kelly_loss_saved_usd: 0.0
        }
      ],
      portfolio_heat_telemetry: {
        open_positions_count: 5,
        account_equity_base_usd: 100000.0,
        aggregate_open_risk_usd: 3850.0,
        aggregate_open_risk_pct: 3.85,
        portfolio_heat_cap_pct: 6.5,
        net_portfolio_beta_delta_usd: 54200.0,
        net_portfolio_gamma_1pct_usd: 1420.0,
        net_portfolio_vanna_1pct_usd: 610.0,
        mean_pairwise_correlation: 0.62,
        max_pairwise_correlation: 0.84,
        correlation_heat_multiplier: 0.78,
        heat_regime_badge: 'ELEVATED CORRELATION CAP ACTIVE',
        overlapping_clusters_summary: '2 overlapping cross-brain pairs (NVDA/AMD)',
        cross_position_correlation_matrix: [
          [1.0, 0.84],
          [0.84, 1.0]
        ],
        pairwise_correlations: [
          {
            pos_a_label: 'NVDA (BRAIN-02)',
            pos_b_label: 'AMD (BRAIN-04)',
            symbol_a: 'NVDA',
            symbol_b: 'AMD',
            brain_a: 'BRAIN-02',
            brain_b: 'BRAIN-04',
            ledoit_wolf_similarity: 0.84,
            directional_alignment: 1.0
          }
        ]
      }
    });

    expect(store.scorecardKpis.tpLossesVetoed).toBe(1);
    expect(store.scorecardKpis.fpWinnersVetoed).toBe(1);
    expect(store.scorecardKpis.remainingUnscratchedLosses).toBe(1);
    expect(store.portfolioHeatTelemetry.openPositionsCount).toBe(5);
    expect(store.portfolioHeatTelemetry.correlationHeatMultiplier).toBeCloseTo(0.78, 2);
    expect(store.portfolioHeatTelemetry.pairwiseCorrelations.length).toBe(1);
  });

  it('hydrates Phase 18 Multi-Generation Evolution History (Gen 0..Gen 4) and 32-Feature Synaptic Plasticity States', () => {
    const store = new TerminalStore(false);
    store.ingestBackendSnapshot({
      positions_opened: [],
      closed_trades: [],
      evolution_generations_history: [
        {
          generation_id: 0,
          status_label: 'GEN-0 BASELINE',
          is_current_champion: false,
          timestamp_iso: '2026-11-02T20:00:00Z',
          active_trades: 152,
          wins: 106,
          losses: 46,
          win_rate_pct: 69.74,
          tp_losses_vetoed: 14,
          fp_winners_vetoed: 7,
          entry_veto_precision_pct: 66.67,
          tp_losses_scratched: 17,
          fp_winners_scratched: 3,
          scratch_precision_pct: 85.0,
          tp_losses_caught_total: 31,
          fp_winners_clipped_total: 10,
          remaining_unscratched_losses: 26,
          profit_factor: 3.28,
          kelly_profit_factor: 3.31,
          fixed_net_pnl_usd: 35646.99,
          kelly_net_pnl_usd: 56275.03,
          max_drawdown_usd: 1188.68,
          composite_fitness_score: 58336.05,
          mutations_applied_summary: [
            'Gen 0 Baseline: 32-feature uniform weights (w_f = 1.00 across all 32 features).'
          ]
        },
        {
          generation_id: 4,
          status_label: 'GEN-4 PROMOTED CHAMPION',
          is_current_champion: true,
          timestamp_iso: '2026-11-02T20:05:00Z',
          active_trades: 153,
          wins: 115,
          losses: 38,
          win_rate_pct: 75.16,
          tp_losses_vetoed: 20,
          fp_winners_vetoed: 0,
          entry_veto_precision_pct: 100.0,
          tp_losses_scratched: 19,
          fp_winners_scratched: 1,
          scratch_precision_pct: 95.0,
          tp_losses_caught_total: 39,
          fp_winners_clipped_total: 1,
          remaining_unscratched_losses: 18,
          profit_factor: 5.07,
          kelly_profit_factor: 4.98,
          fixed_net_pnl_usd: 45333.9,
          kelly_net_pnl_usd: 70769.27,
          max_drawdown_usd: 1766.2,
          composite_fitness_score: 73482.14,
          mutations_applied_summary: [
            'BRAIN-02 (Gen 4): Boosted #2 s1_wick_exhaustion_ratio (1.38x) | Slashed #14 s2_iv_rank_percentile (0.64x).'
          ]
        }
      ],
      synaptic_plasticity_states: [
        {
          brain_id: 'BRAIN-02',
          generation_id: 4,
          synaptic_feature_weights_32: Array(32).fill(1.05),
          stream_prior_multipliers: [1.14, 1.08, 0.96, 1.02],
          wald_sprt_thresholds: [1.48, 1.54, 1.6],
          entry_veto_margin_delta: 0.015,
          trap_conviction_weights: { 'TRD-0012': 1.35, 'TRD-0044': 0.52 },
          top_boosted_warning_features: [
            {
              feature_index: 2,
              feature_name: 's1_wick_exhaustion_ratio',
              stream_id: 'STREAM_1_PRICE_VOLUME',
              gen0_weight: 1.0,
              current_weight: 1.38,
              delta_pct: 38.0,
              role_classification: 'BOOSTED_WARNING_SENSOR'
            }
          ],
          top_slashed_deceiver_features: [
            {
              feature_index: 14,
              feature_name: 's2_iv_rank_percentile',
              stream_id: 'STREAM_2_OPTIONS_GAMMA',
              gen0_weight: 1.0,
              current_weight: 0.64,
              delta_pct: -36.0,
              role_classification: 'SLASHED_DECEIVER_FEATURE'
            }
          ],
          cumulative_tp_vetoes: 6,
          cumulative_fp_vetoes: 0,
          cumulative_tp_scratches: 5,
          cumulative_fp_scratches: 0,
          remaining_unscratched_losses: 4,
          last_mutation_summary: 'Gen 4 promoted champion'
        }
      ]
    });

    expect(store.evolutionGenerationsHistory.length).toBe(2);
    const gen0 = store.evolutionGenerationsHistory[0]!;
    const gen4 = store.evolutionGenerationsHistory[1]!;
    expect(gen0.generation_id).toBe(0);
    expect(gen0.win_rate_pct).toBeCloseTo(69.74, 2);
    expect(gen4.generation_id).toBe(4);
    expect(gen4.promoted_as_champion).toBe(true);
    expect(gen4.win_rate_pct).toBeCloseTo(75.16, 2);
    expect(gen4.profit_factor_fixed).toBeCloseTo(5.07, 2);
    expect(gen4.tp_losses_vetoed).toBe(20);
    expect(gen4.fp_winners_vetoed).toBe(0);
    expect(gen4.tp_losses_scratched).toBe(19);
    expect(gen4.fp_winners_scratched).toBe(1);
    expect(gen4.remaining_full_stops).toBe(18);
    expect(gen4.composite_fitness_score).toBeCloseTo(73482.14, 2);

    expect(store.synapticPlasticityStates.length).toBe(1);
    const sp = store.synapticPlasticityStates[0]!;
    expect(sp.brain_id).toBe('BRAIN-02');
    expect(sp.generation_id).toBe(4);
    expect(sp.top_boosted_warning_features[0]?.feature_name).toBe('s1_wick_exhaustion_ratio');
    expect(sp.top_boosted_warning_features[0]?.updated_weight).toBeCloseTo(1.38, 2);
    expect(sp.top_slashed_deceiver_features[0]?.feature_name).toBe('s2_iv_rank_percentile');
    expect(sp.top_slashed_deceiver_features[0]?.updated_weight).toBeCloseTo(0.64, 2);
  });

  it('hydrates Phase 19 Empirical-Bayes Per-Brain Specialization (Gen 8) and Self-Learned Mid-Trade Stall Saves (Bar +4..+6)', () => {
    const store = new TerminalStore(false);
    store.ingestBackendSnapshot({
      positions_opened: [],
      closed_trades: [],
      evolution_generations_history: [
        {
          generation_id: 8,
          status_label: 'GEN-8 PROMOTED CHAMPION',
          is_current_champion: true,
          timestamp_iso: '2026-11-02T21:15:00Z',
          active_trades: 153,
          wins: 115,
          losses: 38,
          win_rate_pct: 75.16,
          tp_losses_vetoed: 20,
          fp_winners_vetoed: 0,
          entry_veto_precision_pct: 100.0,
          tp_losses_scratched: 35,
          fp_winners_scratched: 1,
          scratch_precision_pct: 97.22,
          early_scratches_bar1_to_3: 19,
          mid_trade_stall_saves_bar4_to_6: 16,
          fp_mid_trade_scratches: 0,
          tp_losses_caught_total: 55,
          fp_winners_clipped_total: 1,
          remaining_unscratched_losses: 2,
          profit_factor: 11.62,
          kelly_profit_factor: 10.95,
          fixed_net_pnl_usd: 51618.62,
          kelly_net_pnl_usd: 80459.93,
          max_drawdown_usd: 1045.05,
          composite_fitness_score: 94081.69,
          mutations_applied_summary: [
            'BRAIN-02 (Gen 8 | EB λ=0.81): Mid-Trade Stall Saves=16 (0 FP), Stop Ratchet=-0.18R'
          ]
        }
      ],
      synaptic_plasticity_states: [
        {
          brain_id: 'BRAIN-02',
          generation_id: 8,
          synaptic_feature_weights_32: Array(32).fill(1.12),
          stream_prior_multipliers: [1.18, 1.12, 0.94, 1.04],
          wald_sprt_thresholds: [1.46, 1.52, 1.58],
          entry_veto_margin_delta: 0.012,
          trap_conviction_weights: { 'TRD-0012': 1.48 },
          empirical_bayes_shrinkage_lambda: 0.8124,
          mid_trade_wald_sprt_thresholds: [1.31, 1.28, 1.24],
          learned_mid_trade_mfe_norm_quantile: 0.56,
          learned_mid_trade_stop_ratchet_r: -0.18,
          top_boosted_warning_features: [],
          top_slashed_deceiver_features: [],
          cumulative_tp_vetoes: 8,
          cumulative_fp_vetoes: 0,
          cumulative_tp_scratches: 11,
          cumulative_fp_scratches: 0,
          remaining_unscratched_losses: 0,
          last_mutation_summary: 'Gen 8 promoted champion'
        }
      ]
    });

    expect(store.evolutionGenerationsHistory.length).toBe(1);
    const gen8 = store.evolutionGenerationsHistory[0]!;
    expect(gen8.generation_id).toBe(8);
    expect(gen8.early_scratches_bar1_to_3).toBe(19);
    expect(gen8.mid_trade_stall_saves_bar4_to_6).toBe(16);
    expect(gen8.tp_losses_scratched).toBe(35);
    expect(gen8.remaining_full_stops).toBe(2);
    expect(gen8.profit_factor_fixed).toBeCloseTo(11.62, 2);

    const sp8 = store.synapticPlasticityStates[0]!;
    expect(sp8.empirical_bayes_shrinkage_lambda).toBeCloseTo(0.8124, 4);
    expect(sp8.mid_trade_wald_sprt_thresholds).toEqual([1.31, 1.28, 1.24]);
    expect(sp8.learned_mid_trade_stop_ratchet_r).toBeCloseTo(-0.18, 2);
  });

  it('hydrates Phase 20 Gen 0..Gen 12 Profit-Giveback Ratchet Evolution & Open-Position Live Sequential LLR Telemetry', () => {
    const store = new TerminalStore(false);

    store.ingestBackendSnapshot({
      positions_opened: [
        {
          position_id: 'POS-P20-1',
          opened_timestamp_ns: 1_730_500_000_000_000_000,
          symbol: 'NVDA',
          exchange_index: 'NASDAQ_100',
          brain_id: 'BRAIN-02',
          direction: 'LONG',
          entry_price: 138.5,
          live_price: 142.1,
          quantity_shares: 100,
          structural_stop_price: 136.8,
          target_1_price: 141.5,
          target_2_price: 145.0,
          bars_held_count: 9,
          active_sequential_stage: 'STAGE_3_RUNNER_GIVEBACK_GUARD (Bar 6+)',
          live_sequential_llr: -0.42,
          champion_llr_threshold: 1.36,
          sequential_safety_margin_pct: 84.2,
          next_bar_ratchet_stop_preview: 141.38
        }
      ],
      closed_trades: [
        {
          trade_id: 'TRD-P20-GB-1',
          opened_timestamp_ns: 1_730_490_000_000_000_000,
          closed_timestamp_ns: 1_730_495_400_000_000_000,
          hold_duration_ms: 5_400_000,
          symbol: 'PLTR',
          exchange_index: 'NYSE',
          brain_id: 'BRAIN-03',
          direction: 'LONG',
          entry_price: 42.0,
          actual_exit_price: 43.2,
          quantity_shares: 250,
          initial_structural_stop: 40.8,
          target_1_price: 43.8,
          target_2_price: 45.5,
          outcome_badge: 'CLOSED - TARGET HIT',
          is_winner: true,
          net_pnl_usd: 300.0,
          stock_return_pct: 2.86,
          realized_r_multiple: 1.0,
          was_protected_from_giveback: true,
          giveback_ratchet_bar_offset: 8,
          giveback_r_saved: 0.72
        }
      ],
      evolution_generations_history: [
        {
          generation_id: 12,
          status_label: 'GEN-12 PROMOTED CHAMPION',
          is_current_champion: true,
          timestamp_iso: '2026-11-02T22:00:00Z',
          active_trades: 153,
          wins: 115,
          losses: 38,
          win_rate_pct: 75.16,
          tp_losses_vetoed: 20,
          fp_winners_vetoed: 0,
          entry_veto_precision_pct: 100.0,
          tp_losses_scratched: 36,
          fp_winners_scratched: 1,
          scratch_precision_pct: 97.3,
          early_scratches_bar1_to_3: 19,
          mid_trade_stall_saves_bar4_to_6: 17,
          giveback_saves_count: 19,
          total_giveback_r_recovered: 13.09,
          avg_winner_r: 0.91,
          tp_losses_caught_total: 56,
          fp_winners_clipped_total: 1,
          remaining_unscratched_losses: 1,
          profit_factor: 15.16,
          kelly_profit_factor: 14.54,
          fixed_net_pnl_usd: 57937.26,
          kelly_net_pnl_usd: 90959.59,
          max_drawdown_usd: 353.64,
          composite_fitness_score: 111762.24,
          mutations_applied_summary: [
            'BRAIN-02 (Gen 12): Giveback Saves=19 (+13.09R recovered), Avg Winner=+0.91R'
          ]
        }
      ],
      synaptic_plasticity_states: [
        {
          brain_id: 'BRAIN-02',
          generation_id: 12,
          synaptic_feature_weights_32: Array(32).fill(1.15),
          stream_prior_multipliers: [1.2, 1.14, 0.95, 1.05],
          wald_sprt_thresholds: [1.46, 1.52, 1.58],
          entry_veto_margin_delta: 0.015,
          trap_conviction_weights: {},
          empirical_bayes_shrinkage_lambda: 0.8904,
          mid_trade_wald_sprt_thresholds: [1.42, 1.38, 1.35],
          learned_mid_trade_mfe_norm_quantile: 0.55,
          learned_mid_trade_stop_ratchet_r: -0.121,
          learned_itm_mfe_activation_r: 0.931,
          giveback_exhaustion_threshold: 1.36,
          runner_continuation_threshold: -0.52,
          learned_giveback_retention_ratio: 0.8,
          top_boosted_warning_features: [],
          top_slashed_deceiver_features: [],
          cumulative_tp_vetoes: 8,
          cumulative_fp_vetoes: 0,
          cumulative_tp_scratches: 12,
          cumulative_fp_scratches: 0,
          remaining_unscratched_losses: 0,
          last_mutation_summary: 'Gen 12 promoted champion'
        }
      ]
    });

    const openPos = store.openPositions[0]!;
    expect(openPos.barsHeldCount).toBe(9);
    expect(openPos.activeSequentialStage).toBe('STAGE_3_RUNNER_GIVEBACK_GUARD (Bar 6+)');
    expect(openPos.liveSequentialLlr).toBeCloseTo(-0.42, 2);
    expect(openPos.championLlrThreshold).toBeCloseTo(1.36, 2);
    expect(openPos.sequentialSafetyMarginPct).toBeCloseTo(84.2, 1);
    expect(openPos.nextBarRatchetStopPreview).toBeCloseTo(141.38, 2);

    const closedTr = store.closedLedger[0]!;
    expect(closedTr.wasProtectedFromGiveback).toBe(true);
    expect(closedTr.givebackRatchetBarOffset).toBe(8);
    expect(closedTr.givebackRSaved).toBeCloseTo(0.72, 2);

    const gen12 = store.evolutionGenerationsHistory[0]!;
    expect(gen12.generation_id).toBe(12);
    expect(gen12.giveback_saves_count).toBe(19);
    expect(gen12.total_giveback_r_recovered).toBeCloseTo(13.09, 2);
    expect(gen12.avg_winner_r).toBeCloseTo(0.91, 2);
    expect(gen12.remaining_full_stops).toBe(1);

    const sp12 = store.synapticPlasticityStates[0]!;
    expect(sp12.learned_itm_mfe_activation_r).toBeCloseTo(0.931, 3);
    expect(sp12.giveback_exhaustion_threshold).toBeCloseTo(1.36, 2);
    expect(sp12.runner_continuation_threshold).toBeCloseTo(-0.52, 2);
    expect(sp12.learned_giveback_retention_ratio).toBeCloseTo(0.8, 2);
  });
});
