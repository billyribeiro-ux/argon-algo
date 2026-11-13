import { describe, expect, it, vi } from 'vitest';
import rustWireFixture from './__fixtures__/phase30_rust_wire_contract.json';
import autopsyWireFixture from './__fixtures__/phase32_autopsy_wire_contract.json';
import phase34ShadowWireFixture from './__fixtures__/phase34_shadow_bandit_wire_contract.json';
import phase35EigenConvexityWireFixture from './__fixtures__/phase35_portfolio_eigen_convexity_wire_contract.json';
import phase36ApexWireFixture from './__fixtures__/phase36_apex_intelligence_wire_contract.json';
import phase37SovereignWireFixture from './__fixtures__/phase37_sovereign_intelligence_wire_contract.json';
import {
  TerminalStore,
  formatSubPennyPrice,
  CSV_LEDGER_HEADERS,
  type StagedOrderReceipt,
  type StageSundayOrdersResponse,
  type StageSundayOrdersErrorResponse,
  type RedactedWebhookConfigResponse,
  type WebhookTestPingResponse
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

  it('hydrates Phase 21 Gen 0..Gen 16 House-Money Runner Pyramiding, Zero-Tail-Loss Autopsy, & CBOE Gamma Roll-Up Stager', () => {
    const store = new TerminalStore(false);
    store.ingestBackendSnapshot({
      positions_opened: [
        {
          position_id: 'OPEN-P21-AMD',
          opened_timestamp_ns: 1_730_510_000_000_000_000,
          live_hold_duration_ms: 86_400_000,
          symbol: 'AMD',
          exchange_index: 'NASDAQ_100',
          brain_id: 'BRAIN-04',
          direction: 'LONG',
          entry_price: 566.42,
          live_price: 567.854,
          quantity_shares: 18.0,
          kelly_fraction_pct: 2.1,
          risk_budget_usd: 972.39,
          notional_allocated_usd: 10195.56,
          tranche_count: 1,
          scale_in_history: ['T1: Entry 18.00 sh @ $566.42'],
          entry_bar_idx: 24,
          structural_stop_price: 512.39848,
          target_1_price: 631.76916,
          target_2_price: 683.08175,
          unrealized_stock_pnl_usd: 25.81,
          unrealized_stock_pnl_pct: 0.25,
          option_structure: 'DIAGONAL_SPREAD',
          matched_option_contract: 'AMD 2026-12-18 $565.00C',
          option_entry_premium: 38.4,
          option_live_premium: 39.25,
          option_contracts_count: 2,
          unrealized_option_pnl_usd: 170.0,
          unrealized_option_pnl_pct: 2.21,
          contract_delta: 0.62,
          contract_gamma: 0.018,
          contract_vanna: 0.009,
          contract_theta: -0.24,
          contract_vega: 0.52,
          contract_iv: 0.44,
          current_r_multiple: 1.18,
          mfe_pct: 3.85,
          mae_pct: 0.12,
          ai_confidence_pct: 74.0,
          primary_driver: 'GAMMA_SQUEEZE',
          engine_mode: 'CONSENSUS_BOTH_ENGINES',
          strategy_archetype: 'GAMMA_SQUEEZE_IGNITION',
          trace_health_badge: 'OPTIMAL',
          precedent_matches_count: 25,
          precedent_win_rate_pct: 88.0,
          bars_held_count: 11,
          active_sequential_stage: 'STAGE_3_RUNNER_GIVEBACK_GUARD (Bar 6+)',
          live_sequential_llr: -0.58,
          champion_llr_threshold: 1.36,
          sequential_safety_margin_pct: 91.5,
          next_bar_ratchet_stop_preview: 598.15,
          house_money_scale_eligible: true,
          learned_house_money_phi: 0.524,
          locked_profit_r: 0.587,
          guaranteed_net_floor_r: 0.279,
          next_gamma_wall_strike: 655.0,
          next_gamma_wall_contract: 'AMD 2026-12-18 $655.00C (GAMMA WALL ROLL)',
          banked_option_gain_usd: 89.08,
          rolled_option_budget_usd: 80.92
        }
      ],
      closed_trades: [
        {
          trade_id: 'TRD-P21-RUNNER',
          opened_timestamp_ns: 1_730_450_000_000_000_000,
          closed_timestamp_ns: 1_730_456_000_000_000_000,
          hold_duration_ms: 6_000_000,
          symbol: 'NVDA',
          exchange_index: 'NASDAQ_100',
          brain_id: 'BRAIN-04',
          direction: 'LONG',
          entry_price: 149.87554,
          actual_exit_price: 168.93722,
          quantity_shares: 100.0,
          kelly_fraction_pct: 2.35,
          risk_budget_usd: 850.0,
          notional_allocated_usd: 14987.55,
          kelly_net_pnl_usd: 3420.0,
          initial_structural_stop: 140.48,
          target_1_price: 161.5,
          target_2_price: 170.0,
          outcome_badge: 'CLOSED - TARGET HIT',
          is_winner: true,
          net_pnl_usd: 2131.55,
          stock_return_pct: 12.72,
          option_contracts_count: 2,
          option_net_pnl_usd: 640.0,
          realized_r_multiple: 2.03,
          mfe_pct: 13.1,
          mae_pct: 0.25,
          ai_confidence_pct: 91.5,
          primary_driver: 'GAMMA_SQUEEZE',
          engine_mode: 'CONSENSUS_BOTH_ENGINES',
          strategy_archetype: 'GAMMA_SQUEEZE_IGNITION',
          was_vetoed_by_loss_autopsy: false,
          was_scratched_by_loss_autopsy: false,
          was_protected_from_giveback: false,
          was_scaled_with_house_money: true,
          house_money_scale_bar_offset: 6,
          house_money_additional_r_gained: 0.89,
          adaptive_outcome_badge: 'CLOSED - TARGET HIT (+HOUSE-MONEY PYRAMID)',
          adaptive_exit_price: 168.93722,
          adaptive_scratch_exit_r: 2.92,
          adaptive_net_pnl_usd: 3066.2,
          adaptive_kelly_net_pnl_usd: 4919.8,
          adaptive_loss_saved_usd: 934.65,
          adaptive_kelly_loss_saved_usd: 1499.8
        }
      ],
      evolution_generations_history: [
        {
          generation_id: 16,
          status_label: 'GEN-16 PROMOTED CHAMPION',
          is_current_champion: true,
          timestamp_iso: '2026-11-02T23:59:59Z',
          active_trades: 153,
          wins: 115,
          losses: 38,
          win_rate_pct: 75.16,
          tp_losses_vetoed: 20,
          fp_winners_vetoed: 0,
          entry_veto_precision_pct: 100.0,
          tp_losses_scratched: 37,
          fp_winners_scratched: 1,
          scratch_precision_pct: 97.37,
          early_scratches_bar1_to_3: 19,
          mid_trade_stall_saves_bar4_to_6: 18,
          giveback_saves_count: 19,
          total_giveback_r_recovered: 13.09,
          house_money_scaled_runners_count: 23,
          total_house_money_r_gained: 17.14,
          avg_winner_r: 1.06,
          tp_losses_caught_total: 57,
          fp_winners_clipped_total: 1,
          remaining_unscratched_losses: 0,
          profit_factor: 18.98,
          kelly_profit_factor: 18.32,
          fixed_net_pnl_usd: 68551.95,
          kelly_net_pnl_usd: 107175.98,
          max_drawdown_usd: 310.3,
          composite_fitness_score: 133796.03,
          mutations_applied_summary: [
            'Gen 16: 0 Remaining Unscratched Losses (KULR Bar+15 scratched at -0.22R), 23 House-Money Runners (+17.14R)'
          ]
        }
      ],
      synaptic_plasticity_states: [
        {
          brain_id: 'BRAIN-04',
          generation_id: 16,
          synaptic_feature_weights_32: Array(32).fill(1.18),
          stream_prior_multipliers: [1.16, 1.08, 1.14, 1.28],
          wald_sprt_thresholds: [1.46, 1.38, 1.44],
          entry_veto_margin_delta: 0.015,
          trap_conviction_weights: {},
          empirical_bayes_shrinkage_lambda: 0.8344,
          mid_trade_wald_sprt_thresholds: [1.42, 1.38, 1.35],
          learned_mid_trade_mfe_norm_quantile: 0.55,
          learned_mid_trade_stop_ratchet_r: -0.121,
          learned_itm_mfe_activation_r: 1.024,
          giveback_exhaustion_threshold: 1.36,
          runner_continuation_threshold: -0.52,
          learned_giveback_retention_ratio: 0.807,
          learned_profit_lock_activation_r: 0.416,
          learned_house_money_reinvestment_fraction: 0.524,
          late_blowoff_scratch_threshold: 1.28,
          top_boosted_warning_features: [],
          top_slashed_deceiver_features: [],
          cumulative_tp_vetoes: 6,
          cumulative_fp_vetoes: 0,
          cumulative_tp_scratches: 7,
          cumulative_fp_scratches: 0,
          remaining_unscratched_losses: 0,
          last_mutation_summary: 'Gen 16 promoted champion'
        }
      ]
    });

    const openPos = store.openPositions[0]!;
    expect(openPos.houseMoneyScaleEligible).toBe(true);
    expect(openPos.learnedHouseMoneyPhi).toBeCloseTo(0.524, 3);
    expect(openPos.lockedProfitR).toBeCloseTo(0.587, 3);
    expect(openPos.guaranteedNetFloorR).toBeCloseTo(0.279, 3);
    expect(openPos.nextGammaWallStrike).toBeCloseTo(655.0, 2);

    const closedRunner = store.closedLedger[0]!;
    expect(closedRunner.wasScaledWithHouseMoney).toBe(true);
    expect(closedRunner.houseMoneyScaleBarOffset).toBe(6);
    expect(closedRunner.houseMoneyAdditionalRGained).toBeCloseTo(0.89, 2);

    const gen16 = store.evolutionGenerationsHistory[0]!;
    expect(gen16.generation_id).toBe(16);
    expect(gen16.remaining_full_stops).toBe(0);
    expect(gen16.house_money_scaled_runners_count).toBe(23);
    expect(gen16.total_house_money_r_gained).toBeCloseTo(17.14, 2);
    expect(gen16.avg_winner_r).toBeCloseTo(1.06, 2);

    const sp16 = store.synapticPlasticityStates[0]!;
    expect(sp16.learned_profit_lock_activation_r).toBeCloseTo(0.416, 3);
    expect(sp16.learned_house_money_reinvestment_fraction).toBeCloseTo(0.524, 3);
    expect(sp16.late_blowoff_scratch_threshold).toBeCloseTo(1.28, 2);
  });

  it('hydrates Phase 22 Gen 20 Shakeout Rescue (116/116 Winners, 0 FP), Freed-Risk Capital Recycler, & Atomic 2-Leg BAG Combo Router', async () => {
    const store = new TerminalStore(false);
    store.ingestBackendSnapshot({
      positions_opened: [
        {
          position_id: 'OPEN-P22-AMD',
          opened_timestamp_ns: 1_793_739_300_000_000_000,
          symbol: 'AMD',
          exchange_index: 'SP_500',
          brain_id: 'BRAIN-04',
          direction: 'LONG',
          entry_price: 566.42,
          live_price: 567.854,
          quantity_shares: 18,
          kelly_fraction_pct: 1.9,
          risk_budget_usd: 972.39,
          notional_allocated_usd: 10195.56,
          tranche_count: 2,
          structural_stop_price: 512.39848,
          target_1_price: 651.48057,
          target_2_price: 692.39653,
          unrealized_stock_pnl_usd: 25.81,
          unrealized_stock_pnl_pct: 0.25,
          option_structure: 'DIAGONAL_SPREAD',
          matched_option_contract: 'AMD 2026-12-18 $565.00C',
          option_entry_premium: 42.5,
          option_live_premium: 157.7,
          option_contracts_count: 5,
          unrealized_option_pnl_usd: 57600.0,
          unrealized_option_pnl_pct: 271.06,
          contract_delta: 0.6222,
          contract_gamma: 0.018,
          contract_vanna: 0.011,
          contract_theta: -0.14,
          contract_vega: 0.35,
          contract_iv: 0.48,
          current_r_multiple: 0.68,
          mfe_pct: 4.5,
          mae_pct: 0.2,
          ai_confidence_pct: 77.7,
          bars_held_count: 5,
          active_sequential_stage: 'STAGE_3_GIVEBACK_RATCHET & HOUSE_MONEY_PYRAMID',
          live_sequential_llr: -0.64,
          champion_llr_threshold: -0.52,
          sequential_safety_margin_pct: 84.2,
          next_bar_ratchet_stop_preview: 598.4,
          house_money_scale_eligible: true,
          learned_house_money_phi: 0.524,
          locked_profit_r: 0.587,
          guaranteed_net_floor_r: 0.279,
          next_gamma_wall_strike: 655.0,
          next_gamma_wall_contract: 'AMD 2026-12-18 $655.00C',
          banked_option_gain_usd: 12141.44,
          rolled_option_budget_usd: 9129.0
        }
      ],
      closed_trades: [],
      portfolio_heat_telemetry: {
        account_equity_usd: 205128.62,
        open_positions_count: 5,
        aggregate_open_risk_usd: 2810.0,
        aggregate_open_risk_pct: 1.37,
        portfolio_heat_cap_pct: 8.5,
        gross_portfolio_notional_usd: 52000.0,
        net_portfolio_beta_delta_usd: 14200.0,
        net_portfolio_gamma_1pct_usd: 1850.0,
        net_portfolio_vanna_1pct_usd: 640.0,
        mean_pairwise_correlation: 0.18,
        max_pairwise_correlation: 0.34,
        correlation_heat_multiplier: 1.0,
        heat_regime_badge: 'LOW HEAT — FULL 1.00x KELLY CAPACITY',
        position_labels: ['AMD (BRAIN-04)'],
        cross_position_correlation_matrix: [[1.0]],
        pairwise_correlations: [],
        overlapping_clusters_summary: 'Independent multi-horizon exposure',
        recycled_risk_pool_usd: 48670.25,
        recycled_runner_boost_count: 24,
        recycled_additional_r_compounded: 9.65,
        cross_brain_recycle_multiplier: 1.33
      },
      evolution_generations_history: [
        {
          generation_id: 20,
          status_label: 'GEN-20 PROMOTED CHAMPION',
          is_current_champion: true,
          timestamp_iso: '2026-11-03T17:00:00Z',
          active_trades: 153,
          wins: 116,
          losses: 37,
          win_rate_pct: 75.82,
          tp_losses_vetoed: 20,
          fp_winners_vetoed: 0,
          entry_veto_precision_pct: 100.0,
          tp_losses_scratched: 37,
          fp_winners_scratched: 0,
          scratch_precision_pct: 100.0,
          early_scratches_bar1_to_3: 19,
          mid_trade_stall_saves_bar4_to_6: 18,
          giveback_saves_count: 19,
          total_giveback_r_recovered: 13.09,
          house_money_scaled_runners_count: 24,
          total_house_money_r_gained: 27.77,
          recycled_risk_pool_usd: 48670.25,
          recycled_runner_boost_count: 24,
          recycled_additional_r_compounded: 9.65,
          avg_winner_r: 1.15,
          tp_losses_caught_total: 57,
          fp_winners_clipped_total: 0,
          remaining_unscratched_losses: 0,
          profit_factor: 22.87,
          kelly_profit_factor: 22.4,
          fixed_net_pnl_usd: 76615.42,
          kelly_net_pnl_usd: 120660.5,
          max_drawdown_usd: 282.48,
          composite_fitness_score: 153494.0,
          mutations_applied_summary: [
            'Gen 20: 116/116 Winners Preserved (0 FP Scratches), 24 Recycled House-Money Runners (+27.77R total, +9.65R recycled)'
          ]
        }
      ],
      synaptic_plasticity_states: [
        {
          brain_id: 'BRAIN-04',
          generation_id: 20,
          synaptic_feature_weights_32: Array(32).fill(1.22),
          stream_prior_multipliers: [1.18, 1.12, 1.15, 1.31],
          wald_sprt_thresholds: [1.46, 1.38, 1.44],
          entry_veto_margin_delta: 0.015,
          trap_conviction_weights: {},
          empirical_bayes_shrinkage_lambda: 0.8344,
          mid_trade_wald_sprt_thresholds: [1.42, 1.38, 1.35],
          learned_mid_trade_mfe_norm_quantile: 0.55,
          learned_mid_trade_stop_ratchet_r: -0.121,
          learned_itm_mfe_activation_r: 1.024,
          giveback_exhaustion_threshold: 1.36,
          runner_continuation_threshold: -0.52,
          learned_giveback_retention_ratio: 0.807,
          learned_profit_lock_activation_r: 0.416,
          learned_house_money_reinvestment_fraction: 0.524,
          late_blowoff_scratch_threshold: 1.28,
          learned_shakeout_absorption_quantile: 0.68,
          learned_cross_brain_recycle_multiplier: 1.33,
          top_boosted_warning_features: [],
          top_slashed_deceiver_features: [],
          cumulative_tp_vetoes: 6,
          cumulative_fp_vetoes: 0,
          cumulative_tp_scratches: 6,
          cumulative_fp_scratches: 0,
          remaining_unscratched_losses: 0,
          last_mutation_summary: 'Gen 20 promoted champion'
        }
      ]
    });

    const gen20 = store.evolutionGenerationsHistory[0]!;
    expect(gen20.generation_id).toBe(20);
    expect(gen20.winners_preserved_count).toBe(116);
    expect(gen20.fp_winners_scratched).toBe(0);
    expect(gen20.scratch_precision_pct).toBeCloseTo(100.0, 2);
    expect(gen20.recycled_risk_pool_usd).toBeCloseTo(48670.25, 2);
    expect(gen20.recycled_runner_boost_count).toBe(24);
    expect(gen20.recycled_additional_r_compounded).toBeCloseTo(9.65, 2);

    expect(store.portfolioHeatTelemetry.recycledRiskPoolUsd).toBeCloseTo(48670.25, 2);
    expect(store.portfolioHeatTelemetry.crossBrainRecycleMultiplier).toBeCloseTo(1.33, 2);

    const rollTicket = await store.stageOptionRollTicket('OPEN-P22-AMD');
    expect(rollTicket).not.toBeNull();
    expect(rollTicket?.comboOrderType).toBe('IBKR_BAG_VERTICAL_ROLL_COMBO');
    expect(rollTicket?.leg1Close.action).toBe('SELL_TO_CLOSE');
    expect(rollTicket?.leg2Open.action).toBe('BUY_TO_OPEN');
    expect(rollTicket?.nbboLimitWalkLadder.length).toBe(3);

    const executed = await store.executeRollComboOrder('OPEN-P22-AMD', 'IBKR_TWS');
    expect(executed).toBe(true);
    expect(store.activeStagedRollTicketModal?.rollStatus).toBe('ATOMIC_2LEG_BAG_EXECUTED_FILLED');
  });

  it('hydrates Phase 23 Gen-24 Pareto Dominance Champion, Almgren-Chriss Child Execution Schedule, and SPY/QQQ Vanna Tail-Hedge Overlay', async () => {
    const store = new TerminalStore(false);
    store.ingestBackendSnapshot({
      positions_opened: [
        {
          position_id: 'OPEN-P23-NVDA',
          opened_timestamp_ns: 1_793_739_300_000_000_000,
          symbol: 'NVDA',
          exchange_index: 'NASDAQ_100',
          brain_id: 'BRAIN-02',
          direction: 'LONG',
          entry_price: 210.0,
          live_price: 218.5,
          quantity_shares: 120,
          kelly_fraction_pct: 2.15,
          risk_budget_usd: 1120.0,
          notional_allocated_usd: 25200.0,
          tranche_count: 2,
          structural_stop_price: 204.5,
          target_1_price: 222.0,
          target_2_price: 235.0,
          unrealized_stock_pnl_usd: 1020.0,
          unrealized_stock_pnl_pct: 4.05,
          option_structure: 'LONG_CALL_GAMMA',
          matched_option_contract: 'NVDA 2026-11-20 $210.00C',
          option_entry_premium: 6.4,
          option_live_premium: 11.8,
          option_contracts_count: 6,
          unrealized_option_pnl_usd: 3240.0,
          unrealized_option_pnl_pct: 84.38
        }
      ],
      closed_trades: [],
      portfolio_heat_telemetry: {
        account_equity_usd: 227406.09,
        open_positions_count: 5,
        aggregate_open_risk_usd: 2740.0,
        aggregate_open_risk_pct: 1.2,
        portfolio_heat_cap_pct: 8.5,
        gross_portfolio_notional_usd: 54000.0,
        net_portfolio_beta_delta_usd: 14800.0,
        net_portfolio_gamma_1pct_usd: 1920.0,
        net_portfolio_vanna_1pct_usd: 680.0,
        mean_pairwise_correlation: 0.18,
        max_pairwise_correlation: 0.34,
        correlation_heat_multiplier: 1.0,
        heat_regime_badge: 'LOW HEAT — FULL 1.00x KELLY CAPACITY',
        position_labels: ['NVDA (BRAIN-02)'],
        cross_position_correlation_matrix: [[1.0]],
        pairwise_correlations: [],
        overlapping_clusters_summary: 'Independent multi-horizon exposure',
        recycled_risk_pool_usd: 50507.68,
        recycled_runner_boost_count: 24,
        recycled_additional_r_compounded: 10.63,
        cross_brain_recycle_multiplier: 1.33,
        avg_execution_slippage_bps: 0.74,
        slippage_saved_usd: 2701.91,
        tail_hedge_ratio_theta: 0.0558,
        active_tail_hedge_count: 44,
        tail_hedge_convexity_payout_usd: 1765.62
      },
      evolution_generations_history: [
        {
          generation_id: 24,
          status_label: 'GEN-24 PROMOTED CHAMPION',
          is_current_champion: true,
          timestamp_iso: '2026-11-03T18:30:00Z',
          active_trades: 153,
          wins: 116,
          losses: 37,
          win_rate_pct: 75.82,
          tp_losses_vetoed: 20,
          fp_winners_vetoed: 0,
          entry_veto_precision_pct: 100.0,
          tp_losses_scratched: 37,
          fp_winners_scratched: 0,
          scratch_precision_pct: 100.0,
          early_scratches_bar1_to_3: 19,
          mid_trade_stall_saves_bar4_to_6: 18,
          giveback_saves_count: 19,
          total_giveback_r_recovered: 13.09,
          house_money_scaled_runners_count: 24,
          total_house_money_r_gained: 27.77,
          recycled_risk_pool_usd: 50507.68,
          recycled_runner_boost_count: 24,
          recycled_additional_r_compounded: 10.63,
          avg_execution_slippage_bps: 0.74,
          slippage_saved_usd: 2701.91,
          tail_hedge_ratio_theta: 0.0558,
          active_tail_hedge_count: 44,
          tail_hedge_convexity_payout_usd: 1765.62,
          pareto_gate_passed: true,
          avg_winner_r: 1.15,
          tp_losses_caught_total: 57,
          fp_winners_clipped_total: 0,
          remaining_unscratched_losses: 0,
          profit_factor: 29.36,
          kelly_profit_factor: 28.05,
          fixed_net_pnl_usd: 79464.72,
          kelly_net_pnl_usd: 127406.09,
          max_drawdown_usd: 234.0,
          composite_fitness_score: 168817.43,
          mutations_applied_summary: [
            'Gen 24: Pareto Gate Passed (116/116 Winners, 57/57 Full Stops Eliminated), Almgren-Chriss Slippage 0.74 bps (+$2,701.91 saved), Vanna Tail-Hedge θ*=5.58% (+$1,765.62 convexity)'
          ]
        }
      ],
      synaptic_plasticity_states: [
        {
          brain_id: 'BRAIN-02',
          generation_id: 24,
          synaptic_feature_weights_32: Array(32).fill(1.24),
          stream_prior_multipliers: [1.2, 1.15, 1.12, 1.28],
          wald_sprt_thresholds: [1.44, 1.36, 1.42],
          entry_veto_margin_delta: 0.015,
          trap_conviction_weights: {},
          empirical_bayes_shrinkage_lambda: 0.892,
          learned_almgren_chriss_risk_aversion_lambda: 0.42,
          learned_almgren_chriss_urgency_kappa: 0.94,
          learned_correlation_spike_quantile: 0.34,
          learned_vanna_tail_hedge_ratio_theta: 0.0558,
          top_boosted_warning_features: [],
          top_slashed_deceiver_features: [],
          cumulative_tp_vetoes: 4,
          cumulative_fp_vetoes: 0,
          cumulative_tp_scratches: 9,
          cumulative_fp_scratches: 0,
          remaining_unscratched_losses: 0,
          last_mutation_summary: 'Gen 24 promoted champion'
        }
      ]
    });

    const gen24 = store.evolutionGenerationsHistory[0]!;
    expect(gen24.generation_id).toBe(24);
    expect(gen24.winners_preserved_count).toBe(116);
    expect(gen24.remaining_full_stops).toBe(0);
    expect(gen24.pareto_gate_passed).toBe(true);
    expect(gen24.avg_execution_slippage_bps).toBeCloseTo(0.74, 2);
    expect(gen24.slippage_saved_usd).toBeCloseTo(2701.91, 2);
    expect(gen24.tail_hedge_ratio_theta).toBeCloseTo(0.0558, 4);
    expect(gen24.active_tail_hedge_count).toBe(44);
    expect(gen24.tail_hedge_convexity_payout_usd).toBeCloseTo(1765.62, 2);

    expect(store.portfolioHeatTelemetry.avgExecutionSlippageBps).toBeCloseTo(0.74, 2);
    expect(store.portfolioHeatTelemetry.slippageSavedUsd).toBeCloseTo(2701.91, 2);
    expect(store.portfolioHeatTelemetry.tailHedgeRatioTheta).toBeCloseTo(0.0558, 4);

    const ticket = await store.stageBrokerOrderTicket('OPEN-P23-NVDA', 'IBKR_TWS');
    expect(ticket).not.toBeNull();
    expect(ticket?.childExecutionSlices?.length).toBe(4);
    expect(ticket?.childExecutionSlices?.[0]?.equityPegStrategy).toBe(
      'IBKR_PEG_MID_ISLAND_PRIMARY_OFFSET'
    );
    expect(ticket?.childExecutionSlices?.[0]?.optionPegStrategy).toBe(
      'IBKR_IBUSOPT_PEGGED_TO_MIDPOINT_0005'
    );
  });

  it('hydrates Phase 24 Gen-28 Cross-Venue NBBO SOR Markout Toxicity, Dealer Charm/Vanna Unpinning, and OCC TIMS Portfolio Margin Relief', async () => {
    const store = new TerminalStore(false);
    store.ingestBackendSnapshot({
      positions_opened: [
        {
          position_id: 'OPEN-P24-NVDA',
          opened_timestamp_ns: 1_793_739_300_000_000_000,
          symbol: 'NVDA',
          exchange_index: 'NASDAQ_100',
          brain_id: 'BRAIN-02',
          direction: 'LONG',
          entry_price: 210.0,
          live_price: 221.4,
          quantity_shares: 120,
          kelly_fraction_pct: 2.15,
          risk_budget_usd: 1120.0,
          notional_allocated_usd: 25200.0,
          tranche_count: 2,
          structural_stop_price: 204.5,
          target_1_price: 222.0,
          target_2_price: 235.0,
          unrealized_stock_pnl_usd: 1368.0,
          unrealized_stock_pnl_pct: 5.43,
          option_structure: 'LONG_CALL_GAMMA',
          matched_option_contract: 'NVDA 2026-11-20 $210.00C',
          option_entry_premium: 6.4,
          option_live_premium: 13.2,
          option_contracts_count: 6,
          unrealized_option_pnl_usd: 4080.0,
          unrealized_option_pnl_pct: 106.25
        }
      ],
      closed_trades: [],
      portfolio_heat_telemetry: {
        account_equity_usd: 230449.62,
        open_positions_count: 5,
        aggregate_open_risk_usd: 2680.0,
        aggregate_open_risk_pct: 1.16,
        portfolio_heat_cap_pct: 8.5,
        gross_portfolio_notional_usd: 54000.0,
        net_portfolio_beta_delta_usd: 14800.0,
        net_portfolio_gamma_1pct_usd: 1920.0,
        net_portfolio_vanna_1pct_usd: 680.0,
        mean_pairwise_correlation: 0.18,
        max_pairwise_correlation: 0.34,
        correlation_heat_multiplier: 1.0,
        heat_regime_badge: 'LOW HEAT — FULL 1.00x KELLY CAPACITY',
        position_labels: ['NVDA (BRAIN-02)'],
        cross_position_correlation_matrix: [[1.0]],
        pairwise_correlations: [],
        overlapping_clusters_summary: 'Independent multi-horizon exposure',
        recycled_risk_pool_usd: 50507.68,
        recycled_runner_boost_count: 24,
        recycled_additional_r_compounded: 10.63,
        cross_brain_recycle_multiplier: 1.33,
        avg_execution_slippage_bps: 0.6,
        slippage_saved_usd: 3325.08,
        tail_hedge_ratio_theta: 0.0558,
        active_tail_hedge_count: 44,
        tail_hedge_convexity_payout_usd: 1765.62,
        sor_venue_weights_simplex: [0.169, 0.6063, 0.197, 0.0277],
        sor_markout_toxicity_bps: 0.45,
        charm_vanna_unpin_boosts_count: 24,
        charm_vanna_additional_r_gained: 2.32,
        reg_t_standalone_margin_usd: 23705.28,
        occ_tims_portfolio_margin_usd: 15035.94,
        occ_tims_pm_relief_ratio_pct: 43.06,
        return_on_margin_rom_pct: 541.47
      },
      evolution_generations_history: [
        {
          generation_id: 28,
          status_label: 'GEN-28 PROMOTED CHAMPION',
          is_current_champion: true,
          timestamp_iso: '2026-11-03T19:10:00Z',
          active_trades: 153,
          wins: 116,
          losses: 37,
          win_rate_pct: 75.82,
          tp_losses_vetoed: 20,
          fp_winners_vetoed: 0,
          entry_veto_precision_pct: 100.0,
          tp_losses_scratched: 37,
          fp_winners_scratched: 0,
          scratch_precision_pct: 100.0,
          early_scratches_bar1_to_3: 19,
          mid_trade_stall_saves_bar4_to_6: 18,
          giveback_saves_count: 19,
          total_giveback_r_recovered: 13.09,
          house_money_scaled_runners_count: 24,
          total_house_money_r_gained: 30.08,
          recycled_risk_pool_usd: 50507.68,
          recycled_runner_boost_count: 24,
          recycled_additional_r_compounded: 10.63,
          avg_execution_slippage_bps: 0.6,
          slippage_saved_usd: 3325.08,
          tail_hedge_ratio_theta: 0.0558,
          active_tail_hedge_count: 44,
          tail_hedge_convexity_payout_usd: 1765.62,
          sor_markout_toxicity_bps: 0.45,
          charm_vanna_unpin_boosts_count: 24,
          charm_vanna_additional_r_gained: 2.32,
          occ_tims_pm_relief_ratio_pct: 43.06,
          return_on_margin_rom_pct: 541.47,
          pareto_gate_passed: true,
          avg_winner_r: 1.15,
          tp_losses_caught_total: 57,
          fp_winners_clipped_total: 0,
          remaining_unscratched_losses: 0,
          profit_factor: 31.05,
          kelly_profit_factor: 29.61,
          fixed_net_pnl_usd: 81415.76,
          kelly_net_pnl_usd: 130449.62,
          max_drawdown_usd: 225.31,
          composite_fitness_score: 174226.84,
          mutations_applied_summary: [
            'Gen 28: Pareto Gate Passed (116/116 Winners, 57/57 Full Stops Eliminated), Cross-Venue SOR Slippage 0.60 bps (Markout Toxicity 0.45 bps), Charm/Vanna Unpin +2.32R, OCC TIMS PM Relief 43.06% (ROM 541.47%)'
          ]
        }
      ],
      synaptic_plasticity_states: [
        {
          brain_id: 'BRAIN-02',
          generation_id: 28,
          synaptic_feature_weights_32: Array(32).fill(1.26),
          stream_prior_multipliers: [1.22, 1.17, 1.14, 1.3],
          wald_sprt_thresholds: [1.44, 1.36, 1.42],
          entry_veto_margin_delta: 0.015,
          trap_conviction_weights: {},
          empirical_bayes_shrinkage_lambda: 0.892,
          learned_almgren_chriss_risk_aversion_lambda: 0.42,
          learned_almgren_chriss_urgency_kappa: 0.94,
          learned_correlation_spike_quantile: 0.34,
          learned_vanna_tail_hedge_ratio_theta: 0.0558,
          learned_sor_venue_weights_4: [0.169, 0.6063, 0.197, 0.0277],
          learned_sor_markout_toxicity_bps: 0.45,
          learned_charm_vanna_pin_quantile: -0.18,
          learned_charm_vanna_unpin_quantile: 0.34,
          learned_occ_tims_sigma_shock_pct: 7.82,
          top_boosted_warning_features: [],
          top_slashed_deceiver_features: [],
          cumulative_tp_vetoes: 4,
          cumulative_fp_vetoes: 0,
          cumulative_tp_scratches: 9,
          cumulative_fp_scratches: 0,
          remaining_unscratched_losses: 0,
          last_mutation_summary: 'Gen 28 promoted champion'
        }
      ]
    });

    const gen28 = store.evolutionGenerationsHistory[0]!;
    expect(gen28.generation_id).toBe(28);
    expect(gen28.winners_preserved_count).toBe(116);
    expect(gen28.remaining_full_stops).toBe(0);
    expect(gen28.profit_factor_fixed).toBeCloseTo(31.05, 2);
    expect(gen28.sor_markout_toxicity_bps).toBeCloseTo(0.45, 2);
    expect(gen28.charm_vanna_unpin_boosts_count).toBe(24);
    expect(gen28.charm_vanna_additional_r_gained).toBeCloseTo(2.32, 2);
    expect(gen28.occ_tims_pm_relief_ratio_pct).toBeCloseTo(43.06, 2);
    expect(gen28.return_on_margin_rom_pct).toBeCloseTo(541.47, 2);

    expect(store.portfolioHeatTelemetry.sorMarkoutToxicityBps).toBeCloseTo(0.45, 2);
    expect(store.portfolioHeatTelemetry.occTimsPmReliefRatioPct).toBeCloseTo(43.06, 2);
    expect(store.portfolioHeatTelemetry.returnOnMarginRomPct).toBeCloseTo(541.47, 2);

    const ticket = await store.stageBrokerOrderTicket('OPEN-P24-NVDA', 'IBKR_TWS');
    expect(ticket).not.toBeNull();
    expect(ticket?.sorExpectedMarkoutToxicityBps).toBeCloseTo(0.45, 2);
    expect(ticket?.childExecutionSlices?.[0]?.sorSelectedVenue).toBe('IBKR_PEG_MID');
  });

  it('hydrates Phase 28 Isolated Paper-Trading Overlay (173 untouched baseline + overlay trades), Atomic 2-Leg BAG Vertical Roll tickets (leg1_ratio: 1, leg2_ratio: 1, penny-reconciled credit), and Gen 36 synaptic plasticity shifts', async () => {
    const store = new TerminalStore(false);
    seedYahooBackendState(store);

    const paperTelemetryMock = {
      mode: 'CONTINUOUS_AUTONOMOUS_PAPER_OVERLAY',
      baseline_ledger_locked: true,
      baseline_trade_count: 173,
      baseline_sha256: 'a41f3200e74dcfd64158ad4ef9392124fea711068a87589b5417f4039fb071af',
      overlay_closed_count: 7,
      open_paper_positions: 2,
      recent_atomic_bag_rolls: 5,
      brain_plasticity_shifts: [
        {
          brain_id: 'BRAIN-01' as const,
          tau_veto: 0.082,
          eb_lambda: 0.884,
          learned_profit_lock_activation_r: 0.411,
          learned_conformal_veto_quantile: 0.045
        },
        {
          brain_id: 'BRAIN-02' as const,
          tau_veto: 0.076,
          eb_lambda: 0.892,
          learned_profit_lock_activation_r: 0.388,
          learned_conformal_veto_quantile: 0.042
        },
        {
          brain_id: 'BRAIN-03' as const,
          tau_veto: 0.091,
          eb_lambda: 0.865,
          learned_profit_lock_activation_r: 0.442,
          learned_conformal_veto_quantile: 0.048
        },
        {
          brain_id: 'BRAIN-04' as const,
          tau_veto: 0.068,
          eb_lambda: 0.905,
          learned_profit_lock_activation_r: 0.375,
          learned_conformal_veto_quantile: 0.038
        },
        {
          brain_id: 'BRAIN-05' as const,
          tau_veto: 0.055,
          eb_lambda: 0.925,
          learned_profit_lock_activation_r: 0.35,
          learned_conformal_veto_quantile: 0.032
        }
      ],
      timestamp_iso: '2026-11-09T22:00:00Z'
    };

    const originalFetch = globalThis.fetch;
    globalThis.fetch = vi.fn().mockImplementation(async (url: string) => {
      if (url.includes('/api/v1/scanner/paper-overlay')) {
        return {
          ok: true,
          json: async () => paperTelemetryMock
        };
      }
      if (url.includes('/api/v1/execution/stage-roll')) {
        return {
          ok: true,
          json: async () => ({
            roll_ticket_id: 'ROLL-OPEN-P22-SOFI',
            position_id: 'OPEN-P22-SOFI',
            staged_at_iso: '2026-11-09T22:00:00Z',
            symbol: 'SOFI',
            brain_id: 'BRAIN-02',
            direction: 'LONG',
            leg1_ratio: 1,
            leg2_ratio: 1,
            combo_order_type: 'IBKR_BAG_VERTICAL_ROLL_COMBO',
            net_combo_limit_price: 0.1,
            nbbo_limit_walk_ladder: [0.1, 0.08, 0.05],
            leg_1_close: {
              leg_id: 1,
              action: 'SELL_TO_CLOSE',
              ratio_qty: 5,
              occ_symbol: 'SOFI  261120C00017000',
              contract_desc: 'SOFI 2026-11-20 $17.00C',
              strike_price: 17.0,
              cboe_mid_price: 0.15
            },
            leg_2_open: {
              leg_id: 2,
              action: 'BUY_TO_OPEN',
              ratio_qty: 5,
              occ_symbol: 'SOFI  261120C00018000',
              contract_desc: 'SOFI 2026-11-20 $18.00C',
              strike_price: 18.0,
              cboe_mid_price: 0.05
            }
          })
        };
      }
      return { ok: false };
    });

    try {
      const data = await store.fetchPaperOverlayState();
      expect(data).not.toBeNull();
      expect(data?.baseline_ledger_locked).toBe(true);
      expect(data?.baseline_trade_count).toBe(173);
      expect(data?.baseline_sha256).toBe('a41f3200e74dcfd64158ad4ef9392124fea711068a87589b5417f4039fb071af');
      expect(data?.overlay_closed_count).toBe(7);
      expect(data?.brain_plasticity_shifts.length).toBe(5);

      // Verify Atomic 2-Leg BAG Combo Roll Ticket with 1:1 Leg Ratio and Penny-Reconciled Bounds
      const stagedRoll = await store.stageOptionRollTicket('OPEN-P22-SOFI');
      expect(stagedRoll).not.toBeNull();
      if (stagedRoll) {
        expect(stagedRoll.leg1Ratio ?? 1).toBe(1);
        expect(stagedRoll.leg2Ratio ?? 1).toBe(1);
        expect(stagedRoll.comboOrderType).toBe('IBKR_BAG_VERTICAL_ROLL_COMBO');
        const netMidDiff = Number((stagedRoll.leg1Close.liveMidPrice - stagedRoll.leg2Open.liveMidPrice).toFixed(2));
        expect(Math.abs(stagedRoll.netComboLimitPrice - netMidDiff)).toBeLessThanOrEqual(0.01);
      }
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('hydrates Phase 29/30 24-Pair Institutional Watchlist & Sunday Alpha Protocol Stager, enforces zero expired CBOE contracts, verifies Cross-Brain Conflict blocking on opposing multi-horizon signals, and validates Sunday Alpha CSV + Webhook dispatch state', async () => {
    const store = new TerminalStore(false);

    const nowIso = '2026-11-09T22:30:00Z';
    const evalEpochS = Math.floor(new Date(nowIso).getTime() / 1000);

    const watchlistMock = {
      universe_size: 24,
      actionable_setups_count: 19,
      active_generation: 'Gen 36',
      total_allocated_kelly_pct: 10.0,
      net_portfolio_delta_beta_weighted: -2.71,
      cross_brain_conflicts_blocked: 1,
      estimated_occ_tims_margin_relief_pct: 46.5,
      timestamp_iso: nowIso,
      eval_epoch_s: evalEpochS,
      watchlist: [
        {
          rank: 1,
          symbol: 'NVDA',
          brain_id: 'BRAIN-02' as const,
          brain_name: 'BRAIN-02 INTRADAY DAY TRADER',
          direction: 'LONG' as const,
          strategy_archetype: 'GAMMA_SQUEEZE_IGNITION',
          engine_a_confidence_pct: 98.8,
          engine_b_confidence_pct: 95.5,
          consensus_confidence_pct: 97.2,
          active_engine_mode: 'CONSENSUS_BOTH_ENGINES',
          entry_price: 178.6,
          structural_stop_price: 176.89,
          stop_distance_pct: 0.96,
          target_1_price: 184.24,
          target_2_price: 195.75,
          expected_r_multiple: 10.04,
          stage0_conformal_status: 'PASS' as const,
          mahalanobis_win_distance: 0.082,
          mahalanobis_loss_distance: 0.285,
          conformal_llr: 0.094,
          conformal_threshold: 0.05,
          conformal_margin: 0.044,
          action_status: 'ACTIONABLE_STAGE_READY',
          venue_ticket: 'IBKR_PEG_MID_SMART',
          matched_option_contract: 'NVDA 2026-11-13 $177.50C',
          option_structure: 'LONG_CALL',
          kelly_fraction_pct: 0.33,
          is_actionable: true
        },
        {
          rank: 9,
          symbol: 'AMD',
          brain_id: 'BRAIN-02' as const,
          brain_name: 'BRAIN-02 INTRADAY DAY TRADER',
          direction: 'SHORT' as const,
          strategy_archetype: 'POC_VOLUME_ABSORPTION',
          engine_a_confidence_pct: 94.7,
          engine_b_confidence_pct: 78.9,
          consensus_confidence_pct: 86.8,
          active_engine_mode: 'CONSENSUS_BOTH_ENGINES',
          entry_price: 509.01,
          structural_stop_price: 515.47,
          stop_distance_pct: 1.27,
          target_1_price: 493.87,
          target_2_price: 488.24,
          expected_r_multiple: 3.22,
          stage0_conformal_status: 'PASS' as const,
          mahalanobis_win_distance: 0.112,
          mahalanobis_loss_distance: 0.312,
          conformal_llr: 0.145,
          conformal_threshold: 0.048,
          conformal_margin: 0.097,
          action_status: 'ACTIONABLE_STAGE_READY',
          venue_ticket: 'IBKR_PEG_MID_SMART',
          matched_option_contract: 'AMD 2026-11-13 $510.00P',
          option_structure: 'LONG_PUT',
          kelly_fraction_pct: 0.31,
          is_actionable: true
        },
        {
          rank: 21,
          symbol: 'AMD',
          brain_id: 'BRAIN-04' as const,
          brain_name: 'BRAIN-04 CORE SWING',
          direction: 'LONG' as const,
          strategy_archetype: 'HTF_MULTI_SCALE_BREAKOUT',
          engine_a_confidence_pct: 71.2,
          engine_b_confidence_pct: 69.4,
          consensus_confidence_pct: 70.3,
          active_engine_mode: 'CONSENSUS_BOTH_ENGINES',
          entry_price: 509.01,
          structural_stop_price: 460.42,
          stop_distance_pct: 9.55,
          target_1_price: 585.36,
          target_2_price: 622.15,
          expected_r_multiple: 2.33,
          stage0_conformal_status: 'PASS' as const,
          mahalanobis_win_distance: 0.185,
          mahalanobis_loss_distance: 0.224,
          conformal_llr: 0.052,
          conformal_threshold: 0.045,
          conformal_margin: 0.007,
          action_status: 'BLOCKED_CROSS_BRAIN_CONFLICT',
          venue_ticket: 'SCHWAB_2LEG_NET_ROLL',
          matched_option_contract: 'AMD 2026-12-18 $510.00C',
          option_structure: 'LONG_CALL',
          kelly_fraction_pct: 0.28,
          is_actionable: false
        }
      ]
    };

    const sundayProtocolMock = {
      protocol_generation: 'Gen 36',
      staged_for_execution_epoch_iso: nowIso,
      universe_size: 24,
      actionable_count: 19,
      total_allocated_kelly_pct: 10.0,
      net_portfolio_delta_beta_weighted: -2.71,
      cross_brain_conflicts_blocked: 1,
      estimated_occ_tims_margin_relief_pct: 46.5,
      staged_setups: [
        {
          rank: 1,
          symbol: 'NVDA',
          brain_id: 'BRAIN-02' as const,
          brain_name: 'BRAIN-02 INTRADAY DAY TRADER',
          direction: 'LONG' as const,
          strategy_archetype: 'GAMMA_SQUEEZE_IGNITION',
          consensus_confidence_pct: 97.2,
          active_engine_mode: 'CONSENSUS_BOTH_ENGINES',
          entry_price: 178.6,
          structural_stop_price: 176.89,
          target_1_price: 184.24,
          target_2_price: 195.75,
          expected_r_multiple: 10.04,
          stage0_conformal_status: 'PASS',
          conformal_margin: 0.044,
          venue_ticket: 'IBKR_PEG_MID_SMART',
          matched_option_contract: 'NVDA 2026-11-13 $177.50C',
          kelly_fraction_pct: 0.33,
          action_status: 'ACTIONABLE_STAGE_READY'
        }
      ],
      csv_export_endpoint: '/api/v1/scanner/sunday-protocol/export/csv',
      webhook_dispatch_endpoint: '/api/v1/scanner/sunday-protocol/dispatch'
    };

    const dispatchMock = {
      status: 'STAGED_TO_OUTBOX',
      dispatched_at_iso: nowIso,
      discord_embed_queued: true,
      telegram_payload_queued: true,
      slack_blocks_queued: true,
      oms_bundle_queued: true,
      staged_count: 19,
      total_portfolio_kelly_pct: 10.0,
      net_delta_beta_weighted: -2.71,
      conflicts_blocked: 1
    };

    const originalFetch = globalThis.fetch;
    globalThis.fetch = vi.fn().mockImplementation(async (url: string) => {
      if (url.includes('/api/v1/scanner/institutional-watchlist')) {
        return { ok: true, json: async () => watchlistMock };
      }
      if (url.includes('/api/v1/scanner/sunday-protocol/dispatch')) {
        return { ok: true, json: async () => dispatchMock };
      }
      if (url.includes('/api/v1/scanner/sunday-protocol/export/csv')) {
        return {
          ok: true,
          blob: async () => new Blob(['Rank,Symbol\n1,NVDA'], { type: 'text/csv' })
        };
      }
      if (url.includes('/api/v1/scanner/sunday-protocol')) {
        return { ok: true, json: async () => sundayProtocolMock };
      }
      return { ok: false };
    });

    try {
      // 1. Hydrate Institutional Watchlist
      const wData = await store.fetchInstitutionalWatchlist();
      expect(wData).not.toBeNull();
      expect(wData?.universe_size).toBe(24);
      expect(wData?.actionable_setups_count).toBe(19);
      expect(wData?.total_allocated_kelly_pct).toBe(10.0);
      expect(wData?.cross_brain_conflicts_blocked).toBe(1);

      // 2. Zero Expired Option Contracts Verification
      const evalDate = new Date(nowIso);
      for (const row of wData?.watchlist ?? []) {
        const tokens = row.matched_option_contract.split(' ');
        expect(tokens.length).toBeGreaterThanOrEqual(2);
        const expDate = new Date(tokens[1]!);
        expect(expDate.getTime()).toBeGreaterThanOrEqual(evalDate.getTime());
      }

      // 3. Multi-Horizon Canonical Spot Reference & Cross-Brain Conflict Netting Gate
      const amdM5 = wData?.watchlist.find((r) => r.symbol === 'AMD' && r.brain_id === 'BRAIN-02');
      const amdD1 = wData?.watchlist.find((r) => r.symbol === 'AMD' && r.brain_id === 'BRAIN-04');
      expect(amdM5).toBeDefined();
      expect(amdD1).toBeDefined();
      expect(amdM5?.entry_price).toBe(509.01);
      expect(amdD1?.entry_price).toBe(509.01); // Coherent canonical spot!
      expect(amdM5?.direction).toBe('SHORT');
      expect(amdD1?.direction).toBe('LONG');
      // Cross-Brain Netting Gate blocks the opposing lower-expectancy setup
      expect(amdM5?.action_status).toBe('ACTIONABLE_STAGE_READY');
      expect(amdM5?.is_actionable).toBe(true);
      expect(amdD1?.action_status).toBe('BLOCKED_CROSS_BRAIN_CONFLICT');
      expect(amdD1?.is_actionable).toBe(false);

      // 4. Sunday Alpha Protocol & Webhook Outbox Dispatch
      const spData = await store.fetchSundayAlphaProtocol();
      expect(spData).not.toBeNull();
      expect(spData?.staged_setups.length).toBe(1);

      const dispatchRes = await store.dispatchSundayProtocolWebhook();
      expect(dispatchRes).not.toBeNull();
      expect(dispatchRes?.status).toBe('STAGED_TO_OUTBOX');
      expect(dispatchRes?.discord_embed_queued).toBe(true);
      expect(dispatchRes?.telegram_payload_queued).toBe(true);
      expect(dispatchRes?.conflicts_blocked).toBe(1);
      expect(store.lastDispatchResult).toEqual(dispatchMock);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('validates Rust-to-Svelte wire contract schema parity against real serde-serialized JSON fixture (#21)', async () => {
    const store = new TerminalStore();
    const originalFetch = globalThis.fetch;

    globalThis.fetch = vi.fn().mockImplementation(async (url: string) => {
      if (url.includes('/api/v1/scanner/institutional-watchlist')) {
        return { ok: true, json: async () => rustWireFixture };
      }
      return { ok: false };
    });

    try {
      const data = await store.fetchInstitutionalWatchlist();
      expect(data).not.toBeNull();
      // Verify root fields
      expect(data?.universe_size).toBe(24);
      expect(data?.actionable_setups_count).toBe(rustWireFixture.actionable_setups_count);
      expect(data?.actionable_setups_count).toBe(data?.watchlist.filter((r) => r.is_actionable).length);
      expect(data?.active_generation_id).toBe(36);
      expect(data?.active_generation).toBe('GEN-36');
      expect(data?.total_allocated_kelly_pct).toBe(10.0);
      expect(data?.net_portfolio_delta_beta_weighted).toBe(rustWireFixture.net_portfolio_delta_beta_weighted);
      expect(data?.cross_brain_conflicts_blocked).toBe(rustWireFixture.cross_brain_conflicts_blocked);
      // Zero hardcoded 45.2 proof: dynamic OCC TIMS margin relief
      expect(data?.estimated_occ_tims_margin_relief_pct).toBe(rustWireFixture.estimated_occ_tims_margin_relief_pct);
      expect(data?.estimated_occ_tims_margin_relief_pct).toBeGreaterThan(15.0);
      expect(data?.estimated_occ_tims_margin_relief_pct).not.toBe(45.2);

      // Verify row-level wire contract parity
      expect(data?.watchlist.length).toBe(24);
      for (const row of data?.watchlist ?? []) {
        expect(row.symbol).toBeDefined();
        expect(row.brain_id).toMatch(/^BRAIN-0[1-5]$/);
        expect(['LONG', 'SHORT']).toContain(row.direction);
        expect(row.consensus_confidence_pct).toBeGreaterThan(0);
        expect(row.entry_price).toBeGreaterThan(0);
        expect(row.structural_stop_price).toBeGreaterThan(0);
        expect(row.target_1_price).toBeGreaterThan(0);
        expect(row.target_2_price).toBeGreaterThan(0);
        expect(row.matched_option_contract).toMatch(/^(SHARES_ONLY|[A-Z]+\s+\d{4}-\d{2}-\d{2}\s+\$[\d\.]+(\/\$[\d\.]+)*[CP])/);
        expect(row.venue_ticket).toBeDefined();
        expect(row.kelly_fraction_pct).toBeGreaterThan(0);
        expect(typeof row.is_actionable).toBe('boolean');
        expect(typeof row.action_status).toBe('string');
        // Ensure zero undefined fields from snake_case / camelCase mismatch
        expect(row.conformal_margin).toBeDefined();
        expect(row.engine_a_confidence_pct).toBeDefined();
        expect(row.engine_b_confidence_pct).toBeDefined();
      }

      // Verify Cross-Brain conflict netting in fixture
      const blockedRows = (data?.watchlist ?? []).filter((r) => r.action_status === 'BLOCKED_CROSS_BRAIN_CONFLICT');
      expect(blockedRows.length).toBe(rustWireFixture.cross_brain_conflicts_blocked);
      for (const blockedRow of blockedRows) {
        expect(blockedRow.is_actionable).toBe(false);
      }
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('stages approved Sunday Alpha orders into isolated Paper-OMS with hardware safety lock and validates redacted webhook configuration + test ping (#22)', async () => {
    const store = new TerminalStore();
    const originalFetch = globalThis.fetch;

    const paperReceiptMock: StagedOrderReceipt = {
      client_order_id: 'PAPER-AMD-BRAIN-02-1762780824',
      symbol: 'AMD',
      brain_id: 'BRAIN-02',
      direction: 'SHORT',
      execution_safety_mode: 'PAPER_ISOLATED_ONLY',
      execution_venue: 'IEX_DLIMIT_DAY',
      limit_entry_price: 483.3,
      structural_stop_loss: 488.29,
      profit_target_1: 467.96,
      profit_target_2: 461.83,
      kelly_fraction_pct: 0.625,
      allocated_notional_usd: 6250.0,
      matched_cboe_contract: 'AMD 2026-11-20 $480.00P',
      almgren_chriss_slices: [0.4, 0.35, 0.25],
      staged_at_iso: '2026-11-10T13:20:24Z'
    };

    const stageSuccessMock: StageSundayOrdersResponse = {
      status: 'SUCCESS',
      execution_safety_mode: 'PAPER_ISOLATED_ONLY',
      staged_orders_count: 1,
      total_staged_kelly_pct: 0.625,
      receipts: [paperReceiptMock],
      open_paper_positions_count: 6,
      ledger_baseline_trades_untouched: 173
    };

    const safetyLockErrorMock: StageSundayOrdersErrorResponse = {
      status: 'REJECTED_SAFETY_LOCK',
      error_code: 'PAPER_SAFETY_LOCK_ACTIVE',
      message: 'Live brokerage routing blocked by hardware safety latch: ARGON_ALLOW_LIVE_BROKER_ROUTING=1 and token required',
      execution_safety_mode: 'PAPER_ISOLATED_ONLY'
    };

    const redactedWebhookConfigMock: RedactedWebhookConfigResponse = {
      discord_webhook_url_masked: 'https://discord.com/api/webhooks/1234...****abcd',
      discord_configured: true,
      discord_enabled: true,
      telegram_bot_token_masked: '987654...****xyz',
      telegram_chat_id_masked: '-100...****789',
      telegram_configured: true,
      telegram_enabled: true,
      configured: true,
      dry_run_mode: true,
      outbox_count: 4,
      active_channels: ['DISCORD', 'TELEGRAM']
    };

    const testPingMock: WebhookTestPingResponse = {
      status: 'TEST_PING_ACCEPTED',
      delivery_status: 'RECORDED_IN_OUTBOX_DRY_RUN',
      receipt_id: 'PING-REC-1762780824',
      dispatched_at_iso: '2026-11-10T13:20:24Z',
      channels_targeted: ['DISCORD', 'TELEGRAM'],
      dry_run_mode: true,
      outbox_count: 5
    };

    globalThis.fetch = vi.fn().mockImplementation(async (url: string, init?: RequestInit) => {
      if (url.includes('/api/v1/scanner/sunday-protocol/stage-orders')) {
        const body = init?.body ? JSON.parse(init.body as string) : {};
        if (body.live === true && !body.confirmation_token) {
          return {
            ok: false,
            status: 403,
            json: async () => safetyLockErrorMock
          };
        }
        return {
          ok: true,
          status: 200,
          json: async () => stageSuccessMock
        };
      }
      if (url.includes('/api/v1/scanner/webhooks/config')) {
        if (init?.method === 'POST') {
          return {
            ok: true,
            status: 200,
            json: async () => ({
              ...redactedWebhookConfigMock,
              outbox_count: 5
            })
          };
        }
        return {
          ok: true,
          status: 200,
          json: async () => redactedWebhookConfigMock
        };
      }
      if (url.includes('/api/v1/scanner/webhooks/test-ping')) {
        return {
          ok: true,
          status: 200,
          json: async () => testPingMock
        };
      }
      if (url.includes('/api/v1/scanner/paper-overlay')) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            active_engine_mode: 'PAPER_OVERLAY_ACTIVE',
            baseline_closed_count: 173,
            overlay_closed_count: 0,
            total_active_closed_count: 173,
            open_paper_positions: 6,
            paper_positions: []
          })
        };
      }
      return { ok: false };
    });

    try {
      // 1. Hardware Safety Lock Enforcement: attempt unauthorized live routing -> must fail with 403
      const liveAttempt = await store.stageSundayOrders({
        symbols: ['AMD'],
        live: true
      });
      expect(liveAttempt).toBeNull();
      expect(store.lastStagingError).not.toBeNull();
      expect(store.lastStagingError?.error_code).toBe('PAPER_SAFETY_LOCK_ACTIVE');
      expect(store.lastStagingError?.execution_safety_mode).toBe('PAPER_ISOLATED_ONLY');

      // 2. Paper-Isolated Staging: approved setup staged into isolated OMS
      const paperOrder = await store.stageSundayOrders({
        symbols: ['AMD']
      });
      expect(paperOrder).not.toBeNull();
      expect(paperOrder?.status).toBe('SUCCESS');
      expect(paperOrder?.execution_safety_mode).toBe('PAPER_ISOLATED_ONLY');
      expect(paperOrder?.staged_orders_count).toBe(1);
      expect(paperOrder?.ledger_baseline_trades_untouched).toBe(173);
      expect(paperOrder?.receipts.length).toBe(1);
      expect(paperOrder?.receipts[0]?.client_order_id).toBe('PAPER-AMD-BRAIN-02-1762780824');
      expect(paperOrder?.receipts[0]?.almgren_chriss_slices).toEqual([0.4, 0.35, 0.25]);
      expect(store.stagedOrdersReceipts.length).toBe(1);

      // 3. Strict Secret Redaction: GET webhooks config must never expose raw credentials
      const whConfig = await store.fetchWebhookConfig();
      expect(whConfig).not.toBeNull();
      expect(whConfig?.discord_webhook_url_masked).toContain('****');
      expect(whConfig?.telegram_bot_token_masked).toContain('****');
      expect(whConfig?.telegram_chat_id_masked).toContain('****');
      expect(whConfig?.dry_run_mode).toBe(true);

      // 4. Update Webhook Config & Execute Test Ping
      const updatedConfig = await store.updateWebhookConfig({
        discord_webhook_url: 'https://discord.com/api/webhooks/new-token-123456'
      });
      expect(updatedConfig).not.toBeNull();

      const pingRes = await store.testPingWebhook();
      expect(pingRes).not.toBeNull();
      expect(pingRes?.status).toBe('TEST_PING_ACCEPTED');
      expect(pingRes?.delivery_status).toBe('RECORDED_IN_OUTBOX_DRY_RUN');
      expect(pingRes?.receipt_id).toBe('PING-REC-1762780824');
      expect(store.lastTestPingResult).toEqual(testPingMock);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('validates Phase 32 Closed-Loop Loss-Autopsy & Dual-Gate Pareto Evolution wire parity (#23)', async () => {
    const store = new TerminalStore(false);
    const originalFetch = globalThis.fetch;

    globalThis.fetch = vi.fn().mockImplementation(async (url: string) => {
      if (url.includes('/api/v1/scanner/autopsy-evolution')) {
        return {
          ok: true,
          status: 200,
          json: async () => autopsyWireFixture
        };
      }
      if (url.includes('/api/v1/scanner/evolve-cycle')) {
        return {
          ok: true,
          status: 200,
          json: async () => autopsyWireFixture
        };
      }
      if (url.includes('/api/v1/evolution/generations')) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            current_champion_generation: autopsyWireFixture.candidate_generation_id,
            generations_history: [],
            synaptic_plasticity_states: []
          })
        };
      }
      return { ok: false };
    });

    try {
      // 1. Fetch Autopsy Evolution State
      const autoRes = await store.fetchAutopsyEvolution();
      expect(autoRes).not.toBeNull();
      expect(store.autopsyEvolutionState).toEqual(autopsyWireFixture);
      expect(store.autopsyEvolutionState?.candidate_generation_id).toBeGreaterThanOrEqual(37);
      expect(store.autopsyEvolutionState?.is_promoted).toBe(true);
      expect(store.autopsyEvolutionState?.status).toBe('PROMOTED_CHAMPION');

      // 2. Dual-Gate Pareto Audit Telemetry
      const audit = store.autopsyEvolutionState!.gate_audit;
      expect(audit.gate1_baseline_lock_passed).toBe(true);
      expect(audit.gate1_baseline_winners_preserved).toBe(116);
      expect(audit.gate1_baseline_winners_target).toBe(116);
      expect(audit.gate1_candidate_expectancy_r).toBeGreaterThan(0.20);
      expect(audit.gate1_candidate_pf).toBeGreaterThanOrEqual(1.50);
      expect(audit.gate2_walkforward_passed).toBe(true);
      expect(audit.gate2_candidate_walkforward_net_r).toBeGreaterThan(audit.gate2_current_walkforward_net_r);
      expect(audit.gate2_offending_losses_reduced).toBeGreaterThanOrEqual(1);
      expect(audit.gate2_walkforward_winners_preserved).toBe(true);

      // 3. Decoded Trade Autopsies & 32-Feature Blame Attribution
      const autopsies = store.autopsyEvolutionState!.autopsies_decoded;
      expect(autopsies.length).toBeGreaterThanOrEqual(2);
      const modes = autopsies.map((a) => a.failure_mode);
      expect(modes).toContain('ENTRY_LIQUIDITY_VACUUM_TRAP');
      expect(modes).toContain('MFE_STALL_GIVEBACK_EXHAUSTION');

      for (const aut of autopsies) {
        expect(aut.blame_attribution_32.length).toBe(32);
        expect(aut.top_culprit_names.length).toBe(5);
        expect(aut.top_culprit_feature_indices.length).toBe(5);
        expect(aut.mahalanobis_distance_shift).toBeGreaterThan(0.0);
        expect(typeof aut.summary).toBe('string');
        expect(aut.summary.length).toBeGreaterThan(5);
      }

      // 4. Trigger Run Self-Learning Evolution Cycle
      const runRes = await store.runEvolveCycle();
      expect(runRes).not.toBeNull();
      expect(runRes?.is_promoted).toBe(true);
      expect(store.autopsyEvolutionState?.candidate_generation_id).toBe(autopsyWireFixture.candidate_generation_id);

      // Drawer toggle
      store.toggleEvolutionDrawer(true);
      expect(store.evolutionDrawerOpen).toBe(true);
      store.toggleEvolutionDrawer();
      expect(store.evolutionDrawerOpen).toBe(false);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('filters 24-pair institutional watchlist across execution tiers (SCALP_0DTE, DAY, SWING, WEALTH) (#24)', async () => {
    const store = new TerminalStore(false);
    const originalFetch = globalThis.fetch;

    globalThis.fetch = vi.fn().mockImplementation(async (url: string) => {
      if (url.includes('/api/v1/scanner/watchlist') || url.includes('/api/v1/scanner/institutional-watchlist')) {
        return {
          ok: true,
          status: 200,
          json: async () => rustWireFixture
        };
      }
      return { ok: false };
    });

    try {
      await store.fetchInstitutionalWatchlist();
      expect(store.institutionalWatchlist).not.toBeNull();
      const rows = store.institutionalWatchlist!.watchlist;
      expect(rows.length).toBe(24);

      // Verify execution tier presence on all 24 rows
      for (const r of rows) {
        expect(r.execution_tier).toBeDefined();
        expect(typeof r.execution_tier).toBe('string');
        expect(['SCALP_0DTE', 'DAY_TRADE', 'SWING', 'LONG_TERM_WEALTH']).toContain(r.execution_tier);
      }

      // Tier: SCALP_0DTE (SPY, QQQ, NVDA, TSLA)
      store.setSelectedExecutionTier('SCALP_0DTE');
      expect(store.selectedExecutionTier).toBe('SCALP_0DTE');
      const scalpRows = rows.filter((r) => r.execution_tier === 'SCALP_0DTE');
      expect(scalpRows.length).toBeGreaterThanOrEqual(2);
      const scalpSymbols = scalpRows.map((r) => r.symbol);
      expect(scalpSymbols).toContain('SPY');
      expect(scalpSymbols).toContain('QQQ');

      // Tier: DAY_TRADE
      store.setSelectedExecutionTier('DAY_TRADE');
      const dayRows = rows.filter((r) => r.execution_tier === 'DAY_TRADE');
      expect(dayRows.length).toBeGreaterThanOrEqual(4);

      // Tier: SWING
      store.setSelectedExecutionTier('SWING');
      const swingRows = rows.filter((r) => r.execution_tier === 'SWING');
      expect(swingRows.length).toBeGreaterThanOrEqual(5);

      // Tier: LONG_TERM_WEALTH
      store.setSelectedExecutionTier('LONG_TERM_WEALTH');
      const wealthRows = rows.filter((r) => r.execution_tier === 'LONG_TERM_WEALTH');
      expect(wealthRows.length).toBeGreaterThanOrEqual(1);

      // Reset to ALL
      store.setSelectedExecutionTier('ALL');
      expect(store.selectedExecutionTier).toBe('ALL');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('verifies real bar-replay Gate 1/2 metrics, baseline PF/Expectancy comparison, and closed-loop hard-negative trap count from Rust wire fixture (#25)', async () => {
    const store = new TerminalStore(false);
    const originalFetch = globalThis.fetch;

    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/api/v1/scanner/autopsy-evolution') || url.includes('/api/v1/scanner/evolve-cycle')) {
        return new Response(JSON.stringify(autopsyWireFixture), {
          status: 200,
          headers: { 'Content-Type': 'application/json' }
        });
      }
      if (url.includes('/api/v1/evolution/status')) {
        return new Response(JSON.stringify({ generations: [], synaptic_states: [] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' }
        });
      }
      return new Response('{}', { status: 404 });
    }) as unknown as typeof fetch;

    try {
      const fetched = await store.fetchAutopsyEvolution();
      expect(fetched).not.toBeNull();
      const cycled = await store.runEvolveCycle();
      expect(cycled).not.toBeNull();

      const state = store.autopsyEvolutionState!;
      expect(state.is_promoted).toBe(true);
      expect(state.promoted_generation_id).toBe(37);
      expect(state.hard_negative_traps_added_count).toBeGreaterThanOrEqual(1);

      const audit = state.gate_audit;
      expect(audit.gate1_baseline_lock_passed).toBe(true);
      expect(audit.gate1_baseline_winners_preserved).toBe(116);
      expect(audit.gate1_baseline_winners_target).toBe(116);
      expect(Number.isFinite(audit.current_baseline_exp_r)).toBe(true);
      expect(Number.isFinite(audit.candidate_baseline_exp_r)).toBe(true);
      expect(audit.candidate_baseline_exp_r).toBeGreaterThanOrEqual(audit.current_baseline_exp_r - 1e-6);
      expect(Number.isFinite(audit.current_baseline_pf)).toBe(true);
      expect(Number.isFinite(audit.candidate_baseline_pf)).toBe(true);
      expect(audit.candidate_baseline_pf).toBeGreaterThanOrEqual(audit.current_baseline_pf - 1e-6);

      expect(audit.gate2_walkforward_passed).toBe(true);
      expect(audit.gate2_walkforward_winners_preserved).toBe(true);
      expect(audit.candidate_wf_net_r).toBeGreaterThan(audit.current_wf_net_r);
      expect(audit.hard_negative_traps_added_count).toBe(state.hard_negative_traps_added_count);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('hydrates Phase 34 Mondrian regime conformal p-values, Chi^2_32 OOD Kelly multipliers, Shadow-Tape Net Veto Alpha, auto-quarantine counts, and 32-Simplex Mirror KL telemetry (#26)', async () => {
    const store = new TerminalStore(false);
    const originalFetch = globalThis.fetch;

    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (
        url.includes('/api/v1/scanner/watchlist') ||
        url.includes('/api/v1/scanner/institutional-watchlist') ||
        url.includes('/api/v1/scanner/sunday-watchlist')
      ) {
        return new Response(JSON.stringify(rustWireFixture), {
          status: 200,
          headers: { 'Content-Type': 'application/json' }
        });
      }
      if (url.includes('/api/v1/scanner/autopsy-evolution') || url.includes('/api/v1/scanner/evolve-cycle')) {
        return new Response(JSON.stringify(phase34ShadowWireFixture), {
          status: 200,
          headers: { 'Content-Type': 'application/json' }
        });
      }
      if (url.includes('/api/v1/evolution/status')) {
        return new Response(JSON.stringify({ generations: [], synaptic_states: [] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' }
        });
      }
      return new Response('{}', { status: 404 });
    }) as unknown as typeof fetch;

    try {
      await store.fetchInstitutionalWatchlist();
      expect(store.institutionalWatchlist).not.toBeNull();
      const rows = store.institutionalWatchlist!.watchlist;
      expect(rows.length).toBe(24);

      const validRegimes = [
        'POS_GAMMA_PIN_MEAN_REVERT',
        'NEG_GAMMA_MOMENTUM_TREND',
        'LIQUIDITY_VACUUM_HIGH_TOXICITY'
      ];
      for (const r of rows) {
        expect(validRegimes).toContain(r.regime_bucket);
        expect(Number.isFinite(r.conformal_p_value)).toBe(true);
        expect(r.conformal_p_value!).toBeGreaterThan(0);
        expect(r.conformal_p_value!).toBeLessThanOrEqual(1.0);
        expect(Number.isFinite(r.epistemic_ood_score)).toBe(true);
        expect(r.epistemic_ood_score!).toBeGreaterThanOrEqual(0);
        expect(Number.isFinite(r.kelly_epistemic_multiplier)).toBe(true);
        expect(r.kelly_epistemic_multiplier!).toBeGreaterThanOrEqual(0.15);
        expect(r.kelly_epistemic_multiplier!).toBeLessThanOrEqual(1.0);
      }

      const evolved = await store.runEvolveCycle();
      expect(evolved).not.toBeNull();
      const evoState = store.autopsyEvolutionState!;
      expect(evoState.shadow_vetoed_evaluated_count!).toBeGreaterThanOrEqual(4);
      expect(Number.isFinite(evoState.shadow_net_veto_alpha_r)).toBe(true);
      expect(evoState.quarantined_traps_count!).toBeGreaterThanOrEqual(1);
      expect(Number.isFinite(evoState.mirror_descent_kl_divergence)).toBe(true);
      expect(evoState.mirror_descent_kl_divergence!).toBeGreaterThan(0);

      expect(evoState.gate_audit.shadow_vetoed_evaluated_count).toBe(evoState.shadow_vetoed_evaluated_count);
      expect(evoState.gate_audit.shadow_net_veto_alpha_r!).toBeCloseTo(evoState.shadow_net_veto_alpha_r!, 6);
      expect(evoState.gate_audit.quarantined_traps_count).toBe(evoState.quarantined_traps_count);
      expect(evoState.gate_audit.mirror_descent_kl_divergence!).toBeCloseTo(
        evoState.mirror_descent_kl_divergence!,
        6
      );
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('hydrates Phase 35 Portfolio Eigen-Covariance AR/N_eff, Contextual NIG Simplex Meta-Allocator, Smart Options Convexity Router, and 0DTE Theta-Burn Stall Guard (#27)', async () => {
    const store = new TerminalStore(false);
    const originalFetch = globalThis.fetch;

    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (
        url.includes('/api/v1/scanner/watchlist') ||
        url.includes('/api/v1/scanner/institutional-watchlist') ||
        url.includes('/api/v1/scanner/sunday-watchlist')
      ) {
        return new Response(JSON.stringify(phase35EigenConvexityWireFixture.watchlist_response), {
          status: 200,
          headers: { 'Content-Type': 'application/json' }
        });
      }
      return new Response('{}', { status: 404 });
    }) as unknown as typeof fetch;

    try {
      await store.fetchInstitutionalWatchlist();
      expect(store.institutionalWatchlist).not.toBeNull();
      const wl = store.institutionalWatchlist!;

      // Book-level Eigen-Covariance & Contextual NIG Simplex assertions
      expect(Number.isFinite(wl.absorption_ratio)).toBe(true);
      expect(wl.absorption_ratio!).toBeGreaterThan(0);
      expect(wl.absorption_ratio!).toBeLessThanOrEqual(1.0);
      expect(Number.isFinite(wl.effective_independent_bets_neff)).toBe(true);
      expect(wl.effective_independent_bets_neff!).toBeGreaterThanOrEqual(1.0);
      expect(Number.isFinite(wl.eigen_kelly_multiplier)).toBe(true);
      expect(wl.eigen_kelly_multiplier!).toBeGreaterThanOrEqual(0.25);
      expect(wl.eigen_kelly_multiplier!).toBeLessThanOrEqual(1.0);

      expect(wl.contextual_nig_allocations).toBeDefined();
      expect(wl.contextual_nig_allocations!.length).toBe(6);
      const sumPi = wl.contextual_nig_allocations!.reduce((acc, row) => acc + row.capital_weight_pi, 0);
      expect(sumPi).toBeCloseTo(1.0, 5);
      for (const alloc of wl.contextual_nig_allocations!) {
        expect(alloc.capital_weight_pi).toBeGreaterThanOrEqual(0.0999);
        expect(alloc.capital_weight_pi).toBeLessThanOrEqual(0.4201);
        expect(Number.isFinite(alloc.posterior_t_sharpe)).toBe(true);
      }

      // Row-level Conformal + Eigen + Contextual + Convexity wire assertions
      expect(wl.watchlist.length).toBe(24);
      for (const r of wl.watchlist) {
        expect(['SINGLE_LONG_CONVEX', 'DEFINED_RISK_VERTICAL_BAG']).toContain(r.recommended_option_structure);
        expect(Number.isFinite(r.gamma_theta_efficiency)).toBe(true);
        expect(r.gamma_theta_efficiency!).toBeGreaterThanOrEqual(0);
        expect(Number.isFinite(r.theta_burn_hazard)).toBe(true);
        expect(r.theta_burn_hazard!).toBeGreaterThanOrEqual(0);
        expect(r.theta_burn_hazard!).toBeLessThanOrEqual(1.0);
        expect(Number.isFinite(r.conformal_kelly_scalar)).toBe(true);
        expect(r.conformal_kelly_scalar!).toBeGreaterThanOrEqual(0.35);
        expect(r.conformal_kelly_scalar!).toBeLessThanOrEqual(1.0);
        expect(Number.isFinite(r.eigen_kelly_multiplier)).toBe(true);
        expect(Number.isFinite(r.contextual_brain_weight_pi)).toBe(true);
      }

      // Orthogonal vs Crowded Eigen-Risk & Regime Contextual Shift & 0DTE Theta Stall audit proofs
      const orth = phase35EigenConvexityWireFixture.eigen_orthogonal_report;
      const crowd = phase35EigenConvexityWireFixture.eigen_crowded_report;
      expect(orth.absorption_ratio).toBeLessThan(0.55);
      expect(orth.effective_independent_bets_neff).toBeGreaterThan(2.5);
      expect(orth.vetoed).toBe(false);
      expect(crowd.absorption_ratio).toBeGreaterThan(0.78);
      expect(crowd.vetoed).toBe(true);
      expect(crowd.veto_reason).toBe('PORTFOLIO_VETOED_EIGEN_CORRELATION_CROWDING');

      const negB02 = phase35EigenConvexityWireFixture.nig_neg_gamma_allocations.find(
        (a) => a.slot_id === 'BRAIN-02'
      )!;
      const negScalp = phase35EigenConvexityWireFixture.nig_neg_gamma_allocations.find(
        (a) => a.slot_id === 'SCALP_0DTE'
      )!;
      const posScalp = phase35EigenConvexityWireFixture.nig_pos_gamma_allocations.find(
        (a) => a.slot_id === 'SCALP_0DTE'
      )!;
      const posB02 = phase35EigenConvexityWireFixture.nig_pos_gamma_allocations.find(
        (a) => a.slot_id === 'BRAIN-02'
      )!;
      expect(negB02.capital_weight_pi).toBeGreaterThan(negScalp.capital_weight_pi);
      expect(posScalp.capital_weight_pi).toBeGreaterThan(posB02.capital_weight_pi);

      expect(phase35EigenConvexityWireFixture.convexity_low_iv_trend.recommended_option_structure).toBe(
        'SINGLE_LONG_CONVEX'
      );
      expect(phase35EigenConvexityWireFixture.convexity_high_iv_pin.recommended_option_structure).toBe(
        'DEFINED_RISK_VERTICAL_BAG'
      );

      const guarded = phase35EigenConvexityWireFixture.zero_dte_stall_guarded;
      const unguarded = phase35EigenConvexityWireFixture.zero_dte_stall_unguarded;
      expect(guarded.exited_via_theta_stall).toBe(true);
      expect(guarded.exit_bar_index).toBe(4);
      expect(guarded.realized_option_r - unguarded.realized_option_r).toBeGreaterThan(0.35);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('hydrates Phase 36 Dealer Charm/Vanna Greeks, BOCPD Gamma-Flip Adaptive Memory, Ville Martingale E-Process Alpha Shield, and Microprice Execution Router (#28)', async () => {
    const store = new TerminalStore(false);
    const originalFetch = globalThis.fetch;

    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (
        url.includes('/api/v1/scanner/watchlist') ||
        url.includes('/api/v1/scanner/institutional-watchlist') ||
        url.includes('/api/v1/scanner/sunday-watchlist')
      ) {
        return new Response(JSON.stringify(phase36ApexWireFixture.watchlist_response), {
          status: 200,
          headers: { 'Content-Type': 'application/json' }
        });
      }
      return new Response('{}', { status: 404 });
    }) as unknown as typeof fetch;

    try {
      await store.fetchInstitutionalWatchlist();
      expect(store.institutionalWatchlist).not.toBeNull();
      const wl = store.institutionalWatchlist!;

      // System Header Telemetry: BOCPD & Adaptive Memory & Ville Martingale Alpha Shield
      expect(Number.isFinite(wl.bocpd_changepoint_prob)).toBe(true);
      expect(wl.bocpd_changepoint_prob!).toBeGreaterThanOrEqual(0.0);
      expect(wl.bocpd_changepoint_prob!).toBeLessThanOrEqual(1.0);
      expect(Number.isFinite(wl.bocpd_expected_run_length)).toBe(true);
      expect(wl.bocpd_expected_run_length!).toBeGreaterThanOrEqual(0.0);
      expect(Number.isFinite(wl.adaptive_forgetting_gamma)).toBe(true);
      expect(wl.adaptive_forgetting_gamma!).toBeGreaterThanOrEqual(0.76);
      expect(wl.adaptive_forgetting_gamma!).toBeLessThanOrEqual(0.992);

      expect(wl.martingale_shield_states).toBeDefined();
      expect(wl.martingale_shield_states!.length).toBeGreaterThanOrEqual(5);
      for (const m of wl.martingale_shield_states!) {
        expect(m.slot_id).toBeDefined();
        expect(Number.isFinite(m.martingale_e_value)).toBe(true);
        expect(m.martingale_e_value).toBeGreaterThanOrEqual(0.25);
        expect(Number.isFinite(m.martingale_kelly_multiplier)).toBe(true);
        expect(m.martingale_kelly_multiplier).toBeGreaterThanOrEqual(0.15);
        expect(m.martingale_kelly_multiplier).toBeLessThanOrEqual(1.0);
        expect(typeof m.probation_shadow_only).toBe('boolean');
      }

      // Per-watchlist row assertions: Surface Greeks, Microprice Execution Plan, Martingale Shield
      expect(wl.watchlist.length).toBe(24);
      for (const r of wl.watchlist) {
        expect(Number.isFinite(r.optimal_long_strike)).toBe(true);
        expect(r.optimal_long_strike!).toBeGreaterThan(0);
        if (r.recommended_option_structure === 'DEFINED_RISK_VERTICAL_BAG') {
          expect(r.optimal_short_strike).not.toBeNull();
          expect(Number.isFinite(r.optimal_short_strike!)).toBe(true);
        }
        expect(Number.isFinite(r.net_delta)).toBe(true);
        expect(Number.isFinite(r.net_gamma)).toBe(true);
        expect(Number.isFinite(r.net_theta_per_bar)).toBe(true);
        expect(Number.isFinite(r.net_vanna)).toBe(true);
        expect(Number.isFinite(r.net_charm)).toBe(true);

        expect(['PASSIVE_MID_PEG_LIMIT', 'ADAPTIVE_MICROPRICE_PEG', 'IMMEDIATE_ALMGREN_CHRISS_SWEEP']).toContain(
          r.execution_tactic
        );
        expect(Number.isFinite(r.microprice_limit_price)).toBe(true);
        expect(r.microprice_limit_price!).toBeGreaterThan(0);
        expect(Number.isFinite(r.execution_urgency_score)).toBe(true);
        expect(r.execution_urgency_score!).toBeGreaterThanOrEqual(0.0);
        expect(r.execution_urgency_score!).toBeLessThanOrEqual(1.0);
        expect(Number.isFinite(r.expected_slippage_bps)).toBe(true);

        expect(Number.isFinite(r.martingale_e_value)).toBe(true);
        expect(r.martingale_e_value!).toBeGreaterThanOrEqual(0.25);
        expect(Number.isFinite(r.martingale_kelly_multiplier)).toBe(true);
        expect(r.martingale_kelly_multiplier!).toBeGreaterThanOrEqual(0.15);
        expect(r.martingale_kelly_multiplier!).toBeLessThanOrEqual(1.0);
        expect(typeof r.probation_shadow_only).toBe('boolean');
      }

      // Phase 35.1 and Phase 36 Proof Audits from real Rust engine fixture
      const looProof = phase36ApexWireFixture.rank1_loo_proof;
      expect(looProof.loo_winners_preserved).toBe(116);
      expect(looProof.total_baseline_winners).toBe(116);
      expect(looProof.min_downdate_l2_norm).toBeGreaterThan(0.0);

      const greeksProof = phase36ApexWireFixture.strike_exact_greeks_proof;
      expect(greeksProof.theta_diff).toBeLessThan(1e-12);
      expect(greeksProof.charm_diff).toBeLessThan(1e-12);
      expect(greeksProof.bag_bleed).toBeLessThan(greeksProof.naked_bleed * 0.70);

      const bocpdProof = phase36ApexWireFixture.bocpd_gamma_flip_proof;
      expect(bocpdProof.calm_p_cp).toBeLessThan(0.10);
      expect(bocpdProof.calm_gamma).toBeGreaterThan(0.97);
      expect(bocpdProof.shock_p_cp).toBeGreaterThan(0.70);
      expect(bocpdProof.shock_gamma).toBeLessThan(0.84);
      expect(bocpdProof.dynamic_pivot_speedup).toBeGreaterThan(2.0);

      const martProof = phase36ApexWireFixture.martingale_probation_proof;
      expect(martProof.healthy_e).toBeLessThan(1.50);
      expect(martProof.quarantined_e).toBeGreaterThanOrEqual(20.0);
      expect(martProof.recovered_e).toBeLessThan(4.0);
      expect(martProof.probation_recovered).toBe(true);

      const routerProof = phase36ApexWireFixture.microprice_router_proof;
      expect(routerProof.passive_plan.execution_tactic).toBe('PASSIVE_MID_PEG_LIMIT');
      expect(routerProof.adaptive_plan.execution_tactic).toBe('ADAPTIVE_MICROPRICE_PEG');
      expect(routerProof.sweep_plan.execution_tactic).toBe('IMMEDIATE_ALMGREN_CHRISS_SWEEP');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('hydrates Phase 37 Sovereign Hawkes Absorption, Dealer Gamma Wall Pinning, 1-Wasserstein DRO Kelly-CVaR, and Telescoping Multi-Stage Credit (#29)', async () => {
    const store = new TerminalStore(false);
    const originalFetch = globalThis.fetch;

    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (
        url.includes('/api/v1/scanner/watchlist') ||
        url.includes('/api/v1/scanner/institutional-watchlist') ||
        url.includes('/api/v1/scanner/sunday-watchlist')
      ) {
        return new Response(JSON.stringify(phase37SovereignWireFixture.watchlist_response), {
          status: 200,
          headers: { 'Content-Type': 'application/json' }
        });
      }
      return new Response('{}', { status: 404 });
    }) as unknown as typeof fetch;

    try {
      await store.fetchInstitutionalWatchlist();
      expect(store.institutionalWatchlist).not.toBeNull();
      const rows = store.institutionalWatchlist!.watchlist;
      expect(rows.length).toBeGreaterThan(0);

      // Verify all 10 Phase 37 fields on hydrated rows
      for (const r of rows) {
        expect(Number.isFinite(r.hawkes_branching_ratio)).toBe(true);
        expect(r.hawkes_branching_ratio!).toBeGreaterThanOrEqual(0.0);
        expect(Number.isFinite(r.hawkes_absorption_toxicity)).toBe(true);
        expect(r.hawkes_absorption_toxicity!).toBeGreaterThanOrEqual(0.0);
        expect(r.hawkes_absorption_toxicity!).toBeLessThanOrEqual(1.0);
        expect(typeof r.hawkes_regime_state).toBe('string');
        expect(Number.isFinite(r.zero_gamma_flip_price)).toBe(true);
        expect(Number.isFinite(r.nearest_gamma_wall_price)).toBe(true);
        expect(Number.isFinite(r.dealer_pin_resistance)).toBe(true);
        expect(Number.isFinite(r.warped_target_1_price)).toBe(true);
        expect(Number.isFinite(r.wasserstein_ambiguity_eps)).toBe(true);
        expect(r.wasserstein_ambiguity_eps!).toBeGreaterThan(0.0);
        expect(Number.isFinite(r.dro_optimal_kelly_pct)).toBe(true);
        expect(Number.isFinite(r.marginal_cvar95_r)).toBe(true);
      }

      // Proof Audits from real Rust engine telemetry
      const bocpdProof = phase37SovereignWireFixture.bocpd_isolation_proof;
      expect(bocpdProof.macro_p_cp).toBeLessThan(0.25);
      expect(bocpdProof.xyz_p_cp).toBeGreaterThan(0.40);
      expect(bocpdProof.fused_p_cp).toBeGreaterThanOrEqual(bocpdProof.macro_p_cp);
      expect(bocpdProof.symmetric_buy_to_open_parity).toBe(true);
      expect(bocpdProof.symmetric_sell_to_open_parity).toBe(true);

      const hawkesProof = phase37SovereignWireFixture.hawkes_microstructure_proof;
      expect(hawkesProof.branching_ratio).toBeGreaterThan(0.60);
      expect(hawkesProof.absorption_toxicity).toBeGreaterThan(0.68);
      expect(hawkesProof.long_iceberg_vetoed).toBe(true);
      expect(hawkesProof.short_iceberg_vetoed).toBe(false);
      expect(hawkesProof.climax_ratchet_floor).toBeGreaterThanOrEqual(0.85);

      const gammaProof = phase37SovereignWireFixture.dealer_gamma_field_proof;
      expect(gammaProof.zero_gamma_flip).toBeLessThan(gammaProof.spot);
      expect(gammaProof.call_wall).toBeGreaterThan(gammaProof.spot);
      expect(gammaProof.warped_t1).toBeLessThan(gammaProof.call_wall);
      expect(gammaProof.pinned_short_strike).toBe(gammaProof.call_wall);
      expect(gammaProof.unwarped_t1).toBe(gammaProof.raw_t1_safe);

      const droProof = phase37SovereignWireFixture.wasserstein_dro_proof;
      expect(droProof.calm_eps).toBeLessThan(droProof.elevated_eps);
      expect(droProof.elevated_eps).toBeLessThan(droProof.stress_eps);
      expect(droProof.calm_f_dro).toBeGreaterThan(droProof.elevated_f_dro);
      expect(droProof.elevated_f_dro).toBeGreaterThan(droProof.stress_f_dro);
      expect(droProof.cvar_capped).toBe(true);
      expect(droProof.cvar_tail_heat_pct).toBeLessThanOrEqual(0.0651);

      const shapleyProof = phase37SovereignWireFixture.telescoping_shapley_proof;
      expect(shapleyProof.baseline_trades_verified).toBe(173);
      expect(shapleyProof.max_telescoping_error).toBe(0.0);
      expect(shapleyProof.total_alpha_lift).toBeGreaterThan(0.0);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});
