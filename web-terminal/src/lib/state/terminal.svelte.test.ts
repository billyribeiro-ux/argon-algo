import { describe, expect, it } from 'vitest';
import {
  TerminalStore,
  formatSubPennyPrice,
  CSV_LEDGER_HEADERS
} from './terminal.svelte';

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
});
