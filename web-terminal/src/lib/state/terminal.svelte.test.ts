import { describe, expect, it } from 'vitest';
import {
  TerminalStore,
  formatSubPennyPrice,
  CSV_LEDGER_HEADERS
} from './terminal.svelte';

describe('ARGON-4 Phase 4 TerminalStore & Hyper-Filter Verification', () => {
  it('formats sub-penny ($0.00042) and mega-cap ($142.85) prices with exact tabular precision', () => {
    expect(formatSubPennyPrice(0.00042)).toBe('$0.00042');
    expect(formatSubPennyPrice(0.00001)).toBe('$0.00001');
    expect(formatSubPennyPrice(142.85)).toBe('$142.85');
  });

  it('filters scanner candidates across all 5 Brains and sub-penny vs institutional price tiers', () => {
    const store = new TerminalStore();
    expect(store.filteredScannerCandidates.length).toBe(6);

    store.selectBrain('BRAIN-01', false);
    expect(store.filteredScannerCandidates.every((c) => c.brainId === 'BRAIN-01')).toBe(true);

    store.selectBrain('ALL_5_BRAINS', false);
    store.universeTierMode = 'SUB_PENNY_ONLY';
    expect(store.filteredScannerCandidates.length).toBeGreaterThanOrEqual(1);
    expect(store.filteredScannerCandidates.every((c) => c.livePrice <= 5.0)).toBe(true);

    store.universeTierMode = 'INSTITUTIONAL_ONLY';
    expect(store.filteredScannerCandidates.every((c) => c.livePrice > 5.0)).toBe(true);
  });

  it('transitions open positions from POSITIONS OPENED into CLOSED TRADE LEDGER and updates KPIs', () => {
    const store = new TerminalStore();
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
    const store = new TerminalStore();

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

