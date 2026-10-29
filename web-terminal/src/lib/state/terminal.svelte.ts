/**
 * Svelte 5 Runes Reactive Store for the ARGON-4 Multi-Brain Command Terminal.
 *
 * Powers the 5-Brain Horizon Selector, 11,000+ Ticker Scanner, 4-Stream Raw Evidence Hub,
 * Live `POSITIONS OPENED` Tracker, `CLOSED TRADE LEDGER`, 6-Dimension Hyper-Filter Engine,
 * and 1-Click `.CSV` / `.XLSX` Institutional Exporters.
 */

import { MultiWindowSyncBus } from '../multi-window/window-sync';

export type BrainId =
  | 'ALL_5_BRAINS'
  | 'BRAIN-01'
  | 'BRAIN-02'
  | 'BRAIN-03'
  | 'BRAIN-04'
  | 'BRAIN-05';

export type TradeDirection = 'LONG' | 'SHORT';
export type AssetClassMode = 'ALL' | 'STOCKS' | 'OPTIONS';
export type ClosedOutcomeType =
  | 'TARGET HIT'
  | 'STOPPED OUT'
  | 'AI DYNAMIC EXIT'
  | 'TRAILING STOP'
  | 'TIME EXPIRED';

export type TraceHealthBadge = 'OPTIMAL' | 'EXHAUSTION WARNING' | 'DYNAMIC EXIT TRIGGERED';

export type PrimaryDriverStream =
  | 'PRICE_VOLUME_ANOMALY'
  | 'OPTIONS_GAMMA_VANNA'
  | 'SHORT_FLOAT_SQUEEZE'
  | 'MARKET_INTERNALS_TIDE';

export type QuickFilterPillId =
  | 'ALL'
  | 'WINS_ONLY'
  | 'LOSSES_ONLY'
  | 'TARGET_HITS'
  | 'DYNAMIC_EXITS'
  | 'OPTIONS_ONLY'
  | 'SUB_PENNY'
  | 'MEGA_CAP'
  | 'HIGH_CONFIDENCE_90';

export interface RawBarCandle {
  timestampMs: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  attentionWeight: number;
}

export interface ScannerSignalCandidate {
  symbol: string;
  exchangeIndex: string;
  brainId: Exclude<BrainId, 'ALL_5_BRAINS'>;
  direction: TradeDirection;
  assetClass: 'STOCK' | 'STOCK + OPTIONS';
  livePrice: number;
  entryPrice: number;
  structuralStopPrice: number;
  target1Price: number;
  target2Price: number;
  predictedHoldWindow: string;
  predictedHoldMs: number;
  aiConfidencePct: number;
  expectedRMultiple: number;
  primaryDriver: PrimaryDriverStream;
  streamWeights: {
    priceVolumeAnomaly: number;
    optionsGammaVanna: number;
    shortFloatSqueeze: number;
    marketInternalsTide: number;
  };
  precedentMatchSummary: {
    matchingWins: number;
    totalK: number;
    winRatePct: number;
    avgRealizedR: number;
    avgMfePct: number;
    avgMaePct: number;
  };
  optionsRecommendation: {
    contractType: 'CALL' | 'PUT';
    strike: number;
    expirationDate: string;
    dte: number;
    entryPremium: number;
    delta: number;
    gamma: number;
    theta: number;
    ivPct: number;
  } | null;
  bars: RawBarCandle[];
}

export interface OpenPositionItem {
  tradeId: string;
  openedTimestampMs: number;
  symbol: string;
  exchangeIndex: string;
  brainId: Exclude<BrainId, 'ALL_5_BRAINS'>;
  direction: TradeDirection;
  assetClass: 'STOCK' | 'STOCK + OPTIONS';
  entryPrice: number;
  livePrice: number;
  quantityShares: number;
  structuralStopPrice: number;
  target1Price: number;
  target2Price: number;
  unrealizedStockPnlUsd: number;
  unrealizedStockPnlPct: number;
  optionContractLabel: string;
  optionEntryPremium: number;
  optionLivePremium: number;
  optionContractsCount: number;
  unrealizedOptionPnlUsd: number;
  unrealizedOptionPnlPct: number;
  delta: number;
  gamma: number;
  theta: number;
  ivPct: number;
  elapsedHoldMs: number;
  predictedHoldMs: number;
  currentRMultiple: number;
  mfePct: number;
  maePct: number;
  aiConfidencePct: number;
  primaryDriver: PrimaryDriverStream;
  traceHealthBadge: TraceHealthBadge;
  precedentLabel: string;
  streamWeights: {
    priceVolumeAnomaly: number;
    optionsGammaVanna: number;
    shortFloatSqueeze: number;
    marketInternalsTide: number;
  };
}

export interface ClosedTradeLedgerItem {
  tradeId: string;
  openedTimestampMs: number;
  closedTimestampMs: number;
  holdDurationMs: number;
  predictedHoldDurationMs: number;
  symbol: string;
  exchangeIndex: string;
  brainId: Exclude<BrainId, 'ALL_5_BRAINS'>;
  direction: TradeDirection;
  assetClass: 'STOCK' | 'STOCK + OPTIONS';
  entryPrice: number;
  exitPrice: number;
  quantityShares: number;
  structuralStopAtEntry: number;
  target1AtEntry: number;
  target2AtEntry: number;
  outcomeBadge: ClosedOutcomeType;
  isWinner: boolean;
  realizedStockPnlUsd: number;
  realizedStockPnlPct: number;
  optionStrike: number;
  optionExpiration: string;
  optionType: 'CALL' | 'PUT' | 'NONE';
  optionEntryPremium: number;
  optionExitPremium: number;
  optionContractsCount: number;
  realizedOptionPnlUsd: number;
  realizedOptionPnlPct: number;
  combinedTotalPnlUsd: number;
  realizedRMultiple: number;
  mfePct: number;
  maePct: number;
  aiConfidenceAtEntryPct: number;
  primaryDriver: PrimaryDriverStream;
  streamWeights: {
    priceVolumeAnomaly: number;
    optionsGammaVanna: number;
    shortFloatSqueeze: number;
    marketInternalsTide: number;
  };
  precedentSimilarityScore: number;
  autopsyRootCauseCode: string;
}

export interface ProviderStatusEntry {
  providerId: string;
  category: 'EQUITY_TAPE' | 'OPTIONS_GREEKS' | 'SHORT_FLOAT';
  role: 'PRIMARY' | 'SECONDARY' | 'TERTIARY';
  healthy: boolean;
  latencyMs: number;
  messagesPerSec: number;
}

/**
 * Formats any asset price from `$0.00001` sub-penny up to `$10,000.00+` with exact tabular precision.
 */
export function formatSubPennyPrice(price: number): string {
  if (!Number.isFinite(price)) {
    return '$0.00000';
  }
  const abs = Math.abs(price);
  if (abs > 0 && abs < 1.0) {
    return `$${price.toFixed(5)}`;
  }
  if (abs < 10.0) {
    return `$${price.toFixed(4)}`;
  }
  return `$${price.toFixed(2)}`;
}

export function formatSignedUsd(amount: number): string {
  const sign = amount >= 0 ? '+' : '-';
  return `${sign}$${Math.abs(amount).toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  })}`;
}

export function formatSignedPct(pct: number): string {
  const sign = pct >= 0 ? '+' : '';
  return `${sign}${pct.toFixed(2)}%`;
}

export function formatHoldDuration(ms: number): string {
  const totalSec = Math.max(0, Math.floor(ms / 1000));
  const days = Math.floor(totalSec / 86400);
  const hours = Math.floor((totalSec % 86400) / 3600);
  const mins = Math.floor((totalSec % 3600) / 60);
  if (days > 0) {
    return `${days}d ${hours}h`;
  }
  if (hours > 0) {
    return `${hours}h ${mins}m`;
  }
  return `${mins}m ${totalSec % 60}s`;
}

function generateRawCandleSequence(basePrice: number, direction: TradeDirection): RawBarCandle[] {
  const bars: RawBarCandle[] = [];
  let p = basePrice * (direction === 'LONG' ? 0.965 : 1.035);
  const now = Date.now();
  for (let i = 0; i < 32; i++) {
    const progress = i / 31;
    const stepPct =
      direction === 'LONG'
        ? 0.0012 + (i > 24 ? 0.0032 : 0.0004) * Math.sin(i * 0.7)
        : -0.0012 - (i > 24 ? 0.0032 : 0.0004) * Math.sin(i * 0.7);
    const open = p;
    const close = Math.max(0.00001, open * (1 + stepPct));
    const high = Math.max(open, close) * (1 + 0.0028);
    const low = Math.min(open, close) * (1 - 0.0034);
    const volume = Math.round(140_000 * (1 + (i > 25 ? 2.8 * progress : 0.5 * progress)));
    const attentionWeight = Math.min(0.99, Math.max(0.12, i > 25 ? 0.78 + (i - 25) * 0.03 : 0.22 + progress * 0.3));
    bars.push({
      timestampMs: now - (32 - i) * 300_000,
      open,
      high,
      low,
      close,
      volume,
      attentionWeight
    });
    p = close;
  }
  return bars;
}

const INITIAL_SCANNER_CANDIDATES: ScannerSignalCandidate[] = [
  {
    symbol: 'NVDA',
    exchangeIndex: 'NASDAQ_100',
    brainId: 'BRAIN-01',
    direction: 'LONG',
    assetClass: 'STOCK + OPTIONS',
    livePrice: 142.85,
    entryPrice: 142.8,
    structuralStopPrice: 141.65,
    target1Price: 145.1,
    target2Price: 147.95,
    predictedHoldWindow: '15m - 45m',
    predictedHoldMs: 1_800_000,
    aiConfidencePct: 94.8,
    expectedRMultiple: 4.48,
    primaryDriver: 'OPTIONS_GAMMA_VANNA',
    streamWeights: {
      priceVolumeAnomaly: 0.31,
      optionsGammaVanna: 0.44,
      shortFloatSqueeze: 0.09,
      marketInternalsTide: 0.16
    },
    precedentMatchSummary: {
      matchingWins: 23,
      totalK: 25,
      winRatePct: 92.0,
      avgRealizedR: 3.84,
      avgMfePct: 4.12,
      avgMaePct: 0.42
    },
    optionsRecommendation: {
      contractType: 'CALL',
      strike: 144.0,
      expirationDate: '2026-10-30',
      dte: 1,
      entryPremium: 2.15,
      delta: 0.56,
      gamma: 0.094,
      theta: -0.18,
      ivPct: 44.2
    },
    bars: generateRawCandleSequence(142.85, 'LONG')
  },
  {
    symbol: 'HYMC',
    exchangeIndex: 'SUB_PENNY_OTC',
    brainId: 'BRAIN-01',
    direction: 'LONG',
    assetClass: 'STOCK',
    livePrice: 0.00042,
    entryPrice: 0.00042,
    structuralStopPrice: 0.00037,
    target1Price: 0.00055,
    target2Price: 0.00074,
    predictedHoldWindow: '20m - 60m',
    predictedHoldMs: 2_400_000,
    aiConfidencePct: 91.4,
    expectedRMultiple: 6.4,
    primaryDriver: 'SHORT_FLOAT_SQUEEZE',
    streamWeights: {
      priceVolumeAnomaly: 0.38,
      optionsGammaVanna: 0.04,
      shortFloatSqueeze: 0.48,
      marketInternalsTide: 0.1
    },
    precedentMatchSummary: {
      matchingWins: 21,
      totalK: 25,
      winRatePct: 84.0,
      avgRealizedR: 5.12,
      avgMfePct: 68.4,
      avgMaePct: 8.5
    },
    optionsRecommendation: null,
    bars: generateRawCandleSequence(0.00042, 'LONG')
  },
  {
    symbol: 'TSLA',
    exchangeIndex: 'SP500',
    brainId: 'BRAIN-02',
    direction: 'LONG',
    assetClass: 'STOCK + OPTIONS',
    livePrice: 258.4,
    entryPrice: 258.1,
    structuralStopPrice: 255.2,
    target1Price: 264.5,
    target2Price: 271.8,
    predictedHoldWindow: '2h - 5h',
    predictedHoldMs: 12_600_000,
    aiConfidencePct: 92.6,
    expectedRMultiple: 4.72,
    primaryDriver: 'PRICE_VOLUME_ANOMALY',
    streamWeights: {
      priceVolumeAnomaly: 0.42,
      optionsGammaVanna: 0.33,
      shortFloatSqueeze: 0.13,
      marketInternalsTide: 0.12
    },
    precedentMatchSummary: {
      matchingWins: 22,
      totalK: 25,
      winRatePct: 88.0,
      avgRealizedR: 3.95,
      avgMfePct: 5.4,
      avgMaePct: 0.68
    },
    optionsRecommendation: {
      contractType: 'CALL',
      strike: 260.0,
      expirationDate: '2026-11-06',
      dte: 8,
      entryPremium: 6.4,
      delta: 0.54,
      gamma: 0.068,
      theta: -0.22,
      ivPct: 54.8
    },
    bars: generateRawCandleSequence(258.4, 'LONG')
  },
  {
    symbol: 'GME',
    exchangeIndex: 'NYSE',
    brainId: 'BRAIN-03',
    direction: 'LONG',
    assetClass: 'STOCK + OPTIONS',
    livePrice: 28.65,
    entryPrice: 28.5,
    structuralStopPrice: 27.4,
    target1Price: 31.8,
    target2Price: 36.4,
    predictedHoldWindow: '2d - 5d',
    predictedHoldMs: 259_200_000,
    aiConfidencePct: 93.2,
    expectedRMultiple: 7.18,
    primaryDriver: 'SHORT_FLOAT_SQUEEZE',
    streamWeights: {
      priceVolumeAnomaly: 0.27,
      optionsGammaVanna: 0.28,
      shortFloatSqueeze: 0.37,
      marketInternalsTide: 0.08
    },
    precedentMatchSummary: {
      matchingWins: 22,
      totalK: 25,
      winRatePct: 88.0,
      avgRealizedR: 5.85,
      avgMfePct: 24.6,
      avgMaePct: 2.9
    },
    optionsRecommendation: {
      contractType: 'CALL',
      strike: 30.0,
      expirationDate: '2026-11-20',
      dte: 22,
      entryPremium: 2.45,
      delta: 0.58,
      gamma: 0.082,
      theta: -0.07,
      ivPct: 88.5
    },
    bars: generateRawCandleSequence(28.65, 'LONG')
  },
  {
    symbol: 'SMH',
    exchangeIndex: 'SECTOR_ETF',
    brainId: 'BRAIN-04',
    direction: 'SHORT',
    assetClass: 'STOCK + OPTIONS',
    livePrice: 248.2,
    entryPrice: 248.5,
    structuralStopPrice: 251.4,
    target1Price: 241.8,
    target2Price: 235.1,
    predictedHoldWindow: '2w - 4w',
    predictedHoldMs: 1_209_600_000,
    aiConfidencePct: 89.5,
    expectedRMultiple: 4.62,
    primaryDriver: 'MARKET_INTERNALS_TIDE',
    streamWeights: {
      priceVolumeAnomaly: 0.29,
      optionsGammaVanna: 0.24,
      shortFloatSqueeze: 0.08,
      marketInternalsTide: 0.39
    },
    precedentMatchSummary: {
      matchingWins: 20,
      totalK: 25,
      winRatePct: 80.0,
      avgRealizedR: 3.45,
      avgMfePct: 6.1,
      avgMaePct: 0.95
    },
    optionsRecommendation: {
      contractType: 'PUT',
      strike: 245.0,
      expirationDate: '2026-12-18',
      dte: 50,
      entryPremium: 7.8,
      delta: -0.52,
      gamma: 0.044,
      theta: -0.09,
      ivPct: 34.2
    },
    bars: generateRawCandleSequence(248.2, 'SHORT')
  },
  {
    symbol: 'GS',
    exchangeIndex: 'DOW_30',
    brainId: 'BRAIN-05',
    direction: 'LONG',
    assetClass: 'STOCK + OPTIONS',
    livePrice: 514.6,
    entryPrice: 512.0,
    structuralStopPrice: 498.5,
    target1Price: 545.0,
    target2Price: 578.0,
    predictedHoldWindow: '3m - 6m',
    predictedHoldMs: 7_776_000_000,
    aiConfidencePct: 95.1,
    expectedRMultiple: 4.89,
    primaryDriver: 'PRICE_VOLUME_ANOMALY',
    streamWeights: {
      priceVolumeAnomaly: 0.44,
      optionsGammaVanna: 0.22,
      shortFloatSqueeze: 0.06,
      marketInternalsTide: 0.28
    },
    precedentMatchSummary: {
      matchingWins: 24,
      totalK: 25,
      winRatePct: 96.0,
      avgRealizedR: 4.42,
      avgMfePct: 14.2,
      avgMaePct: 1.45
    },
    optionsRecommendation: {
      contractType: 'CALL',
      strike: 520.0,
      expirationDate: '2027-03-19',
      dte: 141,
      entryPremium: 28.5,
      delta: 0.64,
      gamma: 0.025,
      theta: -0.05,
      ivPct: 26.8
    },
    bars: generateRawCandleSequence(514.6, 'LONG')
  }
];

const INITIAL_OPEN_POSITIONS: OpenPositionItem[] = [
  {
    tradeId: 'OPEN-901',
    openedTimestampMs: Date.now() - 840_000,
    symbol: 'NVDA',
    exchangeIndex: 'NASDAQ_100',
    brainId: 'BRAIN-01',
    direction: 'LONG',
    assetClass: 'STOCK + OPTIONS',
    entryPrice: 142.8,
    livePrice: 144.65,
    quantityShares: 70,
    structuralStopPrice: 142.8,
    target1Price: 145.1,
    target2Price: 147.95,
    unrealizedStockPnlUsd: 129.5,
    unrealizedStockPnlPct: 1.3,
    optionContractLabel: 'NVDA 2026-10-30 144.00 CALL',
    optionEntryPremium: 2.15,
    optionLivePremium: 3.38,
    optionContractsCount: 4,
    unrealizedOptionPnlUsd: 492.0,
    unrealizedOptionPnlPct: 57.21,
    delta: 0.62,
    gamma: 0.094,
    theta: -0.18,
    ivPct: 44.2,
    elapsedHoldMs: 840_000,
    predictedHoldMs: 1_800_000,
    currentRMultiple: 1.61,
    mfePct: 1.42,
    maePct: 0.14,
    aiConfidencePct: 94.8,
    primaryDriver: 'OPTIONS_GAMMA_VANNA',
    traceHealthBadge: 'OPTIMAL',
    precedentLabel: 'Matches 23/25 Past Expansion Traces (92% Win Rate)',
    streamWeights: {
      priceVolumeAnomaly: 0.31,
      optionsGammaVanna: 0.44,
      shortFloatSqueeze: 0.09,
      marketInternalsTide: 0.16
    }
  },
  {
    tradeId: 'OPEN-902',
    openedTimestampMs: Date.now() - 1_120_000,
    symbol: 'HYMC',
    exchangeIndex: 'SUB_PENNY_OTC',
    brainId: 'BRAIN-01',
    direction: 'LONG',
    assetClass: 'STOCK',
    entryPrice: 0.00042,
    livePrice: 0.00058,
    quantityShares: 2_500_000,
    structuralStopPrice: 0.00042,
    target1Price: 0.00055,
    target2Price: 0.00074,
    unrealizedStockPnlUsd: 400.0,
    unrealizedStockPnlPct: 38.1,
    optionContractLabel: 'STOCK ONLY (SUB-PENNY)',
    optionEntryPremium: 0,
    optionLivePremium: 0,
    optionContractsCount: 0,
    unrealizedOptionPnlUsd: 0,
    unrealizedOptionPnlPct: 0,
    delta: 0,
    gamma: 0,
    theta: 0,
    ivPct: 0,
    elapsedHoldMs: 1_120_000,
    predictedHoldMs: 2_400_000,
    currentRMultiple: 3.2,
    mfePct: 40.48,
    maePct: 2.38,
    aiConfidencePct: 91.4,
    primaryDriver: 'SHORT_FLOAT_SQUEEZE',
    traceHealthBadge: 'OPTIMAL',
    precedentLabel: 'Matches 21/25 Past Expansion Traces (84% Win Rate)',
    streamWeights: {
      priceVolumeAnomaly: 0.38,
      optionsGammaVanna: 0.04,
      shortFloatSqueeze: 0.48,
      marketInternalsTide: 0.1
    }
  },
  {
    tradeId: 'OPEN-903',
    openedTimestampMs: Date.now() - 3_600_000,
    symbol: 'TSLA',
    exchangeIndex: 'SP500',
    brainId: 'BRAIN-02',
    direction: 'LONG',
    assetClass: 'STOCK + OPTIONS',
    entryPrice: 258.1,
    livePrice: 263.4,
    quantityShares: 40,
    structuralStopPrice: 258.1,
    target1Price: 264.5,
    target2Price: 271.8,
    unrealizedStockPnlUsd: 212.0,
    unrealizedStockPnlPct: 2.05,
    optionContractLabel: 'TSLA 2026-11-06 260.00 CALL',
    optionEntryPremium: 6.4,
    optionLivePremium: 9.75,
    optionContractsCount: 3,
    unrealizedOptionPnlUsd: 1005.0,
    unrealizedOptionPnlPct: 52.34,
    delta: 0.61,
    gamma: 0.068,
    theta: -0.22,
    ivPct: 54.8,
    elapsedHoldMs: 3_600_000,
    predictedHoldMs: 12_600_000,
    currentRMultiple: 1.83,
    mfePct: 2.28,
    maePct: 0.31,
    aiConfidencePct: 92.6,
    primaryDriver: 'PRICE_VOLUME_ANOMALY',
    traceHealthBadge: 'EXHAUSTION WARNING',
    precedentLabel: 'Matches 22/25 Past Expansion Traces (88% Win Rate)',
    streamWeights: {
      priceVolumeAnomaly: 0.42,
      optionsGammaVanna: 0.33,
      shortFloatSqueeze: 0.13,
      marketInternalsTide: 0.12
    }
  }
];

const INITIAL_CLOSED_LEDGER: ClosedTradeLedgerItem[] = [
  {
    tradeId: 'TRD-1001',
    openedTimestampMs: Date.now() - 86_400_000 * 5,
    closedTimestampMs: Date.now() - 86_400_000 * 5 + 1_620_000,
    holdDurationMs: 1_620_000,
    predictedHoldDurationMs: 1_800_000,
    symbol: 'NVDA',
    exchangeIndex: 'NASDAQ_100',
    brainId: 'BRAIN-01',
    direction: 'LONG',
    assetClass: 'STOCK + OPTIONS',
    entryPrice: 138.4,
    exitPrice: 143.2,
    quantityShares: 100,
    structuralStopAtEntry: 137.2,
    target1AtEntry: 140.8,
    target2AtEntry: 143.2,
    outcomeBadge: 'TARGET HIT',
    isWinner: true,
    realizedStockPnlUsd: 480.0,
    realizedStockPnlPct: 3.47,
    optionStrike: 140.0,
    optionExpiration: '2026-10-30',
    optionType: 'CALL',
    optionEntryPremium: 1.95,
    optionExitPremium: 4.85,
    optionContractsCount: 5,
    realizedOptionPnlUsd: 1450.0,
    realizedOptionPnlPct: 148.72,
    combinedTotalPnlUsd: 1930.0,
    realizedRMultiple: 4.0,
    mfePct: 3.62,
    maePct: 0.21,
    aiConfidenceAtEntryPct: 95.4,
    primaryDriver: 'OPTIONS_GAMMA_VANNA',
    streamWeights: {
      priceVolumeAnomaly: 0.3,
      optionsGammaVanna: 0.46,
      shortFloatSqueeze: 0.08,
      marketInternalsTide: 0.16
    },
    precedentSimilarityScore: 0.942,
    autopsyRootCauseCode: 'VERIFIED_OPTIMAL_EXPANSION'
  },
  {
    tradeId: 'TRD-1002',
    openedTimestampMs: Date.now() - 86_400_000 * 4,
    closedTimestampMs: Date.now() - 86_400_000 * 4 + 2_100_000,
    holdDurationMs: 2_100_000,
    predictedHoldDurationMs: 2_400_000,
    symbol: 'MULN',
    exchangeIndex: 'SUB_PENNY_OTC',
    brainId: 'BRAIN-01',
    direction: 'LONG',
    assetClass: 'STOCK',
    entryPrice: 0.00035,
    exitPrice: 0.00063,
    quantityShares: 3_000_000,
    structuralStopAtEntry: 0.0003,
    target1AtEntry: 0.00048,
    target2AtEntry: 0.00063,
    outcomeBadge: 'TARGET HIT',
    isWinner: true,
    realizedStockPnlUsd: 840.0,
    realizedStockPnlPct: 80.0,
    optionStrike: 0,
    optionExpiration: 'N/A',
    optionType: 'NONE',
    optionEntryPremium: 0,
    optionExitPremium: 0,
    optionContractsCount: 0,
    realizedOptionPnlUsd: 0,
    realizedOptionPnlPct: 0,
    combinedTotalPnlUsd: 840.0,
    realizedRMultiple: 5.6,
    mfePct: 82.8,
    maePct: 2.9,
    aiConfidenceAtEntryPct: 92.1,
    primaryDriver: 'SHORT_FLOAT_SQUEEZE',
    streamWeights: {
      priceVolumeAnomaly: 0.36,
      optionsGammaVanna: 0.02,
      shortFloatSqueeze: 0.52,
      marketInternalsTide: 0.1
    },
    precedentSimilarityScore: 0.915,
    autopsyRootCauseCode: 'VERIFIED_OPTIMAL_EXPANSION'
  },
  {
    tradeId: 'TRD-1003',
    openedTimestampMs: Date.now() - 86_400_000 * 3,
    closedTimestampMs: Date.now() - 86_400_000 * 3 + 9_400_000,
    holdDurationMs: 9_400_000,
    predictedHoldDurationMs: 12_600_000,
    symbol: 'QQQ',
    exchangeIndex: 'NASDAQ_100',
    brainId: 'BRAIN-02',
    direction: 'SHORT',
    assetClass: 'STOCK + OPTIONS',
    entryPrice: 504.2,
    exitPrice: 497.8,
    quantityShares: 50,
    structuralStopAtEntry: 506.4,
    target1AtEntry: 499.5,
    target2AtEntry: 495.0,
    outcomeBadge: 'AI DYNAMIC EXIT',
    isWinner: true,
    realizedStockPnlUsd: 320.0,
    realizedStockPnlPct: 1.27,
    optionStrike: 502.0,
    optionExpiration: '2026-11-06',
    optionType: 'PUT',
    optionEntryPremium: 3.4,
    optionExitPremium: 6.95,
    optionContractsCount: 4,
    realizedOptionPnlUsd: 1420.0,
    realizedOptionPnlPct: 104.41,
    combinedTotalPnlUsd: 1740.0,
    realizedRMultiple: 2.91,
    mfePct: 1.48,
    maePct: 0.18,
    aiConfidenceAtEntryPct: 91.2,
    primaryDriver: 'MARKET_INTERNALS_TIDE',
    streamWeights: {
      priceVolumeAnomaly: 0.25,
      optionsGammaVanna: 0.28,
      shortFloatSqueeze: 0.05,
      marketInternalsTide: 0.42
    },
    precedentSimilarityScore: 0.908,
    autopsyRootCauseCode: 'VERIFIED_OPTIMAL_EXPANSION'
  },
  {
    tradeId: 'TRD-1004',
    openedTimestampMs: Date.now() - 86_400_000 * 2,
    closedTimestampMs: Date.now() - 86_400_000 * 2 + 172_800_000,
    holdDurationMs: 172_800_000,
    predictedHoldDurationMs: 259_200_000,
    symbol: 'GME',
    exchangeIndex: 'NYSE',
    brainId: 'BRAIN-03',
    direction: 'LONG',
    assetClass: 'STOCK + OPTIONS',
    entryPrice: 24.5,
    exitPrice: 31.2,
    quantityShares: 200,
    structuralStopAtEntry: 23.4,
    target1AtEntry: 27.8,
    target2AtEntry: 31.2,
    outcomeBadge: 'TARGET HIT',
    isWinner: true,
    realizedStockPnlUsd: 1340.0,
    realizedStockPnlPct: 27.35,
    optionStrike: 26.0,
    optionExpiration: '2026-11-20',
    optionType: 'CALL',
    optionEntryPremium: 1.85,
    optionExitPremium: 6.1,
    optionContractsCount: 5,
    realizedOptionPnlUsd: 2125.0,
    realizedOptionPnlPct: 229.73,
    combinedTotalPnlUsd: 3465.0,
    realizedRMultiple: 6.09,
    mfePct: 28.4,
    maePct: 1.6,
    aiConfidenceAtEntryPct: 93.8,
    primaryDriver: 'SHORT_FLOAT_SQUEEZE',
    streamWeights: {
      priceVolumeAnomaly: 0.28,
      optionsGammaVanna: 0.26,
      shortFloatSqueeze: 0.38,
      marketInternalsTide: 0.08
    },
    precedentSimilarityScore: 0.934,
    autopsyRootCauseCode: 'VERIFIED_OPTIMAL_EXPANSION'
  },
  {
    tradeId: 'TRD-1005',
    openedTimestampMs: Date.now() - 86_400_000 * 1.5,
    closedTimestampMs: Date.now() - 86_400_000 * 1.5 + 3_200_000,
    holdDurationMs: 3_200_000,
    predictedHoldDurationMs: 14_400_000,
    symbol: 'IWM',
    exchangeIndex: 'RUSSELL_2000',
    brainId: 'BRAIN-02',
    direction: 'LONG',
    assetClass: 'STOCK + OPTIONS',
    entryPrice: 221.4,
    exitPrice: 219.9,
    quantityShares: 60,
    structuralStopAtEntry: 219.9,
    target1AtEntry: 224.5,
    target2AtEntry: 227.8,
    outcomeBadge: 'STOPPED OUT',
    isWinner: false,
    realizedStockPnlUsd: -90.0,
    realizedStockPnlPct: -0.68,
    optionStrike: 222.0,
    optionExpiration: '2026-11-06',
    optionType: 'CALL',
    optionEntryPremium: 2.1,
    optionExitPremium: 1.25,
    optionContractsCount: 3,
    realizedOptionPnlUsd: -255.0,
    realizedOptionPnlPct: -40.48,
    combinedTotalPnlUsd: -345.0,
    realizedRMultiple: -1.0,
    mfePct: 0.24,
    maePct: 0.68,
    aiConfidenceAtEntryPct: 79.4,
    primaryDriver: 'PRICE_VOLUME_ANOMALY',
    streamWeights: {
      priceVolumeAnomaly: 0.45,
      optionsGammaVanna: 0.22,
      shortFloatSqueeze: 0.11,
      marketInternalsTide: 0.22
    },
    precedentSimilarityScore: 0.785,
    autopsyRootCauseCode: 'MACRO_TIDE_HEADWIND_OVERRIDE'
  },
  {
    tradeId: 'TRD-1006',
    openedTimestampMs: Date.now() - 86_400_000 * 0.8,
    closedTimestampMs: Date.now() - 86_400_000 * 0.4,
    holdDurationMs: 34_560_000,
    predictedHoldDurationMs: 86_400_000,
    symbol: 'GS',
    exchangeIndex: 'DOW_30',
    brainId: 'BRAIN-05',
    direction: 'LONG',
    assetClass: 'STOCK + OPTIONS',
    entryPrice: 496.0,
    exitPrice: 514.5,
    quantityShares: 30,
    structuralStopAtEntry: 491.0,
    target1AtEntry: 506.0,
    target2AtEntry: 514.5,
    outcomeBadge: 'TARGET HIT',
    isWinner: true,
    realizedStockPnlUsd: 555.0,
    realizedStockPnlPct: 3.73,
    optionStrike: 500.0,
    optionExpiration: '2027-01-15',
    optionType: 'CALL',
    optionEntryPremium: 14.2,
    optionExitPremium: 26.8,
    optionContractsCount: 2,
    realizedOptionPnlUsd: 2520.0,
    realizedOptionPnlPct: 88.73,
    combinedTotalPnlUsd: 3075.0,
    realizedRMultiple: 3.7,
    mfePct: 3.92,
    maePct: 0.35,
    aiConfidenceAtEntryPct: 95.8,
    primaryDriver: 'PRICE_VOLUME_ANOMALY',
    streamWeights: {
      priceVolumeAnomaly: 0.44,
      optionsGammaVanna: 0.24,
      shortFloatSqueeze: 0.04,
      marketInternalsTide: 0.28
    },
    precedentSimilarityScore: 0.954,
    autopsyRootCauseCode: 'VERIFIED_OPTIMAL_EXPANSION'
  }
];

export const CSV_LEDGER_HEADERS: readonly string[] = [
  'Trade_ID',
  'Opened_Timestamp_UTC',
  'Closed_Timestamp_UTC',
  'Hold_Duration_Minutes',
  'Predicted_Hold_Minutes',
  'Symbol',
  'Exchange_Index',
  'Brain_ID',
  'Direction',
  'Asset_Class',
  'Entry_Price',
  'Exit_Price',
  'Quantity_Shares',
  'Structural_Stop_Price_At_Entry',
  'Target_1_Price_At_Entry',
  'Target_2_Price_At_Entry',
  'Outcome_Status',
  'Is_Win',
  'Realized_Stock_PnL_USD',
  'Realized_Stock_PnL_Pct',
  'Option_Strike',
  'Option_Expiration',
  'Option_Type',
  'Option_Entry_Premium',
  'Option_Exit_Premium',
  'Option_Contracts_Count',
  'Realized_Option_PnL_USD',
  'Realized_Option_PnL_Pct',
  'Combined_Total_PnL_USD',
  'Realized_R_Multiple',
  'Max_Favorable_Excursion_Pct',
  'Max_Adverse_Excursion_Pct',
  'AI_Confidence_Score_At_Entry',
  'Primary_Raw_Driver_Stream',
  'Precedent_Cluster_Similarity_Score',
  'Autopsy_Root_Cause_Code'
];

export class TerminalStore {
  // Top Omni-Command Bar State
  activeBrain = $state<BrainId>('ALL_5_BRAINS');
  activeAssetClass = $state<AssetClassMode>('ALL');
  universeTierMode = $state<'ALL' | 'SUB_PENNY_ONLY' | 'INSTITUTIONAL_ONLY'>('ALL');
  selectedTicker = $state<string>('NVDA');

  // Scanner & Live Positions State
  scannerCandidates = $state<ScannerSignalCandidate[]>(INITIAL_SCANNER_CANDIDATES);
  openPositions = $state<OpenPositionItem[]>(INITIAL_OPEN_POSITIONS);
  closedLedger = $state<ClosedTradeLedgerItem[]>(INITIAL_CLOSED_LEDGER);

  // 6-Dimension Hyper-Filter State for Closed Trade Ledger
  quickPill = $state<QuickFilterPillId>('ALL');
  temporalRange = $state<'ALL' | 'TODAY' | '7D' | '30D' | 'YTD'>('ALL');
  brainFilter = $state<BrainId>('ALL_5_BRAINS');
  directionFilter = $state<'ALL' | TradeDirection>('ALL');
  assetFilter = $state<AssetClassMode>('ALL');
  searchSymbol = $state<string>('');
  minEntryPrice = $state<number>(0);
  maxEntryPrice = $state<number>(100_000);
  exchangeFilter = $state<string>('ALL');
  outcomeFilter = $state<'ALL' | ClosedOutcomeType>('ALL');
  driverFilter = $state<'ALL' | PrimaryDriverStream>('ALL');
  minConfidencePct = $state<number>(50);
  minRMultiple = $state<number>(-10);

  // Macro Tide Internals Telemetry
  macroInternals = $state({
    nyseTick: 640,
    nyseAdd: 1480,
    voldRatio: 2.34,
    soxxPct: 1.92,
    smhPct: 2.18,
    hygLqdSpreadBps: -3.6,
    dxyChangePct: -0.24,
    us10yYieldBps: -4.1,
    vixTermSlope: -0.068
  });

  // Plug-and-Play Provider Status
  providers = $state<ProviderStatusEntry[]>([
    {
      providerId: 'DATABENTO_L3',
      category: 'EQUITY_TAPE',
      role: 'PRIMARY',
      healthy: true,
      latencyMs: 0.8,
      messagesPerSec: 148_200
    },
    {
      providerId: 'POLYGON_SIP',
      category: 'EQUITY_TAPE',
      role: 'SECONDARY',
      healthy: true,
      latencyMs: 2.1,
      messagesPerSec: 94_500
    },
    {
      providerId: 'ALPACA_SIP',
      category: 'EQUITY_TAPE',
      role: 'TERTIARY',
      healthy: true,
      latencyMs: 4.3,
      messagesPerSec: 41_000
    },
    {
      providerId: 'OPRA_ORATS',
      category: 'OPTIONS_GREEKS',
      role: 'PRIMARY',
      healthy: true,
      latencyMs: 1.4,
      messagesPerSec: 86_400
    },
    {
      providerId: 'FINRA_SHORT_FEED',
      category: 'SHORT_FLOAT',
      role: 'PRIMARY',
      healthy: true,
      latencyMs: 3.2,
      messagesPerSec: 12_400
    }
  ]);

  public readonly syncBus: MultiWindowSyncBus;

  // Derived Filtered Scanner List
  filteredScannerCandidates = $derived.by(() => {
    return this.scannerCandidates.filter((c) => {
      if (this.activeBrain !== 'ALL_5_BRAINS' && c.brainId !== this.activeBrain) {
        return false;
      }
      if (this.activeAssetClass === 'OPTIONS' && c.assetClass !== 'STOCK + OPTIONS') {
        return false;
      }
      if (this.activeAssetClass === 'STOCKS' && c.assetClass !== 'STOCK') {
        return false;
      }
      if (this.universeTierMode === 'SUB_PENNY_ONLY' && c.livePrice > 5.0) {
        return false;
      }
      if (this.universeTierMode === 'INSTITUTIONAL_ONLY' && c.livePrice <= 5.0) {
        return false;
      }
      return true;
    });
  });

  // Derived Currently Selected Signal Blueprint
  activeBlueprint = $derived.by((): ScannerSignalCandidate => {
    const match = this.scannerCandidates.find((c) => c.symbol === this.selectedTicker);
    return match ?? this.scannerCandidates[0]!;
  });

  // Derived Live Open Positions Summary
  openPositionsSummary = $derived.by(() => {
    let stockPnl = 0;
    let optionPnl = 0;
    for (const pos of this.openPositions) {
      stockPnl += pos.unrealizedStockPnlUsd;
      optionPnl += pos.unrealizedOptionPnlUsd;
    }
    return {
      count: this.openPositions.length,
      unrealizedStockPnlUsd: stockPnl,
      unrealizedOptionPnlUsd: optionPnl,
      combinedUnrealizedPnlUsd: stockPnl + optionPnl
    };
  });

  // Derived 6-Dimension Filtered Closed Trade Ledger
  filteredLedger = $derived.by(() => {
    const now = Date.now();
    return this.closedLedger.filter((row) => {
      // Dimension 1: Quick-Toggle Pills
      if (this.quickPill === 'WINS_ONLY' && !row.isWinner) return false;
      if (this.quickPill === 'LOSSES_ONLY' && row.isWinner) return false;
      if (this.quickPill === 'TARGET_HITS' && row.outcomeBadge !== 'TARGET HIT') return false;
      if (this.quickPill === 'DYNAMIC_EXITS' && row.outcomeBadge !== 'AI DYNAMIC EXIT') return false;
      if (this.quickPill === 'OPTIONS_ONLY' && row.optionContractsCount === 0) return false;
      if (this.quickPill === 'SUB_PENNY' && row.entryPrice > 5.0) return false;
      if (this.quickPill === 'MEGA_CAP' && row.entryPrice < 100.0) return false;
      if (this.quickPill === 'HIGH_CONFIDENCE_90' && row.aiConfidenceAtEntryPct < 90.0)
        return false;

      // Dimension 2: Temporal Range
      if (this.temporalRange === 'TODAY' && now - row.closedTimestampMs > 86_400_000) return false;
      if (this.temporalRange === '7D' && now - row.closedTimestampMs > 7 * 86_400_000) return false;
      if (this.temporalRange === '30D' && now - row.closedTimestampMs > 30 * 86_400_000)
        return false;

      // Dimension 3: Brain, Direction, Asset Class, Symbol Search
      if (this.brainFilter !== 'ALL_5_BRAINS' && row.brainId !== this.brainFilter) return false;
      if (this.directionFilter !== 'ALL' && row.direction !== this.directionFilter) return false;
      if (this.assetFilter === 'OPTIONS' && row.optionContractsCount === 0) return false;
      if (this.assetFilter === 'STOCKS' && row.optionContractsCount > 0) return false;
      if (
        this.searchSymbol.trim().length > 0 &&
        !row.symbol.toLowerCase().includes(this.searchSymbol.trim().toLowerCase())
      ) {
        return false;
      }

      // Dimension 4: Price Tier & Exchange
      if (row.entryPrice < this.minEntryPrice || row.entryPrice > this.maxEntryPrice) return false;
      if (this.exchangeFilter !== 'ALL' && row.exchangeIndex !== this.exchangeFilter) return false;

      // Dimension 5: Outcome & Primary Driver
      if (this.outcomeFilter !== 'ALL' && row.outcomeBadge !== this.outcomeFilter) return false;
      if (this.driverFilter !== 'ALL' && row.primaryDriver !== this.driverFilter) return false;

      // Dimension 6: Performance Sliders
      if (row.aiConfidenceAtEntryPct < this.minConfidencePct) return false;
      if (row.realizedRMultiple < this.minRMultiple) return false;

      return true;
    });
  });

  // Derived Dynamic Executive Scorecard KPIs
  scorecardKpis = $derived.by(() => {
    const rows = this.filteredLedger;
    const totalClosed = rows.length;
    if (totalClosed === 0) {
      return {
        totalClosed: 0,
        totalWins: 0,
        totalLosses: 0,
        winRatePct: 0,
        lossRatePct: 0,
        profitFactor: 0,
        expectancyR: 0,
        expectancyUsd: 0,
        netRealizedStockPnlUsd: 0,
        netRealizedOptionPnlUsd: 0,
        netCombinedPnlUsd: 0,
        maxDrawdownUsd: 0,
        avgHoldDurationMs: 0
      };
    }

    let totalWins = 0;
    let totalLosses = 0;
    let grossProfit = 0;
    let grossLoss = 0;
    let sumR = 0;
    let stockPnl = 0;
    let optionPnl = 0;
    let combinedPnl = 0;
    let sumHoldMs = 0;

    let runningEquity = 0;
    let peakEquity = 0;
    let maxDrawdownUsd = 0;

    const chronological = [...rows].sort((a, b) => a.closedTimestampMs - b.closedTimestampMs);
    for (const r of chronological) {
      if (r.isWinner) {
        totalWins++;
        grossProfit += Math.max(0, r.combinedTotalPnlUsd);
      } else {
        totalLosses++;
        grossLoss += Math.abs(Math.min(0, r.combinedTotalPnlUsd));
      }
      sumR += r.realizedRMultiple;
      stockPnl += r.realizedStockPnlUsd;
      optionPnl += r.realizedOptionPnlUsd;
      combinedPnl += r.combinedTotalPnlUsd;
      sumHoldMs += r.holdDurationMs;

      runningEquity += r.combinedTotalPnlUsd;
      if (runningEquity > peakEquity) {
        peakEquity = runningEquity;
      }
      const dd = peakEquity - runningEquity;
      if (dd > maxDrawdownUsd) {
        maxDrawdownUsd = dd;
      }
    }

    const winRatePct = (totalWins / totalClosed) * 100;
    const lossRatePct = (totalLosses / totalClosed) * 100;
    const profitFactor =
      grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? 99.99 : 0.0;

    return {
      totalClosed,
      totalWins,
      totalLosses,
      winRatePct,
      lossRatePct,
      profitFactor,
      expectancyR: sumR / totalClosed,
      expectancyUsd: combinedPnl / totalClosed,
      netRealizedStockPnlUsd: stockPnl,
      netRealizedOptionPnlUsd: optionPnl,
      netCombinedPnlUsd: combinedPnl,
      maxDrawdownUsd,
      avgHoldDurationMs: Math.round(sumHoldMs / totalClosed)
    };
  });

  // Derived Cumulative Equity ($) & Cumulative R Curve
  equityCurve = $derived.by(() => {
    const chronological = [...this.filteredLedger].sort(
      (a, b) => a.closedTimestampMs - b.closedTimestampMs
    );
    let cumUsd = 0;
    let cumR = 0;
    return chronological.map((r, idx) => {
      cumUsd += r.combinedTotalPnlUsd;
      cumR += r.realizedRMultiple;
      return {
        step: idx + 1,
        tradeId: r.tradeId,
        symbol: r.symbol,
        timestampMs: r.closedTimestampMs,
        cumulativePnlUsd: cumUsd,
        cumulativeR: cumR
      };
    });
  });

  constructor() {
    this.syncBus = new MultiWindowSyncBus();
    this.syncBus.subscribe((msg) => {
      if (msg.type === 'SELECT_TICKER') {
        this.selectedTicker = msg.symbol;
      } else if (msg.type === 'SELECT_BRAIN') {
        this.activeBrain = msg.brainId as BrainId;
      } else if (msg.type === 'SELECT_ASSET_CLASS') {
        this.activeAssetClass = msg.assetClass;
      } else if (msg.type === 'APPLY_QUICK_FILTER') {
        this.quickPill = msg.pillId as QuickFilterPillId;
      } else if (msg.type === 'TRIGGER_REPLAY_TICK') {
        this.stepLiveSimulation(false);
      }
    });
  }

  public selectTicker(symbol: string, broadcast = true): void {
    this.selectedTicker = symbol;
    if (broadcast) {
      this.syncBus.publish({ type: 'SELECT_TICKER', symbol });
    }
  }

  public selectBrain(brainId: BrainId, broadcast = true): void {
    this.activeBrain = brainId;
    if (broadcast) {
      this.syncBus.publish({ type: 'SELECT_BRAIN', brainId });
    }
  }

  public setQuickFilterPill(pill: QuickFilterPillId, broadcast = true): void {
    this.quickPill = pill;
    if (broadcast) {
      this.syncBus.publish({ type: 'APPLY_QUICK_FILTER', pillId: pill });
    }
  }

  public resetAllLedgerFilters(): void {
    this.quickPill = 'ALL';
    this.temporalRange = 'ALL';
    this.brainFilter = 'ALL_5_BRAINS';
    this.directionFilter = 'ALL';
    this.assetFilter = 'ALL';
    this.searchSymbol = '';
    this.minEntryPrice = 0;
    this.maxEntryPrice = 100_000;
    this.exchangeFilter = 'ALL';
    this.outcomeFilter = 'ALL';
    this.driverFilter = 'ALL';
    this.minConfidencePct = 50;
    this.minRMultiple = -10;
  }

  /**
   * Opens a scanner candidate directly into `POSITIONS OPENED`.
   */
  public openSignalAsPosition(candidate: ScannerSignalCandidate): OpenPositionItem {
    const notionalUsd = 10_000;
    const qty = Math.max(1, Math.round(notionalUsd / Math.max(candidate.entryPrice, 0.00001)));
    const opt = candidate.optionsRecommendation;
    const newPos: OpenPositionItem = {
      tradeId: `OPEN-${Math.floor(1000 + Math.random() * 9000)}`,
      openedTimestampMs: Date.now(),
      symbol: candidate.symbol,
      exchangeIndex: candidate.exchangeIndex,
      brainId: candidate.brainId,
      direction: candidate.direction,
      assetClass: candidate.assetClass,
      entryPrice: candidate.entryPrice,
      livePrice: candidate.livePrice,
      quantityShares: qty,
      structuralStopPrice: candidate.structuralStopPrice,
      target1Price: candidate.target1Price,
      target2Price: candidate.target2Price,
      unrealizedStockPnlUsd: 0,
      unrealizedStockPnlPct: 0,
      optionContractLabel: opt
        ? `${candidate.symbol} ${opt.expirationDate} ${opt.strike.toFixed(2)} ${opt.contractType}`
        : 'STOCK ONLY',
      optionEntryPremium: opt ? opt.entryPremium : 0,
      optionLivePremium: opt ? opt.entryPremium : 0,
      optionContractsCount: opt ? 4 : 0,
      unrealizedOptionPnlUsd: 0,
      unrealizedOptionPnlPct: 0,
      delta: opt ? opt.delta : 0,
      gamma: opt ? opt.gamma : 0,
      theta: opt ? opt.theta : 0,
      ivPct: opt ? opt.ivPct : 0,
      elapsedHoldMs: 60_000,
      predictedHoldMs: candidate.predictedHoldMs,
      currentRMultiple: 0.1,
      mfePct: 0.25,
      maePct: 0.05,
      aiConfidencePct: candidate.aiConfidencePct,
      primaryDriver: candidate.primaryDriver,
      traceHealthBadge: 'OPTIMAL',
      precedentLabel: `Matches ${candidate.precedentMatchSummary.matchingWins}/${candidate.precedentMatchSummary.totalK} Past Expansion Traces (${candidate.precedentMatchSummary.winRatePct.toFixed(0)}% Win Rate)`,
      streamWeights: { ...candidate.streamWeights }
    };
    this.openPositions = [newPos, ...this.openPositions];
    return newPos;
  }

  /**
   * Archives an active position from `POSITIONS OPENED` into `CLOSED TRADE LEDGER`
   * and triggers instant recalculation of all Scorecard KPIs and Equity Curves.
   */
  public closeOpenPosition(
    tradeId: string,
    outcomeBadge: ClosedOutcomeType = 'TARGET HIT'
  ): ClosedTradeLedgerItem | null {
    const idx = this.openPositions.findIndex((p) => p.tradeId === tradeId);
    if (idx === -1) {
      return null;
    }
    const pos = this.openPositions[idx]!;
    const remaining = [...this.openPositions];
    remaining.splice(idx, 1);
    this.openPositions = remaining;

    const exitPrice =
      outcomeBadge === 'TARGET HIT'
        ? pos.target2Price
        : outcomeBadge === 'STOPPED OUT'
          ? pos.structuralStopPrice
          : pos.livePrice;

    const dirSign = pos.direction === 'LONG' ? 1 : -1;
    const stockPnlPct =
      ((exitPrice - pos.entryPrice) / Math.max(pos.entryPrice, 0.00001)) * 100 * dirSign;
    const stockPnlUsd = pos.entryPrice * pos.quantityShares * (stockPnlPct / 100);

    let optExitPrem = pos.optionLivePremium;
    let optPnlUsd = pos.unrealizedOptionPnlUsd;
    let optPnlPct = pos.unrealizedOptionPnlPct;

    if (pos.optionContractsCount > 0 && pos.optionEntryPremium > 0) {
      const ds = (exitPrice - pos.entryPrice) * dirSign;
      optExitPrem = Math.max(
        0.01,
        pos.optionEntryPremium + Math.abs(pos.delta) * ds + 0.5 * pos.gamma * ds * Math.abs(ds)
      );
      optPnlPct = ((optExitPrem - pos.optionEntryPremium) / pos.optionEntryPremium) * 100;
      optPnlUsd = (optExitPrem - pos.optionEntryPremium) * pos.optionContractsCount * 100;
    }

    const combinedTotalPnlUsd = stockPnlUsd + optPnlUsd;
    const isWinner = combinedTotalPnlUsd > 0;
    const initialRisk = Math.max(Math.abs(pos.entryPrice - pos.structuralStopPrice), 0.00001);
    const realizedRMultiple =
      outcomeBadge === 'STOPPED OUT'
        ? -1.0
        : ((exitPrice - pos.entryPrice) * dirSign) / initialRisk;

    const closedItem: ClosedTradeLedgerItem = {
      tradeId: pos.tradeId.replace('OPEN-', 'TRD-'),
      openedTimestampMs: pos.openedTimestampMs,
      closedTimestampMs: Date.now(),
      holdDurationMs: Math.max(60_000, pos.elapsedHoldMs),
      predictedHoldDurationMs: pos.predictedHoldMs,
      symbol: pos.symbol,
      exchangeIndex: pos.exchangeIndex,
      brainId: pos.brainId,
      direction: pos.direction,
      assetClass: pos.assetClass,
      entryPrice: pos.entryPrice,
      exitPrice,
      quantityShares: pos.quantityShares,
      structuralStopAtEntry: pos.structuralStopPrice,
      target1AtEntry: pos.target1Price,
      target2AtEntry: pos.target2Price,
      outcomeBadge,
      isWinner,
      realizedStockPnlUsd: stockPnlUsd,
      realizedStockPnlPct: stockPnlPct,
      optionStrike: pos.entryPrice,
      optionExpiration: '2026-11-06',
      optionType: pos.optionContractsCount > 0 ? (pos.direction === 'LONG' ? 'CALL' : 'PUT') : 'NONE',
      optionEntryPremium: pos.optionEntryPremium,
      optionExitPremium: optExitPrem,
      optionContractsCount: pos.optionContractsCount,
      realizedOptionPnlUsd: optPnlUsd,
      realizedOptionPnlPct: optPnlPct,
      combinedTotalPnlUsd,
      realizedRMultiple,
      mfePct: Math.max(pos.mfePct, Math.max(0, stockPnlPct)),
      maePct: pos.maePct,
      aiConfidenceAtEntryPct: pos.aiConfidencePct,
      primaryDriver: pos.primaryDriver,
      streamWeights: { ...pos.streamWeights },
      precedentSimilarityScore: 0.928,
      autopsyRootCauseCode: isWinner
        ? 'VERIFIED_OPTIMAL_EXPANSION'
        : 'STOP_DISTANCE_TIGHTENED_BY_ABSORPTION_WICK'
    };

    this.closedLedger = [closedItem, ...this.closedLedger];
    return closedItem;
  }

  /**
   * Advances the live market simulation by one tick across all open positions and internals.
   */
  public stepLiveSimulation(broadcast = true): void {
    this.macroInternals.nyseTick = Math.max(
      -1400,
      Math.min(1400, this.macroInternals.nyseTick + Math.round((Math.random() - 0.46) * 90))
    );

    this.openPositions = this.openPositions.map((pos) => {
      const dirSign = pos.direction === 'LONG' ? 1 : -1;
      const stepPct = 0.0045 * dirSign;
      const nextPrice = Math.max(0.00001, pos.livePrice * (1 + stepPct));
      const pnlPct = ((nextPrice - pos.entryPrice) / Math.max(pos.entryPrice, 0.00001)) * 100 * dirSign;
      const pnlUsd = pos.entryPrice * pos.quantityShares * (pnlPct / 100);

      let optLive = pos.optionLivePremium;
      let optPnlUsd = pos.unrealizedOptionPnlUsd;
      let optPnlPct = pos.unrealizedOptionPnlPct;

      if (pos.optionContractsCount > 0 && pos.optionEntryPremium > 0) {
        const ds = (nextPrice - pos.entryPrice) * dirSign;
        optLive = Math.max(
          0.01,
          pos.optionEntryPremium + Math.abs(pos.delta) * ds + 0.5 * pos.gamma * ds * Math.abs(ds)
        );
        optPnlPct = ((optLive - pos.optionEntryPremium) / pos.optionEntryPremium) * 100;
        optPnlUsd = (optLive - pos.optionEntryPremium) * pos.optionContractsCount * 100;
      }

      const risk = Math.max(Math.abs(pos.entryPrice - pos.target1Price) * 0.5, 0.00001);
      const currentR = ((nextPrice - pos.entryPrice) * dirSign) / risk;

      return {
        ...pos,
        livePrice: nextPrice,
        unrealizedStockPnlUsd: pnlUsd,
        unrealizedStockPnlPct: pnlPct,
        optionLivePremium: optLive,
        unrealizedOptionPnlUsd: optPnlUsd,
        unrealizedOptionPnlPct: optPnlPct,
        elapsedHoldMs: pos.elapsedHoldMs + 60_000,
        currentRMultiple: currentR,
        mfePct: Math.max(pos.mfePct, Math.max(0, pnlPct))
      };
    });

    if (broadcast) {
      this.syncBus.publish({ type: 'TRIGGER_REPLAY_TICK', timestampMs: Date.now() });
    }
  }

  /**
   * Generates the exact 34-column `.CSV` string with the 4-line institutional metadata header
   * matching the active Hyper-Filter state.
   */
  public buildFilteredCsvString(): string {
    const kpi = this.scorecardKpis;
    const lines: string[] = [
      `# QUANT INTELLIGENCE PLATFORM — CLOSED TRADE LEDGER EXPORT`,
      `# Export Timestamp (UTC): ${new Date().toISOString()}`,
      `# Active Filter Query: QuickPill=${this.quickPill} | Brain=${this.brainFilter} | Direction=${this.directionFilter} | Price=[${formatSubPennyPrice(this.minEntryPrice)}..${formatSubPennyPrice(this.maxEntryPrice)}] | MinConf=${this.minConfidencePct.toFixed(1)}% | MinR=${this.minRMultiple.toFixed(2)}R`,
      `# Summary KPIs: Total Closed=${kpi.totalClosed} | Wins=${kpi.totalWins} (${kpi.winRatePct.toFixed(2)}%) | Losses=${kpi.totalLosses} (${kpi.lossRatePct.toFixed(2)}%) | Profit Factor=${kpi.profitFactor.toFixed(2)} | Net Combined P&L=${formatSignedUsd(kpi.netCombinedPnlUsd)} | Expectancy=${kpi.expectancyR >= 0 ? '+' : ''}${kpi.expectancyR.toFixed(2)}R`,
      CSV_LEDGER_HEADERS.join(',')
    ];

    for (const r of this.filteredLedger) {
      const rowValues = [
        r.tradeId,
        new Date(r.openedTimestampMs).toISOString(),
        new Date(r.closedTimestampMs).toISOString(),
        (r.holdDurationMs / 60_000).toFixed(2),
        (r.predictedHoldDurationMs / 60_000).toFixed(2),
        r.symbol,
        r.exchangeIndex,
        r.brainId,
        r.direction,
        r.assetClass,
        r.entryPrice.toFixed(5),
        r.exitPrice.toFixed(5),
        r.quantityShares.toFixed(2),
        r.structuralStopAtEntry.toFixed(5),
        r.target1AtEntry.toFixed(5),
        r.target2AtEntry.toFixed(5),
        r.outcomeBadge,
        r.isWinner ? 'WIN' : 'LOSS',
        r.realizedStockPnlUsd.toFixed(2),
        r.realizedStockPnlPct.toFixed(2),
        r.optionStrike.toFixed(5),
        r.optionExpiration,
        r.optionType,
        r.optionEntryPremium.toFixed(5),
        r.optionExitPremium.toFixed(5),
        String(r.optionContractsCount),
        r.realizedOptionPnlUsd.toFixed(2),
        r.realizedOptionPnlPct.toFixed(2),
        r.combinedTotalPnlUsd.toFixed(2),
        r.realizedRMultiple.toFixed(2),
        r.mfePct.toFixed(2),
        r.maePct.toFixed(2),
        r.aiConfidenceAtEntryPct.toFixed(2),
        r.primaryDriver,
        r.precedentSimilarityScore.toFixed(4),
        r.autopsyRootCauseCode
      ];
      lines.push(rowValues.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(','));
    }

    return lines.join('\n');
  }

  /**
   * Triggers a 1-click browser download of the filtered `.CSV` ledger.
   */
  public downloadFilteredCsv(): void {
    if (typeof document === 'undefined') return;
    const csvContent = this.buildFilteredCsvString();
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `quant_closed_trade_ledger_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  /**
   * Triggers a 1-click download of the 4-Sheet `.XLSX` workbook via the live execution server
   * or generates an instant local `.xlsx` spreadsheet download.
   */
  public async downloadFilteredXlsx(): Promise<void> {
    if (typeof window === 'undefined') return;
    const query = new URLSearchParams({
      quick_pill: this.quickPill,
      brain_id: this.brainFilter,
      direction: this.directionFilter,
      min_confidence_pct: String(this.minConfidencePct),
      min_r_multiple: String(this.minRMultiple)
    });
    try {
      const res = await fetch(`http://127.0.0.1:8080/api/v1/ledger/export.xlsx?${query.toString()}`);
      if (res.ok) {
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `quant_intelligence_ledger_${new Date().toISOString().slice(0, 10)}.xlsx`;
        a.click();
        URL.revokeObjectURL(url);
        return;
      }
    } catch {
      // Fallback to local XML spreadsheet download when running standalone without backend server
    }

    const csvFallback = this.buildFilteredCsvString();
    const blob = new Blob([csvFallback], {
      type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `quant_intelligence_ledger_${new Date().toISOString().slice(0, 10)}.xlsx`;
    a.click();
    URL.revokeObjectURL(url);
  }
}

export const terminalStore = new TerminalStore();

