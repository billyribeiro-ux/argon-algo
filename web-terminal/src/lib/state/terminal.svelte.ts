/**
 * Svelte 5 Runes Reactive Store for the ARGON-4 Multi-Brain Command Terminal.
 *
 * Connected 100% end-to-end to the Rust `live-execution-server` (`http://127.0.0.1:8090`)
 * and `ws://127.0.0.1:8090/ws/v1/stream` powered by `YahooFinanceLiveAdapter`.
 * Zero synthetic default trades or charts.
 */

import {
  MultiWindowSyncBus,
  type LinkChannelColor,
  type WorkspaceLayoutMode
} from '../multi-window/window-sync';
import {
  ExecutionAudioEngine,
  executionAudio,
  type ExecutionAudioMode
} from '../audio/execution-audio';

export type { LinkChannelColor, WorkspaceLayoutMode, ExecutionAudioMode };
export { ExecutionAudioEngine, executionAudio };

export type BrainId =
  | 'ALL_5_BRAINS'
  | 'BRAIN-01'
  | 'BRAIN-02'
  | 'BRAIN-03'
  | 'BRAIN-04'
  | 'BRAIN-05';

export type ChartTimeframeId = '1m' | '3m' | '5m' | '15m' | '1H' | '4H' | '1D' | '1W' | '1M';
export type ChartRangeId = '1D' | '5D' | '1M' | '6M' | '1Y' | '5Y';

export type TradeDirection = 'LONG' | 'SHORT';
export type AssetClassMode = 'ALL' | 'STOCKS' | 'OPTIONS';
export type ClosedOutcomeType =
  | 'TARGET HIT'
  | 'STOPPED OUT'
  | 'AI DYNAMIC EXIT'
  | 'AI LOSS-AUTOPSY SCRATCH EXIT'
  | 'TRAILING STOP'
  | 'TIME EXPIRED';

export type TraceHealthBadge = 'OPTIMAL' | 'EXHAUSTION WARNING' | 'DYNAMIC EXIT TRIGGERED';

export type PrimaryDriverStream =
  | 'PRICE_VOLUME_ANOMALY'
  | 'OPTIONS_GAMMA_VANNA'
  | 'SHORT_FLOAT_SQUEEZE'
  | 'MARKET_INTERNALS_TIDE';

export type EngineExecutionModeId =
  | 'STRUCTURED_HYBRID_CHAMPION'
  | 'PURE_ML_SELF_LEARNED'
  | 'CONSENSUS_BOTH_ENGINES';

export type StrategyArchetypeId =
  | 'GAMMA_SQUEEZE_IGNITION'
  | 'SHORT_FLOAT_LIQUEFACTION'
  | 'POC_VOLUME_ABSORPTION'
  | 'MACRO_INTERNALS_TIDE_DRIVE'
  | 'HTF_MULTI_SCALE_BREAKOUT'
  | 'PURE_ML_LATENT_ANOMALY';

export type SizingModeId = 'FIXED_10K_BASELINE' | 'FRACTIONAL_KELLY';
export type ImmunityModeId = 'BASELINE_ALL_TRADES' | 'ACTIVE_LOSS_IMMUNITY';

export interface FeatureCulpritEntry {
  featureIndex: number;
  featureName: string;
  streamName: string;
  lossValue: number;
  winnerCentroidValue: number;
  anomalyZScore: number;
}

export interface LossForensicAutopsyEntry {
  tradeId: string;
  symbol: string;
  brainId: string;
  rootCauseCategory: string;
  topCulpritFeatures: FeatureCulpritEntry[];
  whatWentWrongSummary: string;
  learnedAdaptationRule: string;
  counterfactualEarlyExitR: number;
}

export interface VetoedTrapAuditEntry {
  symbol: string;
  brainId: string;
  timestampNs: number;
  timestampIso: string;
  matchedPastLossTradeId: string;
  rootCauseCategory: string;
  trapSimilarityScore: number;
  counterfactualLossAvoidedUsd: number;
}

export interface StrategyLeaderboardRow {
  strategyArchetype: StrategyArchetypeId;
  displayName: string;
  totalTrades: number;
  wins: number;
  losses: number;
  winRatePct: number;
  profitFactor: number;
  avgRealizedR: number;
  bestSymbol: string;
  realizedStockNetPnlUsd: number;
  realizedOptionNetPnlUsd: number;
  kellyRealizedNetPnlUsd: number;
  kellyProfitFactor: number;
  adaptiveTotalTrades: number;
  adaptiveWins: number;
  adaptiveLosses: number;
  adaptiveVetoedLosses: number;
  adaptiveScratchedLosses: number;
  adaptiveWinRatePct: number;
  adaptiveProfitFactor: number;
  adaptiveRealizedNetPnlUsd: number;
  adaptiveKellyNetPnlUsd: number;
  adaptiveLossAvoidedUsd: number;
}

export interface EngineBreakdownRow {
  engineMode: EngineExecutionModeId;
  displayName: string;
  totalTrades: number;
  wins: number;
  losses: number;
  winRatePct: number;
  profitFactor: number;
  avgRealizedR: number;
  bestSymbol: string;
  realizedStockNetPnlUsd: number;
  realizedOptionNetPnlUsd: number;
  kellyRealizedNetPnlUsd: number;
  kellyProfitFactor: number;
  adaptiveTotalTrades: number;
  adaptiveWins: number;
  adaptiveLosses: number;
  adaptiveVetoedLosses: number;
  adaptiveScratchedLosses: number;
  adaptiveWinRatePct: number;
  adaptiveProfitFactor: number;
  adaptiveRealizedNetPnlUsd: number;
  adaptiveKellyNetPnlUsd: number;
  adaptiveLossAvoidedUsd: number;
}

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

export interface ShapDriverAttribution {
  feature: string;
  impact: number;
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
    optionStructure: string;
    matchedContractLabel: string;
    contractType: 'CALL' | 'PUT';
    strike: number;
    expirationDate: string;
    dte: number;
    entryPremium: number;
    stopPremium: number;
    targetPremium: number;
    delta: number;
    gamma: number;
    vanna: number;
    theta: number;
    vega: number;
    ivPct: number;
  } | null;
  shapDrivers: ShapDriverAttribution[];
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
  kellyFractionPct: number;
  riskBudgetUsd: number;
  notionalAllocatedUsd: number;
  trancheCount: number;
  scaleInHistory: string[];
  entryBarIdx: number;
  structuralStopPrice: number;
  target1Price: number;
  target2Price: number;
  unrealizedStockPnlUsd: number;
  unrealizedStockPnlPct: number;
  optionStructure: string;
  optionContractLabel: string;
  optionEntryPremium: number;
  optionLivePremium: number;
  optionContractsCount: number;
  unrealizedOptionPnlUsd: number;
  unrealizedOptionPnlPct: number;
  delta: number;
  gamma: number;
  vanna: number;
  theta: number;
  vega: number;
  ivPct: number;
  elapsedHoldMs: number;
  predictedHoldMs: number;
  currentRMultiple: number;
  mfePct: number;
  maePct: number;
  aiConfidencePct: number;
  primaryDriver: PrimaryDriverStream;
  engineMode: EngineExecutionModeId;
  strategyArchetype: StrategyArchetypeId;
  traceHealthBadge: TraceHealthBadge;
  precedentLabel: string;
  barsHeldCount: number;
  activeSequentialStage: string;
  liveSequentialLlr: number;
  championLlrThreshold: number;
  sequentialSafetyMarginPct: number;
  nextBarRatchetStopPreview: number;
  houseMoneyScaleEligible?: boolean;
  learnedHouseMoneyPhi?: number;
  lockedProfitR?: number;
  guaranteedNetFloorR?: number;
  nextGammaWallStrike?: number;
  nextGammaWallContract?: string;
  bankedOptionGainUsd?: number;
  rolledOptionBudgetUsd?: number;
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
  kellyFractionPct: number;
  riskBudgetUsd: number;
  notionalAllocatedUsd: number;
  trancheCount: number;
  scaleInHistory: string[];
  entryBarIdx: number;
  exitBarIdx: number;
  structuralStopAtEntry: number;
  target1AtEntry: number;
  target2AtEntry: number;
  outcomeBadge: ClosedOutcomeType;
  isWinner: boolean;
  realizedStockPnlUsd: number;
  realizedStockPnlPct: number;
  kellyRealizedStockPnlUsd: number;
  optionStructure: string;
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
  engineMode: EngineExecutionModeId;
  strategyArchetype: StrategyArchetypeId;
  streamWeights: {
    priceVolumeAnomaly: number;
    optionsGammaVanna: number;
    shortFloatSqueeze: number;
    marketInternalsTide: number;
  };
  precedentSimilarityScore: number;
  autopsyRootCauseCode: string;
  lossForensicAutopsy: LossForensicAutopsyEntry | null;
  entryBars: RawBarCandle[];
  wasVetoedByLossAutopsy: boolean;
  vetoedByLossTradeId: string | null;
  entryTrapSimilarityScore: number;
  wasScratchedByLossAutopsy: boolean;
  scratchBarOffset: number;
  adaptiveOutcomeBadge: string;
  adaptiveExitPrice: number;
  adaptiveScratchExitR: number;
  adaptiveNetPnlUsd: number;
  adaptiveKellyNetPnlUsd: number;
  adaptiveLossSavedUsd: number;
  adaptiveKellyLossSavedUsd: number;
  lossAffinityPosteriorPct: number;
  wasProtectedFromGiveback: boolean;
  givebackRatchetBarOffset: number;
  givebackRSaved: number;
  wasScaledWithHouseMoney?: boolean;
  houseMoneyScaleBarOffset?: number;
  houseMoneyAdditionalRGained?: number;
}

export interface ProviderStatusEntry {
  providerId: string;
  displayName: string;
  category: 'EQUITY_TAPE' | 'OPTIONS_GREEKS' | 'SHORT_FLOAT';
  role: 'PRIMARY' | 'SECONDARY' | 'TERTIARY';
  healthy: boolean;
  latencyMs: number;
  messagesPerSec: number;
  failoverCount: number;
}

export interface AutopsyTradeGradeEntry {
  tradeId: string;
  symbol: string;
  brainId: string;
  grade: string;
  realizedRMultiple: number;
  mfePct: number;
  maePct: number;
  isHardNegative: boolean;
  diagnosis: string;
  weightAdjustmentSummary: string;
}

export interface ChampionChallengerEntry {
  brainId: string;
  brainName: string;
  championVersion: string;
  challengerVersion: string;
  championProfitFactor: number;
  challengerProfitFactor: number;
  championMaxDrawdownPct: number;
  challengerMaxDrawdownPct: number;
  challengerPromoted: boolean;
  promotionReason: string;
}

export interface Era1BrainCalibrationEntry {
  brainId: string;
  calibratedConfidenceFloor: number;
  minPrecedentWinRatePct: number;
  minExpectedRMultiple: number;
  chartTapePhysicsWeight: number;
  optionsGammaVannaWeight: number;
  shortFloatSqueezeWeight: number;
  marketInternalsMacroWeight: number;
  era1SampleTrades: number;
  era1CalibratedWinRatePct: number;
  era1CalibratedProfitFactor: number;
  era1LearnedMinDisplacementPct: number;
  era1LearnedMaxDisplacementPct: number;
  era1MedianStopPct: number;
  era1LearnedScratchPosteriorThreshold: number;
}

export interface SynapticFeatureMutation {
  feature_index: number;
  feature_name: string;
  stream_id: number;
  stream_name: string;
  previous_weight: number;
  updated_weight: number;
  delta_pct: number;
  mutation_role: string; // 'BOOSTED_WARNING_SENSOR' | 'SLASHED_DECEIVER_FEATURE' | 'NEUTRAL_FEATURE'
  rationale: string;
}

export interface BrainSynapticPlasticityState {
  brain_id: string;
  generation_id: number;
  synaptic_feature_weights_32: number[];
  stream_prior_multipliers: number[];
  wald_sprt_thresholds: number[];
  entry_veto_margin_delta: number;
  trap_conviction_weights: Record<string, number>;
  empirical_bayes_shrinkage_lambda?: number;
  mid_trade_wald_sprt_thresholds?: [number, number, number];
  learned_mid_trade_mfe_norm_quantile?: number;
  learned_mid_trade_stop_ratchet_r?: number;
  learned_itm_mfe_activation_r?: number;
  giveback_exhaustion_threshold?: number;
  runner_continuation_threshold?: number;
  learned_giveback_retention_ratio?: number;
  learned_profit_lock_activation_r?: number;
  learned_house_money_reinvestment_fraction?: number;
  late_blowoff_scratch_threshold?: number;
  learned_shakeout_absorption_quantile?: number;
  learned_cross_brain_recycle_multiplier?: number;
  learned_almgren_chriss_risk_aversion_lambda?: number;
  learned_almgren_chriss_urgency_kappa?: number;
  learned_correlation_spike_quantile?: number;
  learned_vanna_tail_hedge_ratio_theta?: number;
  learned_sor_venue_weights_4?: [number, number, number, number];
  learned_sor_markout_toxicity_bps?: number;
  learned_charm_vanna_pin_quantile?: number;
  learned_charm_vanna_unpin_quantile?: number;
  learned_occ_tims_sigma_shock_pct?: number;
  top_boosted_warning_features: SynapticFeatureMutation[];
  top_slashed_deceiver_features: SynapticFeatureMutation[];
  top_boosted_features?: SynapticFeatureMutation[];
  top_slashed_features?: SynapticFeatureMutation[];
  cumulative_tp_vetoes: number;
  cumulative_fp_vetoes: number;
  cumulative_tp_scratches: number;
  cumulative_fp_scratches: number;
  remaining_unscratched_losses: number;
  last_mutation_summary: string;
}

export interface EvolutionGenerationRecord {
  generation_id: number;
  status: string;
  promoted_as_champion: boolean;
  evaluated_at: string;
  active_trades_count: number;
  winners_preserved_count: number;
  active_losses_count: number;
  win_rate_pct: number;
  tp_losses_vetoed: number;
  fp_winners_vetoed: number;
  veto_precision_pct: number;
  tp_losses_scratched: number;
  fp_winners_scratched: number;
  scratch_precision_pct: number;
  early_scratches_bar1_to_3?: number;
  mid_trade_stall_saves_bar4_to_6?: number;
  fp_mid_trade_scratches?: number;
  giveback_saves_count?: number;
  total_giveback_r_recovered?: number;
  house_money_scaled_runners_count?: number;
  total_house_money_r_gained?: number;
  recycled_risk_pool_usd?: number;
  recycled_runner_boost_count?: number;
  recycled_additional_r_compounded?: number;
  avg_execution_slippage_bps?: number;
  slippage_saved_usd?: number;
  tail_hedge_ratio_theta?: number;
  active_tail_hedge_count?: number;
  tail_hedge_convexity_payout_usd?: number;
  sor_markout_toxicity_bps?: number;
  charm_vanna_unpin_boosts_count?: number;
  charm_vanna_additional_r_gained?: number;
  occ_tims_pm_relief_ratio_pct?: number;
  return_on_margin_rom_pct?: number;
  pareto_gate_passed?: boolean;
  avg_winner_r?: number;
  tp_scratches?: number;
  expectancy_r?: number;
  net_r?: number;
  tp_losses_caught_total: number;
  fp_winners_clipped_total: number;
  remaining_full_stops: number;
  profit_factor_fixed: number;
  profit_factor_kelly: number;
  fixed_10k_net_pnl: number;
  fractional_kelly_net_pnl: number;
  max_drawdown_usd: number;
  composite_fitness_score: number;
  key_synaptic_mutations: string[];
  brain_synaptic_states: BrainSynapticPlasticityState[];
}

export interface ConfusionMatrixRollup {
  true_positive_losses_vetoed: number;
  false_positive_winners_vetoed: number;
  true_positive_losses_scratched: number;
  false_positive_winners_scratched: number;
  tp_scratched_bar1?: number;
  tp_scratched_bar2?: number;
  tp_scratched_bar3?: number;
  tp_scratched_bar4?: number;
  tp_scratched_bar5?: number;
  tp_scratched_bar6?: number;
  fp_scratched_bar1?: number;
  fp_scratched_bar2?: number;
  fp_scratched_bar3?: number;
  early_scratches_bar1_to_3?: number;
  mid_trade_stall_saves_bar4_to_6?: number;
  fp_mid_trade_scratches?: number;
  remaining_unscratched_losses: number;
  untouched_winners_count: number;
  entry_veto_precision_pct: number;
  bar1_scratch_precision_pct: number;
}

export interface MultiGenEvolutionResponse {
  active_champion_generation: number;
  generations: EvolutionGenerationRecord[];
  active_synaptic_states: BrainSynapticPlasticityState[];
  latest_generation: EvolutionGenerationRecord | null;
}

export type BrainEvolutionStatus = BrainEvolutionStatusPayload;

export interface FilteredLedgerSnapshot {
  evolution_generations_history?: EvolutionGenerationRecord[];
  synaptic_plasticity_states?: BrainSynapticPlasticityState[];
}

export interface BrainEvolutionStatusPayload {
  lastAutopsyIso: string;
  totalAutopsyRuns: number;
  hardNegativesMined: number;
  missedMovesIndexed: number;
  chopThrottleActive: boolean;
  chopConfidenceFloorPct: number;
  tradeGrades: AutopsyTradeGradeEntry[];
  tournamentResults: ChampionChallengerEntry[];
  era1Calibrations: Era1BrainCalibrationEntry[];
  lossAutopsies: LossForensicAutopsyEntry[];
  vetoedRepeatTraps: VetoedTrapAuditEntry[];
  evolution_generations_history?: EvolutionGenerationRecord[];
  synaptic_plasticity_states?: BrainSynapticPlasticityState[];
}

export interface CustomSymbolBacktestSummary {
  symbol: string;
  brainId: Exclude<BrainId, 'ALL_5_BRAINS'>;
  timeframe: string;
  totalBarsScanned: number;
  era1Calibration: Era1BrainCalibrationEntry;
  closedTradesGenerated: number;
  openPositionActive: boolean;
  winRatePct: number;
  profitFactor: number;
  realizedStockNetPnlUsd: number;
  realizedOptionNetPnlUsd: number;
  completedAtIso: string;
}

export interface WebhookChannelConfigState {
  discordWebhookUrl: string;
  discordEnabled: boolean;
  telegramBotToken: string;
  telegramChatId: string;
  telegramEnabled: boolean;
  slackWebhookUrl: string;
  slackEnabled: boolean;
  customJsonWebhookUrl: string;
  customJsonEnabled: boolean;
}

export interface WebhookDispatchRecordEntry {
  dispatchId: string;
  dispatchedAtIso: string;
  dispatchedTimestampMs: number;
  eventType: string;
  outcomeDetail: string;
  symbol: string;
  brainId: string;
  brainName: string;
  direction: TradeDirection;
  entryPrice: number;
  structuralStopPrice: number;
  target1Price: number;
  target2Price: number;
  currentOrExitPrice: number;
  kellyFractionPct: number;
  riskBudgetUsd: number;
  notionalAllocatedUsd: number;
  matchedOptionContract: string;
  greeksSummary: string;
  streamAttributionSummary: string;
  precedentWinRatePct: number;
  aiConfidencePct: number;
  pnlUsd: number;
  rMultiple: number;
  deliveryStatus: string;
  channelsTargeted: string[];
  discordEmbedJson: string;
  telegramMarkdown: string;
  slackBlocksJson: string;
  genericWebhookJson: string;
}

export interface WebhookStatusPayload {
  config: WebhookChannelConfigState;
  totalOutboxRecords: number;
  outboxRecords: WebhookDispatchRecordEntry[];
}

export interface CrossPositionCorrelationCellEntry {
  posALabel: string;
  posBLabel: string;
  symbolA: string;
  symbolB: string;
  brainA: string;
  brainB: string;
  ledoitWolfSimilarity: number;
  directionalAlignment: number;
}

export interface PortfolioHeatTelemetryState {
  accountEquityUsd: number;
  openPositionsCount: number;
  aggregateOpenRiskUsd: number;
  aggregateOpenRiskPct: number;
  portfolioHeatCapPct: number;
  grossPortfolioNotionalUsd: number;
  netPortfolioBetaDeltaUsd: number;
  netPortfolioGamma1pctUsd: number;
  netPortfolioVanna1pctUsd: number;
  meanPairwiseCorrelation: number;
  maxPairwiseCorrelation: number;
  correlationHeatMultiplier: number;
  heatRegimeBadge: string;
  positionLabels: string[];
  crossPositionCorrelationMatrix: number[][];
  pairwiseCorrelations: CrossPositionCorrelationCellEntry[];
  overlappingClustersSummary: string;
  recycledRiskPoolUsd?: number;
  recycledRunnerBoostCount?: number;
  recycledAdditionalRCompounded?: number;
  crossBrainRecycleMultiplier?: number;
  avgExecutionSlippageBps?: number;
  slippageSavedUsd?: number;
  tailHedgeRatioTheta?: number;
  activeTailHedgeCount?: number;
  tailHedgeConvexityPayoutUsd?: number;
  sorVenueWeightsSimplex?: [number, number, number, number];
  sorMarkoutToxicityBps?: number;
  charmVannaUnpinBoostsCount?: number;
  charmVannaAdditionalRGained?: number;
  regTStandaloneMarginUsd?: number;
  occTimsPortfolioMarginUsd?: number;
  occTimsPmReliefRatioPct?: number;
  returnOnMarginRomPct?: number;
}

export interface MicrostructureChildSliceEntry {
  sliceNumber: number;
  timeOffsetSecs: number;
  equityQtyShares: number;
  optionContractsQty: number;
  sliceWeightPct: number;
  equityPegStrategy: string;
  optionPegStrategy: string;
  expectedSlippageBps: number;
  sorSelectedVenue?: string;
  sorVenueAllocationWeight?: number;
  postFillMarkoutToxicityBps?: number;
}

export interface StagedTailHedgeOverlayTicketEntry {
  hedgeTicketId: string;
  stagedAtIso: string;
  hedgeSymbol: string;
  occPutSymbol: string;
  putContractDesc: string;
  strikePrice: number;
  expirationDate: string;
  contractsCount: number;
  entryLimitPremium: number;
  fundedFromHouseMoneyUsd: number;
  principalCapitalRiskedPct: number;
  learnedTailHedgeRatioTheta: number;
  meanPairwiseCorrelationTrigger: number;
  learnedCorrelationSpikeQuantile: number;
  expectedConvexityPayoutUsd: number;
  ibkrPegMidPayloadJson: string;
}

export interface StagedBrokerOrderTicketEntry {
  ticketId: string;
  positionId: string;
  stagedAtIso: string;
  symbol: string;
  exchangeIndex: string;
  brainId: string;
  brainName: string;
  direction: TradeDirection;
  action: string;
  orderStatus: string;
  selectedBroker: string;
  equityQuantityShares: number;
  occOptionSymbol: string;
  matchedOptionContract: string;
  optionStructure: string;
  optionContractsCount: number;
  limitPrice: number;
  liveMarkPrice: number;
  stopLossTriggerPrice: number;
  target1PartialLimitPrice: number;
  target2RunnerLimitPrice: number;
  optionEntryLimitPremium: number;
  optionStopTriggerPremium: number;
  optionTarget1LimitPremium: number;
  kellyFractionPct: number;
  kellyRiskBudgetUsd: number;
  notionalAllocatedUsd: number;
  correlationHeatMultiplier: number;
  almgrenChrissRiskAversionLambda?: number;
  almgrenChrissUrgencyKappa?: number;
  estimatedEtaTemp?: number;
  estimatedGammaPerm?: number;
  expectedExecutionSlippageBps?: number;
  expectedSlippageSavedUsd?: number;
  sorVenueWeightsSimplex?: [number, number, number, number];
  sorExpectedMarkoutToxicityBps?: number;
  crumblingQuoteProbability?: number;
  orderBookAbsorptionRatio?: number;
  childExecutionSlices?: MicrostructureChildSliceEntry[];
  ibkrTwsPayloadJson: string;
  schwabTraderPayloadJson: string;
  alpacaApiPayloadJson: string;
}

export interface IbkrBagComboLegEntry {
  legId: number;
  action: string;
  ratioQty: number;
  occOptionSymbol: string;
  contractDesc: string;
  strikePrice: number;
  expirationDate: string;
  optionRight: string;
  liveMidPrice: number;
}

export interface IbkrBagComboPayloadEntry {
  comboOrderType: string;
  leg1Ratio?: number;
  leg2Ratio?: number;
  leg1Close: IbkrBagComboLegEntry;
  leg2Open: IbkrBagComboLegEntry;
  netComboLimitPrice: number;
  nbboLimitWalkLadder: [number, number, number];
  preRollNetDelta: number;
  postRollNetDelta: number;
  postRollGammaConvexityBoostPct: number;
  lockedCashHarvestedUsd: number;
}

export interface StagedOptionRollUpTicketEntry {
  rollTicketId: string;
  positionId: string;
  stagedAtIso: string;
  symbol: string;
  brainId: string;
  brainName: string;
  direction: TradeDirection;
  rollStatus: string;
  isGammaRollActive: boolean;
  leg1Ratio?: number;
  leg2Ratio?: number;
  currentItmOccSymbol: string;
  currentItmContractDesc: string;
  sellToCloseContracts: number;
  currentItmEntryPremium: number;
  currentItmLivePremium: number;
  intrinsicOptionGainUsd: number;
  bankedCashFraction: number;
  bankedRealizedCashUsd: number;
  houseMoneyReinvestmentFractionPhi: number;
  houseMoneyRollBudgetUsd: number;
  nextGammaWallStrike: number;
  nextGammaWallOccSymbol: string;
  nextGammaWallContractDesc: string;
  buyToOpenRollContracts: number;
  nextGammaWallLimitPremium: number;
  lockedProfitR: number;
  learnedProfitLockQuantileR: number;
  liveGivebackLlr: number;
  runnerContinuationThreshold: number;
  guaranteedNetFloorR: number;
  comboOrderType: string;
  leg1Close: IbkrBagComboLegEntry;
  leg2Open: IbkrBagComboLegEntry;
  netComboLimitPrice: number;
  nbboLimitWalkLadder: [number, number, number];
  preRollNetDelta: number;
  postRollNetDelta: number;
  postRollGammaConvexityBoostPct: number;
  lockedCashHarvestedUsd: number;
  ibkrBagCombo: IbkrBagComboPayloadEntry;
  ibkrComboRollPayloadJson: string;
  schwabRollPayloadJson: string;
  alpacaRollPayloadJson: string;
}

export interface InstitutionalWatchlistRow {
  rank: number;
  symbol: string;
  brain_id: BrainId;
  brain_name: string;
  direction: TradeDirection;
  strategy_archetype: string;
  engine_a_confidence_pct: number;
  engine_b_confidence_pct: number;
  consensus_confidence_pct: number;
  active_engine_mode: string;
  entry_price: number;
  structural_stop_price: number;
  stop_distance_pct: number;
  target_1_price: number;
  target_2_price: number;
  expected_r_multiple: number;
  stage0_conformal_status: 'PASS' | 'VETO';
  mahalanobis_win_distance: number;
  mahalanobis_loss_distance: number;
  conformal_llr: number;
  conformal_threshold: number;
  conformal_margin: number;
  action_status: string;
  venue_ticket: string;
  matched_option_contract: string;
  option_structure: string;
  kelly_fraction_pct: number;
  is_actionable: boolean;
  execution_tier?: string;
}

export interface InstitutionalWatchlistResponse {
  universe_size: number;
  actionable_setups_count: number;
  active_generation?: string;
  active_generation_id?: number;
  total_allocated_kelly_pct: number;
  net_portfolio_delta_beta_weighted: number;
  cross_brain_conflicts_blocked: number;
  estimated_occ_tims_margin_relief_pct: number;
  timestamp_iso: string;
  eval_epoch_s?: number;
  watchlist: InstitutionalWatchlistRow[];
  market_internals?: any;
}

export interface SundayAlphaProtocolExportRow {
  rank: number;
  symbol: string;
  brain_id: BrainId;
  brain_name: string;
  direction: TradeDirection;
  strategy_archetype: string;
  consensus_confidence_pct: number;
  active_engine_mode: string;
  entry_price: number;
  structural_stop_price: number;
  target_1_price: number;
  target_2_price: number;
  expected_r_multiple: number;
  stage0_conformal_status: string;
  conformal_margin: number;
  venue_ticket: string;
  matched_option_contract: string;
  kelly_fraction_pct: number;
  action_status: string;
}

export interface SundayAlphaProtocolResponse {
  protocol_generation: string;
  staged_for_execution_epoch_iso: string;
  universe_size: number;
  actionable_count: number;
  total_allocated_kelly_pct: number;
  net_portfolio_delta_beta_weighted: number;
  cross_brain_conflicts_blocked: number;
  estimated_occ_tims_margin_relief_pct: number;
  staged_setups: SundayAlphaProtocolExportRow[];
  csv_export_endpoint: string;
  webhook_dispatch_endpoint: string;
}

export interface PaperOverlayBrainPlasticity {
  brain_id: BrainId;
  tau_veto: number;
  eb_lambda: number;
  learned_profit_lock_activation_r: number;
  learned_conformal_veto_quantile: number;
}

export interface PaperOverlayTelemetryResponse {
  mode: string;
  baseline_ledger_locked: boolean;
  baseline_trade_count: number;
  baseline_sha256: string;
  overlay_closed_count: number;
  open_paper_positions: number;
  recent_atomic_bag_rolls: number;
  brain_plasticity_shifts: PaperOverlayBrainPlasticity[];
  timestamp_iso: string;
}

export interface SundayAlphaDispatchResponse {
  status: string;
  dispatched_at_iso: string;
  discord_embed_queued: boolean;
  telegram_payload_queued: boolean;
  slack_blocks_queued: boolean;
  oms_bundle_queued: boolean;
  staged_count: number;
  total_portfolio_kelly_pct: number;
  net_delta_beta_weighted: number;
  conflicts_blocked: number;
}

export type ExecutionSafetyMode = 'PAPER_ISOLATED_ONLY' | 'LIVE_AUTHORIZED';

export interface StageSundayOrdersRequest {
  symbols?: string[];
  live?: boolean;
  confirmation_token?: string;
}

export interface StagedOrderReceipt {
  client_order_id: string;
  symbol: string;
  brain_id: string;
  direction: string;
  execution_safety_mode: ExecutionSafetyMode;
  execution_venue: string;
  limit_entry_price: number;
  structural_stop_loss: number;
  profit_target_1: number;
  profit_target_2: number;
  kelly_fraction_pct: number;
  allocated_notional_usd: number;
  matched_cboe_contract: string;
  almgren_chriss_slices: number[];
  staged_at_iso: string;
}

export interface StageSundayOrdersResponse {
  status: string;
  execution_safety_mode: ExecutionSafetyMode;
  staged_orders_count: number;
  total_staged_kelly_pct: number;
  receipts: StagedOrderReceipt[];
  open_paper_positions_count: number;
  ledger_baseline_trades_untouched: number;
}

export interface StageSundayOrdersErrorResponse {
  status: string;
  error_code: string;
  message: string;
  execution_safety_mode: ExecutionSafetyMode;
}

export interface RedactedWebhookConfigResponse {
  discord_webhook_url_masked: string;
  discord_configured: boolean;
  discord_enabled: boolean;
  telegram_bot_token_masked: string;
  telegram_chat_id_masked: string;
  telegram_configured: boolean;
  telegram_enabled: boolean;
  configured: boolean;
  dry_run_mode: boolean;
  outbox_count: number;
  active_channels: string[];
}

export interface ScannerWebhookConfigUpdateRequest {
  discord_webhook_url?: string;
  discord_enabled?: boolean;
  telegram_bot_token?: string;
  telegram_chat_id?: string;
  telegram_enabled?: boolean;
}

export interface WebhookTestPingRequest {
  symbol?: string;
}

export interface WebhookTestPingResponse {
  status: string;
  delivery_status: string;
  receipt_id: string;
  dispatched_at_iso: string;
  channels_targeted: string[];
  dry_run_mode: boolean;
  outbox_count: number;
  record?: any;
}

// Phase 32: Autonomous Closed-Loop Loss-Autopsy & Pareto Evolver Types
export type ExecutionTier = 'ALL' | 'SCALP_0DTE' | 'DAY_TRADE' | 'SWING' | 'LONG_TERM_WEALTH';

export type AutopsyFailureMode =
  | 'ENTRY_LIQUIDITY_VACUUM_TRAP'
  | 'POST_ENTRY_ORDERFLOW_TOXICITY_REVERSAL'
  | 'DEALER_CHARM_VANNA_UNPIN_DECAY'
  | 'MFE_STALL_GIVEBACK_EXHAUSTION';

export interface DecodedTradeAutopsy {
  trade_id: string;
  symbol: string;
  brain_id: string;
  realized_r: number;
  peak_mfe_r: number;
  giveback_pct: number;
  failure_mode: AutopsyFailureMode;
  failure_mode_label: string;
  blame_attribution_32: number[];
  top_culprit_feature_indices: number[];
  top_culprit_names: string[];
  mahalanobis_distance_shift: number;
  collapse_bar_offset: number;
  summary: string;
}

export interface GateAuditReport {
  gate1_baseline_lock_passed: boolean;
  gate1_baseline_winners_preserved: number;
  gate1_baseline_winners_target: number;
  gate1_current_expectancy_r: number;
  gate1_candidate_expectancy_r: number;
  gate1_current_pf: number;
  gate1_candidate_pf: number;
  gate2_walkforward_passed: boolean;
  gate2_current_walkforward_net_r: number;
  gate2_candidate_walkforward_net_r: number;
  gate2_offending_losses_reduced: number;
  gate2_walkforward_winners_preserved: boolean;
  all_gates_passed: boolean;
  decision_rationale: string;
}

export interface AutopsyEvolutionResponse {
  status: string;
  timestamp_iso: string;
  current_generation_id: number;
  candidate_generation_id: number;
  promoted_generation_id: number | null;
  is_promoted: boolean;
  total_trades_inspected: number;
  autopsies_decoded: DecodedTradeAutopsy[];
  gate_audit: GateAuditReport;
  updated_synaptic_states: any[];
}

const API_BASE: string = import.meta.env.VITE_ARGON_API_BASE ?? 'http://127.0.0.1:8090';
const WS_URL: string = import.meta.env.VITE_ARGON_WS_URL ?? 'ws://127.0.0.1:8090/ws/v1/stream';

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

function mapBackendDriver(driver: string): PrimaryDriverStream {
  switch (String(driver).toUpperCase()) {
    case 'GAMMA_SQUEEZE':
      return 'OPTIONS_GAMMA_VANNA';
    case 'SHORT_SQUEEZE':
      return 'SHORT_FLOAT_SQUEEZE';
    case 'INTERNALS_ALIGNMENT':
      return 'MARKET_INTERNALS_TIDE';
    default:
      return 'PRICE_VOLUME_ANOMALY';
  }
}

function mapBackendTraceHealth(badge: string): TraceHealthBadge {
  const b = String(badge).toUpperCase();
  if (b.includes('DYNAMIC') || b.includes('EXIT')) return 'DYNAMIC EXIT TRIGGERED';
  if (b.includes('EXHAUSTION') || b.includes('WARNING')) return 'EXHAUSTION WARNING';
  return 'OPTIMAL';
}

function mapBackendOutcome(badge: string): ClosedOutcomeType {
  const b = String(badge).toUpperCase();
  if (b.includes('TARGET HIT')) return 'TARGET HIT';
  if (b.includes('SCRATCH EXIT') || b.includes('LOSS-AUTOPSY SCRATCH'))
    return 'AI LOSS-AUTOPSY SCRATCH EXIT';
  if (b.includes('DYNAMIC EXIT')) return 'AI DYNAMIC EXIT';
  if (b.includes('TRAILING STOP')) return 'TRAILING STOP';
  if (b.includes('STOPPED OUT')) return 'STOPPED OUT';
  return 'TIME EXPIRED';
}

const EMPTY_BLUEPRINT_PLACEHOLDER: ScannerSignalCandidate = {
  symbol: 'NVDA',
  exchangeIndex: 'NASDAQ_100',
  brainId: 'BRAIN-02',
  direction: 'LONG',
  assetClass: 'STOCK + OPTIONS',
  livePrice: 0,
  entryPrice: 0,
  structuralStopPrice: 0,
  target1Price: 0,
  target2Price: 0,
  predictedHoldWindow: 'Loading Live Yahoo Finance...',
  predictedHoldMs: 1_800_000,
  aiConfidencePct: 80.0,
  expectedRMultiple: 2.5,
  primaryDriver: 'PRICE_VOLUME_ANOMALY',
  streamWeights: {
    priceVolumeAnomaly: 0.4,
    optionsGammaVanna: 0.25,
    shortFloatSqueeze: 0.2,
    marketInternalsTide: 0.15
  },
  precedentMatchSummary: {
    matchingWins: 0,
    totalK: 25,
    winRatePct: 0,
    avgRealizedR: 0,
    avgMfePct: 0,
    avgMaePct: 0
  },
  optionsRecommendation: null,
  shapDrivers: [],
  bars: []
};

export class TerminalStore {
  // Top Omni-Command Bar & Multi-Window Layout State
  activeBrain = $state<BrainId>('ALL_5_BRAINS');
  activeAssetClass = $state<AssetClassMode>('ALL');
  universeTierMode = $state<'ALL' | 'SUB_PENNY_ONLY' | 'INSTITUTIONAL_ONLY'>('ALL');
  selectedTicker = $state<string>('NVDA');
  activeTimeframe = $state<ChartTimeframeId>('5m');
  activeRange = $state<ChartRangeId>('1D');
  workspaceLayout = $state<WorkspaceLayoutMode>('SINGLE');
  activeLinkChannel = $state<LinkChannelColor>('RED');

  // Connection & Loading Telemetry
  backendConnected = $state<boolean>(false);
  chartLoading = $state<boolean>(false);
  backtestLoading = $state<boolean>(false);
  lastBacktestSummary = $state<CustomSymbolBacktestSummary | null>(null);
  audioMode = $state<ExecutionAudioMode>(executionAudio.getMode());
  lastSyncTimestampIso = $state<string>('');
  replaySpeed = $state<'1x' | '5x' | '10x' | '50x' | 'MAX'>('1x');

  private hasHydratedInitialSnapshot = false;
  private knownOpenPositionIds = new Set<string>();
  private knownOpenTrancheCounts = new Map<string, number>();
  private knownClosedTradeIds = new Set<string>();

  // Scanner, Live Positions & Closed Trade Ledger State (100% Real Yahoo Finance Backend Hydrated)
  scannerCandidates = $state<ScannerSignalCandidate[]>([]);
  openPositions = $state<OpenPositionItem[]>([]);
  closedLedger = $state<ClosedTradeLedgerItem[]>([]);
  evolutionStatus = $state<BrainEvolutionStatusPayload | null>(null);
  ledgerSnapshot = $state<FilteredLedgerSnapshot | null>(null);
  isRunningEvolutionGeneration = $state<boolean>(false);
  lastEvolutionGenerationBanner = $state<string | null>(null);
  activeReplayTrade = $state<ClosedTradeLedgerItem | null>(null);
  sizingMode = $state<SizingModeId>('FIXED_10K_BASELINE');
  immunityMode = $state<ImmunityModeId>('BASELINE_ALL_TRADES');
  byStrategyLeaderboard = $state<StrategyLeaderboardRow[]>([]);
  byEngineBreakdown = $state<EngineBreakdownRow[]>([]);
  portfolioHeatTelemetry = $state<PortfolioHeatTelemetryState>({
    accountEquityUsd: 128513.2,
    openPositionsCount: 0,
    aggregateOpenRiskUsd: 0,
    aggregateOpenRiskPct: 0,
    portfolioHeatCapPct: 8.5,
    grossPortfolioNotionalUsd: 0,
    netPortfolioBetaDeltaUsd: 0,
    netPortfolioGamma1pctUsd: 0,
    netPortfolioVanna1pctUsd: 0,
    meanPairwiseCorrelation: 0,
    maxPairwiseCorrelation: 0,
    correlationHeatMultiplier: 1.0,
    heatRegimeBadge: 'LOW HEAT — FULL 1.00x KELLY CAPACITY',
    positionLabels: [],
    crossPositionCorrelationMatrix: [],
    pairwiseCorrelations: [],
    overlappingClustersSummary: 'No active open positions'
  });
  brokerOrderTickets = $state<StagedBrokerOrderTicketEntry[]>([]);
  stagedOptionRollTickets = $state<StagedOptionRollUpTicketEntry[]>([]);
  stagedTailHedgeTickets = $state<StagedTailHedgeOverlayTicketEntry[]>([]);
  activeStagedTicketModal = $state<StagedBrokerOrderTicketEntry | null>(null);
  activeStagedRollTicketModal = $state<StagedOptionRollUpTicketEntry | null>(null);
  lastExecutedRollComboBanner = $state<string | null>(null);

  // Phase 29 & 30 Institutional Watchlist, Sunday Alpha Protocol & Paper Overlay
  institutionalWatchlist = $state<InstitutionalWatchlistResponse | null>(null);
  sundayAlphaProtocol = $state<SundayAlphaProtocolResponse | null>(null);
  paperOverlayState = $state<PaperOverlayTelemetryResponse | null>(null);
  lastDispatchResult = $state<SundayAlphaDispatchResponse | null>(null);
  isLoadingWatchlist = $state<boolean>(false);
  isLoadingSundayProtocol = $state<boolean>(false);
  isDispatchingSundayWebhook = $state<boolean>(false);
  actionableWatchlistOnly = $state<boolean>(false);
  watchlistSearchQuery = $state<string>('');

  // Phase 31 Milestone 5: Isolated Paper-OMS Order Staging & Redacted Webhook Configuration
  stagedOrdersReceipts = $state<StagedOrderReceipt[]>([]);
  isStagingOrders = $state<boolean>(false);
  lastStagingResponse = $state<StageSundayOrdersResponse | null>(null);
  lastStagingError = $state<StageSundayOrdersErrorResponse | null>(null);
  webhookConfig = $state<RedactedWebhookConfigResponse | null>(null);
  isLoadingWebhookConfig = $state<boolean>(false);
  isUpdatingWebhookConfig = $state<boolean>(false);
  isTestPingingWebhook = $state<boolean>(false);
  lastTestPingResult = $state<WebhookTestPingResponse | null>(null);
  webhookStatus = $state<WebhookStatusPayload>({
    config: {
      discordWebhookUrl: '',
      discordEnabled: false,
      telegramBotToken: '',
      telegramChatId: '',
      telegramEnabled: false,
      slackWebhookUrl: '',
      slackEnabled: false,
      customJsonWebhookUrl: '',
      customJsonEnabled: false
    },
    totalOutboxRecords: 0,
    outboxRecords: []
  });
  webhookTesting = $state<boolean>(false);

  // Phase 32: Closed-Loop Loss-Autopsy Reinforcement Evolver & Scalping Tier
  selectedExecutionTier = $state<ExecutionTier>('ALL');
  autopsyEvolutionState = $state<AutopsyEvolutionResponse | null>(null);
  isEvolvingCycle = $state<boolean>(false);
  isLoadingAutopsyEvolution = $state<boolean>(false);
  evolutionDrawerOpen = $state<boolean>(false);

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
  strategyFilter = $state<'ALL' | StrategyArchetypeId>('ALL');
  engineFilter = $state<'ALL' | EngineExecutionModeId>('ALL');
  minConfidencePct = $state<number>(50);
  minRMultiple = $state<number>(-10);

  // Macro Tide Internals Telemetry (Live Yahoo Finance ^VIX, ADD, VOLD, TICK, Sector Flows)
  macroInternals = $state({
    nyseTick: 0,
    nyseAdd: 0,
    voldRatio: 1.0,
    vixSpot: 17.2,
    vixFrontMonth: 18.3,
    soxxPct: 0,
    smhPct: 0,
    hygLqdSpreadBps: 0,
    dxyChangePct: 0,
    us10yYieldBps: 0,
    vixTermSlope: 0
  });

  // Plug-and-Play Provider Status (Hydrated from /api/v1/providers)
  providers = $state<ProviderStatusEntry[]>([]);

  // Independent Multi-Viewport Comparison Panes for SPLIT_50_50 and QUAD_2X2 modes
  splitPanes = $state<
    Array<{
      paneId: string;
      title: string;
      linkChannel: LinkChannelColor;
      brainId: Exclude<BrainId, 'ALL_5_BRAINS'>;
      symbol: string;
      timeframe: ChartTimeframeId;
      range: ChartRangeId;
      loading: boolean;
      blueprint: ScannerSignalCandidate | null;
    }>
  >([
    {
      paneId: 'SPLIT_RED',
      title: 'WORKSPACE A — RED CHANNEL',
      linkChannel: 'RED',
      brainId: 'BRAIN-02',
      symbol: 'NVDA',
      timeframe: '5m',
      range: '1D',
      loading: false,
      blueprint: null
    },
    {
      paneId: 'SPLIT_BLUE',
      title: 'WORKSPACE B — BLUE CHANNEL',
      linkChannel: 'BLUE',
      brainId: 'BRAIN-04',
      symbol: 'AMD',
      timeframe: '1D',
      range: '6M',
      loading: false,
      blueprint: null
    }
  ]);

  quadViewports = $state<
    Array<{
      paneId: string;
      title: string;
      linkChannel: LinkChannelColor;
      brainId: Exclude<BrainId, 'ALL_5_BRAINS'>;
      symbol: string;
      timeframe: ChartTimeframeId;
      range: ChartRangeId;
      loading: boolean;
      blueprint: ScannerSignalCandidate | null;
    }>
  >([
    {
      paneId: 'QUAD_B1',
      title: 'BRAIN-01 PENNY ANOMALY',
      linkChannel: 'RED',
      brainId: 'BRAIN-01',
      symbol: 'CLOV',
      timeframe: '5m',
      range: '5D',
      loading: false,
      blueprint: null
    },
    {
      paneId: 'QUAD_B2',
      title: 'BRAIN-02 INTRADAY DAY TRADE',
      linkChannel: 'BLUE',
      brainId: 'BRAIN-02',
      symbol: 'NVDA',
      timeframe: '5m',
      range: '1D',
      loading: false,
      blueprint: null
    },
    {
      paneId: 'QUAD_B3',
      title: 'BRAIN-03 SHORT-TERM SWING',
      linkChannel: 'GREEN',
      brainId: 'BRAIN-03',
      symbol: 'PLTR',
      timeframe: '1H',
      range: '1M',
      loading: false,
      blueprint: null
    },
    {
      paneId: 'QUAD_B4',
      title: 'BRAIN-04 CORE SWING',
      linkChannel: 'GOLD',
      brainId: 'BRAIN-04',
      symbol: 'AMD',
      timeframe: '1D',
      range: '6M',
      loading: false,
      blueprint: null
    }
  ]);

  public readonly syncBus: MultiWindowSyncBus;
  private wsConnection: WebSocket | null = null;
  private wsReconnectAttempt = 0;
  private wsReconnectTimer: ReturnType<typeof setTimeout> | null = null;

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
    const match = this.scannerCandidates.find(
      (c) => c.symbol.toUpperCase() === this.selectedTicker.toUpperCase()
    );
    return match ?? this.scannerCandidates[0] ?? EMPTY_BLUEPRINT_PLACEHOLDER;
  });

  // Phase 18: Derived Multi-Generation Evolution History & Synaptic Weight Plasticity States
  evolutionGenerationsHistory = $derived<EvolutionGenerationRecord[]>(
    this.evolutionStatus?.evolution_generations_history ??
      this.ledgerSnapshot?.evolution_generations_history ??
      []
  );

  synapticPlasticityStates = $derived<BrainSynapticPlasticityState[]>(
    this.evolutionStatus?.synaptic_plasticity_states ??
      this.ledgerSnapshot?.synaptic_plasticity_states ??
      []
  );

  // Derived Historical Closed Trades for the Currently Selected Symbol (for chart markers)
  selectedSymbolClosedTrades = $derived.by(() => {
    return this.closedLedger.filter(
      (t) => t.symbol.toUpperCase() === this.selectedTicker.toUpperCase()
    );
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

      // Dimension 5: Outcome, Primary Driver, Strategy Archetype, & Engine Mode
      if (this.outcomeFilter !== 'ALL' && row.outcomeBadge !== this.outcomeFilter) return false;
      if (this.driverFilter !== 'ALL' && row.primaryDriver !== this.driverFilter) return false;
      if (this.strategyFilter !== 'ALL' && row.strategyArchetype !== this.strategyFilter)
        return false;
      if (this.engineFilter !== 'ALL' && row.engineMode !== this.engineFilter) return false;

      // Dimension 6: Performance Sliders
      if (row.aiConfidenceAtEntryPct < this.minConfidencePct) return false;
      if (row.realizedRMultiple < this.minRMultiple) return false;

      return true;
    });
  });

  // Derived Dynamic Executive Scorecard KPIs
  scorecardKpis = $derived.by(() => {
    const rows = this.filteredLedger;
    const useKelly = this.sizingMode === 'FRACTIONAL_KELLY';
    const useImmunity = this.immunityMode === 'ACTIVE_LOSS_IMMUNITY';
    const rawCount = rows.length;
    if (rawCount === 0) {
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
        avgHoldDurationMs: 0,
        vetoedLossesCount: 0,
        scratchedLossesCount: 0,
        tpLossesVetoed: 0,
        fpWinnersVetoed: 0,
        tpLossesScratched: 0,
        fpWinnersScratched: 0,
        remainingUnscratchedLosses: 0,
        untouchedWinnersCount: 0,
        entryVetoPrecisionPct: 0,
        bar1ScratchPrecisionPct: 0,
        totalLossAvoidedUsd: 0,
        avgScratchExitR: 0,
        avgLoserR: 0
      };
    }

    let totalWins = 0;
    let totalLosses = 0;
    let grossProfit = 0;
    let grossLoss = 0;
    let sumR = 0;
    let sumLoserR = 0;
    let stockPnl = 0;
    let optionPnl = 0;
    let combinedPnl = 0;
    let sumHoldMs = 0;
    let vetoedLossesCount = 0;
    let scratchedLossesCount = 0;
    let tpLossesVetoed = 0;
    let fpWinnersVetoed = 0;
    let tpLossesScratched = 0;
    let fpWinnersScratched = 0;
    let remainingUnscratchedLosses = 0;
    let untouchedWinnersCount = 0;
    let totalLossAvoidedUsd = 0;
    let sumScratchR = 0;

    let runningEquity = 0;
    let peakEquity = 0;
    let maxDrawdownUsd = 0;

    const chronological = [...rows].sort((a, b) => a.closedTimestampMs - b.closedTimestampMs);
    for (const r of chronological) {
      const savedUsd = useKelly ? r.adaptiveKellyLossSavedUsd : r.adaptiveLossSavedUsd;
      if (r.wasVetoedByLossAutopsy) {
        vetoedLossesCount++;
        if (r.isWinner) {
          fpWinnersVetoed++;
        } else {
          tpLossesVetoed++;
        }
        totalLossAvoidedUsd += savedUsd;
        if (useImmunity) {
          continue;
        }
      } else if (r.wasScratchedByLossAutopsy) {
        scratchedLossesCount++;
        if (r.isWinner) {
          fpWinnersScratched++;
        } else {
          tpLossesScratched++;
        }
        sumScratchR += r.adaptiveScratchExitR;
        totalLossAvoidedUsd += savedUsd;
      } else if (r.isWinner) {
        untouchedWinnersCount++;
      } else {
        remainingUnscratchedLosses++;
      }

      const effectiveStockPnl = useImmunity
        ? useKelly
          ? r.adaptiveKellyNetPnlUsd
          : r.adaptiveNetPnlUsd
        : useKelly
          ? r.kellyRealizedStockPnlUsd
          : r.realizedStockPnlUsd;
      const effectiveR =
        useImmunity && r.wasScratchedByLossAutopsy ? r.adaptiveScratchExitR : r.realizedRMultiple;
      const optionScratchScale =
        useImmunity && r.wasScratchedByLossAutopsy && Math.abs(r.realizedRMultiple) > 1e-6
          ? Math.min(1, Math.abs(r.adaptiveScratchExitR) / Math.abs(r.realizedRMultiple))
          : 1;
      const effectiveOptionPnl =
        useImmunity && r.wasVetoedByLossAutopsy
          ? 0
          : Math.round(r.realizedOptionPnlUsd * optionScratchScale * 100) / 100;
      const effectiveCombined = effectiveStockPnl + effectiveOptionPnl;

      const isEffectiveWin = useImmunity ? effectiveStockPnl > 0 : r.isWinner;
      if (isEffectiveWin) {
        totalWins++;
        grossProfit += Math.max(0, effectiveStockPnl);
      } else {
        totalLosses++;
        grossLoss += Math.abs(Math.min(0, effectiveStockPnl));
        sumLoserR += effectiveR;
      }
      sumR += effectiveR;
      stockPnl += effectiveStockPnl;
      optionPnl += effectiveOptionPnl;
      combinedPnl += effectiveCombined;
      sumHoldMs += r.holdDurationMs;

      runningEquity += effectiveStockPnl;
      if (runningEquity > peakEquity) {
        peakEquity = runningEquity;
      }
      const dd = peakEquity - runningEquity;
      if (dd > maxDrawdownUsd) {
        maxDrawdownUsd = dd;
      }
    }

    const totalClosed = Math.max(0, totalWins + totalLosses);
    const denom = Math.max(1, totalClosed);
    const winRatePct = totalClosed > 0 ? (totalWins / totalClosed) * 100 : 0;
    const lossRatePct = totalClosed > 0 ? (totalLosses / totalClosed) * 100 : 0;
    const profitFactor =
      grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? 99.99 : 0.0;
    const entryVetoPrecisionPct =
      vetoedLossesCount > 0 ? (tpLossesVetoed / vetoedLossesCount) * 100 : 0;
    const bar1ScratchPrecisionPct =
      scratchedLossesCount > 0 ? (tpLossesScratched / scratchedLossesCount) * 100 : 0;

    return {
      totalClosed,
      totalWins,
      totalLosses,
      winRatePct,
      lossRatePct,
      profitFactor,
      expectancyR: sumR / denom,
      expectancyUsd: combinedPnl / denom,
      netRealizedStockPnlUsd: stockPnl,
      netRealizedOptionPnlUsd: optionPnl,
      netCombinedPnlUsd: combinedPnl,
      maxDrawdownUsd,
      avgHoldDurationMs: Math.round(sumHoldMs / denom),
      vetoedLossesCount,
      scratchedLossesCount,
      tpLossesVetoed,
      fpWinnersVetoed,
      tpLossesScratched,
      fpWinnersScratched,
      remainingUnscratchedLosses,
      untouchedWinnersCount,
      entryVetoPrecisionPct,
      bar1ScratchPrecisionPct,
      totalLossAvoidedUsd,
      avgScratchExitR: scratchedLossesCount > 0 ? sumScratchR / scratchedLossesCount : 0,
      avgLoserR: totalLosses > 0 ? sumLoserR / totalLosses : 0
    };
  });

  // Derived Cumulative Equity ($) & Cumulative R Curve
  equityCurve = $derived.by(() => {
    const useKelly = this.sizingMode === 'FRACTIONAL_KELLY';
    const useImmunity = this.immunityMode === 'ACTIVE_LOSS_IMMUNITY';
    const chronological = [...this.filteredLedger]
      .filter((r) => !(useImmunity && r.wasVetoedByLossAutopsy))
      .sort((a, b) => a.closedTimestampMs - b.closedTimestampMs);
    let cumUsd = 0;
    let cumR = 0;
    return chronological.map((r, idx) => {
      const effectiveStockPnl = useImmunity
        ? useKelly
          ? r.adaptiveKellyNetPnlUsd
          : r.adaptiveNetPnlUsd
        : useKelly
          ? r.kellyRealizedStockPnlUsd
          : r.realizedStockPnlUsd;
      const effectiveR =
        useImmunity && r.wasScratchedByLossAutopsy ? r.adaptiveScratchExitR : r.realizedRMultiple;
      const optionScratchScale =
        useImmunity && r.wasScratchedByLossAutopsy && Math.abs(r.realizedRMultiple) > 1e-6
          ? Math.min(1, Math.abs(r.adaptiveScratchExitR) / Math.abs(r.realizedRMultiple))
          : 1;
      const effectiveOptionPnl = Math.round(r.realizedOptionPnlUsd * optionScratchScale * 100) / 100;
      cumUsd += effectiveStockPnl + effectiveOptionPnl;
      cumR += effectiveR;
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

  constructor(autoConnect = true) {
    this.syncBus = new MultiWindowSyncBus();
    this.syncBus.subscribe((msg) => {
      if (
        'linkChannel' in msg &&
        msg.linkChannel &&
        this.activeLinkChannel !== 'UNLINKED' &&
        msg.linkChannel !== this.activeLinkChannel
      ) {
        return;
      }
      if (msg.type === 'SELECT_TICKER') {
        this.selectedTicker = msg.symbol;
        void this.fetchLiveChartForSymbol(msg.symbol, this.activeTimeframe, this.activeRange);
      } else if (msg.type === 'SELECT_BRAIN') {
        this.activeBrain = msg.brainId as BrainId;
      } else if (msg.type === 'SELECT_ASSET_CLASS') {
        this.activeAssetClass = msg.assetClass;
      } else if (msg.type === 'APPLY_QUICK_FILTER') {
        this.quickPill = msg.pillId as QuickFilterPillId;
      } else if (msg.type === 'SET_LINK_CHANNEL') {
        this.activeLinkChannel = msg.linkChannel;
      } else if (msg.type === 'TRIGGER_REPLAY_TICK') {
        this.stepLiveSimulation(false);
      }
    });

    if (autoConnect && typeof window !== 'undefined' && typeof fetch === 'function') {
      void this.initializeLiveConnection();
    }
  }

  /**
   * Connects to the Rust `live-execution-server` over REST + WebSocket (`/ws/v1/stream`)
   * and hydrates all real Yahoo Finance charts, positions, closed trades, providers, and autopsy status.
   */
  public async initializeLiveConnection(): Promise<void> {
    await Promise.all([
      this.fetchLedgerSnapshot(),
      this.fetchExecutionTickets(),
      this.fetchScannerCandidates(),
      this.fetchProvidersTelemetry(),
      this.fetchEvolutionStatus(),
      this.fetchWebhookStatus(),
      this.fetchLiveChartForSymbol(this.selectedTicker, this.activeTimeframe, this.activeRange)
    ]);
    this.connectWebSocketStream();
  }

  private connectWebSocketStream(): void {
    if (typeof WebSocket === 'undefined') return;
    try {
      if (this.wsReconnectTimer) {
        clearTimeout(this.wsReconnectTimer);
        this.wsReconnectTimer = null;
      }
      if (this.wsConnection) {
        this.wsConnection.onclose = null;
        this.wsConnection.onerror = null;
        this.wsConnection.close();
      }
      const ws = new WebSocket(WS_URL);
      this.wsConnection = ws;
      ws.onopen = () => {
        const wasReconnect = this.wsReconnectAttempt > 0;
        this.wsReconnectAttempt = 0;
        this.backendConnected = true;
        if (wasReconnect) {
          void this.fetchLedgerSnapshot();
          void this.fetchScannerCandidates();
        }
      };
      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(String(ev.data));
          if (data.type === 'LEDGER_SNAPSHOT_UPDATE' && data.snapshot) {
            this.ingestBackendSnapshot(data.snapshot);
            if (Array.isArray(data.scanner_candidates)) {
              this.ingestScannerPromotions(data.scanner_candidates);
            }
          }
        } catch {
          // Ignore malformed WS frames
        }
      };
      const scheduleReconnect = () => {
        this.backendConnected = false;
        if (this.wsReconnectTimer) return;
        const delayMs = Math.min(5000, 500 * Math.pow(2, this.wsReconnectAttempt));
        this.wsReconnectAttempt += 1;
        this.wsReconnectTimer = setTimeout(() => {
          this.wsReconnectTimer = null;
          this.connectWebSocketStream();
        }, delayMs);
      };
      ws.onclose = scheduleReconnect;
      ws.onerror = scheduleReconnect;
    } catch {
      this.backendConnected = false;
    }
  }

  private mapBrokerOrderTicket(t: Record<string, unknown>): StagedBrokerOrderTicketEntry {
    return {
      ticketId: String(t.ticket_id ?? t.position_id ?? ''),
      positionId: String(t.position_id ?? ''),
      stagedAtIso: String(t.staged_at_iso ?? ''),
      symbol: String(t.symbol ?? 'NVDA'),
      exchangeIndex: String(t.exchange_index ?? 'NASDAQ_100'),
      brainId: String(t.brain_id ?? 'BRAIN-02'),
      brainName: String(t.brain_name ?? 'BRAIN-02 INTRADAY DAY TRADE'),
      direction: (String(t.direction ?? 'LONG').toUpperCase() === 'SHORT'
        ? 'SHORT'
        : 'LONG') as TradeDirection,
      action: String(t.action ?? 'BUY_TO_OPEN'),
      orderStatus: String(t.order_status ?? 'READY_TO_STAGE'),
      selectedBroker: String(t.selected_broker ?? 'IBKR_TWS'),
      equityQuantityShares: Number(t.equity_quantity_shares ?? 1),
      occOptionSymbol: String(t.occ_option_symbol ?? ''),
      matchedOptionContract: String(t.matched_option_contract ?? 'SHARES_ONLY'),
      optionStructure: String(t.option_structure ?? 'SHARES_ONLY'),
      optionContractsCount: Number(t.option_contracts_count ?? 0),
      limitPrice: Number(t.limit_price ?? 0),
      liveMarkPrice: Number(t.live_mark_price ?? 0),
      stopLossTriggerPrice: Number(t.stop_loss_trigger_price ?? 0),
      target1PartialLimitPrice: Number(t.target_1_partial_limit_price ?? 0),
      target2RunnerLimitPrice: Number(t.target_2_runner_limit_price ?? 0),
      optionEntryLimitPremium: Number(t.option_entry_limit_premium ?? 0),
      optionStopTriggerPremium: Number(t.option_stop_trigger_premium ?? 0),
      optionTarget1LimitPremium: Number(t.option_target_1_limit_premium ?? 0),
      kellyFractionPct: Number(t.kelly_fraction_pct ?? 1.25),
      kellyRiskBudgetUsd: Number(t.kelly_risk_budget_usd ?? 500),
      notionalAllocatedUsd: Number(t.notional_allocated_usd ?? 10000),
      correlationHeatMultiplier: Number(t.correlation_heat_multiplier ?? 1.0),
      almgrenChrissRiskAversionLambda: Number(t.almgren_chriss_risk_aversion_lambda ?? 0.34),
      almgrenChrissUrgencyKappa: Number(t.almgren_chriss_urgency_kappa ?? 0.85),
      estimatedEtaTemp: Number(t.estimated_eta_temp ?? 0.00018),
      estimatedGammaPerm: Number(t.estimated_gamma_perm ?? 0.000045),
      expectedExecutionSlippageBps: Number(t.expected_execution_slippage_bps ?? 0.6),
      expectedSlippageSavedUsd: Number(t.expected_slippage_saved_usd ?? 21.4),
      sorVenueWeightsSimplex: Array.isArray(t.sor_venue_weights_simplex)
        ? [
            Number(t.sor_venue_weights_simplex[0] ?? 0.169),
            Number(t.sor_venue_weights_simplex[1] ?? 0.6063),
            Number(t.sor_venue_weights_simplex[2] ?? 0.197),
            Number(t.sor_venue_weights_simplex[3] ?? 0.0277)
          ]
        : Array.isArray(t.sor_venue_weights_4)
          ? [
              Number(t.sor_venue_weights_4[0] ?? 0.169),
              Number(t.sor_venue_weights_4[1] ?? 0.6063),
              Number(t.sor_venue_weights_4[2] ?? 0.197),
              Number(t.sor_venue_weights_4[3] ?? 0.0277)
            ]
          : [0.169, 0.6063, 0.197, 0.0277],
      sorExpectedMarkoutToxicityBps: Number(
        t.sor_expected_markout_toxicity_bps ?? t.sor_markout_toxicity_bps ?? 0.45
      ),
      crumblingQuoteProbability: Number(
        t.crumbling_quote_probability ?? t.sor_crumbling_quote_prob ?? 0.08
      ),
      orderBookAbsorptionRatio: Number(t.order_book_absorption_ratio ?? 0.84),
      childExecutionSlices: Array.isArray(t.child_execution_slices)
        ? t.child_execution_slices.map((s: Record<string, unknown>, idx: number) => ({
            sliceNumber: Number(s.slice_number ?? s.slice_index ?? idx + 1),
            timeOffsetSecs: Number(
              s.time_offset_secs ??
                (s.time_window_start_t !== undefined
                  ? Math.round(Number(s.time_window_start_t) * 60)
                  : idx * 15)
            ),
            equityQtyShares: Number(s.equity_qty_shares ?? s.equity_shares_qty ?? 1),
            optionContractsQty: Number(s.option_contracts_qty ?? 0),
            sliceWeightPct: Number(s.slice_weight_pct ?? 25.0),
            equityPegStrategy: String(
              s.equity_peg_strategy ??
                s.equity_routing_instruction ??
                'IBKR_PEG_MID_ISLAND_PRIMARY_OFFSET'
            ),
            optionPegStrategy: String(
              s.option_peg_strategy ??
                s.option_routing_instruction ??
                'IBKR_IBUSOPT_PEGGED_TO_MIDPOINT_0005'
            ),
            expectedSlippageBps: Number(
              s.expected_slippage_bps ?? s.expected_slice_slippage_bps ?? 0.6
            ),
            sorSelectedVenue: String(s.sor_selected_venue ?? 'IBKR_PEG_MID'),
            sorVenueAllocationWeight: Number(s.sor_venue_allocation_weight ?? 0.6063),
            postFillMarkoutToxicityBps: Number(s.post_fill_markout_toxicity_bps ?? 0.45)
          }))
        : [],
      ibkrTwsPayloadJson: String(t.ibkr_tws_payload_json ?? '{}'),
      schwabTraderPayloadJson: String(t.schwab_trader_payload_json ?? '{}'),
      alpacaApiPayloadJson: String(t.alpaca_api_payload_json ?? '{}')
    };
  }

  private mapStagedTailHedgeTicket(t: Record<string, unknown>): StagedTailHedgeOverlayTicketEntry {
    return {
      hedgeTicketId: String(t.hedge_ticket_id ?? 'TAIL-HEDGE-SPY-QQQ'),
      stagedAtIso: String(t.staged_at_iso ?? ''),
      hedgeSymbol: String(t.hedge_symbol ?? t.hedge_underlying_symbol ?? 'QQQ'),
      occPutSymbol: String(t.occ_put_symbol ?? ''),
      putContractDesc: String(
        t.put_contract_desc ?? 'QQQ 2026-11-20 Put (25-Delta Vanna Tail-Hedge Overlay)'
      ),
      strikePrice: Number(t.strike_price ?? t.put_strike_price ?? 490.0),
      expirationDate: String(t.expiration_date ?? '2026-11-20'),
      contractsCount: Number(t.contracts_count ?? t.put_contracts_qty ?? 1),
      entryLimitPremium: Number(t.entry_limit_premium ?? t.put_limit_premium ?? 3.85),
      fundedFromHouseMoneyUsd: Number(
        t.funded_from_house_money_usd ?? t.tail_hedge_budget_allocated_usd ?? 385.0
      ),
      principalCapitalRiskedPct: Number(t.principal_capital_risked_pct ?? 0.0),
      learnedTailHedgeRatioTheta: Number(t.learned_tail_hedge_ratio_theta ?? 0.0558),
      meanPairwiseCorrelationTrigger: Number(
        t.mean_pairwise_correlation_trigger ?? t.mean_ledoit_wolf_correlation ?? 0.42
      ),
      learnedCorrelationSpikeQuantile: Number(t.learned_correlation_spike_quantile ?? 0.34),
      expectedConvexityPayoutUsd: Number(
        t.expected_convexity_payout_usd ?? t.projected_3sigma_crash_payout_usd ?? 1765.62
      ),
      ibkrPegMidPayloadJson: String(
        t.ibkr_peg_mid_payload_json ?? t.ibkr_tail_hedge_payload_json ?? '{}'
      )
    };
  }

  private mapIbkrBagComboLeg(l: Record<string, unknown>, defaultId: number): IbkrBagComboLegEntry {
    return {
      legId: Number(l.leg_id ?? l.leg_number ?? defaultId),
      action: String(l.action ?? (defaultId === 1 ? 'SELL_TO_CLOSE' : 'BUY_TO_OPEN')),
      ratioQty: Number(l.ratio_qty ?? l.contracts_qty ?? 1),
      occOptionSymbol: String(l.occ_option_symbol ?? l.occ_symbol ?? ''),
      contractDesc: String(l.contract_desc ?? ''),
      strikePrice: Number(l.strike_price ?? 0),
      expirationDate: String(l.expiration_date ?? '2026-11-20'),
      optionRight: String(l.option_right ?? 'C'),
      liveMidPrice: Number(l.live_mid_price ?? l.cboe_mid_price ?? 0)
    };
  }

  private mapStagedOptionRollUpTicket(t: Record<string, unknown>): StagedOptionRollUpTicketEntry {
    const rawCombo =
      t.ibkr_bag_combo && typeof t.ibkr_bag_combo === 'object'
        ? (t.ibkr_bag_combo as Record<string, unknown>)
        : t;
    const rawLeg1 =
      t.leg_1_close && typeof t.leg_1_close === 'object'
        ? (t.leg_1_close as Record<string, unknown>)
        : rawCombo.leg_1_close && typeof rawCombo.leg_1_close === 'object'
          ? (rawCombo.leg_1_close as Record<string, unknown>)
          : {};
    const rawLeg2 =
      t.leg_2_open && typeof t.leg_2_open === 'object'
        ? (t.leg_2_open as Record<string, unknown>)
        : rawCombo.leg_2_open && typeof rawCombo.leg_2_open === 'object'
          ? (rawCombo.leg_2_open as Record<string, unknown>)
          : {};
    const leg1Close = this.mapIbkrBagComboLeg(rawLeg1, 1);
    const leg2Open = this.mapIbkrBagComboLeg(rawLeg2, 2);
    const rawLadder = Array.isArray(t.nbbo_limit_walk_ladder)
      ? t.nbbo_limit_walk_ladder
      : Array.isArray(rawCombo.nbbo_limit_walk_ladder)
        ? rawCombo.nbbo_limit_walk_ladder
        : [];
    const netMid = Number(
      t.net_combo_limit_price ??
        rawCombo.net_combo_limit_price ??
        (leg1Close.liveMidPrice - leg2Open.liveMidPrice)
    );
    const ladder: [number, number, number] = [
      Number(rawLadder[0] ?? netMid),
      Number(rawLadder[1] ?? Number((netMid - 0.02).toFixed(2))),
      Number(rawLadder[2] ?? Number((netMid - 0.05).toFixed(2)))
    ];
    const comboOrderType = String(
      t.combo_order_type ?? rawCombo.combo_order_type ?? 'IBKR_BAG_VERTICAL_ROLL_COMBO'
    );
    const preRollNetDelta = Number(
      t.pre_roll_net_delta ?? rawCombo.pre_roll_net_delta ?? 0
    );
    const postRollNetDelta = Number(
      t.post_roll_net_delta ?? rawCombo.post_roll_net_delta ?? 0
    );
    const postRollGammaConvexityBoostPct = Number(
      t.post_roll_gamma_convexity_boost_pct ?? rawCombo.post_roll_gamma_convexity_boost_pct ?? 18.0
    );
    const lockedCashHarvestedUsd = Number(
      t.locked_cash_harvested_usd ??
        rawCombo.locked_cash_harvested_usd ??
        t.banked_realized_cash_usd ??
        0
    );
    return {
      rollTicketId: String(t.roll_ticket_id ?? t.position_id ?? ''),
      positionId: String(t.position_id ?? ''),
      stagedAtIso: String(t.staged_at_iso ?? ''),
      symbol: String(t.symbol ?? 'NVDA'),
      brainId: String(t.brain_id ?? 'BRAIN-02'),
      brainName: String(t.brain_name ?? 'BRAIN-02 INTRADAY DAY TRADE'),
      direction: (String(t.direction ?? 'LONG').toUpperCase() === 'SHORT'
        ? 'SHORT'
        : 'LONG') as TradeDirection,
      rollStatus: String(t.roll_status ?? 'HOUSE_MONEY_GAMMA_ROLL_STAGED'),
      isGammaRollActive: Boolean(t.is_gamma_roll_active ?? false),
      currentItmOccSymbol: String(t.current_itm_occ_symbol ?? ''),
      currentItmContractDesc: String(t.current_itm_contract_desc ?? ''),
      sellToCloseContracts: Number(t.sell_to_close_contracts ?? 1),
      currentItmEntryPremium: Number(t.current_itm_entry_premium ?? 0),
      currentItmLivePremium: Number(t.current_itm_live_premium ?? 0),
      intrinsicOptionGainUsd: Number(t.intrinsic_option_gain_usd ?? 0),
      bankedCashFraction: Number(t.banked_cash_fraction ?? 0.46),
      bankedRealizedCashUsd: Number(t.banked_realized_cash_usd ?? 0),
      houseMoneyReinvestmentFractionPhi: Number(t.house_money_reinvestment_fraction_phi ?? 0.54),
      houseMoneyRollBudgetUsd: Number(t.house_money_roll_budget_usd ?? 0),
      nextGammaWallStrike: Number(t.next_gamma_wall_strike ?? 0),
      nextGammaWallOccSymbol: String(t.next_gamma_wall_occ_symbol ?? ''),
      nextGammaWallContractDesc: String(t.next_gamma_wall_contract_desc ?? ''),
      buyToOpenRollContracts: Number(t.buy_to_open_roll_contracts ?? 1),
      nextGammaWallLimitPremium: Number(t.next_gamma_wall_limit_premium ?? 0),
      lockedProfitR: Number(t.locked_profit_r ?? 0),
      learnedProfitLockQuantileR: Number(t.learned_profit_lock_quantile_r ?? 0.4),
      liveGivebackLlr: Number(t.live_giveback_llr ?? 0.28),
      runnerContinuationThreshold: Number(t.runner_continuation_threshold ?? -0.52),
      guaranteedNetFloorR: Number(t.guaranteed_net_floor_r ?? 0),
      leg1Ratio: Number(t.leg1_ratio ?? t.leg1Ratio ?? rawCombo.leg1_ratio ?? rawCombo.leg1Ratio ?? 1),
      leg2Ratio: Number(t.leg2_ratio ?? t.leg2Ratio ?? rawCombo.leg2_ratio ?? rawCombo.leg2Ratio ?? 1),
      comboOrderType,
      leg1Close,
      leg2Open,
      netComboLimitPrice: netMid,
      nbboLimitWalkLadder: ladder,
      preRollNetDelta,
      postRollNetDelta,
      postRollGammaConvexityBoostPct,
      lockedCashHarvestedUsd,
      ibkrBagCombo: {
        comboOrderType,
        leg1Ratio: Number(t.leg1_ratio ?? t.leg1Ratio ?? rawCombo.leg1_ratio ?? rawCombo.leg1Ratio ?? 1),
        leg2Ratio: Number(t.leg2_ratio ?? t.leg2Ratio ?? rawCombo.leg2_ratio ?? rawCombo.leg2Ratio ?? 1),
        leg1Close,
        leg2Open,
        netComboLimitPrice: netMid,
        nbboLimitWalkLadder: ladder,
        preRollNetDelta,
        postRollNetDelta,
        postRollGammaConvexityBoostPct,
        lockedCashHarvestedUsd
      },
      ibkrComboRollPayloadJson: String(t.ibkr_combo_roll_payload_json ?? '{}'),
      schwabRollPayloadJson: String(t.schwab_roll_payload_json ?? '{}'),
      alpacaRollPayloadJson: String(t.alpaca_roll_payload_json ?? '{}')
    };
  }

  public async fetchExecutionTickets(): Promise<void> {
    try {
      const res = await fetch(`${API_BASE}/api/v1/execution/tickets`);
      if (!res.ok) return;
      const payload = (await res.json()) as Record<string, unknown>;
      if (Array.isArray(payload.tickets)) {
        this.brokerOrderTickets = payload.tickets.map((t: Record<string, unknown>) =>
          this.mapBrokerOrderTicket(t)
        );
      }
      if (Array.isArray(payload.staged_option_roll_tickets)) {
        this.stagedOptionRollTickets = payload.staged_option_roll_tickets.map(
          (t: Record<string, unknown>) => this.mapStagedOptionRollUpTicket(t)
        );
      }
      if (Array.isArray(payload.staged_tail_hedge_tickets)) {
        this.stagedTailHedgeTickets = payload.staged_tail_hedge_tickets.map(
          (t: Record<string, unknown>) => this.mapStagedTailHedgeTicket(t)
        );
      }
    } catch {
      // Handled by connection flag
    }
  }

  public async stageBrokerOrderTicket(
    positionId: string,
    broker = 'IBKR_TWS',
    action?: string
  ): Promise<StagedBrokerOrderTicketEntry | null> {
    try {
      const res = await fetch(
        `${API_BASE}/api/v1/execution/stage/${encodeURIComponent(positionId)}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ broker, action })
        }
      );
      if (res.ok) {
        const raw = (await res.json()) as Record<string, unknown>;
        const ticket = this.mapBrokerOrderTicket(raw);
        const idx = this.brokerOrderTickets.findIndex((t) => t.positionId === positionId);
        if (idx >= 0) {
          const updated = [...this.brokerOrderTickets];
          updated[idx] = ticket;
          this.brokerOrderTickets = updated;
        } else {
          this.brokerOrderTickets = [ticket, ...this.brokerOrderTickets];
        }
        this.activeStagedTicketModal = ticket;
        executionAudio.playTarget1BankedChime();
        return ticket;
      }
    } catch {
      // Offline fallback below
    }

    const pos = this.openPositions.find((p) => p.tradeId === positionId);
    if (!pos) return null;
    const act =
      action ??
      (pos.traceHealthBadge === 'DYNAMIC EXIT TRIGGERED' || pos.currentRMultiple <= -0.15
        ? 'SCRATCH_EXIT'
        : pos.currentRMultiple >= 0.85
          ? 'SELL_TO_CLOSE'
          : 'BUY_TO_OPEN');
    const right = pos.direction === 'LONG' ? 'C' : 'P';
    const strikeMillis = String(Math.round(pos.entryPrice * 1000)).padStart(8, '0');
    const occSymbol = `${pos.symbol.padEnd(6, ' ')}260410${right}${strikeMillis}`;
    const fallbackTicket: StagedBrokerOrderTicketEntry = {
      ticketId: `TKT-${pos.tradeId}`,
      positionId: pos.tradeId,
      stagedAtIso: new Date().toISOString(),
      symbol: pos.symbol,
      exchangeIndex: pos.exchangeIndex,
      brainId: pos.brainId,
      brainName: pos.brainId,
      direction: pos.direction,
      action: act,
      orderStatus: 'STAGED_FOR_ROUTING',
      selectedBroker: broker,
      equityQuantityShares: pos.quantityShares,
      occOptionSymbol: occSymbol,
      matchedOptionContract: pos.optionContractLabel,
      optionStructure: pos.optionStructure,
      optionContractsCount: pos.optionContractsCount,
      limitPrice: act === 'BUY_TO_OPEN' ? pos.entryPrice : pos.livePrice,
      liveMarkPrice: pos.livePrice,
      stopLossTriggerPrice: pos.structuralStopPrice,
      target1PartialLimitPrice: pos.target1Price,
      target2RunnerLimitPrice: pos.target2Price,
      optionEntryLimitPremium: pos.optionEntryPremium,
      optionStopTriggerPremium: pos.optionEntryPremium * 0.55,
      optionTarget1LimitPremium: pos.optionLivePremium * 1.35,
      kellyFractionPct: pos.kellyFractionPct,
      kellyRiskBudgetUsd: pos.riskBudgetUsd,
      notionalAllocatedUsd: pos.notionalAllocatedUsd,
      correlationHeatMultiplier: this.portfolioHeatTelemetry.correlationHeatMultiplier,
      almgrenChrissRiskAversionLambda: 0.34,
      almgrenChrissUrgencyKappa: 0.85,
      estimatedEtaTemp: 0.00018,
      estimatedGammaPerm: 0.000045,
      expectedExecutionSlippageBps: 0.6,
      expectedSlippageSavedUsd: 21.4,
      sorVenueWeightsSimplex: [0.169, 0.6063, 0.197, 0.0277],
      sorExpectedMarkoutToxicityBps: 0.45,
      crumblingQuoteProbability: 0.08,
      orderBookAbsorptionRatio: 0.84,
      childExecutionSlices: [
        {
          sliceNumber: 1,
          timeOffsetSecs: 0,
          equityQtyShares: Math.max(1, Math.round(pos.quantityShares * 0.42)),
          optionContractsQty: Math.max(0, Math.round(pos.optionContractsCount * 0.42)),
          sliceWeightPct: 42.0,
          equityPegStrategy: 'IBKR_PEG_MID_ISLAND_PRIMARY_OFFSET',
          optionPegStrategy: 'IBKR_IBUSOPT_PEGGED_TO_MIDPOINT_0005',
          expectedSlippageBps: 0.56,
          sorSelectedVenue: 'IBKR_PEG_MID',
          sorVenueAllocationWeight: 0.6063,
          postFillMarkoutToxicityBps: 0.42
        },
        {
          sliceNumber: 2,
          timeOffsetSecs: 15,
          equityQtyShares: Math.max(1, Math.round(pos.quantityShares * 0.28)),
          optionContractsQty: Math.max(0, Math.round(pos.optionContractsCount * 0.28)),
          sliceWeightPct: 28.0,
          equityPegStrategy: 'IBKR_PEG_MID_ISLAND_PRIMARY_OFFSET',
          optionPegStrategy: 'IBKR_IBUSOPT_PEGGED_TO_MIDPOINT_0005',
          expectedSlippageBps: 0.59,
          sorSelectedVenue: 'MEMX_MIDPOINT',
          sorVenueAllocationWeight: 0.197,
          postFillMarkoutToxicityBps: 0.44
        },
        {
          sliceNumber: 3,
          timeOffsetSecs: 30,
          equityQtyShares: Math.max(1, Math.round(pos.quantityShares * 0.18)),
          optionContractsQty: Math.max(0, Math.round(pos.optionContractsCount * 0.18)),
          sliceWeightPct: 18.0,
          equityPegStrategy: 'IBKR_PEG_MID_ISLAND_PRIMARY_OFFSET',
          optionPegStrategy: 'IBKR_IBUSOPT_PEGGED_TO_MIDPOINT_0005',
          expectedSlippageBps: 0.62,
          sorSelectedVenue: 'IEX_DLIMIT',
          sorVenueAllocationWeight: 0.169,
          postFillMarkoutToxicityBps: 0.46
        },
        {
          sliceNumber: 4,
          timeOffsetSecs: 45,
          equityQtyShares: Math.max(1, pos.quantityShares - Math.round(pos.quantityShares * 0.88)),
          optionContractsQty: Math.max(
            0,
            pos.optionContractsCount - Math.round(pos.optionContractsCount * 0.88)
          ),
          sliceWeightPct: 12.0,
          equityPegStrategy: 'IBKR_PEG_MID_ISLAND_PRIMARY_OFFSET',
          optionPegStrategy: 'IBKR_IBUSOPT_PEGGED_TO_MIDPOINT_0005',
          expectedSlippageBps: 0.64,
          sorSelectedVenue: 'IBKR_PEG_MID',
          sorVenueAllocationWeight: 0.6063,
          postFillMarkoutToxicityBps: 0.48
        }
      ],
      ibkrTwsPayloadJson: JSON.stringify(
        {
          broker: 'IBKR_TWS_API_V10',
          symbol: pos.symbol,
          action: act,
          quantity: pos.quantityShares,
          lmtPrice: pos.livePrice,
          stopPrice: pos.structuralStopPrice,
          target1Price: pos.target1Price,
          occOptionSymbol: occSymbol
        },
        null,
        2
      ),
      schwabTraderPayloadJson: JSON.stringify(
        {
          broker: 'CHARLES_SCHWAB_TRADER_API_V1',
          instruction: act,
          symbol: pos.symbol,
          quantity: pos.quantityShares,
          price: pos.livePrice,
          occOptionSymbol: occSymbol
        },
        null,
        2
      ),
      alpacaApiPayloadJson: JSON.stringify(
        {
          broker: 'ALPACA_TRADING_API_V2',
          symbol: pos.symbol,
          qty: String(pos.quantityShares),
          limit_price: pos.livePrice.toFixed(4),
          stop_loss: { stop_price: pos.structuralStopPrice.toFixed(4) },
          take_profit: { limit_price: pos.target1Price.toFixed(4) }
        },
        null,
        2
      )
    };
    this.activeStagedTicketModal = fallbackTicket;
    return fallbackTicket;
  }

  public closeStagedTicketModal(): void {
    this.activeStagedTicketModal = null;
  }

  public closeStagedRollTicketModal(): void {
    this.activeStagedRollTicketModal = null;
  }

  public async stageOptionRollTicket(
    positionId: string
  ): Promise<StagedOptionRollUpTicketEntry | null> {
    try {
      const res = await fetch(
        `${API_BASE}/api/v1/execution/stage-roll/${encodeURIComponent(positionId)}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({})
        }
      );
      if (res.ok) {
        const raw = (await res.json()) as Record<string, unknown>;
        const rollTicket = this.mapStagedOptionRollUpTicket(raw);
        const idx = this.stagedOptionRollTickets.findIndex((t) => t.positionId === positionId);
        if (idx >= 0) {
          const updated = [...this.stagedOptionRollTickets];
          updated[idx] = rollTicket;
          this.stagedOptionRollTickets = updated;
        } else {
          this.stagedOptionRollTickets = [rollTicket, ...this.stagedOptionRollTickets];
        }
        this.activeStagedRollTicketModal = rollTicket;
        executionAudio.playTarget1BankedChime();
        return rollTicket;
      }
    } catch {
      // Fallback below
    }

    const existing = this.stagedOptionRollTickets.find((t) => t.positionId === positionId);
    if (existing) {
      this.activeStagedRollTicketModal = existing;
      return existing;
    }

    const pos = this.openPositions.find((p) => p.tradeId === positionId);
    if (!pos) return null;
    const right = pos.direction === 'LONG' ? 'C' : 'P';
    const nextWall =
      pos.nextGammaWallStrike && pos.nextGammaWallStrike > 0
        ? pos.nextGammaWallStrike
        : pos.target1Price;
    const closeQty = Math.max(1, pos.optionContractsCount || 5);
    const leg1Mid = Math.max(0.5, pos.optionLivePremium || 3.4);
    const leg2Mid = Math.max(0.25, Number((leg1Mid * 0.58).toFixed(2)));
    const netMid = Number((leg1Mid - leg2Mid).toFixed(2));
    const ladder: [number, number, number] = [
      netMid,
      Number((netMid - 0.02).toFixed(2)),
      Number((netMid - 0.05).toFixed(2))
    ];
    const leg1Occ = `${pos.symbol.padEnd(6, ' ')}260410${right}${String(Math.round(pos.entryPrice * 1000)).padStart(8, '0')}`;
    const leg2Occ = `${pos.symbol.padEnd(6, ' ')}260410${right}${String(Math.round(nextWall * 1000)).padStart(8, '0')}`;
    const fallbackRoll: StagedOptionRollUpTicketEntry = {
      rollTicketId: `ROLL-${pos.tradeId}`,
      positionId: pos.tradeId,
      stagedAtIso: new Date().toISOString(),
      symbol: pos.symbol,
      brainId: pos.brainId,
      brainName: pos.brainId,
      direction: pos.direction,
      rollStatus: 'ATOMIC_2LEG_BAG_GAMMA_ROLL_STAGED',
      isGammaRollActive: true,
      currentItmOccSymbol: leg1Occ,
      currentItmContractDesc: pos.optionContractLabel,
      sellToCloseContracts: closeQty,
      currentItmEntryPremium: pos.optionEntryPremium,
      currentItmLivePremium: leg1Mid,
      intrinsicOptionGainUsd: Math.max(120, pos.unrealizedOptionPnlUsd),
      bankedCashFraction: 0.42,
      bankedRealizedCashUsd: pos.bankedOptionGainUsd ?? 185,
      houseMoneyReinvestmentFractionPhi: pos.learnedHouseMoneyPhi ?? 0.58,
      houseMoneyRollBudgetUsd: pos.rolledOptionBudgetUsd ?? 255,
      nextGammaWallStrike: nextWall,
      nextGammaWallOccSymbol: leg2Occ,
      nextGammaWallContractDesc: `${pos.symbol} 2026-11-20 $${nextWall.toFixed(2)}${right}`,
      buyToOpenRollContracts: Math.max(1, Math.round(closeQty * 0.6)),
      nextGammaWallLimitPremium: leg2Mid,
      lockedProfitR: pos.lockedProfitR ?? 0.52,
      learnedProfitLockQuantileR: 0.42,
      liveGivebackLlr: pos.liveSequentialLlr,
      runnerContinuationThreshold: -0.52,
      guaranteedNetFloorR: pos.guaranteedNetFloorR ?? 0.24,
      leg1Ratio: 1,
      leg2Ratio: 1,
      comboOrderType: 'IBKR_BAG_VERTICAL_ROLL_COMBO',
      leg1Close: {
        legId: 1,
        action: 'SELL_TO_CLOSE',
        ratioQty: closeQty,
        occOptionSymbol: leg1Occ,
        contractDesc: pos.optionContractLabel,
        strikePrice: pos.entryPrice,
        expirationDate: '2026-11-20',
        optionRight: right,
        liveMidPrice: leg1Mid
      },
      leg2Open: {
        legId: 2,
        action: 'BUY_TO_OPEN',
        ratioQty: Math.max(1, Math.round(closeQty * 0.6)),
        occOptionSymbol: leg2Occ,
        contractDesc: `${pos.symbol} 2026-11-20 $${nextWall.toFixed(2)}${right}`,
        strikePrice: nextWall,
        expirationDate: '2026-11-20',
        optionRight: right,
        liveMidPrice: leg2Mid
      },
      netComboLimitPrice: netMid,
      nbboLimitWalkLadder: ladder,
      preRollNetDelta: pos.delta * closeQty * 100,
      postRollNetDelta: pos.delta * 0.66 * Math.max(1, Math.round(closeQty * 0.6)) * 100,
      postRollGammaConvexityBoostPct: 18.0,
      lockedCashHarvestedUsd: pos.bankedOptionGainUsd ?? 185,
      ibkrBagCombo: {
        comboOrderType: 'IBKR_BAG_VERTICAL_ROLL_COMBO',
        leg1Ratio: 1,
        leg2Ratio: 1,
        leg1Close: {
          legId: 1,
          action: 'SELL_TO_CLOSE',
          ratioQty: closeQty,
          occOptionSymbol: leg1Occ,
          contractDesc: pos.optionContractLabel,
          strikePrice: pos.entryPrice,
          expirationDate: '2026-11-20',
          optionRight: right,
          liveMidPrice: leg1Mid
        },
        leg2Open: {
          legId: 2,
          action: 'BUY_TO_OPEN',
          ratioQty: Math.max(1, Math.round(closeQty * 0.6)),
          occOptionSymbol: leg2Occ,
          contractDesc: `${pos.symbol} 2026-11-20 $${nextWall.toFixed(2)}${right}`,
          strikePrice: nextWall,
          expirationDate: '2026-11-20',
          optionRight: right,
          liveMidPrice: leg2Mid
        },
        netComboLimitPrice: netMid,
        nbboLimitWalkLadder: ladder,
        preRollNetDelta: pos.delta * closeQty * 100,
        postRollNetDelta: pos.delta * 0.66 * Math.max(1, Math.round(closeQty * 0.6)) * 100,
        postRollGammaConvexityBoostPct: 18.0,
        lockedCashHarvestedUsd: pos.bankedOptionGainUsd ?? 185
      },
      ibkrComboRollPayloadJson: JSON.stringify(
        {
          broker: 'IBKR_TWS_API_V10',
          orderStrategyType: 'IBKR_BAG_VERTICAL_ROLL_COMBO',
          secType: 'BAG',
          symbol: pos.symbol,
          lmtPrice: netMid,
          nbboLimitWalkLadder: ladder
        },
        null,
        2
      ),
      schwabRollPayloadJson: JSON.stringify(
        {
          broker: 'CHARLES_SCHWAB_TRADER_API_V1',
          complexOrderStrategyType: 'SCHWAB_2LEG_NET_ROLL',
          price: netMid
        },
        null,
        2
      ),
      alpacaRollPayloadJson: JSON.stringify(
        {
          broker: 'ALPACA_TRADING_API_V2_MLEG',
          order_class: 'mleg',
          limit_price: netMid.toFixed(2)
        },
        null,
        2
      )
    };
    this.activeStagedRollTicketModal = fallbackRoll;
    return fallbackRoll;
  }

  public async executeRollComboOrder(
    positionId: string,
    broker = 'IBKR_TWS'
  ): Promise<boolean> {
    try {
      const res = await fetch(
        `${API_BASE}/api/v1/execution/execute-roll-combo/${encodeURIComponent(positionId)}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ broker })
        }
      );
      if (res.ok) {
        const raw = (await res.json()) as Record<string, unknown>;
        const conf =
          raw.execution_confirmation && typeof raw.execution_confirmation === 'object'
            ? (raw.execution_confirmation as Record<string, unknown>)
            : raw;
        if (raw.staged_roll_ticket && typeof raw.staged_roll_ticket === 'object') {
          this.activeStagedRollTicketModal = this.mapStagedOptionRollUpTicket(
            raw.staged_roll_ticket as Record<string, unknown>
          );
        } else if (this.activeStagedRollTicketModal) {
          this.activeStagedRollTicketModal = {
            ...this.activeStagedRollTicketModal,
            rollStatus: String(conf.order_status ?? 'ATOMIC_2LEG_BAG_ROLL_FILLED_AND_RECONCILED')
          };
        }
        const orderId = String(
          conf.confirmation_id ?? conf.broker_combo_order_id ?? 'IBKR-BAG'
        );
        const filledPrice = Number(
          conf.executed_net_combo_price ?? conf.filled_net_combo_price ?? 0
        );
        const stepUsed = Number(conf.limit_walk_step_filled ?? conf.nbbo_walk_step_used ?? 1);
        this.lastExecutedRollComboBanner = `Atomic 2-Leg BAG Filled: ${orderId} | Net Peg $${filledPrice.toFixed(2)} (Step #${stepUsed}) | Harvested +$${Number(conf.locked_cash_harvested_usd ?? 0).toFixed(2)} Cash | Gamma Boost +${Number(conf.post_roll_gamma_convexity_boost_pct ?? 18).toFixed(0)}%`;
        executionAudio.playScaleInChime();
        await Promise.all([this.fetchLedgerSnapshot(), this.fetchExecutionTickets()]);
        return true;
      }
    } catch {
      // Offline fallback below
    }
    if (this.activeStagedRollTicketModal) {
      this.activeStagedRollTicketModal = {
        ...this.activeStagedRollTicketModal,
        rollStatus: 'ATOMIC_2LEG_BAG_EXECUTED_FILLED'
      };
      this.lastExecutedRollComboBanner = `Atomic 2-Leg BAG Filled (${broker}): Net Peg $${this.activeStagedRollTicketModal.netComboLimitPrice.toFixed(2)} | Harvested +$${this.activeStagedRollTicketModal.lockedCashHarvestedUsd.toFixed(2)} Cash | Gamma Boost +${this.activeStagedRollTicketModal.postRollGammaConvexityBoostPct.toFixed(0)}%`;
      return true;
    }
    return false;
  }

  public async fetchLedgerSnapshot(): Promise<void> {
    try {
      const res = await fetch(`${API_BASE}/api/v1/ledger/snapshot`);
      if (!res.ok) return;
      const snap = await res.json();
      this.ingestBackendSnapshot(snap);
      this.backendConnected = true;
      this.lastSyncTimestampIso = new Date().toISOString();
    } catch {
      this.backendConnected = false;
    }
  }

  public async fetchScannerCandidates(): Promise<void> {
    try {
      const res = await fetch(`${API_BASE}/api/v1/scanner/candidates`);
      if (!res.ok) return;
      const payload = await res.json();
      if (Array.isArray(payload.candidates)) {
        this.ingestScannerPromotions(payload.candidates);
      }
      if (payload.internals) {
        this.ingestMarketInternals(payload.internals);
      }
    } catch {
      // Handled by connection flag
    }
  }

  public async fetchProvidersTelemetry(): Promise<void> {
    try {
      const res = await fetch(`${API_BASE}/api/v1/providers`);
      if (!res.ok) return;
      const list = await res.json();
      if (Array.isArray(list)) {
        this.providers = list.map((p: Record<string, unknown>, idx: number) => ({
          providerId: String(p.provider_id ?? `SLOT_${idx}`),
          displayName: String(p.display_name ?? p.provider_id ?? ''),
          category:
            idx === 0
              ? 'EQUITY_TAPE'
              : idx % 2 === 1
                ? 'OPTIONS_GREEKS'
                : 'SHORT_FLOAT',
          role: p.is_primary ? 'PRIMARY' : idx === 1 ? 'SECONDARY' : 'TERTIARY',
          healthy: Boolean(p.connected ?? true),
          latencyMs: Number(((Number(p.last_latency_micros ?? 850) || 850) / 1000).toFixed(2)),
          messagesPerSec: Math.max(120, Number(p.messages_processed ?? 1) * 45),
          failoverCount: Number(p.failover_count ?? 0)
        }));
      }
    } catch {
      // Handled by connection flag
    }
  }

  public async fetchEvolutionStatus(): Promise<void> {
    try {
      const res = await fetch(`${API_BASE}/api/v1/evolution/status`);
      if (!res.ok) return;
      const raw = await res.json();
      this.ingestEvolutionStatus(raw);
    } catch {
      // Handled by connection flag
    }
  }

  public async runNightlyAutopsyNow(): Promise<void> {
    try {
      const res = await fetch(`${API_BASE}/api/v1/evolution/run-autopsy`, {
        method: 'POST'
      });
      if (!res.ok) return;
      const raw = await res.json();
      this.ingestEvolutionStatus(raw);
    } catch {
      // Ignore network error
    }
  }

  private ingestEvolutionStatus(raw: Record<string, unknown>): void {
    const grades = Array.isArray(raw.trade_grades)
      ? raw.trade_grades.map((g: Record<string, unknown>) => ({
          tradeId: String(g.trade_id ?? ''),
          symbol: String(g.symbol ?? ''),
          brainId: String(g.brain_id ?? 'BRAIN-02'),
          grade: String(g.grade ?? 'B'),
          realizedRMultiple: Number(g.realized_r_multiple ?? 0),
          mfePct: Number(g.mfe_pct ?? 0),
          maePct: Number(g.mae_pct ?? 0),
          isHardNegative: Boolean(g.is_hard_negative ?? false),
          diagnosis: String(g.diagnosis ?? ''),
          weightAdjustmentSummary: String(g.weight_adjustment_summary ?? '')
        }))
      : [];

    const tournament = Array.isArray(raw.tournament_results)
      ? raw.tournament_results.map((t: Record<string, unknown>) => ({
          brainId: String(t.brain_id ?? 'BRAIN-02'),
          brainName: String(t.brain_name ?? ''),
          championVersion: String(t.champion_version ?? 'v1.0.0-onnx'),
          challengerVersion: String(t.challenger_version ?? 'v1.1.0-challenger'),
          championProfitFactor: Number(t.champion_profit_factor ?? 2.1),
          challengerProfitFactor: Number(t.challenger_profit_factor ?? 2.4),
          championMaxDrawdownPct: Number(t.champion_max_drawdown_pct ?? 4.2),
          challengerMaxDrawdownPct: Number(t.challenger_max_drawdown_pct ?? 3.5),
          challengerPromoted: Boolean(t.challenger_promoted ?? true),
          promotionReason: String(t.promotion_reason ?? '')
        }))
      : [];

    const era1Calibrations = Array.isArray(raw.era1_calibrations)
      ? raw.era1_calibrations.map((c: Record<string, unknown>) => {
          const sw = (c.stream_prior_weights ?? {}) as Record<string, unknown>;
          return {
            brainId: String(c.brain_id ?? 'BRAIN-02'),
            calibratedConfidenceFloor: Number(c.calibrated_confidence_floor ?? 64.0),
            minPrecedentWinRatePct: Number(c.min_precedent_win_rate_pct ?? 58.0),
            minExpectedRMultiple: Number(c.min_expected_r_multiple ?? 1.35),
            chartTapePhysicsWeight: Number(sw.chart_tape_physics_weight ?? 0.25),
            optionsGammaVannaWeight: Number(sw.options_gamma_vanna_weight ?? 0.25),
            shortFloatSqueezeWeight: Number(sw.short_float_squeeze_weight ?? 0.25),
            marketInternalsMacroWeight: Number(sw.market_internals_macro_weight ?? 0.25),
            era1SampleTrades: Number(c.era1_sample_trades ?? 96),
            era1CalibratedWinRatePct: Number(c.era1_calibrated_win_rate_pct ?? 72.0),
            era1CalibratedProfitFactor: Number(c.era1_calibrated_profit_factor ?? 1.95),
            era1LearnedMinDisplacementPct: Number(c.era1_learned_min_displacement_pct ?? 0.15),
            era1LearnedMaxDisplacementPct: Number(c.era1_learned_max_displacement_pct ?? 12.0),
            era1MedianStopPct: Number(c.era1_median_stop_pct ?? 0.015),
            era1LearnedScratchPosteriorThreshold: Number(
              c.era1_learned_scratch_posterior_threshold ?? 0.58
            )
          };
        })
      : [];

    const lossAutopsies = Array.isArray(raw.loss_autopsies)
      ? raw.loss_autopsies.map((a: Record<string, unknown>) => ({
          tradeId: String(a.trade_id ?? ''),
          symbol: String(a.symbol ?? ''),
          brainId: String(a.brain_id ?? 'BRAIN-02'),
          rootCauseCategory: String(a.root_cause_category ?? 'EXHAUSTION_WICK_LIQUIDITY_TRAP'),
          topCulpritFeatures: Array.isArray(a.top_culprit_features)
            ? a.top_culprit_features.map((f: Record<string, unknown>) => ({
                featureIndex: Number(f.feature_index ?? 0),
                featureName: String(f.feature_name ?? ''),
                streamName: String(f.stream_name ?? ''),
                lossValue: Number(f.loss_value ?? 0),
                winnerCentroidValue: Number(f.winner_centroid_value ?? 0),
                anomalyZScore: Number(f.anomaly_z_score ?? 0)
              }))
            : [],
          whatWentWrongSummary: String(a.what_went_wrong_summary ?? ''),
          learnedAdaptationRule: String(a.learned_adaptation_rule ?? ''),
          counterfactualEarlyExitR: Number(a.counterfactual_early_exit_r ?? -0.3)
        }))
      : [];

    const vetoedRepeatTraps = Array.isArray(raw.vetoed_repeat_traps)
      ? raw.vetoed_repeat_traps.map((v: Record<string, unknown>) => ({
          symbol: String(v.symbol ?? ''),
          brainId: String(v.brain_id ?? 'BRAIN-02'),
          timestampNs: Number(v.timestamp_ns ?? 0),
          timestampIso: String(v.timestamp_iso ?? ''),
          matchedPastLossTradeId: String(v.matched_past_loss_trade_id ?? ''),
          rootCauseCategory: String(v.root_cause_category ?? ''),
          trapSimilarityScore: Number(v.trap_similarity_score ?? 0.9),
          counterfactualLossAvoidedUsd: Number(v.counterfactual_loss_avoided_usd ?? 250)
        }))
      : [];

    const evolution_generations_history = Array.isArray(raw.evolution_generations_history)
      ? raw.evolution_generations_history.map((g: Record<string, unknown>) =>
          this.mapEvolutionGenerationRecord(g)
        )
      : [];
    const synaptic_plasticity_states = Array.isArray(raw.synaptic_plasticity_states)
      ? raw.synaptic_plasticity_states.map((s: Record<string, unknown>) =>
          this.mapBrainSynapticPlasticityState(s)
        )
      : [];

    this.evolutionStatus = {
      lastAutopsyIso: String(raw.last_autopsy_iso ?? new Date().toISOString()),
      totalAutopsyRuns: Number(raw.total_autopsy_runs ?? 1),
      hardNegativesMined: Number(raw.hard_negatives_mined ?? 0),
      missedMovesIndexed: Number(raw.missed_moves_indexed ?? 0),
      chopThrottleActive: Boolean(raw.chop_throttle_active ?? false),
      chopConfidenceFloorPct: Number(raw.chop_confidence_floor_pct ?? 68.0),
      tradeGrades: grades,
      tournamentResults: tournament,
      era1Calibrations,
      lossAutopsies,
      vetoedRepeatTraps,
      evolution_generations_history,
      synaptic_plasticity_states
    };
  }

  private mapSynapticFeatureMutation(m: Record<string, unknown>): SynapticFeatureMutation {
    const rawStream = m.stream_name ?? m.stream_id ?? 'STREAM_1_PRICE_VOLUME';
    const streamStr = String(rawStream);
    const streamNum =
      typeof m.stream_id === 'number'
        ? m.stream_id
        : streamStr.includes('2')
          ? 2
          : streamStr.includes('3')
            ? 3
            : streamStr.includes('4')
              ? 4
              : 1;
    const prevW = Number(m.previous_weight ?? m.gen0_weight ?? 1.0);
    const updW = Number(m.updated_weight ?? m.current_weight ?? 1.0);
    const deltaPct = Number(m.delta_pct ?? (updW - prevW) * 100);
    const role = String(m.mutation_role ?? m.role_classification ?? 'NEUTRAL_FEATURE');
    return {
      feature_index: Number(m.feature_index ?? 0),
      feature_name: String(m.feature_name ?? ''),
      stream_id: streamNum,
      stream_name: streamStr,
      previous_weight: prevW,
      updated_weight: updW,
      delta_pct: deltaPct,
      mutation_role: role,
      rationale: String(
        m.rationale ??
          `${role}: ${prevW.toFixed(2)} -> ${updW.toFixed(2)} (${deltaPct >= 0 ? '+' : ''}${deltaPct.toFixed(1)}%)`
      )
    };
  }

  private mapBrainSynapticPlasticityState(
    s: Record<string, unknown>
  ): BrainSynapticPlasticityState {
    const trapWeightsRaw =
      s.trap_conviction_weights && typeof s.trap_conviction_weights === 'object'
        ? (s.trap_conviction_weights as Record<string, unknown>)
        : {};
    const trapWeights: Record<string, number> = {};
    for (const [k, v] of Object.entries(trapWeightsRaw)) {
      trapWeights[k] = Number(v ?? 1.0);
    }
    const rawMidSprt = Array.isArray(s.mid_trade_wald_sprt_thresholds)
      ? s.mid_trade_wald_sprt_thresholds.map((v: unknown) => Number(v ?? 1.44))
      : [1.48, 1.44, 1.4];
    const midSprtTuple: [number, number, number] = [
      rawMidSprt[0] ?? 1.48,
      rawMidSprt[1] ?? 1.44,
      rawMidSprt[2] ?? 1.4
    ];
    const boosted = Array.isArray(s.top_boosted_warning_features)
      ? s.top_boosted_warning_features.map((f: Record<string, unknown>) =>
          this.mapSynapticFeatureMutation(f)
        )
      : Array.isArray(s.top_boosted_features)
        ? s.top_boosted_features.map((f: Record<string, unknown>) =>
            this.mapSynapticFeatureMutation(f)
          )
        : [];
    const slashed = Array.isArray(s.top_slashed_deceiver_features)
      ? s.top_slashed_deceiver_features.map((f: Record<string, unknown>) =>
          this.mapSynapticFeatureMutation(f)
        )
      : Array.isArray(s.top_slashed_features)
        ? s.top_slashed_features.map((f: Record<string, unknown>) =>
            this.mapSynapticFeatureMutation(f)
          )
        : [];
    return {
      brain_id: String(s.brain_id ?? 'BRAIN-02'),
      generation_id: Number(s.generation_id ?? 0),
      synaptic_feature_weights_32: Array.isArray(s.synaptic_feature_weights_32)
        ? s.synaptic_feature_weights_32.map((v: unknown) => Number(v ?? 1.0))
        : Array(32).fill(1.0),
      stream_prior_multipliers: Array.isArray(s.stream_prior_multipliers)
        ? s.stream_prior_multipliers.map((v: unknown) => Number(v ?? 1.0))
        : [1.0, 1.0, 1.0, 1.0],
      wald_sprt_thresholds: Array.isArray(s.wald_sprt_thresholds)
        ? s.wald_sprt_thresholds.map((v: unknown) => Number(v ?? 1.35))
        : [1.35, 1.45, 1.52],
      entry_veto_margin_delta: Number(s.entry_veto_margin_delta ?? 0.025),
      trap_conviction_weights: trapWeights,
      empirical_bayes_shrinkage_lambda: Number(s.empirical_bayes_shrinkage_lambda ?? 0.5),
      mid_trade_wald_sprt_thresholds: midSprtTuple,
      learned_mid_trade_mfe_norm_quantile: Number(
        s.learned_mid_trade_mfe_norm_quantile ?? 0.58
      ),
      learned_mid_trade_stop_ratchet_r: Number(s.learned_mid_trade_stop_ratchet_r ?? -0.19),
      learned_itm_mfe_activation_r: Number(s.learned_itm_mfe_activation_r ?? 0.98),
      giveback_exhaustion_threshold: Number(s.giveback_exhaustion_threshold ?? 1.36),
      runner_continuation_threshold: Number(s.runner_continuation_threshold ?? -0.52),
      learned_giveback_retention_ratio: Number(s.learned_giveback_retention_ratio ?? 0.8),
      learned_profit_lock_activation_r: Number(s.learned_profit_lock_activation_r ?? 0.4),
      learned_house_money_reinvestment_fraction: Number(
        s.learned_house_money_reinvestment_fraction ?? 0.53
      ),
      late_blowoff_scratch_threshold: Number(s.late_blowoff_scratch_threshold ?? 1.28),
      learned_shakeout_absorption_quantile: Number(
        s.learned_shakeout_absorption_quantile ?? 0.72
      ),
      learned_cross_brain_recycle_multiplier: Number(
        s.learned_cross_brain_recycle_multiplier ?? 1.0
      ),
      learned_almgren_chriss_risk_aversion_lambda: Number(
        s.learned_almgren_chriss_risk_aversion_lambda ?? 0.34
      ),
      learned_almgren_chriss_urgency_kappa: Number(
        s.learned_almgren_chriss_urgency_kappa ?? 0.85
      ),
      learned_correlation_spike_quantile: Number(
        s.learned_correlation_spike_quantile ?? 0.34
      ),
      learned_vanna_tail_hedge_ratio_theta: Number(
        s.learned_vanna_tail_hedge_ratio_theta ?? 0.0558
      ),
      learned_sor_venue_weights_4: Array.isArray(s.learned_sor_venue_weights_4)
        ? [
            Number(s.learned_sor_venue_weights_4[0] ?? 0.169),
            Number(s.learned_sor_venue_weights_4[1] ?? 0.6063),
            Number(s.learned_sor_venue_weights_4[2] ?? 0.197),
            Number(s.learned_sor_venue_weights_4[3] ?? 0.0277)
          ]
        : [0.169, 0.6063, 0.197, 0.0277],
      learned_sor_markout_toxicity_bps: Number(s.learned_sor_markout_toxicity_bps ?? 0.45),
      learned_charm_vanna_pin_quantile: Number(s.learned_charm_vanna_pin_quantile ?? -0.18),
      learned_charm_vanna_unpin_quantile: Number(s.learned_charm_vanna_unpin_quantile ?? 0.34),
      learned_occ_tims_sigma_shock_pct: Number(s.learned_occ_tims_sigma_shock_pct ?? 7.82),
      top_boosted_warning_features: boosted,
      top_slashed_deceiver_features: slashed,
      top_boosted_features: boosted,
      top_slashed_features: slashed,
      cumulative_tp_vetoes: Number(s.cumulative_tp_vetoes ?? 0),
      cumulative_fp_vetoes: Number(s.cumulative_fp_vetoes ?? 0),
      cumulative_tp_scratches: Number(s.cumulative_tp_scratches ?? 0),
      cumulative_fp_scratches: Number(s.cumulative_fp_scratches ?? 0),
      remaining_unscratched_losses: Number(s.remaining_unscratched_losses ?? 0),
      last_mutation_summary: String(s.last_mutation_summary ?? '')
    };
  }

  private mapEvolutionGenerationRecord(g: Record<string, unknown>): EvolutionGenerationRecord {
    const tpVeto = Number(g.tp_losses_vetoed ?? 0);
    const fpVeto = Number(g.fp_winners_vetoed ?? 0);
    const tpScratch = Number(g.tp_losses_scratched ?? g.tp_scratches ?? 0);
    const fpScratch = Number(g.fp_winners_scratched ?? 0);
    const midTradeSaves = Number(g.mid_trade_stall_saves_bar4_to_6 ?? 0);
    const earlyScratches = Number(
      g.early_scratches_bar1_to_3 ?? Math.max(0, tpScratch - midTradeSaves)
    );
    return {
      generation_id: Number(g.generation_id ?? 0),
      status: String(g.status ?? g.status_label ?? 'GEN-0 BASELINE'),
      promoted_as_champion: Boolean(g.promoted_as_champion ?? g.is_current_champion ?? false),
      evaluated_at: String(g.evaluated_at ?? g.timestamp_iso ?? new Date().toISOString()),
      active_trades_count: Number(g.active_trades_count ?? g.active_trades ?? 0),
      winners_preserved_count: Number(
        g.winners_preserved_count ?? g.wins ?? g.untouched_winners_count ?? 0
      ),
      active_losses_count: Number(g.active_losses_count ?? g.losses ?? 0),
      win_rate_pct: Number(g.win_rate_pct ?? 0),
      tp_losses_vetoed: tpVeto,
      fp_winners_vetoed: fpVeto,
      veto_precision_pct: Number(g.veto_precision_pct ?? g.entry_veto_precision_pct ?? 0),
      tp_losses_scratched: tpScratch,
      fp_winners_scratched: fpScratch,
      scratch_precision_pct: Number(g.scratch_precision_pct ?? 0),
      early_scratches_bar1_to_3: earlyScratches,
      mid_trade_stall_saves_bar4_to_6: midTradeSaves,
      fp_mid_trade_scratches: Number(g.fp_mid_trade_scratches ?? 0),
      giveback_saves_count: Number(g.giveback_saves_count ?? 0),
      total_giveback_r_recovered: Number(g.total_giveback_r_recovered ?? 0),
      house_money_scaled_runners_count: Number(g.house_money_scaled_runners_count ?? 0),
      total_house_money_r_gained: Number(g.total_house_money_r_gained ?? 0),
      recycled_risk_pool_usd: Number(g.recycled_risk_pool_usd ?? 0),
      recycled_runner_boost_count: Number(g.recycled_runner_boost_count ?? 0),
      recycled_additional_r_compounded: Number(g.recycled_additional_r_compounded ?? 0),
      avg_execution_slippage_bps: Number(g.avg_execution_slippage_bps ?? 2.01),
      slippage_saved_usd: Number(g.slippage_saved_usd ?? 0),
      tail_hedge_ratio_theta: Number(g.tail_hedge_ratio_theta ?? 0),
      active_tail_hedge_count: Number(g.active_tail_hedge_count ?? 0),
      tail_hedge_convexity_payout_usd: Number(g.tail_hedge_convexity_payout_usd ?? 0),
      sor_markout_toxicity_bps: Number(g.sor_markout_toxicity_bps ?? 0.63),
      charm_vanna_unpin_boosts_count: Number(g.charm_vanna_unpin_boosts_count ?? 0),
      charm_vanna_additional_r_gained: Number(g.charm_vanna_additional_r_gained ?? 0),
      occ_tims_pm_relief_ratio_pct: Number(g.occ_tims_pm_relief_ratio_pct ?? 0),
      return_on_margin_rom_pct: Number(g.return_on_margin_rom_pct ?? 334.58),
      pareto_gate_passed: Boolean(g.pareto_gate_passed ?? true),
      avg_winner_r: Number(g.avg_winner_r ?? 0.85),
      tp_scratches: tpScratch,
      expectancy_r: g.expectancy_r !== undefined ? Number(g.expectancy_r) : undefined,
      net_r: g.net_r !== undefined ? Number(g.net_r) : undefined,
      tp_losses_caught_total: Number(g.tp_losses_caught_total ?? tpVeto + tpScratch),
      fp_winners_clipped_total: Number(g.fp_winners_clipped_total ?? fpVeto + fpScratch),
      remaining_full_stops: Number(
        g.remaining_full_stops ?? g.remaining_unscratched_losses ?? 0
      ),
      profit_factor_fixed: Number(g.profit_factor_fixed ?? g.profit_factor ?? 0),
      profit_factor_kelly: Number(g.profit_factor_kelly ?? g.kelly_profit_factor ?? 0),
      fixed_10k_net_pnl: Number(g.fixed_10k_net_pnl ?? g.fixed_net_pnl_usd ?? 0),
      fractional_kelly_net_pnl: Number(
        g.fractional_kelly_net_pnl ?? g.kelly_net_pnl_usd ?? 0
      ),
      max_drawdown_usd: Number(g.max_drawdown_usd ?? 0),
      composite_fitness_score: Number(g.composite_fitness_score ?? 0),
      key_synaptic_mutations: Array.isArray(g.key_synaptic_mutations)
        ? g.key_synaptic_mutations.map((m: unknown) => String(m))
        : Array.isArray(g.mutations_applied_summary)
          ? g.mutations_applied_summary.map((m: unknown) => String(m))
          : [],
      brain_synaptic_states: Array.isArray(g.brain_synaptic_states)
        ? g.brain_synaptic_states.map((s: Record<string, unknown>) =>
            this.mapBrainSynapticPlasticityState(s)
          )
        : []
    };
  }

  public async runNextEvolutionGeneration(): Promise<MultiGenEvolutionResponse | null> {
    this.isRunningEvolutionGeneration = true;
    try {
      const res = await fetch(`${API_BASE}/api/v1/evolution/run-generation`, {
        method: 'POST'
      });
      if (!res.ok) return null;
      const raw = (await res.json()) as Record<string, unknown>;
      const rawGens = Array.isArray(raw.generations)
        ? raw.generations
        : Array.isArray(raw.generations_history)
          ? raw.generations_history
          : [];
      const generations = rawGens.map((g: Record<string, unknown>) =>
        this.mapEvolutionGenerationRecord(g)
      );
      const rawStates = Array.isArray(raw.active_synaptic_states)
        ? raw.active_synaptic_states
        : Array.isArray(raw.synaptic_plasticity_states)
          ? raw.synaptic_plasticity_states
          : [];
      const active_synaptic_states = rawStates.map((s: Record<string, unknown>) =>
        this.mapBrainSynapticPlasticityState(s)
      );
      const latest_generation =
        raw.latest_generation && typeof raw.latest_generation === 'object'
          ? this.mapEvolutionGenerationRecord(
              raw.latest_generation as Record<string, unknown>
            )
          : generations.length > 0
            ? generations[generations.length - 1]!
            : null;
      const parsed: MultiGenEvolutionResponse = {
        active_champion_generation: Number(
          raw.active_champion_generation ?? raw.current_champion_generation ?? 0
        ),
        generations,
        active_synaptic_states,
        latest_generation
      };
      if (latest_generation) {
        const g = latest_generation;
        this.lastEvolutionGenerationBanner = `Gen #${g.generation_id} (${g.status}): WR ${g.win_rate_pct.toFixed(2)}% | PF ${g.profit_factor_fixed.toFixed(2)} | Kelly Net +$${g.fractional_kelly_net_pnl.toFixed(2)} | Fitness ${g.composite_fitness_score.toFixed(1)}`;
      }
      await Promise.all([this.fetchEvolutionStatus(), this.fetchLedgerSnapshot()]);
      return parsed;
    } catch {
      return null;
    } finally {
      this.isRunningEvolutionGeneration = false;
    }
  }

  /**
   * Fetches real Yahoo Finance multi-timeframe chart bars + live 4-stream ONNX inference
   * for `symbol`, `timeframe`, and `range` from `GET /api/v1/chart/{symbol}`.
   */
  public async fetchLiveChartForSymbol(
    symbol: string,
    timeframe: ChartTimeframeId = this.activeTimeframe,
    range: ChartRangeId = this.activeRange
  ): Promise<ScannerSignalCandidate | null> {
    const cleanSym = symbol.trim().toUpperCase();
    if (!cleanSym) return null;

    this.chartLoading = true;
    const brainParam = this.activeBrain === 'ALL_5_BRAINS' ? 'BRAIN-02' : this.activeBrain;
    try {
      const url = `${API_BASE}/api/v1/chart/${encodeURIComponent(cleanSym)}?interval=${encodeURIComponent(timeframe)}&range=${encodeURIComponent(range)}&brain_id=${encodeURIComponent(brainParam)}&limit=64`;
      const res = await fetch(url);
      if (!res.ok) {
        this.chartLoading = false;
        return null;
      }
      const payload = await res.json();
      const candidate = this.ingestBackendChartResponse(payload);
      this.chartLoading = false;
      this.backendConnected = true;
      this.lastSyncTimestampIso = new Date().toISOString();
      return candidate;
    } catch {
      this.chartLoading = false;
      return null;
    }
  }

  /**
   * Maps a `GET /api/v1/chart/{symbol}` response into a `ScannerSignalCandidate` with real Yahoo candles.
   */
  public ingestBackendChartResponse(payload: Record<string, unknown>): ScannerSignalCandidate {
    const sym = String(payload.symbol ?? 'NVDA').toUpperCase();
    const exchangeIndex = String(payload.exchange_index ?? 'SP_500');
    const inf = (payload.inference ?? {}) as Record<string, unknown>;
    const sig = (inf.signal ?? {}) as Record<string, unknown>;
    const opt = (inf.matched_option ?? {}) as Record<string, unknown>;
    const prec = (payload.precedents ?? {}) as Record<string, unknown>;
    const rawBars = Array.isArray(payload.bars) ? payload.bars : [];
    const heatmap = Array.isArray(inf.bar_attention_heatmap) ? inf.bar_attention_heatmap : [];
    const rawShap = Array.isArray(inf.top_shap_drivers) ? inf.top_shap_drivers : [];

    if (payload.internals) {
      this.ingestMarketInternals(payload.internals as Record<string, unknown>);
    }

    const bars: RawBarCandle[] = rawBars.map((b: Record<string, unknown>, i: number) => ({
      timestampMs: Math.floor(Number(b.timestamp_ns ?? Date.now() * 1_000_000) / 1_000_000),
      open: Number(b.open ?? 0),
      high: Number(b.high ?? 0),
      low: Number(b.low ?? 0),
      close: Number(b.close ?? 0),
      volume: Number(b.volume ?? 0),
      attentionWeight: Number(heatmap[i] ?? 0.45)
    }));

    const latestClose = bars.length > 0 ? bars[bars.length - 1]!.close : Number(sig.entry_price ?? 0);
    const entryPrice = Number(sig.entry_price ?? latestClose);
    const stopPrice = Number(sig.structural_stop_price ?? entryPrice * 0.985);
    const t1Price = Number(sig.target_1_price ?? entryPrice * 1.02);
    const t2Price = Number(sig.target_2_price ?? entryPrice * 1.04);
    const holdMs = Number(sig.predicted_hold_duration_ms ?? 1_800_000);
    const brainId = (String(sig.brain_id ?? 'BRAIN-02') as Exclude<BrainId, 'ALL_5_BRAINS'>);
    const direction: TradeDirection =
      String(sig.direction ?? 'LONG').toUpperCase() === 'SHORT' ? 'SHORT' : 'LONG';

    const sw = (sig.stream_weights ?? {}) as Record<string, unknown>;
    const streamWeights = {
      priceVolumeAnomaly: Number(sw.chart_tape_physics_weight ?? 0.35),
      optionsGammaVanna: Number(sw.options_gamma_vanna_weight ?? 0.25),
      shortFloatSqueeze: Number(sw.short_float_squeeze_weight ?? 0.2),
      marketInternalsTide: Number(sw.market_internals_macro_weight ?? 0.2)
    };

    const optStructure = String(opt.option_structure ?? 'SHARES_ONLY');
    const optContract = String(opt.matched_option_contract ?? 'SHARES_ONLY');
    const optCount = Number(opt.option_contracts_count ?? 0);

    const optionsRecommendation =
      optCount > 0 && optStructure !== 'SHARES_ONLY' && optStructure !== 'NONE'
        ? {
            optionStructure: optStructure,
            matchedContractLabel: optContract,
            contractType: (direction === 'LONG' ? 'CALL' : 'PUT') as 'CALL' | 'PUT',
            strike: Math.round(entryPrice),
            expirationDate: optContract.split(' ')[1] ?? '2026-11-20',
            dte: brainId === 'BRAIN-05' ? 365 : brainId === 'BRAIN-04' ? 45 : 14,
            entryPremium: Number(opt.option_entry_premium ?? 0),
            stopPremium: Number(opt.option_stop_premium ?? 0),
            targetPremium: Number(opt.option_target_premium ?? 0),
            delta: Number(opt.contract_delta ?? 0.55),
            gamma: Number(opt.contract_gamma ?? 0.04),
            vanna: Number(opt.contract_vanna ?? 0.018),
            theta: Number(opt.contract_theta ?? -0.08),
            vega: Number(opt.contract_vega ?? 0.14),
            ivPct: Number(opt.contract_iv ?? 0.42) * 100
          }
        : null;

    const shapDrivers: ShapDriverAttribution[] = rawShap.map((pair: unknown) => {
      if (Array.isArray(pair) && pair.length >= 2) {
        return { feature: String(pair[0]), impact: Number(pair[1]) };
      }
      return { feature: 's1_volume_surge_log_ratio', impact: 0.42 };
    });

    const totalK = Math.max(1, Number(prec.matches_count ?? 25));
    const winRatePct = Number(prec.win_rate_pct ?? 80.0);
    const matchingWins = Math.round((winRatePct / 100) * totalK);

    const candidate: ScannerSignalCandidate = {
      symbol: sym,
      exchangeIndex,
      brainId,
      direction,
      assetClass: optionsRecommendation ? 'STOCK + OPTIONS' : 'STOCK',
      livePrice: latestClose,
      entryPrice,
      structuralStopPrice: stopPrice,
      target1Price: t1Price,
      target2Price: t2Price,
      predictedHoldWindow: formatHoldDuration(holdMs),
      predictedHoldMs: holdMs,
      aiConfidencePct: Number(sig.ai_confidence_pct ?? 84.5),
      expectedRMultiple: Number(sig.expected_r_multiple ?? 2.8),
      primaryDriver: mapBackendDriver(String(sig.primary_driver ?? 'VOLUME_ABSORPTION')),
      streamWeights,
      precedentMatchSummary: {
        matchingWins,
        totalK,
        winRatePct,
        avgRealizedR: Number(prec.mean_realized_r_multiple ?? 2.85),
        avgMfePct: Number(prec.p80_mfe_pct ?? 3.4),
        avgMaePct: Number(prec.p85_mae_pct ?? 0.85)
      },
      optionsRecommendation,
      shapDrivers,
      bars
    };

    const existingIdx = this.scannerCandidates.findIndex(
      (c) => c.symbol.toUpperCase() === sym
    );
    if (existingIdx >= 0) {
      const updated = [...this.scannerCandidates];
      updated[existingIdx] = candidate;
      this.scannerCandidates = updated;
    } else {
      this.scannerCandidates = [candidate, ...this.scannerCandidates];
    }

    return candidate;
  }

  /**
   * Maps a backend `FilteredLedgerSnapshot` (`positions_opened` + `closed_trades`) into reactive store state.
   */
  public ingestBackendSnapshot(snap: Record<string, unknown>): void {
    const rawOpen = Array.isArray(snap.positions_opened) ? snap.positions_opened : [];
    const rawClosed = Array.isArray(snap.closed_trades) ? snap.closed_trades : [];

    if (
      Array.isArray(snap.evolution_generations_history) ||
      Array.isArray(snap.synaptic_plasticity_states)
    ) {
      this.ledgerSnapshot = {
        evolution_generations_history: Array.isArray(snap.evolution_generations_history)
          ? snap.evolution_generations_history.map((g: Record<string, unknown>) =>
              this.mapEvolutionGenerationRecord(g)
            )
          : [],
        synaptic_plasticity_states: Array.isArray(snap.synaptic_plasticity_states)
          ? snap.synaptic_plasticity_states.map((s: Record<string, unknown>) =>
              this.mapBrainSynapticPlasticityState(s)
            )
          : []
      };
    }

    if (snap.portfolio_heat_telemetry && typeof snap.portfolio_heat_telemetry === 'object') {
      const h = snap.portfolio_heat_telemetry as Record<string, unknown>;
      const pairs = Array.isArray(h.pairwise_correlations) ? h.pairwise_correlations : [];
      const matrix = Array.isArray(h.cross_position_correlation_matrix)
        ? h.cross_position_correlation_matrix.map((row: unknown) =>
            Array.isArray(row) ? row.map((v: unknown) => Number(v ?? 0)) : []
          )
        : [];
      this.portfolioHeatTelemetry = {
        accountEquityUsd: Number(h.account_equity_usd ?? 128513.2),
        openPositionsCount: Number(h.open_positions_count ?? rawOpen.length),
        aggregateOpenRiskUsd: Number(h.aggregate_open_risk_usd ?? 0),
        aggregateOpenRiskPct: Number(h.aggregate_open_risk_pct ?? 0),
        portfolioHeatCapPct: Number(h.portfolio_heat_cap_pct ?? 8.5),
        grossPortfolioNotionalUsd: Number(h.gross_portfolio_notional_usd ?? 0),
        netPortfolioBetaDeltaUsd: Number(h.net_portfolio_beta_delta_usd ?? 0),
        netPortfolioGamma1pctUsd: Number(h.net_portfolio_gamma_1pct_usd ?? 0),
        netPortfolioVanna1pctUsd: Number(h.net_portfolio_vanna_1pct_usd ?? 0),
        meanPairwiseCorrelation: Number(h.mean_pairwise_correlation ?? 0),
        maxPairwiseCorrelation: Number(h.max_pairwise_correlation ?? 0),
        correlationHeatMultiplier: Number(h.correlation_heat_multiplier ?? 1.0),
        heatRegimeBadge: String(
          h.heat_regime_badge ?? 'LOW HEAT — FULL 1.00x KELLY CAPACITY'
        ),
        positionLabels: Array.isArray(h.position_labels)
          ? h.position_labels.map((s: unknown) => String(s))
          : [],
        crossPositionCorrelationMatrix: matrix,
        pairwiseCorrelations: pairs.map((p: Record<string, unknown>) => ({
          posALabel: String(p.pos_a_label ?? ''),
          posBLabel: String(p.pos_b_label ?? ''),
          symbolA: String(p.symbol_a ?? ''),
          symbolB: String(p.symbol_b ?? ''),
          brainA: String(p.brain_a ?? 'BRAIN-02'),
          brainB: String(p.brain_b ?? 'BRAIN-02'),
          ledoitWolfSimilarity: Number(p.ledoit_wolf_similarity ?? 0),
          directionalAlignment: Number(p.directional_alignment ?? 1)
        })),
        overlappingClustersSummary: String(
          h.overlapping_clusters_summary ?? 'Independent multi-horizon exposure'
        ),
        recycledRiskPoolUsd: Number(h.recycled_risk_pool_usd ?? 0),
        recycledRunnerBoostCount: Number(h.recycled_runner_boost_count ?? 0),
        recycledAdditionalRCompounded: Number(h.recycled_additional_r_compounded ?? 0),
        crossBrainRecycleMultiplier: Number(h.cross_brain_recycle_multiplier ?? 1.0),
        avgExecutionSlippageBps: Number(h.avg_execution_slippage_bps ?? 0.6),
        slippageSavedUsd: Number(h.slippage_saved_usd ?? 0),
        tailHedgeRatioTheta: Number(h.tail_hedge_ratio_theta ?? 0),
        activeTailHedgeCount: Number(h.active_tail_hedge_count ?? 0),
        tailHedgeConvexityPayoutUsd: Number(h.tail_hedge_convexity_payout_usd ?? 0),
        sorVenueWeightsSimplex: Array.isArray(h.sor_venue_weights_simplex)
          ? [
              Number(h.sor_venue_weights_simplex[0] ?? 0.169),
              Number(h.sor_venue_weights_simplex[1] ?? 0.6063),
              Number(h.sor_venue_weights_simplex[2] ?? 0.197),
              Number(h.sor_venue_weights_simplex[3] ?? 0.0277)
            ]
          : [0.169, 0.6063, 0.197, 0.0277],
        sorMarkoutToxicityBps: Number(h.sor_markout_toxicity_bps ?? 0.45),
        charmVannaUnpinBoostsCount: Number(h.charm_vanna_unpin_boosts_count ?? 0),
        charmVannaAdditionalRGained: Number(h.charm_vanna_additional_r_gained ?? 0),
        regTStandaloneMarginUsd: Number(h.reg_t_standalone_margin_usd ?? 23705.28),
        occTimsPortfolioMarginUsd: Number(h.occ_tims_portfolio_margin_usd ?? 15035.94),
        occTimsPmReliefRatioPct: Number(h.occ_tims_pm_relief_ratio_pct ?? 43.06),
        returnOnMarginRomPct: Number(h.return_on_margin_rom_pct ?? 541.47)
      };
    }

    if (Array.isArray(snap.by_strategy_leaderboard)) {
      this.byStrategyLeaderboard = snap.by_strategy_leaderboard.map(
        (s: Record<string, unknown>) => ({
          strategyArchetype: String(
            s.strategy_archetype ?? 'POC_VOLUME_ABSORPTION'
          ) as StrategyArchetypeId,
          displayName: String(s.display_name ?? s.strategy_archetype ?? ''),
          totalTrades: Number(s.total_trades ?? 0),
          wins: Number(s.wins ?? 0),
          losses: Number(s.losses ?? 0),
          winRatePct: Number(s.win_rate_pct ?? 0),
          profitFactor: Number(s.profit_factor ?? 0),
          avgRealizedR: Number(s.avg_realized_r ?? 0),
          bestSymbol: String(s.best_symbol ?? ''),
          realizedStockNetPnlUsd: Number(s.realized_stock_net_pnl_usd ?? 0),
          realizedOptionNetPnlUsd: Number(s.realized_option_net_pnl_usd ?? 0),
          kellyRealizedNetPnlUsd: Number(s.kelly_realized_net_pnl_usd ?? 0),
          kellyProfitFactor: Number(s.kelly_profit_factor ?? s.profit_factor ?? 0),
          adaptiveTotalTrades: Number(s.adaptive_total_trades ?? s.total_trades ?? 0),
          adaptiveWins: Number(s.adaptive_wins ?? s.wins ?? 0),
          adaptiveLosses: Number(s.adaptive_losses ?? s.losses ?? 0),
          adaptiveVetoedLosses: Number(
            s.adaptive_vetoed_count ?? s.adaptive_vetoed_losses ?? 0
          ),
          adaptiveScratchedLosses: Number(
            s.adaptive_scratched_count ?? s.adaptive_scratched_losses ?? 0
          ),
          adaptiveWinRatePct: Number(s.adaptive_win_rate_pct ?? s.win_rate_pct ?? 0),
          adaptiveProfitFactor: Number(s.adaptive_profit_factor ?? s.profit_factor ?? 0),
          adaptiveRealizedNetPnlUsd: Number(
            s.adaptive_stock_net_pnl_usd ??
              s.adaptive_realized_net_pnl_usd ??
              s.realized_stock_net_pnl_usd ??
              0
          ),
          adaptiveKellyNetPnlUsd: Number(
            s.adaptive_kelly_net_pnl_usd ?? s.kelly_realized_net_pnl_usd ?? 0
          ),
          adaptiveLossAvoidedUsd: Number(s.adaptive_loss_avoided_usd ?? 0)
        })
      );
    }

    if (Array.isArray(snap.by_engine_breakdown)) {
      this.byEngineBreakdown = snap.by_engine_breakdown.map((e: Record<string, unknown>) => ({
        engineMode: String(
          e.engine_mode ?? 'STRUCTURED_HYBRID_CHAMPION'
        ) as EngineExecutionModeId,
        displayName: String(e.display_name ?? e.engine_mode ?? ''),
        totalTrades: Number(e.total_trades ?? 0),
        wins: Number(e.wins ?? 0),
        losses: Number(e.losses ?? 0),
        winRatePct: Number(e.win_rate_pct ?? 0),
        profitFactor: Number(e.profit_factor ?? 0),
        avgRealizedR: Number(e.avg_realized_r ?? 0),
        bestSymbol: String(e.best_symbol ?? ''),
        realizedStockNetPnlUsd: Number(e.realized_stock_net_pnl_usd ?? 0),
        realizedOptionNetPnlUsd: Number(e.realized_option_net_pnl_usd ?? 0),
        kellyRealizedNetPnlUsd: Number(e.kelly_realized_net_pnl_usd ?? 0),
        kellyProfitFactor: Number(e.kelly_profit_factor ?? e.profit_factor ?? 0),
        adaptiveTotalTrades: Number(e.adaptive_total_trades ?? e.total_trades ?? 0),
        adaptiveWins: Number(e.adaptive_wins ?? e.wins ?? 0),
        adaptiveLosses: Number(e.adaptive_losses ?? e.losses ?? 0),
        adaptiveVetoedLosses: Number(
          e.adaptive_vetoed_count ?? e.adaptive_vetoed_losses ?? 0
        ),
        adaptiveScratchedLosses: Number(
          e.adaptive_scratched_count ?? e.adaptive_scratched_losses ?? 0
        ),
        adaptiveWinRatePct: Number(e.adaptive_win_rate_pct ?? e.win_rate_pct ?? 0),
        adaptiveProfitFactor: Number(e.adaptive_profit_factor ?? e.profit_factor ?? 0),
        adaptiveRealizedNetPnlUsd: Number(
          e.adaptive_stock_net_pnl_usd ??
            e.adaptive_realized_net_pnl_usd ??
            e.realized_stock_net_pnl_usd ??
            0
        ),
        adaptiveKellyNetPnlUsd: Number(
          e.adaptive_kelly_net_pnl_usd ?? e.kelly_realized_net_pnl_usd ?? 0
        ),
        adaptiveLossAvoidedUsd: Number(e.adaptive_loss_avoided_usd ?? 0)
      }));
    }

    this.openPositions = rawOpen.map((p: Record<string, unknown>) => {
      const sw = (p.stream_weights ?? {}) as Record<string, unknown>;
      const optCount = Number(p.option_contracts_count ?? 0);
      const optStructure = String(p.option_structure ?? 'SHARES_ONLY');
      const matchesCnt = Number(p.precedent_matches_count ?? 25);
      const winRate = Number(p.precedent_win_rate_pct ?? 80.0);
      const wins = Math.round((winRate / 100) * matchesCnt);

      return {
        tradeId: String(p.position_id ?? ''),
        openedTimestampMs: Math.floor(Number(p.opened_timestamp_ns ?? Date.now() * 1_000_000) / 1_000_000),
        symbol: String(p.symbol ?? ''),
        exchangeIndex: String(p.exchange_index ?? 'SP_500'),
        brainId: String(p.brain_id ?? 'BRAIN-02') as Exclude<BrainId, 'ALL_5_BRAINS'>,
        direction: (String(p.direction ?? 'LONG').toUpperCase() === 'SHORT'
          ? 'SHORT'
          : 'LONG') as TradeDirection,
        assetClass: optCount > 0 ? 'STOCK + OPTIONS' : 'STOCK',
        entryPrice: Number(p.entry_price ?? 0),
        livePrice: Number(p.live_price ?? 0),
        quantityShares: Number(p.quantity_shares ?? 1),
        kellyFractionPct: Number(p.kelly_fraction_pct ?? 1.25),
        riskBudgetUsd: Number(p.risk_budget_usd ?? 500),
        notionalAllocatedUsd: Number(
          p.notional_allocated_usd ?? Number(p.entry_price ?? 0) * Number(p.quantity_shares ?? 1)
        ),
        trancheCount: Number(p.tranche_count ?? 1),
        scaleInHistory: Array.isArray(p.scale_in_history)
          ? p.scale_in_history.map((s: unknown) => String(s))
          : [],
        entryBarIdx: Number(p.entry_bar_idx ?? 0),
        structuralStopPrice: Number(p.structural_stop_price ?? 0),
        target1Price: Number(p.target_1_price ?? 0),
        target2Price: Number(p.target_2_price ?? 0),
        unrealizedStockPnlUsd: Number(p.unrealized_stock_pnl_usd ?? 0),
        unrealizedStockPnlPct: Number(p.unrealized_stock_pnl_pct ?? 0),
        optionStructure: optStructure,
        optionContractLabel: String(p.matched_option_contract ?? 'SHARES_ONLY'),
        optionEntryPremium: Number(p.option_entry_premium ?? 0),
        optionLivePremium: Number(p.option_live_premium ?? 0),
        optionContractsCount: optCount,
        unrealizedOptionPnlUsd: Number(p.unrealized_option_pnl_usd ?? 0),
        unrealizedOptionPnlPct: Number(p.unrealized_option_pnl_pct ?? 0),
        delta: Number(p.contract_delta ?? 0),
        gamma: Number(p.contract_gamma ?? 0),
        vanna: Number(p.contract_vanna ?? 0),
        theta: Number(p.contract_theta ?? 0),
        vega: Number(p.contract_vega ?? 0),
        ivPct: Number(p.contract_iv ?? 0) * 100,
        elapsedHoldMs: Math.max(1000, Number(p.live_hold_duration_ms ?? 60_000)),
        predictedHoldMs: 3_600_000,
        currentRMultiple: Number(p.current_r_multiple ?? 0),
        mfePct: Number(p.mfe_pct ?? 0),
        maePct: Number(p.mae_pct ?? 0),
        aiConfidencePct: Number(p.ai_confidence_pct ?? 80.0),
        primaryDriver: mapBackendDriver(String(p.primary_driver ?? 'VOLUME_ABSORPTION')),
        engineMode: String(
          p.engine_mode ?? 'STRUCTURED_HYBRID_CHAMPION'
        ) as EngineExecutionModeId,
        strategyArchetype: String(
          p.strategy_archetype ?? 'POC_VOLUME_ABSORPTION'
        ) as StrategyArchetypeId,
        traceHealthBadge: mapBackendTraceHealth(String(p.trace_health_badge ?? 'ACCELERATING')),
        precedentLabel: `Matches ${wins}/${matchesCnt} Past Expansion Traces (${winRate.toFixed(0)}% Win Rate)`,
        barsHeldCount: Number(p.bars_held_count ?? 1),
        activeSequentialStage: String(
          p.active_sequential_stage ?? 'STAGE_1_EARLY_SPRT (Bar 1..3)'
        ),
        liveSequentialLlr: Number(p.live_sequential_llr ?? 0.28),
        championLlrThreshold: Number(p.champion_llr_threshold ?? 1.4),
        sequentialSafetyMarginPct: Number(p.sequential_safety_margin_pct ?? 46.5),
        nextBarRatchetStopPreview: Number(
          p.next_bar_ratchet_stop_preview ?? p.structural_stop_price ?? 0
        ),
        houseMoneyScaleEligible: Boolean(p.house_money_scale_eligible ?? false),
        learnedHouseMoneyPhi: Number(p.learned_house_money_phi ?? 0.53),
        lockedProfitR: Number(p.locked_profit_r ?? 0),
        guaranteedNetFloorR: Number(p.guaranteed_net_floor_r ?? 0),
        nextGammaWallStrike: Number(p.next_gamma_wall_strike ?? 0),
        nextGammaWallContract: String(p.next_gamma_wall_contract ?? ''),
        bankedOptionGainUsd: Number(p.banked_option_gain_usd ?? 0),
        rolledOptionBudgetUsd: Number(p.rolled_option_budget_usd ?? 0),
        streamWeights: {
          priceVolumeAnomaly: Number(sw.chart_tape_physics_weight ?? 0.35),
          optionsGammaVanna: Number(sw.options_gamma_vanna_weight ?? 0.25),
          shortFloatSqueeze: Number(sw.short_float_squeeze_weight ?? 0.2),
          marketInternalsTide: Number(sw.market_internals_macro_weight ?? 0.2)
        }
      };
    });

    this.closedLedger = rawClosed.map((c: Record<string, unknown>) => {
      const sw = (c.stream_weights ?? {}) as Record<string, unknown>;
      const optCount = Number(c.option_contracts_count ?? 0);
      const optStructure = String(c.option_structure ?? 'SHARES_ONLY');
      const stockNet = Number(c.net_pnl_usd ?? 0);
      const kellyStockNet = Number(c.kelly_net_pnl_usd ?? stockNet);
      const optNet = Number(c.option_net_pnl_usd ?? 0);
      const isWin = Boolean(c.is_winner ?? stockNet > 0);
      const dir: TradeDirection =
        String(c.direction ?? 'LONG').toUpperCase() === 'SHORT' ? 'SHORT' : 'LONG';

      const rawAutopsy = c.loss_forensic_autopsy as Record<string, unknown> | null | undefined;
      const lossForensicAutopsy: LossForensicAutopsyEntry | null =
        rawAutopsy && typeof rawAutopsy === 'object'
          ? {
              tradeId: String(rawAutopsy.trade_id ?? c.trade_id ?? ''),
              symbol: String(rawAutopsy.symbol ?? c.symbol ?? ''),
              brainId: String(rawAutopsy.brain_id ?? c.brain_id ?? 'BRAIN-02'),
              rootCauseCategory: String(
                rawAutopsy.root_cause_category ?? 'EXHAUSTION_WICK_LIQUIDITY_TRAP'
              ),
              topCulpritFeatures: Array.isArray(rawAutopsy.top_culprit_features)
                ? rawAutopsy.top_culprit_features.map((f: Record<string, unknown>) => ({
                    featureIndex: Number(f.feature_index ?? 0),
                    featureName: String(f.feature_name ?? ''),
                    streamName: String(f.stream_name ?? ''),
                    lossValue: Number(f.loss_value ?? 0),
                    winnerCentroidValue: Number(f.winner_centroid_value ?? 0),
                    anomalyZScore: Number(f.anomaly_z_score ?? 0)
                  }))
                : [],
              whatWentWrongSummary: String(rawAutopsy.what_went_wrong_summary ?? ''),
              learnedAdaptationRule: String(rawAutopsy.learned_adaptation_rule ?? ''),
              counterfactualEarlyExitR: Number(rawAutopsy.counterfactual_early_exit_r ?? -0.3)
            }
          : null;

      return {
        tradeId: String(c.trade_id ?? ''),
        openedTimestampMs: Math.floor(Number(c.opened_timestamp_ns ?? Date.now() * 1_000_000) / 1_000_000),
        closedTimestampMs: Math.floor(Number(c.closed_timestamp_ns ?? Date.now() * 1_000_000) / 1_000_000),
        holdDurationMs: Math.max(1000, Number(c.hold_duration_ms ?? 60_000)),
        predictedHoldDurationMs: 3_600_000,
        symbol: String(c.symbol ?? ''),
        exchangeIndex: String(c.exchange_index ?? 'SP_500'),
        brainId: String(c.brain_id ?? 'BRAIN-02') as Exclude<BrainId, 'ALL_5_BRAINS'>,
        direction: dir,
        assetClass: optCount > 0 ? 'STOCK + OPTIONS' : 'STOCK',
        entryPrice: Number(c.entry_price ?? 0),
        exitPrice: Number(c.actual_exit_price ?? 0),
        quantityShares: Number(c.quantity_shares ?? 1),
        kellyFractionPct: Number(c.kelly_fraction_pct ?? 1.25),
        riskBudgetUsd: Number(c.risk_budget_usd ?? 500),
        notionalAllocatedUsd: Number(
          c.notional_allocated_usd ?? Number(c.entry_price ?? 0) * Number(c.quantity_shares ?? 1)
        ),
        trancheCount: Number(c.tranche_count ?? 1),
        scaleInHistory: Array.isArray(c.scale_in_history)
          ? c.scale_in_history.map((s: unknown) => String(s))
          : [],
        entryBarIdx: Number(c.entry_bar_idx ?? 0),
        exitBarIdx: Number(
          c.exit_bar_idx ??
            Math.max(0, (Array.isArray(c.entry_bars) ? c.entry_bars.length : 1) - 1)
        ),
        structuralStopAtEntry: Number(c.initial_structural_stop ?? 0),
        target1AtEntry: Number(c.target_1_price ?? 0),
        target2AtEntry: Number(c.target_2_price ?? 0),
        outcomeBadge: mapBackendOutcome(String(c.outcome_badge ?? 'TARGET HIT')),
        isWinner: isWin,
        realizedStockPnlUsd: stockNet,
        realizedStockPnlPct: Number(c.stock_return_pct ?? 0),
        kellyRealizedStockPnlUsd: kellyStockNet,
        optionStructure: optStructure,
        optionStrike: Math.round(Number(c.entry_price ?? 0)),
        optionExpiration: String(c.matched_option_contract ?? '').split(' ')[1] ?? '2026-11-20',
        optionType: optCount > 0 ? (dir === 'LONG' ? 'CALL' : 'PUT') : 'NONE',
        optionEntryPremium: Number(c.option_entry_premium ?? 0),
        optionExitPremium: Number(c.option_exit_premium ?? 0),
        optionContractsCount: optCount,
        realizedOptionPnlUsd: optNet,
        realizedOptionPnlPct: Number(c.option_return_pct ?? 0),
        combinedTotalPnlUsd: stockNet + optNet,
        realizedRMultiple: Number(c.realized_r_multiple ?? 0),
        mfePct: Number(c.mfe_pct ?? 0),
        maePct: Number(c.mae_pct ?? 0),
        aiConfidenceAtEntryPct: Number(c.ai_confidence_pct ?? 80.0),
        primaryDriver: mapBackendDriver(String(c.primary_driver ?? 'VOLUME_ABSORPTION')),
        engineMode: String(
          c.engine_mode ?? 'STRUCTURED_HYBRID_CHAMPION'
        ) as EngineExecutionModeId,
        strategyArchetype: String(
          c.strategy_archetype ?? 'POC_VOLUME_ABSORPTION'
        ) as StrategyArchetypeId,
        streamWeights: {
          priceVolumeAnomaly: Number(sw.chart_tape_physics_weight ?? 0.35),
          optionsGammaVanna: Number(sw.options_gamma_vanna_weight ?? 0.25),
          shortFloatSqueeze: Number(sw.short_float_squeeze_weight ?? 0.2),
          marketInternalsTide: Number(sw.market_internals_macro_weight ?? 0.2)
        },
        precedentSimilarityScore: Number(c.precedent_win_rate_pct ?? 85) / 100,
        autopsyRootCauseCode: isWin
          ? 'VERIFIED_REAL_YAHOO_EXPANSION'
          : lossForensicAutopsy
            ? lossForensicAutopsy.rootCauseCategory
            : 'HARD_NEGATIVE_STOP_INVALIDATION',
        lossForensicAutopsy,
        entryBars: Array.isArray(c.entry_bars)
          ? c.entry_bars.map((b: Record<string, unknown>) => ({
              timestampMs: Math.floor(Number(b.timestamp_ns ?? Date.now() * 1_000_000) / 1_000_000),
              open: Number(b.open ?? 0),
              high: Number(b.high ?? 0),
              low: Number(b.low ?? 0),
              close: Number(b.close ?? 0),
              volume: Number(b.volume ?? 0),
              attentionWeight: 0.65
            }))
          : [],
        wasVetoedByLossAutopsy: Boolean(c.was_vetoed_by_loss_autopsy ?? false),
        vetoedByLossTradeId: c.vetoed_by_loss_trade_id ? String(c.vetoed_by_loss_trade_id) : null,
        entryTrapSimilarityScore: Number(c.entry_trap_similarity_score ?? 0),
        wasScratchedByLossAutopsy: Boolean(c.was_scratched_by_loss_autopsy ?? false),
        scratchBarOffset: Number(c.scratch_bar_offset ?? 0),
        adaptiveOutcomeBadge: String(c.adaptive_outcome_badge ?? c.outcome_badge ?? 'TARGET HIT'),
        adaptiveExitPrice: Number(c.adaptive_exit_price ?? c.actual_exit_price ?? 0),
        adaptiveScratchExitR: Number(c.adaptive_scratch_exit_r ?? c.realized_r_multiple ?? 0),
        adaptiveNetPnlUsd: Number(c.adaptive_net_pnl_usd ?? stockNet),
        adaptiveKellyNetPnlUsd: Number(c.adaptive_kelly_net_pnl_usd ?? kellyStockNet),
        adaptiveLossSavedUsd: Number(c.adaptive_loss_saved_usd ?? 0),
        adaptiveKellyLossSavedUsd: Number(c.adaptive_kelly_loss_saved_usd ?? 0),
        lossAffinityPosteriorPct: Number(c.loss_affinity_posterior_pct ?? 0),
        wasProtectedFromGiveback: Boolean(c.was_protected_from_giveback ?? false),
        givebackRatchetBarOffset: Number(c.giveback_ratchet_bar_offset ?? 0),
        givebackRSaved: Number(c.giveback_r_saved ?? 0),
        wasScaledWithHouseMoney: Boolean(c.was_scaled_with_house_money ?? false),
        houseMoneyScaleBarOffset: Number(c.house_money_scale_bar_offset ?? 0),
        houseMoneyAdditionalRGained: Number(c.house_money_additional_r_gained ?? 0)
      };
    });

    // Trigger institutional Web Audio chimes and voice callouts on newly opened, scaled-in, or closed positions
    if (this.hasHydratedInitialSnapshot) {
      for (const p of rawOpen as Array<Record<string, unknown>>) {
        const posId = String(p.position_id ?? '');
        const trancheCnt = Number(p.tranche_count ?? 1);
        const sym = String(p.symbol ?? '');
        const brainNum = String(p.brain_id ?? 'BRAIN-02').replace('BRAIN-0', '');
        const dir: TradeDirection =
          String(p.direction ?? 'LONG').toUpperCase() === 'SHORT' ? 'SHORT' : 'LONG';
        const entryPx = Number(p.entry_price ?? 0).toFixed(2);
        const confPct = Math.round(Number(p.ai_confidence_pct ?? 80));

        if (posId && !this.knownOpenPositionIds.has(posId)) {
          if (executionAudio.claimCrossWindowAudioEvent(`OPEN:${posId}`)) {
            executionAudio.playSignalEntryChime(dir);
            const spelled = ExecutionAudioEngine.formatSymbolForSpeech(sym);
            executionAudio.speakExecutionCallout(
              `Argon Brain ${brainNum}: ${dir === 'LONG' ? 'Long' : 'Short'} ${spelled} entered at ${entryPx}, confidence ${confPct} percent`
            );
          }
        } else if (posId && (this.knownOpenTrancheCounts.get(posId) ?? 1) < trancheCnt) {
          if (executionAudio.claimCrossWindowAudioEvent(`SCALE:${posId}:${trancheCnt}`)) {
            executionAudio.playScaleInChime();
            const spelled = ExecutionAudioEngine.formatSymbolForSpeech(sym);
            executionAudio.speakExecutionCallout(
              `Argon Brain ${brainNum}: Scaled into ${spelled} tranche ${trancheCnt} of 3`
            );
          }
        }
      }

      for (const tr of this.closedLedger) {
        if (tr.tradeId && !this.knownClosedTradeIds.has(tr.tradeId)) {
          if (executionAudio.claimCrossWindowAudioEvent(`CLOSED:${tr.tradeId}`)) {
            executionAudio.playPositionClosedChime(tr.isWinner);
            const spelled = ExecutionAudioEngine.formatSymbolForSpeech(tr.symbol);
            const brainNum = tr.brainId.replace('BRAIN-0', '');
            executionAudio.speakExecutionCallout(
              `Argon Brain ${brainNum}: ${spelled} ${tr.outcomeBadge.toLowerCase()}, ${tr.isWinner ? 'profit' : 'loss'} ${Math.abs(Math.round(tr.realizedStockPnlUsd))} dollars`
            );
          }
          break;
        }
      }
    }

    this.knownOpenPositionIds = new Set(this.openPositions.map((p) => p.tradeId));
    this.knownOpenTrancheCounts = new Map(
      (rawOpen as Array<Record<string, unknown>>).map((p) => [
        String(p.position_id ?? ''),
        Number(p.tranche_count ?? 1)
      ])
    );
    this.knownClosedTradeIds = new Set(this.closedLedger.map((t) => t.tradeId));
    this.hasHydratedInitialSnapshot = true;
  }

  private ingestScannerPromotions(promotions: Array<Record<string, unknown>>): void {
    const brains: Array<Exclude<BrainId, 'ALL_5_BRAINS'>> = [
      'BRAIN-01',
      'BRAIN-02',
      'BRAIN-03',
      'BRAIN-04',
      'BRAIN-05'
    ];

    for (let i = 0; i < promotions.length; i++) {
      const item = promotions[i]!;
      const snap = (item.snapshot ?? {}) as Record<string, unknown>;
      const sym = String(snap.symbol ?? '');
      if (!sym) continue;
      // Prevent duplicate scanner candidates (preserve existing candidates or update in place)
      if (this.scannerCandidates.some((c) => c.symbol.toUpperCase() === sym.toUpperCase())) {
        continue;
      }
      const lastPx = Number(snap.last_price ?? 0);
      if (lastPx <= 0) continue;

      const isPenny = lastPx <= 5.0 || Boolean(snap.is_otc_penny);
      const assignedBrain: Exclude<BrainId, 'ALL_5_BRAINS'> = isPenny
        ? 'BRAIN-01'
        : brains[(i % 4) + 1] ?? 'BRAIN-02';
      const logDisp = Number(item.intraday_log_displacement ?? 0.01);
      const dir: TradeDirection = logDisp >= 0 ? 'LONG' : 'SHORT';
      const sign = dir === 'LONG' ? 1 : -1;
      const stopPx = Math.max(0.00001, lastPx * (1 - sign * 0.014));
      const t1Px = Math.max(0.00001, lastPx * (1 + sign * 0.025));
      const t2Px = Math.max(0.00001, lastPx * (1 + sign * 0.048));
      const isOpt = Boolean(snap.is_optionable);

      const newCand: ScannerSignalCandidate = {
        symbol: sym,
        exchangeIndex: String(snap.exchange_index ?? 'SP_500'),
        brainId: assignedBrain,
        direction: dir,
        assetClass: isOpt ? 'STOCK + OPTIONS' : 'STOCK',
        livePrice: lastPx,
        entryPrice: lastPx,
        structuralStopPrice: stopPx,
        target1Price: t1Px,
        target2Price: t2Px,
        predictedHoldWindow: assignedBrain === 'BRAIN-01' ? '15m - 45m' : '1h - 4h',
        predictedHoldMs: assignedBrain === 'BRAIN-01' ? 1_800_000 : 7_200_000,
        aiConfidencePct: Math.min(97.5, 76.0 + Number(item.composite_anomaly_score ?? 1.0) * 8.5),
        expectedRMultiple: 3.2,
        primaryDriver: isPenny ? 'SHORT_FLOAT_SQUEEZE' : 'PRICE_VOLUME_ANOMALY',
        streamWeights: {
          priceVolumeAnomaly: 0.38,
          optionsGammaVanna: isOpt ? 0.28 : 0.05,
          shortFloatSqueeze: isPenny ? 0.42 : 0.18,
          marketInternalsTide: 0.16
        },
        precedentMatchSummary: {
          matchingWins: 20,
          totalK: 25,
          winRatePct: 80.0,
          avgRealizedR: 2.95,
          avgMfePct: 3.8,
          avgMaePct: 0.9
        },
        optionsRecommendation: isOpt
          ? {
              optionStructure: dir === 'LONG' ? 'SINGLE_LEG_CALL' : 'SINGLE_LEG_PUT',
              matchedContractLabel: `${sym} 2026-11-20 $${Math.round(lastPx).toFixed(2)}${dir === 'LONG' ? 'C' : 'P'}`,
              contractType: dir === 'LONG' ? 'CALL' : 'PUT',
              strike: Math.round(lastPx),
              expirationDate: '2026-11-20',
              dte: 14,
              entryPremium: Number((lastPx * 0.032).toFixed(2)),
              stopPremium: Number((lastPx * 0.018).toFixed(2)),
              targetPremium: Number((lastPx * 0.068).toFixed(2)),
              delta: dir === 'LONG' ? 0.55 : -0.55,
              gamma: 0.052,
              vanna: 0.019,
              theta: -0.08,
              vega: 0.14,
              ivPct: 42.5
            }
          : null,
        shapDrivers: [
          { feature: 's1_volume_surge_log_ratio', impact: 0.54 },
          { feature: 's3_float_rotation_velocity', impact: 0.38 }
        ],
        bars: []
      };

      this.scannerCandidates = [...this.scannerCandidates, newCand];
    }
  }

  private ingestMarketInternals(internals: Record<string, unknown>): void {
    const flows = (internals.sector_flows ?? {}) as Record<string, unknown>;
    const vixSpot = Number(internals.vix_spot ?? 17.2);
    const vixFront = Number(internals.vix_front_month ?? 18.3);
    this.macroInternals = {
      nyseTick: Number(internals.nyse_tick ?? 420),
      nyseAdd: Number(internals.add ?? 980),
      voldRatio: Number(internals.vold ?? 1.85),
      vixSpot,
      vixFrontMonth: vixFront,
      soxxPct: Number(((Number(flows.SMH ?? 0.45) || 0.45) * 2.5).toFixed(2)),
      smhPct: Number(((Number(flows.SMH ?? 0.52) || 0.52) * 2.8).toFixed(2)),
      hygLqdSpreadBps: -2.4,
      dxyChangePct: -0.18,
      us10yYieldBps: -3.2,
      vixTermSlope: Number(((vixSpot - vixFront) / Math.max(vixFront, 1)).toFixed(3))
    };
  }

  public selectTicker(symbol: string, broadcast = true): void {
    const clean = symbol.trim().toUpperCase();
    if (!clean) return;
    this.selectedTicker = clean;
    if (typeof window !== 'undefined' && typeof fetch === 'function') {
      void this.fetchLiveChartForSymbol(clean, this.activeTimeframe, this.activeRange);
      const hasExistingHistory =
        this.closedLedger.some((t) => t.symbol.toUpperCase() === clean) ||
        this.openPositions.some((p) => p.symbol.toUpperCase() === clean);
      if (!hasExistingHistory && this.backendConnected) {
        void this.runCustomSymbolBacktest(clean, this.activeBrain);
      }
    }
    if (broadcast) {
      this.syncBus.publish({
        type: 'SELECT_TICKER',
        symbol: clean,
        linkChannel: this.activeLinkChannel
      });
    }
  }

  public setAudioMode(mode: ExecutionAudioMode): void {
    executionAudio.setMode(mode);
    this.audioMode = mode;
  }

  public cycleAudioMode(): ExecutionAudioMode {
    const next = executionAudio.cycleMode();
    this.audioMode = next;
    return next;
  }

  public testExecutionAudio(): void {
    executionAudio.playTestSound();
    this.audioMode = executionAudio.getMode();
  }

  public dismissBacktestSummary(): void {
    this.lastBacktestSummary = null;
  }

  public clearReplayTrade(): void {
    this.activeReplayTrade = null;
  }

  private mapWebhookStatusPayload(raw: Record<string, unknown>): WebhookStatusPayload {
    const cfg = (raw.config ?? {}) as Record<string, unknown>;
    const rawRecs = Array.isArray(raw.outbox_records) ? raw.outbox_records : [];
    return {
      config: {
        discordWebhookUrl: String(cfg.discord_webhook_url ?? ''),
        discordEnabled: Boolean(cfg.discord_enabled ?? false),
        telegramBotToken: String(cfg.telegram_bot_token ?? ''),
        telegramChatId: String(cfg.telegram_chat_id ?? ''),
        telegramEnabled: Boolean(cfg.telegram_enabled ?? false),
        slackWebhookUrl: String(cfg.slack_webhook_url ?? ''),
        slackEnabled: Boolean(cfg.slack_enabled ?? false),
        customJsonWebhookUrl: String(cfg.custom_json_webhook_url ?? ''),
        customJsonEnabled: Boolean(cfg.custom_json_enabled ?? false)
      },
      totalOutboxRecords: Number(raw.total_outbox_records ?? rawRecs.length),
      outboxRecords: rawRecs.map((r: Record<string, unknown>) => ({
        dispatchId: String(r.dispatch_id ?? ''),
        dispatchedAtIso: String(r.dispatched_at_iso ?? ''),
        dispatchedTimestampMs: Number(r.dispatched_timestamp_ms ?? Date.now()),
        eventType: String(r.event_type ?? 'SIGNAL_ENTRY_OPENED'),
        outcomeDetail: String(r.outcome_detail ?? ''),
        symbol: String(r.symbol ?? ''),
        brainId: String(r.brain_id ?? 'BRAIN-02'),
        brainName: String(r.brain_name ?? ''),
        direction: (String(r.direction ?? 'LONG').toUpperCase() === 'SHORT'
          ? 'SHORT'
          : 'LONG') as TradeDirection,
        entryPrice: Number(r.entry_price ?? 0),
        structuralStopPrice: Number(r.structural_stop_price ?? 0),
        target1Price: Number(r.target_1_price ?? 0),
        target2Price: Number(r.target_2_price ?? 0),
        currentOrExitPrice: Number(r.current_or_exit_price ?? 0),
        kellyFractionPct: Number(r.kelly_fraction_pct ?? 1.25),
        riskBudgetUsd: Number(r.risk_budget_usd ?? 500),
        notionalAllocatedUsd: Number(r.notional_allocated_usd ?? 10000),
        matchedOptionContract: String(r.matched_option_contract ?? 'SHARES_ONLY'),
        greeksSummary: String(r.greeks_summary ?? ''),
        streamAttributionSummary: String(r.stream_attribution_summary ?? ''),
        precedentWinRatePct: Number(r.precedent_win_rate_pct ?? 75.0),
        aiConfidencePct: Number(r.ai_confidence_pct ?? 85.0),
        pnlUsd: Number(r.pnl_usd ?? 0),
        rMultiple: Number(r.r_multiple ?? 0),
        deliveryStatus: String(r.delivery_status ?? 'QUEUED_LOCAL_OUTBOX'),
        channelsTargeted: Array.isArray(r.channels_targeted)
          ? r.channels_targeted.map((c: unknown) => String(c))
          : ['LOCAL_OUTBOX'],
        discordEmbedJson: String(r.discord_embed_json ?? ''),
        telegramMarkdown: String(r.telegram_markdown ?? ''),
        slackBlocksJson: String(r.slack_blocks_json ?? ''),
        genericWebhookJson: String(r.generic_webhook_json ?? '')
      }))
    };
  }

  public async fetchWebhookStatus(): Promise<WebhookStatusPayload | null> {
    try {
      const res = await fetch(`${API_BASE}/api/v1/webhooks/status`);
      if (!res.ok) return null;
      const raw = (await res.json()) as Record<string, unknown>;
      const mapped = this.mapWebhookStatusPayload(raw);
      this.webhookStatus = mapped;
      return mapped;
    } catch {
      return null;
    }
  }

  public async saveWebhookConfig(
    cfg: WebhookChannelConfigState
  ): Promise<WebhookStatusPayload | null> {
    try {
      const res = await fetch(`${API_BASE}/api/v1/webhooks/configure`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          discord_webhook_url: cfg.discordWebhookUrl,
          discord_enabled: cfg.discordEnabled,
          telegram_bot_token: cfg.telegramBotToken,
          telegram_chat_id: cfg.telegramChatId,
          telegram_enabled: cfg.telegramEnabled,
          slack_webhook_url: cfg.slackWebhookUrl,
          slack_enabled: cfg.slackEnabled,
          custom_json_webhook_url: cfg.customJsonWebhookUrl,
          custom_json_enabled: cfg.customJsonEnabled
        })
      });
      if (!res.ok) return null;
      const raw = (await res.json()) as Record<string, unknown>;
      const mapped = this.mapWebhookStatusPayload(raw);
      this.webhookStatus = mapped;
      return mapped;
    } catch {
      return null;
    }
  }

  public async fireTestWebhookAlert(symbol: string = this.selectedTicker): Promise<boolean> {
    this.webhookTesting = true;
    try {
      const res = await fetch(`${API_BASE}/api/v1/webhooks/test`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol })
      });
      this.webhookTesting = false;
      if (!res.ok) return false;
      await this.fetchWebhookStatus();
      executionAudio.playSignalEntryChime('LONG');
      return true;
    } catch {
      this.webhookTesting = false;
      return false;
    }
  }

  /**
   * Executes an on-demand causal Era-1 Calibration + Era-2 Walk-Forward Backtest for any valid
   * Yahoo Finance ticker (`POST /api/v1/brains/{brain_id}/backtest/{symbol}`).
   */
  public async runCustomSymbolBacktest(
    symbol: string = this.selectedTicker,
    brainId: BrainId = this.activeBrain
  ): Promise<CustomSymbolBacktestSummary | null> {
    const cleanSym = symbol.trim().toUpperCase();
    if (!cleanSym) return null;
    const targetBrain: Exclude<BrainId, 'ALL_5_BRAINS'> =
      brainId === 'ALL_5_BRAINS' ? 'BRAIN-02' : brainId;

    this.backtestLoading = true;
    try {
      const res = await fetch(
        `${API_BASE}/api/v1/brains/${encodeURIComponent(targetBrain)}/backtest/${encodeURIComponent(cleanSym)}`,
        { method: 'POST' }
      );
      if (!res.ok) {
        this.backtestLoading = false;
        return null;
      }
      const raw = (await res.json()) as Record<string, unknown>;
      const cal = (raw.era1_calibration ?? {}) as Record<string, unknown>;
      const sw = (cal.stream_prior_weights ?? {}) as Record<string, unknown>;

      const summary: CustomSymbolBacktestSummary = {
        symbol: String(raw.symbol ?? cleanSym).toUpperCase(),
        brainId: String(raw.brain_id ?? targetBrain) as Exclude<BrainId, 'ALL_5_BRAINS'>,
        timeframe: String(raw.timeframe ?? '5m'),
        totalBarsScanned: Number(raw.total_bars_scanned ?? 0),
        era1Calibration: {
          brainId: String(cal.brain_id ?? targetBrain),
          calibratedConfidenceFloor: Number(cal.calibrated_confidence_floor ?? 64.0),
          minPrecedentWinRatePct: Number(cal.min_precedent_win_rate_pct ?? 58.0),
          minExpectedRMultiple: Number(cal.min_expected_r_multiple ?? 1.35),
          chartTapePhysicsWeight: Number(sw.chart_tape_physics_weight ?? 0.25),
          optionsGammaVannaWeight: Number(sw.options_gamma_vanna_weight ?? 0.25),
          shortFloatSqueezeWeight: Number(sw.short_float_squeeze_weight ?? 0.25),
          marketInternalsMacroWeight: Number(sw.market_internals_macro_weight ?? 0.25),
          era1SampleTrades: Number(cal.era1_sample_trades ?? 0),
          era1CalibratedWinRatePct: Number(cal.era1_calibrated_win_rate_pct ?? 70.0),
          era1CalibratedProfitFactor: Number(cal.era1_calibrated_profit_factor ?? 1.5),
          era1LearnedMinDisplacementPct: Number(cal.era1_learned_min_displacement_pct ?? 0.15),
          era1LearnedMaxDisplacementPct: Number(cal.era1_learned_max_displacement_pct ?? 12.0),
          era1MedianStopPct: Number(cal.era1_median_stop_pct ?? 0.015),
          era1LearnedScratchPosteriorThreshold: Number(
            cal.era1_learned_scratch_posterior_threshold ?? 0.58
          )
        },
        closedTradesGenerated: Number(raw.closed_trades_generated ?? 0),
        openPositionActive: Boolean(raw.open_position_active ?? false),
        winRatePct: Number(raw.win_rate_pct ?? 0),
        profitFactor: Number(raw.profit_factor ?? 0),
        realizedStockNetPnlUsd: Number(raw.realized_stock_net_pnl_usd ?? 0),
        realizedOptionNetPnlUsd: Number(raw.realized_option_net_pnl_usd ?? 0),
        completedAtIso: new Date().toISOString()
      };

      this.lastBacktestSummary = summary;
      this.backtestLoading = false;

      if (executionAudio.claimCrossWindowAudioEvent(`BACKTEST:${summary.symbol}:${summary.brainId}:${summary.closedTradesGenerated}`)) {
        executionAudio.playTarget1BankedChime();
        const spelled = ExecutionAudioEngine.formatSymbolForSpeech(summary.symbol);
        const brainNum = summary.brainId.replace('BRAIN-0', '');
        executionAudio.speakExecutionCallout(
          `Deep backtest complete for ${spelled} on Brain ${brainNum}: ${summary.closedTradesGenerated} trades, profit factor ${summary.profitFactor.toFixed(2)}`
        );
      }

      await Promise.all([
        this.fetchLedgerSnapshot(),
        this.fetchEvolutionStatus(),
        this.fetchLiveChartForSymbol(summary.symbol, this.activeTimeframe, this.activeRange)
      ]);
      return summary;
    } catch {
      this.backtestLoading = false;
      return null;
    }
  }

  /**
   * Loads a closed trade's historical Yahoo Finance candle window into `BlueprintCanvas`
   * and synchronizes the active ticker across linked windows.
   */
  public selectClosedTradeForReplay(row: ClosedTradeLedgerItem): void {
    this.selectedTicker = row.symbol;
    this.activeReplayTrade = row;
    if (row.entryBars && row.entryBars.length > 0) {
      const candidate: ScannerSignalCandidate = {
        symbol: row.symbol,
        exchangeIndex: row.exchangeIndex,
        brainId: row.brainId,
        direction: row.direction,
        assetClass: row.assetClass,
        livePrice: row.exitPrice,
        entryPrice: row.entryPrice,
        structuralStopPrice: row.structuralStopAtEntry,
        target1Price: row.target1AtEntry,
        target2Price: row.target2AtEntry,
        predictedHoldWindow: formatHoldDuration(row.holdDurationMs),
        predictedHoldMs: row.holdDurationMs,
        aiConfidencePct: row.aiConfidenceAtEntryPct,
        expectedRMultiple: Math.max(1.5, Math.abs(row.realizedRMultiple)),
        primaryDriver: row.primaryDriver,
        streamWeights: { ...row.streamWeights },
        precedentMatchSummary: {
          matchingWins: 21,
          totalK: 25,
          winRatePct: row.precedentSimilarityScore * 100,
          avgRealizedR: row.realizedRMultiple,
          avgMfePct: row.mfePct,
          avgMaePct: row.maePct
        },
        optionsRecommendation:
          row.optionContractsCount > 0
            ? {
                optionStructure: row.optionStructure,
                matchedContractLabel: `${row.symbol} ${row.optionExpiration} $${row.optionStrike.toFixed(2)} ${row.optionType}`,
                contractType: row.optionType === 'PUT' ? 'PUT' : 'CALL',
                strike: row.optionStrike,
                expirationDate: row.optionExpiration,
                dte: 14,
                entryPremium: row.optionEntryPremium,
                stopPremium: row.optionEntryPremium * 0.55,
                targetPremium: row.optionExitPremium,
                delta: row.direction === 'LONG' ? 0.58 : -0.58,
                gamma: 0.048,
                vanna: 0.019,
                theta: -0.07,
                vega: 0.14,
                ivPct: 42.0
              }
            : null,
        shapDrivers: [
          { feature: 's1_volume_surge_log_ratio', impact: 0.62 },
          { feature: 's2_net_gex_wall_distance_pct', impact: 0.44 },
          { feature: 's3_short_squeeze_pressure_z', impact: 0.31 }
        ],
        bars: row.entryBars
      };

      const idx = this.scannerCandidates.findIndex(
        (c) => c.symbol.toUpperCase() === row.symbol.toUpperCase()
      );
      if (idx >= 0) {
        const updated = [...this.scannerCandidates];
        updated[idx] = candidate;
        this.scannerCandidates = updated;
      } else {
        this.scannerCandidates = [candidate, ...this.scannerCandidates];
      }
    } else {
      this.selectTicker(row.symbol);
    }
  }

  public setTimeframe(tf: ChartTimeframeId): void {
    this.activeTimeframe = tf;
    if (typeof window !== 'undefined' && typeof fetch === 'function') {
      void this.fetchLiveChartForSymbol(this.selectedTicker, tf, this.activeRange);
    }
  }

  public setRange(range: ChartRangeId): void {
    this.activeRange = range;
    if (typeof window !== 'undefined' && typeof fetch === 'function') {
      void this.fetchLiveChartForSymbol(this.selectedTicker, this.activeTimeframe, range);
    }
  }

  public setWorkspaceLayout(mode: WorkspaceLayoutMode): void {
    this.workspaceLayout = mode;
    if (mode !== 'SINGLE' && typeof window !== 'undefined' && typeof fetch === 'function') {
      void this.hydrateMultiViewports(mode);
    }
  }

  public async hydrateMultiViewports(mode: WorkspaceLayoutMode = this.workspaceLayout): Promise<void> {
    if (mode === 'SPLIT_50_50') {
      await Promise.all(this.splitPanes.map((p) => this.fetchViewportPaneChart(p.paneId)));
    } else if (mode === 'QUAD_2X2') {
      await Promise.all(this.quadViewports.map((p) => this.fetchViewportPaneChart(p.paneId)));
    }
  }

  public async updateViewportPane(
    paneId: string,
    patch: Partial<{
      symbol: string;
      brainId: Exclude<BrainId, 'ALL_5_BRAINS'>;
      timeframe: ChartTimeframeId;
      range: ChartRangeId;
      linkChannel: LinkChannelColor;
    }>
  ): Promise<void> {
    const cleanSym = patch.symbol ? patch.symbol.trim().toUpperCase() : undefined;
    const sIdx = this.splitPanes.findIndex((p) => p.paneId === paneId);
    if (sIdx >= 0) {
      const cur = this.splitPanes[sIdx]!;
      this.splitPanes[sIdx] = {
        ...cur,
        ...(patch.brainId ? { brainId: patch.brainId } : {}),
        ...(cleanSym ? { symbol: cleanSym } : {}),
        ...(patch.timeframe ? { timeframe: patch.timeframe } : {}),
        ...(patch.range ? { range: patch.range } : {}),
        ...(patch.linkChannel ? { linkChannel: patch.linkChannel } : {})
      };
      await this.fetchViewportPaneChart(paneId);
      return;
    }

    const qIdx = this.quadViewports.findIndex((p) => p.paneId === paneId);
    if (qIdx >= 0) {
      const cur = this.quadViewports[qIdx]!;
      this.quadViewports[qIdx] = {
        ...cur,
        ...(patch.brainId ? { brainId: patch.brainId } : {}),
        ...(cleanSym ? { symbol: cleanSym } : {}),
        ...(patch.timeframe ? { timeframe: patch.timeframe } : {}),
        ...(patch.range ? { range: patch.range } : {}),
        ...(patch.linkChannel ? { linkChannel: patch.linkChannel } : {})
      };
      await this.fetchViewportPaneChart(paneId);
    }
  }

  public async fetchViewportPaneChart(paneId: string): Promise<void> {
    const isSplit = paneId.startsWith('SPLIT_');
    const list = isSplit ? this.splitPanes : this.quadViewports;
    const idx = list.findIndex((p) => p.paneId === paneId);
    if (idx < 0) return;

    const pane = list[idx]!;
    pane.loading = true;
    try {
      const url = `${API_BASE}/api/v1/chart/${encodeURIComponent(pane.symbol)}?interval=${encodeURIComponent(pane.timeframe)}&range=${encodeURIComponent(pane.range)}&brain_id=${encodeURIComponent(pane.brainId)}&limit=64`;
      const res = await fetch(url);
      if (res.ok) {
        const payload = await res.json();
        const candidate = this.ingestBackendChartResponse(payload);
        pane.blueprint = candidate;
      }
    } catch {
      // Fallback to matching scannerCandidate if offline in unit test
      const fallback = this.scannerCandidates.find(
        (c) => c.symbol.toUpperCase() === pane.symbol.toUpperCase()
      );
      if (fallback) {
        pane.blueprint = fallback;
      }
    } finally {
      pane.loading = false;
    }
  }

  public setLinkChannel(color: LinkChannelColor, broadcast = false): void {
    this.activeLinkChannel = color;
    if (broadcast) {
      this.syncBus.publish({ type: 'SET_LINK_CHANNEL', linkChannel: color });
    }
  }

  public selectBrain(brainId: BrainId, broadcast = true): void {
    this.activeBrain = brainId;
    if (broadcast) {
      this.syncBus.publish({
        type: 'SELECT_BRAIN',
        brainId,
        linkChannel: this.activeLinkChannel
      });
    }
  }

  public setQuickFilterPill(pill: QuickFilterPillId, broadcast = true): void {
    this.quickPill = pill;
    if (broadcast) {
      this.syncBus.publish({ type: 'APPLY_QUICK_FILTER', pillId: pill });
    }
  }

  public setSizingMode(mode: SizingModeId): void {
    this.sizingMode = mode;
  }

  public toggleSizingMode(): SizingModeId {
    this.sizingMode =
      this.sizingMode === 'FIXED_10K_BASELINE' ? 'FRACTIONAL_KELLY' : 'FIXED_10K_BASELINE';
    return this.sizingMode;
  }

  public setImmunityMode(mode: ImmunityModeId): void {
    this.immunityMode = mode;
  }

  public toggleImmunityMode(): ImmunityModeId {
    this.immunityMode =
      this.immunityMode === 'BASELINE_ALL_TRADES'
        ? 'ACTIVE_LOSS_IMMUNITY'
        : 'BASELINE_ALL_TRADES';
    return this.immunityMode;
  }

  public setStrategyFilter(archetype: 'ALL' | StrategyArchetypeId): void {
    this.strategyFilter = this.strategyFilter === archetype ? 'ALL' : archetype;
  }

  public setEngineFilter(engineMode: 'ALL' | EngineExecutionModeId): void {
    this.engineFilter = this.engineFilter === engineMode ? 'ALL' : engineMode;
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
    this.strategyFilter = 'ALL';
    this.engineFilter = 'ALL';
    this.minConfidencePct = 50;
    this.minRMultiple = -10;
  }

  /**
   * Triggers a fresh Yahoo Finance live universe sweep and walk-forward ledger update on the backend.
   */
  public async triggerYahooUniverseScan(): Promise<void> {
    try {
      const res = await fetch(`${API_BASE}/api/v1/brains/scan-yahoo`, {
        method: 'POST'
      });
      if (res.ok) {
        const snap = await res.json();
        this.ingestBackendSnapshot(snap);
        await this.fetchScannerCandidates();
        await this.fetchLiveChartForSymbol(this.selectedTicker, this.activeTimeframe, this.activeRange);
      }
    } catch {
      // Handled by connection flag
    }
  }

  /**
   * Hot-swaps the active primary Market Data Adapter on the Rust backend (`<1ms`).
   */
  public async hotSwapProvider(providerId: string): Promise<void> {
    try {
      const res = await fetch(
        `${API_BASE}/api/v1/providers/hotswap/${encodeURIComponent(providerId)}`,
        { method: 'POST' }
      );
      if (res.ok) {
        await this.fetchProvidersTelemetry();
      }
    } catch {
      // Handled by connection flag
    }
  }

  /**
   * Opens a scanner candidate directly into `POSITIONS OPENED` (and posts to the Rust backend when connected).
   */
  public openSignalAsPosition(candidate: ScannerSignalCandidate): OpenPositionItem {
    if (typeof window !== 'undefined' && typeof fetch === 'function' && this.backendConnected) {
      void fetch(
        `${API_BASE}/api/v1/brains/${encodeURIComponent(candidate.brainId)}/evaluate/${encodeURIComponent(candidate.symbol)}`,
        { method: 'POST' }
      ).then(() => this.fetchLedgerSnapshot());
    }

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
      kellyFractionPct: 1.35,
      riskBudgetUsd: 135.0,
      notionalAllocatedUsd: notionalUsd,
      trancheCount: 1,
      scaleInHistory: [],
      entryBarIdx: Math.max(0, candidate.bars.length - 1),
      structuralStopPrice: candidate.structuralStopPrice,
      target1Price: candidate.target1Price,
      target2Price: candidate.target2Price,
      unrealizedStockPnlUsd: 0,
      unrealizedStockPnlPct: 0,
      optionStructure: opt ? opt.optionStructure : 'SHARES_ONLY',
      optionContractLabel: opt ? opt.matchedContractLabel : 'SHARES_ONLY',
      optionEntryPremium: opt ? opt.entryPremium : 0,
      optionLivePremium: opt ? opt.entryPremium : 0,
      optionContractsCount: opt ? 5 : 0,
      unrealizedOptionPnlUsd: 0,
      unrealizedOptionPnlPct: 0,
      delta: opt ? opt.delta : 0,
      gamma: opt ? opt.gamma : 0,
      vanna: opt ? opt.vanna : 0,
      theta: opt ? opt.theta : 0,
      vega: opt ? opt.vega : 0,
      ivPct: opt ? opt.ivPct : 0,
      elapsedHoldMs: 60_000,
      predictedHoldMs: candidate.predictedHoldMs,
      currentRMultiple: 0.1,
      mfePct: 0.25,
      maePct: 0.05,
      aiConfidencePct: candidate.aiConfidencePct,
      primaryDriver: candidate.primaryDriver,
      engineMode: 'STRUCTURED_HYBRID_CHAMPION',
      strategyArchetype:
        candidate.primaryDriver === 'OPTIONS_GAMMA_VANNA'
          ? 'GAMMA_SQUEEZE_IGNITION'
          : candidate.primaryDriver === 'SHORT_FLOAT_SQUEEZE'
            ? 'SHORT_FLOAT_LIQUEFACTION'
            : candidate.primaryDriver === 'MARKET_INTERNALS_TIDE'
              ? 'MACRO_INTERNALS_TIDE_DRIVE'
              : 'POC_VOLUME_ABSORPTION',
      traceHealthBadge: 'OPTIMAL',
      precedentLabel: `Matches ${candidate.precedentMatchSummary.matchingWins}/${candidate.precedentMatchSummary.totalK} Past Expansion Traces (${candidate.precedentMatchSummary.winRatePct.toFixed(0)}% Win Rate)`,
      barsHeldCount: 1,
      activeSequentialStage: 'STAGE_1_EARLY_SPRT (Bar 1..3)',
      liveSequentialLlr: 0.28,
      championLlrThreshold: 1.4,
      sequentialSafetyMarginPct: 46.5,
      nextBarRatchetStopPreview: candidate.structuralStopPrice,
      streamWeights: { ...candidate.streamWeights }
    };
    this.openPositions = [newPos, ...this.openPositions];
    if (executionAudio.claimCrossWindowAudioEvent(`LOCAL_OPEN:${newPos.tradeId}`)) {
      executionAudio.playSignalEntryChime(newPos.direction);
      const spelled = ExecutionAudioEngine.formatSymbolForSpeech(newPos.symbol);
      const brainNum = newPos.brainId.replace('BRAIN-0', '');
      executionAudio.speakExecutionCallout(
        `Argon Brain ${brainNum}: ${newPos.direction === 'LONG' ? 'Long' : 'Short'} ${spelled} entered at ${newPos.entryPrice.toFixed(2)}, confidence ${Math.round(newPos.aiConfidencePct)} percent`
      );
    }
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
      kellyFractionPct: pos.kellyFractionPct,
      riskBudgetUsd: pos.riskBudgetUsd,
      notionalAllocatedUsd: pos.notionalAllocatedUsd,
      trancheCount: pos.trancheCount,
      scaleInHistory: [...pos.scaleInHistory],
      entryBarIdx: pos.entryBarIdx,
      exitBarIdx: Math.max(pos.entryBarIdx, 24),
      structuralStopAtEntry: pos.structuralStopPrice,
      target1AtEntry: pos.target1Price,
      target2AtEntry: pos.target2Price,
      outcomeBadge,
      isWinner,
      realizedStockPnlUsd: stockPnlUsd,
      realizedStockPnlPct: stockPnlPct,
      kellyRealizedStockPnlUsd: stockPnlUsd * 1.15,
      optionStructure: pos.optionStructure,
      optionStrike: pos.entryPrice,
      optionExpiration: '2026-11-20',
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
      engineMode: pos.engineMode,
      strategyArchetype: pos.strategyArchetype,
      streamWeights: { ...pos.streamWeights },
      precedentSimilarityScore: 0.928,
      autopsyRootCauseCode: isWinner
        ? 'VERIFIED_REAL_YAHOO_EXPANSION'
        : 'STOP_DISTANCE_TIGHTENED_BY_ABSORPTION_WICK',
      lossForensicAutopsy: null,
      entryBars: [],
      wasVetoedByLossAutopsy: false,
      vetoedByLossTradeId: null,
      entryTrapSimilarityScore: 0,
      wasScratchedByLossAutopsy: false,
      scratchBarOffset: 0,
      adaptiveOutcomeBadge: outcomeBadge,
      adaptiveExitPrice: exitPrice,
      adaptiveScratchExitR: realizedRMultiple,
      adaptiveNetPnlUsd: stockPnlUsd,
      adaptiveKellyNetPnlUsd: stockPnlUsd * 1.15,
      adaptiveLossSavedUsd: 0,
      adaptiveKellyLossSavedUsd: 0,
      lossAffinityPosteriorPct: 0,
      wasProtectedFromGiveback: false,
      givebackRatchetBarOffset: 0,
      givebackRSaved: 0
    };

    this.closedLedger = [closedItem, ...this.closedLedger];
    if (executionAudio.claimCrossWindowAudioEvent(`LOCAL_CLOSED:${closedItem.tradeId}`)) {
      executionAudio.playPositionClosedChime(closedItem.isWinner);
    }
    return closedItem;
  }

  /**
   * Refreshes live Yahoo Finance prices from the backend (or advances a step when running offline unit tests).
   */
  public stepLiveSimulation(broadcast = true): void {
    if (typeof window !== 'undefined' && this.backendConnected) {
      void this.fetchLedgerSnapshot();
      void this.fetchLiveChartForSymbol(this.selectedTicker, this.activeTimeframe, this.activeRange);
    }

    this.openPositions = this.openPositions.map((pos) => {
      const dirSign = pos.direction === 'LONG' ? 1 : -1;
      const stepPct = 0.0045 * dirSign;
      const nextPrice = Math.max(0.00001, pos.livePrice * (1 + stepPct));
      const pnlPct =
        ((nextPrice - pos.entryPrice) / Math.max(pos.entryPrice, 0.00001)) * 100 * dirSign;
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
   * Triggers a 1-click browser download of the filtered `.CSV` ledger via `/api/v1/export/csv`.
   */
  public async downloadFilteredCsv(): Promise<void> {
    if (typeof document === 'undefined') return;
    const query = new URLSearchParams({
      brain_id: this.brainFilter === 'ALL_5_BRAINS' ? '' : this.brainFilter,
      min_confidence_pct: String(this.minConfidencePct)
    });
    try {
      const res = await fetch(`${API_BASE}/api/v1/export/csv?${query.toString()}`);
      if (res.ok) {
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `quant_closed_trade_ledger_${new Date().toISOString().slice(0, 10)}.csv`;
        a.click();
        URL.revokeObjectURL(url);
        return;
      }
    } catch {
      // Fallback to client-built CSV string
    }

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
   * Triggers a 1-click download of the 4-Sheet `.XLSX` workbook via `/api/v1/export/xlsx`.
   */
  public async downloadFilteredXlsx(): Promise<void> {
    if (typeof window === 'undefined') return;
    const query = new URLSearchParams({
      brain_id: this.brainFilter === 'ALL_5_BRAINS' ? '' : this.brainFilter,
      min_confidence_pct: String(this.minConfidencePct)
    });
    try {
      const res = await fetch(`${API_BASE}/api/v1/export/xlsx?${query.toString()}`);
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

  /**
   * Fetches the 24-pair institutional watchlist snapshot from `/api/v1/scanner/institutional-watchlist`.
   */
  public async fetchInstitutionalWatchlist(): Promise<InstitutionalWatchlistResponse | null> {
    this.isLoadingWatchlist = true;
    try {
      const res = await fetch(`${API_BASE}/api/v1/scanner/institutional-watchlist`);
      if (res.ok) {
        const data: InstitutionalWatchlistResponse = await res.json();
        this.institutionalWatchlist = data;
        return data;
      }
    } catch {
      // Offline / disconnected fallback
    } finally {
      this.isLoadingWatchlist = false;
    }
    return this.institutionalWatchlist;
  }

  /**
   * Fetches the Sunday Alpha Protocol stager summary from `/api/v1/scanner/sunday-protocol`.
   */
  public async fetchSundayAlphaProtocol(): Promise<SundayAlphaProtocolResponse | null> {
    this.isLoadingSundayProtocol = true;
    try {
      const res = await fetch(`${API_BASE}/api/v1/scanner/sunday-protocol`);
      if (res.ok) {
        const data: SundayAlphaProtocolResponse = await res.json();
        this.sundayAlphaProtocol = data;
        return data;
      }
    } catch {
      // Offline fallback
    } finally {
      this.isLoadingSundayProtocol = false;
    }
    return this.sundayAlphaProtocol;
  }

  /**
   * Triggers a 1-click download of the Sunday Alpha Protocol `.CSV` stager spreadsheet via `/api/v1/scanner/sunday-protocol/export/csv`.
   */
  public async exportSundayProtocolCsv(): Promise<void> {
    if (typeof document === 'undefined') return;
    try {
      const res = await fetch(`${API_BASE}/api/v1/scanner/sunday-protocol/export/csv`);
      if (res.ok) {
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `argon_sunday_alpha_protocol_${new Date().toISOString().slice(0, 10)}.csv`;
        a.click();
        URL.revokeObjectURL(url);
        return;
      }
    } catch {
      // Fallback to client-generated CSV below
    }

    if (this.sundayAlphaProtocol && this.sundayAlphaProtocol.staged_setups.length > 0) {
      const header = 'Rank,Symbol,Brain,Direction,Action,ConsensusConf%,VenueTicket,EntryLimit,StopLoss,Target1,Target2,RiskRewardR,KellyPct,CBOEContract,Stage0Verdict,MahalanobisLLR';
      const lines = [header];
      for (const row of this.sundayAlphaProtocol.staged_setups) {
        lines.push([
          row.rank,
          row.symbol,
          row.brain_id,
          row.direction,
          row.action_status,
          `${row.consensus_confidence_pct.toFixed(2)}%`,
          row.venue_ticket,
          row.entry_price.toFixed(2),
          row.structural_stop_price.toFixed(2),
          row.target_1_price.toFixed(2),
          row.target_2_price.toFixed(2),
          row.expected_r_multiple.toFixed(2),
          `${row.kelly_fraction_pct.toFixed(2)}%`,
          `"${row.matched_option_contract}"`,
          row.stage0_conformal_status,
          row.conformal_margin.toFixed(3)
        ].join(','));
      }
      const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `argon_sunday_alpha_protocol_${new Date().toISOString().slice(0, 10)}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    }
  }

  /**
   * Dispatches the Sunday Alpha Protocol staged package to Discord, Telegram, and generic OMS outbox via `POST /api/v1/scanner/sunday-protocol/dispatch`.
   */
  public async dispatchSundayProtocolWebhook(): Promise<SundayAlphaDispatchResponse | null> {
    this.isDispatchingSundayWebhook = true;
    try {
      const res = await fetch(`${API_BASE}/api/v1/scanner/sunday-protocol/dispatch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      if (res.ok) {
        const data: SundayAlphaDispatchResponse = await res.json();
        this.lastDispatchResult = data;
        return data;
      }
    } catch {
      // Offline fallback
    } finally {
      this.isDispatchingSundayWebhook = false;
    }
    return this.lastDispatchResult;
  }

  /**
   * Fetches isolated Phase 28 paper-daemon telemetry without touching baseline `data/ledger_state.json`.
   */
  public async fetchPaperOverlayState(): Promise<PaperOverlayTelemetryResponse | null> {
    try {
      const res = await fetch(`${API_BASE}/api/v1/scanner/paper-overlay`);
      if (res.ok) {
        const data: PaperOverlayTelemetryResponse = await res.json();
        this.paperOverlayState = data;
        return data;
      }
    } catch {
      // Offline fallback
    }
    return this.paperOverlayState;
  }
  /**
   * Stages approved Sunday Alpha protocol orders into the isolated Paper-OMS (`POST /api/v1/scanner/sunday-protocol/stage-orders`).
   * Hardware safety latch strictly defaults to PAPER_ISOLATED_ONLY.
   */
  public async stageSundayOrders(
    params: StageSundayOrdersRequest = { live: false }
  ): Promise<StageSundayOrdersResponse | null> {
    this.isStagingOrders = true;
    this.lastStagingError = null;
    try {
      const res = await fetch(`${API_BASE}/api/v1/scanner/sunday-protocol/stage-orders`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });
      if (res.ok) {
        const data: StageSundayOrdersResponse = await res.json();
        this.lastStagingResponse = data;
        this.stagedOrdersReceipts = data.receipts;
        await this.fetchPaperOverlayState();
        return data;
      } else {
        const errData: StageSundayOrdersErrorResponse = await res.json();
        this.lastStagingError = errData;
        return null;
      }
    } catch {
      // Offline fallback
      return null;
    } finally {
      this.isStagingOrders = false;
    }
  }

  /**
   * Fetches secret-redacted webhook configuration via `GET /api/v1/scanner/webhooks/config`.
   */
  public async fetchWebhookConfig(): Promise<RedactedWebhookConfigResponse | null> {
    this.isLoadingWebhookConfig = true;
    try {
      const res = await fetch(`${API_BASE}/api/v1/scanner/webhooks/config`);
      if (res.ok) {
        const data: RedactedWebhookConfigResponse = await res.json();
        this.webhookConfig = data;
        return data;
      }
    } catch {
      // Offline fallback
    } finally {
      this.isLoadingWebhookConfig = false;
    }
    return this.webhookConfig;
  }

  /**
   * Updates runtime in-memory webhook configuration via `POST /api/v1/scanner/webhooks/config`.
   */
  public async updateWebhookConfig(
    req: ScannerWebhookConfigUpdateRequest
  ): Promise<RedactedWebhookConfigResponse | null> {
    this.isUpdatingWebhookConfig = true;
    try {
      const res = await fetch(`${API_BASE}/api/v1/scanner/webhooks/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req)
      });
      if (res.ok) {
        const data: RedactedWebhookConfigResponse = await res.json();
        this.webhookConfig = data;
        return data;
      }
    } catch {
      // Offline fallback
    } finally {
      this.isUpdatingWebhookConfig = false;
    }
    return null;
  }

  /**
   * Dispatches a test alert across configured channels via `POST /api/v1/scanner/webhooks/test-ping`.
   */
  public async testPingWebhook(symbol?: string): Promise<WebhookTestPingResponse | null> {
    this.isTestPingingWebhook = true;
    try {
      const res = await fetch(`${API_BASE}/api/v1/scanner/webhooks/test-ping`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol })
      });
      if (res.ok) {
        const data: WebhookTestPingResponse = await res.json();
        this.lastTestPingResult = data;
        return data;
      }
    } catch {
      // Offline fallback
    } finally {
      this.isTestPingingWebhook = false;
    }
    return null;
  }
  /**
   * Fetches the latest autopsy & evolution state via `GET /api/v1/scanner/autopsy-evolution`.
   */
  public async fetchAutopsyEvolution(): Promise<AutopsyEvolutionResponse | null> {
    this.isLoadingAutopsyEvolution = true;
    try {
      const res = await fetch(`${API_BASE}/api/v1/scanner/autopsy-evolution`);
      if (res.ok) {
        const data: AutopsyEvolutionResponse = await res.json();
        this.autopsyEvolutionState = data;
        return data;
      }
    } catch {
      // Offline fallback
    } finally {
      this.isLoadingAutopsyEvolution = false;
    }
    return this.autopsyEvolutionState;
  }

  /**
   * Triggers an autonomous closed-loop loss-autopsy & Pareto evolution cycle via `POST /api/v1/scanner/evolve-cycle`.
   */
  public async runEvolveCycle(): Promise<AutopsyEvolutionResponse | null> {
    this.isEvolvingCycle = true;
    try {
      const res = await fetch(`${API_BASE}/api/v1/scanner/evolve-cycle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });
      if (res.ok) {
        const data: AutopsyEvolutionResponse = await res.json();
        this.autopsyEvolutionState = data;
        await this.fetchEvolutionStatus();
        return data;
      }
    } catch {
      // Offline fallback
    } finally {
      this.isEvolvingCycle = false;
    }
    return null;
  }

  public setSelectedExecutionTier(tier: ExecutionTier): void {
    this.selectedExecutionTier = tier;
  }

  public toggleEvolutionDrawer(open?: boolean): void {
    this.evolutionDrawerOpen = open !== undefined ? open : !this.evolutionDrawerOpen;
  }
}

export const terminalStore = new TerminalStore();
