//! Two-Stage Full-Market Scanner Funnel (`Stage 1 Wide-Net Snapshot` -> `Stage 2 Deep WebSocket Promotion`).
//!
//! Evaluates ~11,000+ US Equities & OTC tickers across user-customizable price bounds (`$0.00001` to `$10,000+`),
//! index baskets (`S&P 500`, `Nasdaq 100`, `Russell 2000`, `Dow 30`, `OTC/Penny`), and liquidity floors.
//! Strictly bans lagging human indicators; ranks candidates exclusively by raw physical anomalies.

use std::collections::HashSet;

use raw_fingerprint_core::SubPennyPrice;
use serde::{Deserialize, Serialize};

use crate::types::TickerSnapshot;

/// Selectable Market & Index Universe Baskets (Mix & Match).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IndexBasketFilter {
    #[serde(rename = "FULL_MARKET")]
    FullMarketAutoScan,
    #[serde(rename = "SP_500")]
    Sp500,
    #[serde(rename = "NASDAQ_100")]
    Nasdaq100,
    #[serde(rename = "RUSSELL_2000")]
    Russell2000,
    #[serde(rename = "DOW_30")]
    Dow30,
    #[serde(rename = "OTC_PENNY")]
    OtcPenny,
    #[serde(rename = "CUSTOM_WATCHLIST")]
    CustomWatchlist,
}

/// Asset capability filter toggles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AssetCapabilityToggle {
    #[serde(rename = "ALL_EQUITIES")]
    AllEquities,
    #[serde(rename = "OPTIONABLE_ONLY")]
    OptionableOnly,
    #[serde(rename = "HIGH_SHORT_FLOAT")]
    HighShortFloatOnly,
    #[serde(rename = "LOW_FLOAT")]
    LowFloatOnly,
}

/// User-customizable scanner configuration per AI Brain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UniverseScannerConfig {
    pub active_baskets: HashSet<IndexBasketFilter>,
    pub custom_watchlist: HashSet<String>,
    /// Minimum price with 5-decimal precision (`$0.00001` default minimum).
    pub min_price: SubPennyPrice,
    /// Maximum price with 5-decimal precision (`$100,000.00000` default ceiling).
    pub max_price: SubPennyPrice,
    pub min_daily_share_volume: u64,
    pub min_daily_dollar_turnover: f64,
    pub capability_toggle: AssetCapabilityToggle,
    /// Maximum number of top waking tickers promoted into Stage 2 deep WebSocket evaluation.
    pub stage2_promotion_limit: usize,
}

impl Default for UniverseScannerConfig {
    fn default() -> Self {
        let mut active_baskets = HashSet::new();
        active_baskets.insert(IndexBasketFilter::FullMarketAutoScan);
        Self {
            active_baskets,
            custom_watchlist: HashSet::new(),
            min_price: SubPennyPrice::MIN_VALID, // $0.00001
            max_price: SubPennyPrice::from_scaled_units(1_000_000_000_000)
                .unwrap_or(SubPennyPrice::MIN_VALID), // $10,000,000.00000
            min_daily_share_volume: 100_000,
            min_daily_dollar_turnover: 25_000.0,
            capability_toggle: AssetCapabilityToggle::AllEquities,
            stage2_promotion_limit: 50,
        }
    }
}

/// Stage-2 promoted candidate scored by pure raw physical anomalies (volume surge, float rotation, range displacement).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromotedScannerCandidate {
    pub snapshot: TickerSnapshot,
    /// Raw relative volume anomaly ratio (`day_volume / avg_daily_volume_20d`).
    pub raw_rvol_ratio: f64,
    /// Intraday float rotation ratio (`day_volume / public_float_shares`).
    pub float_rotation_ratio: f64,
    /// Scale-independent intraday log displacement $\ln(\text{last} / \text{prev\_close})$.
    pub intraday_log_displacement: f64,
    /// Composite raw physical wake-up score used to rank Stage-2 promotion priority.
    pub composite_anomaly_score: f64,
}

/// Two-Stage Full-Market Scanner Funnel.
#[derive(Debug, Clone, Default)]
pub struct TwoStageMarketFunnel;

impl TwoStageMarketFunnel {
    #[inline]
    pub const fn new() -> Self {
        Self
    }

    /// Executes Stage 1 (Wide-Net Snapshot Filtering across 11,000+ tickers) and
    /// Stage 2 (Deep Real-Time WebSocket Promotion Ranking).
    pub fn execute_funnel(
        &self,
        snapshots: &[TickerSnapshot],
        config: &UniverseScannerConfig,
    ) -> Vec<PromotedScannerCandidate> {
        let full_market = config
            .active_baskets
            .contains(&IndexBasketFilter::FullMarketAutoScan);

        let mut candidates = Vec::with_capacity(config.stage2_promotion_limit.min(snapshots.len()));

        for snap in snapshots {
            // 1. Price Range Filter (5-decimal $0.00001 precision)
            if snap.last_price < config.min_price || snap.last_price > config.max_price {
                continue;
            }

            // 2. Liquidity & Volume Floor Filter
            if snap.day_volume < config.min_daily_share_volume
                || snap.day_dollar_turnover < config.min_daily_dollar_turnover
            {
                continue;
            }

            // 3. Asset Capability Filter
            match config.capability_toggle {
                AssetCapabilityToggle::AllEquities => {}
                AssetCapabilityToggle::OptionableOnly => {
                    if !snap.is_optionable {
                        continue;
                    }
                }
                AssetCapabilityToggle::HighShortFloatOnly => {
                    let short_pct = (snap.short_interest_shares as f64)
                        / (snap.public_float_shares.max(1) as f64);
                    if short_pct < 0.15 {
                        continue;
                    }
                }
                AssetCapabilityToggle::LowFloatOnly => {
                    if snap.public_float_shares > 25_000_000 {
                        continue;
                    }
                }
            }

            // 4. Universe / Index Basket Filter
            if !full_market {
                let matched_index = match snap.exchange_index.as_str() {
                    "SP_500" => config.active_baskets.contains(&IndexBasketFilter::Sp500),
                    "NASDAQ_100" => config
                        .active_baskets
                        .contains(&IndexBasketFilter::Nasdaq100),
                    "RUSSELL_2000" => config
                        .active_baskets
                        .contains(&IndexBasketFilter::Russell2000),
                    "DOW_30" => config.active_baskets.contains(&IndexBasketFilter::Dow30),
                    "OTC_PENNY" => config.active_baskets.contains(&IndexBasketFilter::OtcPenny),
                    _ => false,
                };
                let matched_custom = config
                    .active_baskets
                    .contains(&IndexBasketFilter::CustomWatchlist)
                    && config.custom_watchlist.contains(&snap.symbol);

                if !matched_index && !matched_custom {
                    continue;
                }
            }

            // Stage 2 Ranking Physics (100% Raw Market Evidence — Zero Lagging Indicators)
            let rvol = (snap.day_volume as f64) / (snap.avg_daily_volume_20d.max(1) as f64);
            let float_rot = (snap.day_volume as f64) / (snap.public_float_shares.max(1) as f64);
            let log_disp = snap.last_price.log_ratio(snap.prev_close);
            let short_ratio =
                (snap.short_interest_shares as f64) / (snap.public_float_shares.max(1) as f64);

            let composite_anomaly_score = rvol.ln_1p() * 0.40
                + float_rot.ln_1p() * 0.25
                + log_disp.abs() * 3.50
                + short_ratio * 0.60;

            candidates.push(PromotedScannerCandidate {
                snapshot: snap.clone(),
                raw_rvol_ratio: rvol,
                float_rotation_ratio: float_rot,
                intraday_log_displacement: log_disp,
                composite_anomaly_score,
            });
        }

        // Sort descending by composite raw anomaly score and truncate to Stage-2 promotion limit
        candidates.sort_by(|a, b| {
            b.composite_anomaly_score
                .partial_cmp(&a.composite_anomaly_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(config.stage2_promotion_limit);
        candidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::MultiFeedRouter;
    use crate::adapters::{AlpacaFreeAdapter, HistoricalReplayAdapter, ReplaySpeed};
    use std::str::FromStr;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_two_stage_11000_ticker_funnel_and_sub_penny_filter() {
        let replay = Arc::new(HistoricalReplayAdapter::new(ReplaySpeed::MaxSpeed, 11_000));
        let router = MultiFeedRouter::new(2048);
        router.register_adapter(replay, true);
        router.register_adapter(Arc::new(AlpacaFreeAdapter::new(None)), false);

        let all_snapshots = router
            .fetch_universe_snapshots_with_failover(None)
            .await
            .expect("fetch 11,000 snapshots");
        assert_eq!(all_snapshots.len(), 11_000);

        let funnel = TwoStageMarketFunnel::new();

        // Filter specifically for BRAIN-01 Penny & Sub-Penny range ($0.00001 to $5.00000)
        let penny_cfg = UniverseScannerConfig {
            min_price: SubPennyPrice::from_str("$0.00001").expect("$0.00001"),
            max_price: SubPennyPrice::from_str("$5.00000").expect("$5.00000"),
            min_daily_dollar_turnover: 100.0,
            stage2_promotion_limit: 25,
            ..UniverseScannerConfig::default()
        };

        let promoted = funnel.execute_funnel(&all_snapshots, &penny_cfg);
        assert_eq!(promoted.len(), 25);
        for item in &promoted {
            assert!(item.snapshot.last_price >= penny_cfg.min_price);
            assert!(item.snapshot.last_price <= penny_cfg.max_price);
        }

        // Verify hot-swapping primary feed to alpaca_free works seamlessly
        router
            .hot_swap_primary("alpaca_free")
            .expect("hot-swap to alpaca_free");
        let bars = router
            .fetch_historical_bars_with_failover("NVDA", crate::types::Timeframe::M5, 64)
            .await
            .expect("fetch bars after hot-swap");
        assert_eq!(bars.len(), 64);
    }
}
