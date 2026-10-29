"""
Isolated Goal-Horizon Configurations & Reward Functions for the 5 Selectable AI Brains.

Zero lagging human indicators are permitted. Each Brain uses the exact same Open-Minded
4-stream discovery engine, differing strictly by:
1. Temporal Resolution Lens (`Tick`, `1m`, `3m`, `5m`, `15m`, `1H`, `4H`, `1D`, `1W`, `1M`)
2. Target Payoff Horizon (`Minutes` to `3+ Years`)
3. Universe & Price Tier Defaults (`$0.00001` sub-penny to `$10,000+`)
4. Horizon-Specific Reward Objective Function
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BrainHorizonConfig:
    brain_id: str
    display_name: str
    temporal_lenses: tuple[str, ...]
    primary_timeframe: str
    target_horizon_label: str
    min_hold_minutes: float
    max_hold_minutes: float
    flatten_by_market_close: bool
    default_min_price_usd: float
    default_max_price_usd: float
    default_index_baskets: tuple[str, ...]
    supports_options_matching: bool
    preferred_option_dte_range: tuple[int, int]
    min_confidence_threshold_pct: float
    dilution_penalty_multiplier: float


ALL_BRAIN_CONFIGS: dict[str, BrainHorizonConfig] = {
    "BRAIN-01": BrainHorizonConfig(
        brain_id="BRAIN-01",
        display_name="Penny & Sub-Penny Anomaly Algo",
        temporal_lenses=("Tick", "1m", "3m", "5m", "1D"),
        primary_timeframe="1m",
        target_horizon_label="Minutes to Multi-Day Parabolic Runs",
        min_hold_minutes=3.0,
        max_hold_minutes=2880.0,
        flatten_by_market_close=False,
        default_min_price_usd=0.00001,
        default_max_price_usd=5.00000,
        default_index_baskets=("OTC_PENNY", "RUSSELL_2000", "FULL_MARKET"),
        supports_options_matching=True,
        preferred_option_dte_range=(7, 45),
        min_confidence_threshold_pct=72.0,
        dilution_penalty_multiplier=3.5,
    ),
    "BRAIN-02": BrainHorizonConfig(
        brain_id="BRAIN-02",
        display_name="Intraday Day-Trading Algo",
        temporal_lenses=("Tick", "1m", "5m", "15m"),
        primary_timeframe="5m",
        target_horizon_label="5 Minutes to 6.5 Hours (Flat by 4:00 PM EST Close)",
        min_hold_minutes=5.0,
        max_hold_minutes=390.0,
        flatten_by_market_close=True,
        default_min_price_usd=5.00000,
        default_max_price_usd=10000.00000,
        default_index_baskets=("NASDAQ_100", "SP_500", "RUSSELL_2000"),
        supports_options_matching=True,
        preferred_option_dte_range=(0, 7),
        min_confidence_threshold_pct=70.0,
        dilution_penalty_multiplier=1.0,
    ),
    "BRAIN-03": BrainHorizonConfig(
        brain_id="BRAIN-03",
        display_name="Short-Term Swing Algo",
        temporal_lenses=("15m", "1H", "4H", "1D"),
        primary_timeframe="1H",
        target_horizon_label="1 to 5 Trading Days",
        min_hold_minutes=390.0,
        max_hold_minutes=5 * 1440.0,
        flatten_by_market_close=False,
        default_min_price_usd=10.00000,
        default_max_price_usd=10000.00000,
        default_index_baskets=("SP_500", "NASDAQ_100", "RUSSELL_2000"),
        supports_options_matching=True,
        preferred_option_dte_range=(7, 21),
        min_confidence_threshold_pct=71.0,
        dilution_penalty_multiplier=1.2,
    ),
    "BRAIN-04": BrainHorizonConfig(
        brain_id="BRAIN-04",
        display_name="Core Swing Trading Algo",
        temporal_lenses=("4H", "1D", "1W"),
        primary_timeframe="4H",
        target_horizon_label="1 to 8 Weeks",
        min_hold_minutes=5 * 1440.0,
        max_hold_minutes=56 * 1440.0,
        flatten_by_market_close=False,
        default_min_price_usd=15.00000,
        default_max_price_usd=10000.00000,
        default_index_baskets=("SP_500", "NASDAQ_100", "DOW_30"),
        supports_options_matching=True,
        preferred_option_dte_range=(21, 90),
        min_confidence_threshold_pct=73.0,
        dilution_penalty_multiplier=1.2,
    ),
    "BRAIN-05": BrainHorizonConfig(
        brain_id="BRAIN-05",
        display_name="Long-Term Portfolio & Wealth Algo",
        temporal_lenses=("1D", "1W", "1M"),
        primary_timeframe="1D",
        target_horizon_label="6 Months to 3+ Years",
        min_hold_minutes=180 * 1440.0,
        max_hold_minutes=1095 * 1440.0,
        flatten_by_market_close=False,
        default_min_price_usd=10.00000,
        default_max_price_usd=100000.00000,
        default_index_baskets=("SP_500", "NASDAQ_100", "DOW_30"),
        supports_options_matching=True,
        preferred_option_dte_range=(240, 730),
        min_confidence_threshold_pct=75.0,
        dilution_penalty_multiplier=1.5,
    ),
}


def get_brain_config(brain_id: str) -> BrainHorizonConfig:
    if brain_id not in ALL_BRAIN_CONFIGS:
        raise KeyError(f"Unknown brain_id '{brain_id}'. Expected one of {list(ALL_BRAIN_CONFIGS)}")
    return ALL_BRAIN_CONFIGS[brain_id]


def compute_brain_reward_objective(
    brain_id: str,
    realized_r_multiple: float,
    mfe_pct: float,
    mae_pct: float,
    hold_duration_minutes: float,
    dilution_hazard: float = 0.0,
) -> float:
    """
    Computes the isolated self-supervised reward score for a completed or simulated trace:
    - Rewards asymmetric R-multiple expansion and clean MFE/MAE excursion ratio.
    - Penalizes holding outside the Brain's target temporal horizon window.
    - Heavily penalizes dilution traps in `BRAIN-01` (`dilution_penalty_multiplier = 3.5`).
    """
    cfg = get_brain_config(brain_id)
    excursion_efficiency = (mfe_pct + 0.1) / max(mae_pct + 0.25, 0.25)
    base_reward = realized_r_multiple * (1.0 + 0.20 * math.tanh(excursion_efficiency - 1.5))

    # Horizon alignment factor
    if hold_duration_minutes < cfg.min_hold_minutes:
        horizon_factor = 0.75
    elif hold_duration_minutes > cfg.max_hold_minutes:
        horizon_factor = 0.65 if not cfg.flatten_by_market_close else 0.35
    else:
        horizon_factor = 1.15

    dilution_penalty = cfg.dilution_penalty_multiplier * max(dilution_hazard, 0.0)
    return float(base_reward * horizon_factor - dilution_penalty)

