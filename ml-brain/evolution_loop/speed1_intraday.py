"""
Speed 1 — Intra-Day Real-Time Tick-by-Tick Adaptation (`speed1_intraday.py`).

1. Evaluates every open position in `POSITIONS OPENED` bar-by-bar via the shared Rust
   `RawFingerprintEngine` (`evaluate_live_trace_health`).
2. Triggers an immediate `LIVE DYNAMIC EXIT` if opposing stopping-volume absorption or
   tape exhaustion prints before the hard target.
3. Dynamically tightens entry confidence thresholds if live `$TICK`, `$ADD`, and `$VOLD`
   market internals shift into high-whipsaw chop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..brains.brain_configs import get_brain_config
from ..world_model.rust_bridge import (
    RawBarInput,
    RawMarketInternalsInput,
    RustFingerprintBridge,
)


@dataclass(frozen=True)
class ActivePositionState:
    position_id: str
    brain_id: str
    symbol: str
    direction: str
    entry_price: float
    structural_stop_price: float
    target_1_price: float
    target_2_price: float
    peak_favorable_price: float


@dataclass(frozen=True)
class IntradayAdaptationResult:
    position_id: str
    symbol: str
    brain_id: str
    trace_health_badge: str
    updated_structural_stop_price: float
    opposing_absorption_score: float
    current_r_multiple: float
    trigger_dynamic_exit: bool
    adjusted_entry_confidence_threshold_pct: float


class Speed1IntradayAdapter:
    """
    Executes Speed-1 real-time bar-by-bar open position evaluation and dynamic
    market-internals threshold adaptation.
    """

    def __init__(self, bridge: RustFingerprintBridge | None = None) -> None:
        self.bridge = bridge or RustFingerprintBridge()

    def compute_dynamic_confidence_threshold(
        self,
        brain_id: str,
        recent_internals: Sequence[RawMarketInternalsInput],
    ) -> float:
        """
        Inspects recent `$TICK` and `$VOLD` internals to detect high-whipsaw chop.
        Tightens the Brain's minimum confidence threshold by up to +8.5% during chop.
        """
        cfg = get_brain_config(brain_id)
        base_thresh = cfg.min_confidence_threshold_pct
        if not recent_internals:
            return base_thresh

        # Measure sign flips in NYSE $TICK and narrow $VOLD hover around 1.0
        sign_flips = 0
        for i in range(1, len(recent_internals)):
            prev_t = recent_internals[i - 1].nyse_tick
            cur_t = recent_internals[i].nyse_tick
            if (prev_t >= 0 > cur_t) or (prev_t < 0 <= cur_t):
                sign_flips += 1

        latest = recent_internals[-1]
        is_narrow_vold = 0.78 <= latest.vold_up_down_ratio <= 1.25
        is_low_breadth = abs(latest.add_breadth) < 350
        vix_backwardation = latest.vix_spot > latest.vix_front_month

        chop_penalty = 0.0
        if len(recent_internals) > 1:
            flip_rate = float(sign_flips) / float(len(recent_internals) - 1)
            chop_penalty += flip_rate * 4.5
        if is_narrow_vold and is_low_breadth:
            chop_penalty += 2.5
        if vix_backwardation:
            chop_penalty += 1.5

        return round(min(base_thresh + chop_penalty, 92.0), 2)

    def evaluate_open_position_bar(
        self,
        position: ActivePositionState,
        recent_bars: Sequence[RawBarInput],
        recent_internals: Sequence[RawMarketInternalsInput],
    ) -> IntradayAdaptationResult:
        latest_internals = (
            recent_internals[-1] if recent_internals else RawMarketInternalsInput()
        )
        rust_health = self.bridge.evaluate_live_trace_health(
            direction=position.direction,
            entry_price=position.entry_price,
            structural_stop_price=position.structural_stop_price,
            target_1_price=position.target_1_price,
            target_2_price=position.target_2_price,
            peak_favorable_price=position.peak_favorable_price,
            recent_bars=recent_bars,
            internals=latest_internals,
        )
        adjusted_threshold = self.compute_dynamic_confidence_threshold(
            position.brain_id, recent_internals
        )

        return IntradayAdaptationResult(
            position_id=position.position_id,
            symbol=position.symbol,
            brain_id=position.brain_id,
            trace_health_badge=rust_health.badge,
            updated_structural_stop_price=rust_health.updated_structural_stop_price,
            opposing_absorption_score=rust_health.opposing_absorption_score,
            current_r_multiple=rust_health.current_r_multiple,
            trigger_dynamic_exit=rust_health.should_exit_immediately,
            adjusted_entry_confidence_threshold_pct=adjusted_threshold,
        )

