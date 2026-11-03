"""
Speed 2 — Nightly 4:15 PM EST Post-Market Autopsy (`speed2_nightly_autopsy.py`).

1. Grades every live signal fired that day (Realized R-Multiple, MFE %, MAE %, Hold Duration,
   Stock & Option Net P&L, and Brain-specific Reward Objective).
2. Hard-Negative Mining: Contrasts stopped-out bull/bear traps against winning traces to extract
   deceptive raw signatures and penalize them in the Brain's readout weights.
3. Missed-Move Discovery: Scans the day's top market gainers/losers across the universe that the
   AI missed, rewinds their 4-stream charts prior to the move, and trains on their pre-move traces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..brains.brain_configs import compute_brain_reward_objective
from ..brains.multi_brain_orchestrator import MultiBrainOrchestrator
from ..world_model.rust_bridge import (
    RawBarInput,
    RawMarketInternalsInput,
    RawOptionsInput,
    RawShortFloatInput,
)


@dataclass(frozen=True)
class ClosedSignalRecord:
    trade_id: str
    brain_id: str
    symbol: str
    direction: str
    outcome_badge: str
    realized_r_multiple: float
    mfe_pct: float
    mae_pct: float
    hold_duration_minutes: float
    net_stock_pnl_usd: float
    net_option_pnl_usd: float
    pre_entry_bars: Sequence[RawBarInput]
    options_snapshot: RawOptionsInput
    short_float_snapshot: RawShortFloatInput
    internals_snapshot: RawMarketInternalsInput


@dataclass(frozen=True)
class MissedMarketMoverRecord:
    symbol: str
    target_brain_id: str
    direction: str
    session_move_pct: float
    pre_move_bars: Sequence[RawBarInput]
    options_snapshot: RawOptionsInput
    short_float_snapshot: RawShortFloatInput
    internals_snapshot: RawMarketInternalsInput


@dataclass(frozen=True)
class NightlyAutopsyReport:
    brain_id: str
    signals_graded_count: int
    winners_count: int
    losers_count: int
    hard_negatives_mined: int
    missed_moves_discovered: int
    mean_reward_score: float
    total_net_pnl_usd: float
    updated_effective_weights: list[float]


class Speed2NightlyAutopsyEngine:
    """
    Executes the 4:15 PM EST Post-Market Autopsy across all 5 AI Brains.
    """

    def __init__(self, orchestrator: MultiBrainOrchestrator) -> None:
        self.orchestrator = orchestrator

    def run_nightly_autopsy_for_brain(
        self,
        brain_id: str,
        closed_signals: Sequence[ClosedSignalRecord],
        missed_movers: Sequence[MissedMarketMoverRecord],
    ) -> NightlyAutopsyReport:
        backbone = self.orchestrator.backbones[brain_id]
        bridge = self.orchestrator.bridge

        brain_signals = [s for s in closed_signals if s.brain_id == brain_id]
        brain_missed = [m for m in missed_movers if m.target_brain_id == brain_id]

        winners = 0
        losers = 0
        hard_negatives = 0
        reward_sum = 0.0
        total_net_pnl = 0.0

        for sig in brain_signals:
            total_net_pnl += sig.net_stock_pnl_usd + sig.net_option_pnl_usd
            tensor_out = bridge.build_sequence_tensor(
                sig.pre_entry_bars,
                sig.options_snapshot,
                sig.short_float_snapshot,
                sig.internals_snapshot,
            )
            last_row = tensor_out.matrix_2d[-1]
            dilution_hazard = last_row[25]

            reward = compute_brain_reward_objective(
                brain_id=brain_id,
                realized_r_multiple=sig.realized_r_multiple,
                mfe_pct=sig.mfe_pct,
                mae_pct=sig.mae_pct,
                hold_duration_minutes=sig.hold_duration_minutes,
                dilution_hazard=dilution_hazard,
            )
            reward_sum += reward

            if sig.realized_r_multiple > 0.0:
                winners += 1
            else:
                losers += 1
                # Hard-Negative Mining on stopped-out bull/bear traps
                if "STOPPED OUT" in sig.outcome_badge or sig.realized_r_multiple <= -0.8:
                    backbone.apply_hard_negative_update(last_row, learning_rate=0.045)
                    hard_negatives += 1

        # Missed-Move Discovery: rewind pre-move bars of top market movers and reinforce trace
        for missed in brain_missed:
            tensor_out = bridge.build_sequence_tensor(
                missed.pre_move_bars,
                missed.options_snapshot,
                missed.short_float_snapshot,
                missed.internals_snapshot,
            )
            last_row = tensor_out.matrix_2d[-1]
            dir_sign = 1.0 if missed.direction.upper() == "LONG" else -1.0
            backbone.apply_missed_move_update(
                last_row, direction_sign=dir_sign, learning_rate=0.035
            )

        mean_reward = reward_sum / max(len(brain_signals), 1)

        return NightlyAutopsyReport(
            brain_id=brain_id,
            signals_graded_count=len(brain_signals),
            winners_count=winners,
            losers_count=losers,
            hard_negatives_mined=hard_negatives,
            missed_moves_discovered=len(brain_missed),
            mean_reward_score=mean_reward,
            total_net_pnl_usd=total_net_pnl,
            updated_effective_weights=backbone.effective_weights(),
        )

