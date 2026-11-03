"""
Speed 3 — Weekend Champion vs. Challenger Evolutionary Tournament (`speed3_weekend_tournament.py`).

1. Spawns mutated Challenger brains around the current Champion backbone.
2. Evaluates both Champion and Challenger via Purged & Embargoed Walk-Forward Cross-Validation
   (enforcing explicit purge and embargo intervals between train and test folds so overlapping
   multi-bar hold horizons never leak information).
3. Promotes a Challenger to live `.onnx` production if and only if it strictly beats the Champion
   across Net Profit ($), Win Rate (%), Profit Factor, and Max Drawdown (%).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..world_model.explainer_and_onnx import OnnxBrainArtifactExporter
from ..world_model.sequence_backbone import MambaFlexWorldModelBackbone
from ..world_model.tokenizer import ReverseTraceSample


@dataclass(frozen=True)
class WalkForwardFoldSplit:
    fold_index: int
    train_indices: list[int]
    test_indices: list[int]
    purged_count: int
    embargoed_count: int


@dataclass(frozen=True)
class TournamentEvaluationMetrics:
    model_version: str
    net_profit_usd: float
    win_rate_pct: float
    profit_factor: float
    max_drawdown_pct: float
    trades_evaluated: int


@dataclass(frozen=True)
class WeekendTournamentResult:
    brain_id: str
    champion_metrics: TournamentEvaluationMetrics
    challenger_metrics: TournamentEvaluationMetrics
    folds_evaluated: int
    promoted_challenger: bool
    production_onnx_path: Path


class Speed3WeekendTournamentEngine:
    """
    Executes Purged & Embargoed Walk-Forward Cross-Validation and promotes winning
    Challenger brains into live `.onnx` production artifacts.
    """

    @staticmethod
    def build_purged_embargoed_splits(
        total_samples: int,
        n_folds: int = 3,
        purge_window: int = 2,
        embargo_window: int = 2,
    ) -> list[WalkForwardFoldSplit]:
        if total_samples < 6:
            raise ValueError(f"Need at least 6 samples for walk-forward CV, got {total_samples}")

        fold_size = max(total_samples // (n_folds + 1), 1)
        splits: list[WalkForwardFoldSplit] = []

        for fold_idx in range(n_folds):
            test_start = (fold_idx + 1) * fold_size
            test_end = min(test_start + fold_size, total_samples)
            if test_start >= total_samples:
                break

            purge_start = max(0, test_start - purge_window)
            embargo_end = min(total_samples, test_end + embargo_window)

            train_indices = [
                i
                for i in range(total_samples)
                if i < purge_start or i >= embargo_end
            ]
            test_indices = list(range(test_start, test_end))

            splits.append(
                WalkForwardFoldSplit(
                    fold_index=fold_idx,
                    train_indices=train_indices,
                    test_indices=test_indices,
                    purged_count=test_start - purge_start,
                    embargoed_count=embargo_end - test_end,
                )
            )
        return splits

    @staticmethod
    def evaluate_backbone_on_samples(
        backbone: MambaFlexWorldModelBackbone,
        samples: Sequence[ReverseTraceSample],
        model_version: str,
        risk_per_trade_usd: float = 1000.0,
    ) -> TournamentEvaluationMetrics:
        gross_profit = 0.0
        gross_loss = 0.0
        wins = 0
        total = 0
        equity = 100_000.0
        peak_equity = equity
        max_dd_pct = 0.0

        for sample in samples:
            pred = backbone.selective_ssm_and_flex_attention_forward(
                sample.rust_tensor.matrix_2d
            )
            aligned = pred.direction == sample.direction
            signed_r = (
                abs(sample.realized_r_multiple)
                if aligned and sample.is_true_expansion
                else -0.85
            )
            trade_pnl = signed_r * risk_per_trade_usd
            total += 1
            if trade_pnl > 0.0:
                wins += 1
                gross_profit += trade_pnl
            else:
                gross_loss += abs(trade_pnl)

            equity += trade_pnl
            if equity > peak_equity:
                peak_equity = equity
            dd_pct = ((peak_equity - equity) / max(peak_equity, 1.0)) * 100.0
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

        net_profit = gross_profit - gross_loss
        win_rate = (float(wins) / float(max(total, 1))) * 100.0
        profit_factor = gross_profit / max(gross_loss, 1.0)

        return TournamentEvaluationMetrics(
            model_version=model_version,
            net_profit_usd=round(net_profit, 2),
            win_rate_pct=round(win_rate, 2),
            profit_factor=round(profit_factor, 3),
            max_drawdown_pct=round(max_dd_pct, 3),
            trades_evaluated=total,
        )

    def run_weekend_tournament(
        self,
        champion_backbone: MambaFlexWorldModelBackbone,
        samples: Sequence[ReverseTraceSample],
        onnx_output_dir: Path,
    ) -> WeekendTournamentResult:
        splits = self.build_purged_embargoed_splits(len(samples), n_folds=3)

        # Spawn mutated Challenger brain trained on the Purged Walk-Forward folds
        challenger_backbone = copy.deepcopy(champion_backbone)
        for split in splits:
            for idx in split.train_indices:
                s = samples[idx]
                last_vec = s.rust_tensor.matrix_2d[-1]
                dir_sign = 1.0 if s.direction == "LONG" else -1.0
                if s.is_true_expansion:
                    challenger_backbone.apply_missed_move_update(
                        last_vec, direction_sign=dir_sign, learning_rate=0.06
                    )
                else:
                    challenger_backbone.apply_hard_negative_update(
                        last_vec, learning_rate=0.06
                    )

        # Collect out-of-sample test indices across all walk-forward folds
        oos_samples = [
            samples[idx] for split in splits for idx in split.test_indices
        ]
        if not oos_samples:
            oos_samples = list(samples)

        champ_metrics = self.evaluate_backbone_on_samples(
            champion_backbone, oos_samples, model_version="v1.0.0-champion"
        )
        chall_metrics = self.evaluate_backbone_on_samples(
            challenger_backbone, oos_samples, model_version="v1.1.0-challenger"
        )

        # Strict multi-objective promotion gate:
        # Challenger must beat or equal Champion on Net Profit, Win %, Profit Factor, and Max Drawdown
        promoted = (
            chall_metrics.net_profit_usd >= champ_metrics.net_profit_usd
            and chall_metrics.win_rate_pct >= champ_metrics.win_rate_pct
            and chall_metrics.profit_factor >= champ_metrics.profit_factor
            and chall_metrics.max_drawdown_pct <= champ_metrics.max_drawdown_pct + 1e-6
            and not math.isclose(chall_metrics.net_profit_usd, champ_metrics.net_profit_usd)
        )

        winning_backbone = challenger_backbone if promoted else champion_backbone
        winning_version = (
            chall_metrics.model_version if promoted else champ_metrics.model_version
        )
        onnx_path = OnnxBrainArtifactExporter.export_brain_to_onnx(
            backbone=winning_backbone,
            output_dir=onnx_output_dir,
            model_version=winning_version,
        )

        return WeekendTournamentResult(
            brain_id=champion_backbone.brain_id,
            champion_metrics=champ_metrics,
            challenger_metrics=chall_metrics,
            folds_evaluated=len(splits),
            promoted_challenger=promoted,
            production_onnx_path=onnx_path,
        )

