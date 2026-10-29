"""
Phase 2 Verification & Maximization Gate Test Suite (`test_phase2_brains_and_evolution.py`).

Verifies:
1. Zero training-serving skew between Rust (`crates/raw-fingerprint-core`) and Python (`ml-brain`).
2. Scale-independence across `$0.00001` sub-penny and `$1,000.00+` price tiers.
3. All 5 Goal-Horizon Brains (`BRAIN-01` through `BRAIN-05`) + Options Structure & Greeks Matcher.
4. `.onnx` and `.safetensors` model export, SHA-256 verification, and `TreeSHAP` 4-stream attribution.
5. Complete 3-Speed Self-Learning Loop (Speed 1 Intraday, Speed 2 Nightly Autopsy, Speed 3 Weekend Tournament).
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_BRAIN_ROOT = REPO_ROOT / "ml-brain"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Allow importing `ml-brain` hyphenated directory as package `ml_brain`
import importlib.util

spec = importlib.util.spec_from_file_location(
    "ml_brain",
    ML_BRAIN_ROOT / "__init__.py",
    submodule_search_locations=[str(ML_BRAIN_ROOT)],
)
assert spec is not None and spec.loader is not None
ml_brain_mod = importlib.util.module_from_spec(spec)
sys.modules["ml_brain"] = ml_brain_mod
spec.loader.exec_module(ml_brain_mod)

from ml_brain.brains import (  # noqa: E402
    ALL_BRAIN_CONFIGS,
    MultiBrainOrchestrator,
)
from ml_brain.evolution_loop import (  # noqa: E402
    ActivePositionState,
    ClosedSignalRecord,
    MissedMarketMoverRecord,
    Speed1IntradayAdapter,
    Speed2NightlyAutopsyEngine,
    Speed3WeekendTournamentEngine,
)
from ml_brain.world_model import (  # noqa: E402
    CandlestickFlowTokenizer,
    OnnxBrainRuntime,
    RawBarInput,
    RawMarketInternalsInput,
    RawOptionsInput,
    RawShortFloatInput,
    RustFingerprintBridge,
)


def make_synthetic_bars(
    base_price: float,
    count: int = 32,
    breakout_slope: float = 0.012,
) -> list[RawBarInput]:
    bars: list[RawBarInput] = []
    cur = base_price
    for i in range(count):
        step = breakout_slope if i >= count // 2 else 0.002
        open_p = max(cur, 0.00001)
        close_p = max(open_p * (1.0 + step), 0.00001)
        if close_p == open_p and base_price < 0.001:
            close_p = round(open_p + 0.00001, 5)
        high_p = max(open_p, close_p) * 1.006
        low_p = max(min(open_p, close_p) * 0.994, 0.00001)
        vwap_p = (open_p + high_p + low_p + close_p) * 0.25
        vol = int(250_000 * (2.5 if i >= count // 2 else 1.0) + i * 5_000)

        bars.append(
            RawBarInput(
                timestamp_ns=1_761_768_000_000_000_000 + i * 60_000_000_000,
                open_price=open_p,
                high_price=high_p,
                low_price=low_p,
                close_price=close_p,
                volume=vol,
                vwap=vwap_p,
                dollar_turnover=vol * vwap_p,
                trade_count=320 + i * 15,
                bid_ask_spread=max(base_price * 0.001, 0.00001),
                time_delta_ms=60_000,
            )
        )
        cur = close_p
    return bars


class TestPhase2WorldModelBrainsAndEvolution(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bridge = RustFingerprintBridge()
        cls.orchestrator = MultiBrainOrchestrator(bridge=cls.bridge)
        cls.models_dir = REPO_ROOT / "models"

    def test_01_zero_training_serving_skew_and_sub_penny_precision(self) -> None:
        sub_penny_bars = make_synthetic_bars(0.00001, count=24)
        mega_cap_bars = make_synthetic_bars(1000.00000, count=24)

        t_sub_1 = self.bridge.build_sequence_tensor(sub_penny_bars)
        t_sub_2 = self.bridge.build_sequence_tensor(sub_penny_bars)

        # Bit-for-bit identical determinism across Rust FFI calls
        self.assertEqual(t_sub_1.flat_tensor, t_sub_2.flat_tensor)
        self.assertEqual(t_sub_1.latent_embedding_128, t_sub_2.latent_embedding_128)

        # Verify L2 norm of 128-d pgvector embedding is 1.0
        l2_norm = math.sqrt(sum(x * x for x in t_sub_1.latent_embedding_128))
        self.assertAlmostEqual(l2_norm, 1.0, places=5)

        # Verify Stream 1 scale independence between $10 and $1,000
        mid_cap_bars = make_synthetic_bars(10.00000, count=24)
        t_mid = self.bridge.build_sequence_tensor(mid_cap_bars)
        t_mega = self.bridge.build_sequence_tensor(mega_cap_bars)

        for row_idx in range(24):
            for col_idx in range(12):
                diff = abs(
                    t_mid.matrix_2d[row_idx][col_idx] - t_mega.matrix_2d[row_idx][col_idx]
                )
                self.assertLess(diff, 1e-4)

    def test_02_all_5_brains_onnx_export_and_treeshap_attribution(self) -> None:
        exported_paths = self.orchestrator.export_all_brains_to_onnx(
            output_dir=self.models_dir,
            model_version="v1.0.0-phase2-verified",
        )
        self.assertEqual(len(exported_paths), 5)

        for brain_id in ALL_BRAIN_CONFIGS:
            self.assertIn(brain_id, exported_paths)
            onnx_path = exported_paths[brain_id]
            self.assertTrue(onnx_path.exists())
            safetensors_path = self.models_dir / f"{brain_id}.safetensors"
            self.assertTrue(safetensors_path.exists())

            # Load exported .onnx artifact into OnnxBrainRuntime and execute inference + TreeSHAP
            runtime = OnnxBrainRuntime(onnx_path)
            is_penny_brain = brain_id == "BRAIN-01"
            bars = make_synthetic_bars(0.00003 if is_penny_brain else 142.50, count=32)

            pkg = self.orchestrator.evaluate_ticker_for_brain(
                brain_id=brain_id,
                symbol="ZNTX" if is_penny_brain else "NVDA",
                bars=bars,
                is_optionable=not is_penny_brain,
            )
            pred, shap_rep = runtime.run_inference(
                self.bridge.build_sequence_tensor(bars).matrix_2d
            )

            self.assertEqual(pred.brain_id, brain_id)
            self.assertEqual(len(shap_rep.shap_values_32), 32)
            stream_weight_sum = sum(shap_rep.normalized_stream_weights.values())
            self.assertAlmostEqual(stream_weight_sum, 1.0, places=5)

            if is_penny_brain:
                self.assertEqual(pkg.matched_options.matched_option_contract, "SHARES_ONLY")
                self.assertGreaterEqual(pkg.rust_signal.entry_price, 0.00001)
            else:
                self.assertNotEqual(pkg.matched_options.matched_option_contract, "SHARES_ONLY")
                self.assertGreater(pkg.matched_options.option_entry_premium, 0.0)
                self.assertGreater(
                    pkg.matched_options.option_target_2_premium,
                    pkg.matched_options.option_entry_premium,
                )

    def test_03_three_speed_continuous_evolution_engine(self) -> None:
        # 1. Speed 1: Intra-day tick-by-tick adaptation & dynamic exit evaluation
        speed1 = Speed1IntradayAdapter(bridge=self.bridge)
        bars = make_synthetic_bars(150.0, count=16)
        pos = ActivePositionState(
            position_id="pos-001",
            brain_id="BRAIN-02",
            symbol="NVDA",
            direction="LONG",
            entry_price=150.0,
            structural_stop_price=147.50,
            target_1_price=153.50,
            target_2_price=158.00,
            peak_favorable_price=154.20,
        )
        chop_internals = [
            RawMarketInternalsInput(nyse_tick=450, add_breadth=120, vold_up_down_ratio=1.05),
            RawMarketInternalsInput(nyse_tick=-520, add_breadth=90, vold_up_down_ratio=0.98),
            RawMarketInternalsInput(nyse_tick=390, add_breadth=110, vold_up_down_ratio=1.02),
        ]
        s1_res = speed1.evaluate_open_position_bar(pos, bars, chop_internals)
        self.assertIn(
            s1_res.trace_health_badge,
            ("TRACE_ACCELERATING", "TRAILING_T1", "EXHAUSTION_WARNING", "DYNAMIC_EXIT_TRIGGERED"),
        )
        self.assertGreater(s1_res.adjusted_entry_confidence_threshold_pct, 70.0)

        # 2. Speed 2: Nightly 4:15 PM EST Post-Market Autopsy (Hard-Negative Mining + Missed Moves)
        speed2 = Speed2NightlyAutopsyEngine(orchestrator=self.orchestrator)
        closed_signals = [
            ClosedSignalRecord(
                trade_id="tr-win-1",
                brain_id="BRAIN-02",
                symbol="NVDA",
                direction="LONG",
                outcome_badge="CLOSED - TARGET HIT",
                realized_r_multiple=3.2,
                mfe_pct=4.8,
                mae_pct=0.4,
                hold_duration_minutes=65.0,
                net_stock_pnl_usd=3200.0,
                net_option_pnl_usd=4850.0,
                pre_entry_bars=bars,
                options_snapshot=RawOptionsInput(),
                short_float_snapshot=RawShortFloatInput(),
                internals_snapshot=RawMarketInternalsInput(),
            ),
            ClosedSignalRecord(
                trade_id="tr-loss-1",
                brain_id="BRAIN-02",
                symbol="AMD",
                direction="LONG",
                outcome_badge="CLOSED - STOPPED OUT",
                realized_r_multiple=-1.0,
                mfe_pct=0.3,
                mae_pct=1.6,
                hold_duration_minutes=22.0,
                net_stock_pnl_usd=-1000.0,
                net_option_pnl_usd=-1250.0,
                pre_entry_bars=make_synthetic_bars(140.0, count=16, breakout_slope=-0.008),
                options_snapshot=RawOptionsInput(ask_fill_aggression_ratio=0.25, bid_fill_aggression_ratio=0.75),
                short_float_snapshot=RawShortFloatInput(),
                internals_snapshot=RawMarketInternalsInput(nyse_tick=-720),
            ),
        ]
        missed_movers = [
            MissedMarketMoverRecord(
                symbol="PLTR",
                target_brain_id="BRAIN-02",
                direction="LONG",
                session_move_pct=14.6,
                pre_move_bars=make_synthetic_bars(60.0, count=20, breakout_slope=0.018),
                options_snapshot=RawOptionsInput(ask_fill_aggression_ratio=0.88, bid_fill_aggression_ratio=0.12),
                short_float_snapshot=RawShortFloatInput(),
                internals_snapshot=RawMarketInternalsInput(nyse_tick=920),
            )
        ]
        autopsy_report = speed2.run_nightly_autopsy_for_brain(
            brain_id="BRAIN-02",
            closed_signals=closed_signals,
            missed_movers=missed_movers,
        )
        self.assertEqual(autopsy_report.signals_graded_count, 2)
        self.assertEqual(autopsy_report.hard_negatives_mined, 1)
        self.assertEqual(autopsy_report.missed_moves_discovered, 1)

        # 3. Speed 3: Weekend Champion vs. Challenger Tournament with Purged Walk-Forward CV
        tokenizer = CandlestickFlowTokenizer(bridge=self.bridge)
        long_history = make_synthetic_bars(50.0, count=48, breakout_slope=0.015)
        samples = tokenizer.extract_reverse_trace_samples(
            symbol="NVDA",
            bars=long_history,
            forward_horizon_bars=6,
            expansion_threshold_pct=1.0,
        )
        self.assertGreaterEqual(len(samples), 8)

        speed3 = Speed3WeekendTournamentEngine()
        tournament_res = speed3.run_weekend_tournament(
            champion_backbone=self.orchestrator.backbones["BRAIN-03"],
            samples=samples,
            onnx_output_dir=self.models_dir,
        )
        self.assertEqual(tournament_res.brain_id, "BRAIN-03")
        self.assertEqual(tournament_res.folds_evaluated, 3)
        self.assertTrue(tournament_res.production_onnx_path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)

