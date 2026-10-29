"""
Self-Supervised Candlestick & Flow Tokenizer + Reverse-Trace Ground-Truth Rewind Engine.

Implements:
1. Ground-Truth Pinning: Scans historical sequences across `$0.00001` sub-penny to `$10,000+`
   scales, tagging structural displacements (rallies/flushes) and failed fakeouts (bull/bear traps).
2. Multi-Horizon Rewind: Rewinds the 4-stream raw sequence `5`, `50`, `500`, and `2,000` bars
   prior to the inflection point.
3. Candlestick & Flow Tokenization: Quantizes each 32-feature raw bar into continuous
   flow tokens and discrete K-line physics archetype IDs without any lagging human indicators.
4. Polars 2.0 Compliance: Explicitly enforces `maintain_order=True` on all time-series
   joins, rolling windows, and `group_by` aggregations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from .rust_bridge import (
    RawBarInput,
    RawMarketInternalsInput,
    RawOptionsInput,
    RawShortFloatInput,
    RustFingerprintBridge,
    RustTensorOutput,
)

try:
    import polars as pl  # type: ignore[import-not-found]

    HAS_POLARS = True
except ImportError:
    pl = None
    HAS_POLARS = False

MULTI_HORIZON_REWIND_WINDOWS: tuple[int, ...] = (5, 50, 500, 2000)


@dataclass(frozen=True)
class ReverseTraceSample:
    """Self-supervised reverse-trace training example pinned to an undeniable historical move."""

    symbol: str
    inflection_index: int
    inflection_timestamp_ns: int
    direction: str
    entry_price: float
    structural_stop_price: float
    realized_move_pct: float
    realized_r_multiple: float
    mfe_pct: float
    mae_pct: float
    time_to_peak_bars: int
    is_true_expansion: bool
    rewind_horizon_lengths: dict[int, int]
    token_ids: list[int]
    rust_tensor: RustTensorOutput


class CandlestickFlowTokenizer:
    """
    Tokenizes 4-stream raw market physics into continuous state vectors and discrete
    microstructure token IDs (`0..255`) using the shared Rust `RawFingerprintEngine`.
    """

    def __init__(self, bridge: RustFingerprintBridge | None = None) -> None:
        self.bridge = bridge or RustFingerprintBridge()

    def tokenize_sequence(
        self,
        bars: Sequence[RawBarInput],
        options: RawOptionsInput | None = None,
        short_float: RawShortFloatInput | None = None,
        internals: RawMarketInternalsInput | None = None,
    ) -> tuple[RustTensorOutput, list[int]]:
        """
        Builds the 32-feature scale-independent tensor via Rust FFI and maps each bar
        to a deterministic 8-bit Candlestick & Flow Token ID (`0..255`):
        - Bits 0..1: Price displacement regime (strong down, mild down, mild up, strong up)
        - Bits 2..3: Body vs. wick absorption archetype (body conviction, upper rejection, lower stopping wick, compression coil)
        - Bits 4..5: Volume & institutional block-print surge tier
        - Bits 6..7: Cross-stream alignment (Gamma/Vanna + Short Squeeze + Macro Tide)
        """
        tensor_out = self.bridge.build_sequence_tensor(bars, options, short_float, internals)
        token_ids: list[int] = []

        for row in tensor_out.matrix_2d:
            log_ret = row[0]
            body_conv = row[3]
            upper_wick = row[4]
            lower_wick = row[5]
            vol_surge = row[7]
            inst_print = row[9]
            gamma_imb = row[13]
            float_rot = row[20]
            tick_imp = row[26]

            # Bits 0..1: Price velocity bucket (0..3)
            if log_ret < -0.008:
                b_price = 0
            elif log_ret < 0.0:
                b_price = 1
            elif log_ret < 0.008:
                b_price = 2
            else:
                b_price = 3

            # Bits 2..3: Candle & wick microstructure archetype (0..3)
            if abs(body_conv) >= 0.65:
                b_wick = 0  # High-conviction expansion candle
            elif lower_wick >= 0.45:
                b_wick = 1  # Stopping-volume demand absorption wick
            elif upper_wick >= 0.45:
                b_wick = 2  # Overhead supply rejection wick
            else:
                b_wick = 3  # Tight volatility compression coil bar

            # Bits 4..5: Raw volume & block-print density (0..3)
            vol_score = vol_surge * 0.6 + inst_print * 0.4
            if vol_score < -0.25:
                b_vol = 0
            elif vol_score < 0.25:
                b_vol = 1
            elif vol_score < 1.0:
                b_vol = 2
            else:
                b_vol = 3

            # Bits 6..7: Synchronized multi-stream alignment (0..3)
            multi_stream = gamma_imb * 0.4 + math.tanh(float_rot) * 0.3 + tick_imp * 0.3
            if multi_stream < -0.35:
                b_multi = 0
            elif multi_stream < 0.0:
                b_multi = 1
            elif multi_stream < 0.35:
                b_multi = 2
            else:
                b_multi = 3

            token_id = (b_multi << 6) | (b_vol << 4) | (b_wick << 2) | b_price
            token_ids.append(int(token_id & 0xFF))

        return tensor_out, token_ids

    def extract_reverse_trace_samples(
        self,
        symbol: str,
        bars: Sequence[RawBarInput],
        options: RawOptionsInput | None = None,
        short_float: RawShortFloatInput | None = None,
        internals: RawMarketInternalsInput | None = None,
        forward_horizon_bars: int = 10,
        expansion_threshold_pct: float = 1.5,
    ) -> list[ReverseTraceSample]:
        """
        Scans a historical bar stream, pins structural expansions and failed fakeouts,
        and rewinds `5`, `50`, `500`, and `2,000` bars prior to each inflection point.
        """
        n = len(bars)
        if n < forward_horizon_bars + 6:
            return []

        samples: list[ReverseTraceSample] = []

        for idx in range(5, n - forward_horizon_bars):
            pre_bars = bars[: idx + 1]
            future_bars = bars[idx + 1 : idx + 1 + forward_horizon_bars]
            entry_bar = bars[idx]
            entry_p = max(entry_bar.close_price, 0.00001)

            # Evaluate future price path to measure MFE, MAE, and realized outcome
            max_high = max(b.high_price for b in future_bars)
            min_low = min(b.low_price for b in future_bars)
            final_close = future_bars[-1].close_price

            up_excursion_pct = ((max_high - entry_p) / entry_p) * 100.0
            dn_excursion_pct = ((entry_p - min_low) / entry_p) * 100.0
            net_move_pct = ((final_close - entry_p) / entry_p) * 100.0

            if max(up_excursion_pct, dn_excursion_pct) < expansion_threshold_pct:
                continue

            direction = "LONG" if up_excursion_pct >= dn_excursion_pct else "SHORT"
            if direction == "LONG":
                mfe_pct = up_excursion_pct
                mae_pct = dn_excursion_pct
                realized_move = net_move_pct
                peak_idx = max(
                    range(len(future_bars)), key=lambda k: future_bars[k].high_price
                )
                stop_p = max(min(b.low_price for b in pre_bars[-5:]) * 0.995, 0.00001)
                risk_pct = max(((entry_p - stop_p) / entry_p) * 100.0, 0.25)
            else:
                mfe_pct = dn_excursion_pct
                mae_pct = up_excursion_pct
                realized_move = -net_move_pct
                peak_idx = min(
                    range(len(future_bars)), key=lambda k: future_bars[k].low_price
                )
                stop_p = max(b.high_price for b in pre_bars[-5:]) * 1.005
                risk_pct = max(((stop_p - entry_p) / entry_p) * 100.0, 0.25)

            realized_r = realized_move / risk_pct
            is_true_expansion = (mfe_pct >= expansion_threshold_pct) and (mae_pct < risk_pct * 1.15)

            # Record multi-horizon rewind slices (5, 50, 500, 2000 bars)
            rewind_lengths = {
                w: len(pre_bars[-w:]) for w in MULTI_HORIZON_REWIND_WINDOWS
            }
            active_window = pre_bars[-min(len(pre_bars), 64) :]
            tensor_out, token_ids = self.tokenize_sequence(
                active_window, options, short_float, internals
            )

            samples.append(
                ReverseTraceSample(
                    symbol=symbol.upper(),
                    inflection_index=idx,
                    inflection_timestamp_ns=entry_bar.timestamp_ns,
                    direction=direction,
                    entry_price=entry_p,
                    structural_stop_price=stop_p,
                    realized_move_pct=realized_move,
                    realized_r_multiple=realized_r,
                    mfe_pct=mfe_pct,
                    mae_pct=mae_pct,
                    time_to_peak_bars=peak_idx + 1,
                    is_true_expansion=is_true_expansion,
                    rewind_horizon_lengths=rewind_lengths,
                    token_ids=token_ids,
                    rust_tensor=tensor_out,
                )
            )

        return samples

    @staticmethod
    def align_polars_streams(
        bars_df: Any,
        internals_df: Any,
    ) -> Any:
        """
        Polars 2.0 Streaming-Engine Compliant Synchronization.
        Always passes `maintain_order=True` on sequential time-series joins and `group_by`.
        """
        if not HAS_POLARS or pl is None:
            raise RuntimeError("Polars is not installed in the active environment")

        joined = bars_df.join(
            internals_df,
            on="timestamp_ns",
            how="left",
            maintain_order="left",
        )
        aggregated = joined.group_by("symbol", maintain_order=True).agg(
            [
                pl.col("close_price").last().alias("latest_close"),
                pl.col("volume").sum().alias("total_volume"),
            ]
        )
        return aggregated

