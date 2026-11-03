"""
Zero Training-Serving Skew Bridge (`rust_bridge.py`).

Links Python (`ml-brain`) directly to the compiled Rust shared library
(`crates/raw-fingerprint-core` -> `libraw_fingerprint_core.dylib` / `.so` / `.dll`)
so historical sequence tensor generation, 128-d `pgvector` embeddings, autonomous
structural stop solving, and Speed-1 live trace health evaluation execute the exact
same machine instructions bit-for-bit in training and live serving.
"""

from __future__ import annotations

import ctypes
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

SUB_PENNY_SCALE: int = 100_000
FINGERPRINT_FEATURE_DIM: int = 32
TRACE_EMBEDDING_DIM: int = 128

BRAIN_CODE_MAP: dict[str, int] = {
    "BRAIN-01": 1,
    "BRAIN-02": 2,
    "BRAIN-03": 3,
    "BRAIN-04": 4,
    "BRAIN-05": 5,
}

DRIVER_CODE_REVERSE: dict[int, str] = {
    0: "VOLUME_ABSORPTION",
    1: "GAMMA_SQUEEZE",
    2: "SHORT_SQUEEZE",
    3: "INTERNALS_ALIGNMENT",
}

BADGE_CODE_REVERSE: dict[int, str] = {
    0: "TRACE_ACCELERATING",
    1: "TRAILING_T1",
    2: "EXHAUSTION_WARNING",
    3: "DYNAMIC_EXIT_TRIGGERED",
}


def price_to_scaled_units(price_usd: float) -> int:
    """Converts a dollar float (`$0.00001` to `$10,000.00+`) into exact 5-decimal integer units."""
    scaled = int(round(price_usd * SUB_PENNY_SCALE))
    return max(scaled, 1)


def scaled_units_to_price(units: int) -> float:
    """Converts 5-decimal integer units back to float dollars."""
    return float(units) / float(SUB_PENNY_SCALE)


class _CAbiRawBar(ctypes.Structure):
    _fields_ = [
        ("timestamp_ns", ctypes.c_int64),
        ("open_scaled", ctypes.c_int64),
        ("high_scaled", ctypes.c_int64),
        ("low_scaled", ctypes.c_int64),
        ("close_scaled", ctypes.c_int64),
        ("volume", ctypes.c_uint64),
        ("vwap_scaled", ctypes.c_int64),
        ("dollar_turnover", ctypes.c_double),
        ("trade_count", ctypes.c_uint32),
        ("spread_scaled", ctypes.c_int64),
        ("time_delta_ms", ctypes.c_uint64),
    ]


class _CAbiOptionsSnapshot(ctypes.Structure):
    _fields_ = [
        ("timestamp_ns", ctypes.c_int64),
        ("call_volume", ctypes.c_uint64),
        ("put_volume", ctypes.c_uint64),
        ("total_open_interest", ctypes.c_uint64),
        ("ask_fill_aggression_ratio", ctypes.c_double),
        ("bid_fill_aggression_ratio", ctypes.c_double),
        ("net_dealer_gamma_gex", ctypes.c_double),
        ("net_dealer_vanna", ctypes.c_double),
        ("zero_dte_volume_share", ctypes.c_double),
        ("front_iv", ctypes.c_double),
        ("back_iv", ctypes.c_double),
        ("dominant_gamma_wall_scaled", ctypes.c_int64),
    ]


class _CAbiShortFloatSnapshot(ctypes.Structure):
    _fields_ = [
        ("timestamp_ns", ctypes.c_int64),
        ("public_float_shares", ctypes.c_uint64),
        ("shares_outstanding", ctypes.c_uint64),
        ("short_interest_shares", ctypes.c_uint64),
        ("days_to_cover", ctypes.c_double),
        ("intraday_short_volume_ratio", ctypes.c_double),
        ("cost_to_borrow_annual_pct", ctypes.c_double),
        ("cumulative_session_volume", ctypes.c_uint64),
    ]


class _CAbiMarketInternalsSnapshot(ctypes.Structure):
    _fields_ = [
        ("timestamp_ns", ctypes.c_int64),
        ("nyse_tick", ctypes.c_int32),
        ("nasdaq_tick", ctypes.c_int32),
        ("add_breadth", ctypes.c_int32),
        ("vold_up_down_ratio", ctypes.c_double),
        ("vix_spot", ctypes.c_double),
        ("vix_front_month", ctypes.c_double),
        ("sector_relative_flow", ctypes.c_double),
    ]


class _CAbiAutonomousSignalOutput(ctypes.Structure):
    _fields_ = [
        ("direction_sign", ctypes.c_int32),
        ("entry_scaled", ctypes.c_int64),
        ("structural_stop_scaled", ctypes.c_int64),
        ("target_1_scaled", ctypes.c_int64),
        ("target_2_scaled", ctypes.c_int64),
        ("predicted_hold_ms", ctypes.c_int64),
        ("ai_confidence_pct", ctypes.c_double),
        ("expected_r_multiple", ctypes.c_double),
        ("primary_driver_code", ctypes.c_int32),
        ("stream_weights", ctypes.c_float * 4),
    ]


class _CAbiLiveTraceHealthOutput(ctypes.Structure):
    _fields_ = [
        ("badge_code", ctypes.c_int32),
        ("updated_stop_scaled", ctypes.c_int64),
        ("opposing_absorption_score", ctypes.c_double),
        ("current_r_multiple", ctypes.c_double),
        ("should_exit_immediately", ctypes.c_int32),
    ]


@dataclass(frozen=True)
class RawBarInput:
    timestamp_ns: int
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    vwap: float
    dollar_turnover: float
    trade_count: int
    bid_ask_spread: float = 0.00001
    time_delta_ms: int = 60_000

    def to_cabi(self) -> _CAbiRawBar:
        return _CAbiRawBar(
            timestamp_ns=int(self.timestamp_ns),
            open_scaled=price_to_scaled_units(self.open_price),
            high_scaled=price_to_scaled_units(self.high_price),
            low_scaled=price_to_scaled_units(self.low_price),
            close_scaled=price_to_scaled_units(self.close_price),
            volume=max(int(self.volume), 0),
            vwap_scaled=price_to_scaled_units(self.vwap),
            dollar_turnover=float(self.dollar_turnover),
            trade_count=max(int(self.trade_count), 1),
            spread_scaled=max(int(round(self.bid_ask_spread * SUB_PENNY_SCALE)), 1),
            time_delta_ms=max(int(self.time_delta_ms), 1),
        )


@dataclass(frozen=True)
class RawOptionsInput:
    timestamp_ns: int = 0
    call_volume: int = 12_000
    put_volume: int = 4_500
    total_open_interest: int = 65_000
    ask_fill_aggression_ratio: float = 0.72
    bid_fill_aggression_ratio: float = 0.28
    net_dealer_gamma_gex: float = 35.0
    net_dealer_vanna: float = 14.0
    zero_dte_volume_share: float = 0.55
    front_iv: float = 0.48
    back_iv: float = 0.36
    dominant_gamma_wall_strike: float = 100.0

    def to_cabi(self) -> _CAbiOptionsSnapshot:
        return _CAbiOptionsSnapshot(
            timestamp_ns=int(self.timestamp_ns),
            call_volume=max(int(self.call_volume), 0),
            put_volume=max(int(self.put_volume), 0),
            total_open_interest=max(int(self.total_open_interest), 1),
            ask_fill_aggression_ratio=float(self.ask_fill_aggression_ratio),
            bid_fill_aggression_ratio=float(self.bid_fill_aggression_ratio),
            net_dealer_gamma_gex=float(self.net_dealer_gamma_gex),
            net_dealer_vanna=float(self.net_dealer_vanna),
            zero_dte_volume_share=float(self.zero_dte_volume_share),
            front_iv=float(self.front_iv),
            back_iv=float(self.back_iv),
            dominant_gamma_wall_scaled=price_to_scaled_units(self.dominant_gamma_wall_strike),
        )


@dataclass(frozen=True)
class RawShortFloatInput:
    timestamp_ns: int = 0
    public_float_shares: int = 25_000_000
    shares_outstanding: int = 28_000_000
    short_interest_shares: int = 6_250_000
    days_to_cover: float = 3.8
    intraday_short_volume_ratio: float = 0.54
    cost_to_borrow_annual_pct: float = 24.5
    cumulative_session_volume: int = 8_500_000

    def to_cabi(self) -> _CAbiShortFloatSnapshot:
        return _CAbiShortFloatSnapshot(
            timestamp_ns=int(self.timestamp_ns),
            public_float_shares=max(int(self.public_float_shares), 1),
            shares_outstanding=max(int(self.shares_outstanding), int(self.public_float_shares), 1),
            short_interest_shares=max(int(self.short_interest_shares), 0),
            days_to_cover=float(self.days_to_cover),
            intraday_short_volume_ratio=float(self.intraday_short_volume_ratio),
            cost_to_borrow_annual_pct=float(self.cost_to_borrow_annual_pct),
            cumulative_session_volume=max(int(self.cumulative_session_volume), 0),
        )


@dataclass(frozen=True)
class RawMarketInternalsInput:
    timestamp_ns: int = 0
    nyse_tick: int = 620
    nasdaq_tick: int = 780
    add_breadth: int = 1480
    vold_up_down_ratio: float = 2.75
    vix_spot: float = 14.5
    vix_front_month: float = 15.9
    sector_relative_flow: float = 0.65

    def to_cabi(self) -> _CAbiMarketInternalsSnapshot:
        return _CAbiMarketInternalsSnapshot(
            timestamp_ns=int(self.timestamp_ns),
            nyse_tick=int(self.nyse_tick),
            nasdaq_tick=int(self.nasdaq_tick),
            add_breadth=int(self.add_breadth),
            vold_up_down_ratio=float(self.vold_up_down_ratio),
            vix_spot=float(self.vix_spot),
            vix_front_month=float(self.vix_front_month),
            sector_relative_flow=float(self.sector_relative_flow),
        )


@dataclass(frozen=True)
class RustTensorOutput:
    seq_len: int
    feature_dim: int
    matrix_2d: list[list[float]]
    flat_tensor: list[float]
    latent_embedding_128: list[float]
    stream_weights: dict[str, float]


@dataclass(frozen=True)
class RustAutonomousSignalOutput:
    symbol: str
    brain_id: str
    direction: str
    entry_price: float
    structural_stop_price: float
    target_1_price: float
    target_2_price: float
    entry_scaled: int
    structural_stop_scaled: int
    target_1_scaled: int
    target_2_scaled: int
    predicted_hold_duration_ms: int
    ai_confidence_pct: float
    expected_r_multiple: float
    primary_driver: str
    stream_weights: dict[str, float]


@dataclass(frozen=True)
class RustLiveTraceHealthOutput:
    badge: str
    updated_structural_stop_price: float
    updated_stop_scaled: int
    opposing_absorption_score: float
    current_r_multiple: float
    should_exit_immediately: bool


class RustFingerprintBridge:
    """
    Zero-Skew FFI Wrapper around `libraw_fingerprint_core`.
    """

    def __init__(self, dylib_path: Path | None = None) -> None:
        self._lib_path = dylib_path or self._discover_dylib()
        self._lib = ctypes.CDLL(str(self._lib_path))
        self._bind_signatures()

    @staticmethod
    def _discover_dylib() -> Path:
        repo_root = Path(__file__).resolve().parents[2]
        ext = {
            "Darwin": "dylib",
            "Linux": "so",
            "Windows": "dll",
        }.get(platform.system(), "dylib")
        prefix = "" if platform.system() == "Windows" else "lib"
        filename = f"{prefix}raw_fingerprint_core.{ext}"

        candidates = [
            repo_root / "target" / "release" / filename,
            repo_root / "target" / "debug" / filename,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Compiled Rust library {filename} not found in {candidates}. "
            "Run `cargo build --release -p raw-fingerprint-core` first."
        )

    def _bind_signatures(self) -> None:
        self._lib.argon_build_fingerprint_tensor_cabi.argtypes = [
            ctypes.POINTER(_CAbiRawBar),
            ctypes.c_size_t,
            ctypes.POINTER(_CAbiOptionsSnapshot),
            ctypes.POINTER(_CAbiShortFloatSnapshot),
            ctypes.POINTER(_CAbiMarketInternalsSnapshot),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self._lib.argon_build_fingerprint_tensor_cabi.restype = ctypes.c_int32

        self._lib.argon_evaluate_autonomous_signal_cabi.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(_CAbiRawBar),
            ctypes.c_size_t,
            ctypes.POINTER(_CAbiOptionsSnapshot),
            ctypes.POINTER(_CAbiShortFloatSnapshot),
            ctypes.POINTER(_CAbiMarketInternalsSnapshot),
            ctypes.POINTER(_CAbiAutonomousSignalOutput),
        ]
        self._lib.argon_evaluate_autonomous_signal_cabi.restype = ctypes.c_int32

        self._lib.argon_evaluate_live_trace_health_cabi.argtypes = [
            ctypes.c_int32,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(_CAbiRawBar),
            ctypes.c_size_t,
            ctypes.POINTER(_CAbiMarketInternalsSnapshot),
            ctypes.POINTER(_CAbiLiveTraceHealthOutput),
        ]
        self._lib.argon_evaluate_live_trace_health_cabi.restype = ctypes.c_int32

    def build_sequence_tensor(
        self,
        bars: Sequence[RawBarInput],
        options: RawOptionsInput | None = None,
        short_float: RawShortFloatInput | None = None,
        internals: RawMarketInternalsInput | None = None,
    ) -> RustTensorOutput:
        seq_len = len(bars)
        if seq_len < 2:
            raise ValueError(f"Expected at least 2 bars, got {seq_len}")

        opts_c = (options or RawOptionsInput()).to_cabi()
        sf_c = (short_float or RawShortFloatInput()).to_cabi()
        int_c = (internals or RawMarketInternalsInput()).to_cabi()

        bar_array_type = _CAbiRawBar * seq_len
        c_bars = bar_array_type(*(b.to_cabi() for b in bars))

        out_tensor_c = (ctypes.c_float * (seq_len * FINGERPRINT_FEATURE_DIM))()
        out_emb_c = (ctypes.c_float * TRACE_EMBEDDING_DIM)()
        out_weights_c = (ctypes.c_float * 4)()

        rc = self._lib.argon_build_fingerprint_tensor_cabi(
            c_bars,
            seq_len,
            ctypes.byref(opts_c),
            ctypes.byref(sf_c),
            ctypes.byref(int_c),
            out_tensor_c,
            out_emb_c,
            out_weights_c,
        )
        if rc != 0:
            raise RuntimeError(f"argon_build_fingerprint_tensor_cabi failed with status {rc}")

        flat = [float(x) for x in out_tensor_c]
        matrix_2d = [
            flat[i * FINGERPRINT_FEATURE_DIM : (i + 1) * FINGERPRINT_FEATURE_DIM]
            for i in range(seq_len)
        ]
        emb_128 = [float(x) for x in out_emb_c]
        weights = {
            "chart_tape_physics": float(out_weights_c[0]),
            "options_gamma_vanna": float(out_weights_c[1]),
            "short_float_squeeze": float(out_weights_c[2]),
            "market_internals_macro": float(out_weights_c[3]),
        }

        return RustTensorOutput(
            seq_len=seq_len,
            feature_dim=FINGERPRINT_FEATURE_DIM,
            matrix_2d=matrix_2d,
            flat_tensor=flat,
            latent_embedding_128=emb_128,
            stream_weights=weights,
        )

    def evaluate_autonomous_signal(
        self,
        brain_id: str,
        symbol: str,
        bars: Sequence[RawBarInput],
        options: RawOptionsInput | None = None,
        short_float: RawShortFloatInput | None = None,
        internals: RawMarketInternalsInput | None = None,
    ) -> RustAutonomousSignalOutput:
        seq_len = len(bars)
        if seq_len < 2:
            raise ValueError(f"Expected at least 2 bars, got {seq_len}")

        brain_code = BRAIN_CODE_MAP.get(brain_id, 2)
        opts_c = (options or RawOptionsInput()).to_cabi()
        sf_c = (short_float or RawShortFloatInput()).to_cabi()
        int_c = (internals or RawMarketInternalsInput()).to_cabi()

        bar_array_type = _CAbiRawBar * seq_len
        c_bars = bar_array_type(*(b.to_cabi() for b in bars))
        out_sig = _CAbiAutonomousSignalOutput()

        rc = self._lib.argon_evaluate_autonomous_signal_cabi(
            brain_code,
            c_bars,
            seq_len,
            ctypes.byref(opts_c),
            ctypes.byref(sf_c),
            ctypes.byref(int_c),
            ctypes.byref(out_sig),
        )
        if rc != 0:
            raise RuntimeError(f"argon_evaluate_autonomous_signal_cabi failed with status {rc}")

        return RustAutonomousSignalOutput(
            symbol=symbol.upper(),
            brain_id=brain_id,
            direction="LONG" if out_sig.direction_sign >= 0 else "SHORT",
            entry_price=scaled_units_to_price(out_sig.entry_scaled),
            structural_stop_price=scaled_units_to_price(out_sig.structural_stop_scaled),
            target_1_price=scaled_units_to_price(out_sig.target_1_scaled),
            target_2_price=scaled_units_to_price(out_sig.target_2_scaled),
            entry_scaled=int(out_sig.entry_scaled),
            structural_stop_scaled=int(out_sig.structural_stop_scaled),
            target_1_scaled=int(out_sig.target_1_scaled),
            target_2_scaled=int(out_sig.target_2_scaled),
            predicted_hold_duration_ms=int(out_sig.predicted_hold_ms),
            ai_confidence_pct=float(out_sig.ai_confidence_pct),
            expected_r_multiple=float(out_sig.expected_r_multiple),
            primary_driver=DRIVER_CODE_REVERSE.get(int(out_sig.primary_driver_code), "VOLUME_ABSORPTION"),
            stream_weights={
                "chart_tape_physics": float(out_sig.stream_weights[0]),
                "options_gamma_vanna": float(out_sig.stream_weights[1]),
                "short_float_squeeze": float(out_sig.stream_weights[2]),
                "market_internals_macro": float(out_sig.stream_weights[3]),
            },
        )

    def evaluate_live_trace_health(
        self,
        direction: str,
        entry_price: float,
        structural_stop_price: float,
        target_1_price: float,
        target_2_price: float,
        peak_favorable_price: float,
        recent_bars: Sequence[RawBarInput],
        internals: RawMarketInternalsInput | None = None,
    ) -> RustLiveTraceHealthOutput:
        seq_len = len(recent_bars)
        if seq_len < 1:
            raise ValueError("recent_bars must contain at least 1 bar")

        dir_sign = 1 if direction.upper() == "LONG" else -1
        bar_array_type = _CAbiRawBar * seq_len
        c_bars = bar_array_type(*(b.to_cabi() for b in recent_bars))
        int_c = (internals or RawMarketInternalsInput()).to_cabi()
        out_health = _CAbiLiveTraceHealthOutput()

        rc = self._lib.argon_evaluate_live_trace_health_cabi(
            dir_sign,
            price_to_scaled_units(entry_price),
            price_to_scaled_units(structural_stop_price),
            price_to_scaled_units(target_1_price),
            price_to_scaled_units(target_2_price),
            price_to_scaled_units(peak_favorable_price),
            c_bars,
            seq_len,
            ctypes.byref(int_c),
            ctypes.byref(out_health),
        )
        if rc != 0:
            raise RuntimeError(f"argon_evaluate_live_trace_health_cabi failed with status {rc}")

        return RustLiveTraceHealthOutput(
            badge=BADGE_CODE_REVERSE.get(int(out_health.badge_code), "TRACE_ACCELERATING"),
            updated_structural_stop_price=scaled_units_to_price(out_health.updated_stop_scaled),
            updated_stop_scaled=int(out_health.updated_stop_scaled),
            opposing_absorption_score=float(out_health.opposing_absorption_score),
            current_r_multiple=float(out_health.current_r_multiple),
            should_exit_immediately=bool(out_health.should_exit_immediately),
        )

