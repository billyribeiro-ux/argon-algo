//! Zero Training-Serving Skew FFI Bridge (`C-ABI` + `PyO3 0.29.2` Adapter).
//!
//! Exposes `RawFingerprintEngine::build_sequence_tensor` directly to Python (`ml-brain`)
//! so historical training sequences and live Rust execution ticks execute the exact
//! same compiled machine code bit-for-bit.

use std::slice;

use crate::price::SubPennyPrice;
use crate::streams::{
    BrainHorizonId, PrimaryFingerprintDriver, RawBarRecord, RawMarketInternalsSnapshot,
    RawOptionsSnapshot, RawShortFloatSnapshot, TraceHealthBadge, TradeDirection,
};
use crate::tensor::{
    FINGERPRINT_FEATURE_DIM, OpenPositionEvaluationContext, RawFingerprintEngine,
    TRACE_EMBEDDING_DIM,
};

/// C-compatible raw bar struct for zero-copy array passing from Python / NumPy / Polars.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CAbiRawBar {
    pub timestamp_ns: i64,
    pub open_scaled: i64,
    pub high_scaled: i64,
    pub low_scaled: i64,
    pub close_scaled: i64,
    pub volume: u64,
    pub vwap_scaled: i64,
    pub dollar_turnover: f64,
    pub trade_count: u32,
    pub spread_scaled: i64,
    pub time_delta_ms: u64,
}

/// C-compatible Options / Gamma / Vanna snapshot struct.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CAbiOptionsSnapshot {
    pub timestamp_ns: i64,
    pub call_volume: u64,
    pub put_volume: u64,
    pub total_open_interest: u64,
    pub ask_fill_aggression_ratio: f64,
    pub bid_fill_aggression_ratio: f64,
    pub net_dealer_gamma_gex: f64,
    pub net_dealer_vanna: f64,
    pub zero_dte_volume_share: f64,
    pub front_iv: f64,
    pub back_iv: f64,
    pub dominant_gamma_wall_scaled: i64,
}

/// C-compatible Short & Float squeeze snapshot struct.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CAbiShortFloatSnapshot {
    pub timestamp_ns: i64,
    pub public_float_shares: u64,
    pub shares_outstanding: u64,
    pub short_interest_shares: u64,
    pub days_to_cover: f64,
    pub intraday_short_volume_ratio: f64,
    pub cost_to_borrow_annual_pct: f64,
    pub cumulative_session_volume: u64,
}

/// C-compatible Market Internals snapshot struct.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CAbiMarketInternalsSnapshot {
    pub timestamp_ns: i64,
    pub nyse_tick: i32,
    pub nasdaq_tick: i32,
    pub add_breadth: i32,
    pub vold_up_down_ratio: f64,
    pub vix_spot: f64,
    pub vix_front_month: f64,
    pub sector_relative_flow: f64,
}

/// Computes the 4-stream `[seq_len, 32]` normalized tensor, `[128]` latent embedding,
/// and `[4]` stream attribution weights directly into pre-allocated caller buffers.
///
/// # Safety
/// Caller must ensure:
/// - `bars_ptr` points to `seq_len` valid `CAbiRawBar` elements (`seq_len >= 2`).
/// - `options_ptr`, `short_float_ptr`, and `internals_ptr` are valid non-null pointers.
/// - `out_tensor_ptr` has capacity for at least `seq_len * 32` `f32` elements.
/// - `out_embedding_128_ptr` has capacity for at least `128` `f32` elements.
/// - `out_weights_4_ptr` has capacity for at least `4` `f32` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn argon_build_fingerprint_tensor_cabi(
    bars_ptr: *const CAbiRawBar,
    seq_len: usize,
    options_ptr: *const CAbiOptionsSnapshot,
    short_float_ptr: *const CAbiShortFloatSnapshot,
    internals_ptr: *const CAbiMarketInternalsSnapshot,
    out_tensor_ptr: *mut f32,
    out_embedding_128_ptr: *mut f32,
    out_weights_4_ptr: *mut f32,
) -> i32 {
    if bars_ptr.is_null()
        || options_ptr.is_null()
        || short_float_ptr.is_null()
        || internals_ptr.is_null()
        || out_tensor_ptr.is_null()
        || out_embedding_128_ptr.is_null()
        || out_weights_4_ptr.is_null()
        || seq_len < 2
    {
        return -1;
    }

    let c_bars = unsafe { slice::from_raw_parts(bars_ptr, seq_len) };
    let c_opts = unsafe { &*options_ptr };
    let c_sf = unsafe { &*short_float_ptr };
    let c_int = unsafe { &*internals_ptr };

    let mut bars = Vec::with_capacity(seq_len);
    for b in c_bars {
        let open = SubPennyPrice::from_scaled_units(b.open_scaled.max(1))
            .unwrap_or(SubPennyPrice::MIN_VALID);
        let high = SubPennyPrice::from_scaled_units(b.high_scaled.max(1))
            .unwrap_or(SubPennyPrice::MIN_VALID);
        let low = SubPennyPrice::from_scaled_units(b.low_scaled.max(1))
            .unwrap_or(SubPennyPrice::MIN_VALID);
        let close = SubPennyPrice::from_scaled_units(b.close_scaled.max(1))
            .unwrap_or(SubPennyPrice::MIN_VALID);
        let vwap = SubPennyPrice::from_scaled_units(b.vwap_scaled.max(1)).unwrap_or(close);
        let spread = SubPennyPrice::from_scaled_units_allow_zero(b.spread_scaled.max(0))
            .unwrap_or(SubPennyPrice::ZERO);

        bars.push(RawBarRecord {
            timestamp_ns: b.timestamp_ns,
            open,
            high,
            low,
            close,
            volume: b.volume,
            vwap,
            dollar_turnover: b.dollar_turnover,
            trade_count: b.trade_count,
            bid_ask_spread: spread,
            time_delta_ms: b.time_delta_ms,
        });
    }

    let options = RawOptionsSnapshot {
        timestamp_ns: c_opts.timestamp_ns,
        call_volume: c_opts.call_volume,
        put_volume: c_opts.put_volume,
        total_open_interest: c_opts.total_open_interest,
        ask_fill_aggression_ratio: c_opts.ask_fill_aggression_ratio,
        bid_fill_aggression_ratio: c_opts.bid_fill_aggression_ratio,
        net_dealer_gamma_gex: c_opts.net_dealer_gamma_gex,
        net_dealer_vanna: c_opts.net_dealer_vanna,
        zero_dte_volume_share: c_opts.zero_dte_volume_share,
        front_iv: c_opts.front_iv,
        back_iv: c_opts.back_iv,
        dominant_gamma_wall_strike: SubPennyPrice::from_scaled_units(
            c_opts.dominant_gamma_wall_scaled.max(1),
        )
        .unwrap_or(SubPennyPrice::MIN_VALID),
    };

    let short_float = RawShortFloatSnapshot {
        timestamp_ns: c_sf.timestamp_ns,
        public_float_shares: c_sf.public_float_shares,
        shares_outstanding: c_sf.shares_outstanding,
        short_interest_shares: c_sf.short_interest_shares,
        days_to_cover: c_sf.days_to_cover,
        intraday_short_volume_ratio: c_sf.intraday_short_volume_ratio,
        cost_to_borrow_annual_pct: c_sf.cost_to_borrow_annual_pct,
        cumulative_session_volume: c_sf.cumulative_session_volume,
    };

    let internals = RawMarketInternalsSnapshot {
        timestamp_ns: c_int.timestamp_ns,
        nyse_tick: c_int.nyse_tick,
        nasdaq_tick: c_int.nasdaq_tick,
        add_breadth: c_int.add_breadth,
        vold_up_down_ratio: c_int.vold_up_down_ratio,
        vix_spot: c_int.vix_spot,
        vix_front_month: c_int.vix_front_month,
        sector_relative_flow: c_int.sector_relative_flow,
    };

    let engine = RawFingerprintEngine::new();
    match engine.build_sequence_tensor(&bars, &options, &short_float, &internals) {
        Ok(res) => {
            let out_tensor = unsafe {
                slice::from_raw_parts_mut(out_tensor_ptr, seq_len * FINGERPRINT_FEATURE_DIM)
            };
            out_tensor.copy_from_slice(&res.flat_tensor);

            let out_emb =
                unsafe { slice::from_raw_parts_mut(out_embedding_128_ptr, TRACE_EMBEDDING_DIM) };
            out_emb.copy_from_slice(&res.latent_embedding_128);

            let out_w = unsafe { slice::from_raw_parts_mut(out_weights_4_ptr, 4) };
            out_w[0] = res.stream_weights.chart_tape_physics_weight;
            out_w[1] = res.stream_weights.options_gamma_vanna_weight;
            out_w[2] = res.stream_weights.short_float_squeeze_weight;
            out_w[3] = res.stream_weights.market_internals_macro_weight;
            0
        }
        Err(_) => -2,
    }
}

/// C-compatible output struct for `argon_evaluate_autonomous_signal_cabi`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CAbiAutonomousSignalOutput {
    pub direction_sign: i32,
    pub entry_scaled: i64,
    pub structural_stop_scaled: i64,
    pub target_1_scaled: i64,
    pub target_2_scaled: i64,
    pub predicted_hold_ms: i64,
    pub ai_confidence_pct: f64,
    pub expected_r_multiple: f64,
    pub primary_driver_code: i32,
    pub stream_weights: [f32; 4],
}

/// C-compatible output struct for `argon_evaluate_live_trace_health_cabi` (Speed-1 Intraday Loop).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CAbiLiveTraceHealthOutput {
    pub badge_code: i32,
    pub updated_stop_scaled: i64,
    pub opposing_absorption_score: f64,
    pub current_r_multiple: f64,
    pub should_exit_immediately: i32,
}

/// Evaluates a complete autonomous signal blueprint over C-ABI with zero training-serving skew.
///
/// # Safety
/// All pointer arguments must be valid and non-null, and `bars_ptr` must contain `seq_len >= 2` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn argon_evaluate_autonomous_signal_cabi(
    brain_code: i32,
    bars_ptr: *const CAbiRawBar,
    seq_len: usize,
    options_ptr: *const CAbiOptionsSnapshot,
    short_float_ptr: *const CAbiShortFloatSnapshot,
    internals_ptr: *const CAbiMarketInternalsSnapshot,
    out_signal_ptr: *mut CAbiAutonomousSignalOutput,
) -> i32 {
    if bars_ptr.is_null()
        || options_ptr.is_null()
        || short_float_ptr.is_null()
        || internals_ptr.is_null()
        || out_signal_ptr.is_null()
        || seq_len < 2
    {
        return -1;
    }

    let brain_id = match brain_code {
        1 => BrainHorizonId::Brain01PennyAnomaly,
        2 => BrainHorizonId::Brain02IntradayDayTrade,
        3 => BrainHorizonId::Brain03ShortTermSwing,
        4 => BrainHorizonId::Brain04CoreSwing,
        _ => BrainHorizonId::Brain05LongTermWealth,
    };

    let c_bars = unsafe { slice::from_raw_parts(bars_ptr, seq_len) };
    let c_opts = unsafe { &*options_ptr };
    let c_sf = unsafe { &*short_float_ptr };
    let c_int = unsafe { &*internals_ptr };

    let mut bars = Vec::with_capacity(seq_len);
    for b in c_bars {
        bars.push(RawBarRecord {
            timestamp_ns: b.timestamp_ns,
            open: SubPennyPrice::from_scaled_units(b.open_scaled.max(1))
                .unwrap_or(SubPennyPrice::MIN_VALID),
            high: SubPennyPrice::from_scaled_units(b.high_scaled.max(1))
                .unwrap_or(SubPennyPrice::MIN_VALID),
            low: SubPennyPrice::from_scaled_units(b.low_scaled.max(1))
                .unwrap_or(SubPennyPrice::MIN_VALID),
            close: SubPennyPrice::from_scaled_units(b.close_scaled.max(1))
                .unwrap_or(SubPennyPrice::MIN_VALID),
            volume: b.volume,
            vwap: SubPennyPrice::from_scaled_units(b.vwap_scaled.max(1))
                .unwrap_or(SubPennyPrice::MIN_VALID),
            dollar_turnover: b.dollar_turnover,
            trade_count: b.trade_count,
            bid_ask_spread: SubPennyPrice::from_scaled_units_allow_zero(b.spread_scaled.max(0))
                .unwrap_or(SubPennyPrice::ZERO),
            time_delta_ms: b.time_delta_ms,
        });
    }

    let options = RawOptionsSnapshot {
        timestamp_ns: c_opts.timestamp_ns,
        call_volume: c_opts.call_volume,
        put_volume: c_opts.put_volume,
        total_open_interest: c_opts.total_open_interest,
        ask_fill_aggression_ratio: c_opts.ask_fill_aggression_ratio,
        bid_fill_aggression_ratio: c_opts.bid_fill_aggression_ratio,
        net_dealer_gamma_gex: c_opts.net_dealer_gamma_gex,
        net_dealer_vanna: c_opts.net_dealer_vanna,
        zero_dte_volume_share: c_opts.zero_dte_volume_share,
        front_iv: c_opts.front_iv,
        back_iv: c_opts.back_iv,
        dominant_gamma_wall_strike: SubPennyPrice::from_scaled_units(
            c_opts.dominant_gamma_wall_scaled.max(1),
        )
        .unwrap_or(SubPennyPrice::MIN_VALID),
    };

    let short_float = RawShortFloatSnapshot {
        timestamp_ns: c_sf.timestamp_ns,
        public_float_shares: c_sf.public_float_shares,
        shares_outstanding: c_sf.shares_outstanding,
        short_interest_shares: c_sf.short_interest_shares,
        days_to_cover: c_sf.days_to_cover,
        intraday_short_volume_ratio: c_sf.intraday_short_volume_ratio,
        cost_to_borrow_annual_pct: c_sf.cost_to_borrow_annual_pct,
        cumulative_session_volume: c_sf.cumulative_session_volume,
    };

    let internals = RawMarketInternalsSnapshot {
        timestamp_ns: c_int.timestamp_ns,
        nyse_tick: c_int.nyse_tick,
        nasdaq_tick: c_int.nasdaq_tick,
        add_breadth: c_int.add_breadth,
        vold_up_down_ratio: c_int.vold_up_down_ratio,
        vix_spot: c_int.vix_spot,
        vix_front_month: c_int.vix_front_month,
        sector_relative_flow: c_int.sector_relative_flow,
    };

    let engine = RawFingerprintEngine::new();
    match engine.evaluate_autonomous_signal(
        "CABI",
        brain_id,
        &bars,
        &options,
        &short_float,
        &internals,
    ) {
        Ok(sig) => {
            let driver_code = match sig.primary_driver {
                PrimaryFingerprintDriver::VolumeAbsorption => 0,
                PrimaryFingerprintDriver::GammaSqueeze => 1,
                PrimaryFingerprintDriver::ShortSqueeze => 2,
                PrimaryFingerprintDriver::InternalsAlignment => 3,
            };
            unsafe {
                *out_signal_ptr = CAbiAutonomousSignalOutput {
                    direction_sign: match sig.direction {
                        TradeDirection::Long => 1,
                        TradeDirection::Short => -1,
                    },
                    entry_scaled: sig.entry_price.scaled_units(),
                    structural_stop_scaled: sig.structural_stop_price.scaled_units(),
                    target_1_scaled: sig.target_1_price.scaled_units(),
                    target_2_scaled: sig.target_2_price.scaled_units(),
                    predicted_hold_ms: sig.predicted_hold_duration_ms,
                    ai_confidence_pct: sig.ai_confidence_pct,
                    expected_r_multiple: sig.expected_r_multiple,
                    primary_driver_code: driver_code,
                    stream_weights: [
                        sig.stream_weights.chart_tape_physics_weight,
                        sig.stream_weights.options_gamma_vanna_weight,
                        sig.stream_weights.short_float_squeeze_weight,
                        sig.stream_weights.market_internals_macro_weight,
                    ],
                };
            }
            0
        }
        Err(_) => -2,
    }
}

/// Evaluates Speed-1 Intraday bar-by-bar dynamic exit and trace health over C-ABI.
///
/// # Safety
/// All pointer arguments must be valid and non-null, and `bars_ptr` must contain `seq_len >= 1` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn argon_evaluate_live_trace_health_cabi(
    direction_sign: i32,
    entry_scaled: i64,
    structural_stop_scaled: i64,
    target_1_scaled: i64,
    target_2_scaled: i64,
    peak_favorable_scaled: i64,
    bars_ptr: *const CAbiRawBar,
    seq_len: usize,
    internals_ptr: *const CAbiMarketInternalsSnapshot,
    out_health_ptr: *mut CAbiLiveTraceHealthOutput,
) -> i32 {
    if bars_ptr.is_null() || internals_ptr.is_null() || out_health_ptr.is_null() || seq_len == 0 {
        return -1;
    }

    let c_bars = unsafe { slice::from_raw_parts(bars_ptr, seq_len) };
    let c_int = unsafe { &*internals_ptr };

    let mut bars = Vec::with_capacity(seq_len);
    for b in c_bars {
        bars.push(RawBarRecord {
            timestamp_ns: b.timestamp_ns,
            open: SubPennyPrice::from_scaled_units(b.open_scaled.max(1))
                .unwrap_or(SubPennyPrice::MIN_VALID),
            high: SubPennyPrice::from_scaled_units(b.high_scaled.max(1))
                .unwrap_or(SubPennyPrice::MIN_VALID),
            low: SubPennyPrice::from_scaled_units(b.low_scaled.max(1))
                .unwrap_or(SubPennyPrice::MIN_VALID),
            close: SubPennyPrice::from_scaled_units(b.close_scaled.max(1))
                .unwrap_or(SubPennyPrice::MIN_VALID),
            volume: b.volume,
            vwap: SubPennyPrice::from_scaled_units(b.vwap_scaled.max(1))
                .unwrap_or(SubPennyPrice::MIN_VALID),
            dollar_turnover: b.dollar_turnover,
            trade_count: b.trade_count,
            bid_ask_spread: SubPennyPrice::from_scaled_units_allow_zero(b.spread_scaled.max(0))
                .unwrap_or(SubPennyPrice::ZERO),
            time_delta_ms: b.time_delta_ms,
        });
    }

    let internals = RawMarketInternalsSnapshot {
        timestamp_ns: c_int.timestamp_ns,
        nyse_tick: c_int.nyse_tick,
        nasdaq_tick: c_int.nasdaq_tick,
        add_breadth: c_int.add_breadth,
        vold_up_down_ratio: c_int.vold_up_down_ratio,
        vix_spot: c_int.vix_spot,
        vix_front_month: c_int.vix_front_month,
        sector_relative_flow: c_int.sector_relative_flow,
    };

    let ctx = OpenPositionEvaluationContext {
        direction: if direction_sign >= 0 {
            TradeDirection::Long
        } else {
            TradeDirection::Short
        },
        entry_price: SubPennyPrice::from_scaled_units(entry_scaled.max(1))
            .unwrap_or(SubPennyPrice::MIN_VALID),
        structural_stop_price: SubPennyPrice::from_scaled_units(structural_stop_scaled.max(1))
            .unwrap_or(SubPennyPrice::MIN_VALID),
        target_1_price: SubPennyPrice::from_scaled_units(target_1_scaled.max(1))
            .unwrap_or(SubPennyPrice::MIN_VALID),
        target_2_price: SubPennyPrice::from_scaled_units(target_2_scaled.max(1))
            .unwrap_or(SubPennyPrice::MIN_VALID),
        peak_favorable_price: SubPennyPrice::from_scaled_units(peak_favorable_scaled.max(1))
            .unwrap_or(SubPennyPrice::MIN_VALID),
    };

    let engine = RawFingerprintEngine::new();
    match engine.evaluate_live_trace_health(&ctx, &bars, &internals) {
        Ok(eval) => {
            let badge_code = match eval.badge {
                TraceHealthBadge::Accelerating => 0,
                TraceHealthBadge::TrailingTarget1 => 1,
                TraceHealthBadge::ExhaustionWarning => 2,
                TraceHealthBadge::DynamicExitTriggered => 3,
            };
            unsafe {
                *out_health_ptr = CAbiLiveTraceHealthOutput {
                    badge_code,
                    updated_stop_scaled: eval.updated_structural_stop.scaled_units(),
                    opposing_absorption_score: eval.opposing_absorption_score,
                    current_r_multiple: eval.current_r_multiple,
                    should_exit_immediately: i32::from(eval.should_exit_immediately),
                };
            }
            0
        }
        Err(_) => -2,
    }
}

/// PyO3 0.29.2 Native Rust-to-Python Binding Module (`Python::attach` / `Python::detach` / `.cast()`).
#[cfg(feature = "pyo3-bindings")]
pub mod pyo3_bridge {
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    use crate::price::SubPennyPrice;
    use crate::streams::{
        RawBarRecord, RawMarketInternalsSnapshot, RawOptionsSnapshot, RawShortFloatSnapshot,
    };
    use crate::tensor::RawFingerprintEngine;

    pub type RawBarTuple = (i64, i64, i64, i64, i64, u64, i64, f64, u32, i64, u64);

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    pub fn build_fingerprint_tensor_py(
        bars_scaled: Vec<RawBarTuple>,
        call_volume: u64,
        put_volume: u64,
        total_oi: u64,
        ask_agg: f64,
        bid_agg: f64,
        dealer_gex: f64,
        dealer_vanna: f64,
        zero_dte_share: f64,
        front_iv: f64,
        back_iv: f64,
        gamma_wall_scaled: i64,
        public_float: u64,
        shares_out: u64,
        short_interest: u64,
        dtc: f64,
        short_vol_ratio: f64,
        ctb_pct: f64,
        session_vol: u64,
        nyse_tick: i32,
        nasdaq_tick: i32,
        add_breadth: i32,
        vold_ratio: f64,
        vix_spot: f64,
        vix_front: f64,
        sector_flow: f64,
    ) -> PyResult<(Vec<f32>, Vec<f32>, [f32; 4])> {
        let result = Python::detach(|| {
            let mut bars = Vec::with_capacity(bars_scaled.len());
            for (ts, o, h, l, c, vol, vwap, turn, tc, sp, dt) in bars_scaled {
                bars.push(RawBarRecord {
                    timestamp_ns: ts,
                    open: SubPennyPrice::from_scaled_units(o.max(1))
                        .unwrap_or(SubPennyPrice::MIN_VALID),
                    high: SubPennyPrice::from_scaled_units(h.max(1))
                        .unwrap_or(SubPennyPrice::MIN_VALID),
                    low: SubPennyPrice::from_scaled_units(l.max(1))
                        .unwrap_or(SubPennyPrice::MIN_VALID),
                    close: SubPennyPrice::from_scaled_units(c.max(1))
                        .unwrap_or(SubPennyPrice::MIN_VALID),
                    volume: vol,
                    vwap: SubPennyPrice::from_scaled_units(vwap.max(1))
                        .unwrap_or(SubPennyPrice::MIN_VALID),
                    dollar_turnover: turn,
                    trade_count: tc,
                    bid_ask_spread: SubPennyPrice::from_scaled_units_allow_zero(sp.max(0))
                        .unwrap_or(SubPennyPrice::ZERO),
                    time_delta_ms: dt,
                });
            }

            let opts = RawOptionsSnapshot {
                timestamp_ns: 0,
                call_volume,
                put_volume,
                total_open_interest: total_oi,
                ask_fill_aggression_ratio: ask_agg,
                bid_fill_aggression_ratio: bid_agg,
                net_dealer_gamma_gex: dealer_gex,
                net_dealer_vanna: dealer_vanna,
                zero_dte_volume_share: zero_dte_share,
                front_iv,
                back_iv,
                dominant_gamma_wall_strike: SubPennyPrice::from_scaled_units(
                    gamma_wall_scaled.max(1),
                )
                .unwrap_or(SubPennyPrice::MIN_VALID),
            };

            let sf = RawShortFloatSnapshot {
                timestamp_ns: 0,
                public_float_shares: public_float,
                shares_outstanding: shares_out,
                short_interest_shares: short_interest,
                days_to_cover: dtc,
                intraday_short_volume_ratio: short_vol_ratio,
                cost_to_borrow_annual_pct: ctb_pct,
                cumulative_session_volume: session_vol,
            };

            let internals = RawMarketInternalsSnapshot {
                timestamp_ns: 0,
                nyse_tick,
                nasdaq_tick,
                add_breadth,
                vold_up_down_ratio: vold_ratio,
                vix_spot,
                vix_front_month: vix_front,
                sector_relative_flow: sector_flow,
            };

            let engine = RawFingerprintEngine::new();
            engine.build_sequence_tensor(&bars, &opts, &sf, &internals)
        });

        match result {
            Ok(tensor) => {
                let weights = [
                    tensor.stream_weights.chart_tape_physics_weight,
                    tensor.stream_weights.options_gamma_vanna_weight,
                    tensor.stream_weights.short_float_squeeze_weight,
                    tensor.stream_weights.market_internals_macro_weight,
                ];
                Ok((tensor.flat_tensor, tensor.latent_embedding_128, weights))
            }
            Err(err) => Err(pyo3::exceptions::PyValueError::new_err(err.to_string())),
        }
    }

    pub fn attach_metadata_dict() -> PyResult<String> {
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("engine", "raw-fingerprint-core")?;
            dict.set_item("feature_dim", 32usize)?;
            let any_ref = dict.as_any();
            let casted = any_ref.cast::<PyDict>()?;
            Ok(casted.len().to_string())
        })
    }

    #[pymodule]
    pub fn raw_fingerprint_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(build_fingerprint_tensor_py, m)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cabi_zero_skew_bridge() {
        let c_bars = [
            CAbiRawBar {
                timestamp_ns: 1_000,
                open_scaled: 1,  // $0.00001
                high_scaled: 3,  // $0.00003
                low_scaled: 1,   // $0.00001
                close_scaled: 2, // $0.00002
                volume: 500_000,
                vwap_scaled: 2,
                dollar_turnover: 10.0,
                trade_count: 50,
                spread_scaled: 1,
                time_delta_ms: 1000,
            },
            CAbiRawBar {
                timestamp_ns: 2_000,
                open_scaled: 2,
                high_scaled: 5,
                low_scaled: 2,
                close_scaled: 4, // $0.00004
                volume: 1_200_000,
                vwap_scaled: 3,
                dollar_turnover: 36.0,
                trade_count: 110,
                spread_scaled: 1,
                time_delta_ms: 1000,
            },
        ];
        let opts = CAbiOptionsSnapshot {
            timestamp_ns: 2_000,
            call_volume: 15_000,
            put_volume: 4_000,
            total_open_interest: 50_000,
            ask_fill_aggression_ratio: 0.78,
            bid_fill_aggression_ratio: 0.22,
            net_dealer_gamma_gex: 42.5,
            net_dealer_vanna: 18.0,
            zero_dte_volume_share: 0.65,
            front_iv: 0.62,
            back_iv: 0.45,
            dominant_gamma_wall_scaled: 5,
        };
        let sf = CAbiShortFloatSnapshot {
            timestamp_ns: 2_000,
            public_float_shares: 10_000_000,
            shares_outstanding: 11_000_000,
            short_interest_shares: 3_500_000,
            days_to_cover: 4.2,
            intraday_short_volume_ratio: 0.58,
            cost_to_borrow_annual_pct: 48.5,
            cumulative_session_volume: 1_700_000,
        };
        let int = CAbiMarketInternalsSnapshot {
            timestamp_ns: 2_000,
            nyse_tick: 640,
            nasdaq_tick: 820,
            add_breadth: 1450,
            vold_up_down_ratio: 3.1,
            vix_spot: 14.2,
            vix_front_month: 15.8,
            sector_relative_flow: 0.62,
        };

        let mut out_tensor = vec![0.0_f32; 2 * FINGERPRINT_FEATURE_DIM];
        let mut out_emb = vec![0.0_f32; TRACE_EMBEDDING_DIM];
        let mut out_w = vec![0.0_f32; 4];

        let status = unsafe {
            argon_build_fingerprint_tensor_cabi(
                c_bars.as_ptr(),
                c_bars.len(),
                &opts,
                &sf,
                &int,
                out_tensor.as_mut_ptr(),
                out_emb.as_mut_ptr(),
                out_w.as_mut_ptr(),
            )
        };
        assert_eq!(status, 0);
        let weight_sum: f32 = out_w.iter().sum();
        assert!((weight_sum - 1.0).abs() < 1e-5);
    }
}
