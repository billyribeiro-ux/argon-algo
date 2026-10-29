//! `raw-fingerprint-core` — Shared Scale-Independent 4-Stream Raw Physics Engine.
//!
//! Enforces Law 2 (Zero Lagging Human Indicators) and Law 3 (Zero Training-Serving Skew)
//! across `$0.00001` (5-decimal sub-penny) to `$10,000.00+` price scales.

pub mod ffi;
pub mod price;
pub mod streams;
pub mod tensor;

pub use price::{MIN_SUB_PENNY_UNITS, PriceError, SUB_PENNY_SCALE, SubPennyPrice};
pub use streams::{
    AutonomousSignalBlueprint, BrainHorizonId, FourStreamWeights, PrimaryFingerprintDriver,
    RawBarRecord, RawMarketInternalsSnapshot, RawOptionsSnapshot, RawShortFloatSnapshot,
    TraceHealthBadge, TradeDirection,
};
pub use tensor::{
    FINGERPRINT_FEATURE_DIM, FingerprintError, LiveTraceHealthEvaluation,
    NormalizedFingerprintTensor, OpenPositionEvaluationContext, RawFingerprintEngine,
    TRACE_EMBEDDING_DIM,
};
