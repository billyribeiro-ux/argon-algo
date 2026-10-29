//! 5-Decimal Sub-Penny Fixed-Point Price Arithmetic (`$0.00001` to `$10,000.00+`).
//!
//! Guarantees zero IEEE-754 binary floating-point drift on sub-penny OTC/micro-cap ticks
//! (`$0.00001`) while supporting mega-cap equities and index levels (`$10,000.00+`).

use std::fmt;
use std::ops::{Add, Sub};
use std::str::FromStr;

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Scale factor for 5-decimal sub-penny resolution: `100,000` ticks = `$1.00000`.
/// `1` scaled unit = `$0.00001` (one hundred-thousandth of a dollar).
pub const SUB_PENNY_SCALE: i64 = 100_000;
pub const SUB_PENNY_SCALE_F64: f64 = 100_000.0;
pub const MIN_SUB_PENNY_UNITS: i64 = 1; // $0.00001

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PriceError {
    #[error("Price must be at least $0.00001 (got {0} scaled units)")]
    BelowMinimumSubPenny(i64),
    #[error("Invalid price string format: '{0}'")]
    InvalidFormat(String),
    #[error("Arithmetic overflow in sub-penny calculation")]
    Overflow,
}

/// Exact 5-decimal fixed-point price representation (`1 unit = $0.00001`).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct SubPennyPrice(i64);

impl SubPennyPrice {
    /// Minimum allowed positive market price: `$0.00001`.
    pub const MIN_VALID: Self = Self(MIN_SUB_PENNY_UNITS);

    /// Zero price (`$0.00000`), used for optional fields such as spreads or option premiums.
    pub const ZERO: Self = Self(0);

    /// Creates a `SubPennyPrice` directly from scaled integer units (`1 = $0.00001`).
    #[inline]
    pub const fn from_scaled_units(units: i64) -> Result<Self, PriceError> {
        if units < MIN_SUB_PENNY_UNITS {
            Err(PriceError::BelowMinimumSubPenny(units))
        } else {
            Ok(Self(units))
        }
    }

    /// Creates a `SubPennyPrice` allowing zero (`$0.00000`) for spreads or unassigned option levels.
    #[inline]
    pub const fn from_scaled_units_allow_zero(units: i64) -> Result<Self, PriceError> {
        if units < 0 {
            Err(PriceError::BelowMinimumSubPenny(units))
        } else {
            Ok(Self(units))
        }
    }

    /// Converts an `f64` price into exact nearest 5-decimal scaled units.
    #[inline]
    pub fn from_f64(price: f64) -> Result<Self, PriceError> {
        if !price.is_finite() {
            return Err(PriceError::InvalidFormat(price.to_string()));
        }
        let scaled = (price * SUB_PENNY_SCALE_F64).round() as i64;
        Self::from_scaled_units(scaled)
    }

    /// Converts an `f64` price into exact 5-decimal scaled units, permitting `0.0`.
    #[inline]
    pub fn from_f64_allow_zero(price: f64) -> Result<Self, PriceError> {
        if !price.is_finite() || price < 0.0 {
            return Err(PriceError::InvalidFormat(price.to_string()));
        }
        let scaled = (price * SUB_PENNY_SCALE_F64).round() as i64;
        Self::from_scaled_units_allow_zero(scaled)
    }

    /// Converts from a `rust_decimal::Decimal`, rounding to 5 decimal places.
    #[inline]
    pub fn from_decimal(dec: Decimal) -> Result<Self, PriceError> {
        let scaled_dec = (dec * Decimal::from(SUB_PENNY_SCALE)).round();
        let units: i64 = scaled_dec.try_into().map_err(|_| PriceError::Overflow)?;
        Self::from_scaled_units(units)
    }

    /// Returns the raw integer units (`1 = $0.00001`).
    #[inline]
    pub const fn scaled_units(self) -> i64 {
        self.0
    }

    /// Converts to `f64` for neural network log-ratio math.
    #[inline]
    pub const fn to_f64(self) -> f64 {
        (self.0 as f64) / SUB_PENNY_SCALE_F64
    }

    /// Converts to an exact `rust_decimal::Decimal` with scale 5.
    #[inline]
    pub fn to_decimal(self) -> Decimal {
        Decimal::new(self.0, 5)
    }

    /// Scale-independent natural log ratio $\ln(\text{self} / \text{other})$.
    /// Identical physics whether comparing `$0.00002 / $0.00001` or `$2000.00 / $1000.00`.
    #[inline]
    pub fn log_ratio(self, reference: Self) -> f64 {
        let num = self.0.max(MIN_SUB_PENNY_UNITS) as f64;
        let den = reference.0.max(MIN_SUB_PENNY_UNITS) as f64;
        (num / den).ln()
    }

    /// Exact percentage change `(self - reference) / reference * 100.0` using integer-backed delta.
    #[inline]
    pub fn pct_change(self, reference: Self) -> f64 {
        let ref_units = reference.0.max(MIN_SUB_PENNY_UNITS) as f64;
        let delta_units = (self.0 - reference.0) as f64;
        (delta_units / ref_units) * 100.0
    }

    /// Absolute difference between two prices in `SubPennyPrice` units.
    #[inline]
    pub const fn abs_diff(self, other: Self) -> Self {
        let diff = (self.0 - other.0).abs();
        Self(diff)
    }

    /// Applies a relative basis-point or percentage multiplier and clamps to at least `$0.00001`.
    #[inline]
    pub fn apply_relative_offset(self, relative_ratio: f64) -> Self {
        let base = self.0 as f64;
        let updated = (base * (1.0 + relative_ratio)).round() as i64;
        Self(updated.max(MIN_SUB_PENNY_UNITS))
    }

    /// Formats the price with adaptive institutional precision:
    /// - 5 decimals (`$0.00001`) for sub-penny and penny stocks (`< $1.00`)
    /// - 4 decimals for `$1.00` to `$5.00` when fractional tenth-cents exist
    /// - 2 to 5 decimals preserving exact non-zero sub-penny digits
    pub fn format_institutional(self) -> String {
        let sign = if self.0 < 0 { "-" } else { "" };
        let abs_val = self.0.unsigned_abs();
        let dollars = abs_val / (SUB_PENNY_SCALE as u64);
        let frac = abs_val % (SUB_PENNY_SCALE as u64);

        if abs_val < (SUB_PENNY_SCALE as u64) || !frac.is_multiple_of(1_000) {
            format!("{sign}${dollars}.{frac:05}")
        } else {
            let cents = frac / 1_000;
            format!("{sign}${dollars}.{cents:02}")
        }
    }
}

impl fmt::Debug for SubPennyPrice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let abs_val = self.0.unsigned_abs();
        let dollars = abs_val / (SUB_PENNY_SCALE as u64);
        let frac = abs_val % (SUB_PENNY_SCALE as u64);
        if self.0 < 0 {
            write!(f, "SubPennyPrice(-${dollars}.{frac:05})")
        } else {
            write!(f, "SubPennyPrice(${dollars}.{frac:05})")
        }
    }
}

impl fmt::Display for SubPennyPrice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let abs_val = self.0.unsigned_abs();
        let dollars = abs_val / (SUB_PENNY_SCALE as u64);
        let frac = abs_val % (SUB_PENNY_SCALE as u64);
        if self.0 < 0 {
            write!(f, "-{dollars}.{frac:05}")
        } else {
            write!(f, "{dollars}.{frac:05}")
        }
    }
}

impl FromStr for SubPennyPrice {
    type Err = PriceError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed = s.trim().trim_start_matches('$').replace(',', "");
        let dec =
            Decimal::from_str(&trimmed).map_err(|_| PriceError::InvalidFormat(s.to_string()))?;
        Self::from_decimal(dec)
    }
}

impl Add for SubPennyPrice {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0.saturating_add(rhs.0))
    }
}

impl Sub for SubPennyPrice {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0.saturating_sub(rhs.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sub_penny_exact_arithmetic_no_float_drift() {
        let p1 = SubPennyPrice::from_str("$0.00001").expect("valid $0.00001");
        let p2 = SubPennyPrice::from_str("$0.00002").expect("valid $0.00002");
        let p3 = p1 + p2;
        assert_eq!(p3.scaled_units(), 3);
        assert_eq!(p3.to_string(), "0.00003");
        assert_eq!(p3.format_institutional(), "$0.00003");

        // Verify scale-independent log ratio symmetry between $0.00001 -> $0.00002 and $1000 -> $2000
        let mega_1 = SubPennyPrice::from_str("$1000.00000").expect("valid $1000");
        let mega_2 = SubPennyPrice::from_str("$2000.00000").expect("valid $2000");

        let log_sub_penny = p2.log_ratio(p1);
        let log_mega_cap = mega_2.log_ratio(mega_1);
        assert!((log_sub_penny - log_mega_cap).abs() < 1e-15);
        assert!((p2.pct_change(p1) - 100.0).abs() < 1e-12);
        assert!((mega_2.pct_change(mega_1) - 100.0).abs() < 1e-12);
    }

    #[test]
    fn test_decimal_round_trip() {
        let prices = [
            "0.00001",
            "0.00015",
            "0.12345",
            "4.99999",
            "145.67000",
            "12450.75000",
        ];
        for s in prices {
            let parsed = SubPennyPrice::from_str(s).expect("parse price");
            assert_eq!(parsed.to_string(), s);
            let dec = parsed.to_decimal();
            let round_trip = SubPennyPrice::from_decimal(dec).expect("round-trip decimal");
            assert_eq!(parsed, round_trip);
        }
    }
}
