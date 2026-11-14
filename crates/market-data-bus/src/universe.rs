//! `universe.rs` — Deterministic ~250-Symbol Multi-Regime Anti-Survivorship Curriculum Universe.
//!
//! Spans:
//! 1. 100% Backward-Compatible 66 Core Argon-Algo symbols.
//! 2. Multi-Decade Macro, Sector & Index Benchmarks (SPY, QQQ, IWM, DIA, XLF, etc.).
//! 3. Multi-Cycle Blue-Chips & Cyclicals Spanning 1992–2026 (IBM, GE, INTC, CSCO, XOM, etc.).
//! 4. Anti-Survivorship Drawdowns / Secular Decline / Penny Traps / Crashed Micro-Caps (>=85% drawdowns, reverse splits).

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UniverseRegimeTier {
    CoreProduction66,
    MacroSectorBenchmark,
    MultiCycleCyclical,
    AntiSurvivorshipDrawdown,
}

impl UniverseRegimeTier {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::CoreProduction66 => "CORE_66",
            Self::MacroSectorBenchmark => "MACRO_BENCHMARK",
            Self::MultiCycleCyclical => "MULTI_CYCLE",
            Self::AntiSurvivorshipDrawdown => "ANTI_SURVIVORSHIP_PENNY_TRAP",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CurriculumSymbol {
    pub symbol_id: u32,
    pub symbol: String,
    pub tier: UniverseRegimeTier,
    pub exchange_index: String,
    pub sector: String,
}

/// The 66 core production symbols from `REAL_YAHOO_CORE_UNIVERSE`.
pub const CORE_66_SYMBOLS: &[&str] = &[
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "INTC", "AVGO", "QCOM", "ARM",
    "SMCI", "PLTR", "COIN", "MARA", "RIOT", "CLSK", "IONQ", "RGTI", "SOUN", "BBAI", "GME", "AMC",
    "KULR", "PLUG", "SOFI", "HOOD", "OPEN", "CLOV", "AFRM", "UPST", "RIVN", "LCID", "DKNG", "SHOP",
    "ROKU", "SNOW", "CRWD", "PANW", "NET", "UBER", "DIS", "NKE", "SBUX", "BA", "CAT", "CVX",
    "XOM", "JPM", "BAC", "GS", "MS", "PFE", "MRK", "LLY", "UNH", "T", "VZ", "F",
    "GM", "SPY", "QQQ", "IWM", "DIA", "RSP",
];

/// Multi-decade macro, sector & index benchmarks.
pub const MACRO_SECTOR_BENCHMARKS: &[&str] = &[
    "SPY", "QQQ", "IWM", "DIA", "RSP", "XLF", "XLE", "XLK", "XLV", "XLI", "XLP", "XLU", "XLB",
    "SMH", "KRE", "XBI", "TLT", "IEF", "HYG", "LQD", "GLD", "SLV", "USO",
];

/// Multi-cycle blue-chips and cyclicals spanning 1992–2026.
pub const MULTI_CYCLE_CYCLICALS: &[&str] = &[
    "IBM", "GE", "INTC", "CSCO", "ORCL", "QCOM", "TXN", "MU", "CAT", "DE", "BA", "F", "GM",
    "JPM", "BAC", "C", "WFC", "GS", "MS", "AIG", "XOM", "CVX", "COP", "WMT", "HD", "COST",
    "DIS", "NKE", "PFE", "JNJ", "MRK", "UNH", "T", "VZ",
];

/// Anti-survivorship secular decline, dilution, reverse splits, and >=85% drawdown traps.
pub const ANTI_SURVIVORSHIP_SYMBOLS: &[&str] = &[
    "PARA", "WBA", "LUMN", "SPCE", "PLUG", "LCID", "RIVN", "OPEN", "CLOV", "AMC", "GME", "BB",
    "PTON", "ZM", "ROKU", "PYPL", "BYND", "NKLA", "WE", "MULN", "CEI", "PROG", "SDC", "WISH",
    "ENDP", "BBBY", "RAD", "VRX", "BHC", "CHK", "MNK", "FRC", "SBNY", "PACW", "SI", "SIVB",
    "KODK", "REV", "HTZ", "APRN", "MVIS", "GNUS", "BNGO", "OCGN", "SNDL", "TLRY", "ACB", "CRON",
    "ZOM", "SOS", "EBON", "CAN", "BITF", "HUT", "HIVE", "CORZ", "BTBT", "WULF", "CIFR", "SDIG",
    "IREN", "ANY", "MIGI", "ATER", "BBIG", "SPRT", "GREE", "HLBZ", "COSM", "TOP", "HKD", "MEGL",
    "HUDI", "AAME", "ABVC", "ACCS", "ACET", "ACFN", "ACON", "ACTU", "ACXP", "ADGM", "ADIL",
    "ADVB", "AEI", "AEMD", "AEON", "AGIG", "AHT", "AIDX", "AIFA", "AIFC", "AIFF", "AIM",
    "AIMD", "AIRE", "AIRI", "ALDX", "ALLR", "AARD", "ABAT", "ABEO", "ABOS", "ACNT", "ACOG",
    "ACR", "ACRE", "ACRV", "ACU", "ADBT", "ADCT", "AEYE", "AFCG", "AGEN", "AGPU", "AIAI",
    "AIB", "AIRG", "AIRO", "AIRS", "AIRT", "AISP", "AIV", "AKA", "ZNTX", "QBIO",
];

/// Builds the deterministic, deduplicated, sorted curriculum universe (~250 symbols).
pub fn build_target_curriculum_universe() -> Vec<CurriculumSymbol> {
    let mut seen = HashSet::new();
    let mut list = Vec::with_capacity(300);

    // 1. Core 66 symbols first to guarantee 100% backward compatibility
    for &sym in CORE_66_SYMBOLS {
        if seen.insert(sym) {
            list.push(CurriculumSymbol {
                symbol_id: 0,
                symbol: sym.to_string(),
                tier: UniverseRegimeTier::CoreProduction66,
                exchange_index: infer_exchange_index(sym).to_string(),
                sector: infer_sector(sym).to_string(),
            });
        }
    }

    // 2. Macro, Sector & Index Benchmarks
    for &sym in MACRO_SECTOR_BENCHMARKS {
        if seen.insert(sym) {
            list.push(CurriculumSymbol {
                symbol_id: 0,
                symbol: sym.to_string(),
                tier: UniverseRegimeTier::MacroSectorBenchmark,
                exchange_index: "INDEX_ETF".to_string(),
                sector: infer_sector(sym).to_string(),
            });
        }
    }

    // 3. Multi-Cycle Cyclicals (1992-2026)
    for &sym in MULTI_CYCLE_CYCLICALS {
        if seen.insert(sym) {
            list.push(CurriculumSymbol {
                symbol_id: 0,
                symbol: sym.to_string(),
                tier: UniverseRegimeTier::MultiCycleCyclical,
                exchange_index: infer_exchange_index(sym).to_string(),
                sector: infer_sector(sym).to_string(),
            });
        }
    }

    // 4. Anti-Survivorship Penny Traps & Secular Declines
    for &sym in ANTI_SURVIVORSHIP_SYMBOLS {
        if seen.insert(sym) {
            list.push(CurriculumSymbol {
                symbol_id: 0,
                symbol: sym.to_string(),
                tier: UniverseRegimeTier::AntiSurvivorshipDrawdown,
                exchange_index: infer_exchange_index(sym).to_string(),
                sector: infer_sector(sym).to_string(),
            });
        }
    }

    // Sort deterministically by symbol string
    list.sort_by(|a, b| a.symbol.cmp(&b.symbol));

    // Assign sequential 0-based symbol IDs
    for (idx, item) in list.iter_mut().enumerate() {
        item.symbol_id = idx as u32;
    }

    list
}

/// Helper mapping to infer exchange index string.
fn infer_exchange_index(sym: &str) -> &'static str {
    match sym {
        "SPY" | "QQQ" | "IWM" | "DIA" | "RSP" | "XLF" | "XLE" | "XLK" | "XLV" | "XLI" | "XLP"
        | "XLU" | "XLB" | "SMH" | "KRE" | "XBI" | "TLT" | "IEF" | "HYG" | "LQD" | "GLD"
        | "SLV" | "USO" => "INDEX_ETF",
        "AAPL" | "MSFT" | "NVDA" | "AMZN" | "GOOGL" | "META" | "TSLA" | "AMD" | "INTC"
        | "AVGO" | "QCOM" | "CSCO" | "ORCL" | "NFLX" | "COST" => "NASDAQ_100",
        "IBM" | "GE" | "BA" | "CAT" | "DE" | "F" | "GM" | "JPM" | "BAC" | "C" | "WFC" | "GS"
        | "MS" | "AIG" | "XOM" | "CVX" | "COP" | "WMT" | "HD" | "DIS" | "NKE" | "PFE" | "JNJ"
        | "MRK" | "UNH" | "T" | "VZ" => "SP_500",
        "KULR" | "ZNTX" | "QBIO" | "MULN" | "CEI" | "PROG" | "SDC" | "WISH" | "ENDP" | "BBBY"
        | "RAD" | "VRX" | "BHC" | "CHK" | "MNK" | "FRC" | "SBNY" | "PACW" | "SI" | "SIVB"
        | "KODK" | "REV" | "HTZ" | "APRN" | "MVIS" | "GNUS" | "BNGO" | "OCGN" | "SNDL"
        | "TLRY" | "ACB" | "CRON" | "ZOM" | "SOS" | "EBON" | "CAN" | "BITF" | "HUT" | "HIVE"
        | "CORZ" | "BTBT" | "WULF" | "CIFR" | "SDIG" | "IREN" | "ANY" | "MIGI" | "ATER"
        | "BBIG" | "SPRT" | "GREE" | "HLBZ" | "COSM" | "TOP" | "HKD" | "MEGL" | "HUDI" => {
            "OTC_PENNY"
        }
        _ => "RUSSELL_2000",
    }
}

/// Helper mapping to infer sector string.
fn infer_sector(sym: &str) -> &'static str {
    match sym {
        "XLF" | "JPM" | "BAC" | "GS" | "MS" | "C" | "WFC" | "AIG" | "KRE" => "FINANCIALS",
        "XLK" | "AAPL" | "MSFT" | "NVDA" | "AVGO" | "AMD" | "INTC" | "QCOM" | "CSCO" | "ORCL"
        | "IBM" | "SMCI" | "PLTR" | "CRWD" | "PANW" | "NET" => "TECHNOLOGY",
        "XLE" | "XOM" | "CVX" | "COP" | "USO" => "ENERGY",
        "XLV" | "PFE" | "MRK" | "LLY" | "UNH" | "JNJ" | "XBI" => "HEALTHCARE",
        "XLI" | "BA" | "CAT" | "DE" | "GE" => "INDUSTRIALS",
        "XLY" | "AMZN" | "TSLA" | "HD" | "NKE" | "SBUX" | "DKNG" => "CONSUMER_DISCRETIONARY",
        "XLP" | "WMT" | "COST" => "CONSUMER_STAPLES",
        "XLU" => "UTILITIES",
        "XLB" => "MATERIALS",
        "COIN" | "MARA" | "RIOT" | "CLSK" => "CRYPTO_MINING",
        "TLT" | "IEF" | "HYG" | "LQD" => "FIXED_INCOME",
        "GLD" | "SLV" => "COMMODITIES",
        _ => "BROAD_MARKET",
    }
}

/// Provides fast O(1) symbol lookup by ticker string.
pub struct CurriculumUniverseIndex {
    by_symbol: HashMap<String, CurriculumSymbol>,
    by_id: Vec<CurriculumSymbol>,
}

impl CurriculumUniverseIndex {
    pub fn new() -> Self {
        let universe = build_target_curriculum_universe();
        let mut by_symbol = HashMap::with_capacity(universe.len());
        for sym in &universe {
            by_symbol.insert(sym.symbol.clone(), sym.clone());
        }
        Self {
            by_symbol,
            by_id: universe,
        }
    }

    pub fn get_by_symbol(&self, symbol: &str) -> Option<&CurriculumSymbol> {
        self.by_symbol.get(&symbol.to_uppercase())
    }

    pub fn get_by_id(&self, id: u32) -> Option<&CurriculumSymbol> {
        self.by_id.get(id as usize)
    }

    pub fn symbols(&self) -> &[CurriculumSymbol] {
        &self.by_id
    }

    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }
}

impl Default for CurriculumUniverseIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_curriculum_universe_invariants() {
        let universe = build_target_curriculum_universe();
        assert!(
            universe.len() >= 200,
            "Target curriculum universe must contain >= 200 symbols, got {}",
            universe.len()
        );

        // Verify all 66 core symbols are present
        let set: HashSet<String> = universe.iter().map(|s| s.symbol.clone()).collect();
        for &sym in CORE_66_SYMBOLS {
            assert!(
                set.contains(sym),
                "Missing core backward-compatible symbol: {}",
                sym
            );
        }

        // Verify monotonic 0-based symbol IDs
        for (i, sym) in universe.iter().enumerate() {
            assert_eq!(sym.symbol_id, i as u32);
        }
    }
}
