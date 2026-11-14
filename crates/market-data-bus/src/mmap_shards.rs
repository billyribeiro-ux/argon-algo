//! `mmap_shards.rs` — Zero-Copy Memory-Mapped (`memmap2`) Structure-of-Arrays (SoA) Binary Shard Engine.
//!
//! Features:
//! - 64-byte cache-line aligned, deterministic binary layout (`data/shards/{timeframe}/shard.bin`).
//! - Little-endian Pod / Zeroable structures via `bytemuck`.
//! - Instant zero-copy cold-open via `memmap2::Mmap`.
//! - Multi-timeframe causal rollup (1m -> 5m, 15m, 1h, 1d) with zero lookahead.
//! - Accurate `America/New_York` DST session tagging via `chrono-tz` (1992–2026).
//! - Accompanied by `manifest.json`.

use bytemuck::{Pod, Zeroable};
use chrono::{Datelike, Timelike};
use chrono_tz::America::New_York;
use crc32fast::Hasher;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use thiserror::Error;
use tracing::info;

pub const SHARD_MAGIC: &[u8; 8] = b"ARGNSHD1";
pub const SHARD_VERSION: u32 = 1;

#[derive(Debug, Error)]
pub enum ShardError {
    #[error("I/O error at {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("Corrupt shard header in {path}: invalid magic bytes {magic:?}")]
    InvalidMagic { path: String, magic: [u8; 8] },
    #[error("Unsupported shard version {version} in {path}")]
    UnsupportedVersion { path: String, version: u32 },
    #[error("Checksum mismatch in {path}: expected {expected:#x}, got {computed:#x}")]
    ChecksumMismatch {
        path: String,
        expected: u32,
        computed: u32,
    },
    #[error("Alignment error at offset {offset}: must be 64-byte aligned")]
    MisalignedOffset { offset: usize },
    #[error("Buffer bounds error in shard: {0}")]
    BoundsError(String),
}

/// 64-byte aligned Shard Header.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardHeader {
    pub magic: [u8; 8],           // b"ARGNSHD1"
    pub version: u32,             // 1
    pub timeframe_minutes: u32,   // 1, 5, 15, 60, 1440
    pub num_symbols: u32,
    pub checksum: u32,            // CRC32
    pub total_bars: u64,
    pub min_ts_sec: i64,
    pub max_ts_sec: i64,
    pub symbols_table_offset: u64,
    pub days_table_offset: u64,
    pub total_days: u32,
    pub _padding: [u8; 4],
}

/// 64-byte Symbol Directory Entry (64 bytes, 8-byte aligned, zero padding).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardSymbolEntry {
    pub bar_start_idx: u64,
    pub bar_len: u64,
    pub day_index_start: u64,
    pub symbol_id: u32,
    pub day_count: u32,
    pub tier: u32,
    pub _pad0: u32,
    pub ticker: [u8; 16],          // ASCII, null-padded
    pub _padding: [u8; 8],
}

impl ShardSymbolEntry {
    pub fn ticker_str(&self) -> &str {
        let len = self.ticker.iter().position(|&b| b == 0).unwrap_or(self.ticker.len());
        std::str::from_utf8(&self.ticker[..len]).unwrap_or("")
    }
}

/// 32-byte (Symbol, TradingDay) O(1) Index Entry (32 bytes, 4-byte aligned, zero padding).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq, Serialize, Deserialize)]
pub struct ShardDayIndexEntry {
    pub day_yyyymmdd: u32,
    pub rth_start_offset: u32,
    pub rth_bar_count: u16,
    pub session_flag: u16,
    pub day_open_raw: f32,
    pub day_high_raw: f32,
    pub day_low_raw: f32,
    pub day_close_raw: f32,
    pub _pad0: [u8; 4],
}

/// Structure representing a contiguous Bar before SoA serialization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IngestedBar {
    pub ts_sec: i64,
    pub adj_open: f32,
    pub adj_high: f32,
    pub adj_low: f32,
    pub adj_close: f32,
    pub adj_volume: f32,
    pub raw_close: f32,
    pub raw_split_factor: f32,
    pub div_factor: f32,
    pub session_mins_rem: u16,
    pub flags: u16,
}

/// Human-readable manifest JSON stored alongside `shard.bin`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardManifest {
    pub magic: String,
    pub version: u32,
    pub timeframe: String,
    pub timeframe_minutes: u32,
    pub num_symbols: u32,
    pub total_bars: u64,
    pub total_days: u32,
    pub min_date_iso: String,
    pub max_date_iso: String,
    pub file_size_bytes: u64,
    pub checksum_crc32: String,
    pub columns: Vec<String>,
}

/// Zero-copy read-only view over a memory-mapped shard file.
pub struct MmapShardReader {
    _mmap: Mmap,
    header: ShardHeader,
    symbols: Vec<ShardSymbolEntry>,
    symbol_lookup: HashMap<String, usize>,
    days_slice: *const ShardDayIndexEntry,
    total_days: usize,
    // Column slices
    ts_sec: *const i64,
    adj_open: *const f32,
    adj_high: *const f32,
    adj_low: *const f32,
    adj_close: *const f32,
    adj_volume: *const f32,
    raw_close: *const f32,
    raw_split_factor: *const f32,
    div_factor: *const f32,
    session_mins_rem: *const u16,
    flags: *const u16,
    total_bars: usize,
}

// Safety: MmapShardReader only reads from immutable mmap memory.
unsafe impl Send for MmapShardReader {}
unsafe impl Sync for MmapShardReader {}

impl MmapShardReader {
    /// Opens and validates a memory-mapped shard file in microseconds.
    pub fn open(path: &Path) -> Result<Self, ShardError> {
        let file = File::open(path).map_err(|source| ShardError::Io {
            path: path.display().to_string(),
            source,
        })?;

        let mmap = unsafe {
            Mmap::map(&file).map_err(|source| ShardError::Io {
                path: path.display().to_string(),
                source,
            })?
        };

        if mmap.len() < std::mem::size_of::<ShardHeader>() {
            return Err(ShardError::BoundsError(format!(
                "File too small for header: {} bytes",
                mmap.len()
            )));
        }

        let header_slice = &mmap[..std::mem::size_of::<ShardHeader>()];
        let header: ShardHeader = *bytemuck::from_bytes(header_slice);

        if &header.magic != SHARD_MAGIC {
            return Err(ShardError::InvalidMagic {
                path: path.display().to_string(),
                magic: header.magic,
            });
        }

        if header.version != SHARD_VERSION {
            return Err(ShardError::UnsupportedVersion {
                path: path.display().to_string(),
                version: header.version,
            });
        }

        let num_symbols = header.num_symbols as usize;
        let sym_table_offset = header.symbols_table_offset as usize;
        let sym_size = std::mem::size_of::<ShardSymbolEntry>();
        let sym_table_bytes = num_symbols * sym_size;

        if mmap.len() < sym_table_offset + sym_table_bytes {
            return Err(ShardError::BoundsError("Symbols table out of bounds".into()));
        }

        let sym_bytes = &mmap[sym_table_offset..sym_table_offset + sym_table_bytes];
        let symbols_slice: &[ShardSymbolEntry] = bytemuck::cast_slice(sym_bytes);
        let symbols = symbols_slice.to_vec();

        let mut symbol_lookup = HashMap::with_capacity(num_symbols);
        for (idx, entry) in symbols.iter().enumerate() {
            symbol_lookup.insert(entry.ticker_str().to_string(), idx);
        }

        let total_days = header.total_days as usize;
        let days_offset = header.days_table_offset as usize;
        let day_size = std::mem::size_of::<ShardDayIndexEntry>();
        let days_bytes = total_days * day_size;

        if mmap.len() < days_offset + days_bytes {
            return Err(ShardError::BoundsError("Days table out of bounds".into()));
        }

        let days_ptr = mmap[days_offset..].as_ptr() as *const ShardDayIndexEntry;

        let total_bars = header.total_bars as usize;
        let mut curr_offset = align_64(days_offset + days_bytes);

        // Map contiguous columns
        macro_rules! map_col {
            ($t:ty) => {{
                let byte_len = total_bars * std::mem::size_of::<$t>();
                if mmap.len() < curr_offset + byte_len {
                    return Err(ShardError::BoundsError(format!(
                        "Column {} out of bounds",
                        stringify!($t)
                    )));
                }
                let ptr = mmap[curr_offset..].as_ptr() as *const $t;
                curr_offset = align_64(curr_offset + byte_len);
                ptr
            }};
        }

        let ts_sec = map_col!(i64);
        let adj_open = map_col!(f32);
        let adj_high = map_col!(f32);
        let adj_low = map_col!(f32);
        let adj_close = map_col!(f32);
        let adj_volume = map_col!(f32);
        let raw_close = map_col!(f32);
        let raw_split_factor = map_col!(f32);
        let div_factor = map_col!(f32);
        let session_mins_rem = map_col!(u16);
        let flags = map_col!(u16);
        let _ = curr_offset;

        Ok(Self {
            _mmap: mmap,
            header,
            symbols,
            symbol_lookup,
            days_slice: days_ptr,
            total_days,
            ts_sec,
            adj_open,
            adj_high,
            adj_low,
            adj_close,
            adj_volume,
            raw_close,
            raw_split_factor,
            div_factor,
            session_mins_rem,
            flags,
            total_bars,
        })
    }

    pub fn header(&self) -> &ShardHeader {
        &self.header
    }

    pub fn symbols(&self) -> &[ShardSymbolEntry] {
        &self.symbols
    }

    pub fn total_bars(&self) -> usize {
        self.total_bars
    }

    pub fn get_symbol_entry(&self, ticker: &str) -> Option<&ShardSymbolEntry> {
        let idx = self.symbol_lookup.get(&ticker.to_uppercase())?;
        self.symbols.get(*idx)
    }

    /// Returns zero-copy slice views over all SoA columns for a specific symbol.
    pub fn get_symbol_soa_view<'a>(&'a self, ticker: &str) -> Option<SymbolSoAView<'a>> {
        let entry = self.get_symbol_entry(ticker)?;
        let start = entry.bar_start_idx as usize;
        let len = entry.bar_len as usize;

        if start + len > self.total_bars {
            return None;
        }

        unsafe {
            Some(SymbolSoAView {
                symbol_id: entry.symbol_id,
                ticker: entry.ticker_str(),
                ts_sec: std::slice::from_raw_parts(self.ts_sec.add(start), len),
                adj_open: std::slice::from_raw_parts(self.adj_open.add(start), len),
                adj_high: std::slice::from_raw_parts(self.adj_high.add(start), len),
                adj_low: std::slice::from_raw_parts(self.adj_low.add(start), len),
                adj_close: std::slice::from_raw_parts(self.adj_close.add(start), len),
                adj_volume: std::slice::from_raw_parts(self.adj_volume.add(start), len),
                raw_close: std::slice::from_raw_parts(self.raw_close.add(start), len),
                raw_split_factor: std::slice::from_raw_parts(self.raw_split_factor.add(start), len),
                div_factor: std::slice::from_raw_parts(self.div_factor.add(start), len),
                session_mins_rem: std::slice::from_raw_parts(self.session_mins_rem.add(start), len),
                flags: std::slice::from_raw_parts(self.flags.add(start), len),
            })
        }
    }

    /// Returns all trading days for a symbol.
    pub fn get_symbol_days(&self, ticker: &str) -> Option<&[ShardDayIndexEntry]> {
        let entry = self.get_symbol_entry(ticker)?;
        let start = entry.day_index_start as usize;
        let len = entry.day_count as usize;

        if start + len > self.total_days {
            return None;
        }

        unsafe {
            Some(std::slice::from_raw_parts(self.days_slice.add(start), len))
        }
    }
}

/// Zero-copy SoA column slices for an individual symbol.
pub struct SymbolSoAView<'a> {
    pub symbol_id: u32,
    pub ticker: &'a str,
    pub ts_sec: &'a [i64],
    pub adj_open: &'a [f32],
    pub adj_high: &'a [f32],
    pub adj_low: &'a [f32],
    pub adj_close: &'a [f32],
    pub adj_volume: &'a [f32],
    pub raw_close: &'a [f32],
    pub raw_split_factor: &'a [f32],
    pub div_factor: &'a [f32],
    pub session_mins_rem: &'a [u16],
    pub flags: &'a [u16],
}

impl<'a> SymbolSoAView<'a> {
    pub fn len(&self) -> usize {
        self.ts_sec.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ts_sec.is_empty()
    }
}

/// Builder that compiles a sorted list of symbols and bars into a cache-aligned binary shard.
pub struct ShardBuilder {
    timeframe_minutes: u32,
    symbols_data: Vec<(String, u32, Vec<IngestedBar>)>, // (ticker, tier, bars)
}

impl ShardBuilder {
    pub fn new(timeframe_minutes: u32) -> Self {
        Self {
            timeframe_minutes,
            symbols_data: Vec::new(),
        }
    }

    pub fn add_symbol_bars(
        &mut self,
        ticker: &str,
        tier: u32,
        bars: Vec<IngestedBar>,
    ) {
        if !bars.is_empty() {
            self.symbols_data.push((ticker.to_uppercase(), tier, bars));
        }
    }

    /// Compiles and writes `shard.bin` + `manifest.json`.
    pub fn build_and_write(
        mut self,
        dest_dir: &Path,
    ) -> Result<(usize, u64), ShardError> {
        std::fs::create_dir_all(dest_dir).map_err(|source| ShardError::Io {
            path: dest_dir.display().to_string(),
            source,
        })?;

        // Sort symbols deterministically
        self.symbols_data.sort_by(|a, b| a.0.cmp(&b.0));

        let num_symbols = self.symbols_data.len();
        let mut total_bars: u64 = 0;
        let mut min_ts = i64::MAX;
        let mut max_ts = i64::MIN;

        for (_, _, bars) in &self.symbols_data {
            total_bars += bars.len() as u64;
            if let Some(first) = bars.first() {
                min_ts = min_ts.min(first.ts_sec);
            }
            if let Some(last) = bars.last() {
                max_ts = max_ts.max(last.ts_sec);
            }
        }

        if total_bars == 0 {
            min_ts = 0;
            max_ts = 0;
        }

        let mut symbol_entries = Vec::with_capacity(num_symbols);
        let mut day_entries = Vec::new();

        let mut cur_bar_idx: u64 = 0;
        let mut cur_day_idx: u64 = 0;

        for (sym_id, (ticker, tier, bars)) in self.symbols_data.iter().enumerate() {
            let start_bar = cur_bar_idx;
            let bar_len = bars.len() as u64;
            let start_day = cur_day_idx;

            // Group bars by trading day
            let mut days_map: BTreeMap<u32, Vec<(usize, IngestedBar)>> = BTreeMap::new();
            for (offset, bar) in bars.iter().enumerate() {
                let ymd = unix_sec_to_et_yyyymmdd(bar.ts_sec);
                days_map.entry(ymd).or_default().push((offset, *bar));
            }

            let day_count = days_map.len() as u32;

            for (ymd, day_bars) in days_map {
                let rth_bars: Vec<_> = day_bars.iter().filter(|(_, b)| (b.flags & 1) != 0).collect();
                let (rth_offset, rth_count) = if let Some(first) = rth_bars.first() {
                    (first.0 as u32, rth_bars.len() as u16)
                } else if let Some(first) = day_bars.first() {
                    (first.0 as u32, day_bars.len() as u16)
                } else {
                    (0, 0)
                };

                let first_bar = day_bars.first().map(|b| b.1).unwrap_or(bars[0]);
                let last_bar = day_bars.last().map(|b| b.1).unwrap_or(bars[0]);
                let max_high = day_bars.iter().map(|(_, b)| b.adj_high * b.raw_split_factor).fold(0.0_f32, f32::max);
                let min_low = day_bars.iter().map(|(_, b)| b.adj_low * b.raw_split_factor).fold(f32::MAX, f32::min);

                day_entries.push(ShardDayIndexEntry {
                    day_yyyymmdd: ymd,
                    rth_start_offset: rth_offset,
                    rth_bar_count: rth_count,
                    session_flag: 1,
                    day_open_raw: first_bar.adj_open * first_bar.raw_split_factor,
                    day_high_raw: max_high,
                    day_low_raw: if min_low == f32::MAX { first_bar.adj_low * first_bar.raw_split_factor } else { min_low },
                    day_close_raw: last_bar.raw_close,
                    _pad0: [0u8; 4],
                });
                cur_day_idx += 1;
            }

            let mut ticker_bytes = [0u8; 16];
            let raw_t = ticker.as_bytes();
            let copy_len = raw_t.len().min(16);
            ticker_bytes[..copy_len].copy_from_slice(&raw_t[..copy_len]);

            symbol_entries.push(ShardSymbolEntry {
                bar_start_idx: start_bar,
                bar_len,
                day_index_start: start_day,
                symbol_id: sym_id as u32,
                day_count,
                tier: *tier,
                _pad0: 0,
                ticker: ticker_bytes,
                _padding: [0u8; 8],
            });

            cur_bar_idx += bar_len;
        }

        // Layout byte offsets aligned to 64 bytes
        let header_size = std::mem::size_of::<ShardHeader>();
        let symbols_offset = align_64(header_size);
        let symbols_byte_size = symbol_entries.len() * std::mem::size_of::<ShardSymbolEntry>();
        let days_offset = align_64(symbols_offset + symbols_byte_size);
        let days_byte_size = day_entries.len() * std::mem::size_of::<ShardDayIndexEntry>();

        let mut header = ShardHeader {
            magic: *SHARD_MAGIC,
            version: SHARD_VERSION,
            timeframe_minutes: self.timeframe_minutes,
            num_symbols: num_symbols as u32,
            checksum: 0,
            total_bars,
            min_ts_sec: min_ts,
            max_ts_sec: max_ts,
            symbols_table_offset: symbols_offset as u64,
            days_table_offset: days_offset as u64,
            total_days: day_entries.len() as u32,
            _padding: [0u8; 4],
        };

        // Collect flat arrays
        let mut ts_sec_vec = Vec::with_capacity(total_bars as usize);
        let mut adj_open_vec = Vec::with_capacity(total_bars as usize);
        let mut adj_high_vec = Vec::with_capacity(total_bars as usize);
        let mut adj_low_vec = Vec::with_capacity(total_bars as usize);
        let mut adj_close_vec = Vec::with_capacity(total_bars as usize);
        let mut adj_volume_vec = Vec::with_capacity(total_bars as usize);
        let mut raw_close_vec = Vec::with_capacity(total_bars as usize);
        let mut raw_split_factor_vec = Vec::with_capacity(total_bars as usize);
        let mut div_factor_vec = Vec::with_capacity(total_bars as usize);
        let mut session_mins_rem_vec = Vec::with_capacity(total_bars as usize);
        let mut flags_vec = Vec::with_capacity(total_bars as usize);

        for (_, _, bars) in &self.symbols_data {
            for b in bars {
                ts_sec_vec.push(b.ts_sec);
                adj_open_vec.push(b.adj_open);
                adj_high_vec.push(b.adj_high);
                adj_low_vec.push(b.adj_low);
                adj_close_vec.push(b.adj_close);
                adj_volume_vec.push(b.adj_volume);
                raw_close_vec.push(b.raw_close);
                raw_split_factor_vec.push(b.raw_split_factor);
                div_factor_vec.push(b.div_factor);
                session_mins_rem_vec.push(b.session_mins_rem);
                flags_vec.push(b.flags);
            }
        }

        // Compute checksum
        let mut hasher = Hasher::new();
        hasher.update(bytemuck::cast_slice(&symbol_entries));
        hasher.update(bytemuck::cast_slice(&day_entries));
        hasher.update(bytemuck::cast_slice(&ts_sec_vec));
        hasher.update(bytemuck::cast_slice(&adj_close_vec));
        header.checksum = hasher.finalize();

        // Write binary shard
        let bin_path = dest_dir.join("shard.bin");
        let mut f = File::create(&bin_path).map_err(|source| ShardError::Io {
            path: bin_path.display().to_string(),
            source,
        })?;

        // 1. Header
        f.write_all(bytemuck::bytes_of(&header)).map_err(|source| ShardError::Io {
            path: bin_path.display().to_string(),
            source,
        })?;

        // Pad to symbols_offset
        write_padding(&mut f, symbols_offset - header_size)?;

        // 2. Symbols Table
        f.write_all(bytemuck::cast_slice(&symbol_entries)).map_err(|source| ShardError::Io {
            path: bin_path.display().to_string(),
            source,
        })?;

        // Pad to days_offset
        let written_so_far = symbols_offset + symbols_byte_size;
        write_padding(&mut f, days_offset - written_so_far)?;

        // 3. Days Table
        f.write_all(bytemuck::cast_slice(&day_entries)).map_err(|source| ShardError::Io {
            path: bin_path.display().to_string(),
            source,
        })?;

        let mut curr = days_offset + days_byte_size;

        macro_rules! write_col {
            ($vec:expr) => {{
                let target_offset = align_64(curr);
                write_padding(&mut f, target_offset - curr)?;
                let bytes = bytemuck::cast_slice(&$vec);
                f.write_all(bytes).map_err(|source| ShardError::Io {
                    path: bin_path.display().to_string(),
                    source,
                })?;
                curr = target_offset + bytes.len();
            }};
        }

        write_col!(ts_sec_vec);
        write_col!(adj_open_vec);
        write_col!(adj_high_vec);
        write_col!(adj_low_vec);
        write_col!(adj_close_vec);
        write_col!(adj_volume_vec);
        write_col!(raw_close_vec);
        write_col!(raw_split_factor_vec);
        write_col!(div_factor_vec);
        write_col!(session_mins_rem_vec);
        write_col!(flags_vec);

        f.flush().map_err(|source| ShardError::Io {
            path: bin_path.display().to_string(),
            source,
        })?;

        let total_file_bytes = curr as u64;

        // Write manifest.json
        let manifest = ShardManifest {
            magic: "ARGNSHD1".to_string(),
            version: SHARD_VERSION,
            timeframe: format_tf_label(self.timeframe_minutes),
            timeframe_minutes: self.timeframe_minutes,
            num_symbols: num_symbols as u32,
            total_bars,
            total_days: day_entries.len() as u32,
            min_date_iso: chrono::DateTime::from_timestamp(min_ts, 0)
                .map(|dt| dt.to_rfc3339())
                .unwrap_or_default(),
            max_date_iso: chrono::DateTime::from_timestamp(max_ts, 0)
                .map(|dt| dt.to_rfc3339())
                .unwrap_or_default(),
            file_size_bytes: total_file_bytes,
            checksum_crc32: format!("{:#010x}", header.checksum),
            columns: vec![
                "ts_sec: i64".into(),
                "adj_open: f32".into(),
                "adj_high: f32".into(),
                "adj_low: f32".into(),
                "adj_close: f32".into(),
                "adj_volume: f32".into(),
                "raw_close: f32".into(),
                "raw_split_factor: f32".into(),
                "div_factor: f32".into(),
                "session_mins_rem: u16".into(),
                "flags: u16".into(),
            ],
        };

        let manifest_path = dest_dir.join("manifest.json");
        let manifest_json = serde_json::to_string_pretty(&manifest).map_err(|e| {
            ShardError::BoundsError(format!("Failed to serialize manifest: {e}"))
        })?;
        std::fs::write(&manifest_path, manifest_json).map_err(|source| ShardError::Io {
            path: manifest_path.display().to_string(),
            source,
        })?;

        info!(
            "Compiled binary shard for {} ({} symbols, {} bars, {:.2} MB)",
            manifest.timeframe, num_symbols, total_bars, (total_file_bytes as f64) / 1_048_576.0
        );

        Ok((num_symbols, total_file_bytes))
    }
}

/// Helper converting unix timestamp to America/New_York (YYYYMMDD, HH, MM).
pub fn unix_sec_to_et(ts_sec: i64) -> (u32, u32, u32) {
    if let Some(utc_dt) = chrono::DateTime::from_timestamp(ts_sec, 0) {
        let et_dt = utc_dt.with_timezone(&New_York);
        let ymd = (et_dt.year() as u32) * 10000 + et_dt.month() * 100 + et_dt.day();
        (ymd, et_dt.hour(), et_dt.minute())
    } else {
        (0, 0, 0)
    }
}

pub fn unix_sec_to_et_yyyymmdd(ts_sec: i64) -> u32 {
    unix_sec_to_et(ts_sec).0
}

/// Tags an intraday bar with session flag and session minutes remaining.
/// Flags:
/// - bit 0: RTH flag (1 = Regular Market Hours 09:30-16:00 ET)
/// - bit 1: Pre-market (04:00-09:30 ET)
/// - bit 2: After-hours (16:00-20:00 ET)
pub fn tag_session_and_minutes_remaining(ts_sec: i64) -> (u16, u16) {
    let (_, h, m) = unix_sec_to_et(ts_sec);
    let mins_from_midnight = h * 60 + m;

    let rth_open_mins = 9 * 60 + 30; // 570 mins (09:30 ET)
    let rth_close_mins = 16 * 60;    // 960 mins (16:00 ET)

    if mins_from_midnight >= rth_open_mins && mins_from_midnight < rth_close_mins {
        let rem = (rth_close_mins - mins_from_midnight) as u16;
        (rem, 1) // RTH
    } else if mins_from_midnight >= 4 * 60 && mins_from_midnight < rth_open_mins {
        (0, 2) // Pre-Market
    } else if mins_from_midnight >= rth_close_mins && mins_from_midnight < 20 * 60 {
        (0, 4) // After-Hours
    } else {
        (0, 0) // Closed
    }
}

/// Rolls up 1-minute bars into a target timeframe (5m, 15m, 60m, 1440m/1d) causally.
pub fn rollup_1m_bars(bars_1m: &[IngestedBar], target_mins: u32) -> Vec<IngestedBar> {
    if target_mins <= 1 || bars_1m.is_empty() {
        return bars_1m.to_vec();
    }

    let mut rolled = Vec::new();
    let bucket_step_sec = (target_mins as i64) * 60;

    // Group by bucket timestamp
    let mut cur_bucket_ts = (bars_1m[0].ts_sec / bucket_step_sec) * bucket_step_sec;
    let mut cur_open = bars_1m[0].adj_open;
    let mut cur_high = bars_1m[0].adj_high;
    let mut cur_low = bars_1m[0].adj_low;
    let mut cur_close = bars_1m[0].adj_close;
    let mut cur_volume = 0.0_f32;
    let mut cur_raw_close = bars_1m[0].raw_close;
    let mut cur_raw_factor = bars_1m[0].raw_split_factor;
    let mut cur_div_factor = bars_1m[0].div_factor;
    let mut cur_rem = bars_1m[0].session_mins_rem;
    let mut cur_flags = bars_1m[0].flags;
    let mut has_bars = false;

    for b in bars_1m {
        let bucket = if target_mins >= 1440 {
            // For 1D, bucket by New York trading day start
            let ymd = unix_sec_to_et_yyyymmdd(b.ts_sec);
            crate::corporate_actions::yyyymmdd_to_unix_day_range(ymd).0 + 57600 // 16:00 ET close
        } else {
            (b.ts_sec / bucket_step_sec) * bucket_step_sec
        };

        if bucket != cur_bucket_ts && has_bars {
            rolled.push(IngestedBar {
                ts_sec: cur_bucket_ts,
                adj_open: cur_open,
                adj_high: cur_high,
                adj_low: cur_low,
                adj_close: cur_close,
                adj_volume: cur_volume,
                raw_close: cur_raw_close,
                raw_split_factor: cur_raw_factor,
                div_factor: cur_div_factor,
                session_mins_rem: cur_rem,
                flags: cur_flags,
            });

            cur_bucket_ts = bucket;
            cur_open = b.adj_open;
            cur_high = b.adj_high;
            cur_low = b.adj_low;
            cur_close = b.adj_close;
            cur_volume = b.adj_volume;
            cur_raw_close = b.raw_close;
            cur_raw_factor = b.raw_split_factor;
            cur_div_factor = b.div_factor;
            cur_rem = b.session_mins_rem;
            cur_flags = b.flags;
        } else {
            if !has_bars {
                cur_bucket_ts = bucket;
                cur_open = b.adj_open;
                cur_high = b.adj_high;
                cur_low = b.adj_low;
                has_bars = true;
            } else {
                cur_high = cur_high.max(b.adj_high);
                cur_low = cur_low.min(b.adj_low);
            }
            cur_close = b.adj_close;
            cur_volume += b.adj_volume;
            cur_raw_close = b.raw_close;
            cur_raw_factor = b.raw_split_factor;
            cur_div_factor = b.div_factor;
            cur_rem = b.session_mins_rem;
            cur_flags = b.flags;
        }
    }

    if has_bars {
        rolled.push(IngestedBar {
            ts_sec: cur_bucket_ts,
            adj_open: cur_open,
            adj_high: cur_high,
            adj_low: cur_low,
            adj_close: cur_close,
            adj_volume: cur_volume,
            raw_close: cur_raw_close,
            raw_split_factor: cur_raw_factor,
            div_factor: cur_div_factor,
            session_mins_rem: cur_rem,
            flags: cur_flags,
        });
    }

    rolled
}

fn align_64(offset: usize) -> usize {
    (offset + 63) & !63
}

fn write_padding(f: &mut File, pad_len: usize) -> Result<(), ShardError> {
    if pad_len > 0 {
        let pad = vec![0u8; pad_len];
        f.write_all(&pad).map_err(|source| ShardError::Io {
            path: "padding".into(),
            source,
        })?;
    }
    Ok(())
}

fn format_tf_label(mins: u32) -> String {
    match mins {
        1 => "1m".into(),
        5 => "5m".into(),
        15 => "15m".into(),
        60 => "1h".into(),
        1440 => "1d".into(),
        _ => format!("{mins}m"),
    }
}

use std::collections::BTreeMap;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shard_compilation_and_zero_copy_read() {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(12345);
        let temp_dir = std::env::temp_dir().join(format!("test_shard_{now_ns}"));
        let mut builder = ShardBuilder::new(5);

        let bars = vec![
            IngestedBar {
                ts_sec: 1718026200, // 09:30 ET
                adj_open: 120.0,
                adj_high: 121.0,
                adj_low: 119.5,
                adj_close: 120.5,
                adj_volume: 50_000.0,
                raw_close: 120.5,
                raw_split_factor: 1.0,
                div_factor: 1.0,
                session_mins_rem: 390,
                flags: 1,
            },
            IngestedBar {
                ts_sec: 1718026500, // 09:35 ET
                adj_open: 120.5,
                adj_high: 122.0,
                adj_low: 120.2,
                adj_close: 121.8,
                adj_volume: 80_000.0,
                raw_close: 121.8,
                raw_split_factor: 1.0,
                div_factor: 1.0,
                session_mins_rem: 385,
                flags: 1,
            },
        ];

        builder.add_symbol_bars("AAPL", 0, bars);
        builder.build_and_write(&temp_dir).expect("build shard");

        let reader = MmapShardReader::open(&temp_dir.join("shard.bin")).expect("open shard");
        assert_eq!(reader.header().num_symbols, 1);
        assert_eq!(reader.total_bars(), 2);

        let view = reader.get_symbol_soa_view("AAPL").expect("symbol view");
        assert_eq!(view.len(), 2);
        assert_eq!(view.adj_open[0], 120.0);
        assert_eq!(view.adj_close[1], 121.8);
        assert_eq!(view.session_mins_rem[0], 390);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
