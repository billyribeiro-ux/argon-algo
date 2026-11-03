//! Real-Time Outbound Multi-Channel Webhook Dispatcher (`webhooks.rs`).
//!
//! Formats and dispatches institutional execution lifecycle alerts across:
//! 1. **Discord Rich Embeds** (color-coded hex bars with structured field grids)
//! 2. **Telegram Bot API** (Markdown/HTML formatted institutional desk alerts)
//! 3. **Slack Block Kit** (header, section fields, and context blocks)
//! 4. **Generic JSON Webhooks** (canonical machine-readable telemetry schema)
//!
//! Automatically emits outbound events on 4 lifecycle transitions:
//! - `SIGNAL_ENTRY_OPENED`
//! - `SCALE_IN_TRANCHE_ADDED`
//! - `TARGET_1_PARTIAL_BANKED_AND_STOP_RATCHETED`
//! - `POSITION_CLOSED` (`TARGET HIT`, `TRAILING STOP`, `AI DYNAMIC EXIT`, `STOPPED OUT`)
//!
//! Maintains a bounded ring-buffer **Webhook Outbox & Delivery Audit Log** (`VecDeque<WebhookDispatchRecord>`,
//! max 100 records, persisted in `data/ledger_state.json`) so users can inspect every formatted
//! Discord/Telegram/Slack payload in real time even before external URLs are configured.

use std::collections::VecDeque;

use chrono::{TimeZone, Utc};
use parking_lot::RwLock;
use raw_fingerprint_core::{
    BrainHorizonId, FourStreamWeights, SubPennyPrice, TradeDirection,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use uuid::Uuid;

/// Maximum number of webhook dispatch records retained in the persistent ring-buffer outbox.
pub const MAX_WEBHOOK_OUTBOX_RECORDS: usize = 100;

/// Lifecycle transition event type for outbound institutional execution alerts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WebhookEventType {
    #[serde(rename = "SIGNAL_ENTRY_OPENED")]
    SignalEntryOpened,
    #[serde(rename = "SCALE_IN_TRANCHE_ADDED")]
    ScaleInTrancheAdded,
    #[serde(rename = "TARGET_1_PARTIAL_BANKED_AND_STOP_RATCHETED")]
    Target1PartialBankedAndStopRatcheted,
    #[serde(rename = "POSITION_CLOSED")]
    PositionClosed,
}

impl WebhookEventType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SignalEntryOpened => "SIGNAL_ENTRY_OPENED",
            Self::ScaleInTrancheAdded => "SCALE_IN_TRANCHE_ADDED",
            Self::Target1PartialBankedAndStopRatcheted => {
                "TARGET_1_PARTIAL_BANKED_AND_STOP_RATCHETED"
            }
            Self::PositionClosed => "POSITION_CLOSED",
        }
    }

    pub const fn display_badge(self) -> &'static str {
        match self {
            Self::SignalEntryOpened => "🚀 SIGNAL ENTRY OPENED",
            Self::ScaleInTrancheAdded => "➕ SCALE-IN TRANCHE ADDED",
            Self::Target1PartialBankedAndStopRatcheted => "🏦 50% BANKED @ T1 & STOP RATCHETED",
            Self::PositionClosed => "🏁 POSITION CLOSED",
        }
    }
}

/// User-configurable outbound webhook channel endpoints and toggles.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WebhookChannelConfig {
    #[serde(default)]
    pub discord_webhook_url: String,
    #[serde(default)]
    pub discord_enabled: bool,
    #[serde(default)]
    pub telegram_bot_token: String,
    #[serde(default)]
    pub telegram_chat_id: String,
    #[serde(default)]
    pub telegram_enabled: bool,
    #[serde(default)]
    pub slack_webhook_url: String,
    #[serde(default)]
    pub slack_enabled: bool,
    #[serde(default)]
    pub custom_json_webhook_url: String,
    #[serde(default)]
    pub custom_json_enabled: bool,
}

impl WebhookChannelConfig {
    /// Returns the list of active external channels that have both their toggle enabled and non-empty credentials.
    pub fn active_channel_names(&self) -> Vec<String> {
        let mut channels = Vec::new();
        if self.discord_enabled && !self.discord_webhook_url.trim().is_empty() {
            channels.push("DISCORD".to_string());
        }
        if self.telegram_enabled
            && !self.telegram_bot_token.trim().is_empty()
            && !self.telegram_chat_id.trim().is_empty()
        {
            channels.push("TELEGRAM".to_string());
        }
        if self.slack_enabled && !self.slack_webhook_url.trim().is_empty() {
            channels.push("SLACK".to_string());
        }
        if self.custom_json_enabled && !self.custom_json_webhook_url.trim().is_empty() {
            channels.push("CUSTOM_JSON".to_string());
        }
        if channels.is_empty() {
            channels.push("LOCAL_OUTBOX".to_string());
        }
        channels
    }
}

/// Input telemetry for constructing a multi-channel execution alert.
#[derive(Debug, Clone)]
pub struct WebhookEventPayload {
    pub event_type: WebhookEventType,
    pub outcome_detail: String,
    pub symbol: String,
    pub brain_id: BrainHorizonId,
    pub brain_name: String,
    pub direction: TradeDirection,
    pub entry_price: SubPennyPrice,
    pub structural_stop_price: SubPennyPrice,
    pub target_1_price: SubPennyPrice,
    pub target_2_price: SubPennyPrice,
    pub current_or_exit_price: SubPennyPrice,
    pub kelly_fraction_pct: f64,
    pub risk_budget_usd: f64,
    pub notional_allocated_usd: f64,
    pub matched_option_contract: String,
    pub contract_delta: f64,
    pub contract_gamma: f64,
    pub contract_vanna: f64,
    pub contract_iv: f64,
    pub stream_weights: FourStreamWeights,
    pub precedent_win_rate_pct: f64,
    pub ai_confidence_pct: f64,
    pub pnl_usd: f64,
    pub r_multiple: f64,
    pub tranche_count: u8,
    pub event_timestamp_ns: i64,
}

/// Immutable audit record stored in the bounded ring-buffer Webhook Outbox (`VecDeque<WebhookDispatchRecord>`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookDispatchRecord {
    pub dispatch_id: Uuid,
    pub dispatched_at_iso: String,
    pub dispatched_timestamp_ms: i64,
    pub event_type: WebhookEventType,
    pub outcome_detail: String,
    pub symbol: String,
    pub brain_id: BrainHorizonId,
    pub brain_name: String,
    pub direction: TradeDirection,
    pub entry_price: SubPennyPrice,
    pub structural_stop_price: SubPennyPrice,
    pub target_1_price: SubPennyPrice,
    pub target_2_price: SubPennyPrice,
    pub current_or_exit_price: SubPennyPrice,
    pub kelly_fraction_pct: f64,
    pub risk_budget_usd: f64,
    pub notional_allocated_usd: f64,
    pub matched_option_contract: String,
    pub greeks_summary: String,
    pub stream_attribution_summary: String,
    pub precedent_win_rate_pct: f64,
    pub ai_confidence_pct: f64,
    pub pnl_usd: f64,
    pub r_multiple: f64,
    pub delivery_status: String,
    pub channels_targeted: Vec<String>,
    pub discord_embed_json: String,
    pub telegram_markdown: String,
    pub slack_blocks_json: String,
    pub generic_webhook_json: String,
}

/// Response payload for `GET /api/v1/webhooks/status`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookStatusResponse {
    pub config: WebhookChannelConfig,
    pub total_outbox_records: usize,
    pub outbox_records: Vec<WebhookDispatchRecord>,
}

/// Multi-Channel Outbound Webhook Dispatcher with persistent bounded outbox.
pub struct OutboundWebhookDispatcher {
    config: RwLock<WebhookChannelConfig>,
    outbox: RwLock<VecDeque<WebhookDispatchRecord>>,
    http_client: reqwest::Client,
}

impl Default for OutboundWebhookDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl OutboundWebhookDispatcher {
    pub fn new() -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap_or_default();
        Self {
            config: RwLock::new(WebhookChannelConfig::default()),
            outbox: RwLock::new(VecDeque::with_capacity(MAX_WEBHOOK_OUTBOX_RECORDS)),
            http_client,
        }
    }

    pub fn get_config(&self) -> WebhookChannelConfig {
        self.config.read().clone()
    }

    pub fn update_config(&self, new_cfg: WebhookChannelConfig) -> WebhookChannelConfig {
        *self.config.write() = new_cfg.clone();
        new_cfg
    }

    pub fn export_outbox(&self) -> Vec<WebhookDispatchRecord> {
        self.outbox.read().iter().cloned().collect()
    }

    pub fn restore_state(
        &self,
        cfg: WebhookChannelConfig,
        records: Vec<WebhookDispatchRecord>,
    ) {
        *self.config.write() = cfg;
        let mut guard = self.outbox.write();
        guard.clear();
        for rec in records.into_iter().take(MAX_WEBHOOK_OUTBOX_RECORDS) {
            guard.push_back(rec);
        }
    }

    pub fn clear_outbox(&self) {
        self.outbox.write().clear();
    }

    /// Returns the current webhook configuration and the most recent 50 outbox dispatch records (newest first).
    pub fn status_snapshot(&self) -> WebhookStatusResponse {
        let cfg = self.config.read().clone();
        let guard = self.outbox.read();
        let total = guard.len();
        let recent: Vec<WebhookDispatchRecord> = guard.iter().rev().take(50).cloned().collect();
        WebhookStatusResponse {
            config: cfg,
            total_outbox_records: total,
            outbox_records: recent,
        }
    }

    /// Formats a lifecycle transition into Discord, Telegram, Slack, and Generic JSON payloads,
    /// appends the record to the bounded 100-item ring buffer, and asynchronously dispatches HTTP POSTs
    /// if any external webhook URLs are enabled.
    pub fn record_lifecycle_event(&self, payload: WebhookEventPayload) -> WebhookDispatchRecord {
        let cfg = self.config.read().clone();
        let channels_targeted = cfg.active_channel_names();
        let has_remote_target = channels_targeted.iter().any(|c| c != "LOCAL_OUTBOX");

        let ts_ms = if payload.event_timestamp_ns > 0 {
            payload.event_timestamp_ns / 1_000_000
        } else {
            Utc::now().timestamp_millis()
        };
        let dispatched_at_iso = Utc
            .timestamp_millis_opt(ts_ms)
            .single()
            .map(|dt| dt.format("%Y-%m-%dT%H:%M:%S.%3fZ").to_string())
            .unwrap_or_else(|| Utc::now().format("%Y-%m-%dT%H:%M:%S.%3fZ").to_string());

        let greeks_summary = format!(
            "Δ {:+.2} | Γ {:.3} | Vanna {:+.3} | IV {:.1}%",
            payload.contract_delta,
            payload.contract_gamma,
            payload.contract_vanna,
            payload.contract_iv * 100.0
        );
        let stream_attribution_summary = format!(
            "Tape {:.0}% | Opt Γ/V {:.0}% | Squeeze {:.0}% | Macro {:.0}%",
            payload.stream_weights.chart_tape_physics_weight * 100.0,
            payload.stream_weights.options_gamma_vanna_weight * 100.0,
            payload.stream_weights.short_float_squeeze_weight * 100.0,
            payload.stream_weights.market_internals_macro_weight * 100.0,
        );

        let embed_color_hex: u32 = match payload.event_type {
            WebhookEventType::SignalEntryOpened => 0x00B8D4, // Institutional Cyan
            WebhookEventType::ScaleInTrancheAdded => 0x2979FF, // Electric Blue
            WebhookEventType::Target1PartialBankedAndStopRatcheted => 0x00E676, // Emerald Profit Lock
            WebhookEventType::PositionClosed => {
                if payload.pnl_usd >= 0.0 {
                    0x00E676 // Emerald Winner
                } else {
                    0xFF5252 // Laser Crimson Stop
                }
            }
        };

        let pnl_sign = if payload.pnl_usd >= 0.0 { "+" } else { "" };
        let r_sign = if payload.r_multiple >= 0.0 { "+" } else { "" };

        let discord_val = json!({
            "username": "Argon Quant Execution Desk",
            "embeds": [{
                "title": format!(
                    "{} — {} ({}) [{}]",
                    payload.event_type.display_badge(),
                    payload.symbol,
                    payload.direction.as_str(),
                    payload.brain_id.code()
                ),
                "description": format!(
                    "**{}** | Brain: `{}` | Tranche: `{}/3` | Precedent Win Rate: `{:.1}%`",
                    payload.outcome_detail,
                    payload.brain_name,
                    payload.tranche_count,
                    payload.precedent_win_rate_pct
                ),
                "color": embed_color_hex,
                "timestamp": dispatched_at_iso,
                "fields": [
                    {
                        "name": "Entry / Current-Exit",
                        "value": format!("`{}` → `{}`", payload.entry_price, payload.current_or_exit_price),
                        "inline": true
                    },
                    {
                        "name": "Structural Stop",
                        "value": format!("`{}`", payload.structural_stop_price),
                        "inline": true
                    },
                    {
                        "name": "Targets (T1 / T2)",
                        "value": format!("`{}` / `{}`", payload.target_1_price, payload.target_2_price),
                        "inline": true
                    },
                    {
                        "name": "Fractional-Kelly & Risk Budget",
                        "value": format!(
                            "`{:.2}%` Equity (`${:.2}` Risk | `${:.0}` Notional)",
                            payload.kelly_fraction_pct,
                            payload.risk_budget_usd,
                            payload.notional_allocated_usd
                        ),
                        "inline": true
                    },
                    {
                        "name": "Realized / Unrealized P&L",
                        "value": format!("`{}{:.2} USD` (`{}{:.2}R`)", pnl_sign, payload.pnl_usd, r_sign, payload.r_multiple),
                        "inline": true
                    },
                    {
                        "name": "Matched Option & Greeks",
                        "value": format!("`{}` ({})", payload.matched_option_contract, greeks_summary),
                        "inline": false
                    },
                    {
                        "name": "4-Stream Attribution & k-NN",
                        "value": format!("{} | AI Conf `{:.1}%` | k-NN WR `{:.1}%`", stream_attribution_summary, payload.ai_confidence_pct, payload.precedent_win_rate_pct),
                        "inline": false
                    }
                ]
            }]
        });

        let telegram_markdown = format!(
            "*{badge}*\n\
            *Symbol:* `{sym}` ({dir}) — `{brain_code}` ({brain_name})\n\
            *Transition:* `{detail}` (Tranche `{tranche}/3`)\n\
            *Entry / Price:* `{entry}` → `{px}`\n\
            *Structural Stop:* `{stop}` | *T1 / T2:* `{t1}` / `{t2}`\n\
            *Kelly Sizing:* `{kelly:.2}%` (`${risk:.2}` 1R Risk | `${notional:.0}` Notional)\n\
            *Option & Greeks:* `{opt}` (`{greeks}`)\n\
            *4-Stream Weights:* `{streams}`\n\
            *k-NN Precedent WR:* `{knn:.1}%` | *P&L:* `{pnl_sign}${pnl:.2}` (`{r_sign}{r:.2}R`)",
            badge = payload.event_type.display_badge(),
            sym = payload.symbol,
            dir = payload.direction.as_str(),
            brain_code = payload.brain_id.code(),
            brain_name = payload.brain_name,
            detail = payload.outcome_detail,
            tranche = payload.tranche_count,
            entry = payload.entry_price,
            px = payload.current_or_exit_price,
            stop = payload.structural_stop_price,
            t1 = payload.target_1_price,
            t2 = payload.target_2_price,
            kelly = payload.kelly_fraction_pct,
            risk = payload.risk_budget_usd,
            notional = payload.notional_allocated_usd,
            opt = payload.matched_option_contract,
            greeks = greeks_summary,
            streams = stream_attribution_summary,
            knn = payload.precedent_win_rate_pct,
            pnl_sign = pnl_sign,
            pnl = payload.pnl_usd,
            r_sign = r_sign,
            r = payload.r_multiple,
        );

        let slack_val = json!({
            "text": format!(
                "{} — {} ({}) [{}] | {} | P&L: {}${:.2} ({}{:.2}R)",
                payload.event_type.display_badge(),
                payload.symbol,
                payload.direction.as_str(),
                payload.brain_id.code(),
                payload.outcome_detail,
                pnl_sign,
                payload.pnl_usd,
                r_sign,
                payload.r_multiple
            ),
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": format!(
                            "{} — {} ({}) [{}]",
                            payload.event_type.display_badge(),
                            payload.symbol,
                            payload.direction.as_str(),
                            payload.brain_id.code()
                        )
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        { "type": "mrkdwn", "text": format!("*Transition:*\n`{}`", payload.outcome_detail) },
                        { "type": "mrkdwn", "text": format!("*Entry → Price:*\n`{}` → `{}`", payload.entry_price, payload.current_or_exit_price) },
                        { "type": "mrkdwn", "text": format!("*Stop / Targets:*\nStop `{}` | T1 `{}` | T2 `{}`", payload.structural_stop_price, payload.target_1_price, payload.target_2_price) },
                        { "type": "mrkdwn", "text": format!("*Fractional-Kelly:*\n`{:.2}%` (`${:.2}` Risk | `${:.0}` Notional)", payload.kelly_fraction_pct, payload.risk_budget_usd, payload.notional_allocated_usd) },
                        { "type": "mrkdwn", "text": format!("*Option & Greeks:*\n`{}` ({})", payload.matched_option_contract, greeks_summary) },
                        { "type": "mrkdwn", "text": format!("*P&L & Precedent WR:*\n`{}${:.2}` (`{}{:.2}R`) | k-NN `{:.1}%`", pnl_sign, payload.pnl_usd, r_sign, payload.r_multiple, payload.precedent_win_rate_pct) }
                    ]
                },
                {
                    "type": "context",
                    "elements": [
                        { "type": "mrkdwn", "text": format!("4-Stream Attribution: {} | Dispatched: {}", stream_attribution_summary, dispatched_at_iso) }
                    ]
                }
            ]
        });

        let generic_val = json!({
            "event_type": payload.event_type.as_str(),
            "outcome_detail": payload.outcome_detail,
            "dispatched_at_iso": dispatched_at_iso,
            "dispatched_timestamp_ms": ts_ms,
            "symbol": payload.symbol,
            "brain_id": payload.brain_id.code(),
            "brain_name": payload.brain_name,
            "direction": payload.direction.as_str(),
            "tranche_count": payload.tranche_count,
            "entry_price_usd": payload.entry_price.to_f64(),
            "current_or_exit_price_usd": payload.current_or_exit_price.to_f64(),
            "structural_stop_price_usd": payload.structural_stop_price.to_f64(),
            "target_1_price_usd": payload.target_1_price.to_f64(),
            "target_2_price_usd": payload.target_2_price.to_f64(),
            "kelly_fraction_pct": payload.kelly_fraction_pct,
            "risk_budget_usd": payload.risk_budget_usd,
            "notional_allocated_usd": payload.notional_allocated_usd,
            "matched_option_contract": payload.matched_option_contract,
            "greeks": {
                "delta": payload.contract_delta,
                "gamma": payload.contract_gamma,
                "vanna": payload.contract_vanna,
                "iv": payload.contract_iv,
            },
            "stream_weights": payload.stream_weights,
            "precedent_win_rate_pct": payload.precedent_win_rate_pct,
            "ai_confidence_pct": payload.ai_confidence_pct,
            "pnl_usd": payload.pnl_usd,
            "r_multiple": payload.r_multiple,
        });

        let delivery_status = if has_remote_target {
            "DISPATCHING_ASYNC".to_string()
        } else {
            "QUEUED_LOCAL_OUTBOX".to_string()
        };

        let dispatch_id = Uuid::new_v4();
        let record = WebhookDispatchRecord {
            dispatch_id,
            dispatched_at_iso,
            dispatched_timestamp_ms: ts_ms,
            event_type: payload.event_type,
            outcome_detail: payload.outcome_detail,
            symbol: payload.symbol,
            brain_id: payload.brain_id,
            brain_name: payload.brain_name,
            direction: payload.direction,
            entry_price: payload.entry_price,
            structural_stop_price: payload.structural_stop_price,
            target_1_price: payload.target_1_price,
            target_2_price: payload.target_2_price,
            current_or_exit_price: payload.current_or_exit_price,
            kelly_fraction_pct: payload.kelly_fraction_pct,
            risk_budget_usd: payload.risk_budget_usd,
            notional_allocated_usd: payload.notional_allocated_usd,
            matched_option_contract: payload.matched_option_contract,
            greeks_summary,
            stream_attribution_summary,
            precedent_win_rate_pct: payload.precedent_win_rate_pct,
            ai_confidence_pct: payload.ai_confidence_pct,
            pnl_usd: payload.pnl_usd,
            r_multiple: payload.r_multiple,
            delivery_status,
            channels_targeted,
            discord_embed_json: serde_json::to_string_pretty(&discord_val).unwrap_or_default(),
            telegram_markdown,
            slack_blocks_json: serde_json::to_string_pretty(&slack_val).unwrap_or_default(),
            generic_webhook_json: serde_json::to_string_pretty(&generic_val).unwrap_or_default(),
        };

        {
            let mut guard = self.outbox.write();
            if guard.len() >= MAX_WEBHOOK_OUTBOX_RECORDS {
                guard.pop_front();
            }
            guard.push_back(record.clone());
        }

        if has_remote_target && tokio::runtime::Handle::try_current().is_ok() {
            let client = self.http_client.clone();
            let cfg_clone = cfg.clone();
            let rec_clone = record.clone();
            tokio::spawn(async move {
                let _ = Self::deliver_over_http(&client, &cfg_clone, &rec_clone).await;
            });
        }

        record
    }

    /// Fires a live test alert across configured webhooks (or logs to local outbox) and awaits HTTP delivery status.
    pub async fn fire_test_alert(&self, custom_symbol: Option<String>) -> WebhookDispatchRecord {
        let sym = custom_symbol
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| "NVDA".to_string())
            .to_uppercase();
        let payload = WebhookEventPayload {
            event_type: WebhookEventType::SignalEntryOpened,
            outcome_detail: "LIVE_WEBHOOK_DISPATCH_VERIFICATION_TEST".to_string(),
            symbol: sym.clone(),
            brain_id: BrainHorizonId::Brain02IntradayDayTrade,
            brain_name: BrainHorizonId::Brain02IntradayDayTrade
                .display_name()
                .to_string(),
            direction: TradeDirection::Long,
            entry_price: SubPennyPrice::from_f64(138.42000).unwrap_or(SubPennyPrice::ZERO),
            structural_stop_price: SubPennyPrice::from_f64(136.85000).unwrap_or(SubPennyPrice::ZERO),
            target_1_price: SubPennyPrice::from_f64(140.30000).unwrap_or(SubPennyPrice::ZERO),
            target_2_price: SubPennyPrice::from_f64(142.85000).unwrap_or(SubPennyPrice::ZERO),
            current_or_exit_price: SubPennyPrice::from_f64(138.42000).unwrap_or(SubPennyPrice::ZERO),
            kelly_fraction_pct: 1.85,
            risk_budget_usd: 1_850.00,
            notional_allocated_usd: 34_881.84,
            matched_option_contract: format!("{sym} 140C EXP +7D"),
            contract_delta: 0.56,
            contract_gamma: 0.084,
            contract_vanna: 0.032,
            contract_iv: 0.44,
            stream_weights: FourStreamWeights {
                chart_tape_physics_weight: 0.34,
                options_gamma_vanna_weight: 0.31,
                short_float_squeeze_weight: 0.21,
                market_internals_macro_weight: 0.14,
            },
            precedent_win_rate_pct: 78.6,
            ai_confidence_pct: 89.4,
            pnl_usd: 0.0,
            r_multiple: 0.0,
            tranche_count: 1,
            event_timestamp_ns: Utc::now().timestamp_nanos_opt().unwrap_or(0),
        };

        let mut record = self.record_lifecycle_event(payload);
        let cfg = self.config.read().clone();
        let has_remote_target = cfg
            .active_channel_names()
            .iter()
            .any(|c| c != "LOCAL_OUTBOX");
        if has_remote_target {
            let status = Self::deliver_over_http(&self.http_client, &cfg, &record).await;
            record.delivery_status = status.clone();
            let mut guard = self.outbox.write();
            if let Some(existing) = guard
                .iter_mut()
                .find(|r| r.dispatch_id == record.dispatch_id)
            {
                existing.delivery_status = status;
            }
        }
        record
    }

    async fn deliver_over_http(
        client: &reqwest::Client,
        cfg: &WebhookChannelConfig,
        record: &WebhookDispatchRecord,
    ) -> String {
        let mut statuses = Vec::new();

        if cfg.discord_enabled && !cfg.discord_webhook_url.trim().is_empty() {
            match client
                .post(cfg.discord_webhook_url.trim())
                .header("Content-Type", "application/json")
                .body(record.discord_embed_json.clone())
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => statuses.push("DISCORD:DELIVERED_200".to_string()),
                Ok(resp) => statuses.push(format!("DISCORD:HTTP_{}", resp.status().as_u16())),
                Err(e) => statuses.push(format!("DISCORD:ERR({})", e)),
            }
        }

        if cfg.telegram_enabled
            && !cfg.telegram_bot_token.trim().is_empty()
            && !cfg.telegram_chat_id.trim().is_empty()
        {
            let url = format!(
                "https://api.telegram.org/bot{}/sendMessage",
                cfg.telegram_bot_token.trim()
            );
            let body = json!({
                "chat_id": cfg.telegram_chat_id.trim(),
                "text": record.telegram_markdown,
                "parse_mode": "Markdown"
            });
            match client.post(&url).json(&body).send().await {
                Ok(resp) if resp.status().is_success() => statuses.push("TELEGRAM:DELIVERED_200".to_string()),
                Ok(resp) => statuses.push(format!("TELEGRAM:HTTP_{}", resp.status().as_u16())),
                Err(e) => statuses.push(format!("TELEGRAM:ERR({})", e)),
            }
        }

        if cfg.slack_enabled && !cfg.slack_webhook_url.trim().is_empty() {
            match client
                .post(cfg.slack_webhook_url.trim())
                .header("Content-Type", "application/json")
                .body(record.slack_blocks_json.clone())
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => statuses.push("SLACK:DELIVERED_200".to_string()),
                Ok(resp) => statuses.push(format!("SLACK:HTTP_{}", resp.status().as_u16())),
                Err(e) => statuses.push(format!("SLACK:ERR({})", e)),
            }
        }

        if cfg.custom_json_enabled && !cfg.custom_json_webhook_url.trim().is_empty() {
            match client
                .post(cfg.custom_json_webhook_url.trim())
                .header("Content-Type", "application/json")
                .body(record.generic_webhook_json.clone())
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => statuses.push("CUSTOM_JSON:DELIVERED_200".to_string()),
                Ok(resp) => statuses.push(format!("CUSTOM_JSON:HTTP_{}", resp.status().as_u16())),
                Err(e) => statuses.push(format!("CUSTOM_JSON:ERR({})", e)),
            }
        }

        if statuses.is_empty() {
            "QUEUED_LOCAL_OUTBOX".to_string()
        } else if statuses.iter().all(|s| s.ends_with("DELIVERED_200")) {
            "DELIVERED_200".to_string()
        } else {
            statuses.join(" | ")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_webhook_dispatcher_formats_all_channels_and_maintains_bounded_outbox() {
        let dispatcher = OutboundWebhookDispatcher::new();
        let rec = dispatcher.fire_test_alert(Some("PLTR".to_string())).await;

        assert_eq!(rec.symbol, "PLTR");
        assert_eq!(rec.event_type, WebhookEventType::SignalEntryOpened);
        assert_eq!(rec.delivery_status, "QUEUED_LOCAL_OUTBOX");
        assert!(rec.discord_embed_json.contains("Argon Quant Execution Desk"));
        assert!(rec.discord_embed_json.contains("Fractional-Kelly & Risk Budget"));
        assert!(rec.telegram_markdown.contains("*Symbol:* `PLTR`"));
        assert!(rec.slack_blocks_json.contains("\"blocks\""));
        assert!(rec.generic_webhook_json.contains("\"kelly_fraction_pct\": 1.85"));

        let snap = dispatcher.status_snapshot();
        assert_eq!(snap.total_outbox_records, 1);
        assert_eq!(snap.outbox_records[0].symbol, "PLTR");
    }
}
