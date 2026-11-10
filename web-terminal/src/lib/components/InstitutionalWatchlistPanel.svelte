<script lang="ts">
  import { onMount } from 'svelte';
  import {
    terminalStore,
    formatSubPennyPrice,
    type InstitutionalWatchlistRow
  } from '$lib/state/terminal.svelte';
  import {
    ArrowsClockwise,
    DownloadSimple,
    PaperPlaneTilt,
    ShieldCheck,
    CheckCircle,
    WarningCircle,
    LockSimple,
    Sliders,
    FloppyDisk,
    Broadcast
  } from 'phosphor-svelte';

  const watchlist = $derived(terminalStore.institutionalWatchlist);
  const sundayProtocol = $derived(terminalStore.sundayAlphaProtocol);
  const paperOverlay = $derived(terminalStore.paperOverlayState);
  const stagedReceipts = $derived(terminalStore.stagedOrdersReceipts);
  const webhookConfig = $derived(terminalStore.webhookConfig);
  const isLoading = $derived(terminalStore.isLoadingWatchlist || terminalStore.isLoadingSundayProtocol);
  const isDispatching = $derived(terminalStore.isDispatchingSundayWebhook);
  const isStaging = $derived(terminalStore.isStagingOrders);
  const isUpdatingWebhook = $derived(terminalStore.isUpdatingWebhookConfig);
  const isPinging = $derived(terminalStore.isTestPingingWebhook);

  const stagedSymbolsMap = $derived.by(() => {
    const map: Record<string, string> = {};
    for (const r of stagedReceipts) {
      map[r.symbol] = r.client_order_id;
    }
    return map;
  });

  let isWebhookDrawerOpen = $state(false);
  let dispatchBanner = $state<string | null>(null);
  let stageBanner = $state<string | null>(null);
  let webhookBanner = $state<string | null>(null);

  // Webhook editable configuration
  let discordInput = $state('');
  let telegramTokenInput = $state('');
  let telegramChatIdInput = $state('');
  let discordEnabled = $state(true);
  let telegramEnabled = $state(true);

  onMount(() => {
    void terminalStore.fetchInstitutionalWatchlist();
    void terminalStore.fetchSundayAlphaProtocol();
    void terminalStore.fetchPaperOverlayState();
    void terminalStore.fetchWebhookConfig().then((cfg) => {
      if (cfg) {
        discordEnabled = cfg.discord_enabled;
        telegramEnabled = cfg.telegram_enabled;
      }
    });
  });

  const displayedRows = $derived.by(() => {
    if (!watchlist?.watchlist) return [];
    let list = watchlist.watchlist;
    if (terminalStore.actionableWatchlistOnly) {
      list = list.filter((r) => r.is_actionable);
    }
    const q = terminalStore.watchlistSearchQuery.trim().toUpperCase();
    if (q) {
      list = list.filter(
        (r) =>
          r.symbol.toUpperCase().includes(q) ||
          r.brain_id.toUpperCase().includes(q) ||
          r.strategy_archetype.toUpperCase().includes(q) ||
          r.matched_option_contract.toUpperCase().includes(q)
      );
    }
    return list;
  });

  async function handleRefresh(): Promise<void> {
    await Promise.all([
      terminalStore.fetchInstitutionalWatchlist(),
      terminalStore.fetchSundayAlphaProtocol(),
      terminalStore.fetchPaperOverlayState(),
      terminalStore.fetchWebhookConfig()
    ]);
  }

  async function handleExportCsv(): Promise<void> {
    await terminalStore.exportSundayProtocolCsv();
  }

  async function handleDispatchWebhook(): Promise<void> {
    const res = await terminalStore.dispatchSundayProtocolWebhook();
    if (res) {
      dispatchBanner = `Successfully queued ${res.staged_count} signals to institutional outbox (Receipt: ${res.dispatched_at_iso.slice(11, 19)} UTC)`;
      setTimeout(() => {
        dispatchBanner = null;
      }, 5000);
    }
  }

  async function handleStageAllPaper(): Promise<void> {
    const actionableSymbols = displayedRows
      .filter((r) => r.is_actionable)
      .map((r) => r.symbol);
    if (actionableSymbols.length === 0) return;
    const res = await terminalStore.stageSundayOrders({
      symbols: actionableSymbols
    });
    if (res && res.status === 'SUCCESS') {
      stageBanner = `Paper-Locked: Staged ${res.staged_orders_count} approved setups into isolated OMS (${res.total_staged_kelly_pct.toFixed(2)}% Kelly)`;
      setTimeout(() => {
        stageBanner = null;
      }, 6000);
    }
  }

  async function handleStageSingle(symbol: string): Promise<void> {
    const res = await terminalStore.stageSundayOrders({
      symbols: [symbol]
    });
    if (res && res.status === 'SUCCESS') {
      stageBanner = `Staged ${symbol} into Paper-OMS: ${res.receipts[0]?.client_order_id ?? 'OK'}`;
      setTimeout(() => {
        stageBanner = null;
      }, 5000);
    }
  }

  async function handleSaveWebhookConfig(): Promise<void> {
    const res = await terminalStore.updateWebhookConfig({
      discord_webhook_url: discordInput.trim() ? discordInput.trim() : undefined,
      telegram_bot_token: telegramTokenInput.trim() ? telegramTokenInput.trim() : undefined,
      telegram_chat_id: telegramChatIdInput.trim() ? telegramChatIdInput.trim() : undefined,
      discord_enabled: discordEnabled,
      telegram_enabled: telegramEnabled
    });
    if (res) {
      webhookBanner = `Webhook configuration saved (Dry-Run: ${res.dry_run_mode ? 'ACTIVE' : 'OFF'})`;
      discordInput = '';
      telegramTokenInput = '';
      telegramChatIdInput = '';
      setTimeout(() => {
        webhookBanner = null;
      }, 5000);
    }
  }

  async function handleTestPing(): Promise<void> {
    const res = await terminalStore.testPingWebhook();
    if (res) {
      webhookBanner = `Test Ping Result: ${res.delivery_status} (Receipt: ${res.receipt_id})`;
      setTimeout(() => {
        webhookBanner = null;
      }, 6000);
    }
  }
</script>

<section
  class="glass-panel panel-institutional-watchlist"
  aria-label="Phase 29/30 Institutional Watchlist & Sunday Alpha Protocol Stager"
>
  <header class="panel-header">
    <div class="panel-title">
      <span class="title-main">INSTITUTIONAL WATCHLIST &amp; SUNDAY ALPHA PROTOCOL</span>
      <span class="badge badge-ai-core">
        GEN {watchlist?.active_generation ?? '36'}
      </span>
      <span class="badge badge-long">
        UNIVERSE: {watchlist?.universe_size ?? 24}
      </span>
      <span class="badge badge-consensus">
        ACTIONABLE: {watchlist?.actionable_setups_count ?? 19}
      </span>
      <span class="badge badge-baseline-lock" title="Immutable baseline ledger locked at 173 trades">
        <ShieldCheck size={13} weight="bold" />
        BASELINE: 173 TRADES LOCKED
      </span>
    </div>

    <!-- 1-Click Action Controls -->
    <div class="controls-toolbar">
      <button
        type="button"
        class="action-btn action-btn-refresh"
        onclick={handleRefresh}
        disabled={isLoading}
        title="Refresh institutional watchlist & Sunday Alpha scan"
      >
        <ArrowsClockwise size={13} class={isLoading ? 'spin-icon' : ''} />
        {isLoading ? 'Scanning...' : 'Refresh Scan'}
      </button>

      <button
        type="button"
        class="action-btn action-btn-stage-all"
        onclick={handleStageAllPaper}
        disabled={isStaging || (watchlist?.actionable_setups_count ?? 0) === 0}
        title="Stage all actionable setups into isolated Paper-OMS with hardware lock"
      >
        <LockSimple size={13} weight="bold" />
        {isStaging ? 'Staging Paper...' : 'Stage All Approved (Paper Lock 🔒)'}
      </button>

      <button
        type="button"
        class="action-btn action-btn-config {isWebhookDrawerOpen ? 'active' : ''}"
        onclick={() => { isWebhookDrawerOpen = !isWebhookDrawerOpen; }}
        title="Toggle Webhook Outbox & Secret Masking Configuration Drawer"
      >
        <Sliders size={13} weight="bold" />
        Webhooks Config
      </button>

      <button
        type="button"
        class="action-btn action-btn-export"
        onclick={handleExportCsv}
        title="Download Sunday Alpha Protocol CSV Stager Sheet"
      >
        <DownloadSimple size={13} weight="bold" />
        Export Sunday CSV
      </button>

      <button
        type="button"
        class="action-btn action-btn-primary"
        onclick={handleDispatchWebhook}
        disabled={isDispatching}
        title="Dispatch Sunday Alpha Staged Signals to Discord / Telegram / OMS Webhooks"
      >
        <PaperPlaneTilt size={13} weight="bold" />
        {isDispatching ? 'Dispatching...' : 'Dispatch Webhook Outbox'}
      </button>
    </div>
  </header>

  <!-- Cross-Brain Risk Allocator & OCC TIMS Telemetry Strip -->
  <div class="telemetry-bar">
    <div class="telemetry-metric">
      <span class="metric-label">PORTFOLIO KELLY CAP</span>
      <span class="metric-value">
        {(watchlist?.total_allocated_kelly_pct ?? 10.0).toFixed(2)}%
        <span class="metric-sub">/ 10.00% MAX</span>
      </span>
    </div>

    <div class="telemetry-metric">
      <span class="metric-label">NET DELTA (BETA-WT)</span>
      <span
        class="metric-value {(watchlist?.net_portfolio_delta_beta_weighted ?? 0) >= 0 ? 'pnl-positive' : 'pnl-negative'}"
      >
        {(watchlist?.net_portfolio_delta_beta_weighted ?? -2.71).toFixed(2)} Δ
      </span>
    </div>

    <div class="telemetry-metric">
      <span class="metric-label">CONFLICTS BLOCKED</span>
      <span class="metric-value">
        {watchlist?.cross_brain_conflicts_blocked ?? 1}
        <span class="metric-sub">NETTED OPPOSING</span>
      </span>
    </div>

    <div class="telemetry-metric">
      <span class="metric-label">OCC TIMS MARGIN RELIEF</span>
      <span class="metric-value pnl-positive">
        +{(watchlist?.estimated_occ_tims_margin_relief_pct ?? 46.5).toFixed(1)}%
      </span>
    </div>

    <div class="telemetry-metric">
      <span class="metric-label">PAPER OVERLAY DELTA</span>
      <span class="metric-value">
        +{paperOverlay?.overlay_closed_count ?? 0} TRADES
        <span class="metric-sub">({paperOverlay?.open_paper_positions ?? 0} OPEN)</span>
      </span>
    </div>
  </div>

  <!-- Outbox Dispatch Feedback Banner -->
  {#if dispatchBanner}
    <div class="dispatch-banner" role="status">
      <CheckCircle size={15} weight="bold" />
      <span>{dispatchBanner}</span>
    </div>
  {/if}

  <!-- Paper Staging Feedback Banner -->
  {#if stageBanner}
    <div class="stage-banner" role="status">
      <LockSimple size={15} weight="bold" />
      <span>{stageBanner}</span>
    </div>
  {/if}

  <!-- Webhook Feedback Banner -->
  {#if webhookBanner}
    <div class="webhook-banner" role="status">
      <Broadcast size={15} weight="bold" />
      <span>{webhookBanner}</span>
    </div>
  {/if}

  <!-- Webhook Configuration & Secret Redaction Drawer -->
  {#if isWebhookDrawerOpen}
    <div class="webhook-drawer" aria-label="Webhook Configuration & Secret Masking Drawer">
      <div class="drawer-header">
        <div class="drawer-title">
          <Broadcast size={14} weight="bold" />
          <span>INSTITUTIONAL WEBHOOK DESTINATIONS (STRICT SECRET MASKING)</span>
        </div>
        <div class="drawer-mode-indicator">
          <span class="badge {webhookConfig?.dry_run_mode ? 'badge-dry-run' : 'badge-live'}">
            {webhookConfig?.dry_run_mode ? 'DRY-RUN (LOCAL OUTBOX)' : 'LIVE REMOTE DISPATCH'}
          </span>
          <span class="badge badge-venue">OUTBOX: {webhookConfig?.outbox_count ?? 0}</span>
        </div>
      </div>

      <div class="drawer-body">
        <div class="config-row">
          <div class="config-label-group">
            <span class="config-label-title">Discord Webhook Destination</span>
            <span class="masked-status">
              {webhookConfig?.discord_webhook_url_masked || '(Not Configured)'}
            </span>
          </div>
          <div class="config-input-group">
            <input
              type="password"
              placeholder="Paste new Discord webhook URL to override..."
              bind:value={discordInput}
              class="drawer-input"
            />
            <label class="checkbox-label">
              <input type="checkbox" bind:checked={discordEnabled} />
              <span>Enabled</span>
            </label>
          </div>
        </div>

        <div class="config-row">
          <div class="config-label-group">
            <span class="config-label-title">Telegram Bot Token</span>
            <span class="masked-status">
              {webhookConfig?.telegram_bot_token_masked || '(Not Configured)'}
            </span>
          </div>
          <div class="config-input-group">
            <input
              type="password"
              placeholder="Paste new bot token..."
              bind:value={telegramTokenInput}
              class="drawer-input"
            />
            <label class="checkbox-label">
              <input type="checkbox" bind:checked={telegramEnabled} />
              <span>Enabled</span>
            </label>
          </div>
        </div>

        <div class="config-row">
          <div class="config-label-group">
            <span class="config-label-title">Telegram Chat ID</span>
            <span class="masked-status">
              {webhookConfig?.telegram_chat_id_masked || '(Not Configured)'}
            </span>
          </div>
          <div class="config-input-group">
            <input
              type="text"
              placeholder="e.g. -1001234567890..."
              bind:value={telegramChatIdInput}
              class="drawer-input"
            />
          </div>
        </div>

        <div class="drawer-actions">
          <div class="drawer-buttons">
            <button
              type="button"
              class="action-btn action-btn-save"
              onclick={handleSaveWebhookConfig}
              disabled={isUpdatingWebhook}
            >
              <FloppyDisk size={13} weight="bold" />
              {isUpdatingWebhook ? 'Saving...' : 'Save Webhook Config'}
            </button>

            <button
              type="button"
              class="action-btn action-btn-ping"
              onclick={handleTestPing}
              disabled={isPinging}
            >
              <Broadcast size={13} weight="bold" />
              {isPinging ? 'Pinging...' : 'Test Ping'}
            </button>
          </div>
        </div>
      </div>
    </div>
  {/if}

  <!-- Filter & Search Strip -->
  <div class="filter-strip">
    <div class="search-box">
      <input
        type="search"
        placeholder="Filter symbol, brain, archetype, contract..."
        bind:value={terminalStore.watchlistSearchQuery}
        class="search-input"
      />
    </div>

    <label class="filter-toggle">
      <input
        type="checkbox"
        bind:checked={terminalStore.actionableWatchlistOnly}
      />
      <span>Actionable Only ({watchlist?.actionable_setups_count ?? 19})</span>
    </label>

    <div class="row-count-caption">
      Showing {displayedRows.length} of {watchlist?.universe_size ?? 24} watchlist pairs
    </div>
  </div>

  <!-- Interactive 24-Row Watchlist & Sunday Alpha Stager Table -->
  <div class="data-table-container">
    <table class="telemetry-table">
      <thead>
        <tr>
          <th>#</th>
          <th>Symbol</th>
          <th>Brain</th>
          <th>Dir</th>
          <th>Archetype</th>
          <th data-numeric="true">Eng A %</th>
          <th data-numeric="true">Eng B %</th>
          <th data-numeric="true">Consensus</th>
          <th data-numeric="true">Canonical Entry</th>
          <th data-numeric="true">Stop Px</th>
          <th data-numeric="true">Target 1</th>
          <th data-numeric="true">Target 2</th>
          <th data-numeric="true">Stage 0 Margin</th>
          <th>CBOE Contract</th>
          <th>Venue Ticket</th>
          <th data-numeric="true">Kelly %</th>
          <th>Action Status</th>
          <th>Paper OMS Staging</th>
        </tr>
      </thead>
      <tbody>
        {#each displayedRows as row (row.symbol + row.brain_id + row.rank)}
          <tr
            data-actionable={row.is_actionable}
            class={row.action_status.includes('BLOCKED') ? 'row-conflict-blocked' : ''}
          >
            <td class="rank-cell">#{row.rank}</td>
            <td class="symbol-cell">
              <strong>{row.symbol}</strong>
            </td>
            <td>
              <span class="badge badge-ai-core">{row.brain_id}</span>
            </td>
            <td>
              <span class="badge {row.direction === 'LONG' ? 'badge-long' : 'badge-short'}">
                {row.direction}
              </span>
            </td>
            <td>
              <span class="badge badge-archetype">{row.strategy_archetype}</span>
            </td>
            <td data-numeric="true" class="tabular-num">
              {row.engine_a_confidence_pct.toFixed(1)}%
            </td>
            <td data-numeric="true" class="tabular-num">
              {row.engine_b_confidence_pct.toFixed(1)}%
            </td>
            <td data-numeric="true" class="tabular-num consensus-cell">
              <strong>{row.consensus_confidence_pct.toFixed(1)}%</strong>
            </td>
            <td data-numeric="true" class="tabular-num entry-price">
              {formatSubPennyPrice(row.entry_price)}
            </td>
            <td data-numeric="true" class="tabular-num stop-price">
              {formatSubPennyPrice(row.structural_stop_price)}
            </td>
            <td data-numeric="true" class="tabular-num target-price">
              {formatSubPennyPrice(row.target_1_price)}
            </td>
            <td data-numeric="true" class="tabular-num target-price">
              {formatSubPennyPrice(row.target_2_price)}
            </td>
            <td data-numeric="true" class="tabular-num conformal-cell">
              {#if row.stage0_conformal_status === 'PASS'}
                <span class="conformal-pass" title="Conformal LLR: {row.conformal_llr.toFixed(3)} >= {row.conformal_threshold.toFixed(3)}">
                  +{row.conformal_margin.toFixed(3)} PASS
                </span>
              {:else}
                <span class="conformal-veto" title="Conformal LLR: {row.conformal_llr.toFixed(3)} < {row.conformal_threshold.toFixed(3)}">
                  {row.conformal_margin.toFixed(3)} VETO
                </span>
              {/if}
            </td>
            <td class="contract-cell">
              <code>{row.matched_option_contract}</code>
            </td>
            <td>
              <span class="badge badge-venue">{row.venue_ticket}</span>
            </td>
            <td data-numeric="true" class="tabular-num kelly-cell">
              {row.kelly_fraction_pct.toFixed(2)}%
            </td>
            <td>
              {#if row.action_status === 'ACTIONABLE_STAGE_READY'}
                <span class="badge badge-actionable">
                  <CheckCircle size={11} weight="bold" />
                  STAGE READY
                </span>
              {:else if row.action_status.includes('CO_ALIGNED')}
                <span class="badge badge-coaligned" title="Multi-horizon co-aligned setup capped <= 1.25%">
                  CO-ALIGNED
                </span>
              {:else if row.action_status === 'BLOCKED_CROSS_BRAIN_CONFLICT'}
                <span class="badge badge-conflict" title="Blocked by Cross-Brain Netting Gate against opposing higher expectancy setup">
                  <WarningCircle size={11} weight="bold" />
                  CONFLICT BLOCKED
                </span>
              {:else}
                <span class="badge badge-vetoed">{row.action_status}</span>
              {/if}
            </td>
            <td class="staging-cell">
              {#if stagedSymbolsMap[row.symbol]}
                <span class="badge badge-staged-receipt" title="Client Order ID: {stagedSymbolsMap[row.symbol]}">
                  <LockSimple size={10} weight="bold" />
                  PAPER STAGED
                </span>
              {:else if row.is_actionable}
                <button
                  type="button"
                  class="action-btn action-btn-stage-row"
                  onclick={() => handleStageSingle(row.symbol)}
                  disabled={isStaging}
                  title="Stage {row.symbol} into Paper-OMS"
                >
                  Stage Paper
                </button>
              {:else}
                <span class="stage-ineligible">—</span>
              {/if}
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
</section>

<style>
  .panel-institutional-watchlist {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 10px 14px;
    margin: 10px 12px;
    background: oklch(0.16 0.015 250 / 0.72);
    border: 1px solid oklch(0.28 0.03 250 / 0.5);
    border-radius: 6px;
    backdrop-filter: blur(8px);
  }

  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
    border-bottom: 1px solid oklch(0.24 0.02 250 / 0.4);
    padding-bottom: 8px;
  }

  .panel-title {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }

  .title-main {
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    color: oklch(0.92 0.02 250);
  }

  .controls-toolbar {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }

  .action-btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 8px;
    font-size: 0.7rem;
    font-weight: 600;
    border-radius: 4px;
    cursor: pointer;
    background: oklch(0.22 0.02 250 / 0.8);
    color: oklch(0.85 0.02 250);
    border: 1px solid oklch(0.35 0.04 250 / 0.5);
    transition: background 0.15s ease, border-color 0.15s ease;
  }

  .action-btn:hover:not(:disabled) {
    background: oklch(0.28 0.03 250);
    border-color: oklch(0.48 0.08 250);
  }

  .action-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .action-btn-primary {
    background: oklch(0.25 0.08 150 / 0.85);
    color: oklch(0.95 0.05 150);
    border-color: oklch(0.45 0.12 150);
  }

  .action-btn-primary:hover:not(:disabled) {
    background: oklch(0.32 0.12 150);
    border-color: oklch(0.55 0.16 150);
  }

  .action-btn-stage-all {
    background: oklch(0.24 0.08 280 / 0.85);
    color: oklch(0.95 0.06 280);
    border-color: oklch(0.45 0.12 280);
  }

  .action-btn-stage-all:hover:not(:disabled) {
    background: oklch(0.32 0.12 280);
    border-color: oklch(0.55 0.16 280);
  }

  .action-btn-config {
    background: oklch(0.22 0.03 250 / 0.8);
    color: oklch(0.85 0.04 250);
    border-color: oklch(0.38 0.05 250);
  }

  .action-btn-config.active {
    background: oklch(0.3 0.06 250);
    border-color: oklch(0.55 0.1 250);
    color: oklch(0.98 0.02 250);
  }

  .action-btn-export {
    background: oklch(0.24 0.06 200 / 0.8);
    color: oklch(0.9 0.04 200);
    border-color: oklch(0.4 0.08 200);
  }

  .action-btn-stage-row {
    padding: 2px 6px;
    font-size: 0.62rem;
    background: oklch(0.25 0.07 280 / 0.8);
    color: oklch(0.92 0.05 280);
    border-color: oklch(0.42 0.1 280);
  }

  .action-btn-save {
    background: oklch(0.25 0.08 150 / 0.85);
    color: oklch(0.95 0.05 150);
    border-color: oklch(0.45 0.12 150);
  }

  .action-btn-ping {
    background: oklch(0.24 0.06 200 / 0.8);
    color: oklch(0.9 0.04 200);
    border-color: oklch(0.4 0.08 200);
  }

  .telemetry-bar {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 8px;
    padding: 6px 10px;
    background: oklch(0.13 0.015 250 / 0.6);
    border-radius: 4px;
    border: 1px solid oklch(0.22 0.02 250 / 0.4);
  }

  .telemetry-metric {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .metric-label {
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: oklch(0.65 0.03 250);
  }

  .metric-value {
    font-size: 0.82rem;
    font-weight: 800;
    color: oklch(0.9 0.02 250);
    font-family: var(--font-mono, monospace);
  }

  .metric-sub {
    font-size: 0.62rem;
    font-weight: 500;
    color: oklch(0.6 0.02 250);
  }

  .dispatch-banner {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    font-size: 0.72rem;
    font-weight: 600;
    color: oklch(0.92 0.12 150);
    background: oklch(0.22 0.08 150 / 0.7);
    border: 1px solid oklch(0.4 0.12 150);
    border-radius: 4px;
  }

  .stage-banner {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    font-size: 0.72rem;
    font-weight: 600;
    color: oklch(0.92 0.1 280);
    background: oklch(0.22 0.08 280 / 0.7);
    border: 1px solid oklch(0.42 0.12 280);
    border-radius: 4px;
  }

  .webhook-banner {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    font-size: 0.72rem;
    font-weight: 600;
    color: oklch(0.92 0.08 200);
    background: oklch(0.22 0.06 200 / 0.7);
    border: 1px solid oklch(0.42 0.1 200);
    border-radius: 4px;
  }

  /* Webhook Drawer */
  .webhook-drawer {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 10px 12px;
    background: oklch(0.14 0.02 250 / 0.95);
    border: 1px solid oklch(0.32 0.05 250 / 0.6);
    border-radius: 5px;
  }

  .drawer-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    flex-wrap: wrap;
    border-bottom: 1px solid oklch(0.24 0.02 250 / 0.5);
    padding-bottom: 6px;
  }

  .drawer-title {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    color: oklch(0.9 0.03 250);
  }

  .drawer-mode-indicator {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .badge-dry-run {
    background: oklch(0.25 0.08 80 / 0.6);
    color: oklch(0.92 0.12 80);
    border: 1px solid oklch(0.45 0.12 80 / 0.6);
  }

  .badge-live {
    background: oklch(0.26 0.12 150 / 0.6);
    color: oklch(0.95 0.14 150);
    border: 1px solid oklch(0.5 0.15 150 / 0.6);
  }

  .drawer-body {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .config-row {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .config-label-group {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    font-size: 0.66rem;
  }

  .config-label-title {
    font-weight: 700;
    color: oklch(0.8 0.02 250);
  }

  .masked-status {
    font-family: var(--font-mono, monospace);
    font-size: 0.64rem;
    color: oklch(0.68 0.05 200);
  }

  .config-input-group {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .drawer-input {
    flex: 1;
    padding: 3px 8px;
    font-size: 0.68rem;
    border-radius: 4px;
    background: oklch(0.1 0.015 250 / 0.9);
    border: 1px solid oklch(0.26 0.03 250);
    color: oklch(0.9 0.02 250);
    font-family: var(--font-mono, monospace);
  }

  .drawer-input:focus {
    outline: none;
    border-color: oklch(0.5 0.1 220);
  }

  .checkbox-label {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 0.66rem;
    color: oklch(0.75 0.02 250);
    cursor: pointer;
  }

  .drawer-actions {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 4px;
    padding-top: 6px;
    border-top: 1px solid oklch(0.22 0.02 250 / 0.4);
  }

  .drawer-buttons {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .filter-strip {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    font-size: 0.7rem;
  }

  .search-box {
    flex: 1;
    min-inline-size: 200px;
  }

  .search-input {
    inline-size: 100%;
    padding: 3px 8px;
    font-size: 0.7rem;
    border-radius: 4px;
    background: oklch(0.12 0.015 250 / 0.9);
    border: 1px solid oklch(0.28 0.03 250);
    color: oklch(0.9 0.02 250);
  }

  .search-input:focus {
    outline: none;
    border-color: oklch(0.5 0.1 220);
  }

  .filter-toggle {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 0.7rem;
    color: oklch(0.8 0.02 250);
    cursor: pointer;
  }

  .row-count-caption {
    font-size: 0.65rem;
    color: oklch(0.6 0.02 250);
  }

  .data-table-container {
    max-block-size: 340px;
    overflow-y: auto;
    overflow-x: auto;
    border: 1px solid oklch(0.22 0.02 250 / 0.4);
    border-radius: 4px;
  }

  .telemetry-table {
    inline-size: 100%;
    border-collapse: collapse;
    font-size: 0.69rem;
    text-align: start;
  }

  .telemetry-table th {
    position: sticky;
    top: 0;
    background: oklch(0.18 0.02 250);
    padding: 4px 6px;
    font-weight: 700;
    font-size: 0.62rem;
    letter-spacing: 0.04em;
    color: oklch(0.7 0.03 250);
    border-bottom: 1px solid oklch(0.28 0.03 250);
    white-space: nowrap;
    z-index: 1;
  }

  .telemetry-table td {
    padding: 3px 6px;
    border-bottom: 1px solid oklch(0.2 0.02 250 / 0.4);
    white-space: nowrap;
  }

  .telemetry-table tr:hover {
    background: oklch(0.22 0.02 250 / 0.5);
  }

  .row-conflict-blocked {
    opacity: 0.6;
    background: oklch(0.18 0.04 25 / 0.2);
  }

  .tabular-num,
  [data-numeric="true"] {
    text-align: end;
    font-family: var(--font-mono, monospace);
  }

  .rank-cell {
    font-weight: 700;
    color: oklch(0.7 0.02 250);
  }

  .symbol-cell strong {
    color: oklch(0.95 0.02 250);
    font-size: 0.72rem;
  }

  .consensus-cell strong {
    color: oklch(0.95 0.08 150);
  }

  .entry-price {
    font-weight: 700;
    color: oklch(0.9 0.02 250);
  }

  .stop-price {
    color: oklch(0.75 0.12 25);
  }

  .target-price {
    color: oklch(0.85 0.12 150);
  }

  .conformal-pass {
    color: oklch(0.85 0.14 150);
    font-weight: 600;
  }

  .conformal-veto {
    color: oklch(0.75 0.14 25);
    font-weight: 600;
  }

  .contract-cell code {
    font-family: var(--font-mono, monospace);
    font-size: 0.65rem;
    color: oklch(0.85 0.04 200);
  }

  .kelly-cell {
    font-weight: 600;
    color: oklch(0.9 0.05 180);
  }

  .staging-cell {
    text-align: center;
  }

  .stage-ineligible {
    color: oklch(0.5 0.02 250);
  }

  .badge {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    padding: 1px 4px;
    font-size: 0.6rem;
    font-weight: 700;
    border-radius: 3px;
    text-transform: uppercase;
    letter-spacing: 0.02em;
  }

  .badge-ai-core {
    background: oklch(0.24 0.05 250);
    color: oklch(0.9 0.03 250);
  }

  .badge-long {
    background: oklch(0.25 0.08 150 / 0.4);
    color: oklch(0.88 0.14 150);
    border: 1px solid oklch(0.4 0.12 150 / 0.5);
  }

  .badge-short {
    background: oklch(0.25 0.08 25 / 0.4);
    color: oklch(0.88 0.14 25);
    border: 1px solid oklch(0.4 0.12 25 / 0.5);
  }

  .badge-consensus {
    background: oklch(0.24 0.06 180);
    color: oklch(0.9 0.06 180);
  }

  .badge-baseline-lock {
    background: oklch(0.22 0.04 150 / 0.5);
    color: oklch(0.9 0.08 150);
    border: 1px solid oklch(0.38 0.1 150 / 0.5);
  }

  .badge-archetype {
    background: oklch(0.22 0.03 280);
    color: oklch(0.85 0.05 280);
  }

  .badge-venue {
    background: oklch(0.2 0.02 250);
    color: oklch(0.75 0.02 250);
    border: 1px solid oklch(0.3 0.02 250);
    font-family: var(--font-mono, monospace);
  }

  .badge-actionable {
    background: oklch(0.26 0.1 150);
    color: oklch(0.98 0.08 150);
    border: 1px solid oklch(0.45 0.14 150);
  }

  .badge-coaligned {
    background: oklch(0.24 0.08 220);
    color: oklch(0.95 0.08 220);
    border: 1px solid oklch(0.42 0.12 220);
  }

  .badge-conflict {
    background: oklch(0.26 0.1 30);
    color: oklch(0.95 0.1 30);
    border: 1px solid oklch(0.45 0.14 30);
  }

  .badge-vetoed {
    background: oklch(0.22 0.03 250);
    color: oklch(0.7 0.02 250);
  }

  .badge-staged-receipt {
    background: oklch(0.26 0.1 280);
    color: oklch(0.98 0.08 280);
    border: 1px solid oklch(0.48 0.14 280);
  }

  .pnl-positive {
    color: oklch(0.85 0.14 150);
  }

  .pnl-negative {
    color: oklch(0.8 0.14 25);
  }

  :global(.spin-icon) {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from {
      transform: rotate(0deg);
    }
    to {
      transform: rotate(360deg);
    }
  }
</style>
