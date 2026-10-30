<script lang="ts">
  import { onMount } from 'svelte';
  import { terminalStore, type WebhookChannelConfigState } from '$lib/state/terminal.svelte';
  import BrainAutopsyLab from '$lib/components/BrainAutopsyLab.svelte';

  const internals = $derived(terminalStore.macroInternals);
  const providers = $derived(terminalStore.providers);
  const webhookStatus = $derived(terminalStore.webhookStatus);
  const REPLAY_SPEEDS: Array<'1x' | '5x' | '10x' | '50x' | 'MAX'> = ['1x', '5x', '10x', '50x', 'MAX'];

  let scanningYahoo = $state(false);
  let swappingProviderId = $state<string | null>(null);
  let savingWebhooks = $state(false);
  let selectedOutboxEventId = $state<string | null>(null);
  let activePayloadTab = $state<'DISCORD' | 'TELEGRAM' | 'SLACK'>('DISCORD');

  let draftConfig = $state<WebhookChannelConfigState>({
    discordWebhookUrl: '',
    discordEnabled: false,
    telegramBotToken: '',
    telegramChatId: '',
    telegramEnabled: false,
    slackWebhookUrl: '',
    slackEnabled: false,
    customJsonWebhookUrl: '',
    customJsonEnabled: false
  });

  onMount(() => {
    void terminalStore.fetchWebhookStatus().then(() => {
      draftConfig = { ...terminalStore.webhookStatus.config };
    });
  });

  async function handleYahooScan(): Promise<void> {
    scanningYahoo = true;
    await terminalStore.triggerYahooUniverseScan();
    scanningYahoo = false;
  }

  async function handleHotSwap(providerId: string): Promise<void> {
    swappingProviderId = providerId;
    await terminalStore.hotSwapProvider(providerId);
    swappingProviderId = null;
  }

  async function handleSaveWebhooks(): Promise<void> {
    savingWebhooks = true;
    await terminalStore.saveWebhookConfig(draftConfig);
    draftConfig = { ...terminalStore.webhookStatus.config };
    savingWebhooks = false;
  }

  async function handleFireTestWebhook(): Promise<void> {
    await terminalStore.fireTestWebhookAlert();
    if (terminalStore.webhookStatus.outboxRecords.length > 0 && !selectedOutboxEventId) {
      selectedOutboxEventId = terminalStore.webhookStatus.outboxRecords[0]!.dispatchId;
    }
  }
</script>

<div class="command-bento-grid">
  <section class="glass-panel" style="grid-column: span 6;">
    <header class="panel-header" style="flex-wrap: wrap; gap: 8px;">
      <div class="panel-title">
        <span>Plug-and-Play Market Data Bus &amp; Zero-Downtime Failover Matrix</span>
      </div>

      <div style="display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
        <div style="display: flex; align-items: center; gap: 3px;">
          <span style="font-size: 0.66rem; color: var(--text-secondary);">REPLAY:</span>
          {#each REPLAY_SPEEDS as spd (spd)}
            <button
              type="button"
              class="filter-pill"
              data-selected={terminalStore.replaySpeed === spd}
              onclick={() => (terminalStore.replaySpeed = spd)}
              style="padding: 2px 6px; font-size: 0.65rem;"
            >
              {spd}
            </button>
          {/each}
        </div>

        <button
          type="button"
          class="action-btn action-btn-primary"
          onclick={handleYahooScan}
          disabled={scanningYahoo}
        >
          {scanningYahoo
            ? 'Scanning Live Yahoo Finance...'
            : '[ Scan Real Yahoo Finance Universe Now ]'}
        </button>
      </div>
    </header>

    <div class="data-table-container">
      <table class="telemetry-table">
        <thead>
          <tr>
            <th>Provider Adapter</th>
            <th>Stream Category</th>
            <th>Failover Tier</th>
            <th>Health Status</th>
            <th data-numeric="true">Wire Latency</th>
            <th data-numeric="true">Throughput</th>
            <th>Hot-Swap Control</th>
          </tr>
        </thead>
        <tbody>
          {#each providers as p (p.providerId)}
            <tr>
              <td>
                <strong>{p.providerId}</strong>
                <div style="font-size: 0.64rem; color: var(--text-secondary);">{p.displayName}</div>
              </td>
              <td><span class="badge badge-ai-core">{p.category}</span></td>
              <td><span class="badge badge-gamma">{p.role}</span></td>
              <td>
                <span class="badge {p.healthy ? 'badge-long' : 'badge-short'}">
                  {p.healthy ? 'ONLINE (ACTIVE)' : 'FAILOVER STANDBY'}
                </span>
              </td>
              <td data-numeric="true">{p.latencyMs.toFixed(2)} ms</td>
              <td data-numeric="true">{p.messagesPerSec.toLocaleString('en-US')} /s</td>
              <td>
                {#if p.role === 'PRIMARY'}
                  <span class="badge badge-long">ACTIVE PRIMARY</span>
                {:else}
                  <button
                    type="button"
                    class="action-btn"
                    onclick={() => handleHotSwap(p.providerId)}
                    disabled={swappingProviderId === p.providerId}
                  >
                    {swappingProviderId === p.providerId ? 'Swapping...' : 'Hot-Swap Primary'}
                  </button>
                {/if}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </section>

  <section class="glass-panel" style="grid-column: span 6;">
    <header class="panel-header">
      <div class="panel-title">
        <span>Stream 4: Live Yahoo Finance Macro Tide &amp; Cross-Asset Internals</span>
      </div>
    </header>
    <div class="scorecard-kpi-ribbon">
      <div class="kpi-card">
        <span class="kpi-label">NYSE $TICK Impulse</span>
        <span class="kpi-metric-value {internals.nyseTick >= 0 ? 'pnl-positive' : 'pnl-negative'}">
          {internals.nyseTick >= 0 ? '+' : ''}{internals.nyseTick}
        </span>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">Advance/Decline ($ADD)</span>
        <span class="kpi-metric-value {internals.nyseAdd >= 0 ? 'pnl-positive' : 'pnl-negative'}">
          {internals.nyseAdd >= 0 ? '+' : ''}{internals.nyseAdd}
        </span>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">Up/Down Vol ($VOLD)</span>
        <span class="kpi-metric-value pnl-positive">{internals.voldRatio.toFixed(2)}x</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">Semiconductor Tide (SOXX/SMH)</span>
        <span class="kpi-metric-value {internals.smhPct >= 0 ? 'pnl-positive' : 'pnl-negative'}">
          {internals.soxxPct >= 0 ? '+' : ''}{internals.soxxPct.toFixed(2)}% /
          {internals.smhPct >= 0 ? '+' : ''}{internals.smhPct.toFixed(2)}%
        </span>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">VIX Spot / Front Month</span>
        <span class="kpi-metric-value">
          {internals.vixSpot.toFixed(2)} / {internals.vixFrontMonth.toFixed(2)}
        </span>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">VIX Term Slope (VX1/VX2)</span>
        <span class="kpi-metric-value">{internals.vixTermSlope.toFixed(3)}</span>
      </div>
    </div>
  </section>

  <!-- 3-Speed Continuous Self-Learning & Nightly Autopsy Lab -->
  <BrainAutopsyLab />

  <!-- Outbound Execution Webhooks (Discord / Telegram / Slack / Custom JSON) & Live Outbox Inspector -->
  <section
    class="glass-panel"
    style="grid-column: span 12;"
    aria-label="Outbound Execution Webhooks and Live Outbox Inspector"
  >
    <header class="panel-header" style="flex-wrap: wrap; gap: 8px;">
      <div class="panel-title">
        <span>Outbound Execution Webhooks (Discord / Telegram / Slack / Custom JSON) &amp; Live Outbox Inspector</span>
        <span class="badge badge-ai-core">{webhookStatus.totalOutboxRecords} EVENTS RECORDED</span>
      </div>

      <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
        <button
          type="button"
          class="action-btn"
          onclick={handleSaveWebhooks}
          disabled={savingWebhooks}
        >
          {savingWebhooks ? 'Saving...' : '[ 💾 Save Webhook Config ]'}
        </button>
        <button
          type="button"
          class="action-btn action-btn-primary"
          onclick={handleFireTestWebhook}
          disabled={terminalStore.webhookTesting}
        >
          {terminalStore.webhookTesting
            ? 'Dispatching Test Alert...'
            : '[ 🚀 Fire Test Execution Alert ]'}
        </button>
      </div>
    </header>

    <!-- Channel Configuration Matrix -->
    <div
      style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px; padding: 12px; background: var(--surface-1); border-block-end: 1px solid var(--border-subtle); font-size: 0.72rem;"
    >
      <div style="display: flex; flex-direction: column; gap: 6px; padding: 8px; background: var(--surface-0); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <strong>Discord Rich Embed Webhook</strong>
          <label style="display: inline-flex; align-items: center; gap: 4px; cursor: pointer;">
            <input type="checkbox" bind:checked={draftConfig.discordEnabled} />
            <span>Enabled</span>
          </label>
        </div>
        <input
          type="url"
          class="input-control"
          placeholder="https://discord.com/api/webhooks/..."
          bind:value={draftConfig.discordWebhookUrl}
          aria-label="Discord Webhook URL"
        />
      </div>

      <div style="display: flex; flex-direction: column; gap: 6px; padding: 8px; background: var(--surface-0); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <strong>Telegram Bot API Markdown</strong>
          <label style="display: inline-flex; align-items: center; gap: 4px; cursor: pointer;">
            <input type="checkbox" bind:checked={draftConfig.telegramEnabled} />
            <span>Enabled</span>
          </label>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px;">
          <input
            type="text"
            class="input-control"
            placeholder="Bot Token (123456:ABC...)"
            bind:value={draftConfig.telegramBotToken}
            aria-label="Telegram Bot Token"
          />
          <input
            type="text"
            class="input-control"
            placeholder="Chat ID (-100...)"
            bind:value={draftConfig.telegramChatId}
            aria-label="Telegram Chat ID"
          />
        </div>
      </div>

      <div style="display: flex; flex-direction: column; gap: 6px; padding: 8px; background: var(--surface-0); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <strong>Slack Block Kit Webhook</strong>
          <label style="display: inline-flex; align-items: center; gap: 4px; cursor: pointer;">
            <input type="checkbox" bind:checked={draftConfig.slackEnabled} />
            <span>Enabled</span>
          </label>
        </div>
        <input
          type="url"
          class="input-control"
          placeholder="https://hooks.slack.com/services/..."
          bind:value={draftConfig.slackWebhookUrl}
          aria-label="Slack Webhook URL"
        />
      </div>

      <div style="display: flex; flex-direction: column; gap: 6px; padding: 8px; background: var(--surface-0); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <strong>Generic JSON Execution Webhook</strong>
          <label style="display: inline-flex; align-items: center; gap: 4px; cursor: pointer;">
            <input type="checkbox" bind:checked={draftConfig.customJsonEnabled} />
            <span>Enabled</span>
          </label>
        </div>
        <input
          type="url"
          class="input-control"
          placeholder="https://your-prop-desk.internal/webhooks/argon"
          bind:value={draftConfig.customJsonWebhookUrl}
          aria-label="Custom JSON Webhook URL"
        />
      </div>
    </div>

    <!-- Live Webhook Delivery Outbox Table -->
    <div class="data-table-container">
      <table class="telemetry-table">
        <thead>
          <tr>
            <th>Event ID</th>
            <th>Lifecycle Event Type</th>
            <th>Symbol</th>
            <th>Brain</th>
            <th>Dir</th>
            <th data-numeric="true">Price / Stop / T1 / T2</th>
            <th data-numeric="true">Kelly % ($1R Risk / Notional)</th>
            <th data-numeric="true">P&amp;L / R-Mult</th>
            <th>HTTP / Outbox State</th>
            <th>Formatted Payload Preview</th>
          </tr>
        </thead>
        <tbody>
          {#each webhookStatus.outboxRecords.slice(0, 25) as rec (rec.dispatchId)}
            <tr
              data-selected={selectedOutboxEventId === rec.dispatchId}
              onclick={() =>
                (selectedOutboxEventId =
                  selectedOutboxEventId === rec.dispatchId ? null : rec.dispatchId)}
              style="cursor: pointer;"
            >
              <td class="tabular-num">{rec.dispatchId}</td>
              <td>
                <span
                  class="badge {rec.eventType === 'POSITION_CLOSED'
                    ? 'badge-long'
                    : rec.eventType === 'TARGET_1_PARTIAL_BANKED_AND_STOP_RATCHETED'
                      ? 'badge-squeeze'
                      : 'badge-ai-core'}"
                >
                  {rec.eventType}
                </span>
              </td>
              <td><strong>{rec.symbol}</strong></td>
              <td><span class="badge badge-ai-core">{rec.brainId}</span></td>
              <td>
                <span class="badge {rec.direction === 'LONG' ? 'badge-long' : 'badge-short'}">
                  {rec.direction}
                </span>
              </td>
              <td data-numeric="true" class="tabular-num">
                ${rec.entryPrice.toFixed(2)} / Stop ${rec.structuralStopPrice.toFixed(2)} / T1 ${rec.target1Price.toFixed(2)} / T2 ${rec.target2Price.toFixed(2)}
              </td>
              <td data-numeric="true" class="tabular-num">
                <strong>{rec.kellyFractionPct.toFixed(2)}%</strong>
                (${rec.riskBudgetUsd.toFixed(0)} / ${rec.notionalAllocatedUsd.toFixed(0)})
              </td>
              <td
                data-numeric="true"
                class="tabular-num {rec.pnlUsd >= 0 ? 'pnl-positive' : 'pnl-negative'}"
              >
                {rec.pnlUsd >= 0 ? '+' : ''}${rec.pnlUsd.toFixed(2)} ({rec.rMultiple >= 0 ? '+' : ''}{rec.rMultiple.toFixed(2)}R)
              </td>
              <td>
                <span class="badge badge-long">{rec.deliveryStatus}</span>
              </td>
              <td>
                <button
                  type="button"
                  class="action-btn"
                  onclick={(e) => {
                    e.stopPropagation();
                    selectedOutboxEventId =
                      selectedOutboxEventId === rec.dispatchId ? null : rec.dispatchId;
                  }}
                >
                  {selectedOutboxEventId === rec.dispatchId ? 'Hide Preview' : 'Inspect Embed / Markdown'}
                </button>
              </td>
            </tr>

            {#if selectedOutboxEventId === rec.dispatchId}
              <tr>
                <td colspan="10" style="background: oklch(0.14 0.02 260); padding: 10px 14px;">
                  <div style="display: flex; justify-content: space-between; align-items: center; margin-block-end: 8px; flex-wrap: wrap; gap: 8px;">
                    <div style="display: flex; align-items: center; gap: 6px;">
                      <strong style="font-size: 0.72rem;">Formatted Wire Payload ({rec.dispatchId}):</strong>
                      {#each ['DISCORD', 'TELEGRAM', 'SLACK'] as tab (tab)}
                        <button
                          type="button"
                          class="filter-pill"
                          data-selected={activePayloadTab === tab}
                          onclick={() => (activePayloadTab = tab as 'DISCORD' | 'TELEGRAM' | 'SLACK')}
                          style="padding: 2px 8px; font-size: 0.66rem;"
                        >
                          {tab === 'DISCORD'
                            ? 'Discord Rich Embed JSON'
                            : tab === 'TELEGRAM'
                              ? 'Telegram Bot Markdown'
                              : 'Slack Block Kit JSON'}
                        </button>
                      {/each}
                    </div>
                    <span style="font-size: 0.68rem; color: var(--text-secondary);">
                      Option: {rec.matchedOptionContract} | Outcome: {rec.outcomeDetail}
                    </span>
                  </div>
                  <pre
                    style="margin: 0; padding: 10px; background: oklch(0.11 0.015 255); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); font-family: monospace; font-size: 0.68rem; overflow-x: auto; white-space: pre-wrap; color: oklch(0.88 0.04 240);"
                  >{activePayloadTab === 'DISCORD'
                      ? rec.discordEmbedJson
                      : activePayloadTab === 'TELEGRAM'
                        ? rec.telegramMarkdown
                        : rec.slackBlocksJson}</pre>
                </td>
              </tr>
            {/if}
          {/each}
        </tbody>
      </table>
    </div>
  </section>
</div>
