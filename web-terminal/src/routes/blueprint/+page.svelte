<script lang="ts">
  import BlueprintCanvas from '$lib/components/BlueprintCanvas.svelte';
  import RawEvidencePanel from '$lib/components/RawEvidencePanel.svelte';
</script>

<div class="command-bento-grid">
  <div style="grid-column: span 8;">
    <BlueprintCanvas />
  </div>
  <div style="grid-column: span 4;">
    <RawEvidencePanel />
  </div>
</div>

