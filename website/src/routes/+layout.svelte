<script lang="ts">
  import Seo from "$lib/common/Seo.svelte";
  import "../app.css";

  let { children } = $props();
</script>

<Seo />

{@render children()}
