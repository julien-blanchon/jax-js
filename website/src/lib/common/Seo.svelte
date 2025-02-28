<script lang="ts">
  const {
    title = "jax-js",
    description = "JavaScript library for numerical and GPU computing on the web",
  }: { title?: string; description?: string } = $props();
</script>

<svelte:head>
  <meta property="og:type" content="website" />
  <meta property="og:site_name" content="jax-js" />
  <meta property="og:title" content={title} />
  <meta property="og:description" content={description} />
  <!-- <meta property="og:image" content={imageUrl} /> -->
</svelte:head>
