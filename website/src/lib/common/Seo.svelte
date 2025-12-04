<script lang="ts">
  const {
    title = "jax-js",
    description = "Machine learning library and compiler in JavaScript, generating fast WebAssembly and WebGPU kernels",
  }: { title?: string; description?: string } = $props();
</script>

<svelte:head>
  <meta property="og:type" content="website" />
  <meta property="og:site_name" content="jax-js" />
  <meta property="og:title" content={title} />
  <meta property="og:description" content={description} />
  <!-- <meta property="og:image" content={imageUrl} /> -->
</svelte:head>
