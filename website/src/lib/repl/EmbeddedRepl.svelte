<script lang="ts">
  import ReplEditor from "./ReplEditor.svelte";

  let {
    initialText,
  }: {
    initialText: string;
  } = $props();
</script>

<div
  class="h-full flex flex-col border border-gray-200 rounded-lg overflow-hidden"
>
  <div class="h-8 border-b border-gray-200"></div>
  <div class="flex-1">
    <ReplEditor
      {initialText}
      editorOptions={{
        lineNumbersMinChars: 4,
        padding: {
          top: 8,
          bottom: 8,
        },
        minimap: { enabled: false },
        scrollbar: { alwaysConsumeMouseWheel: false, useShadows: false },
        scrollBeyondLastLine: false,
      }}
    />
  </div>
</div>
