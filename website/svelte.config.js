import adapter from "@sveltejs/adapter-static";
import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";

/** @type {import("@sveltejs/kit").Config} */
const config = {
  preprocess: vitePreprocess(),

  kit: {
    // Fallback is used for dynamic routes that aren't prerendered.
    // https://svelte.dev/docs/kit/adapter-static
    adapter: adapter({ fallback: "200.html" }),
    alias: {
      "$app.css": "src/app.css",
    },
    paths: {
      base: process.argv.includes("dev") ? "" : process.env.BASE_PATH,
    },
  },
};

export default config;
