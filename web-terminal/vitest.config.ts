import { svelte, vitePreprocess } from '@sveltejs/vite-plugin-svelte';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  plugins: [
    svelte({
      preprocess: vitePreprocess(),
      compilerOptions: { dev: true }
    })
  ],
  resolve: {
    conditions: ['browser']
  },
  test: {
    include: ['src/**/*.{test,spec}.ts'],
    exclude: ['node_modules/**'],
    environment: 'node'
  }
});

