import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [sveltekit()],
  css: {
    transformer: 'lightningcss',
    lightningcss: {
      drafts: {
        customMedia: true
      }
    }
  },
  optimizeDeps: {
    rolldownOptions: {
      target: 'es2025'
    }
  },
  server: {
    port: 5173,
    strictPort: false,
    forwardConsole: true
  },
  build: {
    target: 'es2025',
    cssMinify: 'lightningcss'
  }
});

