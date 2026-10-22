import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    open: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        secure: false,
        timeout: 600000,
        configure: (proxy, _options) => {
          proxy.on('error', (err, _req, res) => {
            // Gracefully handle aborted or temporary client resets without crashing Vite proxy
            if (res && !res.headersSent && typeof res.writeHead === 'function') {
              try {
                res.writeHead(502, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ detail: 'Proxy temporarily unable to reach backend' }));
              } catch (_) {}
            }
          });
        }
      },
      '/storage': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        secure: false,
        timeout: 600000,
        configure: (proxy, _options) => {
          proxy.on('error', (err, _req, _res) => {
            // Silently ignore aborted/reset image fetches when components unmount
          });
        }
      }
    }
  }
})
