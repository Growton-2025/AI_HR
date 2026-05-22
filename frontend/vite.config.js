import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const apiProxyTarget = process.env.VITE_API_PROXY_TARGET || 'http://127.0.0.1:8000'

export default defineConfig({
    plugins: [react()],
    server: {
        port: 3000,
        strictPort: true,
        hmr: {
            host: '127.0.0.1',
            port: 3000
        },
        proxy: {
            '/api': {
                target: apiProxyTarget,
                changeOrigin: true,
                ws: true,
                // Large CSV/XLSX imports can run longer than typical API calls; short timeouts
                // surface as failed commit/upload requests in dev.
                timeout: 300000,
                proxyTimeout: 300000,
            }
        }
    }
})
