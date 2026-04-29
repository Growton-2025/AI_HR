import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

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
                target: 'http://127.0.0.1:8000',
                changeOrigin: true,
                ws: true,
                timeout: 8000,
                proxyTimeout: 8000,
            }
        }
    }
})
