import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  root: './',
  build: {
    rollupOptions: {
      input: {
        main: './test-voip.html'
      }
    }
  },
  server: {
    port: 3005,
    host: true,
    open: '/test-voip.html'
  }
})
