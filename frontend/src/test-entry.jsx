import React from 'react'
import ReactDOM from 'react-dom/client'
import TestVoIP from './pages/TestVoIP'
import './index.css'
import { Toaster } from 'sonner'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <div style={{ background: '#f8fafc', minHeight: '100vh' }}>
      <TestVoIP />
      <Toaster position="top-right" expand={false} richColors />
    </div>
  </React.StrictMode>,
)
