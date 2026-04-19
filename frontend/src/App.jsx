import { Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { useEffect, useState } from 'react'
import axios from 'axios'
import { useAppStore } from './store/useAppStore'
import Sidebar from './components/Sidebar'
import Screening from './pages/Screening'
import Roles from './pages/Roles'
import ComingSoon from './pages/ComingSoon'
import Login from './pages/Login'
import UserManagement from './pages/UserManagement'
import TalentPool from './pages/TalentPool'
import Dashboard from './pages/Dashboard'
import Calls from './pages/Calls'
import { VoIPProvider } from './context/VoIPContext'


// Set axios header immediately on module load (before any component renders)
const initToken = localStorage.getItem('token')
if (initToken) {
    axios.defaults.headers.common['Authorization'] = `Bearer ${initToken}`
}

function App() {
    const { isAuthenticated, token, fetchStats, fetchRoles, sidebarWidth } = useAppStore()
    const location = useLocation()
    const [isReady, setIsReady] = useState(false)
    const [isProfileLoaded, setIsProfileLoaded] = useState(false)

    // Set up axios interceptor to always use latest token
    useEffect(() => {
        if (token) {
            axios.defaults.headers.common['Authorization'] = `Bearer ${token}`
        } else {
            delete axios.defaults.headers.common['Authorization']
        }
        // Mark as ready after token is set
        setIsReady(true)
    }, [token])

    // Load initial data once authenticated and ready
    useEffect(() => {
        if (isAuthenticated && isReady) {
            const loadData = async () => {
                try { await useAppStore.getState().fetchProfile() } catch (e) { console.error(e) }
                setIsProfileLoaded(true)
                try { await fetchStats() } catch (e) { console.error(e) }
                try { await fetchRoles() } catch (e) { console.error(e) }
            }
            loadData()
        }
    }, [isAuthenticated, isReady])

    // Show nothing while initializing auth
    if (!isReady && isAuthenticated) {
        return null
    }

    if (!isAuthenticated) {
        return (
            <Routes>
                <Route path="/login" element={<Login />} />
                <Route path="*" element={<Navigate to="/login" replace />} />
            </Routes>
        )
    }

    if (!isProfileLoaded) {
        return null // don't render until profile and role are fetched
    }

    const isFullBleed = ['/', '/dashboard', '/talent-pool', '/screening', '/calls'].includes(location.pathname)

    return (
        <div className="app-layout">
            <Sidebar />
            <main
                className="main-content"
                style={{
                    marginLeft: sidebarWidth,
                    width: `calc(100% - ${sidebarWidth}px)`,
                    padding: isFullBleed ? 0 : undefined,
                    minHeight: '100vh',
                    boxSizing: 'border-box',
                }}
            >
                {!isFullBleed ? (
                    <div style={{ padding: '40px 60px' }}>
                        <div style={{ marginBottom: '36px' }}>
                            <h1 className="main-title">Talent Intelligence Platform</h1>
                            <p className="subtitle">Find and engage top candidates with AI-powered matching</p>
                        </div>
                        <Routes>
                            <Route path="/login" element={<Navigate to="/" replace />} />
                            <Route path="/" element={<Dashboard />} />
                            <Route path="/dashboard" element={<Navigate to="/" replace />} />
                            <Route path="/screening" element={<Screening />} />
                            <Route path="/roles" element={<Roles />} />
                            <Route path="/talent-pool" element={<TalentPool />} />
                            <Route path="/campaigns" element={<ComingSoon title="Campaign Management" description="Track and manage recruitment campaigns in one place" />} />
                            <Route path="/messages" element={<ComingSoon title="Messaging Center" description="Communicate with candidates directly from the platform" />} />
                            <Route path="/calls" element={<Calls />} />
                            <Route path="/admin/users" element={<UserManagement />} />
                        </Routes>
                    </div>
                ) : (
                    <Routes>
                        <Route path="/login" element={<Navigate to="/" replace />} />
                        <Route path="/" element={<Dashboard />} />
                        <Route path="/dashboard" element={<Navigate to="/" replace />} />
                        <Route path="/screening" element={<Screening />} />
                        <Route path="/roles" element={<Roles />} />
                        <Route path="/talent-pool" element={<TalentPool />} />
                        <Route path="/campaigns" element={<ComingSoon title="Campaign Management" description="Track and manage recruitment campaigns in one place" />} />
                        <Route path="/messages" element={<ComingSoon title="Messaging Center" description="Communicate with candidates directly from the platform" />} />
                        <Route path="/calls" element={<Calls />} />
                        <Route path="/admin/users" element={<UserManagement />} />
                    </Routes>
                )}
            </main>
            {/* Floating VoIP call popup — appears over any page when a bridge call arrives */}

        </div>
    )
}

function AppWithVoIP() {
    return (
        <VoIPProvider>
            <App />
        </VoIPProvider>
    )
}

export default AppWithVoIP


