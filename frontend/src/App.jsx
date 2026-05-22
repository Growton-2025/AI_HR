import { Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { lazy, Suspense, useCallback, useEffect, useLayoutEffect, useRef, useState, Component } from 'react'
import axios from 'axios'
import { useShallow } from 'zustand/react/shallow'
import { useAppStore } from './store/useAppStore'
import Sidebar from './components/Sidebar'
import ComingSoon from './pages/ComingSoon'

const Screening = lazy(() => import('./pages/Screening'))
const Roles = lazy(() => import('./pages/Roles'))
const Login = lazy(() => import('./pages/Login'))
const UserManagement = lazy(() => import('./pages/UserManagement'))
const TalentPool = lazy(() => import('./pages/TalentPool'))
const Dashboard = lazy(() => import('./pages/Dashboard'))
const Calls = lazy(() => import('./pages/Calls'))

class TalentPoolErrorBoundary extends Component {
    constructor(props) {
        super(props)
        this.state = { error: null }
    }
    static getDerivedStateFromError(error) {
        return { error }
    }
    render() {
        if (this.state.error) {
            return (
                <div style={{ padding: '48px 32px', maxWidth: 720, margin: '0 auto', fontFamily: 'system-ui, sans-serif' }}>
                    <h1 style={{ fontSize: 22, color: '#0f172a', marginBottom: 12 }}>Talent Pool could not load</h1>
                    <p style={{ color: '#64748b', lineHeight: 1.6, marginBottom: 16 }}>
                        Something went wrong while rendering this page. Open the browser developer console (F12 → Console) for the full stack trace.
                    </p>
                    <pre
                        style={{
                            background: '#f8fafc',
                            border: '1px solid #e2e8f0',
                            borderRadius: 12,
                            padding: 16,
                            overflow: 'auto',
                            fontSize: 13,
                            color: '#b91c1c',
                            whiteSpace: 'pre-wrap',
                        }}
                    >
                        {String(this.state.error?.message || this.state.error)}
                    </pre>
                    <button
                        type="button"
                        onClick={() => window.location.reload()}
                        style={{
                            marginTop: 20,
                            padding: '10px 18px',
                            borderRadius: 10,
                            border: 'none',
                            background: '#f97316',
                            color: '#fff',
                            fontWeight: 700,
                            cursor: 'pointer',
                        }}
                    >
                        Reload page
                    </button>
                </div>
            )
        }
        return this.props.children
    }
}

// Set axios header immediately on module load (before any component renders)
const initToken = localStorage.getItem('token')
if (initToken) {
    axios.defaults.headers.common['Authorization'] = `Bearer ${initToken}`
}

function App() {
    const { isAuthenticated, token, fetchProfile, sidebarWidth } = useAppStore(useShallow((state) => ({
        isAuthenticated: state.isAuthenticated,
        token: state.token,
        fetchProfile: state.fetchProfile,
        sidebarWidth: state.sidebarWidth,
    })))
    const location = useLocation()
    const [isReady, setIsReady] = useState(false)
    const [isProfileLoaded, setIsProfileLoaded] = useState(false)
    const topAnchorRef = useRef(null)

    const resetViewportPosition = useCallback(() => {
        if (typeof window === 'undefined') return

        window.scrollTo({ top: 0, left: 0, behavior: 'auto' })

        const scrollRoots = [
            document.scrollingElement,
            document.documentElement,
            document.body,
            document.querySelector('.main-content'),
        ]

        scrollRoots.forEach((node) => {
            if (!node) return
            node.scrollTop = 0
            node.scrollLeft = 0
        })

        topAnchorRef.current?.scrollIntoView({
            block: 'start',
            inline: 'nearest',
            behavior: 'auto',
        })
    }, [])

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
            let isMounted = true
            setIsProfileLoaded(false)

            const loadData = async () => {
                try {
                    await fetchProfile()
                } catch (e) {
                    console.error(e)
                } finally {
                    if (isMounted) {
                        setIsProfileLoaded(true)
                    }
                }
            }

            loadData()

            return () => {
                isMounted = false
            }
        }

        setIsProfileLoaded(false)
    }, [fetchProfile, isAuthenticated, isReady])

    useLayoutEffect(() => {
        if (typeof window === 'undefined') return undefined

        const previousRestoration = 'scrollRestoration' in window.history
            ? window.history.scrollRestoration
            : undefined

        if ('scrollRestoration' in window.history) {
            window.history.scrollRestoration = 'manual'
        }

        resetViewportPosition()

        let lateRaf = 0
        const initialRaf = window.requestAnimationFrame(() => {
            resetViewportPosition()
            lateRaf = window.requestAnimationFrame(() => {
                resetViewportPosition()
            })
        })
        const timeoutId = window.setTimeout(() => {
            resetViewportPosition()
        }, 180)

        return () => {
            window.cancelAnimationFrame(initialRaf)
            window.cancelAnimationFrame(lateRaf)
            window.clearTimeout(timeoutId)

            if (previousRestoration && 'scrollRestoration' in window.history) {
                window.history.scrollRestoration = previousRestoration
            }
        }
    }, [location.pathname, resetViewportPosition])

    useEffect(() => {
        if (!isProfileLoaded) return undefined

        let attempt = 0
        let timeoutId = 0

        const syncViewport = () => {
            resetViewportPosition()

            attempt += 1
            if (attempt < 6) {
                timeoutId = window.setTimeout(syncViewport, attempt < 2 ? 80 : 180)
            }
        }

        timeoutId = window.setTimeout(syncViewport, 0)

        return () => {
            window.clearTimeout(timeoutId)
        }
    }, [isProfileLoaded, location.pathname, resetViewportPosition])

    const AppLoadingShell = ({ title, description }) => (
        <div
            style={{
                minHeight: '100vh',
                width: '100%',
                background: '#f8fafc',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '32px',
                boxSizing: 'border-box',
            }}
        >
            <div
                style={{
                    width: 'min(520px, 100%)',
                    background: '#fff',
                    borderRadius: '28px',
                    border: '1px solid #e2e8f0',
                    boxShadow: '0 20px 25px -5px rgba(15, 23, 42, 0.06)',
                    padding: '32px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '18px',
                }}
            >
                <div
                    style={{
                        width: '44px',
                        height: '44px',
                        borderRadius: '14px',
                        border: '3px solid rgba(249, 115, 22, 0.18)',
                        borderTopColor: '#f97316',
                        animation: 'spin 0.9s linear infinite',
                    }}
                />
                <div>
                    <div style={{ fontSize: '24px', fontWeight: 800, color: '#0f172a', marginBottom: '8px' }}>{title}</div>
                    <div style={{ fontSize: '14px', color: '#64748b', lineHeight: 1.6 }}>{description}</div>
                </div>
                <div style={{ display: 'grid', gap: '12px' }}>
                    <div style={{ height: '16px', width: '72%', borderRadius: '999px', background: '#e2e8f0' }} />
                    <div style={{ height: '16px', width: '100%', borderRadius: '999px', background: '#f1f5f9' }} />
                    <div style={{ height: '16px', width: '84%', borderRadius: '999px', background: '#f1f5f9' }} />
                </div>
            </div>
            <style>{`
                @keyframes spin {
                    from { transform: rotate(0deg); }
                    to { transform: rotate(360deg); }
                }
            `}</style>
        </div>
    )

    const RouteLoadingShell = ({ fullBleed }) => {
        if (fullBleed) {
            return (
                <div style={{ minHeight: '100vh', background: '#f8fafc', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', color: '#64748b', fontWeight: 700 }}>
                        <span
                            style={{
                                width: '18px',
                                height: '18px',
                                borderRadius: '999px',
                                border: '2px solid rgba(249, 115, 22, 0.22)',
                                borderTopColor: '#f97316',
                                animation: 'spin 0.9s linear infinite',
                            }}
                        />
                        Loading page...
                    </div>
                </div>
            )
        }

        return (
            <div style={{ padding: '40px 60px' }}>
                <div style={{ marginBottom: '36px' }}>
                    <h1 className="main-title">Talent Intelligence Platform</h1>
                    <p className="subtitle">Find and engage top candidates with AI-powered matching</p>
                </div>
                <div style={{ display: 'grid', gap: '18px' }}>
                    <div style={{ height: '140px', borderRadius: '24px', background: '#fff', border: '1px solid #e2e8f0' }} />
                    <div style={{ height: '320px', borderRadius: '24px', background: '#fff', border: '1px solid #e2e8f0' }} />
                </div>
            </div>
        )
    }

    // Show nothing while initializing auth
    if (!isReady && isAuthenticated) {
        return (
            <AppLoadingShell
                title="Preparing your workspace"
                description="Restoring your session and reconnecting the dashboard."
            />
        )
    }

    if (!isAuthenticated) {
        return (
            <Suspense fallback={<AppLoadingShell title="Opening sign in" description="Loading the secure sign-in experience." />}>
                <Routes>
                    <Route path="/login" element={<Login />} />
                    <Route path="*" element={<Navigate to="/login" replace />} />
                </Routes>
            </Suspense>
        )
    }

    if (!isProfileLoaded) {
        return (
            <AppLoadingShell
                title="Loading your profile"
                description="Pulling your permissions and getting the app ready."
            />
        )
    }

    const isFullBleed = ['/', '/dashboard', '/talent-pool', '/screening', '/calls'].includes(location.pathname)
    const routes = (
        <Suspense fallback={<RouteLoadingShell fullBleed={isFullBleed} />}>
            <Routes>
                <Route path="/login" element={<Navigate to="/" replace />} />
                <Route path="/" element={<Dashboard />} />
                <Route path="/dashboard" element={<Navigate to="/" replace />} />
                <Route path="/screening" element={<Screening />} />
                <Route path="/roles" element={<Roles />} />
                <Route
                    path="/talent-pool"
                    element={
                        <TalentPoolErrorBoundary>
                            <TalentPool />
                        </TalentPoolErrorBoundary>
                    }
                />
                <Route path="/campaigns" element={<ComingSoon title="Campaign Management" description="Track and manage recruitment campaigns in one place" />} />
                <Route path="/messages" element={<ComingSoon title="Messaging Center" description="Communicate with candidates directly from the platform" />} />
                <Route path="/calls" element={<Calls />} />
                <Route path="/admin/users" element={<UserManagement />} />
            </Routes>
        </Suspense>
    )

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
                <div
                    ref={topAnchorRef}
                    aria-hidden="true"
                    style={{ width: 0, height: 0, overflow: 'hidden' }}
                />
                {!isFullBleed ? (
                    <div style={{ padding: '40px 60px' }}>
                        <div style={{ marginBottom: '36px' }}>
                            <h1 className="main-title">Talent Intelligence Platform</h1>
                            <p className="subtitle">Find and engage top candidates with AI-powered matching</p>
                        </div>
                        {routes}
                    </div>
                ) : (
                    routes
                )}
            </main>
        </div>
    )
}

export default App
