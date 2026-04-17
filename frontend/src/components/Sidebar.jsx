import { NavLink } from 'react-router-dom'
import { useEffect, useRef, useCallback } from 'react'
import { useAppStore } from '../store/useAppStore'
import { Search, Briefcase, Activity, MessageSquareMore, Phone, BarChart2, MoreHorizontal, LogOut, Users, PanelLeftClose, PanelLeftOpen } from 'lucide-react'

const MIN_WIDTH = 56
const MAX_WIDTH = 420
const ICON_THRESHOLD = 160

function Sidebar() {
    const analytics = useAppStore(state => state.analytics)
    const fetchAnalytics = useAppStore(state => state.fetchAnalytics)
    const { user, sidebarWidth, setSidebarWidth, isSidebarCollapsed, toggleSidebar } = useAppStore()
    const role = user?.role
    const permissions = user?.permissions || {}
    const isDragging = useRef(false)
    const startX = useRef(0)
    const startWidth = useRef(sidebarWidth)

    useEffect(() => { fetchAnalytics() }, [fetchAnalytics])

    const isIconOnly = sidebarWidth < ICON_THRESHOLD

    const allNavItems = [
        { path: "/",            label: "Dashboard",    icon: BarChart2,         id: "dashboard"   },
        { path: '/screening',   label: 'Screening',    icon: Search,            id: 'screening'   },
        { path: '/talent-pool', label: 'Talent Pool',  icon: Users,             id: 'talent_pool' },
        { path: '/roles',       label: 'Manage Roles', icon: Briefcase,         id: 'roles'       },
        { path: '/campaigns',   label: 'Campaigns',    icon: Activity,          id: 'campaigns'   },
        { path: '/messages',    label: 'Messages',     icon: MessageSquareMore, id: 'messages'    },
        { path: '/calls',       label: 'Calls',        icon: Phone,             id: 'calls'       }
    ]

    const navItems = role === 'admin'
        ? allNavItems
        : allNavItems.filter(item => item.id === 'dashboard' || permissions[item.id])

    // ── Drag Resize ────────────────────────────────────────────
    const onMouseDown = useCallback((e) => {
        isDragging.current = true
        startX.current = e.clientX
        startWidth.current = sidebarWidth
        document.body.style.cursor = 'col-resize'
        document.body.style.userSelect = 'none'
        e.preventDefault()
    }, [sidebarWidth])

    useEffect(() => {
        const onMove = (e) => {
            if (!isDragging.current) return
            const newW = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, startWidth.current + e.clientX - startX.current))
            setSidebarWidth(newW)
        }
        const onUp = () => {
            if (!isDragging.current) return
            isDragging.current = false
            document.body.style.cursor = ''
            document.body.style.userSelect = ''
        }
        window.addEventListener('mousemove', onMove)
        window.addEventListener('mouseup', onUp)
        return () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp) }
    }, [setSidebarWidth])

    const navPad = isIconOnly ? '4px 6px' : '4px 10px'
    const itemPad = isIconOnly ? '11px 0' : '9px 12px'
    const iconMr  = isIconOnly ? 0 : '9px'

    return (
        <aside className="app-sidebar" style={{ width: sidebarWidth, display: 'flex', flexDirection: 'column' }}>

            {/* ── Logo & Toggle ── */}
            <div style={{
                padding: isIconOnly ? '16px 0' : '20px 20px',
                borderBottom: '1px solid rgba(255,255,255,0.06)',
                display: 'flex', alignItems: 'center', justifyContent: isIconOnly ? 'center' : 'space-between', flexShrink: 0,
            }}>
                <img
                    src="https://cdn.prod.website-files.com/65e41b0d7632a225ef3abc4e/693509bb8da9f3b5e10554be_LOGO.svg"
                    alt="Growton"
                    style={{ filter: 'brightness(0) invert(1)', width: isIconOnly ? '26px' : '90px', height: 'auto', objectFit: 'contain', flexShrink: 0 }}
                />
                {!isIconOnly && (
                    <button onClick={toggleSidebar} style={{ background: 'none', border: 'none', color: '#52525b', cursor: 'pointer', padding: '4px' }}>
                        <PanelLeftClose size={18} />
                    </button>
                )}
            </div>

            {isIconOnly && (
                <button onClick={toggleSidebar} style={{ 
                    position: 'absolute', top: '70px', right: '-12px', background: '#f97316', 
                    borderRadius: '50%', width: '24px', height: '24px', display: 'flex', 
                    alignItems: 'center', justifyContent: 'center', border: 'none', color: '#fff',
                    cursor: 'pointer', zIndex: 10001, boxShadow: '0 2px 8px rgba(249,115,22,0.4)' 
                }}>
                    <PanelLeftOpen size={14} />
                </button>
            )}

            {/* ── Top Section (nav) ── */}
            <div style={{ flex: '0 0 auto' }}>
                {!isIconOnly && (
                    <div style={{ fontSize: '9px', fontWeight: 800, color: '#3f3f46', textTransform: 'uppercase', letterSpacing: '1.5px', padding: '14px 16px 4px' }}>
                        Navigation
                    </div>
                )}
                <nav style={{ padding: navPad, display: 'flex', flexDirection: 'column', gap: '2px' }}>
                    {navItems.map(item => {
                        const Icon = item.icon
                        return (
                            <NavLink key={item.path} to={item.path}
                                className={({ isActive }) => `nav-button ${isActive ? 'active' : ''}`}
                                style={{ justifyContent: isIconOnly ? 'center' : 'flex-start', padding: itemPad, overflow: 'hidden', whiteSpace: 'nowrap' }}
                                title={isIconOnly ? item.label : ''}
                            >
                                <Icon size={17} style={{ flexShrink: 0, marginRight: iconMr }} />
                                {!isIconOnly && <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{item.label}</span>}
                            </NavLink>
                        )
                    })}
                </nav>

                {/* Admin */}
                {role === 'admin' && (
                    <>
                        <div style={{ margin: '6px 10px', borderBottom: '1px solid rgba(255,255,255,0.06)' }} />
                        {!isIconOnly && (
                            <div style={{ fontSize: '9px', fontWeight: 800, color: '#3f3f46', textTransform: 'uppercase', letterSpacing: '1.5px', padding: '0 16px 4px' }}>Admin</div>
                        )}
                        <div style={{ padding: navPad }}>
                            <NavLink to="/admin/users"
                                className={({ isActive }) => `nav-button ${isActive ? 'active' : ''}`}
                                style={{ justifyContent: isIconOnly ? 'center' : 'flex-start', padding: itemPad, overflow: 'hidden', whiteSpace: 'nowrap' }}
                                title={isIconOnly ? 'User Management' : ''}
                            >
                                <MoreHorizontal size={17} style={{ flexShrink: 0, marginRight: iconMr }} />
                                {!isIconOnly && <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>User Management</span>}
                            </NavLink>
                        </div>
                    </>
                )}
            </div>

            {/* ── Pipeline Stats ── always visible, adapts layout ── */}
            <div style={{ flex: '0 0 auto', padding: isIconOnly ? '10px 6px' : '10px', marginTop: isIconOnly ? 'auto' : '12px' }}>
                {!isIconOnly && (
                    <>
                        <div style={{ borderBottom: '1px solid rgba(255,255,255,0.06)', marginBottom: '10px' }} />
                        <div style={{ fontSize: '9px', fontWeight: 800, color: '#3f3f46', textTransform: 'uppercase', letterSpacing: '1.5px', paddingLeft: '6px', marginBottom: '8px' }}>
                            Pipeline
                        </div>
                    </>
                )}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    {/* Sourced */}
                    <div style={{
                        background: 'rgba(249,115,22,0.08)', border: '1px solid rgba(249,115,22,0.15)',
                        borderRadius: '8px', padding: isIconOnly ? '8px 0' : '8px 12px',
                        display: 'flex', flexDirection: isIconOnly ? 'column' : 'row',
                        alignItems: 'center', gap: isIconOnly ? '2px' : '8px',
                    }}>
                        <div style={{ fontSize: isIconOnly ? '14px' : '18px', fontWeight: 800, color: '#f97316', lineHeight: 1 }}>
                            {analytics?.summary?.total_sourced?.toLocaleString() || 0}
                        </div>
                        {isIconOnly
                            ? <div style={{ fontSize: '7px', color: '#52525b', fontWeight: 700, textAlign: 'center', textTransform: 'uppercase', letterSpacing: '0.5px' }}>SRC</div>
                            : <div style={{ fontSize: '9px', textTransform: 'uppercase', color: '#52525b', fontWeight: 700, letterSpacing: '0.05em' }}>Total Sourced</div>
                        }
                    </div>
                    {/* Shortlisted */}
                    <div style={{
                        background: 'rgba(34,197,94,0.08)', border: '1px solid rgba(34,197,94,0.15)',
                        borderRadius: '8px', padding: isIconOnly ? '8px 0' : '8px 12px',
                        display: 'flex', flexDirection: isIconOnly ? 'column' : 'row',
                        alignItems: 'center', gap: isIconOnly ? '2px' : '8px',
                    }}>
                        <div style={{ fontSize: isIconOnly ? '14px' : '18px', fontWeight: 800, color: '#22c55e', lineHeight: 1 }}>
                            {analytics?.summary?.shortlisted || 0}
                        </div>
                        {isIconOnly
                            ? <div style={{ fontSize: '7px', color: '#52525b', fontWeight: 700, textAlign: 'center', textTransform: 'uppercase', letterSpacing: '0.5px' }}>SHL</div>
                            : <div style={{ fontSize: '9px', textTransform: 'uppercase', color: '#52525b', fontWeight: 700, letterSpacing: '0.05em' }}>Shortlisted</div>
                        }
                    </div>
                </div>
            </div>

            {/* ── Spacer only when expanded (not icon-only) ── */}
            {!isIconOnly && <div style={{ flex: 1 }} />}

            {/* ── User Profile ── */}
            <div style={{
                padding: isIconOnly ? '10px 6px' : '12px 10px',
                borderTop: '1px solid rgba(255,255,255,0.07)',
                display: 'flex', alignItems: 'center',
                gap: isIconOnly ? 0 : '8px',
                flexDirection: isIconOnly ? 'column' : 'row',
                flexShrink: 0, marginTop: isIconOnly ? '0' : '0',
            }}>
                <div style={{
                    width: 30, height: 30, borderRadius: '50%', flexShrink: 0,
                    background: 'linear-gradient(135deg, #f97316 0%, #c2410c 100%)',
                    color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontWeight: 700, fontSize: '10px', boxShadow: '0 2px 6px rgba(249,115,22,0.35)',
                }}>
                    {user?.full_name ? user.full_name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0,2) : '??'}
                </div>
                {!isIconOnly && (
                    <div style={{ flex: 1, overflow: 'hidden', minWidth: 0 }}>
                        <div style={{ fontSize: '12px', fontWeight: 600, color: '#f4f4f5', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {user?.full_name || 'User'}
                        </div>
                        <div style={{ fontSize: '10px', color: '#52525b' }}>
                            {role === 'admin' ? 'Admin' : 'Recruiter'}
                        </div>
                    </div>
                )}
                <button onClick={() => useAppStore.getState().logout()}
                    style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '3px', color: '#52525b', flexShrink: 0 }}
                    title="Log Out"
                >
                    <LogOut size={14} />
                </button>
            </div>

            {/* ── Drag Handle ── */}
            <div onMouseDown={onMouseDown}
                style={{ position: 'absolute', top: 0, right: 0, bottom: 0, width: '6px', cursor: 'col-resize', zIndex: 100 }}
                onMouseEnter={e => e.currentTarget.style.background = 'rgba(249,115,22,0.3)'}
                onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                title="Drag to resize"
            />
        </aside>
    )
}

export default Sidebar
