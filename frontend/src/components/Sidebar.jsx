import { NavLink } from 'react-router-dom'
import { useEffect, useRef, useCallback } from 'react'
import { useAppStore } from '../store/useAppStore'
import { Search, Briefcase, Activity, MessageSquareMore, Phone, BarChart2, MoreHorizontal, LogOut, Users, PanelLeftClose, PanelLeftOpen } from 'lucide-react'
import { useShallow } from 'zustand/react/shallow'
import HayasaBrand from './HayasaBrand'

const MIN_WIDTH = 72
const MAX_WIDTH = 420
const ICON_THRESHOLD = 160

const normalizeCount = (value) => {
    if (value == null || value === '') return null
    const count = Number(value)
    return Number.isFinite(count) ? count : null
}

const getStatusCount = (counts, statusName) => {
    if (!counts || typeof counts !== 'object') return null
    if (counts[statusName] != null) return normalizeCount(counts[statusName])

    const normalizedStatus = statusName.trim().toLowerCase()
    const matchedKey = Object.keys(counts).find(key => key.trim().toLowerCase() === normalizedStatus)
    return matchedKey ? normalizeCount(counts[matchedKey]) : null
}

function Sidebar() {
    const analytics = useAppStore(state => state.analytics)
    const { user, sidebarWidth, setSidebarWidth, toggleSidebar, tpScopeTotal, tpScopeStatusCounts, tpScopeSummaryLastParamsString, buildTalentPoolScopeQuery, talentPoolViewScope, talentPoolRecruiterFilterId, talentPoolRoleFilterId } = useAppStore(useShallow((state) => ({
        user: state.user,
        sidebarWidth: state.sidebarWidth,
        setSidebarWidth: state.setSidebarWidth,
        toggleSidebar: state.toggleSidebar,
        tpScopeTotal: state.tpScopeTotal,
        tpScopeStatusCounts: state.tpScopeStatusCounts,
        tpScopeSummaryLastParamsString: state.tpScopeSummaryLastParamsString,
        buildTalentPoolScopeQuery: state.buildTalentPoolScopeQuery,
        talentPoolViewScope: state.talentPoolViewScope,
        talentPoolRecruiterFilterId: state.talentPoolRecruiterFilterId,
        talentPoolRoleFilterId: state.talentPoolRoleFilterId,
    })))
    const role = user?.role
    const permissions = user?.permissions || {}
    const isDragging = useRef(false)
    const startX = useRef(0)
    const startWidth = useRef(sidebarWidth)

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

    const navPad = isIconOnly ? '8px 8px' : '4px 10px'
    const itemPad = isIconOnly ? '12px 0' : '9px 12px'
    const iconMr  = isIconOnly ? 0 : '9px'
    const summaryTotal = normalizeCount(tpScopeTotal)
    const summaryShortlisted = getStatusCount(tpScopeStatusCounts, 'Shortlisted')
    const analyticsTotal = normalizeCount(analytics?.summary?.total_sourced)
    const analyticsShortlisted = normalizeCount(analytics?.summary?.shortlisted)
    const activeSummaryParams = buildTalentPoolScopeQuery()
    const summaryMatchesScope = tpScopeSummaryLastParamsString === activeSummaryParams
    const summaryStatusEmpty = !tpScopeStatusCounts || Object.keys(tpScopeStatusCounts).length === 0
    const summaryLooksCold = summaryMatchesScope && summaryTotal === 0 && summaryStatusEmpty && Number(analyticsTotal || 0) > 0
    const scopedSummaryReady = summaryMatchesScope && !summaryLooksCold && summaryTotal != null
    const sourcedCount = scopedSummaryReady ? summaryTotal : (analyticsTotal ?? summaryTotal ?? 0)
    const shortlistedCount = scopedSummaryReady
        ? (summaryShortlisted ?? 0)
        : (analyticsShortlisted ?? summaryShortlisted ?? 0)

    return (
        <aside
            className="app-sidebar"
            data-collapsed={isIconOnly ? 'true' : 'false'}
            style={{ width: sidebarWidth, display: 'flex', flexDirection: 'column' }}
        >

            {/* ── Logo & Toggle ── */}
            <div style={{
                padding: isIconOnly ? '16px 0 14px' : '18px 18px 14px',
                borderBottom: '1px solid rgba(255,255,255,0.07)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: isIconOnly ? 'center' : 'space-between',
                flexDirection: isIconOnly ? 'column' : 'row',
                gap: isIconOnly ? '12px' : 0,
                flexShrink: 0,
            }}>
                <div
                    style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: isIconOnly ? 'center' : 'flex-start',
                        minWidth: isIconOnly ? '34px' : '142px',
                        minHeight: isIconOnly ? '34px' : '36px',
                        padding: 0,
                        flexShrink: 0,
                    }}
                >
                    <HayasaBrand size="sidebar" tone="dark" iconOnly={isIconOnly} layout="sidebarStack" />
                </div>
                <button
                    onClick={toggleSidebar}
                    style={{
                        width: isIconOnly ? '34px' : '32px',
                        height: isIconOnly ? '34px' : '32px',
                        background: 'rgba(255,255,255,0.03)',
                        border: '1px solid rgba(255,255,255,0.08)',
                        color: '#9ca3af',
                        cursor: 'pointer',
                        padding: 0,
                        borderRadius: '11px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        boxShadow: isIconOnly ? '0 10px 20px rgba(15,23,42,0.2)' : 'none',
                    }}
                    title={isIconOnly ? 'Expand sidebar' : 'Collapse sidebar'}
                >
                    {isIconOnly ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={18} />}
                </button>
            </div>

            {/* ── Top Section (nav) ── */}
            <div style={{ flex: '0 0 auto' }}>
                {!isIconOnly && (
                    <div style={{ fontSize: '9px', fontWeight: 800, color: '#7c8697', textTransform: 'uppercase', letterSpacing: '1.5px', padding: '16px 16px 6px' }}>
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
                        <div style={{ margin: '10px 10px 8px', borderBottom: '1px solid rgba(255,255,255,0.06)' }} />
                        {!isIconOnly && (
                            <div style={{ fontSize: '9px', fontWeight: 800, color: '#7c8697', textTransform: 'uppercase', letterSpacing: '1.5px', padding: '0 16px 6px' }}>Admin</div>
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
            <div style={{ flex: '0 0 auto', padding: isIconOnly ? '12px 6px' : '12px 10px', marginTop: isIconOnly ? 'auto' : '14px' }}>
                {!isIconOnly && (
                    <>
                        <div style={{ borderBottom: '1px solid rgba(255,255,255,0.06)', marginBottom: '12px' }} />
                        <div style={{ fontSize: '9px', fontWeight: 800, color: '#7c8697', textTransform: 'uppercase', letterSpacing: '1.5px', paddingLeft: '6px', marginBottom: '10px' }}>
                            Pipeline
                        </div>
                    </>
                )}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    {/* Sourced */}
                    <div style={{
                        background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)',
                        borderRadius: '12px', padding: isIconOnly ? '10px 0' : '10px 12px',
                        display: 'flex', flexDirection: isIconOnly ? 'column' : 'row',
                        alignItems: 'center', gap: isIconOnly ? '2px' : '8px',
                        boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.02)',
                    }}>
                        <div style={{ fontSize: isIconOnly ? '14px' : '18px', fontWeight: 800, color: '#f4f4f5', lineHeight: 1 }}>
                            {sourcedCount == null ? '...' : sourcedCount.toLocaleString()}
                        </div>
                        {isIconOnly
                            ? <div style={{ fontSize: '7px', color: '#a1a1aa', fontWeight: 700, textAlign: 'center', textTransform: 'uppercase', letterSpacing: '0.5px' }}>SRC</div>
                            : <div style={{ fontSize: '9px', textTransform: 'uppercase', color: '#a1a1aa', fontWeight: 700, letterSpacing: '0.08em' }}>Total Sourced</div>
                        }
                    </div>
                    {/* Shortlisted */}
                    <div style={{
                        background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)',
                        borderRadius: '12px', padding: isIconOnly ? '10px 0' : '10px 12px',
                        display: 'flex', flexDirection: isIconOnly ? 'column' : 'row',
                        alignItems: 'center', gap: isIconOnly ? '2px' : '8px',
                        boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.02)',
                    }}>
                        <div style={{ fontSize: isIconOnly ? '14px' : '18px', fontWeight: 800, color: '#f4f4f5', lineHeight: 1 }}>
                            {shortlistedCount == null ? '...' : shortlistedCount.toLocaleString()}
                        </div>
                        {isIconOnly
                            ? <div style={{ fontSize: '7px', color: '#a1a1aa', fontWeight: 700, textAlign: 'center', textTransform: 'uppercase', letterSpacing: '0.5px' }}>SHL</div>
                            : <div style={{ fontSize: '9px', textTransform: 'uppercase', color: '#a1a1aa', fontWeight: 700, letterSpacing: '0.08em' }}>Shortlisted</div>
                        }
                    </div>
                </div>
            </div>

            {/* ── Spacer only when expanded (not icon-only) ── */}
            {!isIconOnly && <div style={{ flex: 1 }} />}

            {/* ── User Profile ── */}
            <div style={{
                padding: isIconOnly ? '10px 6px' : '14px 10px 12px',
                borderTop: isIconOnly ? '1px solid rgba(255,255,255,0.07)' : 'none',
                display: 'flex', alignItems: 'center',
                gap: isIconOnly ? 0 : '8px',
                flexDirection: isIconOnly ? 'column' : 'row',
                flexShrink: 0, marginTop: isIconOnly ? '0' : '0',
            }}>
                <div style={{
                    width: '100%',
                    padding: isIconOnly ? '0' : '12px 12px',
                    borderRadius: isIconOnly ? '0' : '16px',
                    border: isIconOnly ? 'none' : '1px solid rgba(255,255,255,0.07)',
                    background: isIconOnly ? 'transparent' : 'rgba(255,255,255,0.03)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: isIconOnly ? 0 : '10px',
                    flexDirection: isIconOnly ? 'column' : 'row',
                }}>
                    <div style={{
                        width: 30, height: 30, borderRadius: '50%', flexShrink: 0,
                        background: 'linear-gradient(135deg, #c97b35 0%, #7c3f13 100%)',
                        color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontWeight: 700, fontSize: '10px', boxShadow: '0 10px 18px rgba(15,23,42,0.24)',
                    }}>
                        {user?.full_name ? user.full_name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0,2) : '??'}
                    </div>
                    {!isIconOnly && (
                        <div style={{ flex: 1, overflow: 'hidden', minWidth: 0 }}>
                            <div style={{ fontSize: '12px', fontWeight: 600, color: '#f4f4f5', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                {user?.full_name || 'User'}
                            </div>
                            <div style={{ fontSize: '10px', color: '#8b93a1' }}>
                                {role === 'admin' ? 'Admin' : 'Recruiter'}
                            </div>
                        </div>
                    )}
                    <button onClick={() => useAppStore.getState().logout()}
                        style={{
                            background: 'none',
                            border: 'none',
                            cursor: 'pointer',
                            padding: '3px',
                            color: '#8b93a1',
                            flexShrink: 0,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                        }}
                        title="Log Out"
                    >
                        <LogOut size={14} />
                    </button>
                </div>
            </div>

            {/* ── Drag Handle ── */}
            <div onMouseDown={onMouseDown}
                style={{ position: 'absolute', top: 0, right: 0, bottom: 0, width: '6px', cursor: 'col-resize', zIndex: 100 }}
                onMouseEnter={e => e.currentTarget.style.background = 'rgba(249,115,22,0.14)'}
                onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                title="Drag to resize"
            />
        </aside>
    )
}

export default Sidebar
