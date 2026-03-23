import { NavLink } from 'react-router-dom'
import { useEffect } from 'react'
import { useAppStore } from '../store/useAppStore'
import { Search, Briefcase, Activity, MessageSquareMore, Phone, BarChart2, MoreHorizontal, LogOut, Users } from 'lucide-react'

function Sidebar() {
    const analytics = useAppStore(state => state.analytics)
    const fetchAnalytics = useAppStore(state => state.fetchAnalytics)

    useEffect(() => {
        fetchAnalytics()
    }, [fetchAnalytics])

    const user = useAppStore(state => state.user)
    const role = user?.role
    const permissions = user?.permissions || {}

    const allNavItems = [
        { path: "/", label: "Dashboard", icon: BarChart2, id: "dashboard" },
        { path: '/screening', label: 'Screening', icon: Search, id: 'screening' },
        { path: '/talent-pool', label: 'Talent Pool', icon: Users, id: 'talent_pool' },
        { path: '/roles', label: 'Manage Roles', icon: Briefcase, id: 'roles' },
        { path: '/campaigns', label: 'Campaigns', icon: Activity, id: 'campaigns' },
        { path: '/messages', label: 'Messages', icon: MessageSquareMore, id: 'messages' },
        { path: '/calls', label: 'Calls', icon: Phone, id: 'calls' }
    ]

    // Admins see everything, recruiters see only what's permitted
    const navItems = role === 'admin' 
        ? allNavItems 
        : allNavItems.filter(item => item.id === 'dashboard' || permissions[item.id])

    return (
        <aside className="app-sidebar">
            {/* Logo */}
            <div className="logo-container">
                <img
                    src="https://cdn.prod.website-files.com/65e41b0d7632a225ef3abc4e/693509bb8da9f3b5e10554be_LOGO.svg"
                    alt="Growton"
                    style={{ maxWidth: '160px', height: 'auto' }}
                />
                <div className="logo-subtitle">Talent Intelligence Platform</div>
            </div>

            {/* Navigation */}
            <div className="section-label">Navigation</div>
            <nav className="nav-menu">
                {navItems.map(item => {
                    const Icon = item.icon
                    return (
                        <NavLink
                            key={item.path}
                            to={item.path}
                            className={({ isActive }) => `nav-button ${isActive ? 'active' : ''}`}
                        >
                            <Icon size={18} style={{ marginRight: '12px' }} />
                            {item.label}
                        </NavLink>
                    )
                })}
                
                {/* Admin Only section */}
                {role === 'admin' && (
                    <>
                        <div style={{ margin: '20px 20px', borderBottom: '1px solid rgba(255,255,255,0.08)' }} />
                        <div className="section-label">Admin Controls</div>
                        <NavLink
                            to="/admin/users"
                            className={({ isActive }) => `nav-button ${isActive ? 'active' : ''}`}
                        >
                            <MoreHorizontal size={18} style={{ marginRight: '12px' }} />
                            User Management
                        </NavLink>
                    </>
                )}
            </nav>

            {/* Divider */}
            <div style={{ margin: '20px 20px', borderBottom: '1px solid rgba(255,255,255,0.08)' }} />

            {/* Stats */}
            <div className="section-label" style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
                <BarChart2 size={14} /> Pipeline Insights
            </div>
            
            <div className="stats-container" style={{ padding: '0 16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <div className="stat-card" style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)', borderRadius: '12px', padding: '12px' }}>
                    <div className="stat-value" style={{ fontSize: '20px', fontWeight: 800, color: '#f97316' }}>
                        {analytics?.summary?.total_sourced?.toLocaleString() || 0}
                    </div>
                    <div className="stat-label" style={{ fontSize: '10px', textTransform: 'uppercase', color: '#64748b', fontWeight: 700, letterSpacing: '0.05em' }}>
                        Total Sourced
                    </div>
                </div>

                <div className="stat-card" style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)', borderRadius: '12px', padding: '12px' }}>
                    <div className="stat-value" style={{ fontSize: '20px', fontWeight: 800, color: '#22c55e' }}>
                        {analytics?.summary?.shortlisted || 0}
                    </div>
                    <div className="stat-label" style={{ fontSize: '10px', textTransform: 'uppercase', color: '#64748b', fontWeight: 700, letterSpacing: '0.05em' }}>
                        Shortlisted
                    </div>
                </div>
            </div>

            {/* User Profile (Bottom) */}
            <div className="user-profile">
                <div className="user-avatar">
                    {user?.full_name ? user.full_name.split(' ').map(n => n[0]).join('').toUpperCase() : '??'}
                </div>
                <div className="user-info">
                    <div className="user-name">{user?.full_name || 'User'}</div>
                    <div className="user-role">
                        {role === 'admin' ? 'Administrator' : role === 'recruiter' ? 'Recruiter' : 'Loading...'}
                    </div>
                </div>

                {/* Logout */}
                <button
                    onClick={() => useAppStore.getState().logout()}
                    style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '4px', color: '#64748b', marginLeft: 'auto' }}
                    title="Log Out"
                >
                    <LogOut size={18} />
                </button>
            </div>
        </aside>
    )
}

export default Sidebar
