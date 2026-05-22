import React, { useEffect, useState } from 'react';
import { useAppStore } from '../store/useAppStore';
import {
  UserPlus, Trash2, X, Mail, User, Lock,
  ChevronDown, Loader2, Monitor,
  LayoutGrid, MessagesSquare, Mic, Search, RefreshCw,
  Shield, CheckCircle2, Users, BarChart2
} from 'lucide-react';
import { toast } from 'sonner';
import { useShallow } from 'zustand/react/shallow';

// ── Tool definitions ─────────────────────────────────────────
const TOOLS = [
  { id: 'screening', label: 'Screening',    icon: Search,         color: '#475569', bg: '#f8fafc' },
  { id: 'roles',     label: 'Manage Roles', icon: LayoutGrid,     color: '#8b6b44', bg: '#fcf8f2' },
  { id: 'campaigns', label: 'Campaigns',    icon: Monitor,        color: '#166534', bg: '#f3faf5' },
  { id: 'messages',  label: 'Messages',     icon: MessagesSquare, color: '#1d4ed8', bg: '#f5f9ff' },
  { id: 'calls',     label: 'Calls',        icon: Mic,            color: '#334155', bg: '#f8fafc' },
  { id: 'talent_pool', label: 'Talent Pool', icon: Users,          color: '#7c3f13', bg: '#faf5ef' },
];

const PANEL_STYLE = {
  background: 'rgba(255,255,255,0.84)',
  backdropFilter: 'blur(16px)',
  border: '1px solid rgba(226,232,240,0.92)',
  boxShadow: '0 18px 36px rgba(15,23,42,0.05)',
};

const PRIMARY_BUTTON_STYLE = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: '8px',
  padding: '10px 18px',
  background: '#111827',
  color: '#fff',
  border: '1px solid #111827',
  borderRadius: '12px',
  fontWeight: 700,
  fontSize: '14px',
  cursor: 'pointer',
  fontFamily: 'inherit',
  boxShadow: '0 12px 24px rgba(15,23,42,0.12)',
  transition: 'all 0.15s',
};

const SECONDARY_BUTTON_STYLE = {
  padding: '12px',
  background: '#fff',
  border: '1px solid rgba(203,213,225,0.9)',
  borderRadius: '10px',
  color: '#475569',
  fontWeight: 600,
  fontSize: '14px',
  cursor: 'pointer',
  fontFamily: 'inherit',
};

// ── Compact Toggle ────────────────────────────────────────────
function Toggle({ checked, onChange, color }) {
  return (
    <button
      type="button"
      onClick={e => { e.stopPropagation(); onChange(); }}
      style={{
        position: 'relative',
        width: '34px', height: '20px',
        background: checked ? color : '#e2e8f0',
        border: 'none', borderRadius: '10px',
        cursor: 'pointer', transition: 'background 0.2s',
        outline: 'none', flexShrink: 0,
      }}
    >
      <span style={{
        position: 'absolute', top: '3px',
        left: checked ? '17px' : '3px',
        width: '14px', height: '14px',
        background: '#fff', borderRadius: '50%',
        transition: 'left 0.18s',
        boxShadow: '0 1px 4px rgba(0,0,0,0.18)',
      }} />
    </button>
  );
}

// ── Avatar ────────────────────────────────────────────────────
function Avatar({ name }) {
  const initials = (name || '?').split(' ').map(n => n[0]).join('').slice(0, 2).toUpperCase();
  const palette = ['#475569','#8b6b44','#166534','#334155','#1d4ed8','#7c3f13','#6b7280'];
  const color = palette[(name?.charCodeAt(0) || 0) % palette.length];
  return (
    <div style={{
      width: '36px', height: '36px', borderRadius: '10px',
      background: color + '18', border: `1.5px solid ${color}35`,
      color, fontWeight: 800, fontSize: '12px',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      flexShrink: 0, letterSpacing: '-0.5px', fontFamily: 'inherit',
    }}>
      {initials}
    </div>
  );
}

// ── Input for Modal ───────────────────────────────────────────
function ModalInput({ icon: Icon, label, type = 'text', value, onChange, placeholder, required }) {
  const [focused, setFocused] = useState(false);
  return (
    <div style={{ marginBottom: '16px' }}>
      <label style={{ display: 'block', fontSize: '12px', fontWeight: 700, color: '#64748b', marginBottom: '7px', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
        {label}
      </label>
      <div style={{ position: 'relative' }}>
        <span style={{ position: 'absolute', left: '13px', top: '50%', transform: 'translateY(-50%)', color: '#94a3b8', display: 'flex', alignItems: 'center' }}>
          <Icon size={16} />
        </span>
        <input
          type={type} value={value} onChange={onChange}
          placeholder={placeholder} required={required}
          onFocus={() => setFocused(true)} onBlur={() => setFocused(false)}
          style={{
            width: '100%', padding: '11px 13px 11px 40px',
            background: '#fff',
            border: `1px solid ${focused ? 'rgba(194, 124, 63, 0.4)' : 'rgba(203,213,225,0.9)'}`,
            borderRadius: '10px', color: '#0f172a', fontSize: '14px',
            outline: 'none', transition: 'border-color 0.2s, box-shadow 0.2s',
            boxShadow: focused ? '0 0 0 3px rgba(194,124,63,0.10)' : 'none',
            fontFamily: 'inherit',
          }}
        />
      </div>
    </div>
  );
}

// ── Skeleton Row ──────────────────────────────────────────────
function SkeletonRow() {
  return (
    <tr>
      <td style={{ padding: '16px 20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ width: 36, height: 36, borderRadius: 10, background: '#f1f5f9' }} />
          <div>
            <div style={{ width: 110, height: 11, borderRadius: 6, background: '#f1f5f9', marginBottom: 6 }} />
            <div style={{ width: 150, height: 9, borderRadius: 6, background: '#f8fafc' }} />
          </div>
        </div>
      </td>
      <td style={{ padding: '16px 20px' }}>
        <div style={{ display: 'flex', gap: 8 }}>
          {[80, 100, 90].map(w => <div key={w} style={{ width: w, height: 24, borderRadius: 6, background: '#f1f5f9' }} />)}
        </div>
      </td>
      <td style={{ padding: '16px 20px', textAlign: 'right' }}>
        <div style={{ width: 36, height: 36, borderRadius: 10, background: '#f1f5f9', marginLeft: 'auto' }} />
      </td>
    </tr>
  );
}

// ── Main Component ────────────────────────────────────────────
const UserManagement = () => {
  const { recruiters, fetchRecruiters, createRecruiter, updateRecruiterPermissions, deleteRecruiter } = useAppStore(useShallow((state) => ({
    recruiters: state.recruiters,
    fetchRecruiters: state.fetchRecruiters,
    createRecruiter: state.createRecruiter,
    updateRecruiterPermissions: state.updateRecruiterPermissions,
    deleteRecruiter: state.deleteRecruiter,
  })));
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [newUser, setNewUser] = useState({ name: '', email: '', password: '' });
  const [loading, setLoading] = useState(false);
  const [isFetching, setIsFetching] = useState(false);
  const [expandedId, setExpandedId] = useState(null);
  const [deletingId, setDeletingId] = useState(null);

  useEffect(() => {
    setIsFetching(true);
    fetchRecruiters().finally(() => setIsFetching(false));
  }, []);

  const handleAddUser = async (e) => {
    e.preventDefault();
    if (!newUser.name || !newUser.email || !newUser.password) {
      toast.error('Please fill in all fields.'); return;
    }
    setLoading(true);
    const res = await createRecruiter(newUser);
    if (res.success) {
      setIsModalOpen(false);
      setNewUser({ name: '', email: '', password: '' });
      toast.success('Recruiter added!');
    } else {
      toast.error(res.error || 'Failed to create recruiter.');
    }
    setLoading(false);
  };

  const togglePermission = async (recruiter, toolId) => {
    const cur = recruiter.permissions || {};
    const res = await updateRecruiterPermissions(recruiter.id, { ...cur, [toolId]: !cur[toolId] });
    if (!res.success) toast.error('Failed to update permission.');
  };

  const handleDelete = async (id, name) => {
    if (!window.confirm(`Archive "${name}"? They will no longer be able to sign in; their candidate pool remains visible to admins.`)) return;
    setDeletingId(id);
    const res = await deleteRecruiter(id);
    if (res.success) toast.success('Recruiter archived.');
    else toast.error('Failed to archive recruiter.');
    setDeletingId(null);
  };

  const refresh = async () => {
    setIsFetching(true);
    await fetchRecruiters();
    setIsFetching(false);
    toast.success('Refreshed!');
  };

  // Stats
  const totalPerms = recruiters.reduce((sum, r) => sum + TOOLS.filter(t => r.permissions?.[t.id]).length, 0);

  // Recruiter Chart
  const recruiterStats = useAppStore.getState().analytics?.recruiter_performance || [];
  const maxSourced = Math.max(...recruiterStats.map(r => r.sourced), 1);

  return (
    <div style={{ fontFamily: '"Inter", -apple-system, sans-serif', padding: '8px 0 12px' }}>

      {/* Header */}
      <div style={{ ...PANEL_STYLE, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '24px', padding: '24px 28px', borderRadius: '24px' }}>
        <div>
          <div style={{ fontSize: '11px', fontWeight: 700, color: '#8b6b44', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '6px' }}>
            Team operations
          </div>
          <h1 style={{ fontSize: '24px', fontWeight: 800, color: '#0f172a', marginBottom: '5px', letterSpacing: '-0.5px' }}>
            Recruiter Management
          </h1>
          <p style={{ color: '#94a3b8', fontSize: '14px' }}>
            Manage your team members and their tool permissions
          </p>
        </div>
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <button
            onClick={refresh}
            disabled={isFetching}
            title="Refresh"
            style={{
              width: '40px', height: '40px',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              background: '#f8fafc', border: '1.5px solid #e2e8f0',
              borderRadius: '10px', cursor: 'pointer', color: '#64748b',
            }}
          >
            <RefreshCw size={16} style={{ animation: isFetching ? 'spin 1s linear infinite' : 'none' }} />
          </button>
          <button
            onClick={() => setIsModalOpen(true)}
            style={PRIMARY_BUTTON_STYLE}
            onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-1px)'; e.currentTarget.style.boxShadow = '0 16px 28px rgba(15,23,42,0.16)'; }}
            onMouseLeave={e => { e.currentTarget.style.transform = 'none'; e.currentTarget.style.boxShadow = '0 12px 24px rgba(15,23,42,0.12)'; }}
          >
            <UserPlus size={16} /> Add Recruiter
          </button>
        </div>
      </div>

      {/* Stats */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '14px', marginBottom: '24px' }}>
        {[
          { label: 'Team Members', value: recruiters.length, accent: '#8b6b44', icon: User, loading: isFetching && recruiters.length === 0 },
          { label: 'Permissions Granted', value: totalPerms, accent: '#475569', icon: Shield, loading: isFetching && recruiters.length === 0 },
          { label: 'Tools Available', value: TOOLS.length, accent: '#166534', icon: CheckCircle2, loading: false },
        ].map(({ label, value, accent, icon: Icon, loading }) => (
          <div key={label} style={{
            ...PANEL_STYLE,
            borderRadius: '14px', padding: '18px 20px',
            display: 'flex', alignItems: 'center', gap: '14px',
          }}>
            <div style={{ width: 40, height: 40, borderRadius: '10px', background: accent + '12', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Icon size={18} color={accent} />
            </div>
            <div>
              {loading ? (
                <div style={{ width: 48, height: 22, borderRadius: 6, background: 'linear-gradient(90deg,#f1f5f9 25%,#e2e8f0 50%,#f1f5f9 75%)', backgroundSize: '200% 100%', animation: 'shimmer 1.2s infinite', marginBottom: 6 }} />
              ) : (
                <div style={{ fontSize: '22px', fontWeight: 900, color: '#0f172a', letterSpacing: '-0.5px', lineHeight: 1 }}>{value}</div>
              )}
              <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '2px' }}>{label}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Charts Section (Admin Special) */}
      <div style={{ 
        ...PANEL_STYLE,
        borderRadius: '20px', 
        padding: '24px', 
        marginBottom: '24px',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <div>
            <h2 style={{ fontSize: '18px', fontWeight: 800, color: '#0f172a', marginBottom: '4px' }}>Recruiter Performance</h2>
            <p style={{ fontSize: '13px', color: '#94a3b8' }}>Comparing sourcing volume and successful conversions</p>
          </div>
          <BarChart2 size={20} color="#64748b" />
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {recruiterStats.map((stat, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
              <div style={{ width: '120px', fontSize: '13px', fontWeight: 700, color: '#475569', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {stat.recruiter}
              </div>
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '4px' }}>
                {/* Sourced Bar */}
                <div style={{ height: '16px', background: '#f8fafc', borderRadius: '4px', overflow: 'hidden', position: 'relative', display: 'flex', alignItems: 'center' }}>
                  <div style={{ 
                    width: `${(stat.sourced / maxSourced) * 100}%`, 
                    height: '100%', 
                    background: '#8b6b44',
                    transition: 'width 0.6s cubic-bezier(0.16, 1, 0.3, 1)'
                  }} />
                  <div style={{ position: 'absolute', right: '8px', fontSize: '10px', fontWeight: 800, color: stat.sourced / maxSourced > 0.8 ? '#fff' : '#64748b' }}>
                    {stat.sourced}
                  </div>
                </div>
                {/* Shortlisted Bar */}
                <div style={{ height: '16px', background: '#f8fafc', borderRadius: '4px', overflow: 'hidden', position: 'relative', display: 'flex', alignItems: 'center' }}>
                  <div style={{ 
                    width: `${(stat.shortlisted / maxSourced) * 100}%`, 
                    height: '100%', 
                    background: '#22c55e',
                    transition: 'width 0.6s cubic-bezier(0.16, 1, 0.3, 1)'
                  }} />
                  <div style={{ position: 'absolute', right: '8px', fontSize: '10px', fontWeight: 800, color: stat.shortlisted / maxSourced > 0.8 ? '#fff' : '#64748b' }}>
                    {stat.shortlisted}
                  </div>
                </div>
              </div>
              <div style={{ width: '80px', display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '2px' }}>
                 <span style={{ fontSize: '11px', fontWeight: 700, color: '#8b6b44' }}>Sourced</span>
                 <span style={{ fontSize: '11px', fontWeight: 700, color: '#22c55e' }}>Shortlisted</span>
              </div>
            </div>
          ))}
          {recruiterStats.length === 0 && (
            <div style={{ padding: '40px', textAlign: 'center', color: '#cbd5e1', fontSize: '14px' }}>
              No performance data available yet
            </div>
          )}
        </div>
      </div>

      {/* Table */}
      <div style={{ ...PANEL_STYLE, borderRadius: '18px', overflow: 'hidden' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ background: '#f8fafc', borderBottom: '1.5px solid #f1f5f9' }}>
              {['Recruiter', 'Active Permissions', 'Actions'].map((h, i) => (
                <th key={h} style={{
                  padding: '12px 20px', textAlign: i === 2 ? 'right' : 'left',
                  fontSize: '11px', fontWeight: 700, color: '#94a3b8',
                  textTransform: 'uppercase', letterSpacing: '0.07em',
                  width: i === 0 ? '30%' : i === 2 ? '90px' : 'auto',
                }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {/* Loading skeleton - only when no cached data */}
            {isFetching && recruiters.length === 0 && (
              <><SkeletonRow /><SkeletonRow /><SkeletonRow /></>
            )}

            {/* Empty state */}
            {!isFetching && recruiters.length === 0 && (
              <tr>
                <td colSpan={3} style={{ padding: '70px 20px', textAlign: 'center' }}>
                  <div style={{ display: 'inline-flex', flexDirection: 'column', alignItems: 'center', gap: '12px' }}>
                    <div style={{ width: 56, height: 56, background: '#f8fafc', border: '1.5px solid #e2e8f0', borderRadius: '16px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <UserPlus size={24} color="#cbd5e1" />
                    </div>
                    <p style={{ color: '#64748b', fontWeight: 600, fontSize: '15px', margin: 0 }}>No recruiters yet</p>
                    <p style={{ color: '#cbd5e1', fontSize: '13px', margin: 0 }}>Add your first team member to get started</p>
                    <button onClick={() => setIsModalOpen(true)} style={{ marginTop: '4px', padding: '8px 18px', background: '#fff', border: '1px solid rgba(203,213,225,0.9)', borderRadius: '10px', color: '#334155', fontWeight: 700, fontSize: '13px', cursor: 'pointer', fontFamily: 'inherit' }}>
                      + Add First Recruiter
                    </button>
                  </div>
                </td>
              </tr>
            )}

            {/* Recruiter rows */}
            {recruiters.map((recruiter, idx) => {
              const isExpanded = expandedId === recruiter.id;
              const enabledTools = TOOLS.filter(t => recruiter.permissions?.[t.id]);
              return (
                <React.Fragment key={recruiter.id}>
                  <tr
                    style={{
                      borderBottom: idx < recruiters.length - 1 || isExpanded ? '1px solid #f8fafc' : 'none',
                      transition: 'background 0.1s',
                      cursor: 'pointer',
                    }}
                    onMouseEnter={e => { if (!isExpanded) e.currentTarget.style.background = '#fafafa'; }}
                    onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; }}
                    onClick={() => setExpandedId(isExpanded ? null : recruiter.id)}
                  >
                    {/* Recruiter info */}
                    <td style={{ padding: '15px 20px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '11px' }}>
                        <Avatar name={recruiter.full_name} />
                        <div>
                          <div style={{ fontWeight: 700, color: '#0f172a', fontSize: '14px' }}>{recruiter.full_name}</div>
                          <div style={{ fontSize: '12px', color: '#94a3b8' }}>{recruiter.email}</div>
                        </div>
                      </div>
                    </td>

                    {/* Permission pills */}
                    <td style={{ padding: '15px 20px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
                        {enabledTools.length === 0 ? (
                          <span style={{ fontSize: '12px', color: '#cbd5e1', background: '#f8fafc', padding: '3px 10px', borderRadius: '6px', border: '1px solid #f1f5f9' }}>No access</span>
                        ) : enabledTools.map(t => {
                          const Icon = t.icon;
                          return (
                            <span key={t.id} style={{
                              display: 'inline-flex', alignItems: 'center', gap: '4px',
                              fontSize: '12px', fontWeight: 600,
                              color: t.color, background: t.bg,
                              padding: '3px 10px', borderRadius: '6px',
                              border: `1px solid ${t.color}25`,
                            }}>
                              <Icon size={11} />{t.label}
                            </span>
                          );
                        })}
                        <ChevronDown size={14} color="#cbd5e1" style={{ marginLeft: '4px', transform: isExpanded ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s', flexShrink: 0 }} />
                      </div>
                    </td>

                    {/* Delete */}
                    <td style={{ padding: '15px 20px', textAlign: 'right' }}>
                      <button
                        onClick={e => { e.stopPropagation(); handleDelete(recruiter.id, recruiter.full_name); }}
                        disabled={deletingId === recruiter.id}
                        style={{
                          width: '36px', height: '36px',
                          display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                          background: '#fff5f5', border: '1.5px solid #fecaca',
                          borderRadius: '9px', color: '#ef4444',
                          cursor: 'pointer', transition: 'all 0.15s',
                        }}
                        onMouseEnter={e => { e.currentTarget.style.background = '#fee2e2'; }}
                        onMouseLeave={e => { e.currentTarget.style.background = '#fff5f5'; }}
                      >
                        {deletingId === recruiter.id
                          ? <Loader2 size={14} style={{ animation: 'spin 1s linear infinite' }} />
                          : <Trash2 size={14} />}
                      </button>
                    </td>
                  </tr>

                  {/* Expanded permission panel */}
                  {isExpanded && (
                    <tr>
                      <td colSpan={3} style={{ padding: '0', background: '#fafbff', borderBottom: idx < recruiters.length - 1 ? '1px solid #f1f5f9' : 'none' }}>
                        <div style={{ padding: '18px 24px 20px' }}>
                          <p style={{ fontSize: '11px', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: '14px' }}>
                            Tool access for {recruiter.full_name}
                          </p>
                          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(170px,1fr))', gap: '10px' }}>
                            {TOOLS.map(tool => {
                              const Icon = tool.icon;
                              const enabled = !!recruiter.permissions?.[tool.id];
                              return (
                                <div
                                  key={tool.id}
                                  onClick={e => { e.stopPropagation(); togglePermission(recruiter, tool.id); }}
                                  style={{
                                    display: 'flex', alignItems: 'center', gap: '10px',
                                    padding: '11px 14px',
                                    background: enabled ? tool.bg : '#fff',
                                    border: `1.5px solid ${enabled ? tool.color + '30' : '#e2e8f0'}`,
                                    borderRadius: '12px', cursor: 'pointer',
                                    transition: 'all 0.15s',
                                    boxShadow: enabled ? `0 1px 4px ${tool.color}15` : 'none',
                                  }}
                                >
                                  <div style={{
                                    width: '32px', height: '32px',
                                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                                    background: enabled ? tool.color + '18' : '#f8fafc',
                                    borderRadius: '8px', flexShrink: 0,
                                  }}>
                                    <Icon size={15} color={enabled ? tool.color : '#94a3b8'} />
                                  </div>
                                  <span style={{ flex: 1, fontSize: '13px', fontWeight: 700, color: enabled ? '#0f172a' : '#94a3b8' }}>
                                    {tool.label}
                                  </span>
                                  <Toggle checked={enabled} onChange={() => togglePermission(recruiter, tool.id)} color={tool.color} />
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Add Recruiter Modal */}
      {isModalOpen && (
        <div
          onClick={e => { if (e.target === e.currentTarget) setIsModalOpen(false); }}
          style={{
            position: 'fixed', inset: 0,
            background: 'rgba(15,23,42,0.35)',
            backdropFilter: 'blur(6px)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            zIndex: 1000, padding: '16px',
          }}
        >
          <div style={{
            background: '#fff',
            border: '1px solid rgba(226,232,240,0.92)',
            borderRadius: '20px',
            width: '100%', maxWidth: '420px',
            padding: '32px',
            boxShadow: '0 20px 60px rgba(0,0,0,0.15)',
            animation: 'slideUp 0.22s ease',
          }}>
            {/* Modal header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '24px' }}>
              <div>
                <h2 style={{ fontSize: '18px', fontWeight: 800, color: '#0f172a', marginBottom: '3px' }}>Add New Recruiter</h2>
                <p style={{ fontSize: '13px', color: '#94a3b8' }}>Create a team member account with login access</p>
              </div>
              <button
                onClick={() => setIsModalOpen(false)}
                style={{ width: '32px', height: '32px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#f8fafc', border: '1.5px solid #e2e8f0', borderRadius: '8px', cursor: 'pointer', color: '#94a3b8' }}
              >
                <X size={15} />
              </button>
            </div>

            <form onSubmit={handleAddUser}>
              <ModalInput icon={User} label="Full Name" value={newUser.name} onChange={e => setNewUser({ ...newUser, name: e.target.value })} placeholder="Jane Smith" required />
              <ModalInput icon={Mail} label="Email Address" type="email" value={newUser.email} onChange={e => setNewUser({ ...newUser, email: e.target.value })} placeholder="jane@company.com" required />
              <ModalInput icon={Lock} label="Temporary Password" type="password" value={newUser.password} onChange={e => setNewUser({ ...newUser, password: e.target.value })} placeholder="Min. 8 characters" required />

              <div style={{ display: 'flex', gap: '10px', marginTop: '24px' }}>
                <button type="button" onClick={() => setIsModalOpen(false)}
                  style={{ ...SECONDARY_BUTTON_STYLE, flex: 1 }}>
                  Cancel
                </button>
                <button type="submit" disabled={loading}
                  style={{ ...PRIMARY_BUTTON_STYLE, flex: 1.5, padding: '12px', opacity: loading ? 0.75 : 1, cursor: loading ? 'not-allowed' : 'pointer' }}>
                  {loading ? <Loader2 size={17} style={{ animation: 'spin 1s linear infinite' }} /> : <><UserPlus size={16} /> Create Recruiter</>}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes slideUp { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes shimmer { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }
      `}</style>
    </div>
  );
};

export default UserManagement;
