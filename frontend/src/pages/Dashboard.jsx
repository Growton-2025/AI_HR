import React, { useEffect, useRef, useState } from 'react';
import { useAppStore } from '../store/useAppStore';
import { Users, Activity, MessageSquareMore, TrendingUp, Globe, BarChart2, Layers, Award, PhoneCall } from 'lucide-react';
import { useShallow } from 'zustand/react/shallow';
import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer
} from 'recharts';

const COLORS_GEO   = ['#6b7280', '#2563eb', '#8b6b44', '#0f766e', '#475569', '#4f46e5', '#15803d', '#334155'];
const COLORS_IND   = ['#475569', '#9a6b28', '#166534', '#1d4ed8', '#0f766e', '#7c3f13', '#5b21b6', '#64748b'];
const COLORS_SEG   = ['#334155', '#8b6b44', '#1d4ed8', '#166534', '#475569', '#0f766e', '#7c3f13', '#6366f1'];

const getEntryColor = (name, index, palette) => {
  if (name === 'Other' || name === 'Unknown') return '#475569';
  return palette[index % palette.length];
};

const DashboardCard = ({ title, value, subtext, icon: Icon, color, loading }) => (
  <div
    style={{
      background: 'rgba(255,255,255,0.92)',
      borderRadius: '14px',
      padding: '18px 20px',
      boxShadow: '0 12px 28px rgba(15,23,42,0.04)',
      border: '1px solid rgba(226,232,240,0.92)',
      display: 'flex', flexDirection: 'column', gap: '14px',
      transition: 'transform 0.2s, box-shadow 0.2s',
      position: 'relative',
      overflow: 'hidden'
    }}
  >
    {loading && <div className="shimmer" style={{ position: 'absolute', inset: 0, opacity: 0.1, zIndex: 1 }} />}
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '10px' }}>
      <div style={{ 
        fontSize: '12px', fontWeight: 800, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em',
        width: loading ? '112px' : 'auto', height: loading ? '14px' : 'auto',
        background: loading ? '#f1f5f9' : 'transparent', borderRadius: '4px'
      }}>
        {!loading && title}
      </div>
      <div style={{ width: '34px', height: '34px', borderRadius: '10px', background: loading ? '#f1f5f9' : `${color}12`, display: 'flex', alignItems: 'center', justifyContent: 'center', border: loading ? 'none' : `1px solid ${color}18`, flexShrink: 0 }}>
        {!loading && <Icon size={17} color={color} />}
      </div>
    </div>
    <div>
      <div style={{ 
        fontSize: '30px', fontWeight: 900, color: '#0f172a', letterSpacing: 0, lineHeight: 1,
        width: loading ? '100px' : 'auto', height: loading ? '30px' : 'auto',
        background: loading ? '#f1f5f9' : 'transparent', borderRadius: '8px'
      }}>{!loading && value}</div>
      {!loading && subtext && <div style={{ fontSize: '12px', color: '#64748b', marginTop: '8px', lineHeight: 1.45 }}>{subtext}</div>}
    </div>
  </div>
);

const ChartCard = ({ title, subtitle, icon: Icon, iconColor, children }) => (
  <div style={{
    background: 'rgba(255,255,255,0.9)',
    backdropFilter: 'blur(16px)',
    borderRadius: '16px',
    padding: '24px',
    border: '1px solid rgba(226,232,240,0.92)',
    boxShadow: '0 14px 30px rgba(15,23,42,0.04)',
  }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
      <h2 style={{ fontSize: '16px', fontWeight: 800, color: '#0f172a', margin: 0 }}>{title}</h2>
      <Icon size={18} color={iconColor || '#64748b'} />
    </div>
    <p style={{ fontSize: '13px', color: '#64748b', marginBottom: '16px', marginTop: 0 }}>{subtitle}</p>
    {children}
  </div>
);

const DonutCenter = ({ total, label }) => (
  <g>
    <text x="50%" y="46%" textAnchor="middle" dominantBaseline="central" style={{ fontSize: '22px', fontWeight: 900, fill: '#0f172a' }}>
      {total?.toLocaleString()}
    </text>
    <text x="50%" y="62%" textAnchor="middle" dominantBaseline="central" style={{ fontSize: '11px', fontWeight: 600, fill: '#94a3b8' }}>
      {label}
    </text>
  </g>
);

const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const d = payload[0];
    return (
      <div style={{ background: '#1e293b', borderRadius: '10px', padding: '10px 14px', color: '#fff', fontSize: '13px', boxShadow: '0 8px 24px rgba(0,0,0,0.2)' }}>
        <div style={{ fontWeight: 700 }}>{d.name}</div>
        <div style={{ color: d.payload.fill, fontWeight: 800, fontSize: '16px' }}>{d.value?.toLocaleString()}</div>
      </div>
    );
  }
  return null;
};

const CHART_MODES = ['Geography', 'Industry', 'Segment'];

const normalizeCount = (value) => {
  if (value == null || value === '') return null;
  const count = Number(value);
  return Number.isFinite(count) ? count : null;
};

const Dashboard = () => {
  const { user, analytics, fetchAnalytics, callStats, fetchCallStats } = useAppStore(useShallow((state) => ({
    user: state.user,
    analytics: state.analytics,
    fetchAnalytics: state.fetchAnalytics,
    callStats: state.callStats,
    fetchCallStats: state.fetchCallStats,
  })));
  const [activeMode, setActiveMode] = useState(0);
  const [isRevalidating, setIsRevalidating] = useState(false);
  const hasLoadedRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      if (hasLoadedRef.current && !cancelled) {
        setIsRevalidating(true);
      }
      await Promise.allSettled([
        fetchAnalytics(),
        fetchCallStats(),
      ]);
      if (cancelled) return;
      hasLoadedRef.current = true;
      setIsRevalidating(false);
    };
    void run();
    return () => {
      cancelled = true;
    };
  }, [fetchAnalytics, fetchCallStats]);

  const isAdmin = user?.role === 'admin';

  const analyticsMetrics = analytics?.summary || {};
  const analyticsTotal = normalizeCount(analyticsMetrics.total_sourced);
  const displayMetrics = {
    ...analyticsMetrics,
    total_sourced: analyticsTotal ?? 0,
    shortlisted: normalizeCount(analyticsMetrics.shortlisted) ?? 0,
    in_conversation: normalizeCount(analyticsMetrics.in_conversation) ?? 0,
  };
    
  // If personal pipeline health, compute in_conversation dynamically if not preset
  let inConv = displayMetrics.in_conversation || 0;
  if (!isAdmin && analytics?.personal?.pipeline_health) {
      inConv = (analytics.personal.pipeline_health['In Conversation'] || 0) + (analytics.personal.pipeline_health['Client Interviewing'] || 0);
  }

  const totalCalls = (callStats?.due_today || 0) + (callStats?.upcoming || 0) + (callStats?.completed || 0);

  const metricCards = [
    { title: 'Total Sourced', value: displayMetrics.total_sourced?.toLocaleString(), subtext: 'Leads in talent pool', icon: Users, color: '#8b6b44' },
    { title: 'Shortlisted', value: displayMetrics.shortlisted?.toLocaleString(), subtext: 'Approved leads', icon: Activity, color: '#166534' },
    { title: 'Call Operations', value: totalCalls.toLocaleString(), subtext: `${callStats?.due_today || 0} Ongoing · ${callStats?.upcoming || 0} Upcoming · ${callStats?.completed || 0} Completed`, icon: PhoneCall, color: '#334155' },
    { title: 'Active Hub', value: ((displayMetrics.email_campaigns_active || 0) + (displayMetrics.linkedin_campaigns_active || 0)).toLocaleString(), subtext: `${displayMetrics.email_campaigns_active || 0} via Email · ${displayMetrics.linkedin_campaigns_active || 0} via LinkedIn`, icon: TrendingUp, color: '#475569' }
  ];

  const distributions = analytics?.distributions || { geo: [], industry: [], segment: [], functional: [] };

  const chartData = [
    { data: distributions.geo || [],        colors: COLORS_GEO,  label: 'leads' },
    { data: distributions.industry || [],   colors: COLORS_IND,  label: 'leads' },
    { data: distributions.segment || [],    colors: COLORS_SEG,  label: 'leads' },
  ];

  const current = chartData[activeMode];
  const total = current.data.reduce((s, d) => s + d.value, 0);

  return (
    <div style={{ padding: '24px 0 12px', fontFamily: '"Inter", sans-serif' }}>
      {/* Header */}
      <div style={{ marginBottom: '24px', padding: '0 4px' }}>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: '11px', fontWeight: 800, color: '#8b6b44', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '7px' }}>
            Team analytics
          </div>
          <h1 style={{ fontSize: '28px', fontWeight: 900, color: '#0f172a', letterSpacing: 0, lineHeight: 1.1, marginBottom: '5px' }}>
            Dashboard
          </h1>
          <p style={{ fontSize: '15px', color: '#64748b', margin: 0 }}>
            Pipeline health across sourcing, screening, calls, and outreach.
          </p>
        </div>
      </div>

      {/* Metric Cards */}
      <div style={{ 
        display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', 
        gap: '16px', marginBottom: '28px',
        opacity: isRevalidating ? 0.7 : 1,
        transition: 'opacity 0.2s'
      }}>
        {metricCards.map(m => (
          <DashboardCard key={m.title} {...m} />
        ))}
      </div>

      {/* Dynamic Charts */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: '24px' }}>

        {/* LEFT: Big donut + toggle */}
        <ChartCard
          title="Lead Distribution"
          subtitle="Explore how all leads are distributed by region, industry, segment, or functional area"
          icon={Globe}
          iconColor="#64748b"
        >
          {/* Tab Switch */}
          <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
            {CHART_MODES.map((mode, i) => (
              <button
                key={mode}
                onClick={() => setActiveMode(i)}
                style={{
                  padding: '6px 14px',
                  borderRadius: '999px',
                  border: activeMode === i ? '1px solid #111827' : '1px solid rgba(203,213,225,0.9)',
                  cursor: 'pointer',
                  fontSize: '12px',
                  fontWeight: 700,
                  transition: 'all 0.15s',
                  background: activeMode === i ? '#111827' : '#fff',
                  color: activeMode === i ? '#fff' : '#64748b',
                }}
              >
                {mode}
              </button>
            ))}
          </div>

          {current.data.length === 0 ? (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '240px', color: '#94a3b8', fontSize: '14px' }}>
              No data available yet.
            </div>
          ) : (
            <>
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie
                    data={current.data}
                    cx="50%"
                    cy="50%"
                    innerRadius={72}
                    outerRadius={100}
                    paddingAngle={2}
                    dataKey="value"
                    minAngle={15}
                    isAnimationActive={true}
                    animationBegin={0}
                    animationDuration={600}
                  >
                    {current.data.map((entry, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={getEntryColor(entry.name, index, current.colors)} 
                        stroke="#fff"
                        strokeWidth={1}
                      />
                    ))}
                    <DonutCenter total={total} label={current.label} />
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                </PieChart>
              </ResponsiveContainer>
              {/* Custom Legend Grid */}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 16px', marginTop: '12px' }}>
                {current.data.map((entry, index) => (
                  <div key={index} style={{ display: 'flex', alignItems: 'center', gap: '6px', minWidth: '120px' }}>
                    <div style={{ 
                      width: '8px', height: '8px', borderRadius: '50%', 
                      background: getEntryColor(entry.name, index, current.colors), 
                      flexShrink: 0 
                    }} />
                    <span style={{ fontSize: '11px', color: '#475569', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '130px' }}>
                      {entry.name}
                    </span>
                  </div>
                ))}
              </div>
            </>
          )}
        </ChartCard>

        {/* RIGHT: Ranked list */}
        <ChartCard
          title={`By ${CHART_MODES[activeMode]}`}
          subtitle="Top segments by lead count"
          icon={BarChart2}
          iconColor={activeMode === 0 ? '#8b6b44' : activeMode === 1 ? '#475569' : activeMode === 2 ? '#0f766e' : '#64748b'}
        >
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {(current.data.slice(0, 8)).map((item, i) => {
              const rawPct = total > 0 ? (item.value / total) * 100 : 0;
              const displayPct = (rawPct > 0 && rawPct < 1) ? '<1' : Math.round(rawPct);
              const color = getEntryColor(item.name, i, current.colors);
              return (
                <div key={i}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                    <span style={{ fontSize: '12px', fontWeight: 700, color: '#475569', maxWidth: '65%', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{item.name}</span>
                    <span style={{ fontSize: '12px', fontWeight: 800, color }}>
                      {item.value?.toLocaleString()} <span style={{ color: '#94a3b8', fontWeight: 600 }}>({displayPct}%)</span>
                    </span>
                  </div>
                  <div style={{ height: '6px', background: '#f1f5f9', borderRadius: '3px', overflow: 'hidden' }}>
                    <div style={{
                      height: '100%',
                      width: `${rawPct}%`,
                      minWidth: rawPct > 0 ? '2px' : '0',
                      background: color,
                      borderRadius: '3px',
                      transition: 'width 0.5s ease'
                    }} />
                  </div>
                </div>
              );
            })}
          </div>
        </ChartCard>
      </div>

      {/* Admin Only: Recruiter Performance */}
      {isAdmin && analytics?.recruiter_performance && analytics.recruiter_performance.length > 0 && (
        <div style={{ marginTop: '36px' }}>
          <ChartCard
            title="Team Performance"
            subtitle="Pipeline metrics broken down by recruiter"
            icon={Award}
            iconColor="#8b6b44"
          >
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', color: '#64748b', textAlign: 'left' }}>
                    <th style={{ padding: '12px 16px', fontWeight: 600 }}>Recruiter</th>
                    <th style={{ padding: '12px 16px', fontWeight: 600 }}>Leads Sourced</th>
                    <th style={{ padding: '12px 16px', fontWeight: 600 }}>Shortlisted</th>
                    <th style={{ padding: '12px 16px', fontWeight: 600 }}>In Conversation</th>
                  </tr>
                </thead>
                <tbody>
                  {analytics.recruiter_performance.map((recruiter, index) => (
                    <tr key={index} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '12px 16px', fontWeight: 500, color: '#0f172a' }}>{recruiter.recruiter}</td>
                      <td style={{ padding: '12px 16px', color: '#475569' }}>{recruiter.sourced?.toLocaleString() || 0}</td>
                      <td style={{ padding: '12px 16px', color: '#475569' }}>{recruiter.shortlisted?.toLocaleString() || 0}</td>
                      <td style={{ padding: '12px 16px', color: '#475569' }}>{recruiter.in_conversation?.toLocaleString() || 0}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </ChartCard>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
