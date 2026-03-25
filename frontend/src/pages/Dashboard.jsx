import React, { useEffect, useState } from 'react';
import { useAppStore } from '../store/useAppStore';
import { Users, Activity, MessageSquareMore, TrendingUp, Globe, BarChart2, Layers, Award } from 'lucide-react';
import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer
} from 'recharts';

const COLORS_GEO   = ['#f97316', '#2563eb', '#7c3aed', '#0891b2', '#db2777', '#4f46e5', '#059669', '#334155'];
const COLORS_IND   = ['#4f46e5', '#d97706', '#059669', '#dc2626', '#0d9488', '#b45309', '#6d28d9', '#475569'];
const COLORS_SEG   = ['#0284c7', '#ea580c', '#6d28d9', '#16a34a', '#db2777', '#2563eb', '#475569', '#0d9488'];

const getEntryColor = (name, index, palette) => {
  if (name === 'Other' || name === 'Unknown') return '#475569';
  return palette[index % palette.length];
};

const DashboardCard = ({ title, value, subtext, icon: Icon, color }) => (
  <div
    style={{
      background: '#fff',
      borderRadius: '20px',
      padding: '24px',
      boxShadow: '0 4px 6px -1px rgba(0,0,0,0.02)',
      border: '1.5px solid #f1f5f9',
      display: 'flex', flexDirection: 'column', gap: '12px',
      transition: 'transform 0.2s, box-shadow 0.2s',
    }}
    onMouseEnter={e => {
      e.currentTarget.style.transform = 'translateY(-4px)';
      e.currentTarget.style.boxShadow = '0 20px 25px -5px rgba(0,0,0,0.05)';
    }}
    onMouseLeave={e => {
      e.currentTarget.style.transform = 'none';
      e.currentTarget.style.boxShadow = '0 4px 6px -1px rgba(0,0,0,0.02)';
    }}
  >
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
      <div style={{ width: '48px', height: '48px', borderRadius: '14px', background: `${color}12`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Icon size={24} color={color} />
      </div>
    </div>
    <div>
      <div style={{ fontSize: '32px', fontWeight: 900, color: '#0f172a', letterSpacing: '-1px' }}>{value}</div>
      <div style={{ fontSize: '14px', fontWeight: 700, color: '#64748b', marginTop: '2px' }}>{title}</div>
      {subtext && <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '4px' }}>{subtext}</div>}
    </div>
  </div>
);

const ChartCard = ({ title, subtitle, icon: Icon, iconColor, children }) => (
  <div style={{
    background: '#fff',
    borderRadius: '24px',
    padding: '28px',
    border: '1.5px solid #f1f5f9',
    boxShadow: '0 1px 3px rgba(0,0,0,0.02)',
  }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
      <h2 style={{ fontSize: '17px', fontWeight: 800, color: '#0f172a', margin: 0 }}>{title}</h2>
      <Icon size={18} color={iconColor || '#f97316'} />
    </div>
    <p style={{ fontSize: '13px', color: '#94a3b8', marginBottom: '16px', marginTop: 0 }}>{subtitle}</p>
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

const Dashboard = () => {
  const { user, analytics, fetchAnalytics } = useAppStore();
  const [activeMode, setActiveMode] = useState(0);

  useEffect(() => { fetchAnalytics(); }, [fetchAnalytics]);

  const isAdmin = user?.role === 'admin';

  // Use team-wide summary for all users so the top cards match the charts below
  const displayMetrics = analytics?.summary || { total_sourced: 0, shortlisted: 0, in_conversation: 0 };
    
  // If personal pipeline health, compute in_conversation dynamically if not preset
  let inConv = displayMetrics.in_conversation || 0;
  if (!isAdmin && analytics?.personal?.pipeline_health) {
      inConv = (analytics.personal.pipeline_health['In Conversation'] || 0) + (analytics.personal.pipeline_health['Client Interviewing'] || 0);
  }

  const metricCards = [
    { title: 'Total Sourced', value: displayMetrics.total_sourced?.toLocaleString(), subtext: 'Leads in talent pool', icon: Users, color: '#f97316' },
    { title: 'Shortlisted', value: displayMetrics.shortlisted?.toLocaleString(), subtext: 'Approved leads', icon: Activity, color: '#22c55e' },
    { title: 'In Conversation', value: inConv?.toLocaleString(), subtext: 'Active outreach & follow-ups', icon: MessageSquareMore, color: '#3b82f6' }
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
    <div style={{ padding: '40px 40px 40px', fontFamily: '"Inter", sans-serif' }}>
      {/* Header */}
      <div style={{ marginBottom: '36px' }}>
        <h1 style={{ fontSize: '32px', fontWeight: 900, color: '#0f172a', letterSpacing: '-1px', marginBottom: '6px' }}>
          Welcome back!
        </h1>
        <p style={{ fontSize: '15px', color: '#64748b' }}>
          Here's your team's talent pipeline at a glance.
        </p>
      </div>

      {/* Metric Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '20px', marginBottom: '36px' }}>
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
          iconColor="#f97316"
        >
          {/* Tab Switch */}
          <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
            {CHART_MODES.map((mode, i) => (
              <button
                key={mode}
                onClick={() => setActiveMode(i)}
                style={{
                  padding: '6px 14px',
                  borderRadius: '20px',
                  border: 'none',
                  cursor: 'pointer',
                  fontSize: '12px',
                  fontWeight: 700,
                  transition: 'all 0.15s',
                  background: activeMode === i ? '#f97316' : '#f1f5f9',
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
          iconColor={activeMode === 0 ? '#f97316' : activeMode === 1 ? '#6366f1' : activeMode === 2 ? '#0ea5e9' : '#8b5cf6'}
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
            iconColor="#eab308"
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
