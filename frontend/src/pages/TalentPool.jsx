import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useAppStore, API_BASE } from '../store/useAppStore';
import axios from 'axios';
import {
  Search, ExternalLink, ChevronLeft, ChevronRight, Filter,
  User, Building2, MapPin, Briefcase, BarChart2,
  SlidersHorizontal, RefreshCw, UserPlus, X, ChevronDown,
  Activity, MessageSquareMore, Users, Plus, Edit2, Check, Download, 
  Mail, Phone, MessageSquare, Linkedin, Send
} from 'lucide-react';

// ── Status pill config ────────────────────────────────────────
const RECRUITMENT_STAGES = [
  'To be started', 'Shortlisted', 'Rejected', 'For Future', 
  'Reached out - Linkedin', 'Reached out - Phone', 'Not Interested', 
  'Followup / In conversation', 'Shortlist - Rejected', 'High CTC', 
  'Duplicate', 'Not responding', 'Internal Review', 'Shared with customer'
];

const STATUS_STYLES = {
  'to be started':          { bg: '#f1f5f9', color: '#64748b', dot: '#94a3b8' },
  'shortlisted':            { bg: '#dcfce7', color: '#16a34a', dot: '#16a34a' },
  'rejected':               { bg: '#fee2e2', color: '#b91c1c', dot: '#ef4444' },
  'for future':             { bg: '#fff7ed', color: '#9a3412', dot: '#f97316' },
  'reached out - linkedin': { bg: '#e0f2fe', color: '#0369a1', dot: '#0284c7' },
  'reached out - phone':    { bg: '#dbeafe', color: '#1d4ed8', dot: '#1d4ed8' },
  'not interested':         { bg: '#f1f5f9', color: '#475569', dot: '#64748b' },
  'followup / in conversation': { bg: '#fef9c3', color: '#854d0e', dot: '#ca8a04' },
  'shortlist - rejected':   { bg: '#fef2f2', color: '#991b1b', dot: '#dc2626' },
  'high ctc':               { bg: '#fce7f3', color: '#be185d', dot: '#db2777' },
  'duplicate':              { bg: '#f3f4f6', color: '#374151', dot: '#4b5563' },
  'not responding':         { bg: '#fff1f2', color: '#9f1239', dot: '#e11d48' },
  'internal review':        { bg: '#f3e8ff', color: '#7e22ce', dot: '#9333ea' },
  'shared with customer':   { bg: '#ecfdf5', color: '#065f46', dot: '#059669' },
};

function StatusDropdown({ status, candidateId, onUpdate, onShortlisted }) {
  const [isOpen, setIsOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const dropdownRef = useRef(null);

  useEffect(() => {
    if (!isOpen) return undefined;

    function handleClickOutside(event) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isOpen]);

  const handleUpdate = async (newStatus) => {
    if (newStatus === status) {
      setIsOpen(false);
      return;
    }
    setLoading(true);
    try {
      await axios.post(`${API_BASE}/candidates/${candidateId}/status`, { status: newStatus });
      onUpdate(candidateId, newStatus);
      if (newStatus === 'Shortlisted' && onShortlisted) {
        onShortlisted(candidateId);
      }
    } catch (err) {
      console.error("Failed to update status:", err);
      alert("Failed to update status. Please try again.");
    } finally {
      setLoading(false);
      setIsOpen(false);
    }
  };

  const currentStyle = STATUS_STYLES[(status || '').toLowerCase()] || { bg: '#f1f5f9', color: '#475569', dot: '#94a3b8' };

  return (
    <div style={{ position: 'relative' }} ref={dropdownRef}>
      <button 
        onClick={() => !loading && setIsOpen(!isOpen)}
        disabled={loading}
        style={{
          display: 'inline-flex', alignItems: 'center', gap: '6px',
          padding: '4px 10px', borderRadius: '20px',
          fontSize: '11.5px', fontWeight: 700,
          background: currentStyle.bg, color: currentStyle.color,
          border: 'none', cursor: loading ? 'wait' : 'pointer',
          whiteSpace: 'nowrap', transition: 'filter 0.1s',
          opacity: loading ? 0.7 : 1,
        }}
        onMouseEnter={e => e.currentTarget.style.filter = 'brightness(0.95)'}
        onMouseLeave={e => e.currentTarget.style.filter = 'none'}
      >
        <span style={{ width: 6, height: 6, borderRadius: '50%', background: currentStyle.dot }} />
        {loading ? 'Updating...' : (status || 'Select Status')}
        <ChevronDown size={12} style={{ opacity: 0.5 }} />
      </button>

      {isOpen && (
        <div style={{
          position: 'absolute', top: '100%', left: 0, marginTop: '4px',
          background: '#fff', border: '1px solid #e2e8f0', borderRadius: '12px',
          boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
          zIndex: 50, padding: '6px', minWidth: '180px', maxHeight: '300px', overflowY: 'auto'
        }}>
          {RECRUITMENT_STAGES.map(stage => {
            const style = STATUS_STYLES[stage.toLowerCase()] || { dot: '#94a3b8' };
            const isActive = stage === status;
            return (
              <button
                key={stage}
                onClick={() => handleUpdate(stage)}
                style={{
                  width: '100%', display: 'flex', alignItems: 'center', gap: '8px',
                  padding: '8px 10px', borderRadius: '8px',
                  background: isActive ? '#f8fafc' : 'transparent',
                  border: 'none', cursor: 'pointer', textAlign: 'left',
                  fontSize: '12px', color: isActive ? '#0f172a' : '#475569',
                  fontWeight: isActive ? 700 : 500, transition: 'background 0.1s'
                }}
                onMouseEnter={e => e.currentTarget.style.background = '#f8fafc'}
                onMouseLeave={e => !isActive && (e.currentTarget.style.background = 'transparent')}
              >
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: style.dot }} />
                {stage}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

function ShortlistCard({ data, onClose }) {
  const [closing, setClosing] = useState(false);
  
  useEffect(() => {
    const timer = setTimeout(() => { setClosing(true); setTimeout(onClose, 300); }, 10000);
    return () => clearTimeout(timer);
  }, [onClose]);

  const OutreachBadge = ({ status, label }) => {
    const config = {
      started: { bg: '#dcfce7', color: '#15803d', icon: '✓', text: 'Triggered' },
      error:   { bg: '#fee2e2', color: '#b91c1c', icon: '✕', text: 'Failed' },
      not_started: { bg: '#f1f5f9', color: '#64748b', icon: '—', text: 'No data' },
      no_campaign_id: { bg: '#fef9c3', color: '#854d0e', icon: '!', text: 'No campaign' },
    };
    const c = config[status] || config.not_started;
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px' }}>
        <span style={{ fontWeight: 600, color: '#475569', minWidth: '52px' }}>{label}</span>
        <span style={{ padding: '2px 8px', borderRadius: '10px', background: c.bg, color: c.color, fontWeight: 700, fontSize: '11px' }}>
          {c.icon} {c.text}
        </span>
      </div>
    );
  };

  return (
    <div style={{
      position: 'fixed', bottom: '28px', right: '28px', zIndex: 10000,
      background: '#fff', borderRadius: '18px', padding: '20px 24px',
      boxShadow: '0 20px 25px -5px rgba(0,0,0,0.15), 0 10px 10px -5px rgba(0,0,0,0.07)',
      border: '1px solid #e2e8f0', width: '340px',
      transition: 'opacity 0.3s, transform 0.3s',
      opacity: closing ? 0 : 1, transform: closing ? 'translateY(12px)' : 'translateY(0)'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '14px' }}>
        <div>
          <div style={{ fontSize: '13px', fontWeight: 800, color: '#0f172a', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ background: '#dcfce7', color: '#15803d', padding: '2px 8px', borderRadius: '8px', fontSize: '11px' }}>✓ Shortlisted</span>
            {data.name}
          </div>
          <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '3px' }}>Outreach started automatically</div>
        </div>
        <button onClick={() => { setClosing(true); setTimeout(onClose, 300); }}
          style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#94a3b8', padding: '2px', borderRadius: '50%', lineHeight: 1 }}>
          <X size={16} />
        </button>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', padding: '12px', background: '#f8fafc', borderRadius: '10px', marginBottom: '14px' }}>
        {data.email && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12.5px' }}>
            <Mail size={13} color="#f97316" />
            <span style={{ fontWeight: 600, color: '#0f172a' }}>{data.email}</span>
          </div>
        )}
        {data.phone && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12.5px' }}>
            <Phone size={13} color="#64748b" />
            <span style={{ color: '#374151' }}>{data.phone}</span>
          </div>
        )}
        {data.linkedin && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12.5px' }}>
            <Linkedin size={13} color="#0077b5" />
            <a href={data.linkedin} target="_blank" rel="noreferrer"
               style={{ color: '#0077b5', fontWeight: 600, textDecoration: 'none', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              LinkedIn Profile
            </a>
          </div>
        )}
        {!data.email && !data.phone && !data.linkedin && (
          <span style={{ fontSize: '12px', color: '#94a3b8' }}>No contact info found</span>
        )}
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
        <div style={{ fontSize: '11px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: '4px' }}>Outreach Status</div>
        <OutreachBadge status={data.email_outreach} label="Email" />
        <OutreachBadge status={data.linkedin_outreach} label="LinkedIn" />
      </div>
    </div>
  );
}

function EditableNotes({ candidateId, initialNotes }) {
  const [notes, setNotes] = useState(initialNotes || '');
  const [isEditing, setIsEditing] = useState(false);
  const updateCandidateNotes = useAppStore(state => state.updateCandidateNotes);

  useEffect(() => {
    setNotes(initialNotes || '');
  }, [initialNotes]);

  const handleBlur = async () => {
    setIsEditing(false);
    if (notes !== (initialNotes || '')) {
      await updateCandidateNotes(candidateId, notes);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.target.blur();
    }
  };

  return (
    <div style={{ width: '100%', minWidth: 100 }}>
      {isEditing ? (
        <textarea
          autoFocus
          value={notes}
          onChange={e => setNotes(e.target.value)}
          onBlur={handleBlur}
          onKeyDown={handleKeyDown}
          rows={1}
          style={{
            width: '100%',
            padding: '4px 8px',
            border: '1.5px solid #f97316',
            borderRadius: '6px',
            fontSize: '12px',
            outline: 'none',
            fontFamily: 'inherit',
            resize: 'none',
            display: 'block'
          }}
        />
      ) : (
        <div 
          onClick={() => setIsEditing(true)}
          style={{ 
            fontSize: '12.5px', 
            color: notes ? '#334155' : '#94a3b8',
            fontStyle: notes ? 'normal' : 'italic',
            cursor: 'text',
            minHeight: '20px',
            padding: '4px 0',
            maxWidth: '180px',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap'
          }}
        >
          {notes || 'Add notes...'}
        </div>
      )}
    </div>
  );
}

function ExpBar({ value, max = 40 }) {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '7px' }}>
      <span style={{ fontSize: '13px', fontWeight: 600, color: '#0f172a', minWidth: 32 }}>{value}y</span>
      <div style={{ width: 44, height: 5, background: '#e2e8f0', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: '#3b82f6', borderRadius: 3 }} />
      </div>
    </div>
  );
}

function TagFilterInput({ label, values, inputValue, onInputChange, onTagsChange, placeholder, icon: Icon }) {
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && inputValue.trim()) {
      e.preventDefault();
      const val = inputValue.trim();
      if (!values.includes(val)) {
        onTagsChange([...values, val]);
      }
      onInputChange('');
    } else if (e.key === 'Backspace' && !inputValue && values.length > 0) {
      onTagsChange(values.slice(0, -1));
    }
  };

  const removeTag = (tag) => {
    onTagsChange(values.filter(t => t !== tag));
  };

  return (
    <div style={{ marginBottom: 14 }}>
      <label style={{ display: 'block', fontSize: 11, fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 6 }}>
        {label}
      </label>
      <div style={{ position: 'relative', display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <div style={{ position: 'relative' }}>
          {Icon && (
            <span style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: '#94a3b8', display: 'flex' }}>
              <Icon size={13} />
            </span>
          )}
          <input
            type="text"
            value={inputValue}
            onChange={e => onInputChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={values.length === 0 ? placeholder : "Add more..."}
            style={{
              width: '100%', padding: Icon ? '8px 10px 8px 30px' : '8px 10px',
              background: '#fff', border: '1.5px solid #e2e8f0',
              borderRadius: 8, color: '#0f172a', fontSize: 13,
              outline: 'none', transition: 'border-color 0.15s',
              fontFamily: 'inherit', boxSizing: 'border-box',
            }}
            onFocus={e => e.target.style.borderColor = '#f97316'}
            onBlur={e => e.target.style.borderColor = '#e2e8f0'}
          />
        </div>
        {values.length > 0 && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {values.map(tag => (
              <span key={tag} style={{
                display: 'inline-flex', alignItems: 'center', gap: '6px',
                background: '#f1f5f9', color: '#475569', padding: '3px 8px',
                borderRadius: '6px', fontSize: '11px', fontWeight: 600,
                border: '1px solid #e2e8f0'
              }}>
                {tag}
                <X size={12} style={{ cursor: 'pointer', color: '#94a3b8' }} onClick={() => removeTag(tag)} />
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function SelectFilter({ label, value, onChange, options, placeholder }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <label style={{ display: 'block', fontSize: 11, fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 6 }}>
        {label}
      </label>
      <div style={{ position: 'relative' }}>
        <select
          value={value}
          onChange={e => onChange(e.target.value)}
          style={{
            width: '100%', padding: '8px 28px 8px 10px',
            background: '#fff', border: '1.5px solid #e2e8f0',
            borderRadius: 8, color: value ? '#0f172a' : '#94a3b8', fontSize: 13,
            outline: 'none', appearance: 'none', cursor: 'pointer',
            fontFamily: 'inherit', boxSizing: 'border-box',
          }}
          onFocus={e => e.target.style.borderColor = '#f97316'}
          onBlur={e => e.target.style.borderColor = '#e2e8f0'}
        >
          <option value="">{placeholder}</option>
          {options.map(o => <option key={o} value={o}>{o}</option>)}
        </select>
        <ChevronDown size={13} color="#94a3b8" style={{ position: 'absolute', right: 8, top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none' }} />
      </div>
    </div>
  );
}

function RangeSlider({ label, min, max, minValue, maxValue, onChange }) {
  const minPos = ((minValue - min) / (max - min)) * 100;
  const maxPos = ((maxValue - min) / (max - min)) * 100;

  return (
    <div style={{ marginBottom: 18 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <label style={{ fontSize: 11, fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          {label}
        </label>
        <span style={{ fontSize: 11, fontWeight: 700, color: '#94a3b8' }}>
          {minValue} - {maxValue} yrs
        </span>
      </div>
      
      <div style={{ position: 'relative', height: 20, display: 'flex', alignItems: 'center', padding: '0 4px' }}>
        {/* Track Background */}
        <div style={{ position: 'absolute', left: 4, right: 4, height: 4, background: '#e2e8f0', borderRadius: 2 }} />
        
        {/* Active Range Highlight */}
        <div style={{ 
          position: 'absolute', 
          left: `calc(4px + ${minPos}%)`, 
          width: `${maxPos - minPos}%`, 
          height: 4, 
          background: '#3b82f6', 
          borderRadius: 2,
          zIndex: 1
        }} />
        
        {/* Dual Range Inputs Layered */}
        <input 
          type="range" min={min} max={max} value={minValue} 
          onChange={e => {
            const val = Math.min(Number(e.target.value), maxValue - 1);
            onChange(val, maxValue);
          }}
          style={{
            position: 'absolute', width: '100%', left: 0, appearance: 'none', background: 'none',
            pointerEvents: 'none', zIndex: 2, cursor: 'pointer', outline: 'none'
          }}
          className="dual-range-input"
        />
        <input 
          type="range" min={min} max={max} value={maxValue} 
          onChange={e => {
            const val = Math.max(Number(e.target.value), minValue + 1);
            onChange(minValue, val);
          }}
          style={{
            position: 'absolute', width: '100%', left: 0, appearance: 'none', background: 'none',
            pointerEvents: 'none', zIndex: 3, cursor: 'pointer', outline: 'none'
          }}
          className="dual-range-input"
        />
        
        <style dangerouslySetInnerHTML={{ __html: `
          .dual-range-input::-webkit-slider-thumb {
            appearance: none;
            pointer-events: auto;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #2563eb;
            border: 2px solid #fff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.15);
            cursor: pointer;
            z-index: 10;
          }
          .dual-range-input::-moz-range-thumb {
            appearance: none;
            pointer-events: auto;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #2563eb;
            border: 2px solid #fff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.15);
            cursor: pointer;
            z-index: 10;
          }
        `}} />
      </div>
    </div>
  );
}

const DEFAULT_PAGE_SIZE = 25;
const PAGE_SIZE_OPTIONS = [10, 25, 50, 100];
const CONTACT_INFO_STORAGE_KEY = 'talent-pool-contact-info-v1';

const readPersistedContactInfo = () => {
  if (typeof window === 'undefined') return {};
  try {
    const raw = window.sessionStorage.getItem(CONTACT_INFO_STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
};

function StatisticsDashboard({ analytics, role, onStatClick, onRecruiterClick }) {
  if (!analytics) return null;
  const { summary } = analytics;
  
  const conversionRate = summary.total_sourced > 0 
    ? Math.round((summary.shortlisted / summary.total_sourced) * 100) 
    : 0;

  const cards = [
    { label: 'Total Sourced', value: summary.total_sourced, icon: UserPlus, color: '#f97316' },
    { label: 'Pipeline Success', value: summary.shortlisted, icon: Activity, color: '#22c55e', status: 'Shortlisted' },
    { label: 'Conversion', value: `${conversionRate}%`, icon: BarChart2, color: '#3b82f6' },
    { label: 'Following Up', value: summary.pipeline_health['Followup / In conversation'] || 0, icon: MessageSquareMore, color: '#eab308', status: 'Followup / In conversation' },
  ];

  const recruiterPerf = analytics.recruiter_performance || [];

  return (
    <div style={{ padding: '20px', background: '#fff', borderBottom: '1.5px solid #f1f5f9' }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginBottom: role === 'admin' && recruiterPerf.length > 0 ? '24px' : '0' }}>
        {cards.map((card, i) => {
          const Icon = card.icon;
          return (
            <div 
              key={i}
              onClick={() => card.status && onStatClick(card.status)}
              style={{ 
                background: '#fff', padding: '16px', borderRadius: '16px', border: '1.5px solid #f1f5f9',
                cursor: card.status ? 'pointer' : 'default', transition: 'all 0.2s',
                display: 'flex', alignItems: 'center', gap: '16px',
                boxShadow: '0 1px 3px rgba(0,0,0,0.02)'
              }}
              onMouseEnter={e => card.status && (e.currentTarget.style.transform = 'translateY(-2px)', e.currentTarget.style.boxShadow = '0 10px 15px -3px rgba(0,0,0,0.05)')}
              onMouseLeave={e => card.status && (e.currentTarget.style.transform = 'none', e.currentTarget.style.boxShadow = '0 1px 3px rgba(0,0,0,0.02)')}
            >
              <div style={{ width: 44, height: 44, borderRadius: '12px', background: `${card.color}12`, display: 'flex', alignItems: 'center', justifyContent: 'center', color: card.color }}>
                <Icon size={20} />
              </div>
              <div>
                 <div style={{ fontSize: '11px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.02em', marginTop: '4px' }}>{card.label}</div>
              </div>
            </div>
          );
        })}
      </div>

      {role === 'admin' && recruiterPerf.length > 0 && (
        <div style={{ background: '#f8fafc', borderRadius: '16px', padding: '16px', border: '1px solid #e2e8f0' }}>
          <div style={{ fontSize: '12px', fontWeight: 800, color: '#0f172a', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Users size={14} color="#f97316" /> Recruiter Leaderboard
          </div>
          <div style={{ display: 'flex', gap: '12px', overflowX: 'auto', paddingBottom: '4px' }}>
            {recruiterPerf.map((perf, i) => (
              <div 
                key={i} 
                onClick={() => onRecruiterClick(perf.recruiter)}
                style={{ 
                  minWidth: '160px', background: '#fff', padding: '10px 14px', borderRadius: '12px', border: '1px solid #e2e8f0', 
                  display: 'flex', flexDirection: 'column', gap: '4px', cursor: 'pointer', transition: 'all 0.15s'
                }}
                onMouseEnter={e => e.currentTarget.style.borderColor = '#f97316'}
                onMouseLeave={e => e.currentTarget.style.borderColor = '#e2e8f0'}
              >
                <div style={{ fontSize: '13px', fontWeight: 700, color: '#0f172a', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{perf.recruiter}</div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '11px', color: '#64748b' }}>{perf.sourced} sourced</span>
                  <span style={{ fontSize: '11px', fontWeight: 700, color: '#22c55e' }}>{perf.conversion}% hit</span>
                </div>
                <div style={{ height: '4px', width: '100%', background: '#f1f5f9', borderRadius: '2px', overflow: 'hidden', marginTop: '4px' }}>
                  <div style={{ height: '100%', width: `${perf.conversion}%`, background: '#22c55e' }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ConversationModal({ candidate, onClose }) {
  const [platform, setPlatform] = useState(candidate.linkedin ? 'linkedin' : 'email');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [replyText, setReplyText] = useState('');
  const [sending, setSending] = useState(false);
  
  const fetchChatHistory = useAppStore(state => state.fetchChatHistory);
  const sendChatReply = useAppStore(state => state.sendChatReply);
  const messagesEndRef = useRef(null);

  const loadMessages = async () => {
    setLoading(true);
    const res = await fetchChatHistory(0, candidate.id, platform);
    if (res.success) {
        setMessages(res.messages || []);
    } else {
        setMessages([]);
    }
    setLoading(false);
  };

  useEffect(() => {
    loadMessages();
  }, [candidate.id, platform]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!replyText.trim()) return;
    setSending(true);
    const res = await sendChatReply(0, candidate.id, replyText, platform);
    if (res.success) {
      setReplyText('');
      await loadMessages();
    } else {
      alert(res.error || 'Failed to send reply');
    }
    setSending(false);
  };

  const formatTime = (timeStr) => {
    if (!timeStr) return '';
    const d = new Date(timeStr);
    return isNaN(d.getTime()) ? timeStr : d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) + ' ' + d.toLocaleDateString();
  };

  return (
    <div style={{
      position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
      background: 'rgba(15, 23, 42, 0.4)', backdropFilter: 'blur(4px)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      zIndex: 9999, padding: '20px'
    }} onClick={onClose}>
      <div 
        onClick={e => e.stopPropagation()}
        style={{
          background: '#fff', width: '100%', maxWidth: '600px', height: '80vh',
          borderRadius: '20px', display: 'flex', flexDirection: 'column',
          boxShadow: '0 20px 25px -5px rgba(0,0,0,0.1), 0 8px 10px -6px rgba(0,0,0,0.1)',
          overflow: 'hidden'
        }}
      >
        <div style={{ padding: '20px', borderBottom: '1px solid #e2e8f0', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: '#f8fafc' }}>
          <div>
            <div style={{ fontSize: '18px', fontWeight: 800, color: '#0f172a' }}>{candidate.first_name} {candidate.last_name || ''}</div>
            <div style={{ fontSize: '13px', color: '#64748b', display: 'flex', alignItems: 'center', gap: '8px', marginTop: '4px' }}>
               <MessageSquare size={14}/> Conversations
            </div>
          </div>
          <button onClick={onClose} style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '6px', borderRadius: '50%', color: '#64748b' }} onMouseEnter={e => e.currentTarget.style.background = '#e2e8f0'} onMouseLeave={e => e.currentTarget.style.background = 'none'}>
            <X size={20} />
          </button>
        </div>

        <div style={{ display: 'flex', borderBottom: '1px solid #e2e8f0', padding: '0 20px', background: '#fff' }}>
          <button 
            onClick={() => setPlatform('linkedin')}
            style={{
              padding: '14px 20px', background: 'none', border: 'none', borderBottom: platform === 'linkedin' ? '2px solid #0077b5' : '2px solid transparent',
              color: platform === 'linkedin' ? '#0077b5' : '#64748b', fontWeight: platform === 'linkedin' ? 700 : 500,
              fontSize: '13.5px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px', transition: 'all 0.2s'
            }}
          >
            <Linkedin size={16} /> LinkedIn
          </button>
          <button 
            onClick={() => setPlatform('email')}
            style={{
              padding: '14px 20px', background: 'none', border: 'none', borderBottom: platform === 'email' ? '2px solid #f97316' : '2px solid transparent',
              color: platform === 'email' ? '#f97316' : '#64748b', fontWeight: platform === 'email' ? 700 : 500,
              fontSize: '13.5px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px', transition: 'all 0.2s'
            }}
          >
            <Mail size={16} /> Email
          </button>
        </div>

        <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: '#f8fafc' }}>
                <div style={{ flex: 1, padding: '20px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  {loading ? (
                    <div style={{ textAlign: 'center', padding: '40px', color: '#94a3b8', fontSize: '14px' }}>Loading conversations...</div>
                  ) : messages.length === 0 ? (
                    <div style={{ textAlign: 'center', padding: '40px', color: '#94a3b8', fontSize: '14px' }}>No messages found on {platform === 'linkedin' ? 'LinkedIn' : 'Email'}.</div>
                  ) : (
                    messages.map((msg, i) => {
                      const isCandidate = msg.is_reply || msg.type === 'INBOX' || msg.direction === 'inbound';
                      const senderName = isCandidate ? `${candidate.first_name}` : 'You';
                      const time = msg.time || msg.created_at || msg.timestamp;
                      const body = msg.email_body || msg.message || msg.text || '';
                      
                      return (
                        <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: isCandidate ? 'flex-start' : 'flex-end' }}>
                            <div style={{ fontSize: '11px', color: '#94a3b8', marginBottom: '4px', padding: '0 4px', fontWeight: 600 }}>
                                {senderName} &bull; {formatTime(time)}
                            </div>
                            <div style={{
                                maxWidth: '85%',
                                padding: '12px 16px',
                                borderRadius: '16px',
                                background: isCandidate ? '#fff' : (platform === 'linkedin' ? '#0077b5' : '#f97316'),
                                color: isCandidate ? '#0f172a' : '#fff',
                                fontSize: '13.5px',
                                lineHeight: '1.5',
                                boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
                                border: isCandidate ? '1px solid #e2e8f0' : 'none',
                                borderBottomLeftRadius: isCandidate ? '4px' : '16px',
                                borderBottomRightRadius: isCandidate ? '16px' : '4px',
                                whiteSpace: 'pre-wrap'
                            }}>
                                <div dangerouslySetInnerHTML={{ __html: body }} style={{ maxWidth: '100%', overflowWrap: 'break-word', wordWrap: 'break-word' }} />
                            </div>
                        </div>
                      );
                    })
                  )}
                  <div ref={messagesEndRef} />
                </div>
                
                <div style={{ padding: '16px 20px', background: '#fff', borderTop: '1px solid #e2e8f0' }}>
                    <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-end', background: '#f8fafc', padding: '8px 12px', borderRadius: '16px', border: '1px solid #e2e8f0' }}>
                        <textarea
                            value={replyText}
                            onChange={(e) => setReplyText(e.target.value)}
                            placeholder={`Reply via ${platform === 'linkedin' ? 'LinkedIn' : 'Email'}...`}
                            rows={1}
                            style={{
                                flex: 1, border: 'none', background: 'transparent', outline: 'none',
                                resize: 'none', fontSize: '14px', color: '#0f172a', padding: '8px 4px',
                                minHeight: '38px', maxHeight: '120px', fontFamily: 'inherit'
                            }}
                            onInput={(e) => {
                                e.target.style.height = 'auto';
                                e.target.style.height = (e.target.scrollHeight) + 'px';
                            }}
                            onKeyDown={(e) => {
                                if (e.key === 'Enter' && !e.shiftKey) {
                                    e.preventDefault();
                                    handleSend();
                                }
                            }}
                        />
                        <button
                            onClick={handleSend}
                            disabled={sending || !replyText.trim()}
                            style={{
                                width: '38px', height: '38px', borderRadius: '50%',
                                background: sending || !replyText.trim() ? '#e2e8f0' : (platform === 'linkedin' ? '#0077b5' : '#f97316'),
                                color: sending || !replyText.trim() ? '#94a3b8' : '#fff',
                                border: 'none', display: 'flex', alignItems: 'center', justifyContent: 'center',
                                cursor: sending || !replyText.trim() ? 'not-allowed' : 'pointer',
                                transition: 'all 0.2s', padding: 0, flexShrink: 0, marginBottom: '2px'
                            }}
                        >
                            <Send size={18} style={{ marginLeft: '2px' }} />
                        </button>
                    </div>
                </div>
            </div>
        </div>
      </div>
    </div>
  );
}

export default function TalentPool() {
  const { user } = useAppStore();
  const role = user?.role || 'recruiter';
  const permissions = user?.permissions || {};

  if (role !== 'admin' && !permissions['talent_pool']) {
    return (
      <div style={{ padding: '80px 20px', textAlign: 'center', background: '#f8fafc', minHeight: '80vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ padding: '40px', background: '#fff', borderRadius: '24px', boxShadow: '0 20px 25px -5px rgba(0,0,0,0.05)', maxWidth: '440px', border: '1px solid #e2e8f0' }}>
          <div style={{ width: 64, height: 64, borderRadius: '20px', background: '#fff7ed', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 24px', border: '1.5px solid #fed7aa' }}>
            <SlidersHorizontal size={32} color="#f97316" />
          </div>
          <h2 style={{ fontSize: '24px', fontWeight: 800, color: '#0f172a', marginBottom: '12px', letterSpacing: '-0.02em' }}>Access Restricted</h2>
          <p style={{ color: '#64748b', lineHeight: 1.6, marginBottom: '32px', fontSize: '15px' }}>
            The <b>Talent Pool</b> feature has not been enabled for your account yet. Please contact your administrator to enable this tool.
          </p>
          <button 
            onClick={() => window.location.href = '/'}
            style={{ 
              width: '100%', padding: '14px 24px', background: '#f97316', color: '#fff', 
              border: 'none', borderRadius: '14px', fontWeight: 700, cursor: 'pointer',
              fontSize: '15px', transition: 'all 0.2s', boxShadow: '0 4px 6px -1px rgba(249, 115, 22, 0.2)'
            }}
            onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-1px)'}
            onMouseLeave={e => e.currentTarget.style.transform = 'none'}
          >
            Return to Dashboard
          </button>
        </div>
      </div>
    );
  }

  const [candidates, setCandidates] = useState([]);
  const [total, setTotal] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE);
  const [loading, setLoading] = useState(false);
  const [statusCounts, setStatusCounts] = useState({});
  const [meta, setMeta] = useState({ companies: [], cities: [], products: [], statuses: [] });
  const [globalSearch, setGlobalSearch] = useState('');
  const [filters, setFilters] = useState({
    title: [], titleInput: '',
    company: [], companyInput: '',
    city: [], cityInput: '',
    product_service: [], productInput: '',
    status: '', created_by: '',
    min_exp: 0, max_exp: 40,
  });
  const [stability, setStability] = useState(false);
  const [activeStatusTab, setActiveStatusTab] = useState('');
  const [sortBy, setSortBy] = useState('name');
  const [sortDir, setSortDir] = useState('asc');
  const [isSemanticSearch, setIsSemanticSearch] = useState(false);
  const [selectedCandidateForChat, setSelectedCandidateForChat] = useState(null);
  const [shortlistCard, setShortlistCard] = useState(null);
  const [shortlistingId, setShortlistingId] = useState(null);
  const [contactInfo, setContactInfo] = useState(readPersistedContactInfo); // { [candidateId]: { email, phone, enriching } }
  const analytics = useAppStore(state => state.analytics);
  const fetchAnalytics = useAppStore(state => state.fetchAnalytics);
  const shortlistAndOutreach = useAppStore(state => state.shortlistAndOutreach);
  const didInitRef = useRef(false);

  // Poll Clay/DB updates aggressively for enriching records so contact data appears quickly.
  useEffect(() => {
    const enrichingIds = Object.keys(contactInfo).filter(id => contactInfo[id]?.enriching);
    if (enrichingIds.length === 0) return undefined;

    let cancelled = false;
    let timer = null;

    const pollOnce = async () => {
      await Promise.all(
        enrichingIds.map(async (id) => {
          try {
            const res = await axios.get(`${API_BASE}/candidates/${id}`);
            const data = res?.data;
            if (!data) return;

            const email = data.email || '';
            const phone = data.phone || data.mobile_phone || '';
            const done = Boolean(email || phone || data.enrichment_finished);
            if (!done) return;

            setContactInfo(prev => {
              const current = prev[id] || {};
              const next = {
                email: email || current.email || '',
                phone: phone || current.phone || '',
                enriching: false
              };
              if (
                current.email === next.email &&
                current.phone === next.phone &&
                Boolean(current.enriching) === next.enriching
              ) {
                return prev;
              }
              return { ...prev, [id]: next };
            });

            // If this is the currently viewed ShortlistCard, update it too
            setShortlistCard(prev => prev && prev.candidate_id == id
              ? { ...prev, email: email || prev.email || '', phone: phone || prev.phone || '' }
              : prev
            );
          } catch {
            // Keep polling silently.
          }
        })
      );
    };

    const pollLoop = async () => {
      if (cancelled) return;
      await pollOnce();
      if (!cancelled) {
        timer = setTimeout(pollLoop, 1000);
      }
    };

    pollLoop();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [contactInfo]);

  useEffect(() => {
    try {
      window.sessionStorage.setItem(CONTACT_INFO_STORAGE_KEY, JSON.stringify(contactInfo));
    } catch {
      // Ignore storage write errors (private mode, quota, etc.)
    }
  }, [contactInfo]);

  const handleShortlisted = async (candidateId) => {
    setShortlistingId(candidateId);
    // Preserve any existing contact details immediately to avoid flicker while enrichment runs.
    setContactInfo(prev => {
      const current = prev[candidateId] || {};
      const row = candidates.find(c => c.id === candidateId) || {};
      const next = {
        email: current.email || row.email || '',
        phone: current.phone || row.phone || '',
        enriching: !(current.email || current.phone || row.email || row.phone)
      };
      return { ...prev, [candidateId]: next };
    });

    const res = await shortlistAndOutreach(candidateId);
    setShortlistingId(null);
    if (res.success) {
      const d = res.data;
      setContactInfo(prev => ({
        ...prev,
        [candidateId]: {
          email: d.email || prev[candidateId]?.email || '',
          phone: d.phone || prev[candidateId]?.phone || '',
          // If Clay was triggered (no data yet), keep enriching=true
          enriching: d.contact_enriching && !(d.email || prev[candidateId]?.email) && !(d.phone || prev[candidateId]?.phone)
        }
      }));
      setShortlistCard(d);
    } else {
      setContactInfo(prev => ({
        ...prev,
        [candidateId]: {
          email: prev[candidateId]?.email || '',
          phone: prev[candidateId]?.phone || '',
          enriching: false
        }
      }));
      setShortlistCard({ name: 'Candidate', email: '', phone: '', linkedin: '', email_outreach: 'error', linkedin_outreach: 'error' });
    }
  };
  const debounceRef = useRef(null);

  const mergeContactInfoFromRows = useCallback((rows = []) => {
    if (!Array.isArray(rows) || rows.length === 0) return;
    setContactInfo(prev => {
      let changed = false;
      const next = { ...prev };

      for (const row of rows) {
        if (!row?.id) continue;
        const existing = prev[row.id] || {};
        const email = row.email || existing.email || '';
        const phone = row.phone || existing.phone || '';
        const enriching = Boolean(existing.enriching && !(email || phone));

        if (!email && !phone && !existing.enriching) continue;

        if (
          existing.email !== email ||
          existing.phone !== phone ||
          Boolean(existing.enriching) !== enriching
        ) {
          next[row.id] = { email, phone, enriching };
          changed = true;
        }
      }

      return changed ? next : prev;
    });
  }, []);

  // Load metadata (dropdown options)
  useEffect(() => {
    axios.get(`${API_BASE}/candidates/browse/meta`).then(r => setMeta(r.data)).catch(() => {});
    fetchAnalytics();
  }, [fetchAnalytics]);

  const fetchCandidates = useCallback(async (pg = 1) => {
    try {
      const params = new URLSearchParams();
      params.set('page', pg);
      params.set('page_size', pageSize);
      if (globalSearch) params.set('q', globalSearch);
      
      const titleSearch = [...(filters.title || []), filters.titleInput].filter(Boolean).join(',');
      if (titleSearch) params.set('title', titleSearch);
      
      const companySearch = [...(filters.company || []), filters.companyInput].filter(Boolean).join(',');
      if (companySearch) params.set('company', companySearch);
      
      const citySearch = [...(filters.city || []), filters.cityInput].filter(Boolean).join(',');
      if (citySearch) params.set('city', citySearch);
      
      const productSearch = [...(filters.product_service || []), filters.productInput].filter(Boolean).join(',');
      if (productSearch) params.set('product_service', productSearch);

      if (activeStatusTab) params.set('status', activeStatusTab);
      else if (filters.status) params.set('status', filters.status);
      if (filters.min_exp !== undefined && filters.min_exp !== '') params.set('min_exp', filters.min_exp);
      if (filters.max_exp !== undefined && filters.max_exp !== '') params.set('max_exp', filters.max_exp);
      if (filters.created_by) params.set('created_by', filters.created_by);
      if (stability) params.set('min_avg_tenure', '2');
      params.set('sort_by', sortBy);
      params.set('sort_dir', sortDir);

      const paramsString = params.toString();
      const currentCache = useAppStore.getState().talentPoolCache;
      
      const isInstantHit = currentCache?.lastParamsString === paramsString && currentCache?.data;
      
      if (!isInstantHit) {
        setLoading(true);
      } else {
        // Optimistically render from cache instantly
        const d = currentCache.data;
        setCandidates(d.candidates);
        setTotal(d.total);
        setTotalPages(d.total_pages);
        setStatusCounts(d.status_counts || {});
        setIsSemanticSearch(d.is_semantic_search || false);
        mergeContactInfoFromRows(d.candidates);
      }

      const res = await useAppStore.getState().fetchTalentPool(paramsString);
      
      if (res.success && res.data) {
        setCandidates(res.data.candidates);
        setTotal(res.data.total);
        setTotalPages(res.data.total_pages);
        setStatusCounts(res.data.status_counts || {});
        setIsSemanticSearch(res.data.is_semantic_search || false);
        mergeContactInfoFromRows(res.data.candidates);
      }
    } catch (e) {
      console.error('Failed to fetch talent pool:', e);
    } finally {
      setLoading(false);
    }
  }, [globalSearch, filters, activeStatusTab, stability, sortBy, sortDir, pageSize, mergeContactInfoFromRows]);

  // Debounced refetch
  useEffect(() => {
    if (!didInitRef.current) {
      didInitRef.current = true;
      return;
    }
    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      if (page !== 1) {
        setPage(1);
        return;
      }
      fetchCandidates(1);
    }, 350);
    return () => clearTimeout(debounceRef.current);
  }, [fetchCandidates]);

  useEffect(() => { fetchCandidates(page); }, [page]);

  const setFilter = (key, val) => setFilters(f => ({ ...f, [key]: val }));

  const clearFilters = () => {
    setFilters({ 
      title: [], titleInput: '',
      company: [], companyInput: '',
      city: [], cityInput: '',
      product_service: [], productInput: '',
      status: '', created_by: '',
      min_exp: 0, max_exp: 40 
    });
    setGlobalSearch('');
    setStability(false);
    setActiveStatusTab('');
    setPage(1);
  };

  const hasFilters = globalSearch || 
    filters.title || filters.company || filters.city || filters.product_service || filters.status ||
    filters.min_exp !== 0 || filters.max_exp !== 40 ||
    stability || activeStatusTab;

  // Status tabs: All + each status
  const statusTabs = ['', ...Object.keys(statusCounts)];

  // Columns
  const cols = [
    { key: 'first_name', label: 'First Name', w: 100 },
    { key: 'last_name', label: 'Last Name', w: 100 },
    { key: 'title', label: 'Title', w: 160, sortKey: 'title' },
    { key: 'linkedin', label: 'LinkedIn', w: 70 },
    { key: 'company', label: 'Current Company', w: 160, sortKey: 'company' },
    { key: 'product_service', label: 'Product/Service', w: 140 },
    { key: 'city', label: 'City', w: 120, sortKey: 'city' },
    { key: 'location_type', label: 'Location', w: 100 },
    { key: 'total_experience_years', label: 'Total Exp', w: 110, sortKey: 'exp' },
    { key: 'avg_tenure_years', label: 'Avg. Yrs', w: 100, sortKey: 'tenure' },
    { key: 'email', label: 'Email', w: 180 },
    { key: 'phone', label: 'Phone', w: 150 },
    { key: 'status', label: 'Status', w: 140 },
    { key: 'response', label: 'Response', w: 140 },
    { key: 'notes', label: 'Notes', w: 200 },
    { key: 'match_score', label: 'Match Score', w: 110, hidden: !isSemanticSearch },
  ];

  const handleSort = (sortKey) => {
    if (!sortKey) return;
    if (sortBy === sortKey) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortBy(sortKey); setSortDir('asc'); }
  };

  return (
    <div style={{ fontFamily: '"Inter", -apple-system, sans-serif', display: 'flex', gap: 0, height: 'calc(100vh - 80px)', overflow: 'hidden' }}>

      {/* ── Left Filter Sidebar ── */}
      <aside style={{
        width: 200, minWidth: 200, padding: '20px 16px',
        background: '#fff', borderRight: '1.5px solid #f1f5f9',
        overflowY: 'auto', flexShrink: 0,
        display: 'flex', flexDirection: 'column', gap: 0,
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 18 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 7, fontWeight: 800, fontSize: 14, color: '#0f172a' }}>
            <SlidersHorizontal size={15} color="#f97316" /> Filters
          </div>
          {hasFilters && (
            <button onClick={clearFilters} style={{ background: 'none', border: 'none', color: '#ef4444', fontSize: 11, fontWeight: 700, cursor: 'pointer', padding: 0, display: 'flex', alignItems: 'center', gap: 3 }}>
              <X size={11} /> Clear
            </button>
          )}
        </div>

        <TagFilterInput label="Title" values={filters.title} inputValue={filters.titleInput} onInputChange={v => setFilter('titleInput', v)} onTagsChange={v => setFilter('title', v)} placeholder="e.g. Engineer" icon={Briefcase} />
        <TagFilterInput label="Current Company" values={filters.company} inputValue={filters.companyInput} onInputChange={v => setFilter('companyInput', v)} onTagsChange={v => setFilter('company', v)} placeholder="e.g. Google" icon={Building2} />
        <TagFilterInput label="City" values={filters.city} inputValue={filters.cityInput} onInputChange={v => setFilter('cityInput', v)} onTagsChange={v => setFilter('city', v)} placeholder="e.g. San Francisco" icon={MapPin} />

        <TagFilterInput label="Expertise / Product" values={filters.product_service} inputValue={filters.productInput} onInputChange={v => setFilter('productInput', v)} onTagsChange={v => setFilter('product_service', v)} placeholder="e.g. SaaS, Fintech" icon={BarChart2} />
        
        {role === 'admin' && (
          <SelectFilter 
            label="Recruiter" 
            value={filters.created_by || ''} 
            onChange={v => setFilter('created_by', v)} 
            options={meta.recruiters || []} 
            placeholder="All Recruiters" 
          />
        )}

        <SelectFilter label="Status" value={filters.status} onChange={v => setFilter('status', v)} options={meta.statuses} placeholder="All Statuses" />

        <RangeSlider 
          label="Total Experience" 
          min={0} 
          max={40} 
          minValue={filters.min_exp} 
          maxValue={filters.max_exp} 
          onChange={(min, max) => {
            setFilters(prev => ({ ...prev, min_exp: min, max_exp: max }));
          }} 
        />

        {/* Stability */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
          <input type="checkbox" id="stability" checked={stability} onChange={e => setStability(e.target.checked)}
            style={{ width: 14, height: 14, accentColor: '#f97316', cursor: 'pointer' }} />
          <label htmlFor="stability" style={{ fontSize: 12, color: '#475569', cursor: 'pointer', fontWeight: 500 }}>
            &gt; 2 years average
          </label>
        </div>
      </aside>

      {/* ── Main Content ── */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', background: '#f8fafc' }}>
        
        <StatisticsDashboard 
          analytics={analytics} 
          role={role} 
          onStatClick={(status) => {
            setActiveStatusTab(status);
            setPage(1);
          }} 
          onRecruiterClick={(email) => {
            setFilter('created_by', email);
            setPage(1);
          }}
        />

        {/* Top bar */}
        <div style={{ padding: '14px 20px', background: '#fff', borderBottom: '1.5px solid #f1f5f9', display: 'flex', alignItems: 'center', gap: 12 }}>
          {/* Global search */}
          <div style={{ position: 'relative', flex: 1, maxWidth: 380 }}>
            <Search size={15} color="#94a3b8" style={{ position: 'absolute', left: 11, top: '50%', transform: 'translateY(-50%)' }} />
            <input
              type="text"
              value={globalSearch}
              onChange={e => setGlobalSearch(e.target.value)}
              placeholder="Global search across all columns..."
              style={{
                width: '100%', padding: '9px 12px 9px 34px',
                background: '#f8fafc', border: '1.5px solid #e2e8f0',
                borderRadius: 10, color: '#0f172a', fontSize: 13,
                outline: 'none', fontFamily: 'inherit', boxSizing: 'border-box',
                transition: 'border-color 0.15s',
              }}
              onFocus={e => { e.target.style.borderColor = '#f97316'; e.target.style.background = '#fff'; }}
              onBlur={e => { e.target.style.borderColor = '#e2e8f0'; e.target.style.background = '#f8fafc'; }}
            />
          </div>

          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 10 }}>
            {/* Total count */}
            <span style={{ fontSize: 13, color: '#94a3b8', fontWeight: 500 }}>
              {loading ? '...' : `${total.toLocaleString()} candidates`}
            </span>
            <button
              style={{
                display: 'flex', alignItems: 'center', gap: 7,
                padding: '8px 14px', background: '#f97316', border: 'none',
                borderRadius: 9, color: '#fff', fontSize: 13, fontWeight: 700,
                cursor: 'pointer', transition: 'all 0.15s'
              }}
              onMouseEnter={e => e.target.style.background = '#ea580c'}
              onMouseLeave={e => e.target.style.background = '#f97316'}
            >
              <UserPlus size={15} /> Add Candidate
            </button>
            <button onClick={() => fetchCandidates(page)}
              style={{ width: 36, height: 36, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#f8fafc', border: '1.5px solid #e2e8f0', borderRadius: 8, cursor: 'pointer', color: '#64748b' }}>
              <RefreshCw size={14} style={{ animation: loading ? 'spin 1s linear infinite' : 'none' }} />
            </button>
          </div>
        </div>

        {/* Status tabs */}
        <div style={{ padding: '12px 20px', background: '#fff', borderBottom: '1.5px solid #f1f5f9', display: 'flex', gap: 10, overflowX: 'auto', scrollbarWidth: 'none' }}>
          {['', ...RECRUITMENT_STAGES].map(tab => {
            const isActive = activeStatusTab === tab;
            const count = tab === '' ? total : (statusCounts[tab] || 0);
            const style = tab ? (STATUS_STYLES[tab.toLowerCase()] || {}) : { bg: '#f1f5f9', color: '#475569', dot: '#94a3b8' };
            
            return (
              <button key={tab || 'all'}
                onClick={() => { setActiveStatusTab(tab); setPage(1); }}
                style={{
                  padding: '6px 14px', borderRadius: '30px', border: isActive ? '1.5px solid #f97316' : '1.5px solid #e2e8f0',
                  background: isActive ? '#fff7ed' : '#f8fafc', cursor: 'pointer', fontSize: 12, fontWeight: 700,
                  color: isActive ? '#f97316' : '#64748b', whiteSpace: 'nowrap',
                  display: 'flex', alignItems: 'center', gap: 8, fontFamily: 'inherit',
                  transition: 'all 0.15s',
                  boxShadow: isActive ? '0 2px 4px rgba(249, 115, 22, 0.1)' : 'none',
                }}
                onMouseEnter={e => {
                  if (!isActive) {
                    e.currentTarget.style.borderColor = '#cbd5e1';
                    e.currentTarget.style.background = '#f1f5f9';
                  }
                }}
                onMouseLeave={e => {
                  if (!isActive) {
                    e.currentTarget.style.borderColor = '#e2e8f0';
                    e.currentTarget.style.background = '#f8fafc';
                  }
                }}
              >
                {tab === '' ? (
                  <span style={{ color: isActive ? '#f97316' : '#0f172a' }}>All ({total})</span>
                ) : (
                  <>
                    <span style={{ width: 6, height: 6, borderRadius: '50%', background: style.dot || '#94a3b8' }} />
                    {tab}
                    <span style={{ 
                      marginLeft: 4, padding: '1px 6px', borderRadius: 10, fontSize: 10,
                      background: isActive ? '#f97316' : '#e2e8f0', 
                      color: isActive ? '#fff' : '#64748b' 
                    }}>
                      {count}
                    </span>
                  </>
                )}
              </button>
            );
          })}
        </div>

        {/* Table */}
        <div style={{ flex: 1, overflowY: 'auto', overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', minWidth: 2000 }}>
            <thead>
              <tr style={{ background: '#fff', borderBottom: '1.5px solid #f1f5f9', position: 'sticky', top: 0, zIndex: 10 }}>
                {cols.filter(c => !c.hidden).map((col, index) => (
                  <th key={col.key}
                    onClick={() => handleSort(col.sortKey)}
                    style={{
                      padding: '11px 14px', textAlign: 'left',
                      fontSize: 11, fontWeight: 700, color: '#94a3b8',
                      textTransform: 'uppercase', letterSpacing: '0.06em',
                      minWidth: col.w, cursor: col.sortKey ? 'pointer' : 'default',
                      whiteSpace: 'nowrap', userSelect: 'none',
                      borderBottom: '1.5px solid #f1f5f9',
                      borderRight: '1.5px solid #f1f5f9',
                      borderLeft: index === 0 ? '1.5px solid #f1f5f9' : 'none'
                    }}
                  >
                    {col.label}
                    {col.sortKey && sortBy === col.sortKey && (
                      <span style={{ marginLeft: 4 }}>{sortDir === 'asc' ? '↑' : '↓'}</span>
                    )}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {loading && candidates.length === 0 && (
                Array.from({ length: 10 }).map((_, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f8fafc' }}>
                    {cols.filter(c => !c.hidden).map((c, j) => (
                      <td key={j} style={{ padding: '13px 14px' }}>
                        <div style={{ height: 13, borderRadius: 6, background: 'linear-gradient(90deg,#f1f5f9 25%,#e2e8f0 50%,#f1f5f9 75%)', backgroundSize: '200% 100%', animation: 'shimmer 1.2s infinite', width: c.w * 0.6 }} />
                      </td>
                    ))}
                  </tr>
                ))
              )}

              {!loading && candidates.length === 0 && (
                <tr>
                  <td colSpan={cols.filter(c => !c.hidden).length} style={{ padding: '80px 20px', textAlign: 'center' }}>
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}>
                      <div style={{ width: 56, height: 56, background: '#f8fafc', border: '1.5px solid #e2e8f0', borderRadius: 16, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <User size={24} color="#cbd5e1" />
                      </div>
                      <p style={{ color: '#64748b', fontWeight: 600, fontSize: 15, margin: 0 }}>No candidates found</p>
                      <p style={{ color: '#cbd5e1', fontSize: 13, margin: 0 }}>Try adjusting your filters or search query</p>
                      {hasFilters && <button onClick={clearFilters} style={{ padding: '8px 16px', background: '#fff7ed', border: '1.5px solid #fed7aa', borderRadius: 8, color: '#f97316', fontWeight: 700, fontSize: 13, cursor: 'pointer', fontFamily: 'inherit' }}>Clear Filters</button>}
                    </div>
                  </td>
                </tr>
              )}

              {candidates.map((c, idx) => (
                <tr key={c.id || idx}
                  style={{ borderBottom: '1px solid #f8fafc', transition: 'background 0.1s', cursor: 'default' }}
                  onMouseEnter={e => e.currentTarget.style.background = '#fafafa'}
                  onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                >
                  <td style={{ padding: '13px 14px', fontSize: 13, fontWeight: 600, color: '#0f172a', borderRight: '1px solid #f1f5f9', borderLeft: '1px solid #f1f5f9' }}>{c.first_name || ''}</td>
                  <td style={{ padding: '13px 14px', fontSize: 13, color: '#374151', borderRight: '1px solid #f1f5f9' }}>{c.last_name || ''}</td>
                  <td style={{ padding: '13px 14px', fontSize: 13, color: '#374151', maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', borderRight: '1px solid #f1f5f9' }}>{c.title || c.headline || ''}</td>
                  <td style={{ padding: '13px 14px', borderRight: '1px solid #f1f5f9' }}>
                    {c.linkedin
                      ? <a href={c.linkedin} target="_blank" rel="noreferrer" style={{ color: '#2563eb', display: 'flex', alignItems: 'center' }} onClick={e => e.stopPropagation()}>
                          <ExternalLink size={14} />
                        </a>
                      : <span style={{ color: '#cbd5e1' }}></span>}
                  </td>
                  <td style={{ padding: '13px 14px', fontSize: 13, fontWeight: 700, color: '#0f172a', borderRight: '1px solid #f1f5f9' }}>{c.company || ''}</td>
                  <td style={{ padding: '13px 14px', fontSize: 12, color: '#64748b', borderRight: '1px solid #f1f5f9' }}>{c.product_service || ''}</td>
                  <td style={{ padding: '13px 14px', fontSize: 13, color: '#374151', borderRight: '1px solid #f1f5f9' }}>{c.city || ''}</td>
                  <td style={{ padding: '13px 14px', fontSize: 12, color: '#64748b', borderRight: '1px solid #f1f5f9' }}>{c.location_type || ''}</td>
                  <td style={{ padding: '13px 14px', borderRight: '1px solid #f1f5f9' }}>
                    <ExpBar value={c.total_experience_years || 0} />
                  </td>
                  <td style={{ padding: '13px 14px', fontSize: 13, fontWeight: 600, color: '#374151', borderRight: '1px solid #f1f5f9' }}>
                    {c.avg_tenure_years > 0 ? `${c.avg_tenure_years}y` : ''}
                  </td>
                  <td style={{ padding: '13px 14px', fontSize: 12, color: '#0f172a', borderRight: '1px solid #f1f5f9' }}>
                    {contactInfo[c.id]?.enriching && !(contactInfo[c.id]?.email || c.email)
                      ? <span style={{ display: 'flex', alignItems: 'center', gap: 5, color: '#f97316', fontSize: 11, fontWeight: 600, animation: 'pulse 1.5s ease-in-out infinite' }}>
                          <span style={{ width: 7, height: 7, borderRadius: '50%', background: '#f97316', display: 'inline-block', animation: 'pulse 1.5s ease-in-out infinite' }} />
                          Fetching via Clay...
                        </span>
                      : (contactInfo[c.id]?.email || c.email)
                        ? <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}><Mail size={12} color="#f97316" />{contactInfo[c.id]?.email || c.email}</span>
                        : <span style={{ color: '#d1d5db', fontSize: 11 }}>— shortlist to fetch</span>}
                  </td>
                  <td style={{ padding: '13px 14px', fontSize: 12, color: '#374151', borderRight: '1px solid #f1f5f9' }}>
                    {contactInfo[c.id]?.enriching && !(contactInfo[c.id]?.phone || c.phone)
                      ? <span style={{ color: '#94a3b8', fontSize: 11 }}>fetching...</span>
                      : (contactInfo[c.id]?.phone || c.phone)
                        ? <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}><Phone size={12} color="#64748b" />{contactInfo[c.id]?.phone || c.phone}</span>
                        : <span style={{ color: '#d1d5db', fontSize: 11 }}>—</span>}
                  </td>
                  <td style={{ padding: '13px 14px', borderRight: '1px solid #f1f5f9' }}>
                    <StatusDropdown 
                      status={c.status} 
                      candidateId={c.id} 
                      onUpdate={(id, newStatus) => {
                        setCandidates(prev => prev.map(cand => cand.id === id ? { ...cand, status: newStatus } : cand));
                      }}
                      onShortlisted={handleShortlisted}
                    />
                  </td>
                  <td 
                    onClick={() => setSelectedCandidateForChat(c)}
                    style={{ 
                        padding: '13px 14px', fontSize: 12, color: '#2563eb', fontWeight: 600, borderRight: '1px solid #f1f5f9',
                        cursor: 'pointer', textDecoration: c.response ? 'underline' : 'none'
                    }}
                  >
                    {c.response || 'View Chat'}
                  </td>
                  <td style={{ padding: '13px 14px', borderRight: '1px solid #f1f5f9' }}>
                    <EditableNotes candidateId={c.id} initialNotes={c.notes} />
                  </td>
                  {isSemanticSearch && (
                    <td style={{ padding: '13px 14px', borderRight: '1px solid #f1f5f9' }}>
                      <div style={{ 
                        display: 'inline-flex', padding: '4px 8px', borderRadius: '6px', 
                        background: c.match_score > 80 ? '#dcfce7' : c.match_score > 60 ? '#fef9c3' : '#f1f5f9',
                        color: c.match_score > 80 ? '#166534' : c.match_score > 60 ? '#854d0e' : '#475569',
                        fontSize: '11px', fontWeight: 800
                      }}>
                        {c.match_score}% Match
                      </div>
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div style={{ padding: '12px 20px', background: '#fff', borderTop: '1.5px solid #f1f5f9', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 13, color: '#94a3b8' }}>Rows per page:</span>
              <select 
                value={pageSize}
                onChange={e => {
                  setPageSize(Number(e.target.value));
                  setPage(1);
                }}
                style={{
                  padding: '4px 8px', borderRadius: '6px', border: '1.5px solid #e2e8f0',
                  fontSize: '12px', fontWeight: 600, color: '#0f172a', outline: 'none',
                  background: '#f8fafc', cursor: 'pointer'
                }}
              >
                {PAGE_SIZE_OPTIONS.map(opt => <option key={opt} value={opt}>{opt}</option>)}
              </select>
            </div>
            <span style={{ fontSize: 13, color: '#94a3b8' }}>
              Showing {Math.min((page - 1) * pageSize + 1, total)}–{Math.min(page * pageSize, total)} of <strong style={{ color: '#0f172a' }}>{total.toLocaleString()}</strong> candidates
            </span>
          </div>
          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}
              style={{ width: 32, height: 32, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#f8fafc', border: '1.5px solid #e2e8f0', borderRadius: 7, cursor: page === 1 ? 'not-allowed' : 'pointer', opacity: page === 1 ? 0.4 : 1 }}>
              <ChevronLeft size={14} color="#64748b" />
            </button>

            {/* Improved Page numbers with ellipsis */}
            {(() => {
              const pages = [];
              const range = 2; // Number of pages to show around current page
              
              for (let i = 1; i <= totalPages; i++) {
                if (
                  i === 1 || 
                  i === totalPages || 
                  (i >= page - range && i <= page + range)
                ) {
                  pages.push(
                    <button key={i} onClick={() => setPage(i)}
                      style={{ 
                        width: 32, height: 32, display: 'flex', alignItems: 'center', justifyContent: 'center', 
                        background: i === page ? '#f97316' : '#f8fafc', 
                        border: `1.5px solid ${i === page ? '#f97316' : '#e2e8f0'}`, 
                        borderRadius: 7, cursor: 'pointer', fontSize: 13, 
                        fontWeight: i === page ? 700 : 500, color: i === page ? '#fff' : '#64748b',
                        transition: 'all 0.15s'
                      }}
                      onMouseEnter={e => i !== page && (e.currentTarget.style.borderColor = '#cbd5e1')}
                      onMouseLeave={e => i !== page && (e.currentTarget.style.borderColor = '#e2e8f0')}
                    >
                      {i}
                    </button>
                  );
                } else if (
                  i === page - range - 1 || 
                  i === page + range + 1
                ) {
                  pages.push(<span key={`ell-${i}`} style={{ color: '#94a3b8', fontSize: 14 }}>...</span>);
                }
              }
              return pages;
            })()}

            <button onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page === totalPages}
              style={{ width: 32, height: 32, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#f8fafc', border: '1.5px solid #e2e8f0', borderRadius: 7, cursor: page === totalPages ? 'not-allowed' : 'pointer', opacity: page === totalPages ? 0.4 : 1 }}>
              <ChevronRight size={14} color="#64748b" />
            </button>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes shimmer { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
      `}</style>

      {selectedCandidateForChat && (
        <ConversationModal 
          candidate={selectedCandidateForChat} 
          onClose={() => setSelectedCandidateForChat(null)} 
        />
      )}

      {shortlistCard && (
        <ShortlistCard data={shortlistCard} onClose={() => setShortlistCard(null)} />
      )}

      {shortlistingId && (
        <div style={{
          position: 'fixed', bottom: '28px', right: '28px', zIndex: 10001,
          background: '#0f172a', color: '#fff', borderRadius: '14px',
          padding: '14px 20px', display: 'flex', alignItems: 'center', gap: '12px',
          boxShadow: '0 10px 15px -3px rgba(0,0,0,0.2)', fontSize: '13.5px', fontWeight: 600
        }}>
          <span style={{ width: 18, height: 18, border: '2.5px solid rgba(255,255,255,0.3)', borderTopColor: '#fff', borderRadius: '50%', animation: 'spin 0.75s linear infinite', display: 'inline-block', flexShrink: 0 }} />
          Triggering outreach...
        </div>
      )}
    </div>
  );
}
