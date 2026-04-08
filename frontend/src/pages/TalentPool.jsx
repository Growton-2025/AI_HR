import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useAppStore, API_BASE } from '../store/useAppStore';
import axios from 'axios';
import { toast } from 'sonner';
import {
  Search, ExternalLink, ChevronLeft, ChevronRight, Filter,
  User, Building2, MapPin, Briefcase, BarChart2,
  SlidersHorizontal, RefreshCw, UserPlus, X, ChevronDown,
  Activity, MessageSquareMore, Users, Plus, Edit2, Check, Download,
  Mail, Phone, MessageSquare, Linkedin, Send
} from 'lucide-react';
import StatusDropdown, { RECRUITMENT_STAGES, STATUS_STYLES } from '../components/StatusDropdown';

// ── Clickable Editable Cell (renders as <td>) ────────────────
function ClickableEditableCell({ id, field, value, onUpdate, placeholder = '—' }) {
  const [isEditing, setIsEditing] = useState(false);
  const [tempValue, setTempValue] = useState('');
  const [loading, setLoading] = useState(false);

  const isNA = !value || ['na', 'n/a', 'none'].includes(value.toString().toLowerCase());

  const handleSave = async () => {
    if (tempValue !== value) {
      setLoading(true);
      await onUpdate(id, { [field]: tempValue });
      setLoading(false);
    }
    setIsEditing(false);
  };

  if (isEditing) {
    return (
      <td style={{ padding: '0 14px', borderRight: '1px solid #f1f5f9' }}>
        <input
          autoFocus
          value={tempValue}
          onChange={(e) => setTempValue(e.target.value)}
          onBlur={handleSave}
          onKeyDown={(e) => {
            if (e.key === 'Enter') handleSave();
            if (e.key === 'Escape') setIsEditing(false);
          }}
          style={{
            width: '100%', padding: '4px 6px', border: '1px solid #f97316',
            borderRadius: '4px', fontSize: '12px', fontFamily: 'inherit', outline: 'none',
          }}
        />
      </td>
    );
  }

  return (
    <td
      onClick={() => { setIsEditing(true); setTempValue(isNA ? '' : (value || '')); }}
      style={{
        padding: '13px 14px', borderRight: '1px solid #f1f5f9',
        fontSize: '12px', cursor: 'pointer',
        color: isNA ? '#f97316' : '#334155',
      }}
      title="Click to edit"
    >
      <span style={{
        padding: isNA ? '2px 6px' : '0',
        borderRadius: '4px',
        background: isNA ? 'rgba(249,115,22,0.06)' : 'transparent',
        border: isNA ? '1px dashed rgba(249,115,22,0.4)' : 'none',
        display: 'inline-block',
      }}>
        {isNA ? (value || placeholder) : value}
        {loading && ' ...'}
      </span>
    </td>
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
      error: { bg: '#fee2e2', color: '#b91c1c', icon: '✕', text: 'Failed' },
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
  const [isSaving, setIsSaving] = useState(false);
  const updateCandidateNotes = useAppStore(state => state.updateCandidateNotes);

  useEffect(() => {
    setNotes(initialNotes || '');
  }, [initialNotes]);

  const handleSave = async () => {
    if (notes === (initialNotes || '')) {
      setIsEditing(false);
      return;
    }
    setIsSaving(true);
    await updateCandidateNotes(candidateId, notes);
    setIsSaving(false);
    setIsEditing(false);
  };

  return (
    <div style={{ width: '100%', minWidth: 100 }}>
      {isEditing && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
          background: 'rgba(15, 23, 42, 0.4)', backdropFilter: 'blur(4px)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          zIndex: 9999, padding: '20px'
        }} onClick={() => !isSaving && setIsEditing(false)}>
          <div
            onClick={e => e.stopPropagation()}
            style={{
              background: '#fff', width: '100%', maxWidth: '500px',
              borderRadius: '16px', display: 'flex', flexDirection: 'column',
              boxShadow: '0 20px 25px -5px rgba(0,0,0,0.1)', overflow: 'hidden'
            }}
          >
            <div style={{ padding: '20px', borderBottom: '1px solid #e2e8f0', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: '#f8fafc' }}>
              <div style={{ fontSize: '16px', fontWeight: 800, color: '#0f172a' }}>Candidate Notes</div>
              <button disabled={isSaving} onClick={() => setIsEditing(false)} style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '6px', borderRadius: '50%', color: '#64748b' }} onMouseEnter={e => e.currentTarget.style.background = '#e2e8f0'} onMouseLeave={e => e.currentTarget.style.background = 'none'}>
                <X size={18} />
              </button>
            </div>

            <div style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <textarea
                autoFocus
                value={notes}
                maxLength={5000}
                onChange={e => setNotes(e.target.value)}
                placeholder="Add notes about this candidate..."
                rows={6}
                style={{
                  width: '100%', padding: '12px', boxSizing: 'border-box', border: '1.5px solid #e2e8f0',
                  borderRadius: '10px', fontSize: '13.5px', outline: 'none',
                  fontFamily: 'inherit', resize: 'vertical', minHeight: '120px',
                  color: '#0f172a', transition: 'border-color 0.15s'
                }}
                onFocus={e => e.target.style.borderColor = '#f97316'}
                onBlur={e => e.target.style.borderColor = '#e2e8f0'}
              />
              <div style={{ display: 'flex', justifyContent: 'flex-end', fontSize: '12px', color: notes.length >= 5000 ? '#ef4444' : '#64748b', fontWeight: 600 }}>
                {notes.length} / 5000
              </div>
            </div>

            <div style={{ padding: '16px 20px', background: '#f8fafc', borderTop: '1px solid #e2e8f0', display: 'flex', justifyContent: 'flex-end', gap: '12px' }}>
              <button
                disabled={isSaving}
                onClick={() => setIsEditing(false)}
                style={{ padding: '8px 16px', borderRadius: '8px', border: '1.5px solid #e2e8f0', background: '#fff', color: '#475569', fontSize: '13.5px', fontWeight: 600, cursor: 'pointer' }}
              >
                Cancel
              </button>
              <button
                disabled={isSaving}
                onClick={handleSave}
                style={{ padding: '8px 16px', borderRadius: '8px', border: 'none', background: '#f97316', color: '#fff', fontSize: '13.5px', fontWeight: 600, cursor: isSaving ? 'wait' : 'pointer', opacity: isSaving ? 0.7 : 1 }}
              >
                {isSaving ? 'Saving...' : 'Save Notes'}
              </button>
            </div>
          </div>
        </div>
      )}

      <div
        onClick={() => setIsEditing(true)}
        style={{
          fontSize: '12.5px',
          color: initialNotes ? '#334155' : '#94a3b8',
          fontStyle: initialNotes ? 'normal' : 'italic',
          cursor: 'pointer',
          minHeight: '20px',
          padding: '4px 8px',
          borderRadius: '6px',
          maxWidth: '180px',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
          border: '1px solid transparent',
          transition: 'all 0.15s'
        }}
        onMouseEnter={e => {
          e.currentTarget.style.background = '#f1f5f9';
          e.currentTarget.style.borderColor = '#e2e8f0';
        }}
        onMouseLeave={e => {
          e.currentTarget.style.background = 'transparent';
          e.currentTarget.style.borderColor = 'transparent';
        }}
      >
        {notes || 'Click to add notes...'}
      </div>
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

        <style dangerouslySetInnerHTML={{
          __html: `
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
  // Auto-select LinkedIn tab if candidate has an active LinkedIn response
  const hasLiActivity = ['replied', 'message_sent', 'connection_accepted', 'in_campaign'].includes(candidate?.li_status);
  const hasLiResponse = Boolean(candidate?.li_response_text);
  const defaultPlatform = (hasLiResponse || candidate?.li_status === 'replied') ? 'linkedin' : 'email';
  const [platform, setPlatform] = useState(defaultPlatform);
  const [threads, setThreads] = useState({
    email: { messages: [], loaded: false, error: '' },
    linkedin: { messages: [], loaded: false, error: '' }
  });
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [replyText, setReplyText] = useState('');
  const [sending, setSending] = useState(false);
  const [localSentMessagesByPlatform, setLocalSentMessagesByPlatform] = useState({
    email: [],
    linkedin: []
  }); // Buffer for messages sent before server syncs them

  const fetchChatHistory = useAppStore(state => state.fetchChatHistory);
  const sendChatReply = useAppStore(state => state.sendChatReply);
  const { heyreachCampaignId, setHeyreachCampaignId, triggerHeyReachOutreach } = useAppStore();
  const [isTriggering, setIsTriggering] = useState(false);
  const [hasTriggered, setHasTriggered] = useState(false);
  const messagesEndRef = useRef(null);
  const threadsRef = useRef(threads);
  const requestSeqRef = useRef({ email: 0, linkedin: 0 });
  const candidateIdRef = useRef(candidate.id);
  const lastSyncTimeRef = useRef(Date.now());


  useEffect(() => {
    threadsRef.current = threads;
  }, [threads]);

  useEffect(() => {
    candidateIdRef.current = candidate.id;
  }, [candidate.id]);

  const activeThread = threads[platform] || { messages: [], loaded: false, error: '' };
  const messages = activeThread.messages || [];
  const localSentMessages = localSentMessagesByPlatform[platform] || [];
  const conversationAccent = platform === 'linkedin' ? '#0077b5' : '#f97316';
  const hasAnyVisibleMessages = messages.length > 0 || localSentMessages.length > 0;

  const loadMessages = useCallback(async ({ showLoader = false, silent = false, targetPlatform = platform } = {}) => {
    const cachedThread = threadsRef.current[targetPlatform];
    const candidateId = candidate.id;
    const requestSeq = ++requestSeqRef.current[targetPlatform];
    // Always block (show skeleton) on initial load for this platform
    const shouldBlock = showLoader || !cachedThread?.loaded;

    if (targetPlatform === platform) {
      if (!cachedThread?.loaded) {
        setLoading(true);
      } else if (!silent) {
        setRefreshing(true);
      }
    }

    const res = await fetchChatHistory(0, candidateId, targetPlatform);
    if (candidateIdRef.current !== candidateId || requestSeqRef.current[targetPlatform] !== requestSeq) {
      return;
    }

    if (res.success) {
      // Small artificial delay to ensure smooth transition from skeleton to content
      await new Promise(r => setTimeout(r, 200));
      setThreads(prev => ({
        ...prev,
        [targetPlatform]: {
          messages: res.messages || [],
          loaded: true,
          error: ''
        }
      }));
    } else {
      setThreads(prev => ({
        ...prev,
        [targetPlatform]: {
          ...prev[targetPlatform],
          loaded: true,
          error: res.error || `Failed to fetch ${targetPlatform} chat history`
        }
      }));
    }
    if (targetPlatform === platform) {
      setLoading(false);
      if (!silent) setRefreshing(false);
    }
  }, [candidate.id, fetchChatHistory, platform]);

  useEffect(() => {
    const nextThreads = {
      email: { messages: [], loaded: false, error: '' },
      linkedin: { messages: [], loaded: false, error: '' }
    };
    threadsRef.current = nextThreads;
    requestSeqRef.current = { email: 0, linkedin: 0 };
    setThreads(nextThreads);
    setLocalSentMessagesByPlatform({ email: [], linkedin: [] });
    setLoading(true);
    setRefreshing(false);
    setReplyText('');
    setHasTriggered(false);
  }, [candidate.id]);

  useEffect(() => {
    setReplyText('');
    const cached = threadsRef.current[platform];
    if (!cached || !cached.loaded) {
      setLoading(true);
    }
    loadMessages({ targetPlatform: platform });
  }, [candidate.id, platform, loadMessages]);

  // Auto-poll every 5s for new replies
  useEffect(() => {
    const interval = setInterval(() => loadMessages({ silent: true, targetPlatform: platform }), 5000);
    return () => clearInterval(interval);
  }, [candidate.id, platform, loadMessages]);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, localSentMessages]);

  const handleSend = async () => {
    const text = replyText.trim();
    if (!text) return;
    const activePlatform = platform;
    const pendingId = `${activePlatform}-${Date.now()}-${Math.random().toString(36).slice(2)}`;

    // Add to local buffer instantly
    const optimisticMsg = {
      type: 'SENT',
      email_body: text,
      time: new Date().toISOString(),
      sentAt: Date.now(), // Local timestamp for clearing logic
      sender_name: 'You',
      _pendingClientId: pendingId,
      _pending: true
    };
    setLocalSentMessagesByPlatform(prev => ({
      ...prev,
      [activePlatform]: [...(prev[activePlatform] || []), optimisticMsg]
    }));
    setReplyText('');
    setSending(true);

    // Instant scroll to the bottom to show the optimistic message
    setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, 50);

    const res = await sendChatReply(0, candidate.id, text, activePlatform);
    if (res.success) {
      setTimeout(() => loadMessages({ silent: true, targetPlatform: activePlatform }), 3000);
    } else {
      setLocalSentMessagesByPlatform(prev => ({
        ...prev,
        [activePlatform]: (prev[activePlatform] || []).filter(m => m._pendingClientId !== pendingId)
      }));
      toast.error(res.error || 'Failed to send reply');
    }

    setSending(false);
  };

  const buildMessageList = () => {
    const serverMsgs = activeThread.messages || [];
    const localMsgs = localSentMessagesByPlatform[platform] || [];

    // Filter out duplicates from server
    const uniqueServerMsgs = [];
    const serverMsgsSeen = new Set();
    serverMsgs.forEach(msg => {
      const msgKey = `${msg.type}-${(msg.email_body || '').trim().toLowerCase()}-${msg.time}`;
      if (!serverMsgsSeen.has(msgKey)) {
        uniqueServerMsgs.push(msg);
        serverMsgsSeen.add(msgKey);
      }
    });

    // Match local against server
    const pendingToShow = localMsgs.filter(lm => {
      const match = uniqueServerMsgs.some(sm => {
        if (sm.type !== 'SENT') return false;
        const sBody = (sm.email_body || '').trim().toLowerCase();
        const lBody = (lm.email_body || '').trim().toLowerCase();
        // Match if one includes the other, but only for first 50 chars to avoid false hits on generic "hi"
        return sBody === lBody || sBody.includes(lBody) || lBody.includes(sBody);
      });
      // Also expire local messages after 60s if they haven't synced, to prevent ghosting
      const age = (Date.now() - (lm.sentAt || 0)) / 1000;
      return !match && age < 60;
    });

    const combined = [...uniqueServerMsgs, ...pendingToShow];
    return combined.sort((a, b) => {
      const getT = (m) => new Date(m.time || m.created_at || m.timestamp || 0).getTime();
      return getT(a) - getT(b);
    });
  };

  const displayMessages = buildMessageList();

  const formatTime = (timeStr) => {
    if (!timeStr) return '';
    try {
      const d = new Date(timeStr);
      if (isNaN(d.getTime())) return timeStr;
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) + ' ' + d.toLocaleDateString([], { month: '2-digit', day: '2-digit' });
    } catch {
      return timeStr;
    }
  };

  return (
    <div style={{
      position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
      background: 'rgba(15, 23, 42, 0.4)', backdropFilter: 'blur(8px)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      zIndex: 9999, padding: '20px', animation: 'fadeIn 0.2s ease-out'
    }} onClick={onClose}>
       <style>{`
         @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
         @keyframes slideUp { from { transform: translateY(20px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
         @keyframes shimmer { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }
         .hide-scrollbar::-webkit-scrollbar { display: none; }
         .hide-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
         @keyframes spin-anim { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
         .spin-anim { animation: spin-anim 0.8s linear infinite; will-change: transform; }
         .message-bubble { will-change: transform, opacity; }
       `}</style>
      <div
        onClick={e => e.stopPropagation()}
        style={{
          background: '#fff', width: '100%', maxWidth: '750px', height: '85vh',
          borderRadius: '28px', display: 'flex', flexDirection: 'column',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          overflow: 'hidden', animation: 'slideUp 0.3s ease-out'
        }}
      >
        {/* Header */}
        <div style={{ padding: '24px 32px', borderBottom: '1px solid #f1f5f9', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: '#fff' }}>
          <div>
            <div style={{ fontSize: '20px', fontWeight: 800, color: '#0f172a', letterSpacing: '-0.02em', display: 'flex', alignItems: 'center', gap: '10px' }}>
              {candidate.first_name} {candidate.last_name || ''}
              <div style={{ px: '8px', py: '2px', background: '#f1f5f9', borderRadius: '6px', fontSize: '11px', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                ID: {candidate.id}
              </div>
            </div>
            <div style={{ fontSize: '13px', color: '#64748b', display: 'flex', alignItems: 'center', gap: '6px', marginTop: '4px' }}>
              <MessageSquare size={14} />
              <span>Real-time Sync Active</span>
              {refreshing && (
                <span style={{ fontSize: '11px', color: conversationAccent, fontWeight: 700, marginLeft: '4px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                   <div style={{ width: 4, height: 4, borderRadius: '50%', background: conversationAccent, animation: 'pulse 1s infinite' }} />
                   UPDATING
                </span>
              )}
            </div>
          </div>
          <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
            <button 
              onClick={() => loadMessages({ silent: false })}
              style={{
                background: '#fff', border: '1.5px solid #e2e8f0', cursor: 'pointer', padding: '8px 14px', 
                borderRadius: '12px', color: '#64748b', fontSize: '12px', fontWeight: 600,
                display: 'flex', alignItems: 'center', gap: '6px', transition: 'all 0.2s'
              }}
              onMouseEnter={e => e.currentTarget.style.borderColor = '#94a3b8'}
              onMouseLeave={e => e.currentTarget.style.borderColor = '#e2e8f0'}
            >
              <RefreshCw size={14} className={refreshing ? 'spin-anim' : ''} />
              Manual Sync
            </button>
            <button onClick={onClose} style={{ background: '#f1f5f9', border: 'none', cursor: 'pointer', padding: '10px', borderRadius: '12px', color: '#64748b', transition: 'all 0.2s' }} onMouseEnter={e => e.currentTarget.style.background = '#e2e8f0'} onMouseLeave={e => e.currentTarget.style.background = '#f1f5f9'}>
              <X size={20} />
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div style={{ display: 'flex', padding: '0 32px', borderBottom: '1px solid #f1f5f9', background: '#fff', gap: '24px' }}>
          {[
            { id: 'linkedin', label: 'LinkedIn', icon: Linkedin, color: '#0077b5' },
            { id: 'email', label: 'Email', icon: Mail, color: '#f97316' }
          ].map(t => (
            <button
              key={t.id}
              onClick={() => setPlatform(t.id)}
              style={{
                padding: '16px 4px',
                background: 'transparent',
                border: 'none',
                borderBottom: platform === t.id ? `3px solid ${t.color}` : '3px solid transparent',
                color: platform === t.id ? '#0f172a' : '#64748b',
                fontWeight: platform === t.id ? 700 : 500,
                fontSize: '14px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                transition: 'all 0.2s ease',
                position: 'relative'
              }}
            >
              <t.icon size={18} color={platform === t.id ? t.color : '#94a3b8'} strokeWidth={platform === t.id ? 2.5 : 2} />
              {t.label}
              {t.id === 'linkedin' && hasLiResponse && platform !== 'linkedin' && (
                <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#22c55e', position: 'absolute', top: 12, right: -6, border: '2px solid #fff' }} />
              )}
            </button>
          ))}
        </div>

         {/* Messages */}
         <div className="hide-scrollbar" style={{ flex: 1, overflowY: 'auto', padding: '24px', display: 'flex', flexDirection: 'column', gap: '8px', background: '#f8fafc', containment: 'layout style paint' }}>
           {loading ? (
             <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
               {[1, 2, 3].map(i => (
                 <div key={i} style={{ width: i % 2 === 0 ? '55%' : '65%', height: '72px', borderRadius: '18px', background: 'linear-gradient(90deg, #f1f5f9 25%, #e2e8f0 50%, #f1f5f9 75%)', backgroundSize: '200% 100%', animation: 'shimmer 1.5s infinite', alignSelf: i % 2 === 0 ? 'flex-end' : 'flex-start' }} />
               ))}
             </div>
           ) : displayMessages.length === 0 ? (
             <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: '#94a3b8', gap: '16px' }}>
               <div style={{ width: 64, height: 64, borderRadius: '20px', background: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px solid #e2e8f0' }}>
                 <MessageSquare size={32} opacity={0.5} />
               </div>
               <div style={{ textAlign: 'center' }}>
                 <div style={{ fontWeight: 600, color: '#64748b' }}>No messages yet</div>
                 <div style={{ fontSize: '13px' }}>Start the conversation below</div>
               </div>
               {platform === 'linkedin' && !heyreachCampaignId && (
                 <button
                   onClick={async () => {
                     setIsTriggering(true);
                     const res = await triggerHeyReachOutreach([candidate.id]);
                     setIsTriggering(false);
                     if (res.success) {
                       setHasTriggered(true);
                       loadMessages({ silent: true });
                     }
                   }}
                   disabled={isTriggering || hasTriggered}
                   style={{
                     marginTop: '12px', padding: '10px 20px', background: '#0077b5', color: '#fff',
                     border: 'none', borderRadius: '12px', fontWeight: 600, fontSize: '13px', cursor: 'pointer',
                     display: 'flex', alignItems: 'center', gap: '8px'
                   }}
                 >
                   {isTriggering ? 'Starting...' : hasTriggered ? 'Campaign Started' : 'Start LinkedIn Outreach'}
                 </button>
               )}
             </div>
           ) : (
             <>
               {displayMessages.map((msg, idx) => {
                 // Determine if message is from candidate (incoming) or from us (outgoing)
                 // Priority: direction field > type field > fallback
                 let isCandidate = false;
                 if (msg.direction === 'inbound') {
                   isCandidate = true;
                 } else if (msg.direction === 'outbound') {
                   isCandidate = false;
                 } else if (msg.type === 'REPLY' || msg.type === 'INBOX') {
                   isCandidate = true;
                 } else if (msg.type === 'SENT') {
                   isCandidate = false;
                 } else {
                   // Default based on is_reply flag
                   isCandidate = Boolean(msg.is_reply);
                 }
                 
                 const nextMsg = displayMessages[idx + 1];
                 let nextIsCandidate = false;
                 if (nextMsg) {
                   if (nextMsg.direction === 'inbound') {
                     nextIsCandidate = true;
                   } else if (nextMsg.direction === 'outbound') {
                     nextIsCandidate = false;
                   } else if (nextMsg.type === 'REPLY' || nextMsg.type === 'INBOX') {
                     nextIsCandidate = true;
                   } else if (nextMsg.type === 'SENT') {
                     nextIsCandidate = false;
                   } else {
                     nextIsCandidate = Boolean(nextMsg.is_reply);
                   }
                 }
                 
                 const isConsecutive = nextMsg && isCandidate === nextIsCandidate;
                 
                 const rawBody = msg.email_body || msg.message || msg.text || '';
                 const cleanBody = rawBody.replace(/<[^>]+>/g, '')
                   .replace(/&nbsp;/g, ' ')
                   .replace(/&amp;/g, '&')
                   .replace(/&quot;/g, '"')
                   .replace(/&apos;/g, "'")
                   .trim();

                 return (
                   <div key={msg.id || idx} className="message-bubble" style={{
                     display: 'flex',
                     flexDirection: 'column',
                     alignItems: isCandidate ? 'flex-start' : 'flex-end',
                     gap: '4px',
                     maxWidth: '85%',
                     alignSelf: isCandidate ? 'flex-start' : 'flex-end',
                     opacity: msg._pending ? 0.7 : 1,
                     marginBottom: isConsecutive ? '4px' : '16px'
                   }}>
                     <div style={{ display: 'flex', alignItems: 'flex-end', gap: '8px', flexDirection: isCandidate ? 'row' : 'row-reverse' }}>
                       <div style={{
                         width: 28, height: 28, borderRadius: '10px',
                         visibility: isConsecutive ? 'hidden' : 'visible',
                         background: isCandidate ? '#fff' : conversationAccent,
                         display: 'flex', alignItems: 'center', justifyContent: 'center',
                         fontSize: '12px', fontWeight: 700, color: isCandidate ? conversationAccent : '#fff',
                         border: isCandidate ? `1.5px solid ${conversationAccent}20` : 'none',
                         boxShadow: '0 2px 4px rgba(0,0,0,0.05)',
                         flexShrink: 0
                       }}>
                         {isCandidate ? (candidate.first_name?.[0] || 'C') : 'You'}
                       </div>
                       <div style={{
                         padding: '12px 18px',
                         borderRadius: isCandidate ? '18px 18px 18px 4px' : '18px 18px 4px 18px',
                         background: isCandidate ? '#fff' : conversationAccent,
                         color: isCandidate ? '#334155' : '#fff',
                         fontSize: '14.5px',
                         lineHeight: '1.5',
                         boxShadow: isCandidate ? '0 4px 6px -1px rgba(0,0,0,0.05)' : `0 10px 15px -3px ${conversationAccent}15`,
                         border: isCandidate ? '1px solid #f1f5f9' : 'none',
                         whiteSpace: 'pre-wrap',
                         wordBreak: 'break-word',
                         position: 'relative'
                       }}>
                         {cleanBody}
                         {msg._pending && (
                           <div style={{ 
                             position: 'absolute', right: -24, bottom: 4, 
                             display: 'flex', gap: '2px', opacity: 0.5 
                           }}>
                             {[0, 1, 2].map(i => (
                               <div key={i} style={{ 
                                 width: 4, height: 4, borderRadius: '50%', background: '#64748b', 
                                 animation: 'pulse 1s infinite', animationDelay: `${i * 0.2}s` 
                               }} />
                             ))}
                           </div>
                         )}
                       </div>
                     </div>
                     <div style={{ fontSize: '10px', color: '#94a3b8', margin: isCandidate ? '0 42px' : '0 12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                       {msg._pending ? 'Sending' : formatTime(msg.time)}
                     </div>
                   </div>
                 );
               })}
               <div ref={messagesEndRef} />
             </>
           )}
         </div>

        {/* Reply Area */}
        <div style={{ padding: '24px 32px', background: '#fff', borderTop: '1px solid #f1f5f9' }}>
          <div style={{
            display: 'flex', alignItems: 'center', gap: '12px', background: '#f8fafc',
            padding: '8px 8px 8px 16px', borderRadius: '20px', border: '1.5px solid #e2e8f0',
            transition: 'border-color 0.2s', focusWithin: { borderColor: conversationAccent }
          }}>
            <textarea
              value={replyText}
              onChange={e => setReplyText(e.target.value)}
              placeholder={`Reply via ${platform === 'linkedin' ? 'LinkedIn' : 'Email'}...`}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              rows={1}
              style={{
                flex: 1, background: 'none', border: 'none', outline: 'none',
                resize: 'none', padding: '8px 0', fontSize: '14px', color: '#0f172a',
                fontFamily: 'inherit', maxHeight: '120px'
              }}
            />
            <button
              onClick={handleSend}
              disabled={sending || !replyText.trim()}
              style={{
                width: 40, height: 40, borderRadius: '16px',
                background: replyText.trim() ? conversationAccent : '#e2e8f0',
                color: '#fff', border: 'none', cursor: 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                transition: 'all 0.2s', transform: sending ? 'scale(0.9)' : 'none'
              }}
            >
              {sending ? (
                <div style={{ width: 18, height: 18, border: '2px solid #fff', borderTop: '2px solid transparent', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
              ) : (
                <Send size={18} />
              )}
            </button>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '12px', padding: '0 4px' }}>
            <span style={{ fontSize: '11px', color: '#94a3b8' }}>
              Press <b>Enter</b> to send, <b>Shift + Enter</b> for new line
            </span>
            {platform === 'email' && (
              <span style={{ fontSize: '11px', color: '#94a3b8', display: 'flex', alignItems: 'center', gap: '4px' }}>
                <Mail size={12} /> Sending from: <b>Recruitment Team</b>
              </span>
            )}
          </div>
        </div>
      </div>
      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        @keyframes pulse { 0% { transform: scale(0.95); opacity: 0.5; } 50% { transform: scale(1.05); opacity: 1; } 100% { transform: scale(0.95); opacity: 0.5; } }
      `}</style>
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
  const [activeStatusTab, setActiveStatusTab] = useState('');
  const [sortBy, setSortBy] = useState('name');
  const [sortDir, setSortDir] = useState('asc');
  const [isSemanticSearch, setIsSemanticSearch] = useState(false);
  const [selectedCandidateForChat, setSelectedCandidateForChat] = useState(null);
  const [shortlistCard, setShortlistCard] = useState(null);
  const [shortlistingId, setShortlistingId] = useState(null);
  const [selectedIds, setSelectedIds] = useState(new Set());
  const [showAddToListModal, setShowAddToListModal] = useState(false);
  const [contactInfo, setContactInfo] = useState(readPersistedContactInfo); // { [candidateId]: { email, phone, enriching } }
  const analytics = useAppStore(state => state.analytics);
  const fetchAnalytics = useAppStore(state => state.fetchAnalytics);
  const syncOutreachResponses = useAppStore(state => state.syncOutreachResponses);
  const shortlistAndOutreach = useAppStore(state => state.shortlistAndOutreach);
  const updateCandidateField = useAppStore(state => state.updateCandidateField);
  const prefetchCallLists = useAppStore(state => state.fetchCallLists);
  const prefetchCallStats = useAppStore(state => state.fetchCallStats);
  const prefetchCalls = useAppStore(state => state.fetchCalls);
  const { heyreachCampaignId } = useAppStore();
  const didInitRef = useRef(false);
  const talentPoolRequestSeqRef = useRef(0);

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

    const res = await shortlistAndOutreach(candidateId, {
      hr_campaign_id: parseInt(heyreachCampaignId)
    });
    setShortlistingId(null);
    if (res.success) {
      const d = res.data;
      toast.success(`Successfully shortlisted ${d.name || 'candidate'}`);
      // Notify about skips if any
      if (d.email_outreach === 'skipped') {
        toast.warning(`Smartlead: ${d.email_outreach_msg || 'Email missing'}`);
      }
      if (d.linkedin_outreach === 'skipped') {
        toast.warning(`HeyReach: ${d.li_outreach_msg || 'LinkedIn URL missing'}`);
      }
      fetchAnalytics();
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
      toast.error(res.error || 'Operation failed');
      setContactInfo(prev => ({
        ...prev,
        [candidateId]: {
          email: prev[candidateId]?.email || '',
          phone: prev[candidateId]?.phone || '',
          enriching: false
        }
      }));
      // setShortlistCard({ name: 'Candidate', email: '', phone: '', linkedin: '', email_outreach: 'error', linkedin_outreach: 'error' }); // Removed as per instruction, toast handles error
    }
  };
  const updateFieldAndMaybeShortlist = async (candidateId, data) => {
    const res = await updateCandidateField(candidateId, data);
    if (!res.success) return res;

    const field = Object.keys(data)[0];
    const value = data[field];
    const isValuable = value && !['na', 'n/a', 'none', ''].includes(value.toLowerCase());

    if (isValuable && (field === 'email' || field === 'phone')) {
      const candidate = candidates.find(c => c.id === candidateId);
      if (candidate && (candidate.status === 'To be started' || !candidate.status)) {
        try {
          await axios.post(`${API_BASE}/candidates/${candidateId}/status`, { status: 'Shortlisted' });
          setCandidates(prev => prev.map(c => c.id === candidateId ? { ...c, status: 'Shortlisted' } : c));
          toast.success(`Candidate automatically shortlisted`);
          fetchAnalytics();
        } catch (err) {
          console.error('Auto-shortlist failed:', err);
        }
      }
    }
    return res;
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
    axios.get(`${API_BASE}/candidates/browse/meta`).then(r => setMeta(r.data)).catch(() => { });
    fetchAnalytics();
  }, [fetchAnalytics]);

  useEffect(() => {
    prefetchCallLists();
    prefetchCallStats();
    prefetchCalls({ due_filter: 'today', status: 'pending' }, { updateState: false });
  }, [prefetchCallLists, prefetchCallStats, prefetchCalls]);

  const fetchCandidates = useCallback(async (pg = 1) => {
    let requestId = 0;
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
      params.set('sort_by', sortBy);
      params.set('sort_dir', sortDir);

      const paramsString = params.toString();
      requestId = ++talentPoolRequestSeqRef.current;
      setLoading(true);
      setCandidates([]);
      setTotal(0);
      setTotalPages(1);
      setStatusCounts({});

      const res = await useAppStore.getState().fetchTalentPool(paramsString);

      if (requestId !== talentPoolRequestSeqRef.current) {
        return;
      }

      if (res.success && res.data) {
        setCandidates(res.data.candidates);
        setTotal(res.data.total);
        setTotalPages(res.data.total_pages);
        setStatusCounts(res.data.status_counts || {});
        setIsSemanticSearch(res.data.is_semantic_search || false);
        mergeContactInfoFromRows(res.data.candidates);

        // Prewarm LinkedIn chat cache for candidates with active LinkedIn outreach
        const prewarmIds = res.data.candidates
          .filter(c => ['replied', 'message_sent', 'connection_accepted', 'in_campaign'].includes(c.li_status))
          .map(c => c.id);
        if (prewarmIds.length > 0) {
          useAppStore.getState().prewarmLinkedInCache(prewarmIds).catch(console.error);
        }
      }
    } catch (e) {
      console.error('Failed to fetch talent pool:', e);
    } finally {
      if (requestId === talentPoolRequestSeqRef.current) {
        setLoading(false);
      }
    }
  }, [globalSearch, filters, activeStatusTab, sortBy, sortDir, pageSize, mergeContactInfoFromRows]);

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

  // Store fetchCandidates and page in refs to avoid dependency cycles in the sync interval
  const fetchRef = useRef(fetchCandidates);
  const pageRef = useRef(page);
  useEffect(() => {
    fetchRef.current = fetchCandidates;
    pageRef.current = page;
  }, [fetchCandidates, page]);

  useEffect(() => {
    let cancelled = false;

    const syncTalentPoolReplies = async () => {
      const res = await syncOutreachResponses(0);
      const updatedCount = Number(res?.data?.updated_count || 0);
      if (!cancelled && res?.success && updatedCount > 0) {
        // Fetch candidates for the current page only if there were updates
        fetchRef.current(pageRef.current);
      }
    };

    // Run once initially, then stagger
    syncTalentPoolReplies();
    const interval = setInterval(syncTalentPoolReplies, 15000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [syncOutreachResponses]);

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
    setActiveStatusTab('');
    setPage(1);
  };

  const hasFilters = globalSearch ||
    filters.title || filters.company || filters.city || filters.product_service || filters.status ||
    filters.min_exp !== 0 || filters.max_exp !== 40 ||
    activeStatusTab;

  // Status tabs: All + each status
  const statusTabs = ['', ...Object.keys(statusCounts)];

  const cols = [
    { key: 'selection', label: '', w: 40 },
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

  const toggleSelectAll = () => {
    if (selectedIds.size === candidates.length && candidates.length > 0) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(candidates.map(c => c.id)));
    }
  };

  const toggleSelectOne = (id) => {
    setSelectedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const handleSort = (sortKey) => {
    if (!sortKey) return;
    if (sortBy === sortKey) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortBy(sortKey); setSortDir('asc'); }
  };

  return (
    <div style={{ fontFamily: '"Inter", -apple-system, sans-serif', display: 'flex', gap: 0, height: '100vh', overflow: 'hidden' }}>

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

      </aside>

      {/* ── Main Content ── */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', background: '#f8fafc' }}>

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
                    onClick={() => col.key === 'selection' ? toggleSelectAll() : handleSort(col.sortKey)}
                    style={{
                      padding: '11px 14px', textAlign: 'left',
                      fontSize: 11, fontWeight: 700, color: '#94a3b8',
                      textTransform: 'uppercase', letterSpacing: '0.06em',
                      minWidth: col.w, cursor: (col.sortKey || col.key === 'selection') ? 'pointer' : 'default',
                      whiteSpace: 'nowrap', userSelect: 'none',
                      borderBottom: '1.5px solid #f1f5f9',
                      borderRight: '1.5px solid #f1f5f9',
                      borderLeft: index === 0 ? '1.5px solid #f1f5f9' : 'none'
                    }}
                  >
                    {col.key === 'selection' ? (
                      <input type="checkbox" checked={selectedIds.size === candidates.length && candidates.length > 0} readOnly style={{ cursor: 'pointer' }} />
                    ) : (
                      <>
                        {col.label}
                        {col.sortKey && sortBy === col.sortKey && (
                          <span style={{ marginLeft: 4 }}>{sortDir === 'asc' ? '↑' : '↓'}</span>
                        )}
                      </>
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
                  onClick={() => toggleSelectOne(c.id)}
                  style={{ borderBottom: '1px solid #f8fafc', transition: 'background 0.1s', cursor: 'pointer', background: selectedIds.has(c.id) ? '#fff7ed' : 'transparent' }}
                  onMouseEnter={e => { if (!selectedIds.has(c.id)) e.currentTarget.style.background = '#fafafa'; }}
                  onMouseLeave={e => { if (!selectedIds.has(c.id)) e.currentTarget.style.background = 'transparent'; }}
                >
                  <td style={{ padding: '13px 14px', textAlign: 'center', borderRight: '1px solid #f1f5f9', borderLeft: '1px solid #f1f5f9' }}>
                    <input type="checkbox" checked={selectedIds.has(c.id)} readOnly style={{ cursor: 'pointer' }} />
                  </td>
                  <td style={{ padding: '13px 14px', fontSize: 13, color: '#0f172a', borderRight: '1px solid #f1f5f9' }}>{c.first_name || ''}</td>
                  <td style={{ padding: '13px 14px', fontSize: 13, color: '#374151', borderRight: '1px solid #f1f5f9' }}>{c.last_name || ''}</td>
                  <td style={{ padding: '13px 14px', fontSize: 13, color: '#374151', maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', borderRight: '1px solid #f1f5f9' }}>{c.title || c.headline || ''}</td>
                  <td style={{ padding: '13px 14px', borderRight: '1px solid #f1f5f9' }}>
                    {c.linkedin
                      ? <a href={c.linkedin} target="_blank" rel="noreferrer" style={{ color: '#2563eb', display: 'flex', alignItems: 'center' }} onClick={e => e.stopPropagation()}>
                        <ExternalLink size={14} />
                      </a>
                      : <span style={{ color: '#94a3b8', fontSize: 11, fontWeight: 500 }}>NA</span>}
                  </td>
                  <td style={{ padding: '13px 14px', fontSize: 13, color: '#0f172a', borderRight: '1px solid #f1f5f9' }}>{c.company || ''}</td>
                  <td style={{ padding: '13px 14px', fontSize: 12, color: '#64748b', borderRight: '1px solid #f1f5f9' }}>{c.product_service || ''}</td>
                  <td style={{ padding: '13px 14px', fontSize: 13, color: '#374151', borderRight: '1px solid #f1f5f9' }}>{c.city || ''}</td>
                  <td style={{ padding: '13px 14px', fontSize: 12, color: '#64748b', borderRight: '1px solid #f1f5f9' }}>{c.location_type || ''}</td>
                  <td style={{ padding: '13px 14px', borderRight: '1px solid #f1f5f9' }}>
                    <ExpBar value={c.total_experience_years || 0} />
                  </td>
                  <td style={{ padding: '13px 14px', fontSize: 13, color: '#374151', borderRight: '1px solid #f1f5f9' }}>
                    {c.avg_tenure_years > 0 ? `${c.avg_tenure_years}y` : ''}
                  </td>
                  {(() => {
                    const emailVal = contactInfo[c.id]?.email || c.email;
                    const isShortlisted = contactInfo[c.id]?.enriching === false || c.enrichment_finished || ['Shortlisted', 'Followup / In conversation', 'In Conversation', 'Reached out - Linkedin', 'Reached out - Phone', 'Not Interested', 'Shared with customer'].includes(c.status);
                    const isEnriching = contactInfo[c.id]?.enriching && !emailVal;
                    const isNA = !emailVal || ['na', 'n/a', 'none'].includes((emailVal || '').toLowerCase());

                    if (isEnriching) {
                      return <td style={{ padding: '13px 14px', fontSize: 12, color: '#f97316', borderRight: '1px solid #f1f5f9', fontStyle: 'italic' }}>Fetching...</td>;
                    }
                    if (!isShortlisted) {
                      return <td style={{ padding: '13px 14px', fontSize: 12, color: '#d1d5db', borderRight: '1px solid #f1f5f9' }}>—</td>;
                    }
                    if (!isNA) {
                      return <td style={{ padding: '13px 14px', fontSize: 12, color: '#334155', borderRight: '1px solid #f1f5f9' }}>{emailVal}</td>;
                    }
                    return (
                      <ClickableEditableCell
                        id={c.id}
                        field="email"
                        value={emailVal}
                        onUpdate={updateFieldAndMaybeShortlist}
                        placeholder="NA"
                      />
                    );
                  })()}
                  {(() => {
                    const phoneVal = contactInfo[c.id]?.phone || c.mobile_phone;
                    const isShortlisted = contactInfo[c.id]?.enriching === false || c.enrichment_finished || ['Shortlisted', 'Followup / In conversation', 'In Conversation', 'Reached out - Linkedin', 'Reached out - Phone', 'Not Interested', 'Shared with customer'].includes(c.status);
                    const isEnriching = contactInfo[c.id]?.enriching && !phoneVal;
                    const isNA = !phoneVal || ['na', 'n/a', 'none'].includes((phoneVal || '').toLowerCase());

                    if (isEnriching) {
                      return <td style={{ padding: '13px 14px', fontSize: 12, color: '#f97316', borderRight: '1px solid #f1f5f9', fontStyle: 'italic' }}>Fetching...</td>;
                    }
                    if (!isShortlisted) {
                      return <td style={{ padding: '13px 14px', fontSize: 12, color: '#d1d5db', borderRight: '1px solid #f1f5f9' }}>—</td>;
                    }
                    if (!isNA) {
                      return <td style={{ padding: '13px 14px', fontSize: 12, color: '#334155', borderRight: '1px solid #f1f5f9' }}>{phoneVal}</td>;
                    }
                    return (
                      <ClickableEditableCell
                        id={c.id}
                        field="phone"
                        value={phoneVal}
                        onUpdate={updateFieldAndMaybeShortlist}
                        placeholder="NA"
                      />
                    );
                  })()}
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
                    onClick={(e) => { e.stopPropagation(); setSelectedCandidateForChat(c); }}
                    style={{
                      padding: '13px 14px', borderRight: '1px solid #f1f5f9',
                      cursor: 'pointer'
                    }}
                  >
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                      {/* Email response preview */}
                      <div style={{
                        fontSize: 12, color: c.response ? '#2563eb' : '#94a3b8',
                        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 130,
                        textDecoration: c.response ? 'underline' : 'none'
                      }}>
                        {c.response || 'View Chat'}
                      </div>
                      {/* LinkedIn response badge */}
                      {c.li_status && (
                        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                          <Linkedin size={11} color={c.li_status === 'replied' ? '#0077b5' : '#94a3b8'} />
                          <span style={{
                            fontSize: 10, fontWeight: 700,
                            color: c.li_status === 'replied' ? '#0077b5'
                              : c.li_status === 'message_sent' ? '#7c3aed'
                                : c.li_status === 'connection_accepted' ? '#15803d'
                                  : '#64748b',
                            textTransform: 'uppercase', letterSpacing: '0.03em'
                          }}>
                            {c.li_status === 'replied' ? 'LI Replied ✓'
                              : c.li_status === 'message_sent' ? 'LI Msg Sent'
                                : c.li_status === 'connection_accepted' ? 'LI Connected'
                                  : c.li_status === 'in_campaign' ? 'LI In Campaign'
                                    : c.li_status === 'connection_sent' ? 'LI Req Sent'
                                      : c.li_status}
                          </span>
                        </div>
                      )}
                      {c.li_response_text && (
                        <div style={{
                          fontSize: 11, color: '#475569', fontStyle: 'italic',
                          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 130
                        }}>
                          "{c.li_response_text}"
                        </div>
                      )}
                    </div>
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
                        fontSize: '11px'
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

      {showAddToListModal && (
        <AddToListModal
          selectedCount={selectedIds.size}
          onClose={() => setShowAddToListModal(false)}
          onSuccess={() => {
            setSelectedIds(new Set());
            setShowAddToListModal(false);
          }}
          candidateIds={Array.from(selectedIds)}
        />
      )}

      {selectedIds.size > 0 && (
        <div style={{
          position: 'fixed', bottom: 30, left: '50%', transform: 'translateX(-50%)',
          background: '#0f172a', color: '#fff', padding: '12px 24px', borderRadius: '16px',
          display: 'flex', alignItems: 'center', gap: 20, boxShadow: '0 20px 25px -5px rgba(0,0,0,0.3)',
          zIndex: 1000, border: '1px solid rgba(255,255,255,0.1)', animation: 'slideUp 0.3s ease-out'
        }}>
          <span style={{ fontSize: 14, fontWeight: 600 }}>{selectedIds.size} candidates selected</span>
          <div style={{ width: 1, height: 20, background: 'rgba(255,255,255,0.2)' }} />
          <button
            onClick={() => setShowAddToListModal(true)}
            style={{
              padding: '8px 16px', background: '#f97316', color: '#fff', border: 'none',
              borderRadius: '10px', fontSize: 13, fontWeight: 700, cursor: 'pointer',
              display: 'flex', alignItems: 'center', gap: 8, transition: 'all 0.2s'
            }}
            onMouseEnter={e => e.currentTarget.style.background = '#ea580c'}
            onMouseLeave={e => e.currentTarget.style.background = '#f97316'}
          >
            <Phone size={14} /> Add to Call List
          </button>
          <button
            onClick={() => setSelectedIds(new Set())}
            style={{ background: 'none', border: 'none', color: '#94a3b8', fontSize: 13, fontWeight: 600, cursor: 'pointer' }}
          >
            Cancel
          </button>
        </div>
      )}

      <style>{`
        @keyframes slideUp {
          from { transform: translate(-50%, 100%); opacity: 0; }
          to { transform: translate(-50%, 0); opacity: 1; }
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        @keyframes msgFadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
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

function AddToListModal({ selectedCount, onClose, onSuccess, candidateIds }) {
  const { callLists, fetchCallLists, createCallList, addCandidatesToCallList } = useAppStore();
  const [loading, setLoading] = useState(false);
  const [newListName, setNewListName] = useState('');
  const [selectedListId, setSelectedListId] = useState('');
  const [mode, setMode] = useState('select'); // 'select' or 'create'

  useEffect(() => {
    fetchCallLists();
  }, [fetchCallLists]);

  const handleAction = async () => {
    setLoading(true);
    try {
      let listId = selectedListId;
      if (mode === 'create') {
        if (!newListName.trim()) return;
        const res = await createCallList(newListName);
        if (res.success) listId = res.data.id;
        else throw new Error(res.error);
      }

      if (!listId) return;
      const res = await addCandidatesToCallList(candidateIds, listId);
      if (res.success) {
        toast.success(`Added ${selectedCount} candidates to call list`);
        onSuccess();
      } else {
        toast.error(res.error);
      }
    } catch (e) {
      toast.error(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(15, 23, 42, 0.7)', backdropFilter: 'blur(4px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 10000 }}>
      <div style={{ background: '#fff', borderRadius: '24px', width: '100%', maxWidth: '440px', padding: '32px', boxShadow: '0 25px 50px -12px rgba(0,0,0,0.25)', border: '1px solid #e2e8f0' }}>
        <h3 style={{ fontSize: '20px', fontWeight: 800, color: '#0f172a', marginBottom: '8px' }}>Add to Call List</h3>
        <p style={{ color: '#64748b', fontSize: '14px', marginBottom: '24px' }}>Choose a list to add {selectedCount} candidates to.</p>

        <div style={{ display: 'flex', gap: 8, marginBottom: 20, padding: 4, background: '#f8fafc', borderRadius: 12 }}>
          <button
            onClick={() => setMode('select')}
            style={{
              flex: 1, padding: '8px', border: 'none', borderRadius: 8, fontSize: 13, fontWeight: 700,
              background: mode === 'select' ? '#fff' : 'transparent',
              color: mode === 'select' ? '#0f172a' : '#64748b',
              boxShadow: mode === 'select' ? '0 1px 3px rgba(0,0,0,0.1)' : 'none',
              cursor: 'pointer'
            }}
          >Existing List</button>
          <button
            onClick={() => setMode('create')}
            style={{
              flex: 1, padding: '8px', border: 'none', borderRadius: 8, fontSize: 13, fontWeight: 700,
              background: mode === 'create' ? '#fff' : 'transparent',
              color: mode === 'create' ? '#0f172a' : '#64748b',
              boxShadow: mode === 'create' ? '0 1px 3px rgba(0,0,0,0.1)' : 'none',
              cursor: 'pointer'
            }}
          >+ New List</button>
        </div>

        {mode === 'select' ? (
          <select
            value={selectedListId}
            onChange={e => setSelectedListId(e.target.value)}
            style={{
              width: '100%', padding: '12px 16px', borderRadius: 12, border: '1.5px solid #e2e8f0',
              fontSize: 14, outline: 'none', marginBottom: 24, background: '#fff'
            }}
          >
            <option value="">Select a list...</option>
            {callLists.map(l => (
              <option key={l.id} value={l.id}>{l.name} ({l.candidate_count} candidates)</option>
            ))}
          </select>
        ) : (
          <input
            type="text"
            placeholder="List name (e.g. Frontend Devs Today)"
            value={newListName}
            onChange={e => setNewListName(e.target.value)}
            style={{
              width: '100%', padding: '12px 16px', borderRadius: 12, border: '1.5px solid #e2e8f0',
              fontSize: 14, outline: 'none', marginBottom: 24, boxSizing: 'border-box'
            }}
          />
        )}

        <div style={{ display: 'flex', gap: 12 }}>
          <button
            onClick={onClose}
            style={{ flex: 1, padding: '14px', background: '#f1f5f9', color: '#475569', border: 'none', borderRadius: 12, fontWeight: 700, cursor: 'pointer' }}
          >Cancel</button>
          <button
            onClick={handleAction}
            disabled={loading || (mode === 'select' && !selectedListId) || (mode === 'create' && !newListName.trim())}
            style={{
              flex: 1, padding: '14px', background: '#f97316', color: '#fff', border: 'none', borderRadius: 12,
              fontWeight: 700, cursor: (loading || (mode === 'select' && !selectedListId) || (mode === 'create' && !newListName.trim())) ? 'not-allowed' : 'pointer',
              opacity: (loading || (mode === 'select' && !selectedListId) || (mode === 'create' && !newListName.trim())) ? 0.6 : 1
            }}
          >
            {loading ? 'Adding...' : 'Add Candidates'}
          </button>
        </div>
      </div>
    </div>
  );
}
