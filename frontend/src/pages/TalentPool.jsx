import React, { startTransition, useState, useEffect, useCallback, useRef } from 'react';
import { useAppStore, API_BASE } from '../store/useAppStore';
import axios from 'axios';
import { toast } from 'sonner';
import { useShallow } from 'zustand/react/shallow';
import {
  Search, ExternalLink, ChevronLeft, ChevronRight, Filter,
  User, Building2, MapPin, Briefcase, BarChart2,
  SlidersHorizontal, RefreshCw, UserPlus, X, ChevronDown,
  Activity, MessageSquareMore, Users, Plus, Edit2, Check, Download,
  Mail, Phone, MessageSquare, Linkedin, Send
} from 'lucide-react';
import StatusDropdown, { RECRUITMENT_STAGES, STATUS_STYLES } from '../components/StatusDropdown';

// ── Clickable Editable Cell (renders as <td>) ────────────────
// Always editable — click to enter edit mode with existing value pre-filled.
// Pressing Enter or blurring saves. Clearing the value saves empty → renders as NA.
function ClickableEditableCell({ id, field, value, onUpdate, placeholder = 'Not available' }) {
  const [isEditing, setIsEditing] = useState(false);
  const [tempValue, setTempValue] = useState('');
  const [displayValue, setDisplayValue] = useState(value || '');
  const [loading, setLoading] = useState(false);

  // Keep display in sync when parent value changes (e.g. after API refresh)
  useEffect(() => {
    if (!isEditing) setDisplayValue(value || '');
  }, [value, isEditing]);

  const isNA = !displayValue || ['na', 'n/a', 'none'].includes(displayValue.toString().toLowerCase());

  const handleSave = async () => {
    const newVal = tempValue.trim();
    // Always persist — even if unchanged — so the user can intentionally overwrite
    setLoading(true);
    await onUpdate(id, { [field]: newVal });
    setDisplayValue(newVal); // optimistic local update
    setLoading(false);
    setIsEditing(false);
  };

  const startEditing = (e) => {
    e.stopPropagation(); // Don't bubble to row
    setTempValue(displayValue); // Pre-fill with current value so user can edit / overwrite
    setIsEditing(true);
  };

  if (isEditing) {
    return (
      <td
        style={{ padding: '0 14px', borderRight: '1px solid #eef2f7' }}
        onClick={(e) => e.stopPropagation()}
      >
        <input
          autoFocus
          value={tempValue}
          onChange={(e) => setTempValue(e.target.value)}
          onBlur={handleSave}
          onKeyDown={(e) => {
            if (e.key === 'Enter') { e.preventDefault(); handleSave(); }
            if (e.key === 'Escape') { setIsEditing(false); }
          }}
          style={{
            width: '100%', padding: '6px 8px', border: '1px solid rgba(203, 213, 225, 0.9)',
            borderRadius: '8px', fontSize: '12px', fontFamily: 'inherit', outline: 'none',
            background: '#fff', color: '#111827',
          }}
        />
      </td>
    );
  }

  return (
    <td
      onClick={startEditing}
      style={{
        padding: '13px 14px', borderRight: '1px solid #eef2f7',
        fontSize: '12px', cursor: 'text',
        color: isNA ? '#94a3b8' : '#334155',
      }}
      title="Click to edit"
    >
      <span style={{
        padding: isNA ? '2px 6px' : '0',
        borderRadius: '4px',
        background: isNA ? '#f8fafc' : 'transparent',
        border: isNA ? '1px solid #e2e8f0' : 'none',
        display: 'inline-block',
        fontStyle: isNA ? 'italic' : 'normal',
        fontWeight: isNA ? 500 : 400,
      }}>
        {isNA ? placeholder : displayValue}
        {loading && ' ⟳'}
      </span>
    </td>
  );
}

function TableSkeleton({ rows = 10 }) {
  return (
    <>
      {Array.from({ length: rows }).map((_, i) => (
        <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
          <td style={{ padding: '16px 20px' }}>
            <div className="shimmer skeleton-row" style={{ width: '100%', borderRadius: '4px' }} />
          </td>
          <td style={{ padding: '16px 20px' }}>
            <div className="shimmer skeleton-row" style={{ width: '80%', borderRadius: '4px' }} />
          </td>
          <td style={{ padding: '16px 20px' }}>
            <div className="shimmer skeleton-row" style={{ width: '60%', borderRadius: '4px' }} />
          </td>
          <td style={{ padding: '16px 20px' }}>
            <div className="shimmer skeleton-row" style={{ width: '70%', borderRadius: '4px' }} />
          </td>
          <td style={{ padding: '16px 20px' }}>
            <div className="shimmer skeleton-row" style={{ width: '90%', borderRadius: '4px' }} />
          </td>
        </tr>
      ))}
    </>
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
      <div style={{ width: 44, height: 5, background: '#dbe1e8', borderRadius: 999, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: '#0f172a', borderRadius: 999 }} />
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
      <label style={{ display: 'block', fontSize: 11, fontWeight: 700, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 7 }}>
        {label}
      </label>
      <div style={{ position: 'relative', display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <div style={{ position: 'relative' }}>
          {Icon && (
            <span style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', color: '#9ca3af', display: 'flex' }}>
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
              width: '100%', padding: Icon ? '10px 12px 10px 34px' : '10px 12px',
              background: 'rgba(255,255,255,0.92)', border: '1px solid rgba(203, 213, 225, 0.9)',
              borderRadius: 12, color: '#111827', fontSize: 13,
              outline: 'none', transition: 'border-color 0.15s, box-shadow 0.15s, background 0.15s',
              fontFamily: 'inherit', boxSizing: 'border-box',
              boxShadow: 'inset 0 1px 2px rgba(15,23,42,0.03)',
            }}
            onFocus={e => {
              e.target.style.borderColor = 'rgba(194, 124, 63, 0.5)';
              e.target.style.boxShadow = '0 0 0 3px rgba(194, 124, 63, 0.12)';
              e.target.style.background = '#fff';
            }}
            onBlur={e => {
              e.target.style.borderColor = 'rgba(203, 213, 225, 0.9)';
              e.target.style.boxShadow = 'inset 0 1px 2px rgba(15,23,42,0.03)';
              e.target.style.background = 'rgba(255,255,255,0.92)';
            }}
          />
        </div>
        {values.length > 0 && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {values.map(tag => (
              <span key={tag} style={{
                display: 'inline-flex', alignItems: 'center', gap: '6px',
                background: '#eef2f6', color: '#334155', padding: '4px 9px',
                borderRadius: '999px', fontSize: '11px', fontWeight: 600,
                border: '1px solid rgba(203, 213, 225, 0.9)'
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
      <label style={{ display: 'block', fontSize: 11, fontWeight: 700, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 7 }}>
        {label}
      </label>
      <div style={{ position: 'relative' }}>
        <select
          value={value}
          onChange={e => onChange(e.target.value)}
          style={{
            width: '100%', padding: '10px 32px 10px 12px',
            background: 'rgba(255,255,255,0.92)', border: '1px solid rgba(203, 213, 225, 0.9)',
            borderRadius: 12, color: value ? '#111827' : '#94a3b8', fontSize: 13,
            outline: 'none', appearance: 'none', cursor: 'pointer',
            fontFamily: 'inherit', boxSizing: 'border-box',
            boxShadow: 'inset 0 1px 2px rgba(15,23,42,0.03)',
            transition: 'border-color 0.15s, box-shadow 0.15s, background 0.15s',
          }}
          onFocus={e => {
            e.target.style.borderColor = 'rgba(194, 124, 63, 0.5)';
            e.target.style.boxShadow = '0 0 0 3px rgba(194, 124, 63, 0.12)';
            e.target.style.background = '#fff';
          }}
          onBlur={e => {
            e.target.style.borderColor = 'rgba(203, 213, 225, 0.9)';
            e.target.style.boxShadow = 'inset 0 1px 2px rgba(15,23,42,0.03)';
            e.target.style.background = 'rgba(255,255,255,0.92)';
          }}
        >
          <option value="">{placeholder}</option>
          {options.map(o => <option key={o} value={o}>{o}</option>)}
        </select>
        <ChevronDown size={13} color="#94a3b8" style={{ position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none' }} />
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
        <label style={{ fontSize: 11, fontWeight: 700, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
          {label}
        </label>
        <span style={{ fontSize: 11, fontWeight: 700, color: '#64748b' }}>
          {minValue} - {maxValue} yrs
        </span>
      </div>

      <div style={{ position: 'relative', height: 20, display: 'flex', alignItems: 'center', padding: '0 4px' }}>
        {/* Track Background */}
        <div style={{ position: 'absolute', left: 4, right: 4, height: 4, background: '#d8dee8', borderRadius: 999 }} />

        {/* Active Range Highlight */}
        <div style={{
          position: 'absolute',
          left: `calc(4px + ${minPos}%)`,
          width: `${maxPos - minPos}%`,
          height: 4,
          background: '#0f172a',
          borderRadius: 999,
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
            background: #0f172a;
            border: 2px solid #fff;
            box-shadow: 0 6px 12px rgba(15,23,42,0.18);
            cursor: pointer;
            z-index: 10;
          }
          .dual-range-input::-moz-range-thumb {
            appearance: none;
            pointer-events: auto;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #0f172a;
            border: 2px solid #fff;
            box-shadow: 0 6px 12px rgba(15,23,42,0.18);
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

function normalizeTalentPoolText(value) {
  return String(value || '').trim().toLowerCase();
}

function normalizeContactValue(value) {
  const text = String(value || '').trim();
  if (!text) return '';
  const normalized = text.toLowerCase();
  if (['na', 'n/a', 'none', 'null', 'undefined'].includes(normalized)) {
    return '';
  }
  return text;
}

function resolveContactValue(...values) {
  for (const value of values) {
    const normalized = normalizeContactValue(value);
    if (normalized) return normalized;
  }
  return '';
}

function splitTalentPoolFilterValues(values = [], inputValue = '') {
  return [...(values || []), inputValue]
    .flatMap((value) => String(value || '').split(','))
    .map((value) => value.trim())
    .filter(Boolean);
}

function matchesTalentPoolFilters(filterValues, targetValue) {
  if (!filterValues.length) return true;
  const normalizedTarget = normalizeTalentPoolText(targetValue);
  if (!normalizedTarget) return false;
  return filterValues.some((value) => normalizedTarget.includes(normalizeTalentPoolText(value)));
}

function buildTalentPoolParamsString({
  page = 1,
  pageSize = 25,
  globalSearch = '',
  filters = {},
  activeStatusTab = '',
  sortBy = 'name',
  sortDir = 'asc',
}) {
  const params = new URLSearchParams();
  params.set('page', page);
  params.set('page_size', pageSize || 25);

  if (globalSearch) params.set('q', globalSearch);

  const titleSearch = splitTalentPoolFilterValues(filters?.title, filters?.titleInput).join(',');
  if (titleSearch) params.set('title', titleSearch);

  const companySearch = splitTalentPoolFilterValues(filters?.company, filters?.companyInput).join(',');
  if (companySearch) params.set('company', companySearch);

  const citySearch = splitTalentPoolFilterValues(filters?.city, filters?.cityInput).join(',');
  if (citySearch) params.set('city', citySearch);

  const productSearch = splitTalentPoolFilterValues(filters?.product_service, filters?.productInput).join(',');
  if (productSearch) params.set('product_service', productSearch);

  if (activeStatusTab) params.set('status', activeStatusTab);
  else if (filters?.status) params.set('status', filters.status);

  if (filters?.min_exp !== undefined && filters?.min_exp !== '') params.set('min_exp', filters.min_exp);
  if (filters?.max_exp !== undefined && filters?.max_exp !== '') params.set('max_exp', filters.max_exp);
  if (filters?.created_by) params.set('created_by', filters.created_by);

  params.set('sort_by', sortBy);
  params.set('sort_dir', sortDir);

  return params.toString();
}

function buildLocalTalentPoolView(rows = [], {
  globalSearch = '',
  filters = {},
  activeStatusTab = '',
  sortBy = 'name',
  sortDir = 'asc',
  page = 1,
  pageSize = 25,
}) {
  const titleFilters = splitTalentPoolFilterValues(filters?.title, filters?.titleInput);
  const companyFilters = splitTalentPoolFilterValues(filters?.company, filters?.companyInput);
  const cityFilters = splitTalentPoolFilterValues(filters?.city, filters?.cityInput);
  const productFilters = splitTalentPoolFilterValues(filters?.product_service, filters?.productInput);
  const recruiterFilters = splitTalentPoolFilterValues([], filters?.created_by);
  const normalizedSearch = normalizeTalentPoolText(globalSearch);
  const effectiveStatus = activeStatusTab || filters?.status || '';
  const normalizedStatus = normalizeTalentPoolText(effectiveStatus);
  const minExp = Number(filters?.min_exp ?? 0);
  const maxExp = Number(filters?.max_exp ?? 40);

  const filteredRows = rows.filter((row) => {
    const title = row?.title || row?.headline || '';
    const company = row?.company || '';
    const city = row?.city || '';
    const product = row?.product_service || '';
    const recruiter = row?.created_by || '';
    const status = row?.status || 'To be started';
    const totalExperience = Number(row?.total_experience_years || 0);

    if (normalizedSearch) {
      const searchable = [
        row?.name,
        row?.first_name,
        row?.last_name,
        title,
        company,
        city,
        product,
        recruiter,
        row?.email,
        row?.phone,
        row?.mobile_phone,
        status,
      ].map(normalizeTalentPoolText).join(' ');

      if (!searchable.includes(normalizedSearch)) return false;
    }

    if (!matchesTalentPoolFilters(titleFilters, title)) return false;
    if (!matchesTalentPoolFilters(companyFilters, company)) return false;
    if (!matchesTalentPoolFilters(cityFilters, city)) return false;
    if (!matchesTalentPoolFilters(recruiterFilters, recruiter)) return false;
    if (productFilters.length && !matchesTalentPoolFilters(productFilters, product)) return false;
    if (totalExperience < minExp || totalExperience > maxExp) return false;

    return true;
  });

  const statusCounts = {};
  for (const row of filteredRows) {
    const status = (row?.status || 'To be started').trim();
    statusCounts[status] = (statusCounts[status] || 0) + 1;
  }

  let finalRows = filteredRows;
  if (normalizedStatus) {
    finalRows = filteredRows.filter((row) => normalizeTalentPoolText(row?.status || 'To be started') === normalizedStatus);
  }

  const getSortValue = (row) => {
    switch (sortBy) {
      case 'title':
        return row?.title || row?.headline || '';
      case 'company':
        return row?.company || '';
      case 'city':
        return row?.city || '';
      case 'exp':
        return Number(row?.total_experience_years || 0);
      case 'tenure':
        return Number(row?.avg_tenure_years || 0);
      case 'name':
      default:
        return row?.name || `${row?.first_name || ''} ${row?.last_name || ''}`.trim();
    }
  };

  finalRows = [...finalRows].sort((a, b) => {
    const left = getSortValue(a);
    const right = getSortValue(b);

    if (typeof left === 'number' || typeof right === 'number') {
      return sortDir === 'desc' ? Number(right) - Number(left) : Number(left) - Number(right);
    }

    const result = String(left).localeCompare(String(right), undefined, { sensitivity: 'base' });
    return sortDir === 'desc' ? -result : result;
  });

  const total = finalRows.length;
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  const safePage = Math.min(page, totalPages);
  const start = (safePage - 1) * pageSize;

  return {
    candidates: finalRows.slice(start, start + pageSize),
    total,
    totalPages,
    statusCounts,
  };
}

const readPersistedContactInfo = () => {
  if (typeof window === 'undefined') return {};
  try {
    const raw = window.localStorage.getItem(CONTACT_INFO_STORAGE_KEY)
      || window.sessionStorage.getItem(CONTACT_INFO_STORAGE_KEY);
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
    { label: 'Total sourced', value: summary.total_sourced, icon: UserPlus, tone: 'warm' },
    { label: 'Shortlisted', value: summary.shortlisted, icon: Activity, tone: 'emerald', status: 'Shortlisted' },
    { label: 'Conversion rate', value: `${conversionRate}%`, icon: BarChart2, tone: 'slate' },
    { label: 'In follow up', value: summary.pipeline_health?.['Followup / In conversation'] || 0, icon: MessageSquareMore, tone: 'amber', status: 'Followup / In conversation' },
  ];

  const recruiterPerf = analytics.recruiter_performance || [];
  const tones = {
    warm: {
      accent: '#c27c3f',
      iconBg: 'rgba(194, 124, 63, 0.12)',
      cardBg: 'linear-gradient(180deg, #ffffff 0%, #fcf8f3 100%)',
      border: 'rgba(194, 124, 63, 0.18)',
    },
    emerald: {
      accent: '#2f855a',
      iconBg: 'rgba(47, 133, 90, 0.12)',
      cardBg: 'linear-gradient(180deg, #ffffff 0%, #f5fbf7 100%)',
      border: 'rgba(47, 133, 90, 0.16)',
    },
    slate: {
      accent: '#334155',
      iconBg: 'rgba(51, 65, 85, 0.12)',
      cardBg: 'linear-gradient(180deg, #ffffff 0%, #f8fafc 100%)',
      border: 'rgba(148, 163, 184, 0.2)',
    },
    amber: {
      accent: '#9a6b28',
      iconBg: 'rgba(154, 107, 40, 0.12)',
      cardBg: 'linear-gradient(180deg, #ffffff 0%, #fcf8f2 100%)',
      border: 'rgba(154, 107, 40, 0.16)',
    },
  };

  return (
    <div style={{ padding: 0 }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: '12px', marginBottom: role === 'admin' && recruiterPerf.length > 0 ? '16px' : '0' }}>
        {cards.map((card, i) => {
          const Icon = card.icon;
          const tone = tones[card.tone] || tones.slate;
          return (
            <div
              key={i}
              onClick={() => card.status && onStatClick(card.status)}
              style={{
                background: tone.cardBg,
                padding: '16px',
                borderRadius: '18px',
                border: `1px solid ${tone.border}`,
                cursor: card.status ? 'pointer' : 'default',
                transition: 'transform 0.2s ease, box-shadow 0.2s ease',
                display: 'flex',
                alignItems: 'center',
                gap: '14px',
                boxShadow: '0 12px 24px rgba(15,23,42,0.04)'
              }}
              onMouseEnter={e => card.status && (e.currentTarget.style.transform = 'translateY(-1px)', e.currentTarget.style.boxShadow = '0 16px 30px rgba(15,23,42,0.06)')}
              onMouseLeave={e => card.status && (e.currentTarget.style.transform = 'none', e.currentTarget.style.boxShadow = '0 12px 24px rgba(15,23,42,0.04)')}
            >
              <div style={{ width: 42, height: 42, borderRadius: '14px', background: tone.iconBg, display: 'flex', alignItems: 'center', justifyContent: 'center', color: tone.accent }}>
                <Icon size={20} />
              </div>
              <div style={{ minWidth: 0 }}>
                <div style={{ fontSize: '11px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '6px' }}>{card.label}</div>
                <div style={{ fontSize: '28px', lineHeight: 1, fontWeight: 800, color: '#0f172a', letterSpacing: '-0.03em' }}>{card.value?.toLocaleString?.() || card.value}</div>
              </div>
            </div>
          );
        })}
      </div>

      {role === 'admin' && recruiterPerf.length > 0 && (
        <div style={{ background: 'rgba(248, 250, 252, 0.82)', borderRadius: '18px', padding: '14px', border: '1px solid rgba(226, 232, 240, 0.9)' }}>
          <div style={{ fontSize: '12px', fontWeight: 800, color: '#0f172a', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Users size={14} color="#64748b" /> Recruiter performance
          </div>
          <div style={{ display: 'flex', gap: '12px', overflowX: 'auto', paddingBottom: '4px' }}>
            {recruiterPerf.map((perf, i) => (
              <div
                key={i}
                onClick={() => onRecruiterClick(perf.recruiter)}
                style={{
                  minWidth: '176px', background: '#fff', padding: '12px 14px', borderRadius: '14px', border: '1px solid rgba(226, 232, 240, 0.95)',
                  display: 'flex', flexDirection: 'column', gap: '4px', cursor: 'pointer', transition: 'all 0.15s'
                }}
                onMouseEnter={e => e.currentTarget.style.borderColor = 'rgba(194, 124, 63, 0.32)'}
                onMouseLeave={e => e.currentTarget.style.borderColor = 'rgba(226, 232, 240, 0.95)'}
              >
                <div style={{ fontSize: '13px', fontWeight: 700, color: '#0f172a', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{perf.recruiter}</div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '11px', color: '#64748b' }}>{perf.sourced} sourced</span>
                  <span style={{ fontSize: '11px', fontWeight: 700, color: '#334155' }}>{perf.conversion}% hit</span>
                </div>
                <div style={{ height: '4px', width: '100%', background: '#e2e8f0', borderRadius: '999px', overflow: 'hidden', marginTop: '4px' }}>
                  <div style={{ height: '100%', width: `${perf.conversion}%`, background: '#334155' }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Module-level SWR Chat Cache ─────────────────────────────────────────────
// Survives modal open/close so second open is always instant.
// Structure: { [candidateId]: { email: { messages, ts }, linkedin: { messages, ts } } }
const _chatCache = {};
const CHAT_CACHE_TTL_MS = 30_000; // 30 seconds — same as backend cache TTL
const OPTIMISTIC_MESSAGE_TTL_MS = 15 * 60 * 1000;

function _getCached(candidateId, platform) {
  const entry = _chatCache[candidateId]?.[platform];
  if (!entry) return null;
  if (Date.now() - entry.ts > CHAT_CACHE_TTL_MS) return null; // expired
  return entry.messages;
}

function _setCached(candidateId, platform, messages) {
  if (!_chatCache[candidateId]) _chatCache[candidateId] = {};
  _chatCache[candidateId][platform] = { messages, ts: Date.now() };
}
// ─────────────────────────────────────────────────────────────────────────────

function ConversationModal({ candidate, onClose }) {
  // Auto-select LinkedIn tab if candidate has an active LinkedIn response
  const hasLiResponse = Boolean(candidate?.li_response_text);
  const defaultPlatform = (hasLiResponse || candidate?.li_status === 'replied') ? 'linkedin' : 'email';
  const [platform, setPlatform] = useState(defaultPlatform);

  // ── SWR Thread State ───────────────────────────────────────────────────────
  // Seed from module-level cache for instant render on re-open
  const seedThread = (p) => {
    const cached = _getCached(candidate.id, p);
    return cached ? { messages: cached, loaded: true, error: '' } : { messages: [], loaded: false, error: '' };
  };
  const [threads, setThreads] = useState(() => ({
    email: seedThread('email'),
    linkedin: seedThread('linkedin'),
  }));
  // loading = active tab has no messages yet; refreshing = manual sync in progress
  const [loading, setLoading] = useState(!_getCached(candidate.id, defaultPlatform));
  const [refreshing, setRefreshing] = useState(false);
  const [syncingByPlatform, setSyncingByPlatform] = useState({ email: false, linkedin: false });
  const [replyText, setReplyText] = useState('');
  const [sending, setSending] = useState(false);
  const [localSentMessagesByPlatform, setLocalSentMessagesByPlatform] = useState({ email: [], linkedin: [] });

  const fetchChatHistory = useAppStore(state => state.fetchChatHistory);
  const sendChatReply = useAppStore(state => state.sendChatReply);
  const { heyreachCampaignId, triggerHeyReachOutreach } = useAppStore(useShallow((state) => ({
    heyreachCampaignId: state.heyreachCampaignId,
    triggerHeyReachOutreach: state.triggerHeyReachOutreach,
  })));
  const [isTriggering, setIsTriggering] = useState(false);
  const [hasTriggered, setHasTriggered] = useState(false);
  const messagesEndRef = useRef(null);
  const threadsRef = useRef(threads);
  const requestSeqRef = useRef({ email: 0, linkedin: 0 });
  const candidateIdRef = useRef(candidate.id);
  const activePlatformRef = useRef(defaultPlatform);

  useEffect(() => { threadsRef.current = threads; }, [threads]);
  useEffect(() => { candidateIdRef.current = candidate.id; }, [candidate.id]);
  useEffect(() => { activePlatformRef.current = platform; }, [platform]);

  const activeThread = threads[platform] || { messages: [], loaded: false, error: '' };
  const activeThreadSyncing = Boolean(syncingByPlatform[platform]);
  const messages = activeThread.messages || [];
  const localSentMessages = localSentMessagesByPlatform[platform] || [];
  const conversationAccent = '#111827';
  const conversationAccentMuted = '#64748b';
  const conversationBorder = 'rgba(203, 213, 225, 0.92)';
  const platformTabs = [
    { id: 'linkedin', label: 'LinkedIn', icon: Linkedin },
    { id: 'email', label: 'Email', icon: Mail }
  ];
  const normalizeMessageBody = (value) => (value || '').replace(/\s+/g, ' ').trim().toLowerCase();
  const getMessageTimestamp = (msg) => {
    const value = new Date(msg?.time || msg?.created_at || msg?.timestamp || 0).getTime();
    return Number.isFinite(value) ? value : 0;
  };

  // ── Core fetch function ────────────────────────────────────────────────────
  const loadMessages = useCallback(async ({
    silent = false,
    targetPlatform = activePlatformRef.current,
    force = false
  } = {}) => {
    const candidateId = candidate.id;
    const resolvedPlatform = targetPlatform || activePlatformRef.current;
    const requestSeq = ++requestSeqRef.current[resolvedPlatform];
    const isActiveTab = resolvedPlatform === activePlatformRef.current;

    // Show loader only on active tab when no messages cached
    if (isActiveTab && !silent && !threadsRef.current[resolvedPlatform]?.loaded) {
      setLoading(true);
    } else if (isActiveTab && !silent) {
      setRefreshing(true);
    }

    const res = await fetchChatHistory(0, candidateId, resolvedPlatform, force);

    // Discard stale responses (candidate changed or newer request fired)
    if (candidateIdRef.current !== candidateId || requestSeqRef.current[resolvedPlatform] !== requestSeq) return res;

    if (res.success) {
      const msgs = res.messages || [];
      _setCached(candidateId, resolvedPlatform, msgs); // persist in module cache
      setThreads(prev => ({
        ...prev,
        [resolvedPlatform]: { messages: msgs, loaded: true, error: '' }
      }));
      setSyncingByPlatform(prev => ({
        ...prev,
        [resolvedPlatform]: Boolean(res.syncing),
      }));
    } else {
      setThreads(prev => ({
        ...prev,
        [resolvedPlatform]: { ...prev[resolvedPlatform], loaded: true, error: res.error || 'Failed to load' }
      }));
      setSyncingByPlatform(prev => ({
        ...prev,
        [resolvedPlatform]: false,
      }));
    }

    if (isActiveTab) {
      setLoading(false);
      if (!silent) setRefreshing(false);
    }

    return res;
  }, [candidate.id, fetchChatHistory]);

  // ── On candidate change: reset state (but preserve module cache) ───────────
  useEffect(() => {
    const nextThreads = {
      email: seedThread('email'),
      linkedin: seedThread('linkedin'),
    };
    threadsRef.current = nextThreads;
    requestSeqRef.current = { email: 0, linkedin: 0 };
    setThreads(nextThreads);
    setPlatform(defaultPlatform);
    activePlatformRef.current = defaultPlatform;
    setSyncingByPlatform({ email: false, linkedin: false });
    setLocalSentMessagesByPlatform({ email: [], linkedin: [] });
    setLoading(!_getCached(candidate.id, defaultPlatform));
    setRefreshing(false);
    setReplyText('');
    setHasTriggered(false);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [candidate.id, defaultPlatform]);

  // ── On modal open: fetch active thread first, then prefetch the other tab ──
  useEffect(() => {
    const primaryPlatform = defaultPlatform;
    const secondaryPlatform = primaryPlatform === 'linkedin' ? 'email' : 'linkedin';

    void loadMessages({ targetPlatform: primaryPlatform, silent: false });
    void loadMessages({ targetPlatform: secondaryPlatform, silent: true });
  }, [candidate.id, defaultPlatform, loadMessages]);

  // ── Tab switch: if already loaded just show, else fetch ────────────────────
  useEffect(() => {
    setReplyText('');
    const cached = threadsRef.current[platform];
    if (!cached?.loaded) {
      setLoading(true);
      if (requestSeqRef.current[platform] === 0) {
        void loadMessages({ targetPlatform: platform });
      }
    } else {
      setLoading(false);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [platform]);

  // ── Fast poll whichever active thread is currently syncing ─────────────────
  useEffect(() => {
    if (!activeThreadSyncing) return;

    const interval = setInterval(() => {
      void loadMessages({ silent: true, targetPlatform: activePlatformRef.current });
    }, 2000);

    return () => clearInterval(interval);
  }, [activeThreadSyncing, loadMessages]);

  // ── Standard background poll — keep both tabs fresh and apply results ──────
  useEffect(() => {
    const interval = setInterval(() => {
      void loadMessages({ silent: true, targetPlatform: 'email' });
      void loadMessages({ silent: true, targetPlatform: 'linkedin' });
    }, 15000);
    return () => clearInterval(interval);
  }, [candidate.id, loadMessages]);

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
      setSyncingByPlatform(prev => ({
        ...prev,
        [activePlatform]: true,
      }));
      window.setTimeout(() => {
        void loadMessages({ silent: true, targetPlatform: activePlatform, force: true });
      }, 1500);
      window.setTimeout(() => {
        void loadMessages({ silent: true, targetPlatform: activePlatform, force: true });
      }, 5000);
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
      const msgKey = `${msg.type}-${normalizeMessageBody(msg.email_body)}-${getMessageTimestamp(msg)}`;
      if (!serverMsgsSeen.has(msgKey)) {
        uniqueServerMsgs.push(msg);
        serverMsgsSeen.add(msgKey);
      }
    });

    // Match local against server
    const pendingToShow = localMsgs.filter(lm => {
      const match = uniqueServerMsgs.some(sm => {
        if (sm.type !== 'SENT') return false;
        const sBody = normalizeMessageBody(sm.email_body);
        const lBody = normalizeMessageBody(lm.email_body);
        if (!sBody || !lBody || sBody !== lBody) return false;

        const serverTime = getMessageTimestamp(sm);
        const localTime = getMessageTimestamp(lm);
        if (!serverTime || !localTime) return true;

        return Math.abs(serverTime - localTime) <= 10 * 60 * 1000;
      });
      const ageMs = Date.now() - (lm.sentAt || 0);
      return !match && ageMs < OPTIMISTIC_MESSAGE_TTL_MS;
    });

    const combined = [...uniqueServerMsgs, ...pendingToShow];
    return combined.sort((a, b) => {
      return getMessageTimestamp(a) - getMessageTimestamp(b);
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
                <span style={{ fontSize: '11px', color: conversationAccentMuted, fontWeight: 700, marginLeft: '4px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                   <div style={{ width: 4, height: 4, borderRadius: '50%', background: conversationAccentMuted, animation: 'pulse 1s infinite' }} />
                   UPDATING
                </span>
              )}
              {!refreshing && activeThreadSyncing && (
                <span style={{ fontSize: '11px', color: conversationAccentMuted, fontWeight: 600, marginLeft: '4px', display: 'flex', alignItems: 'center', gap: '5px', opacity: 0.8 }}>
                   <div style={{ width: 5, height: 5, borderRadius: '50%', background: conversationAccentMuted, animation: 'pulse 1.4s ease-in-out infinite' }} />
                   Fetching latest…
                </span>
              )}
            </div>
          </div>
           <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
             <button 
               onClick={() => loadMessages({ silent: false, force: true })}
               disabled={refreshing}
               style={{
                 background: '#fff', border: '1.5px solid #e2e8f0', cursor: refreshing ? 'wait' : 'pointer', padding: '8px 14px', 
                 borderRadius: '12px', color: refreshing ? '#94a3b8' : '#64748b', fontSize: '12px', fontWeight: 600,
                 display: 'flex', alignItems: 'center', gap: '6px', transition: 'all 0.2s',
                 opacity: refreshing ? 0.6 : 1
               }}
               onMouseEnter={e => !refreshing && (e.currentTarget.style.borderColor = '#94a3b8')}
               onMouseLeave={e => !refreshing && (e.currentTarget.style.borderColor = '#e2e8f0')}
             >
               <RefreshCw size={14} className={refreshing ? 'spin-anim' : ''} style={{ transition: 'transform 0.2s' }} />
               {refreshing ? 'Syncing...' : 'Manual Sync'}
             </button>
            <button onClick={onClose} style={{ background: '#f1f5f9', border: 'none', cursor: 'pointer', padding: '10px', borderRadius: '12px', color: '#64748b', transition: 'all 0.2s' }} onMouseEnter={e => e.currentTarget.style.background = '#e2e8f0'} onMouseLeave={e => e.currentTarget.style.background = '#f1f5f9'}>
              <X size={20} />
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div style={{ display: 'flex', padding: '0 32px', borderBottom: '1px solid #f1f5f9', background: '#fff', gap: '24px' }}>
          {platformTabs.map(t => (
            <button
              key={t.id}
              onClick={() => setPlatform(t.id)}
              style={{
                padding: '16px 4px',
                background: 'transparent',
                border: 'none',
                borderBottom: platform === t.id ? `3px solid ${conversationAccent}` : '3px solid transparent',
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
              <t.icon size={18} color={platform === t.id ? conversationAccent : '#94a3b8'} strokeWidth={platform === t.id ? 2.3 : 2} />
              {t.label}
              {syncingByPlatform[t.id] && platform !== t.id && (
                <span style={{ width: 8, height: 8, borderRadius: '50%', background: conversationAccentMuted, position: 'absolute', top: 12, right: -6, border: '2px solid #fff', opacity: 0.85 }} />
              )}
              {t.id === 'linkedin' && hasLiResponse && platform !== 'linkedin' && (
                <span style={{ width: 8, height: 8, borderRadius: '50%', background: conversationAccentMuted, position: 'absolute', top: 12, right: -6, border: '2px solid #fff' }} />
              )}
            </button>
          ))}
        </div>

         {/* Messages */}
         <div className="hide-scrollbar" style={{ flex: 1, overflowY: 'auto', padding: '24px', display: 'flex', flexDirection: 'column', gap: '8px', background: '#f8fafc', containment: 'layout style paint' }}>
           {loading ? (
             <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', animation: 'fadeIn 0.3s ease-out' }}>
               {[1, 2, 3].map(i => (
                 <div key={i} style={{ width: i % 2 === 0 ? '55%' : '65%', height: '72px', borderRadius: '18px', background: 'linear-gradient(90deg, #f1f5f9 25%, #e2e8f0 50%, #f1f5f9 75%)', backgroundSize: '200% 100%', animation: 'shimmer 1.5s infinite', alignSelf: i % 2 === 0 ? 'flex-end' : 'flex-start' }} />
               ))}
             </div>
           ) : displayMessages.length === 0 ? (
             <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: '#94a3b8', gap: '16px', animation: 'fadeIn 0.4s ease-out' }}>
               <div style={{ width: 64, height: 64, borderRadius: '20px', background: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.05)' }}>
                 <MessageSquare size={32} opacity={0.5} color={conversationAccentMuted} />
               </div>
               <div style={{ textAlign: 'center', animation: 'slideUp 0.4s ease-out 0.1s backwards' }}>
                  <div style={{ fontWeight: 600, color: '#475569', fontSize: '15px' }}>
                    {activeThreadSyncing ? 'Loading latest messages...' : 'No messages yet'}
                  </div>
                  <div style={{ fontSize: '13px', marginTop: '4px' }}>
                    {activeThreadSyncing ? 'This usually takes a few seconds' : 'Start the conversation below'}
                  </div>
                </div>
               {platform === 'linkedin' && !heyreachCampaignId && (
                 <button
                   onClick={async () => {
                     setIsTriggering(true);
                     const res = await triggerHeyReachOutreach([candidate.id]);
                     setIsTriggering(false);
                     if (res.success) {
                       setHasTriggered(true);
                       setSyncingByPlatform(prev => ({ ...prev, linkedin: true }));
                       void loadMessages({ silent: true, targetPlatform: 'linkedin', force: true });
                     }
                   }}
                   disabled={isTriggering || hasTriggered}
                   style={{
                     marginTop: '12px',
                     padding: '10px 20px',
                     background: conversationAccent,
                     color: '#fff',
                     border: `1px solid ${conversationAccent}`,
                     borderRadius: '12px',
                     fontWeight: 600,
                     fontSize: '13px',
                     cursor: 'pointer',
                     display: 'flex',
                     alignItems: 'center',
                     gap: '8px',
                     boxShadow: '0 12px 24px rgba(15,23,42,0.12)'
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
                 const cleanBody = rawBody
                   .replace(/<(br|div|p)(\s*\/?)>/gi, '\n') // Convert block tags to freshlines before stripping
                   .replace(/<[^>]+>/g, '')
                   .replace(/&nbsp;/g, ' ')
                   .replace(/&amp;/g, '&')
                   .replace(/&quot;/g, '"')
                   .replace(/&apos;/g, "'")
                   .trim()
                   .replace(/\n\s*\n/g, '\n\n'); // Max 2 consecutive newlines

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
                         fontSize: '12px', fontWeight: 700, color: isCandidate ? '#475569' : '#fff',
                         border: isCandidate ? `1px solid ${conversationBorder}` : 'none',
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
                         boxShadow: isCandidate ? '0 4px 6px -1px rgba(0,0,0,0.05)' : '0 12px 24px rgba(15,23,42,0.12)',
                         border: isCandidate ? `1px solid ${conversationBorder}` : '1px solid #111827',
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
            padding: '8px 8px 8px 16px', borderRadius: '20px', border: `1px solid ${conversationBorder}`,
            transition: 'border-color 0.2s'
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
                color: '#fff', border: replyText.trim() ? `1px solid ${conversationAccent}` : '1px solid #e2e8f0', cursor: 'pointer',
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
  const user = useAppStore(state => state.user);
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

  // --- Store Sync (Persistent State) ---
  const candidates = useAppStore(state => state.tpCandidates);
  const total = useAppStore(state => state.tpTotal);
  const totalPages = useAppStore(state => state.tpTotalPages);
  const statusCounts = useAppStore(state => state.tpStatusCounts);
  const filters = useAppStore(state => state.tpFilters);
  const activeStatusTab = useAppStore(state => state.tpActiveStatusTab);
  const sortBy = useAppStore(state => state.tpSortBy);
  const sortDir = useAppStore(state => state.tpSortDir);
  const page = useAppStore(state => state.tpPage);
  const pageSize = useAppStore(state => state.tpPageSize);
  const globalSearch = useAppStore(state => state.tpGlobalSearch);
  
  const setFilters = useAppStore(state => state.setTpFilters);
  const setActiveStatusTab = useAppStore(state => state.setTpActiveStatusTab);
  const setTpPagination = useAppStore(state => state.setTpPagination);
  const setTpSort = useAppStore(state => state.setTpSort);
  const setGlobalSearch = useAppStore(state => state.setTpGlobalSearch);
  const fetchTalentPool = useAppStore(state => state.fetchTalentPool);
  const fetchTalentPoolIndex = useAppStore(state => state.fetchTalentPoolIndex);
  const talentPoolCache = useAppStore(state => state.talentPoolCache);
  const talentPoolIndex = useAppStore(state => state.talentPoolIndex);
  const updateTpCandidate = useAppStore(state => state.updateTpCandidate);

  const setPage = (nextPage) => {
    const currentPage = useAppStore.getState().tpPage;
    const currentPageSize = useAppStore.getState().tpPageSize;
    const resolvedPage = typeof nextPage === 'function' ? nextPage(currentPage) : nextPage;
    setTpPagination(resolvedPage, currentPageSize);
  };
  const setSortBy = (sb) => setTpSort(sb, useAppStore.getState().tpSortDir);
  const setSortDir = (nextSortDir) => {
    const currentSortBy = useAppStore.getState().tpSortBy;
    const currentSortDir = useAppStore.getState().tpSortDir;
    const resolvedSortDir = typeof nextSortDir === 'function' ? nextSortDir(currentSortDir) : nextSortDir;
    setTpSort(currentSortBy, resolvedSortDir);
  };

  const [loading, setLoading] = useState(false);
  const [isFilterCollapsed, setIsFilterCollapsed] = useState(() => {
    return localStorage.getItem('tp-filter-collapsed') === 'true';
  });

  const toggleFilterSidebar = () => {
    setIsFilterCollapsed(prev => {
      const next = !prev;
      localStorage.setItem('tp-filter-collapsed', String(next));
      return next;
    });
  };
  const [meta, setMeta] = useState({ companies: [], cities: [], products: [], statuses: [] });
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
  const heyreachCampaignId = useAppStore(state => state.heyreachCampaignId);
  const didInitRef = useRef(false);
  const talentPoolRequestSeqRef = useRef(0);
  const visibleCandidatesRef = useRef(candidates);
  const canUseInstantLocalFilteringRef = useRef(false);

  const [isRevalidating, setIsRevalidating] = useState(false);

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

            const email = resolveContactValue(data.email);
            const phone = resolveContactValue(data.phone, data.mobile_phone);
            const done = Boolean(email || phone || data.enrichment_finished);
            if (!done) return;

            setContactInfo(prev => {
              const current = prev[id] || {};
              const next = {
                email: resolveContactValue(email, current.email),
                phone: resolveContactValue(phone, current.phone),
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
      window.localStorage.setItem(CONTACT_INFO_STORAGE_KEY, JSON.stringify(contactInfo));
    } catch {
      // Ignore storage write errors (private mode, quota, etc.)
    }
  }, [contactInfo]);

  const handleShortlisted = async (candidateId) => {
    setShortlistingId(candidateId);
    // Preserve any existing contact details immediately to avoid flicker while enrichment runs.
    setContactInfo(prev => {
      const current = prev[candidateId] || {};
      const row = visibleCandidatesRef.current.find(c => c.id === candidateId)
        || talentPoolIndex?.rows?.find(c => c.id === candidateId)
        || candidates.find(c => c.id === candidateId)
        || {};
      const email = resolveContactValue(current.email, row.email);
      const phone = resolveContactValue(current.phone, row.phone, row.mobile_phone);
      const next = {
        email,
        phone,
        enriching: !(email || phone)
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
          email: resolveContactValue(d.email, prev[candidateId]?.email),
          phone: resolveContactValue(d.phone, prev[candidateId]?.phone),
          // If Clay was triggered (no data yet), keep enriching=true
          enriching: d.contact_enriching
            && !resolveContactValue(d.email, prev[candidateId]?.email)
            && !resolveContactValue(d.phone, prev[candidateId]?.phone)
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
      const candidate = visibleCandidatesRef.current.find(c => c.id === candidateId)
        || talentPoolIndex?.rows?.find(c => c.id === candidateId)
        || candidates.find(c => c.id === candidateId);
      if (candidate && (candidate.status === 'To be started' || !candidate.status)) {
        try {
          await axios.post(`${API_BASE}/candidates/${candidateId}/status`, { status: 'Shortlisted' });
          updateTpCandidate(candidateId, { status: 'Shortlisted' });
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
        const email = resolveContactValue(row.email, existing.email);
        const phone = resolveContactValue(row.phone, row.mobile_phone, existing.phone);
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
    let cancelled = false;
    let idleId = null;
    let timeoutId = null;

    axios.get(`${API_BASE}/candidates/browse/meta`).then(r => {
      if (!cancelled) {
        setMeta(r.data);
      }
    }).catch(() => { });

    fetchAnalytics();

    const prefetchTalentPoolIndex = () => {
      if (!cancelled) {
        void fetchTalentPoolIndex();
      }
    };

    if (typeof window !== 'undefined' && typeof window.requestIdleCallback === 'function') {
      idleId = window.requestIdleCallback(prefetchTalentPoolIndex, { timeout: 1200 });
    } else {
      timeoutId = window.setTimeout(prefetchTalentPoolIndex, 250);
    }

    return () => {
      cancelled = true;
      if (idleId !== null && typeof window !== 'undefined' && typeof window.cancelIdleCallback === 'function') {
        window.cancelIdleCallback(idleId);
      }
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId);
      }
    };
  }, [fetchAnalytics, fetchTalentPoolIndex]);

  const hasSemanticProductFilter = splitTalentPoolFilterValues(
    filters?.product_service,
    filters?.productInput,
  ).length > 0;
  const canUseInstantLocalFiltering = Boolean(
    !hasSemanticProductFilter &&
    Array.isArray(talentPoolIndex?.rows) &&
    talentPoolIndex.rows.length > 0
  );
  const localTalentPoolView = canUseInstantLocalFiltering
    ? buildLocalTalentPoolView(talentPoolIndex.rows, {
      globalSearch,
      filters,
      activeStatusTab,
      sortBy,
      sortDir,
      page,
      pageSize,
    })
    : null;
  const displayedCandidates = localTalentPoolView?.candidates || candidates;
  const displayedTotal = localTalentPoolView?.total ?? total;
  const displayedTotalPages = localTalentPoolView?.totalPages ?? totalPages;
  const displayedStatusCounts = localTalentPoolView?.statusCounts || statusCounts;
  const allVisibleSelected = displayedCandidates.length > 0 && displayedCandidates.every((candidate) => selectedIds.has(candidate.id));

  const fetchCandidates = useCallback(async (pg = 1) => {
    let requestId = 0;
    try {
      requestId = ++talentPoolRequestSeqRef.current;
      const paramsString = buildTalentPoolParamsString({
        page: pg,
        pageSize,
        globalSearch,
        filters,
        activeStatusTab,
        sortBy,
        sortDir,
      });

      const cache = useAppStore.getState().talentPoolCache || talentPoolCache;
      const cachedData = cache.lastParamsString === paramsString ? cache.data : null;
      const hasData = visibleCandidatesRef.current.length > 0;

      if (!cachedData && !hasData) {
        setLoading(true);
      } else {
        setIsRevalidating(true);
      }

      const res = await fetchTalentPool(paramsString);

      if (requestId !== talentPoolRequestSeqRef.current) {
        return;
      }

      if (res.success && res.data) {
        setIsSemanticSearch(res.data.is_semantic_search || false);
        mergeContactInfoFromRows(res.data.candidates);
      }
    } catch (e) {
      console.error('Failed to fetch talent pool:', e);
    } finally {
      if (requestId === talentPoolRequestSeqRef.current) {
        setLoading(false);
        setIsRevalidating(false);
      }
    }
  }, [globalSearch, filters, activeStatusTab, sortBy, sortDir, pageSize, mergeContactInfoFromRows, fetchTalentPool, talentPoolCache]);

  useEffect(() => {
    visibleCandidatesRef.current = displayedCandidates;
    canUseInstantLocalFilteringRef.current = canUseInstantLocalFiltering;
  }, [displayedCandidates, canUseInstantLocalFiltering]);

  useEffect(() => {
    if (canUseInstantLocalFiltering) {
      setIsSemanticSearch(false);
      setLoading(false);
      setIsRevalidating(false);
    }
  }, [canUseInstantLocalFiltering]);

  useEffect(() => {
    if (page > displayedTotalPages) {
      setPage(displayedTotalPages);
    }
  }, [page, displayedTotalPages]);

  // Aggressive Pre-warming for LinkedIn Cache
  useEffect(() => {
    if (displayedCandidates.length > 0) {
      const prewarmIds = displayedCandidates
        .filter(c => ['replied', 'message_sent', 'connection_accepted', 'in_campaign'].includes(c.li_status))
        .map(c => c.id);
      if (prewarmIds.length > 0) {
        useAppStore.getState().prewarmLinkedInCache(prewarmIds).catch(console.error);
      }
    }
  }, [displayedCandidates]);

  // Debounced refetch
  useEffect(() => {
    if (!didInitRef.current) {
      didInitRef.current = true;
      return;
    }
    clearTimeout(debounceRef.current);
    if (canUseInstantLocalFiltering) {
      if (page !== 1) {
        setPage(1);
      }
      return () => clearTimeout(debounceRef.current);
    }
    debounceRef.current = setTimeout(() => {
      if (page !== 1) {
        setPage(1);
        return;
      }
      fetchCandidates(1);
    }, 120);
    return () => clearTimeout(debounceRef.current);
  }, [fetchCandidates, canUseInstantLocalFiltering]);

  useEffect(() => {
    if (!canUseInstantLocalFiltering) {
      fetchCandidates(page);
    }
  }, [page, canUseInstantLocalFiltering, fetchCandidates]);

  // Store fetchCandidates and page in refs to avoid dependency cycles in the sync interval
  const fetchRef = useRef(fetchCandidates);
  const pageRef = useRef(page);
  useEffect(() => {
    fetchRef.current = fetchCandidates;
    pageRef.current = page;
  }, [fetchCandidates, page]);

  useEffect(() => {
    let cancelled = false;
    let timeoutId = null;
    let nextDelayMs = 15000;

    const syncTalentPoolReplies = async () => {
      const res = await syncOutreachResponses(0);
      const updatedCount = Number(res?.data?.updated_count || 0);
      if (cancelled) {
        return;
      }

      if (res?.success && updatedCount > 0) {
        if (canUseInstantLocalFilteringRef.current) {
          await fetchTalentPoolIndex({ force: true });
        } else {
          fetchRef.current(pageRef.current);
        }
      }

      nextDelayMs = res?.success ? 15000 : Math.min(nextDelayMs * 2, 60000);
      timeoutId = window.setTimeout(syncTalentPoolReplies, nextDelayMs);
    };

    timeoutId = window.setTimeout(syncTalentPoolReplies, 0);

    return () => {
      cancelled = true;
      if (timeoutId !== null) {
        clearTimeout(timeoutId);
      }
    };
  }, [syncOutreachResponses, fetchTalentPoolIndex]);


  const setFilter = (key, val) => setFilters(prev => ({ ...prev, [key]: val }));
  const clearFilters = () => {
    startTransition(() => {
      setFilters(() => ({
        title: [], titleInput: '',
        company: [], companyInput: '',
        city: [], cityInput: '',
        product_service: [], productInput: '',
        status: '', created_by: '',
        min_exp: 0, max_exp: 40,
      }));
      setGlobalSearch('');
      setActiveStatusTab('');
    });
    setTpPagination(1, pageSize);
  };
  const hasFilters = filters && Object.keys(filters).some(key => {
    if (key.toLowerCase().includes('input')) return false;
    const v = filters[key];
    if (key === 'min_exp') return v > 0;
    if (key === 'max_exp') return v < 40;
    return Array.isArray(v) ? v.length > 0 : !!v;
  });

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
    setSelectedIds(prev => {
      const next = new Set(prev);
      if (allVisibleSelected) {
        displayedCandidates.forEach(candidate => next.delete(candidate.id));
      } else {
        displayedCandidates.forEach(candidate => next.add(candidate.id));
      }
      return next;
    });
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

  const panelSurface = {
    background: 'rgba(255,255,255,0.84)',
    backdropFilter: 'blur(18px)',
    border: '1px solid rgba(226, 232, 240, 0.92)',
    boxShadow: '0 18px 40px rgba(15,23,42,0.06)',
  };
  const surfaceBorder = 'rgba(226, 232, 240, 0.9)';

  return (
    <div style={{ fontFamily: '"Inter", -apple-system, sans-serif', display: 'flex', gap: 18, height: '100vh', overflow: 'hidden', padding: '22px', boxSizing: 'border-box' }}>

      {/* ── Left Filter Sidebar ── */}
      <aside style={{
        width: isFilterCollapsed ? 0 : 220,
        minWidth: isFilterCollapsed ? 0 : 220,
        padding: isFilterCollapsed ? 0 : '20px 18px',
        background: panelSurface.background,
        backdropFilter: panelSurface.backdropFilter,
        border: isFilterCollapsed ? 'none' : panelSurface.border,
        boxShadow: isFilterCollapsed ? 'none' : panelSurface.boxShadow,
        borderRadius: isFilterCollapsed ? 0 : 24,
        overflowX: 'hidden',
        overflowY: 'auto',
        flexShrink: 0,
        display: 'flex',
        flexDirection: 'column',
        gap: 0,
        transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        position: 'relative',
        opacity: isFilterCollapsed ? 0 : 1,
        visibility: isFilterCollapsed ? 'hidden' : 'visible'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 18, paddingBottom: 12, borderBottom: `1px solid ${surfaceBorder}` }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontWeight: 800, fontSize: 14, color: '#0f172a', letterSpacing: '-0.01em' }}>
            <SlidersHorizontal size={15} color="#64748b" /> Filters
          </div>
          {hasFilters && (
            <button onClick={clearFilters} style={{ background: '#eef2f6', border: '1px solid rgba(203, 213, 225, 0.85)', color: '#475569', fontSize: 11, fontWeight: 700, cursor: 'pointer', padding: '6px 10px', display: 'flex', alignItems: 'center', gap: 4, borderRadius: 999 }}>
              <X size={11} /> Clear
            </button>
          )}
        </div>

        <TagFilterInput label="Title" values={filters?.title || []} inputValue={filters?.titleInput || ''} onInputChange={v => setFilter('titleInput', v)} onTagsChange={v => setFilter('title', v)} placeholder="e.g. Engineer" icon={Briefcase} />
        <TagFilterInput label="Current Company" values={filters?.company || []} inputValue={filters?.companyInput || ''} onInputChange={v => setFilter('companyInput', v)} onTagsChange={v => setFilter('company', v)} placeholder="e.g. Google" icon={Building2} />
        <TagFilterInput label="City" values={filters?.city || []} inputValue={filters?.cityInput || ''} onInputChange={v => setFilter('cityInput', v)} onTagsChange={v => setFilter('city', v)} placeholder="e.g. San Francisco" icon={MapPin} />

        <TagFilterInput label="Expertise / Product" values={filters?.product_service || []} inputValue={filters?.productInput || ''} onInputChange={v => setFilter('productInput', v)} onTagsChange={v => setFilter('product_service', v)} placeholder="e.g. SaaS, Fintech" icon={BarChart2} />

        {filters?.created_by !== undefined && (
          <SelectFilter
            label="Recruiter"
            value={filters?.created_by || ''}
            onChange={v => setFilter('created_by', v)}
            options={meta.recruiters || []}
            placeholder="All Recruiters"
          />
        )}

        <SelectFilter label="Status" value={filters?.status || ''} onChange={v => setFilter('status', v)} options={meta.statuses || []} placeholder="All Statuses" />

        <RangeSlider
          label="Total Experience"
          min={0}
          max={40}
          minValue={filters?.min_exp ?? 0}
          maxValue={filters?.max_exp ?? 40}
          onChange={(min, max) => {
            setFilter('min_exp', min);
            setFilter('max_exp', max);
          }}
        />

        <div style={{ marginTop: 'auto', paddingTop: '20px' }}>
          <button
            onClick={clearFilters}
            style={{
              width: '100%', padding: '11px 12px', borderRadius: '12px',
              background: '#fff', border: '1px solid rgba(203, 213, 225, 0.9)',
              color: '#334155', fontSize: '12px', fontWeight: 700,
              cursor: 'pointer', transition: 'all 0.2s',
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px'
            }}
            onMouseEnter={e => { e.currentTarget.style.background = '#f8fafc'; e.currentTarget.style.borderColor = 'rgba(148, 163, 184, 0.75)'; }}
            onMouseLeave={e => { e.currentTarget.style.background = '#fff'; e.currentTarget.style.borderColor = 'rgba(203, 213, 225, 0.9)'; }}
          >
            <RefreshCw size={14} /> Reset All Filters
          </button>
        </div>

      </aside>

      {/* ── Main Content ── */}
      <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden', gap: 18 }}>

        {/* Top bar */}
        <div style={{ ...panelSurface, borderRadius: 24, overflow: 'hidden', flexShrink: 0 }}>
          <div style={{
            padding: '18px 20px',
            display: 'flex',
            alignItems: 'center',
            gap: 14,
            flexWrap: 'wrap',
            borderBottom: analytics ? `1px solid ${surfaceBorder}` : 'none'
          }}>
            <button
              onClick={toggleFilterSidebar}
              title={isFilterCollapsed ? "Show Filters" : "Hide Filters"}
              style={{
                background: isFilterCollapsed ? '#0f172a' : '#fff',
                border: '1px solid rgba(203, 213, 225, 0.9)',
                borderRadius: '12px',
                width: '38px',
                height: '38px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                color: isFilterCollapsed ? '#fff' : '#64748b',
                transition: 'all 0.2s',
                boxShadow: '0 10px 24px rgba(15,23,42,0.05)'
              }}
              onMouseEnter={e => {
                if (!isFilterCollapsed) e.currentTarget.style.borderColor = 'rgba(194, 124, 63, 0.32)';
                e.currentTarget.style.transform = 'translateY(-1px)';
              }}
              onMouseLeave={e => {
                if (!isFilterCollapsed) e.currentTarget.style.borderColor = 'rgba(203, 213, 225, 0.9)';
                e.currentTarget.style.transform = 'none';
              }}
            >
              {isFilterCollapsed ? <Filter size={16} /> : <ChevronLeft size={18} />}
            </button>

            <div style={{ minWidth: 150 }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: '#8b6b44', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 4 }}>
                Candidate operations
              </div>
              <div style={{ fontSize: 22, fontWeight: 800, color: '#0f172a', letterSpacing: '-0.03em' }}>
                Talent Pool
              </div>
            </div>

            <div style={{ position: 'relative', flex: '1 1 320px', maxWidth: 460, minWidth: 220 }}>
              <Search size={15} color="#94a3b8" style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)' }} />
              <input
                type="text"
                value={globalSearch}
                onChange={e => setGlobalSearch(e.target.value)}
                placeholder="Search candidates, titles, companies, cities..."
                style={{
                  width: '100%', padding: '11px 36px 11px 36px',
                  background: 'rgba(255,255,255,0.9)', border: '1px solid rgba(203, 213, 225, 0.9)',
                  borderRadius: 14, color: '#0f172a', fontSize: 13,
                  outline: 'none', fontFamily: 'inherit', boxSizing: 'border-box',
                  transition: 'border-color 0.15s, box-shadow 0.15s, background 0.15s',
                  boxShadow: 'inset 0 1px 2px rgba(15,23,42,0.03)',
                }}
                onFocus={e => {
                  e.target.style.borderColor = 'rgba(194, 124, 63, 0.5)';
                  e.target.style.boxShadow = '0 0 0 3px rgba(194, 124, 63, 0.12)';
                  e.target.style.background = '#fff';
                }}
                onBlur={e => {
                  e.target.style.borderColor = 'rgba(203, 213, 225, 0.9)';
                  e.target.style.boxShadow = 'inset 0 1px 2px rgba(15,23,42,0.03)';
                  e.target.style.background = 'rgba(255,255,255,0.9)';
                }}
              />
              {globalSearch && (
                <X
                  size={14}
                  color="#94a3b8"
                  style={{ position: 'absolute', right: 12, top: '50%', transform: 'translateY(-50%)', cursor: 'pointer' }}
                  onClick={() => setGlobalSearch('')}
                />
              )}
            </div>

            <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 12 }}>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
                <span style={{ fontSize: 13, color: '#0f172a', fontWeight: 700 }}>
                  {loading && !displayedCandidates.length ? '...' : `${displayedTotal.toLocaleString()} candidates`}
                </span>
                {isRevalidating ? (
                  <span style={{ fontSize: 10, color: '#64748b', fontWeight: 600, display: 'flex', alignItems: 'center', gap: 4 }}>
                    <RefreshCw size={10} className="revalidating" /> Syncing updates
                  </span>
                ) : (
                  <span style={{ fontSize: 10, color: '#94a3b8', fontWeight: 600 }}>
                    Live workspace view
                  </span>
                )}
              </div>
              <button onClick={async () => {
                if (canUseInstantLocalFiltering) {
                  setIsRevalidating(true);
                  try {
                    const result = await fetchTalentPoolIndex({ force: true });
                    if (!result?.success) {
                      toast.error(result?.error || 'Failed to refresh candidates');
                    }
                  } finally {
                    setIsRevalidating(false);
                  }
                  return;
                }
                await fetchCandidates(page);
              }}
                style={{ width: 38, height: 38, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#fff', border: '1px solid rgba(203, 213, 225, 0.9)', borderRadius: 12, cursor: 'pointer', color: '#64748b', boxShadow: '0 10px 24px rgba(15,23,42,0.05)' }}>
                <RefreshCw size={14} style={{ animation: loading || isRevalidating ? 'spin 1s linear infinite' : 'none' }} />
              </button>
            </div>
          </div>

          {analytics && (
            <div style={{ padding: '18px 20px 20px' }}>
              <StatisticsDashboard
                analytics={analytics}
                role={role}
                onStatClick={(status) => {
                  startTransition(() => {
                    setActiveStatusTab(status);
                  });
                  setPage(1);
                }}
                onRecruiterClick={(recruiter) => {
                  startTransition(() => {
                    setFilter('created_by', recruiter);
                  });
                  setPage(1);
                }}
              />
            </div>
          )}
        </div>

        <div style={{ ...panelSurface, flex: 1, minHeight: 0, borderRadius: 24, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        {/* Status tabs */}
        <div style={{ padding: '14px 18px', background: 'rgba(248,250,252,0.78)', borderBottom: `1px solid ${surfaceBorder}`, display: 'flex', gap: 10, overflowX: 'auto', scrollbarWidth: 'none' }}>
          {['', ...RECRUITMENT_STAGES].map(tab => {
            const isActive = activeStatusTab === tab;
            const count = tab === '' ? (displayedTotal || 0) : (displayedStatusCounts?.[tab] || 0);
            const style = tab ? (STATUS_STYLES[tab.toLowerCase()] || {}) : { bg: '#f1f5f9', color: '#475569', dot: '#94a3b8' };

            return (
              <button key={tab || 'all'}
                onClick={() => {
                  startTransition(() => {
                    setActiveStatusTab(tab);
                  });
                  setPage(1);
                }}
                style={{
                  padding: '7px 14px', borderRadius: '999px', border: isActive ? '1px solid #111827' : '1px solid rgba(203, 213, 225, 0.9)',
                  background: isActive ? '#111827' : 'rgba(255,255,255,0.72)', cursor: 'pointer', fontSize: 12, fontWeight: 700,
                  color: isActive ? '#fff' : '#64748b', whiteSpace: 'nowrap',
                  display: 'flex', alignItems: 'center', gap: 8, fontFamily: 'inherit',
                  transition: 'all 0.15s',
                  boxShadow: isActive ? '0 10px 18px rgba(15,23,42,0.12)' : 'none',
                }}
                onMouseEnter={e => {
                  if (!isActive) {
                    e.currentTarget.style.borderColor = 'rgba(148, 163, 184, 0.75)';
                    e.currentTarget.style.background = '#fff';
                  }
                }}
                onMouseLeave={e => {
                  if (!isActive) {
                    e.currentTarget.style.borderColor = 'rgba(203, 213, 225, 0.9)';
                    e.currentTarget.style.background = 'rgba(255,255,255,0.72)';
                  }
                }}
              >
                {tab === '' ? (
                  <span style={{ color: isActive ? '#fff' : '#0f172a' }}>All ({displayedTotal})</span>
                ) : (
                  <>
                    <span style={{ width: 6, height: 6, borderRadius: '50%', background: isActive ? '#fff' : (style.dot || '#94a3b8') }} />
                    {tab}
                    <span style={{
                      marginLeft: 4, padding: '1px 6px', borderRadius: 10, fontSize: 10,
                      background: isActive ? 'rgba(255,255,255,0.14)' : '#e2e8f0',
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
        <div style={{ flex: 1, overflowY: 'auto', overflowX: 'auto', background: 'rgba(255,255,255,0.68)' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', minWidth: 2000 }}>
            <thead>
              <tr style={{ background: 'rgba(248,250,252,0.96)', borderBottom: `1px solid ${surfaceBorder}`, position: 'sticky', top: 0, zIndex: 10, backdropFilter: 'blur(12px)' }}>
                {cols.filter(c => !c.hidden).map((col, index) => (
                  <th key={col.key}
                    onClick={() => col.key === 'selection' ? toggleSelectAll() : handleSort(col.sortKey)}
                    style={{
                      padding: '11px 14px', textAlign: 'left',
                      fontSize: 11, fontWeight: 700, color: '#64748b',
                      textTransform: 'uppercase', letterSpacing: '0.08em',
                      minWidth: col.w, cursor: (col.sortKey || col.key === 'selection') ? 'pointer' : 'default',
                      whiteSpace: 'nowrap', userSelect: 'none',
                      borderBottom: `1px solid ${surfaceBorder}`,
                      borderRight: `1px solid ${surfaceBorder}`,
                      borderLeft: index === 0 ? `1px solid ${surfaceBorder}` : 'none'
                    }}
                  >
                    {col.key === 'selection' ? (
                      <input type="checkbox" checked={allVisibleSelected} readOnly style={{ cursor: 'pointer' }} />
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
              {loading && displayedCandidates.length === 0 && (
                Array.from({ length: 10 }).map((_, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #eef2f7' }}>
                    {cols.filter(c => !c.hidden).map((c, j) => (
                      <td key={j} style={{ padding: '13px 14px' }}>
                        <div style={{ height: 13, borderRadius: 6, background: 'linear-gradient(90deg,#f1f5f9 25%,#e2e8f0 50%,#f1f5f9 75%)', backgroundSize: '200% 100%', animation: 'shimmer 1.2s infinite', width: c.w * 0.6 }} />
                      </td>
                    ))}
                  </tr>
                ))
              )}

              {!loading && displayedCandidates.length === 0 && (
                <tr>
                  <td colSpan={cols.filter(c => !c.hidden).length} style={{ padding: '80px 20px', textAlign: 'center' }}>
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}>
                      <div style={{ width: 56, height: 56, background: '#f8fafc', border: '1px solid rgba(203, 213, 225, 0.9)', borderRadius: 18, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <User size={24} color="#cbd5e1" />
                      </div>
                      <p style={{ color: '#64748b', fontWeight: 600, fontSize: 15, margin: 0 }}>No candidates found</p>
                      <p style={{ color: '#cbd5e1', fontSize: 13, margin: 0 }}>Try adjusting your filters or search query</p>
                      {hasFilters && <button onClick={clearFilters} style={{ padding: '8px 16px', background: '#fff', border: '1px solid rgba(203, 213, 225, 0.9)', borderRadius: 10, color: '#334155', fontWeight: 700, fontSize: 13, cursor: 'pointer', fontFamily: 'inherit' }}>Clear Filters</button>}
                    </div>
                  </td>
                </tr>
              )}

              {displayedCandidates.map((c, idx) => (
                <tr key={c.id || idx}
                  style={{ borderBottom: '1px solid #eef2f7', transition: 'background 0.1s', background: selectedIds.has(c.id) ? 'rgba(194, 124, 63, 0.10)' : 'transparent' }}
                  onMouseEnter={e => { if (!selectedIds.has(c.id)) e.currentTarget.style.background = '#f8fafc'; }}
                  onMouseLeave={e => { if (!selectedIds.has(c.id)) e.currentTarget.style.background = 'transparent'; }}
                >
                  {/* Checkbox cell — ONLY this cell toggles row selection */}
                  <td
                    style={{ padding: '13px 14px', textAlign: 'center', borderRight: '1px solid #eef2f7', borderLeft: '1px solid #eef2f7', cursor: 'pointer' }}
                    onClick={(e) => { e.stopPropagation(); toggleSelectOne(c.id); }}
                  >
                    <input
                      type="checkbox"
                      checked={selectedIds.has(c.id)}
                      onChange={() => toggleSelectOne(c.id)}
                      style={{ cursor: 'pointer', pointerEvents: 'none' }}
                    />
                  </td>
                  <td style={{ padding: '13px 14px', fontSize: 13, color: '#0f172a', borderRight: '1px solid #eef2f7' }}>{c.first_name || ''}</td>
                  <td style={{ padding: '13px 14px', fontSize: 13, color: '#374151', borderRight: '1px solid #eef2f7' }}>{c.last_name || ''}</td>
                  <td style={{ padding: '13px 14px', fontSize: 13, color: '#374151', maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', borderRight: '1px solid #eef2f7' }}>{c.title || c.headline || ''}</td>
                  <td style={{ padding: '13px 14px', borderRight: '1px solid #eef2f7' }}>
                    {c.linkedin
                      ? <a href={c.linkedin} target="_blank" rel="noreferrer" style={{ color: '#2563eb', display: 'flex', alignItems: 'center' }} onClick={e => e.stopPropagation()}>
                        <ExternalLink size={14} />
                      </a>
                      : <span style={{ color: '#94a3b8', fontSize: 11, fontWeight: 500, fontStyle: 'italic' }}>Not linked</span>}
                  </td>
                  <td style={{ padding: '13px 14px', fontSize: 13, color: '#0f172a', borderRight: '1px solid #eef2f7' }}>{c.company || ''}</td>
                  <td style={{ padding: '13px 14px', fontSize: 12, color: '#64748b', borderRight: '1px solid #eef2f7' }}>{c.product_service || ''}</td>
                  <td style={{ padding: '13px 14px', fontSize: 13, color: '#374151', borderRight: '1px solid #eef2f7' }}>{c.city || ''}</td>
                  <td style={{ padding: '13px 14px', fontSize: 12, color: '#64748b', borderRight: '1px solid #eef2f7' }}>{c.location_type || ''}</td>
                  <td style={{ padding: '13px 14px', borderRight: '1px solid #eef2f7' }}>
                    <ExpBar value={c.total_experience_years || 0} />
                  </td>
                  <td style={{ padding: '13px 14px', fontSize: 13, color: '#374151', borderRight: '1px solid #eef2f7' }}>
                    {c.avg_tenure_years > 0 ? `${c.avg_tenure_years}y` : ''}
                  </td>
                  {(() => {
                    const emailVal = resolveContactValue(contactInfo[c.id]?.email, c.email);
                    const isEnriching = contactInfo[c.id]?.enriching && !emailVal;

                    if (isEnriching) {
                      return <td style={{ padding: '13px 14px', fontSize: 12, color: '#8b6b44', borderRight: '1px solid #eef2f7', fontStyle: 'italic' }}>Fetching...</td>;
                    }
                    // Always render as editable — even when a value is present
                    return (
                      <ClickableEditableCell
                        key={`email-${c.id}`}
                        id={c.id}
                        field="email"
                        value={emailVal || ''}
                        onUpdate={(id, data) => {
                          // Sync local contactInfo immediately so the cell reflects new value
                          setContactInfo(prev => ({
                            ...prev,
                            [id]: { ...prev[id], email: normalizeContactValue(data.email) }
                          }));
                          return updateFieldAndMaybeShortlist(id, data);
                        }}
                        placeholder="Not available"
                      />
                    );
                  })()}
                  {(() => {
                    const phoneVal = resolveContactValue(contactInfo[c.id]?.phone, c.phone, c.mobile_phone);
                    const isEnriching = contactInfo[c.id]?.enriching && !phoneVal;

                    if (isEnriching) {
                      return <td style={{ padding: '13px 14px', fontSize: 12, color: '#8b6b44', borderRight: '1px solid #eef2f7', fontStyle: 'italic' }}>Fetching...</td>;
                    }
                    // Always render as editable — even when a value is present
                    return (
                      <ClickableEditableCell
                        key={`phone-${c.id}`}
                        id={c.id}
                        field="phone"
                        value={phoneVal || ''}
                        onUpdate={(id, data) => {
                          // Sync local contactInfo immediately so the cell reflects new value
                          setContactInfo(prev => ({
                            ...prev,
                            [id]: { ...prev[id], phone: normalizeContactValue(data.phone) }
                          }));
                          return updateFieldAndMaybeShortlist(id, data);
                        }}
                        placeholder="Not available"
                      />
                    );
                  })()}
                  <td style={{ padding: '13px 14px', borderRight: '1px solid #eef2f7' }}>
                    <StatusDropdown
                      status={c.status}
                      candidateId={c.id}
                      onUpdate={(id, newStatus) => {
                        updateTpCandidate(id, { status: newStatus });
                      }}
                      onShortlisted={handleShortlisted}
                    />
                  </td>
                  <td
                    onClick={(e) => { e.stopPropagation(); setSelectedCandidateForChat(c); }}
                    style={{
                      padding: '13px 14px', borderRight: '1px solid #eef2f7',
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
                  <td style={{ padding: '13px 14px', borderRight: '1px solid #eef2f7' }}>
                    <EditableNotes candidateId={c.id} initialNotes={c.notes} />
                  </td>
                  {isSemanticSearch && (
                    <td style={{ padding: '13px 14px', borderRight: '1px solid #eef2f7' }}>
                      <div style={{
                        display: 'inline-flex', padding: '4px 8px', borderRadius: '999px',
                        background: c.match_score > 80 ? '#e7f6ec' : c.match_score > 60 ? '#f7f0e4' : '#edf2f7',
                        color: c.match_score > 80 ? '#166534' : c.match_score > 60 ? '#8b6b44' : '#475569',
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
        <div style={{ padding: '14px 18px', background: 'rgba(248,250,252,0.78)', borderTop: `1px solid ${surfaceBorder}`, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 13, color: '#64748b' }}>Rows per page:</span>
              <select
                value={pageSize}
                onChange={e => {
                  setTpPagination(1, Number(e.target.value));
                }}
                style={{
                  padding: '6px 10px', borderRadius: '10px', border: '1px solid rgba(203, 213, 225, 0.9)',
                  fontSize: '12px', fontWeight: 600, color: '#0f172a', outline: 'none',
                  background: '#fff', cursor: 'pointer'
                }}
              >
                {PAGE_SIZE_OPTIONS.map(opt => <option key={opt} value={opt}>{opt}</option>)}
              </select>
            </div>
            <span style={{ fontSize: 13, color: '#64748b' }}>
              Showing {displayedTotal === 0 ? 0 : Math.min((page - 1) * pageSize + 1, displayedTotal)}–{Math.min(page * pageSize, displayedTotal)} of <strong style={{ color: '#0f172a' }}>{displayedTotal.toLocaleString()}</strong> candidates
            </span>
          </div>
          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}
              style={{ width: 34, height: 34, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#fff', border: '1px solid rgba(203, 213, 225, 0.9)', borderRadius: 10, cursor: page === 1 ? 'not-allowed' : 'pointer', opacity: page === 1 ? 0.4 : 1 }}>
              <ChevronLeft size={14} color="#64748b" />
            </button>

            {/* Improved Page numbers with ellipsis */}
            {(() => {
              const pages = [];
              const range = 2; // Number of pages to show around current page

              for (let i = 1; i <= displayedTotalPages; i++) {
                if (
                  i === 1 ||
                  i === displayedTotalPages ||
                  (i >= page - range && i <= page + range)
                ) {
                  pages.push(
                    <button key={i} onClick={() => setPage(i)}
                      style={{
                        width: 34, height: 34, display: 'flex', alignItems: 'center', justifyContent: 'center',
                        background: i === page ? '#0f172a' : '#fff',
                        border: `1px solid ${i === page ? '#0f172a' : 'rgba(203, 213, 225, 0.9)'}`,
                        borderRadius: 10, cursor: 'pointer', fontSize: 13,
                        fontWeight: i === page ? 700 : 500, color: i === page ? '#fff' : '#64748b',
                        transition: 'all 0.15s'
                      }}
                      onMouseEnter={e => i !== page && (e.currentTarget.style.borderColor = 'rgba(148, 163, 184, 0.75)')}
                      onMouseLeave={e => i !== page && (e.currentTarget.style.borderColor = 'rgba(203, 213, 225, 0.9)')}
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

            <button onClick={() => setPage(p => Math.min(displayedTotalPages, p + 1))} disabled={page === displayedTotalPages}
              style={{ width: 34, height: 34, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#fff', border: '1px solid rgba(203, 213, 225, 0.9)', borderRadius: 10, cursor: page === displayedTotalPages ? 'not-allowed' : 'pointer', opacity: page === displayedTotalPages ? 0.4 : 1 }}>
              <ChevronRight size={14} color="#64748b" />
            </button>
          </div>
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
              padding: '8px 16px', background: '#fff', color: '#0f172a', border: '1px solid rgba(255,255,255,0.18)',
              borderRadius: '10px', fontSize: 13, fontWeight: 700, cursor: 'pointer',
              display: 'flex', alignItems: 'center', gap: 8, transition: 'all 0.2s'
            }}
            onMouseEnter={e => e.currentTarget.style.background = '#f8fafc'}
            onMouseLeave={e => e.currentTarget.style.background = '#fff'}
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
  const { callLists, fetchCallLists, createCallList, addCandidatesToCallList } = useAppStore(useShallow((state) => ({
    callLists: state.callLists,
    fetchCallLists: state.fetchCallLists,
    createCallList: state.createCallList,
    addCandidatesToCallList: state.addCandidatesToCallList,
  })));
  const [loading, setLoading] = useState(false);
  const [newListName, setNewListName] = useState('');
  const [selectedListId, setSelectedListId] = useState('');
  const [mode, setMode] = useState('select'); // 'select' or 'create'

  useEffect(() => {
    fetchCallLists({ force: true });
  }, [fetchCallLists]);

  const handleAction = async () => {
    setLoading(true);
    try {
      let listId = selectedListId;
      if (mode === 'create') {
        if (!newListName.trim()) return;
        const res = await createCallList(newListName.trim());
        if (res.success) listId = res.data.id;
        else throw new Error(res.error);
      }

      if (!listId) return;
      const res = await addCandidatesToCallList(candidateIds, listId);
      if (res.success) {
        toast.success(
          res.optimistic
            ? `Syncing ${selectedCount} candidate${selectedCount === 1 ? '' : 's'} to call list`
            : `Added ${selectedCount} candidate${selectedCount === 1 ? '' : 's'} to call list`
        );
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
