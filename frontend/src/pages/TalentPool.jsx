import React, { startTransition, useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { useAppStore, API_BASE, getRequestErrorMessage } from '../store/useAppStore';
import { longOperationAxios } from '../api/longTimeoutAxios';
import axios from 'axios';
import { toast } from 'sonner';
import { useShallow } from 'zustand/react/shallow';
import {
  Search, ExternalLink, ChevronLeft, ChevronRight, Filter,
  User, Building2, MapPin, Briefcase, BarChart2,
  SlidersHorizontal, RefreshCw, UserPlus, X, ChevronDown,
  Activity, MessageSquareMore, Users, Plus, Edit2, Check, Download, Trash2,
  Mail, Phone, MessageSquare, Linkedin, Send,
  FileUp, Play, Folder,
} from 'lucide-react';
import StatusDropdown, { RECRUITMENT_STAGES, STATUS_STYLES } from '../components/StatusDropdown';
import AiColumnConfigModal from '../components/AiColumnConfigModal';
import AiColumnCellDrawer from '../components/AiColumnCellDrawer';
import Select, { components } from 'react-select';
import { TagFilterInput, SelectFilter, RangeSlider, uniqueSortedOptions } from '../components/FilterComponents';
import CsvMappingModal from '../components/CsvMappingModal';
import HayasaBrand from '../components/HayasaBrand';
import AddToListModal from '../components/AddToListModal';

export function parseStructuredValue(val) {
  if (val == null) return null;
  if (typeof val === 'object') return val;
  const trimmed = String(val).trim();
  if ((trimmed.startsWith('{') && trimmed.endsWith('}')) || (trimmed.startsWith('[') && trimmed.endsWith(']'))) {
    try {
      return JSON.parse(trimmed);
    } catch (e) {
      try {
        const fn = new Function(`return (${trimmed});`);
        const result = fn();
        if (result && typeof result === 'object') {
          return result;
        }
      } catch (err) {
        try {
          const jsonString = trimmed
            .replace(/'/g, '"')
            .replace(/True/g, 'true')
            .replace(/False/g, 'false')
            .replace(/None/g, 'null');
          return JSON.parse(jsonString);
        } catch (err2) {
          return null;
        }
      }
    }
  }
  return null;
}

export function formatKey(key) {
  let label = key.replace(/[_\.]+/g, ' ').trim();
  if (label.toLowerCase().endsWith(' months')) {
    label = label.slice(0, -7).trim() + ' (months)';
  } else if (label.toLowerCase().endsWith(' years')) {
    label = label.slice(0, -6).trim() + ' (years)';
  }
  return label.charAt(0).toUpperCase() + label.slice(1);
}

const EMPTY_AI_RESPONSE_TEXTS = new Set([
  'no structured response returned',
  'no structured response returned.',
  'needs review',
]);

function isMeaningfulAiValue(value) {
  const text = String(value ?? '').trim();
  const lower = text.toLowerCase();
  return text !== '' && !EMPTY_AI_RESPONSE_TEXTS.has(lower) && !lower.startsWith('needs review:');
}

export function resolveAiCellValue(cell, outputKey, isPrimaryOutput) {
  const outputs = cell?.outputs || {};
  const directOutput = outputs?.[outputKey];
  if (isMeaningfulAiValue(directOutput)) return directOutput;

  if (isPrimaryOutput && isMeaningfulAiValue(cell?.primary_output)) {
    return cell.primary_output;
  }

  const firstOutput = Object.values(outputs).find(isMeaningfulAiValue);
  if (firstOutput != null) return firstOutput;

  return cell ? 'No' : '';
}

function renderAiValuePart(value, depth = 0) {
  if (value == null || String(value).trim() === '') return '—';
  const parsed = parseStructuredValue(value);
  const actual = parsed ?? value;
  if (Array.isArray(actual)) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6, width: '100%' }}>
        {actual.map((item, idx) => (
          <div key={idx} style={{
            padding: '4px 7px',
            borderRadius: 6,
            background: depth ? 'rgba(241,245,249,0.8)' : 'rgba(99, 102, 241, 0.06)',
            border: '1px solid rgba(203,213,225,0.65)',
          }}>
            {renderAiValuePart(item, depth + 1)}
          </div>
        ))}
      </div>
    );
  }
  if (actual && typeof actual === 'object') {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 5, width: '100%' }}>
        {Object.entries(actual).map(([k, v]) => (
          <div key={k} style={{ fontSize: depth ? '10.5px' : '11px', lineHeight: 1.45, color: '#334155' }}>
            <span style={{ fontWeight: 700, color: '#1e293b' }}>{formatKey(k)}:</span>{' '}
            {typeof v === 'object' && v !== null ? (
              <div style={{ marginTop: 3 }}>{renderAiValuePart(v, depth + 1)}</div>
            ) : (
              <span style={{ color: '#475569' }}>{typeof v === 'boolean' ? (v ? 'Yes' : 'No') : String(v ?? '—')}</span>
            )}
          </div>
        ))}
      </div>
    );
  }
  if (typeof actual === 'boolean') return actual ? 'Yes' : 'No';
  return <span style={{ whiteSpace: 'pre-wrap' }}>{String(actual)}</span>;
}

export function renderFriendlyAiValue(aiVal) {
  if (aiVal == null || String(aiVal).trim() === '') {
    return '—';
  }
  const parsed = parseStructuredValue(aiVal);
  if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
    return renderAiValuePart(parsed);
  }
  if (parsed && Array.isArray(parsed)) {
    return renderAiValuePart(parsed);
  }
  return <span style={{ whiteSpace: 'pre-wrap' }}>{String(aiVal)}</span>;
}

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

function EditableNotes({ candidateId, initialNotes, readOnly = false }) {
  const [notes, setNotes] = useState(initialNotes || '');
  const [isEditing, setIsEditing] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const updateCandidateNotes = useAppStore(state => state.updateCandidateNotes);

  useEffect(() => {
    setNotes(initialNotes || '');
  }, [initialNotes]);

  if (readOnly) {
    return (
      <div style={{ fontSize: '12.5px', color: '#64748b', maxWidth: '180px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {initialNotes || '—'}
      </div>
    );
  }

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



const DEFAULT_PAGE_SIZE = 25;
const PAGE_SIZE_OPTIONS = [10, 25, 50, 100];
const CONTACT_INFO_STORAGE_KEY = 'talent-pool-contact-info-v1';
const OUTREACH_REPLY_SYNC_INITIAL_DELAY_MS = 60 * 1000;
const OUTREACH_REPLY_SYNC_INTERVAL_MS = 5 * 60 * 1000;
const OUTREACH_REPLY_SYNC_MAX_BACKOFF_MS = 10 * 60 * 1000;
const isDocumentVisible = () => typeof document === 'undefined' || document.visibilityState === 'visible';

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

function splitTalentPoolFilterValues(values = [], _inputValue = '') {
  const list = Array.isArray(values) ? values : values == null || values === '' ? [] : [values];
  return [...list]
    .flatMap((value) => String(value || '').split(','))
    .map((value) => value.trim())
    .filter(Boolean);
}

function buildTalentPoolParamsString({
  page = 1,
  pageSize = 25,
  globalSearch = '',
  filters = {},
  activeStatusTab = '',
  sortBy = 'name',
  sortDir = 'asc',
  candidateIds = [],
}) {
  const params = new URLSearchParams();
  const normalizedCandidateIds = Array.isArray(candidateIds)
    ? candidateIds.map(Number).filter(Number.isFinite)
    : [];
  params.set('page', normalizedCandidateIds.length ? 1 : page);
  params.set('page_size', normalizedCandidateIds.length ? Math.min(5000, Math.max(pageSize || 25, normalizedCandidateIds.length)) : (pageSize || 25));
  if (normalizedCandidateIds.length) {
    params.set('candidate_ids', normalizedCandidateIds.join(','));
  }

  if (!normalizedCandidateIds.length && globalSearch) params.set('q', globalSearch);

  const titleSearch = splitTalentPoolFilterValues(filters?.title, filters?.titleInput).join(',');
  if (!normalizedCandidateIds.length && titleSearch) params.set('title', titleSearch);

  const companySearch = splitTalentPoolFilterValues(filters?.company, filters?.companyInput).join(',');
  if (!normalizedCandidateIds.length && companySearch) params.set('company', companySearch);

  const citySearch = splitTalentPoolFilterValues(filters?.city, filters?.cityInput).join(',');
  if (!normalizedCandidateIds.length && citySearch) params.set('city', citySearch);

  const productSearch = splitTalentPoolFilterValues(filters?.product_service, filters?.productInput).join(',');
  if (!normalizedCandidateIds.length && productSearch) params.set('product_service', productSearch);

  const statusValues = Array.isArray(filters?.status)
    ? filters.status.filter(Boolean)
    : (filters?.status ? [filters.status] : []);
  if (!normalizedCandidateIds.length && statusValues.length) params.set('status', statusValues.join(','));

  const minExp = Number(filters?.min_exp);
  const maxExp = Number(filters?.max_exp);
  if (!normalizedCandidateIds.length && Number.isFinite(minExp) && minExp > 0) params.set('min_exp', minExp);
  if (!normalizedCandidateIds.length && Number.isFinite(maxExp) && maxExp < 40) params.set('max_exp', maxExp);
  if (!normalizedCandidateIds.length && filters?.created_by) params.set('created_by', filters.created_by);

  params.set('sort_by', sortBy);
  params.set('sort_dir', sortDir);

  return params.toString();
}

export function summarizeAiRun(run) {
  if (!run) return '';
  const total = Number(run.total || 0);
  const finished = Number(run.completed || 0) + Number(run.failed || 0) + Number(run.skipped || 0);
  const st = String(run.status || '').toLowerCase();
  if (!total) return run.status || '';
  const base = `${finished}/${total}`;
  if (finished < total && (st === 'running' || st === 'queued')) {
    return `${base} · ${st}`;
  }
  return base;
}

function buildQueuedAiCells(candidateIds = []) {
  return candidateIds.reduce((acc, id) => {
    acc[id] = { status: 'queued', primary_output: '', outputs: {} };
    return acc;
  }, {});
}

export function createOptimisticAiColumn(definition = {}, candidateIds = [], runId = null) {
  return {
    ...definition,
    id: definition.id,
    name: definition.name || 'Smart Column',
    output_schema: Array.isArray(definition.output_schema) ? definition.output_schema : [],
    cells_by_candidate: {
      ...(definition.cells_by_candidate || {}),
      ...buildQueuedAiCells(candidateIds),
    },
    latest_run: {
      ...(definition.latest_run || {}),
      id: runId || definition.latest_run?.id || null,
      run_id: runId || definition.latest_run?.run_id || null,
      status: 'queued',
      total: candidateIds.length,
      completed: 0,
      failed: 0,
      skipped: 0,
    },
    __optimistic: true,
  };
}

export function mergeAiColumnDefinitions(previous = [], incoming = []) {
  const previousById = new Map((previous || []).map(col => [String(col.id), col]));
  const incomingIds = new Set();
  const merged = (incoming || []).map(nextCol => {
    const key = String(nextCol.id);
    incomingIds.add(key);
    const prevCol = previousById.get(key);
    if (!prevCol) return nextCol;
    return {
      ...prevCol,
      ...nextCol,
      cells_by_candidate: {
        ...(prevCol.cells_by_candidate || {}),
        ...(nextCol.cells_by_candidate || {}),
      },
      latest_run: nextCol.latest_run || prevCol.latest_run,
      __optimistic: Boolean(nextCol.__optimistic),
    };
  });

  for (const prevCol of previous || []) {
    if (incomingIds.has(String(prevCol.id))) continue;
    const status = String(prevCol.latest_run?.status || '').toLowerCase();
    if (prevCol.__optimistic || status === 'queued' || status === 'running') {
      merged.push(prevCol);
    }
  }
  return merged;
}

const AI_CELL_STALE_MS = 24 * 60 * 60 * 1000;

function formatAiCellTimestamp(value) {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
}

function isAiCellStale(value) {
  if (!value) return false;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return false;
  return Date.now() - date.getTime() > AI_CELL_STALE_MS;
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

// Sum of the per-status counts. status_counts ignores the active status filter,
// so its sum is the true "all statuses" total in the SAME scope as any single
// status count (e.g. Shortlisted). Returns null when counts are unavailable.
function sumStatusCounts(counts) {
  if (!counts || typeof counts !== 'object') return null;
  const keys = Object.keys(counts);
  if (!keys.length) return null;
  return keys.reduce((acc, key) => acc + (Number(counts[key]) || 0), 0);
}

function StatisticsDashboard({ analytics, role, onStatClick, onRecruiterClick, tpTotal, tpStatusCounts, countsLoading = false }) {
  const summary = analytics?.summary && typeof analytics.summary === 'object' ? analytics.summary : {};

  // Filters should narrow the table and status tabs, not rewrite the global sourced KPI.
  // The denominator MUST share scope with the Shortlisted numerator (both from
  // status_counts, which excludes the status filter); using the filter-narrowed
  // tpTotal here made conversion exceed 100% (e.g. 1033%) whenever a status tab
  // was selected.
  const statusCountTotal = sumStatusCounts(tpStatusCounts);
  const displayTotal = statusCountTotal != null
    ? statusCountTotal
    : (tpTotal != null ? tpTotal : summary.total_sourced || 0);
  const displayShortlisted = tpStatusCounts?.Shortlisted != null ? tpStatusCounts.Shortlisted : summary.shortlisted || 0;
  const displayPipeline = tpStatusCounts || summary.pipeline_health || {};

  const conversionRate = countsLoading ? '...' : displayTotal > 0
    ? Math.min(100, Math.max(0, Math.round((displayShortlisted / displayTotal) * 100)))
    : 0;

  const cards = [
    { label: 'Total sourced', value: countsLoading ? '...' : displayTotal, icon: UserPlus, tone: 'warm' },
    { label: 'Shortlisted', value: countsLoading ? '...' : displayShortlisted, icon: Activity, tone: 'emerald', status: 'Shortlisted' },
    { label: 'Conversion rate', value: countsLoading ? '...' : `${conversionRate}%`, icon: BarChart2, tone: 'slate' },
    { label: 'In follow up', value: countsLoading ? '...' : displayPipeline?.['Followup / In conversation'] || 0, icon: MessageSquareMore, tone: 'amber', status: 'Followup / In conversation' },
  ];

  const recruiterPerf = analytics?.recruiter_performance || [];
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

function CompactMetricsStrip({ analytics, tpTotal, tpStatusCounts, expanded, onToggle, countsLoading = false }) {
  const summary = analytics?.summary || {};
  // Denominator shares scope with the Shortlisted numerator (see StatisticsDashboard).
  const statusCountTotal = sumStatusCounts(tpStatusCounts);
  const totalSourced = statusCountTotal != null
    ? statusCountTotal
    : (tpTotal != null ? tpTotal : summary.total_sourced || 0);
  const shortlisted = tpStatusCounts?.Shortlisted != null ? tpStatusCounts.Shortlisted : summary.shortlisted || 0;
  const followUp = (tpStatusCounts || summary.pipeline_health || {})['Followup / In conversation'] || 0;
  const conversion = countsLoading ? '...' : totalSourced > 0 ? Math.min(100, Math.max(0, Math.round((shortlisted / totalSourced) * 100))) : 0;
  const items = [
    ['Total sourced', countsLoading ? '...' : totalSourced],
    ['Shortlisted', countsLoading ? '...' : shortlisted],
    ['Conversion', countsLoading ? '...' : `${conversion}%`],
    ['Follow up', countsLoading ? '...' : followUp],
  ];

  return (
    <div style={{
      padding: '9px 20px',
      borderTop: `1px solid rgba(226, 232, 240, 0.9)`,
      display: 'flex',
      alignItems: 'center',
      gap: 14,
      flexWrap: 'wrap',
      background: 'rgba(248,250,252,0.72)',
    }}>
      {items.map(([label, value]) => (
        <div key={label} style={{ display: 'flex', alignItems: 'baseline', gap: 6, minWidth: 0 }}>
          <span style={{ fontSize: 11, color: '#64748b', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            {label}
          </span>
          <span style={{ fontSize: 13, color: '#0f172a', fontWeight: 850 }}>
            {typeof value === 'number' ? value.toLocaleString() : value}
          </span>
        </div>
      ))}
      <button
        type="button"
        onClick={onToggle}
        style={{
          marginLeft: 'auto',
          display: 'inline-flex',
          alignItems: 'center',
          gap: 6,
          border: '1px solid rgba(203, 213, 225, 0.9)',
          background: '#fff',
          color: '#334155',
          borderRadius: 10,
          padding: '6px 10px',
          fontSize: 11.5,
          fontWeight: 800,
          cursor: 'pointer',
        }}
      >
        {expanded ? 'Hide metrics' : 'Show metrics'}
        {expanded ? <ChevronLeft size={13} style={{ transform: 'rotate(90deg)' }} /> : <ChevronRight size={13} style={{ transform: 'rotate(90deg)' }} />}
      </button>
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
      if (isDocumentVisible()) void loadMessages({ silent: true, targetPlatform: activePlatformRef.current });
    }, 2000);

    return () => clearInterval(interval);
  }, [activeThreadSyncing, loadMessages]);

  // ── Standard background poll — keep both tabs fresh and apply results ──────
  useEffect(() => {
    const interval = setInterval(() => {
      if (!isDocumentVisible()) return;
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
  const scopeTotal = useAppStore(state => state.tpScopeTotal);
  const scopeStatusCounts = useAppStore(state => state.tpScopeStatusCounts);
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
  const fetchTalentPoolSummary = useAppStore(state => state.fetchTalentPoolSummary);
  const fetchTalentPoolIndex = useAppStore(state => state.fetchTalentPoolIndex);
  const buildTalentPoolQueryKey = useAppStore(state => state.buildTalentPoolQueryKey);
  const talentPoolCache = useAppStore(state => state.talentPoolCache);
  const talentPoolIndex = useAppStore(state => state.talentPoolIndex);
  const updateTpCandidate = useAppStore(state => state.updateTpCandidate);
  const talentPoolViewScope = useAppStore(state => state.talentPoolViewScope);
  const talentPoolRecruiterFilterId = useAppStore(state => state.talentPoolRecruiterFilterId);
  const talentPoolRoleFilterId = useAppStore(state => state.talentPoolRoleFilterId);
  const setTalentPoolView = useAppStore(state => state.setTalentPoolView);
  const setTalentPoolRoleFilter = useAppStore(state => state.setTalentPoolRoleFilter);
  const invalidateTalentPoolCaches = useAppStore(state => state.invalidateTalentPoolCaches);
  const fetchRecruiters = useAppStore(state => state.fetchRecruiters);
  const recruiters = useAppStore(state => state.recruiters);
  const tpAiRunFocus = useAppStore(state => state.tpAiRunFocus);
  const startTpAiRunFocus = useAppStore(state => state.startTpAiRunFocus);
  const exitTpAiRunFocus = useAppStore(state => state.exitTpAiRunFocus);

  // Master Library should remain fully editable for admins too.
  const poolReadOnly = false;
  const candidateDetailQs = () => {
    if (role !== 'admin') return '';
    let q = `view_scope=${encodeURIComponent(talentPoolViewScope || 'master')}`;
    if (talentPoolViewScope === 'recruiter_pools' && talentPoolRecruiterFilterId) {
      q += `&recruiter_filter_id=${encodeURIComponent(talentPoolRecruiterFilterId)}`;
    }
    return `?${q}`;
  };

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
  const [loadNotice, setLoadNotice] = useState('');
  const [loadError, setLoadError] = useState('');
  const [isFilterCollapsed, setIsFilterCollapsed] = useState(() => {
    return localStorage.getItem('tp-filter-collapsed') === 'true';
  });
  const [metricsExpanded, setMetricsExpanded] = useState(false);

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
  const [allFilteredSelected, setAllFilteredSelected] = useState(false);
  const aiColumns = useAppStore(state => state.aiColumns);
  const setAiColumns = useAppStore(state => state.setAiColumns);
  const aiColumnsLoading = useAppStore(state => state.aiColumnsLoading);
  const setAiColumnsLoading = useAppStore(state => state.setAiColumnsLoading);
  const [aiCellDrawerOpen, setAiCellDrawerOpen] = useState(false);
  const [aiCellDrawerLoading, setAiCellDrawerLoading] = useState(false);
  const [aiCellDrawerDetail, setAiCellDrawerDetail] = useState(null);
  const [aiCellDrawerTitle, setAiCellDrawerTitle] = useState('');
  const [showAddToListModal, setShowAddToListModal] = useState(false);
  const [showAssignModal, setShowAssignModal] = useState(false);
  const [assignTargetRecruiterId, setAssignTargetRecruiterId] = useState('');
  const [assignTargetRoleId, setAssignTargetRoleId] = useState('');
  const [assignRecruiterRoles, setAssignRecruiterRoles] = useState([]);
  const [assignRolesLoading, setAssignRolesLoading] = useState(false);
  const [assignBusy, setAssignBusy] = useState(false);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadHeaders, setUploadHeaders] = useState([]);
  const [uploadSuggested, setUploadSuggested] = useState({});
  const [uploadMappingDetails, setUploadMappingDetails] = useState({});
  const [uploadRequiredTargets, setUploadRequiredTargets] = useState(['first_name', 'last_name', 'linkedin', 'city', 'title']);
  const [uploadTargetOptions, setUploadTargetOptions] = useState([]);
  const [uploadMapping, setUploadMapping] = useState({});
  const [uploadPreviewBusy, setUploadPreviewBusy] = useState(false);
  const [uploadCommitBusy, setUploadCommitBusy] = useState(false);
  const [uploadRowCount, setUploadRowCount] = useState(0);
  const [uploadProgress, setUploadProgress] = useState(null);
  const [uploadEnrichmentMode, setUploadEnrichmentMode] = useState('none');
  const [recentUploads, setRecentUploads] = useState([]);
  const [scopeRoles, setScopeRoles] = useState([]);
  const [contactInfo, setContactInfo] = useState(readPersistedContactInfo); // { [candidateId]: { email, phone, enriching } }
  const [hasLoadedRecruiterScope, setHasLoadedRecruiterScope] = useState(role !== 'recruiter');
  const analytics = useAppStore(state => state.analytics);
  const fetchAnalytics = useAppStore(state => state.fetchAnalytics);
  const syncOutreachResponses = useAppStore(state => state.syncOutreachResponses);
  const shortlistAndOutreach = useAppStore(state => state.shortlistAndOutreach);
  const updateCandidateField = useAppStore(state => state.updateCandidateField);
  const heyreachCampaignId = useAppStore(state => state.heyreachCampaignId);
  const didInitRef = useRef(false);
  const uploadFileRef = useRef(null);
  const talentPoolRequestSeqRef = useRef(0);
  const tableScrollRef = useRef(null);
  const aiColumnsRequestSeqRef = useRef(0);
  const visibleCandidatesRef = useRef(candidates);
  /** Latest fetchCandidates — avoids admin scope effect re-running when this callback identity changes after each fetch. */
  const fetchCandidatesRef = useRef(null);
  const aiColumnDeleteInFlightRef = useRef(null);
  const aiColumnsInFlightKeyRef = useRef('');
  const linkedInPrewarmRef = useRef({ ids: new Set(), signature: '', ts: 0 });
  const talentPoolNonPageQueryKeyRef = useRef('');
  const loadedMetaKeyRef = useRef('');

  const [isRevalidating, setIsRevalidating] = useState(false);
  const talentPoolScopeReady = role !== 'admin'
    || talentPoolViewScope !== 'recruiter_pools'
    || Boolean(talentPoolRecruiterFilterId);

  useEffect(() => {
    setHasLoadedRecruiterScope(role !== 'recruiter');
  }, [role, user?.id]);

  useEffect(() => {
    if (role === 'recruiter' && scopeTotal != null) {
      setHasLoadedRecruiterScope(true);
    }
  }, [role, scopeTotal]);

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
            const res = await axios.get(`${API_BASE}/candidates/${id}${candidateDetailQs()}`);
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
      if (isDocumentVisible()) await pollOnce();
      if (!cancelled) {
        timer = setTimeout(pollLoop, isDocumentVisible() ? 1000 : 5000);
      }
    };

    pollLoop();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [contactInfo, role, talentPoolViewScope, talentPoolRecruiterFilterId]);

  useEffect(() => {
    if (role === 'admin') {
      fetchRecruiters();
    }
  }, [role, fetchRecruiters]);

  useEffect(() => {
    if (!showAssignModal) {
      setAssignRecruiterRoles([]);
      setAssignTargetRoleId('');
      return undefined;
    }

    let cancelled = false;
    setAssignRolesLoading(true);
    // No owner filter: the backend returns every role for admins (their own plus
    // all recruiters') and only the caller's own roles for recruiters.
    axios.get(`${API_BASE}/roles`, { timeout: 60000 })
      .then(res => {
        if (!cancelled) setAssignRecruiterRoles(res.data?.roles || []);
      })
      .catch(error => {
        if (!cancelled) {
          setAssignRecruiterRoles([]);
          toast.error(error.response?.data?.detail || 'Could not load roles');
        }
      })
      .finally(() => { if (!cancelled) setAssignRolesLoading(false); });

    return () => { cancelled = true; };
  }, [showAssignModal]);

  useEffect(() => {
    if (
      role === 'admin' &&
      talentPoolViewScope === 'recruiter_pools' &&
      !talentPoolRecruiterFilterId &&
      recruiters.length > 0
    ) {
      setTalentPoolView('recruiter_pools', recruiters[0].id);
    }
  }, [role, talentPoolViewScope, talentPoolRecruiterFilterId, recruiters, setTalentPoolView]);

  useEffect(() => {
    let cancelled = false;
    const params = new URLSearchParams();
    if (role === 'admin' && talentPoolViewScope === 'recruiter_pools' && !talentPoolRecruiterFilterId) {
      setScopeRoles([]);
      setTalentPoolRoleFilter('');
      return undefined;
    }
    if (role === 'admin' && talentPoolViewScope === 'master') {
      params.set('view_scope', 'master');
    }
    if (role === 'admin' && talentPoolViewScope === 'recruiter_pools' && talentPoolRecruiterFilterId) {
      params.set('view_scope', 'recruiter_pools');
      params.set('owner_user_id', talentPoolRecruiterFilterId);
    }
    if (role === 'admin' && talentPoolViewScope === 'all_recruiter_pools') {
      setScopeRoles([]);
      setTalentPoolRoleFilter('');
      return undefined;
    }
    const url = params.toString() ? `${API_BASE}/roles?${params.toString()}` : `${API_BASE}/roles`;
    axios.get(url, { timeout: 60000 })
      .then(res => {
        if (cancelled) return;
        const nextRoles = res.data?.roles || [];
        setScopeRoles(nextRoles);
        if (talentPoolRoleFilterId && !nextRoles.some(r => String(r.id) === String(talentPoolRoleFilterId))) {
          setTalentPoolRoleFilter('');
        }
      })
      .catch(() => {
        if (!cancelled) setScopeRoles([]);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [role, talentPoolViewScope, talentPoolRecruiterFilterId, setTalentPoolRoleFilter]);

  const adminScopeInitRef = useRef(false);

  useEffect(() => {
    try {
      window.localStorage.setItem(CONTACT_INFO_STORAGE_KEY, JSON.stringify(contactInfo));
    } catch {
      // Ignore storage write errors (private mode, quota, etc.)
    }
  }, [contactInfo]);

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
    if (!talentPoolScopeReady) {
      setMeta({ companies: [], cities: [], products: [], statuses: [], recruiters: [], location_types: [], titles: [] });
      fetchTalentPoolSummary();
      return undefined;
    }

    const qs = useAppStore.getState().buildTalentPoolScopeQuery();
    const metaUrl = qs
      ? `${API_BASE}/candidates/browse/meta?${qs}`
      : `${API_BASE}/candidates/browse/meta`;
    if (loadedMetaKeyRef.current !== metaUrl) {
      axios.get(metaUrl).then(r => {
        if (!cancelled) {
          const data = r.data || {};
          loadedMetaKeyRef.current = metaUrl;
          setMeta({
            companies: Array.isArray(data.companies) ? data.companies : [],
            cities: Array.isArray(data.cities) ? data.cities : [],
            titles: Array.isArray(data.titles) ? data.titles : [],
            products: Array.isArray(data.products) ? data.products : [],
            statuses: Array.isArray(data.statuses) ? data.statuses : [],
            recruiters: Array.isArray(data.recruiters) ? data.recruiters : [],
            location_types: Array.isArray(data.location_types) ? data.location_types : [],
          });
        }
      }).catch(error => {
        console.error('Failed to fetch talent pool filter metadata:', error);
      });
    }

    fetchTalentPoolSummary();

    return () => {
      cancelled = true;
    };
  }, [fetchTalentPoolSummary, role, talentPoolViewScope, talentPoolRecruiterFilterId, talentPoolRoleFilterId, talentPoolScopeReady]);

  useEffect(() => {
    if (role !== 'recruiter') {
      setRecentUploads([]);
      return undefined;
    }
    let cancelled = false;
    axios
      .get(`${API_BASE}/candidates/uploads?limit=8`, { timeout: 60000 })
      .then(r => {
        if (!cancelled) setRecentUploads(r.data.uploads || []);
      })
      .catch(() => { });
    return () => {
      cancelled = true;
    };
    // Do not depend on uploadCommitBusy — toggling it refetched on every import attempt and
    // could amplify errors; refresh explicitly after a successful commit.
  }, [role, uploadOpen]);

  const handleShortlisted = async (candidateId) => {
    if (poolReadOnly) return null;
    setShortlistingId(candidateId);
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
          enriching: d.contact_enriching
            && !resolveContactValue(d.email, prev[candidateId]?.email)
            && !resolveContactValue(d.phone, prev[candidateId]?.phone)
        }
      }));
      setShortlistCard(d);
      invalidateTalentPoolCaches();
      fetchTalentPoolSummary({ force: true, freshnessMs: 0 });
      void fetchCandidates(page);
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
    }
  };

  const updateFieldAndMaybeShortlist = async (candidateId, data) => {
    if (poolReadOnly) {
      toast.error('Master library is read-only');
      return { success: false };
    }
    const res = await updateCandidateField(candidateId, data);
    if (!res.success) return res;

    const field = Object.keys(data)[0];
    const value = data[field];
    const isValuable = value != null && value !== '' && !['na', 'n/a', 'none', ''].includes(String(value).toLowerCase());

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
          invalidateTalentPoolCaches();
          fetchTalentPoolSummary({ force: true, freshnessMs: 0 });
          void fetchCandidates(page);
        } catch (err) {
          console.error('Auto-shortlist failed:', err);
        }
      }
    }
    return res;
  };

  const focusCandidateIds = useMemo(
    () => Array.isArray(tpAiRunFocus?.candidateIds)
      ? tpAiRunFocus.candidateIds.map(Number).filter(Number.isFinite)
      : [],
    [tpAiRunFocus],
  );
  const isAiRunFocusActive = focusCandidateIds.length > 0;
  const focusCandidateIdsKey = useMemo(() => focusCandidateIds.join(','), [focusCandidateIds]);
  const talentPoolNonPageQueryKey = useMemo(() => JSON.stringify({
    pageSize,
    globalSearch,
    filters,
    activeStatusTab,
    sortBy,
    sortDir,
    roleId: talentPoolRoleFilterId,
    viewScope: talentPoolViewScope,
    recruiterId: talentPoolRecruiterFilterId,
    focus: focusCandidateIdsKey,
    scopeReady: talentPoolScopeReady,
    focusStartedAt: tpAiRunFocus?.startedAt || null,
  }), [
    pageSize,
    globalSearch,
    filters,
    activeStatusTab,
    sortBy,
    sortDir,
    talentPoolRoleFilterId,
    talentPoolViewScope,
    talentPoolRecruiterFilterId,
    focusCandidateIdsKey,
    talentPoolScopeReady,
    tpAiRunFocus?.startedAt,
  ]);
  const talentPoolQueryKey = useMemo(
    () => `${isAiRunFocusActive ? 1 : page}|${talentPoolNonPageQueryKey}`,
    [isAiRunFocusActive, page, talentPoolNonPageQueryKey],
  );
  const displayedCandidates = Array.isArray(candidates) ? candidates : [];

  useEffect(() => {
    if (isAiRunFocusActive && tableScrollRef.current) {
      tableScrollRef.current.scrollLeft = 0;
    }
  }, [isAiRunFocusActive]);

  useEffect(() => {
    if (!loading || displayedCandidates.length) {
      setLoadNotice('');
      return undefined;
    }
    const timer = window.setTimeout(() => {
      setLoadNotice('Candidate data is still loading from the server. Keeping the page responsive while it finishes.');
    }, 1200);
    return () => window.clearTimeout(timer);
  }, [loading, displayedCandidates.length]);

  const displayedTotal = total;
  const displayedTotalPages = totalPages;
  const displayedStatusCounts = statusCounts;
  const recruiterCountLoading = role === 'recruiter' && !hasLoadedRecruiterScope;
  const metricsTotal = recruiterCountLoading ? null : (scopeTotal ?? total);
  const metricsStatusCounts = recruiterCountLoading ? {} : (scopeStatusCounts || statusCounts);
  const allVisibleSelected = displayedCandidates.length > 0 && displayedCandidates.every((candidate) => selectedIds.has(candidate.id));
  const statusFilterOptions = useMemo(() => uniqueSortedOptions(
    RECRUITMENT_STAGES,
    meta.statuses,
    scopeStatusCounts,
    statusCounts,
    displayedCandidates.map(candidate => candidate?.status || 'To be started'),
  ), [meta.statuses, scopeStatusCounts, statusCounts, displayedCandidates]);
  const recruiterFilterOptions = useMemo(() => uniqueSortedOptions(
    meta.recruiters,
    talentPoolIndex?.rows?.map(candidate => candidate?.created_by),
    displayedCandidates.map(candidate => candidate?.created_by),
  ), [meta.recruiters, talentPoolIndex?.rows, displayedCandidates]);

  // Stable across new array instances with the same visible ids — avoids hammering GET /ai-columns.
  const aiColumnVisibleIdsKey = (isAiRunFocusActive ? focusCandidateIds : displayedCandidates.map((c) => c.id))
    .filter((id) => id != null && id !== '')
    .slice()
    .sort((a, b) => Number(a) - Number(b))
    .join(',');

  const fetchCandidates = useCallback(async (pg = 1, options = {}) => {
    let requestId = 0;
    try {
      if (!talentPoolScopeReady) {
        setLoading(false);
        setIsRevalidating(false);
        return;
      }
      requestId = ++talentPoolRequestSeqRef.current;
      const paramsString = buildTalentPoolParamsString({
        page: isAiRunFocusActive ? 1 : pg,
        pageSize,
        globalSearch,
        filters,
        activeStatusTab,
        sortBy,
        sortDir,
        candidateIds: isAiRunFocusActive ? focusCandidateIds : [],
      });

      console.log("[DEBUG COMPONENT] fetchCandidates start:", {
        requestId,
        isAiRunFocusActive,
        focusCandidateIds,
        paramsString
      });

      const cache = useAppStore.getState().talentPoolCache || { data: null, lastParamsString: null };
      const cachedData = cache.lastParamsString === buildTalentPoolQueryKey(paramsString) ? cache.data : null;
      const hasData = !isAiRunFocusActive && visibleCandidatesRef.current.length > 0;

      setLoadError('');
      if (!cachedData && !hasData) {
        setLoading(true);
      } else {
        setIsRevalidating(true);
      }

      const res = await fetchTalentPool(paramsString, { force: options.force === true });

      console.log("[DEBUG COMPONENT] fetchCandidates response resolved:", {
        requestId,
        latestRequestId: talentPoolRequestSeqRef.current,
        success: res.success,
        stale: res.stale,
        blocked: res.blocked
      });

      if (requestId !== talentPoolRequestSeqRef.current) {
        console.log("[DEBUG COMPONENT] fetchCandidates ignoring response due to requestId mismatch:", {
          requestId,
          latestRequestId: talentPoolRequestSeqRef.current
        });
        return;
      }

      if (res.success && res.data) {
        setIsSemanticSearch(res.data.is_semantic_search || false);
        mergeContactInfoFromRows(res.data.candidates);
        if (role === 'recruiter') {
          setHasLoadedRecruiterScope(true);
        }
      } else if (!res.success && !res.stale && !res.blocked) {
        setLoadError(res.error || 'Failed to load candidates');
      }
    } catch (e) {
      console.error('Failed to fetch talent pool:', e);
      setLoadError(e?.message || 'Failed to load candidates');
    } finally {
      if (requestId === talentPoolRequestSeqRef.current) {
        setLoading(false);
        setIsRevalidating(false);
      }
    }
  }, [globalSearch, filters, activeStatusTab, sortBy, sortDir, pageSize, talentPoolRoleFilterId, mergeContactInfoFromRows, fetchTalentPool, buildTalentPoolQueryKey, isAiRunFocusActive, focusCandidateIds, talentPoolScopeReady, role]);

  fetchCandidatesRef.current = fetchCandidates;

  useEffect(() => {
    if (role !== 'admin') return;
    if (!talentPoolScopeReady) {
      invalidateTalentPoolCaches({ clearRows: true });
      return;
    }
    if (!adminScopeInitRef.current) {
      adminScopeInitRef.current = true;
      return;
    }
    invalidateTalentPoolCaches();
    setPage(1);
    const run = fetchCandidatesRef.current;
    if (run) void run(1);
  }, [role, talentPoolViewScope, talentPoolRecruiterFilterId, talentPoolRoleFilterId, invalidateTalentPoolCaches, talentPoolScopeReady]);

  useEffect(() => {
    visibleCandidatesRef.current = displayedCandidates;
  }, [displayedCandidates]);

  useEffect(() => {
    if (page > displayedTotalPages) {
      setPage(displayedTotalPages);
    }
  }, [page, displayedTotalPages]);

  // Pre-warm LinkedIn cache once per visible batch; AI-column polling re-renders the grid often.
  useEffect(() => {
    if (!displayedCandidates.length) return;
    const statusesToPrewarm = new Set(['replied', 'message_sent', 'connection_accepted', 'in_campaign']);
    const allIds = displayedCandidates
      .filter(c => statusesToPrewarm.has(c.li_status))
      .map(c => c.id)
      .filter(Boolean);
    if (!allIds.length) return;

    const now = Date.now();
    const cache = linkedInPrewarmRef.current;
    const nextIds = allIds.filter(id => !cache.ids.has(id)).slice(0, 12);
    const signature = allIds.join(',');
    const sameBatchRecently = signature === cache.signature && now - cache.ts < 10 * 60 * 1000;
    if (!nextIds.length || sameBatchRecently) return;

    linkedInPrewarmRef.current = {
      ids: new Set([...cache.ids, ...nextIds]),
      signature,
      ts: now,
    };
    useAppStore.getState().prewarmLinkedInCache(nextIds).catch(console.error);
  }, [displayedCandidates]);

  useEffect(() => {
    const previousNonPageKey = talentPoolNonPageQueryKeyRef.current;
    const nonPageChanged = previousNonPageKey && previousNonPageKey !== talentPoolNonPageQueryKey;
    talentPoolNonPageQueryKeyRef.current = talentPoolNonPageQueryKey;

    if (nonPageChanged && !isAiRunFocusActive && page !== 1) {
      setTpPagination(1, pageSize);
      return undefined;
    }

    clearTimeout(debounceRef.current);
    const delay = didInitRef.current ? 120 : 0;
    didInitRef.current = true;
    debounceRef.current = setTimeout(() => {
      const run = fetchCandidatesRef.current;
      if (run) void run(isAiRunFocusActive ? 1 : page);
    }, delay);
    return () => clearTimeout(debounceRef.current);
  }, [talentPoolQueryKey, talentPoolNonPageQueryKey, isAiRunFocusActive, page, pageSize, setTpPagination]);

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
    let nextDelayMs = OUTREACH_REPLY_SYNC_INTERVAL_MS;

    const syncTalentPoolReplies = async () => {
      if (document.visibilityState !== 'visible') {
        timeoutId = window.setTimeout(syncTalentPoolReplies, nextDelayMs);
        return;
      }
      const res = await syncOutreachResponses(0);
      const updatedCount = Number(res?.data?.updated_count || 0);
      if (cancelled) {
        return;
      }

      if (res?.success && updatedCount > 0) {
        fetchRef.current(pageRef.current);
      }

      nextDelayMs = res?.success
        ? OUTREACH_REPLY_SYNC_INTERVAL_MS
        : Math.min(nextDelayMs * 2, OUTREACH_REPLY_SYNC_MAX_BACKOFF_MS);
      timeoutId = window.setTimeout(syncTalentPoolReplies, nextDelayMs);
    };

    timeoutId = window.setTimeout(syncTalentPoolReplies, OUTREACH_REPLY_SYNC_INITIAL_DELAY_MS);

    return () => {
      cancelled = true;
      if (timeoutId !== null) {
        clearTimeout(timeoutId);
      }
    };
  }, [syncOutreachResponses]);


  const setFilter = (key, val) => {
    if (key === 'status') {
      setActiveStatusTab('');
    }
    setFilters(prev => ({ ...prev, [key]: val }));
    setTpPagination(1, pageSize);
  };
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
    setTalentPoolRoleFilter('');
    });
    setTpPagination(1, pageSize);
  };
  const hasFilters = filters && Object.keys(filters).some(key => {
    if (key.toLowerCase().includes('input')) return false;
    const v = filters[key];
    if (key === 'min_exp') return v > 0;
    if (key === 'max_exp') return v < 40;
    return Array.isArray(v) ? v.length > 0 : !!v;
  }) || Boolean(talentPoolRoleFilterId);

  // ── AI Columns (definition-backed, Clay-style) ───────────────────────────
  const [aiColumnModal, setAiColumnModal] = useState(null);

  const fetchAiColumns = useCallback(async () => {
    const requestedVisibleIdsKey = aiColumnVisibleIdsKey;
    if (!talentPoolScopeReady) {
      setAiColumns([]);
      setAiColumnsLoading(false);
      return;
    }
    if (!requestedVisibleIdsKey) {
      setAiColumnsLoading(false);
      return;
    }
    const params = new URLSearchParams();
    if (requestedVisibleIdsKey) params.set('candidate_ids', requestedVisibleIdsKey);
    if (talentPoolViewScope) params.set('view_scope', talentPoolViewScope);
    if (talentPoolRecruiterFilterId) params.set('recruiter_filter_id', talentPoolRecruiterFilterId);
    if (talentPoolRoleFilterId) params.set('role_id', talentPoolRoleFilterId);
    const requestKey = params.toString();
    if (aiColumnsInFlightKeyRef.current === requestKey) {
      return;
    }
    const requestId = ++aiColumnsRequestSeqRef.current;
    aiColumnsInFlightKeyRef.current = requestKey;
    setAiColumnsLoading(true);
    try {
      const res = await longOperationAxios.get(`${API_BASE}/ai-columns?${requestKey}`, { timeout: 120000 });
      if (requestId !== aiColumnsRequestSeqRef.current) return;
      const nextColumns = res.data?.columns || [];
      setAiColumns(prev => mergeAiColumnDefinitions(prev || [], nextColumns));
    } catch (error) {
      if (requestId !== aiColumnsRequestSeqRef.current) return;
      console.error('Failed to fetch AI columns', error);
      toast.error(
        getRequestErrorMessage(error, 'Could not load smart column data. Check your connection and try refreshing.'),
        { id: 'talent-pool-ai-columns-fetch' },
      );
    } finally {
      if (aiColumnsInFlightKeyRef.current === requestKey) {
        aiColumnsInFlightKeyRef.current = '';
      }
      if (requestId === aiColumnsRequestSeqRef.current) {
        setAiColumnsLoading(false);
      }
    }
  }, [aiColumnVisibleIdsKey, talentPoolViewScope, talentPoolRecruiterFilterId, talentPoolRoleFilterId, talentPoolScopeReady]);

  useEffect(() => {
    const t = window.setTimeout(() => {
      void fetchAiColumns();
    }, 400);
    return () => window.clearTimeout(t);
  }, [fetchAiColumns]);

  useEffect(() => {
    // Poll faster (2s) when any column has an active run OR any visible cell is running/queued
    const hasActiveRuns = (aiColumns || []).some(column => {
      const status = column.latest_run?.status;
      if (status === 'queued' || status === 'running') return true;
      // Also check if any visible cells are still in-progress
      const cells = Object.values(column.cells_by_candidate || {});
      return cells.some(cell => cell.status === 'queued' || cell.status === 'running');
    });
    if (!hasActiveRuns) return undefined;
    const timer = window.setInterval(() => {
      if (isDocumentVisible()) void fetchAiColumns();
    }, 2000);
    return () => window.clearInterval(timer);
  }, [aiColumns, fetchAiColumns]);

  const dynamicAiCols = useMemo(() => {
    return (aiColumns || []).flatMap(definition => {
      const outputs = Array.isArray(definition.output_schema) && definition.output_schema.length
        ? definition.output_schema
        : [{ key: 'result', label: 'Result', primary: true }];
      return outputs.map(output => ({
        key: `__ai__${definition.id}__${output.key}`,
        label: outputs.length === 1 ? definition.name : `${definition.name} · ${output.label}`,
        shortLabel: output.label,
        w: 220,
        isAiCol: true,
        definitionId: definition.id,
        definition,
        outputKey: output.key,
        isPrimaryOutput: Boolean(output.primary),
      }));
    });
  }, [aiColumns]);

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
    ...dynamicAiCols,
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

  const runUploadPreview = async (file) => {
    if (!file) return;
    const fd = new FormData();
    fd.append('file', file);
    fd.append('use_llm', 'false');
    setUploadPreviewBusy(true);
    try {
      const res = await axios.post(`${API_BASE}/candidates/upload/preview`, fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120000,
      });
      const headers = res.data.headers || [];
      const sm = res.data.suggested_mapping || {};
      setUploadHeaders(headers);
      setUploadSuggested(sm);
      setUploadMappingDetails(res.data.mapping_details || {});
      setUploadRequiredTargets(res.data.required_targets || ['first_name', 'last_name', 'linkedin', 'city', 'title']);
      setUploadTargetOptions(res.data.target_options || []);
      setUploadRowCount(Number(res.data.row_count) || 0);
      setUploadProgress(null);
      // Bug-4 fix: auto-enable verified enrichment for Apify/LinkedIn-style CSVs
      // that already contain structured work-history columns (experiences/*).
      // The user can still override this in the modal.
      const hasExperienceCols = headers.some(h => /^experiences\/\d+\//i.test(String(h || '')));
      setUploadEnrichmentMode(hasExperienceCols ? 'verified_profile' : 'none');
      const init = {};
      for (const h of headers) {
        init[h] = sm[h] || 'ignore';
      }
      setUploadMapping(init);
      setUploadFile(file);
      setUploadOpen(true);
    } catch (e) {
      toast.error(e.response?.data?.detail || 'Preview failed');
    } finally {
      setUploadPreviewBusy(false);
    }
  };


  const pollUploadStatus = async (uploadId) => {
    for (let attempt = 0; attempt < 900; attempt += 1) {
      const res = await axios.get(`${API_BASE}/candidates/uploads/${uploadId}`, { timeout: 60000 });
      const next = res.data || {};
      setUploadProgress(next);
      if (['completed', 'completed_with_errors', 'failed'].includes(String(next.status || '').toLowerCase())) {
        return next;
      }
      await new Promise(resolve => window.setTimeout(resolve, 700));
    }
    throw new Error('Upload is still running. Check recent uploads for its status.');
  };

  const commitUpload = async () => {
    if (!uploadFile) return;
    const req = new Set(uploadRequiredTargets?.length ? uploadRequiredTargets : ['first_name', 'last_name', 'linkedin', 'city', 'title']);
    const used = new Set(Object.values(uploadMapping).filter(v => v && v !== 'ignore'));
    if (![...req].every(x => used.has(x))) {
      toast.error('Map first name, last name, LinkedIn, city, and title');
      return;
    }
    const fd = new FormData();
    fd.append('file', uploadFile);
    fd.append('mapping_json', JSON.stringify(uploadMapping));
    fd.append('enrichment_mode', uploadEnrichmentMode || 'none');
    setUploadCommitBusy(true);
    try {
      const res = await axios.post(`${API_BASE}/candidates/upload/commit`, fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120000,
      });
      setUploadProgress(res.data || {});
      const d = await pollUploadStatus(res.data?.upload_id);
      if (String(d.status || '').toLowerCase() === 'failed') {
        toast.error(d.error_message || 'Upload failed');
        return;
      }
      const ins = Number(d.inserted) || 0;
      const upd = Number(d.updated) || 0;
      const skp = Number(d.skipped) || 0;
      const rowCount = Number(d.row_count);
      const errs = Array.isArray(d.errors) ? d.errors : [];
      const toastParts = [];
      if (Number.isFinite(rowCount) && rowCount > 0) {
        toastParts.push(`${rowCount} spreadsheet rows`);
      }
      if (ins > 0) toastParts.push(`${ins} new added to your pool`);
      if (upd > 0) {
        toastParts.push(
          `${upd} already in your pool (profile refreshed — same LinkedIn as an existing row in the file or a past import)`,
        );
      }
      if (skp > 0) toastParts.push(`${skp} row${skp === 1 ? '' : 's'} skipped`);
      const successMsg =
        toastParts.length > 0 ? `Import complete: ${toastParts.join('. ')}.` : 'Import complete.';
      toast.success(successMsg);
      if (errs.length > 0) {
        const preview = errs.slice(0, 3).join(' ');
        const warnBody = preview.length > 300 ? `${preview.slice(0, 297)}…` : preview;
        toast.warning(`Some rows had issues: ${warnBody}`);
      }
      invalidateTalentPoolCaches();
      fetchTalentPoolSummary({ force: true, freshnessMs: 0 });
      await fetchCandidates(1);
      const runIndex = () => {
        void fetchTalentPoolIndex().catch(() => { });
      };
      if (typeof window !== 'undefined' && typeof window.requestIdleCallback === 'function') {
        window.requestIdleCallback(runIndex, { timeout: 5000 });
      } else if (typeof window !== 'undefined') {
        window.setTimeout(runIndex, 1500);
      } else {
        runIndex();
      }
      void axios
        .get(`${API_BASE}/candidates/uploads?limit=8`, { timeout: 60000 })
        .then(r => setRecentUploads(r.data.uploads || []))
        .catch(() => { });
    } catch (e) {
      toast.error(e.response?.data?.detail || e.message || 'Upload failed');
    } finally {
      setUploadCommitBusy(false);
    }
  };

  const runBulkAssign = async () => {
    if (!assignTargetRecruiterId) {
      toast.error('Select a recruiter');
      return;
    }
    setAssignBusy(true);
    try {
      const rawIds = [...selectedIds].map((id) => Number(id)).filter((n) => Number.isFinite(n));
      if (rawIds.length === 0) {
        toast.error('No valid candidate ids selected');
        return;
      }
      const res = await axios.post(
        `${API_BASE}/admin/candidates/assign-to-recruiter`,
        {
          master_candidate_ids: rawIds,
          recruiter_user_id: Number(assignTargetRecruiterId),
          role_id: assignTargetRoleId ? Number(assignTargetRoleId) : null,
        },
        { timeout: 120000 },
      );
      const results = res.data?.results || [];
      const errs = results.filter(r => r.error);
      if (results.length > 0 && errs.length === results.length) {
        toast.error(errs.map(e => `#${e.master_id}: ${e.error}`).join('; ') || 'Assign failed');
        return;
      }
      if (errs.length > 0) {
        toast.warning(
          `Assigned ${results.length - errs.length} of ${results.length}. ${errs.map(e => `#${e.master_id}: ${e.error}`).join('; ')}`,
        );
      } else {
        const roleSuffix = res.data?.role_name ? ` to ${res.data.role_name}` : ' to recruiter pool';
        toast.success(results.length <= 1 ? `Profile assigned${roleSuffix}` : `Assigned ${results.length} profiles${roleSuffix}`);
      }
      setShowAssignModal(false);
      setAssignTargetRoleId('');
      setAssignRecruiterRoles([]);
      setSelectedIds(new Set());
      invalidateTalentPoolCaches();
      fetchTalentPoolSummary({ force: true, freshnessMs: 0 });
      await fetchTalentPoolIndex({ force: true });
      await fetchCandidates(page);
    } catch (e) {
      console.error('Assign request failed', e?.response?.status, e?.response?.data, e?.message);
      const data = e.response?.data;
      const det = data?.detail;
      let msg = 'Assign failed';
      if (typeof det === 'string') msg = det;
      else if (Array.isArray(det)) {
        msg = det.map(x => (x && typeof x === 'object' && x.msg != null ? x.msg : JSON.stringify(x))).join('; ');
      } else if (det != null && typeof det === 'object') msg = JSON.stringify(det);
      else if (data && typeof data === 'object' && !det) msg = JSON.stringify(data);
      else if (e.response?.status) msg = `Assign failed (HTTP ${e.response.status})`;
      else if (e.code === 'ECONNABORTED') msg = 'Assign timed out — try again';
      else if (e.message) msg = e.message;
      toast.error(msg);
    } finally {
      setAssignBusy(false);
    }
  };

  const openAiColumnEditor = (definition) => {
    if (!definition?.id) {
      toast.error('Invalid smart column id — try refreshing the page');
      return;
    }
    setAiColumnModal({ mode: 'edit', definition });
  };

  const deleteAiColumn = async (definition) => {
    if (!window.confirm(`Delete smart column "${definition?.name}"?`)) return;
    const rawId = definition?.id;
    const id = typeof rawId === 'string' ? Number(rawId) : rawId;
    if (rawId == null || id == null || !Number.isFinite(id)) {
      toast.error('Invalid smart column id — try refreshing the page');
      return;
    }
    if (aiColumnDeleteInFlightRef.current === id) return;
    aiColumnDeleteInFlightRef.current = id;
    const previousColumns = aiColumns;
    setAiColumns(prev => (prev || []).filter(col => Number(col.id) !== id));
    try {
      await longOperationAxios.delete(`${API_BASE}/ai-columns/${id}`, { timeout: 120000 });
      toast.success('Smart column removed');
      void fetchAiColumns();
    } catch (error) {
      setAiColumns(previousColumns || []);
      console.error('deleteAiColumn failed', { id, status: error?.response?.status, data: error?.response?.data });
      toast.error(getRequestErrorMessage(error, 'Failed to delete smart column'));
      void fetchAiColumns();
    } finally {
      if (aiColumnDeleteInFlightRef.current === id) aiColumnDeleteInFlightRef.current = null;
    }
  };

  const upsertOptimisticAiColumn = useCallback((runInfo = {}) => {
    const candidateIdArray = Array.isArray(runInfo.candidateIds)
      ? runInfo.candidateIds.map(Number).filter(Number.isFinite)
      : [];
    const rawDefinition = runInfo.definition || {};
    const definition = {
      ...rawDefinition,
      id: rawDefinition.id ?? runInfo.columnDefinitionId,
      name: rawDefinition.name || runInfo.columnName || 'Smart Column',
    };
    if (!definition.id || !candidateIdArray.length) return;
    const optimisticColumn = createOptimisticAiColumn(definition, candidateIdArray, runInfo.runId || null);
    setAiColumns(prev => {
      const existingIndex = (prev || []).findIndex(col => String(col.id) === String(definition.id));
      if (existingIndex === -1) {
        return [...(prev || []), optimisticColumn];
      }
      return (prev || []).map((col, index) => (
        index === existingIndex
          ? mergeAiColumnDefinitions([col], [optimisticColumn])[0]
          : col
      ));
    });
  }, [setAiColumns]);

  const attachAiColumnRun = useCallback((runInfo = {}) => {
    if (!runInfo?.columnDefinitionId || !runInfo?.runId) return;
    setAiColumns(prev => (prev || []).map(col => {
      if (String(col.id) !== String(runInfo.columnDefinitionId)) return col;
      return {
        ...col,
        latest_run: {
          ...(col.latest_run || {}),
          id: runInfo.runId,
          run_id: runInfo.runId,
          status: col.latest_run?.status || 'queued',
        },
      };
    }));
  }, [setAiColumns]);

  const revertOptimisticAiColumn = useCallback((runInfo = {}) => {
    if (!runInfo?.columnDefinitionId) return;
    if (runInfo.isEditMode) {
      void fetchAiColumns();
      return;
    }
    setAiColumns(prev => (prev || []).filter(col => (
      String(col.id) !== String(runInfo.columnDefinitionId) || !col.__optimistic
    )));
  }, [fetchAiColumns, setAiColumns]);

  const rerunAiColumn = async (definition) => {
    const selectedIdArray = Array.from(selectedIds || []);
    if (selectedIdArray.length === 0) {
      toast.error('Select at least one row first, then run the smart column');
      return;
    }
    const previousColumns = useAppStore.getState().aiColumns || [];
    // Optimistic update: immediately show selected rows as "queued" in the UI
    setAiColumns(prev => (prev || []).map(col => {
      if (col.id !== definition.id) return col;
      const updatedCells = { ...(col.cells_by_candidate || {}) };
      selectedIdArray.forEach(id => {
        updatedCells[id] = { ...(updatedCells[id] || {}), status: 'queued', primary_output: '', outputs: {} };
      });
      return {
        ...col,
        cells_by_candidate: updatedCells,
        latest_run: { ...(col.latest_run || {}), status: 'queued', total: selectedIdArray.length, completed: 0, failed: 0, skipped: 0 },
      };
    }));
    try {
      const runRes = await longOperationAxios.post(`${API_BASE}/ai-columns/run`, {
        column_definition_id: definition.id,
        selection_mode: 'selected_ids',
        selected_ids: selectedIdArray,
        view_scope: talentPoolViewScope,
        recruiter_filter_id: talentPoolRecruiterFilterId,
        role_id: talentPoolRoleFilterId || null,
      });
      startTpAiRunFocus({
        runId: runRes.data?.run_id,
        columnDefinitionId: definition.id,
        columnName: definition.name,
        candidateIds: selectedIdArray,
      });
      toast.success(`Running "${definition.name}" on ${selectedIdArray.length} row(s)…`);
      void fetchAiColumns();
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Failed to run smart column');
      setAiColumns(previousColumns);
    }
  };

  const openAiCellDrawer = async (definition, candidate, outputKey) => {
    setAiCellDrawerOpen(true);
    setAiCellDrawerLoading(true);
    setAiCellDrawerTitle(`${definition?.name || 'Smart Column'} • ${(candidate?.name || `${candidate?.first_name || ''} ${candidate?.last_name || ''}`).trim()}`);
    try {
      const res = await longOperationAxios.get(`${API_BASE}/ai-columns/${definition.id}/cells/${candidate.id}`, {
        timeout: 60000,
      });
      const detail = res.data || null;
      if (detail && outputKey && detail.outputs && detail.outputs[outputKey] && outputKey !== 'primary') {
        detail.primary_output = detail.outputs[outputKey];
      }
      setAiCellDrawerDetail(detail);
    } catch (error) {
      if (error.response?.status === 404) {
        setAiCellDrawerDetail(null);
      } else {
        setAiCellDrawerDetail(null);
        toast.error(error.response?.data?.detail || 'Failed to load AI cell details');
      }
    } finally {
      setAiCellDrawerLoading(false);
    }
  };

  const panelSurface = {
    background: 'rgba(255,255,255,0.84)',
    border: '1px solid rgba(226, 232, 240, 0.92)',
    boxShadow: '0 10px 24px rgba(15,23,42,0.05)',
  };
  const surfaceBorder = 'rgba(226, 232, 240, 0.9)';

  return (
    <div className="talent-pool-page" style={{ fontFamily: '"Inter", -apple-system, sans-serif', display: 'flex', gap: 18, height: '100vh', overflow: 'hidden', padding: '22px', boxSizing: 'border-box' }}>

      {/* ── Left Filter Sidebar ── */}
      <aside style={{
        width: isFilterCollapsed ? 0 : 220,
        minWidth: isFilterCollapsed ? 0 : 220,
        padding: isFilterCollapsed ? 0 : '20px 18px',
        background: panelSurface.background,
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
            options={recruiterFilterOptions}
            placeholder="All Recruiters"
          />
        )}

        <TagFilterInput label="Status" values={filters?.status || []} inputValue={filters?.statusInput || ''} onInputChange={v => setFilter('statusInput', v)} onTagsChange={v => { setFilter('status', v); setActiveStatusTab(v.length === 1 ? v[0] : ''); }} placeholder="e.g. Shortlisted" icon={Filter} suggestions={statusFilterOptions} />

        <div style={{ marginTop: 14, marginBottom: 18 }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 7, fontSize: 11, fontWeight: 800, color: '#64748b', textTransform: 'uppercase', marginBottom: 8 }}>
            <Folder size={13} /> Role
          </label>
          <select
            value={talentPoolRoleFilterId || ''}
            onChange={(e) => {
              setTalentPoolRoleFilter(e.target.value);
              setTpPagination(1, pageSize);
            }}
            style={{ width: '100%', padding: '10px 12px', borderRadius: 12, border: '1px solid rgba(203, 213, 225, 0.9)', background: '#fff', color: '#0f172a', fontSize: 12, fontWeight: 700 }}
          >
            <option value="">All roles</option>
            {scopeRoles.map(r => (
              <option key={r.id} value={r.id}>{r.name} ({Number(r.candidate_count || 0)})</option>
            ))}
          </select>
        </div>

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

        {role === 'recruiter' && recentUploads.length > 0 && (
          <div style={{ marginTop: 16, paddingTop: 14, borderTop: `1px solid ${surfaceBorder}` }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: '#64748b', textTransform: 'uppercase', marginBottom: 8 }}>Recent imports</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 200, overflowY: 'auto' }}>
              {recentUploads.map(u => {
                const ins = Number(u.inserted_count) || 0;
                const upd = Number(u.updated_count) || 0;
                const skp = Number(u.skipped_count) || 0;
                const segs = [];
                if (ins > 0) segs.push(`${ins} new`);
                if (upd > 0) segs.push(`${upd} already in pool`);
                if (skp > 0) segs.push(`${skp} skipped`);
                const statsLine = segs.length > 0 ? segs.join(' · ') : 'No rows applied';
                return (
                  <div key={u.id} style={{ fontSize: 11, color: '#334155', background: '#f8fafc', padding: '8px 10px', borderRadius: 10, border: '1px solid #e2e8f0' }}>
                    <div style={{ fontWeight: 700 }}>{u.filename}</div>
                    <div style={{ color: '#94a3b8' }}>{statsLine}</div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

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
            borderBottom: 'none'
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

            {role === 'admin' && (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center' }}>
                <select
                  value={talentPoolViewScope}
                  onChange={(e) => {
                    const v = e.target.value;
                    if (v === 'recruiter_pools') {
                      const rid = talentPoolRecruiterFilterId || recruiters[0]?.id;
                      setTalentPoolView(v, rid || null);
                    } else {
                      setTalentPoolView(v, null);
                    }
                  }}
                  style={{ padding: '8px 12px', borderRadius: 10, border: '1px solid #e2e8f0', fontSize: 12, fontWeight: 600 }}
                >
                  <option value="master">Master Library</option>
                  <option value="recruiter_pools">Recruiter pool</option>
                  <option value="all_recruiter_pools">All recruiter pools</option>
                </select>
                {talentPoolViewScope === 'recruiter_pools' && (
                  <select
                    value={talentPoolRecruiterFilterId || ''}
                    onChange={(e) => setTalentPoolView('recruiter_pools', Number(e.target.value) || null)}
                    style={{ padding: '8px 12px', borderRadius: 10, border: '1px solid #e2e8f0', fontSize: 12, fontWeight: 600, minWidth: 160 }}
                  >
                    <option value="">Select recruiter…</option>
                    {recruiters.map(r => (
                      <option key={r.id} value={r.id}>{r.full_name || r.email}</option>
                    ))}
                  </select>
                )}
                {talentPoolViewScope === 'master' && selectedIds.size > 0 && (
                  <button
                    type="button"
                    onClick={() => setShowAssignModal(true)}
                    style={{ padding: '8px 14px', borderRadius: 10, border: 'none', background: '#0f172a', color: '#fff', fontSize: 12, fontWeight: 700, cursor: 'pointer' }}
                  >
                    Assign ({selectedIds.size})
                  </button>
                )}
              </div>
            )}

            {role === 'recruiter' && (
              <>
                <input
                  ref={uploadFileRef}
                  type="file"
                  accept=".csv,.xlsx,.xls"
                  style={{ display: 'none' }}
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) runUploadPreview(f);
                    e.target.value = '';
                  }}
                />
                <button
                  type="button"
                  disabled={uploadPreviewBusy}
                  onClick={() => uploadFileRef.current?.click()}
                  style={{
                    display: 'inline-flex', alignItems: 'center', gap: 6,
                    padding: '8px 14px', borderRadius: 12, border: '1px solid #e2e8f0',
                    background: '#fff', fontSize: 12, fontWeight: 700, cursor: uploadPreviewBusy ? 'wait' : 'pointer',
                  }}
                >
                  <FileUp size={14} /> {uploadPreviewBusy ? 'Reading…' : 'Upload CSV / Excel'}
                </button>
              </>
            )}

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
              <button
                onClick={() => setAiColumnModal({ mode: 'create' })}
                style={{
                  display: 'inline-flex', alignItems: 'center', gap: 7,
                  padding: '8px 16px', borderRadius: 12,
                  border: '1px solid rgba(203, 213, 225, 0.9)',
                  background: '#fff',
                  color: '#0f172a', fontSize: 12.5, fontWeight: 700,
                  cursor: 'pointer', fontFamily: 'inherit',
                  boxShadow: '0 10px 24px rgba(15,23,42,0.05)',
                  transition: 'border-color 0.15s, background 0.15s',
                  whiteSpace: 'nowrap',
                }}
                onMouseEnter={e => { e.currentTarget.style.borderColor = 'rgba(148, 163, 184, 0.85)'; e.currentTarget.style.background = '#f8fafc'; }}
                onMouseLeave={e => { e.currentTarget.style.borderColor = 'rgba(203, 213, 225, 0.9)'; e.currentTarget.style.background = '#fff'; }}
                title={selectedIds.size > 0 ? `Run on ${selectedIds.size} selected candidates` : 'Select rows, then run a smart column'}
              >
                <HayasaBrand size="compact" tone="light" showGrowton={false} />
                Smart Column
                {selectedIds.size > 0 && (
                  <span style={{ background: '#eef2f7', color: '#334155', borderRadius: 999, padding: '1px 7px', fontSize: 11 }}>
                    {selectedIds.size}
                  </span>
                )}
              </button>

              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
                <span style={{ fontSize: 13, color: '#0f172a', fontWeight: 700 }}>
                  {!talentPoolScopeReady
                    ? 'Select recruiter'
                    : recruiterCountLoading || (loading && !displayedCandidates.length)
                      ? '...'
                      : `${displayedTotal.toLocaleString()} candidates`}
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
                await fetchCandidates(page, { force: true });
              }}
                style={{ width: 38, height: 38, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#fff', border: '1px solid rgba(203, 213, 225, 0.9)', borderRadius: 12, cursor: 'pointer', color: '#64748b', boxShadow: '0 10px 24px rgba(15,23,42,0.05)' }}>
                <RefreshCw size={14} style={{ animation: loading || isRevalidating ? 'spin 1s linear infinite' : 'none' }} />
              </button>
            </div>
          </div>

          <CompactMetricsStrip
            analytics={analytics}
            tpTotal={metricsTotal}
            tpStatusCounts={metricsStatusCounts}
            countsLoading={recruiterCountLoading}
            expanded={metricsExpanded}
            onToggle={() => setMetricsExpanded(prev => !prev)}
          />

          {metricsExpanded && (
            <div style={{ padding: '14px 20px 18px', borderTop: `1px solid ${surfaceBorder}` }}>
              <StatisticsDashboard
                analytics={analytics}
                tpTotal={metricsTotal}
                tpStatusCounts={metricsStatusCounts}
                countsLoading={recruiterCountLoading}
                role={role}
                onStatClick={(status) => {
                  startTransition(() => {
                    setFilters(prev => ({ ...prev, status: '' }));
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
          {(loadNotice || loadError) && displayedCandidates.length === 0 && (
            <div style={{
              padding: '10px 18px',
              background: loadError ? '#fef2f2' : '#fff7ed',
              color: loadError ? '#991b1b' : '#8b6b44',
              borderBottom: `1px solid ${loadError ? '#fecaca' : '#fed7aa'}`,
              fontSize: 12,
              fontWeight: 700,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 12,
            }}>
              <span>{loadError || loadNotice}</span>
              {loadError && (
                <button
                  type="button"
                  onClick={() => fetchCandidates(page)}
                  style={{ padding: '6px 10px', borderRadius: 9, border: '1px solid currentColor', background: '#fff', color: 'inherit', fontSize: 12, fontWeight: 800, cursor: 'pointer' }}
                >
                  Retry
                </button>
              )}
            </div>
          )}
          {isAiRunFocusActive && (
            <div style={{
              padding: '12px 18px',
              background: 'linear-gradient(135deg, rgba(239,246,255,0.92), rgba(240,253,244,0.88))',
              borderBottom: `1px solid ${surfaceBorder}`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 12,
              flexWrap: 'wrap',
            }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <div style={{ fontSize: 13, fontWeight: 800, color: '#0f172a' }}>
                  Showing {focusCandidateIds.length} profile{focusCandidateIds.length === 1 ? '' : 's'} from Smart Column: {tpAiRunFocus?.columnName || 'Smart Column'}
                </div>
                <div style={{ fontSize: 11, color: '#64748b', fontWeight: 600 }}>
                  These are the selected rows from the latest run.
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <button
                  type="button"
                  onClick={() => setSelectedIds(new Set())}
                  style={{ padding: '7px 12px', borderRadius: 10, border: '1px solid rgba(203, 213, 225, 0.9)', background: '#fff', color: '#334155', fontSize: 12, fontWeight: 700, cursor: 'pointer' }}
                >
                  Clear selection
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setSelectedIds(new Set());
                    exitTpAiRunFocus();
                  }}
                  style={{ padding: '7px 12px', borderRadius: 10, border: 'none', background: '#0f172a', color: '#fff', fontSize: 12, fontWeight: 800, cursor: 'pointer' }}
                >
                  Exit run view
                </button>
              </div>
            </div>
          )}
          {/* Status tabs */}
          {!isAiRunFocusActive && (
          <div style={{ padding: '14px 18px', background: 'rgba(248,250,252,0.78)', borderBottom: `1px solid ${surfaceBorder}`, display: 'flex', gap: 10, overflowX: 'auto', scrollbarWidth: 'none' }}>
            {['', ...RECRUITMENT_STAGES].map(tab => {
              const selectedStatuses = Array.isArray(filters?.status) ? filters.status : [];
              const isActive = tab === '' ? selectedStatuses.length === 0 : selectedStatuses.includes(tab);
              const count = tab === '' ? (displayedTotal || 0) : (displayedStatusCounts?.[tab] || 0);
              const style = tab ? (STATUS_STYLES[tab.toLowerCase()] || {}) : { bg: '#f1f5f9', color: '#475569', dot: '#94a3b8' };
              const isLoadingCount = recruiterCountLoading || (loading && !displayedCandidates.length);

              return (
                <button key={tab || 'all'}
                  onClick={() => {
                    const current = Array.isArray(filters?.status) ? filters.status : [];
                    const nextStatus = !tab
                      ? []
                      : (current.includes(tab) ? current.filter(s => s !== tab) : [...current, tab]);
                    startTransition(() => {
                      setFilters(prev => ({ ...prev, status: nextStatus, statusInput: '' }));
                      setActiveStatusTab(nextStatus.length === 1 ? nextStatus[0] : '');
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
                    <span style={{ color: isActive ? '#fff' : '#0f172a' }}>All ({isLoadingCount ? '...' : displayedTotal})</span>
                  ) : (
                    <>
                      <span style={{ width: 6, height: 6, borderRadius: '50%', background: isActive ? '#fff' : (style.dot || '#94a3b8') }} />
                      {tab}
                      <span style={{
                        marginLeft: 4, padding: '1px 6px', borderRadius: 10, fontSize: 10,
                        background: isActive ? 'rgba(255,255,255,0.14)' : '#e2e8f0',
                        color: isActive ? '#fff' : '#64748b'
                      }}>
                        {isLoadingCount ? '...' : count}
                      </span>
                    </>
                  )}
                </button>
              );
            })}
            {Array.isArray(filters?.status) && filters.status.length >= 2 && (
              <button
                onClick={() => {
                  startTransition(() => {
                    setFilters(prev => ({ ...prev, status: [], statusInput: '' }));
                    setActiveStatusTab('');
                  });
                  setPage(1);
                }}
                style={{
                  padding: '7px 12px', borderRadius: '999px', border: '1px dashed rgba(203, 213, 225, 0.9)',
                  background: 'transparent', cursor: 'pointer', fontSize: 12, fontWeight: 700,
                  color: '#64748b', whiteSpace: 'nowrap', display: 'flex', alignItems: 'center', gap: 6,
                  fontFamily: 'inherit', transition: 'all 0.15s',
                }}
                onMouseEnter={e => { e.currentTarget.style.borderColor = '#94a3b8'; e.currentTarget.style.color = '#0f172a'; }}
                onMouseLeave={e => { e.currentTarget.style.borderColor = 'rgba(203, 213, 225, 0.9)'; e.currentTarget.style.color = '#64748b'; }}
              >
                Clear ({filters.status.length})
              </button>
            )}
          </div>
          )}

          {/* Table */}
          <div ref={tableScrollRef} className="talent-pool-table-scroll" style={{ flex: 1, overflowY: 'auto', overflowX: 'auto', background: 'rgba(255,255,255,0.68)', paddingBottom: 12 }}>
            <table style={{ width: 'max-content', minWidth: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ background: 'rgba(248,250,252,0.98)', borderBottom: `1px solid ${surfaceBorder}`, position: 'sticky', top: 0, zIndex: 10 }}>
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
                      ) : col.isAiCol ? (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
                            <span style={{ color: '#0f172a', fontWeight: 800, letterSpacing: 'normal', textTransform: 'none', fontSize: 12 }}>
                              {col.label}
                            </span>
                            {col.isPrimaryOutput && (
                              <span style={{ display: 'flex', gap: 4 }}>
                                <button
                                  type="button"
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    rerunAiColumn(col.definition);
                                  }}
                                  style={{ width: 24, height: 24, borderRadius: 8, border: '1px solid #bbf7d0', background: '#f0fdf4', color: '#16a34a', cursor: 'pointer', display: 'inline-flex', alignItems: 'center', justifyContent: 'center' }}
                                  title={selectedIds?.size > 0 ? `Run on ${selectedIds.size} selected row(s)` : 'Select rows first, then click to run'}
                                >
                                  <Play size={11} fill="currentColor" />
                                </button>
                                <button
                                  type="button"
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    openAiColumnEditor(col.definition);
                                  }}
                                  style={{ width: 24, height: 24, borderRadius: 8, border: '1px solid #e2e8f0', background: '#fff', color: '#64748b', cursor: 'pointer', display: 'inline-flex', alignItems: 'center', justifyContent: 'center' }}
                                  title="Edit smart column"
                                >
                                  <Edit2 size={11} />
                                </button>
                                <button
                                  type="button"
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    deleteAiColumn(col.definition);
                                  }}
                                  style={{ width: 24, height: 24, borderRadius: 8, border: '1px solid #fee2e2', background: '#fff', color: '#ef4444', cursor: 'pointer', display: 'inline-flex', alignItems: 'center', justifyContent: 'center' }}
                                  title="Delete smart column"
                                >
                                  <Trash2 size={11} />
                                </button>
                              </span>
                            )}
                          </div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <div style={{ flex: 1, height: 6, borderRadius: 999, background: '#e2e8f0', overflow: 'hidden' }}>
                              <div
                                style={{
                                  width: `${col.definition?.latest_run?.total ? Math.min(100, (((Number(col.definition.latest_run.completed || 0) + Number(col.definition.latest_run.failed || 0) + Number(col.definition.latest_run.skipped || 0)) / Number(col.definition.latest_run.total || 1)) * 100)) : 0}%`,
                                  height: '100%',
                                  background: col.definition?.latest_run?.status === 'completed_with_errors' ? '#f59e0b' : '#22c55e',
                                }}
                              />
                            </div>
                            <span style={{ fontSize: 10, color: '#64748b', textTransform: 'none', letterSpacing: 'normal' }}>
                              {summarizeAiRun(col.definition?.latest_run)}
                            </span>
                          </div>
                        </div>
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
                        <p style={{ color: '#64748b', fontWeight: 600, fontSize: 15, margin: 0 }}>
                          {loadError ? 'Candidate data did not load' : talentPoolScopeReady ? 'No candidates found' : 'Select a recruiter to load this pool'}
                        </p>
                        <p style={{ color: '#cbd5e1', fontSize: 13, margin: 0 }}>
                          {loadError || (talentPoolScopeReady ? 'Try adjusting your filters or search query' : 'Recruiter-scoped totals load after a recruiter is selected')}
                        </p>
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

                      if (poolReadOnly) {
                        return <td style={{ padding: '13px 14px', fontSize: 12, color: '#475569', borderRight: '1px solid #eef2f7' }}>{emailVal || '—'}</td>;
                      }
                      if (isEnriching) {
                        return <td style={{ padding: '13px 14px', fontSize: 12, color: '#8b6b44', borderRight: '1px solid #eef2f7', fontStyle: 'italic' }}>Fetching...</td>;
                      }
                      return (
                        <ClickableEditableCell
                          key={`email-${c.id}`}
                          id={c.id}
                          field="email"
                          value={emailVal || ''}
                          onUpdate={(id, data) => {
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

                      if (poolReadOnly) {
                        return <td style={{ padding: '13px 14px', fontSize: 12, color: '#475569', borderRight: '1px solid #eef2f7' }}>{phoneVal || '—'}</td>;
                      }
                      if (isEnriching) {
                        return <td style={{ padding: '13px 14px', fontSize: 12, color: '#8b6b44', borderRight: '1px solid #eef2f7', fontStyle: 'italic' }}>Fetching...</td>;
                      }
                      return (
                        <ClickableEditableCell
                          key={`phone-${c.id}`}
                          id={c.id}
                          field="phone"
                          value={phoneVal || ''}
                          onUpdate={(id, data) => {
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
                        disabled={poolReadOnly}
                        onUpdate={(id, newStatus) => {
                          updateTpCandidate(id, { status: newStatus });
                          invalidateTalentPoolCaches();
                          fetchTalentPoolSummary({ force: true, freshnessMs: 0 });
                          void fetchCandidates(page);
                        }}
                        onShortlisted={handleShortlisted}
                      />
                    </td>
                    <td
                      onClick={(e) => { e.stopPropagation(); if (!poolReadOnly) setSelectedCandidateForChat(c); }}
                      style={{
                        padding: '13px 14px', borderRight: '1px solid #eef2f7',
                        cursor: poolReadOnly ? 'default' : 'pointer',
                        opacity: poolReadOnly ? 0.65 : 1,
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
                      <EditableNotes candidateId={c.id} initialNotes={c.notes} readOnly={poolReadOnly} />
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
                    {/* Dynamic AI Columns */}
                    {dynamicAiCols.map(col => {
                      const cellsMap = col.definition?.cells_by_candidate || {};
                      const cell = cellsMap[c.id] ?? cellsMap[String(c.id)] ?? cellsMap[Number(c.id)];
                      const aiVal = resolveAiCellValue(cell, col.outputKey, col.isPrimaryOutput);
                      const st = cell?.status;
                      const aiCreditsDisplay = cell?.ai_credits_display || '';
                      const showAiCreditsDisplay = aiCreditsDisplay && aiCreditsDisplay !== '$0.000000';
                      const isEmpty =
                        !aiVal && st !== 'running' && st !== 'queued';
                      return (
                        <td
                          key={col.key}
                          style={{
                            padding: '10px 14px',
                            borderRight: '1px solid #eef2f7',
                            maxWidth: 240,
                            verticalAlign: 'top',
                            cursor: 'pointer',
                          }}
                          onClick={() => openAiCellDrawer(col.definition, c, col.outputKey)}
                        >
                          {st === 'running' || st === 'queued' ? (
                            <div style={{
                              fontSize: 11,
                              color: '#2563eb',
                              lineHeight: 1.55,
                              background: 'linear-gradient(135deg, #eff6ff, #eef2ff)',
                              border: '1px solid #bfdbfe',
                              borderRadius: 10,
                              padding: '8px 10px',
                              fontWeight: 700,
                            }}>
                              {st === 'queued' ? 'Queued…' : 'Running…'}
                            </div>
                          ) : st === 'failed' ? (
                            <span style={{ color: '#b91c1c', fontSize: 11, lineHeight: 1.45, display: 'block' }}>
                              {cell?.error_message || 'Run failed'}
                            </span>
                          ) : st === 'skipped' ? (
                            <span style={{ color: '#a16207', fontSize: 11, lineHeight: 1.45, display: 'block' }}>
                              {cell?.error_message || 'Skipped'}
                            </span>
                          ) : isEmpty ? (
                            <span style={{ color: '#94a3b8', fontSize: 11, fontStyle: 'italic', lineHeight: 1.45 }}>
                              {cell ? 'No value for this field (open for details)' : '—'}
                            </span>
                          ) : (
                            <div style={{
                              fontSize: 12, color: '#1e1b4b', lineHeight: 1.55,
                              minHeight: 64,
                              whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                              background: 'linear-gradient(135deg, #faf5ff, #eff6ff)',
                              border: '1px solid #e0e7ff',
                              borderRadius: 8, padding: '8px 10px',
                              display: 'flex',
                              flexDirection: 'column',
                              gap: 6,
                            }}>
                                <div style={{
                                  display: '-webkit-box',
                                  WebkitLineClamp: 4,
                                  WebkitBoxOrient: 'vertical',
                                  overflow: 'hidden',
                                }}>
                                  {renderFriendlyAiValue(aiVal, { inline: true })}
                                </div>
                              <div style={{
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'space-between',
                                gap: 8,
                                flexWrap: 'wrap',
                              }}>
                                <span style={{
                                  fontSize: 10,
                                  color: '#6366f1',
                                  fontWeight: 800,
                                  textTransform: 'uppercase',
                                  letterSpacing: '0.06em',
                                  whiteSpace: 'nowrap',
                                }}>
                                  Click for full answer
                                </span>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 6, minWidth: 0, justifyContent: 'flex-end' }}>
                                  {Array.isArray(cell?.details?.sources) && cell.details.sources.filter(s => s.url).length > 0 && (
                                    <a
                                      href={cell.details.sources.find(s => s.url).url}
                                      target="_blank"
                                      rel="noreferrer"
                                      onClick={e => e.stopPropagation()}
                                      title={cell.details.sources.filter(s => s.url).length > 1 ? `Plus ${cell.details.sources.filter(s => s.url).length - 1} more sources` : ''}
                                      style={{
                                        fontSize: 10,
                                        color: '#2563eb',
                                        textDecoration: 'none',
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: 3,
                                        background: '#dbeafe',
                                        padding: '2px 6px',
                                        borderRadius: 6,
                                        fontWeight: 700,
                                        whiteSpace: 'nowrap'
                                      }}
                                    >
                                      <ExternalLink size={10} /> Source
                                    </a>
                                  )}
                                  {showAiCreditsDisplay && (
                                    <span style={{
                                      fontSize: 10,
                                      color: '#0f766e',
                                      fontWeight: 800,
                                      whiteSpace: 'nowrap',
                                    }}>
                                      {aiCreditsDisplay}
                                    </span>
                                  )}
                                </div>
                              </div>
                            </div>
                          )}
                        </td>
                      );
                    })}
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
                {recruiterCountLoading || (loading && !displayedCandidates.length) ? (
                  <>Loading candidates...</>
                ) : (
                  <>Showing {displayedTotal === 0 ? 0 : Math.min((page - 1) * pageSize + 1, displayedTotal)}–{Math.min(page * pageSize, displayedTotal)} of <strong style={{ color: '#0f172a' }}>{displayedTotal.toLocaleString()}</strong> candidates</>
                )}
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

      {(selectedIds.size > 0 || allFilteredSelected) && (
        <div style={{
          position: 'fixed', bottom: 30, left: '50%', transform: 'translateX(-50%)',
          background: '#0f172a', color: '#fff', padding: '12px 24px', borderRadius: '16px',
          display: 'flex', alignItems: 'center', gap: 20, boxShadow: '0 20px 25px -5px rgba(0,0,0,0.3)',
          zIndex: 1000, border: '1px solid rgba(255,255,255,0.1)', animation: 'slideUp 0.3s ease-out'
        }}>
          <span style={{ fontSize: 14, fontWeight: 600 }}>
            {allFilteredSelected ? `${displayedTotal} filtered candidates selected` : `${selectedIds.size} candidates selected`}
          </span>
          <div style={{ width: 1, height: 20, background: 'rgba(255,255,255,0.2)' }} />
          <button
            onClick={() => setAllFilteredSelected(prev => !prev)}
            style={{
              padding: '8px 16px', background: allFilteredSelected ? '#312e81' : '#fff', color: allFilteredSelected ? '#fff' : '#0f172a', border: '1px solid rgba(255,255,255,0.18)',
              borderRadius: '10px', fontSize: 13, fontWeight: 700, cursor: 'pointer',
              display: 'flex', alignItems: 'center', gap: 8, transition: 'all 0.2s'
            }}
          >
            <Check size={14} /> {allFilteredSelected ? 'Using All Filtered' : `Use All Filtered (${displayedTotal})`}
          </button>
          {selectedIds.size > 0 && (
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
          )}
          {selectedIds.size > 0 && (
            <button
              onClick={() => setShowAssignModal(true)}
              style={{
                padding: '8px 16px', background: '#fff', color: '#0f172a', border: '1px solid rgba(255,255,255,0.18)',
                borderRadius: '10px', fontSize: 13, fontWeight: 700, cursor: 'pointer',
                display: 'flex', alignItems: 'center', gap: 8, transition: 'all 0.2s'
              }}
              onMouseEnter={e => e.currentTarget.style.background = '#f8fafc'}
              onMouseLeave={e => e.currentTarget.style.background = '#fff'}
            >
              <Briefcase size={14} /> Add to Role
            </button>
          )}
          <button
            onClick={() => { setSelectedIds(new Set()); setAllFilteredSelected(false); }}
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

      {uploadOpen && (
        <CsvMappingModal
          title="Map Talent Pool columns"
          subtitle="Review the smart suggestions, adjust anything that looks off, then import into your pool."
          headers={uploadHeaders}
          mapping={uploadMapping}
          details={uploadMappingDetails}
          requiredTargets={uploadRequiredTargets}
          targetOptions={uploadTargetOptions}
          rowCount={uploadRowCount}
          progress={uploadProgress}
          enrichmentMode={uploadEnrichmentMode}
          busy={uploadCommitBusy}
          onEnrichmentModeChange={setUploadEnrichmentMode}
          onChange={(header, value) => setUploadMapping(prev => ({ ...prev, [header]: value }))}
          onCancel={() => { setUploadOpen(false); setUploadFile(null); setUploadMappingDetails({}); setUploadProgress(null); setUploadRowCount(0); setUploadEnrichmentMode('none'); }}
          onImport={commitUpload}
        />
      )}

      {showAssignModal && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(15,23,42,0.65)', zIndex: 10002, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20 }}>
          <div style={{ background: '#fff', borderRadius: 20, maxWidth: 460, width: '100%', padding: 24, boxShadow: '0 24px 60px rgba(15,23,42,0.28)' }}>
            <h3 style={{ margin: '0 0 12px', fontSize: 18, fontWeight: 800 }}>Assign to recruiter</h3>
            <p style={{ color: '#64748b', fontSize: 13, marginBottom: 18 }}>{selectedIds.size} master profile(s) will be copied to the recruiter&apos;s pool and can also be placed directly into a role.</p>
            <label style={{ display: 'block', color: '#334155', fontSize: 12, fontWeight: 750, marginBottom: 14 }}>Recruiter
            <select
              value={assignTargetRecruiterId}
              onChange={(e) => setAssignTargetRecruiterId(e.target.value)}
              style={{ width: '100%', padding: '10px 12px', borderRadius: 10, border: '1px solid #e2e8f0', marginTop: 6, fontSize: 13 }}
            >
              <option value="">Choose recruiter…</option>
              {recruiters.map(r => (
                <option key={r.id} value={r.id}>{r.full_name || r.email}</option>
              ))}
            </select>
            </label>

            <label style={{ display: 'block', color: '#334155', fontSize: 12, fontWeight: 750, marginBottom: 20 }}>Role
              <select
                value={assignTargetRoleId}
                onChange={(e) => {
                  const nextRoleId = e.target.value;
                  setAssignTargetRoleId(nextRoleId);
                  const selectedRole = assignRecruiterRoles.find(r => String(r.id) === String(nextRoleId));
                  if (selectedRole && recruiters.some(r => String(r.id) === String(selectedRole.owner_user_id))) {
                    setAssignTargetRecruiterId(String(selectedRole.owner_user_id));
                  }
                }}
                disabled={assignRolesLoading}
                style={{ width: '100%', padding: '10px 12px', borderRadius: 10, border: '1px solid #e2e8f0', marginTop: 6, fontSize: 13, background: '#fff' }}
              >
                <option value="">{assignRolesLoading ? 'Loading roles…' : 'Recruiter pool only (no role)'}</option>
                {assignRecruiterRoles.map(recruiterRole => (
                  <option key={recruiterRole.id} value={recruiterRole.id}>{recruiterRole.name}{recruiterRole.owner_name ? ` · ${recruiterRole.owner_name}` : ''} · {recruiterRole.candidate_count || 0} candidates</option>
                ))}
              </select>
              {!assignRolesLoading && assignRecruiterRoles.length === 0 && (
                <span style={{ display: 'block', color: '#94a3b8', fontSize: 11, marginTop: 6 }}>No roles yet; the profiles will remain in the recruiter&apos;s pool.</span>
              )}
            </label>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
              <button type="button" onClick={() => { setShowAssignModal(false); setAssignTargetRoleId(''); setAssignRecruiterRoles([]); }} style={{ padding: '10px 16px', borderRadius: 10, border: '1px solid #e2e8f0', background: '#fff', fontWeight: 700 }}>Cancel</button>
              <button type="button" disabled={assignBusy || !assignTargetRecruiterId || assignRolesLoading} onClick={runBulkAssign} style={{ padding: '10px 16px', borderRadius: 10, border: 'none', background: '#0f172a', color: '#fff', fontWeight: 700, opacity: assignBusy || !assignTargetRecruiterId || assignRolesLoading ? 0.6 : 1 }}>{assignBusy ? 'Assigning…' : assignTargetRoleId ? 'Assign to role' : 'Assign to pool'}</button>
            </div>
          </div>
        </div>
      )}

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

      {/* AI Column Modal */}
      {aiColumnModal && (
        <AiColumnConfigModal
          selectedIds={selectedIds}
          initialDefinition={aiColumnModal?.mode === 'edit' ? aiColumnModal.definition : null}
          viewScope={talentPoolViewScope}
          recruiterFilterId={talentPoolRecruiterFilterId}
          roleId={talentPoolRoleFilterId}
          onClose={() => setAiColumnModal(null)}
          onColumnDefinitionCreated={(runInfo) => {
            upsertOptimisticAiColumn(runInfo);
          }}
          onColumnRunFailed={(runInfo) => {
            revertOptimisticAiColumn(runInfo);
          }}
          onColumnsCreated={(runInfo) => {
            attachAiColumnRun(runInfo);
            startTpAiRunFocus({
              runId: runInfo?.runId,
              columnDefinitionId: runInfo?.columnDefinitionId,
              columnName: runInfo?.columnName,
              candidateIds: runInfo?.candidateIds,
            });
            setAiColumnModal(null);
            void fetchAiColumns();
          }}
        />
      )}

      <AiColumnCellDrawer
        open={aiCellDrawerOpen}
        loading={aiCellDrawerLoading}
        detail={aiCellDrawerDetail}
        title={aiCellDrawerTitle}
        onClose={() => {
          setAiCellDrawerOpen(false);
          setAiCellDrawerDetail(null);
          setAiCellDrawerTitle('');
        }}
      />
    </div>
  );
}
