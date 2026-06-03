import React, { useMemo, useState } from 'react';
import { Check, AlertTriangle, Loader2, Search, Wand2 } from 'lucide-react';

const DEFAULT_OPTIONS = [
  'ignore',
  'first_name',
  'last_name',
  'linkedin',
  'city',
  'title',
  'company_name',
  'email',
  'phone',
  'location',
  'notes',
  'headline',
  'about',
  'custom',
];

const LABELS = {
  ignore: 'Ignore',
  first_name: 'First name',
  last_name: 'Last name',
  linkedin: 'LinkedIn URL',
  city: 'City',
  title: 'Title',
  company_name: 'Company',
  email: 'Email',
  phone: 'Phone',
  location: 'Location',
  notes: 'Notes',
  headline: 'Headline',
  about: 'About',
  custom: 'Custom data',
};

const GROUP_ORDER = [
  'Required Fields',
  'Recommended Fields',
  'Work History',
  'Education',
  'Contact/Compensation',
  'Other Fields',
  'Needs Review',
];

const STATUS_LABELS = {
  history: 'Preserved for enrichment',
  custom: 'Preserved',
  alias: 'Mapped',
  model: 'Suggested',
  manual: 'Needs review',
};

export default function CsvMappingModal({
  title = 'Map columns',
  subtitle = 'Review the suggested mapping before import.',
  headers = [],
  mapping = {},
  details = {},
  requiredTargets = ['first_name', 'last_name', 'linkedin', 'city', 'title'],
  targetOptions = DEFAULT_OPTIONS,
  rowCount = 0,
  progress = null,
  enrichmentMode = 'none',
  busy = false,
  onEnrichmentModeChange,
  onChange,
  onCancel,
  onImport,
}) {
  const [columnSearch, setColumnSearch] = useState('');
  const [groupFilter, setGroupFilter] = useState('all');

  const truncateSample = (value) => {
    const text = String(value || '').replace(/\s+/g, ' ').trim();
    return text.length > 120 ? `${text.slice(0, 117)}...` : text;
  };

  const missingRequired = useMemo(() => {
    const used = new Set(Object.values(mapping || {}).filter(v => v && v !== 'ignore'));
    return (requiredTargets || []).filter(target => !used.has(target));
  }, [mapping, requiredTargets]);

  const options = targetOptions?.length ? targetOptions : DEFAULT_OPTIONS;
  const progressStatus = String(progress?.status || '').toLowerCase();
  const hasProgress = Boolean(progress?.upload_id || progressStatus);
  const isFailed = progressStatus === 'failed';
  const isComplete = ['completed', 'completed_with_errors'].includes(progressStatus);
  const totalRows = Number(progress?.row_count || rowCount || 0);
  const processedRows = Math.min(Number(progress?.processed_count || 0), totalRows || Number(progress?.processed_count || 0));
  const progressPercent = totalRows > 0 ? Math.max(3, Math.min(100, Math.round((processedRows / totalRows) * 100))) : 6;
  const summaryBits = [
    Number(progress?.inserted) > 0 ? `${Number(progress.inserted)} added` : null,
    Number(progress?.updated) > 0 ? `${Number(progress.updated)} updated` : null,
    Number(progress?.skipped) > 0 ? `${Number(progress.skipped)} skipped` : null,
    Number(progress?.role_assigned_count) > 0 ? `${Number(progress.role_assigned_count)} role assignments` : null,
  ].filter(Boolean);

  const groupedColumns = useMemo(() => {
    const q = columnSearch.trim().toLowerCase();
    const groups = new Map();
    for (const header of headers || []) {
      const detail = details?.[header] || {};
      const category = detail.category || 'Other Fields';
      if (groupFilter !== 'all' && category !== groupFilter) continue;
      const sampleText = Array.isArray(detail.sample_values) ? detail.sample_values.join(' ') : '';
      const haystack = [
        header,
        detail.friendly_label,
        detail.reason,
        detail.preserve_reason,
        mapping?.[header],
        sampleText,
      ].join(' ').toLowerCase();
      if (q && !haystack.includes(q)) continue;
      if (!groups.has(category)) groups.set(category, []);
      groups.get(category).push(header);
    }
    const ordered = [];
    for (const group of GROUP_ORDER) {
      if (groups.has(group)) ordered.push([group, groups.get(group)]);
    }
    for (const [group, items] of groups.entries()) {
      if (!GROUP_ORDER.includes(group)) ordered.push([group, items]);
    }
    return ordered;
  }, [headers, details, mapping, columnSearch, groupFilter]);

  const groupCounts = useMemo(() => {
    const counts = {};
    for (const header of headers || []) {
      const group = details?.[header]?.category || 'Other Fields';
      counts[group] = (counts[group] || 0) + 1;
    }
    return counts;
  }, [headers, details]);

  const mappedCount = useMemo(
    () => Object.values(mapping || {}).filter(value => value && value !== 'ignore').length,
    [mapping],
  );

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(15,23,42,0.65)', zIndex: 10002, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20 }}>
      <div style={{ background: '#fff', borderRadius: 18, maxWidth: 980, width: '100%', maxHeight: '90vh', overflow: 'hidden', boxShadow: '0 25px 50px rgba(0,0,0,0.22)', border: '1px solid #e2e8f0' }}>
        <div style={{ padding: '20px 24px', borderBottom: '1px solid #e2e8f0', display: 'flex', gap: 16, alignItems: 'flex-start', justifyContent: 'space-between' }}>
          <div>
            <h3 style={{ margin: '0 0 6px', fontSize: 20, fontWeight: 800, color: '#0f172a' }}>{title}</h3>
            <p style={{ margin: 0, color: '#64748b', fontSize: 13, lineHeight: 1.5 }}>{subtitle}</p>
          </div>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: 7, padding: '8px 10px', borderRadius: 10, background: '#fff7ed', color: '#9a3412', border: '1px solid #fed7aa', fontSize: 12, fontWeight: 800, whiteSpace: 'nowrap' }}>
            <Wand2 size={14} /> Smart match
          </div>
        </div>

        {!hasProgress && missingRequired.length > 0 && (
          <div style={{ margin: '14px 24px 0', padding: '10px 12px', borderRadius: 10, background: '#fffbeb', border: '1px solid #fde68a', color: '#92400e', fontSize: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
            <AlertTriangle size={14} />
            Map required fields: {missingRequired.map(t => LABELS[t] || t).join(', ')}
          </div>
        )}

        {!hasProgress && (
          <div style={{ margin: '14px 24px 0', padding: 12, borderRadius: 12, background: '#f8fafc', border: '1px solid #e2e8f0', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
            <div>
              <div style={{ fontSize: 13, fontWeight: 800, color: '#0f172a' }}>Import mode</div>
              <div style={{ marginTop: 3, fontSize: 12, color: '#64748b' }}>
                Verified enrichment parses work history, tenure, education, company segment, industry, and function after import.
              </div>
            </div>
            <select
              value={enrichmentMode || 'none'}
              disabled={busy}
              onChange={(e) => onEnrichmentModeChange?.(e.target.value)}
              style={{ minWidth: 220, padding: '9px 10px', borderRadius: 9, border: '1px solid #cbd5e1', fontSize: 12, background: '#fff', color: '#0f172a', fontWeight: 800 }}
            >
              <option value="none">Import only</option>
              <option value="verified_profile">Import + verified enrichment</option>
            </select>
          </div>
        )}

        {hasProgress ? (
          <div style={{ padding: '26px 24px 28px', minHeight: 320, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ width: '100%', maxWidth: 520, border: '1px solid #e2e8f0', borderRadius: 12, padding: '24px 24px 22px', background: '#fbfdff' }}>
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: 14 }}>
                <div style={{ width: 38, height: 38, borderRadius: 10, background: isFailed ? '#fef2f2' : isComplete ? '#ecfdf5' : '#f8fafc', border: `1px solid ${isFailed ? '#fecaca' : isComplete ? '#bbf7d0' : '#e2e8f0'}`, display: 'grid', placeItems: 'center', flexShrink: 0 }}>
                  {isFailed
                    ? <AlertTriangle size={19} color="#dc2626" />
                    : isComplete
                      ? <Check size={19} color="#15803d" />
                      : <Loader2 size={19} color="#334155" style={{ animation: 'csvUploadSpin 1s linear infinite' }} />}
                </div>
                <div style={{ minWidth: 0, flex: 1 }}>
                  <div style={{ fontSize: 16, lineHeight: 1.35, fontWeight: 800, color: '#0f172a' }}>
                    {isFailed
                      ? 'Import failed'
                      : isComplete
                        ? 'Import complete'
                        : progressStatus === 'refreshing'
                          ? 'Finalizing imported profiles'
                          : totalRows > 0
                            ? `Processing profile ${processedRows} of ${totalRows}`
                            : 'Processing uploaded profiles'}
                  </div>
                  <div style={{ marginTop: 5, fontSize: 13, lineHeight: 1.45, color: '#64748b' }}>
                    {isFailed
                      ? (progress?.error_message || 'The upload stopped before it could finish.')
                      : isComplete
                        ? (summaryBits.length ? summaryBits.join(' | ') : `${processedRows || totalRows} spreadsheet rows processed`)
                        : progressStatus === 'enriching'
                          ? 'Work history, tenure, education, and company context are being verified.'
                        : 'Rows are checked, matched, and written as they finish.'}
                  </div>
                </div>
              </div>
              {!isFailed && (
                <>
                  <div style={{ height: 8, borderRadius: 999, overflow: 'hidden', background: '#e2e8f0', marginTop: 22 }}>
                    <div style={{ height: '100%', width: `${progressPercent}%`, borderRadius: 999, background: isComplete ? '#16a34a' : '#334155', transition: 'width 320ms ease' }} />
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, marginTop: 9, color: '#64748b', fontSize: 12, fontWeight: 700 }}>
                    <span>{progressStatus === 'refreshing' ? 'Updating talent views' : 'Spreadsheet rows processed'}</span>
                    <span>{processedRows}{totalRows ? ` / ${totalRows}` : ''}</span>
                  </div>
                </>
              )}
            </div>
          </div>
        ) : (
        <div style={{ padding: '14px 24px 4px', overflow: 'auto', maxHeight: '58vh' }}>
          <div style={{ position: 'sticky', top: -14, zIndex: 3, padding: '10px 0 12px', background: '#fff' }}>
            <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: 10 }}>
              <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 7 }}>
                {(requiredTargets || []).map(target => {
                  const ok = !missingRequired.includes(target);
                  return (
                    <span key={target} style={{ display: 'inline-flex', alignItems: 'center', gap: 5, padding: '6px 8px', borderRadius: 8, border: `1px solid ${ok ? '#bbf7d0' : '#fed7aa'}`, background: ok ? '#f0fdf4' : '#fff7ed', color: ok ? '#166534' : '#9a3412', fontSize: 11, fontWeight: 800 }}>
                      {ok ? <Check size={12} /> : <AlertTriangle size={12} />}
                      {LABELS[target] || target}
                    </span>
                  );
                })}
              </div>
              <div style={{ color: '#64748b', fontSize: 12, fontWeight: 800 }}>
                {mappedCount} of {headers.length} columns mapped or preserved
              </div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(220px, 1fr) 220px', gap: 10, marginTop: 11 }}>
              <label style={{ position: 'relative', display: 'block' }}>
                <Search size={15} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: '#64748b' }} />
                <input
                  value={columnSearch}
                  disabled={busy}
                  onChange={(e) => setColumnSearch(e.target.value)}
                  placeholder="Search any column, sample, or reason"
                  style={{ width: '100%', padding: '9px 10px 9px 32px', borderRadius: 9, border: '1px solid #cbd5e1', fontSize: 12, color: '#0f172a', outline: 'none' }}
                />
              </label>
              <select
                value={groupFilter}
                disabled={busy}
                onChange={(e) => setGroupFilter(e.target.value)}
                style={{ width: '100%', padding: '9px 10px', borderRadius: 9, border: '1px solid #cbd5e1', fontSize: 12, background: '#fff', color: '#0f172a', fontWeight: 800 }}
              >
                <option value="all">All groups ({headers.length})</option>
                {GROUP_ORDER.filter(group => groupCounts[group]).map(group => (
                  <option key={group} value={group}>{group} ({groupCounts[group]})</option>
                ))}
              </select>
            </div>
          </div>

          <div style={{ minWidth: 940, border: '1px solid #e2e8f0', borderRadius: 12, overflow: 'hidden' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1.25fr 1.45fr 1fr 1.2fr', background: '#f8fafc', borderBottom: '1px solid #e2e8f0', color: '#475569', fontSize: 11, fontWeight: 800, textTransform: 'uppercase' }}>
              <div style={{ padding: '10px 12px' }}>Column</div>
              <div style={{ padding: '10px 12px' }}>Sample values</div>
              <div style={{ padding: '10px 12px' }}>Field / action</div>
              <div style={{ padding: '10px 12px' }}>Status</div>
            </div>
            {groupedColumns.length === 0 && (
              <div style={{ padding: 22, color: '#64748b', fontSize: 13, fontWeight: 700 }}>
                No columns match the current filter.
              </div>
            )}
            {groupedColumns.map(([group, groupHeaders]) => (
              <React.Fragment key={group}>
                <div style={{ position: 'sticky', top: 91, zIndex: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, padding: '9px 12px', background: '#eef2f7', borderBottom: '1px solid #dbe3ee', color: '#334155', fontSize: 12, fontWeight: 900 }}>
                  <span>{group}</span>
                  <span>{groupHeaders.length} column{groupHeaders.length === 1 ? '' : 's'}</span>
                </div>
                {groupHeaders.map(header => {
                  const detail = details?.[header] || {};
                  const source = detail.source || 'manual';
                  const confidence = Number(detail.confidence || 0);
                  const samples = Array.isArray(detail.sample_values) ? detail.sample_values.filter(Boolean).map(truncateSample) : [];
                  const isRequiredTarget = requiredTargets.includes(mapping?.[header]);
                  const mapped = Boolean(mapping?.[header] && mapping[header] !== 'ignore');
                  const preserveReason = detail.preserve_reason || (mapping?.[header] === 'custom' ? 'Preserved as imported extra data' : '');
                  const statusText = preserveReason || STATUS_LABELS[source] || (mapped ? 'Mapped' : 'Needs review');
                  const statusColor = detail.category === 'Needs Review' || !mapped ? '#9a3412' : source === 'model' ? '#6d28d9' : source === 'history' ? '#166534' : '#0369a1';
                  return (
                    <div key={header} style={{ display: 'grid', gridTemplateColumns: '1.25fr 1.45fr 1fr 1.2fr', borderBottom: '1px solid #eef2f7', alignItems: 'stretch' }}>
                      <div style={{ padding: '12px', fontSize: 13, color: '#0f172a', wordBreak: 'break-word' }}>
                        <div style={{ fontWeight: 900 }}>{detail.friendly_label || header}</div>
                        {detail.friendly_label && detail.friendly_label !== header && (
                          <div style={{ marginTop: 4, color: '#64748b', fontSize: 11, fontWeight: 700 }}>{header}</div>
                        )}
                      </div>
                      <div style={{ padding: '12px', fontSize: 12, color: '#475569', lineHeight: 1.45 }}>
                        {samples.length ? samples.slice(0, 3).join(' | ') : <span style={{ color: '#94a3b8' }}>No sample value</span>}
                      </div>
                      <div style={{ padding: '9px 12px' }}>
                        <select
                          value={mapping?.[header] || 'ignore'}
                          disabled={busy}
                          onChange={(e) => onChange?.(header, e.target.value)}
                          style={{ width: '100%', padding: '8px 10px', borderRadius: 9, border: `1px solid ${isRequiredTarget ? '#fdba74' : '#cbd5e1'}`, fontSize: 12, background: '#fff', color: '#0f172a', fontWeight: 700 }}
                        >
                          {options.map(option => (
                            <option key={option} value={option}>{option === 'custom' ? 'Preserve as extra data' : LABELS[option] || option}</option>
                          ))}
                        </select>
                      </div>
                      <div style={{ padding: '10px 12px', fontSize: 12, color: '#475569' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 4 }}>
                          {mapped
                            ? <Check size={14} color="#15803d" />
                            : <AlertTriangle size={14} color="#94a3b8" />}
                          <span style={{ fontWeight: 900, color: statusColor }}>
                            {statusText}
                            {confidence > 0 ? ` ${Math.round(confidence * 100)}%` : ''}
                          </span>
                        </div>
                        <div style={{ color: '#94a3b8', lineHeight: 1.35 }}>{detail.reason || 'Review manually'}</div>
                      </div>
                    </div>
                  );
                })}
              </React.Fragment>
            ))}
          </div>
        </div>
        )}

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, padding: '16px 24px 22px', borderTop: '1px solid #e2e8f0', background: '#fff' }}>
          <div style={{ fontSize: 12, color: '#64748b' }}>
            {hasProgress
              ? `${totalRows || processedRows} spreadsheet row${(totalRows || processedRows) === 1 ? '' : 's'} in this import`
              : `${headers.length} column${headers.length === 1 ? '' : 's'} found`}
          </div>
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
            <button type="button" onClick={onCancel} disabled={busy && !isComplete && !isFailed} style={{ padding: '10px 16px', borderRadius: 10, border: '1px solid #e2e8f0', background: '#fff', fontWeight: 800, cursor: busy && !isComplete && !isFailed ? 'not-allowed' : 'pointer', color: '#334155' }}>{hasProgress ? 'Close' : 'Cancel'}</button>
            {!hasProgress && (
              <button type="button" disabled={busy || missingRequired.length > 0} onClick={onImport} style={{ padding: '10px 16px', borderRadius: 10, border: 'none', background: busy || missingRequired.length > 0 ? '#cbd5e1' : '#f97316', color: '#fff', fontWeight: 800, cursor: busy || missingRequired.length > 0 ? 'not-allowed' : 'pointer' }}>
                {busy ? 'Importing...' : 'Import'}
              </button>
            )}
          </div>
        </div>
      </div>
      <style>{`@keyframes csvUploadSpin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
