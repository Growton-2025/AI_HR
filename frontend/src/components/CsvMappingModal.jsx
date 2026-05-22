import React, { useMemo } from 'react';
import { Check, AlertTriangle, Loader2, Wand2 } from 'lucide-react';

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
  busy = false,
  onChange,
  onCancel,
  onImport,
}) {
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
        <div style={{ padding: '16px 24px 4px', overflow: 'auto', maxHeight: '58vh' }}>
          <div style={{ minWidth: 820, border: '1px solid #e2e8f0', borderRadius: 12, overflow: 'hidden' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1.6fr 1fr 1.1fr', background: '#f8fafc', borderBottom: '1px solid #e2e8f0', color: '#475569', fontSize: 11, fontWeight: 800, textTransform: 'uppercase' }}>
              <div style={{ padding: '10px 12px' }}>CSV column</div>
              <div style={{ padding: '10px 12px' }}>Sample values</div>
              <div style={{ padding: '10px 12px' }}>Suggested field</div>
              <div style={{ padding: '10px 12px' }}>Confidence</div>
            </div>
            {headers.map(header => {
              const detail = details?.[header] || {};
              const source = detail.source || 'manual';
              const confidence = Number(detail.confidence || 0);
              const samples = Array.isArray(detail.sample_values) ? detail.sample_values.filter(Boolean) : [];
              const isRequiredTarget = requiredTargets.includes(mapping?.[header]);
              return (
                <div key={header} style={{ display: 'grid', gridTemplateColumns: '1.2fr 1.6fr 1fr 1.1fr', borderBottom: '1px solid #eef2f7', alignItems: 'stretch' }}>
                  <div style={{ padding: '12px', fontSize: 13, fontWeight: 800, color: '#0f172a', wordBreak: 'break-word' }}>
                    {header}
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
                        <option key={option} value={option}>{LABELS[option] || option}</option>
                      ))}
                    </select>
                  </div>
                  <div style={{ padding: '10px 12px', fontSize: 12, color: '#475569' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 4 }}>
                      {mapping?.[header] && mapping[header] !== 'ignore'
                        ? <Check size={14} color="#15803d" />
                        : <AlertTriangle size={14} color="#94a3b8" />}
                      <span style={{ fontWeight: 800, color: source === 'model' ? '#6d28d9' : source === 'alias' ? '#0369a1' : '#64748b' }}>
                        {source === 'model' ? 'Model' : source === 'alias' ? 'Alias' : 'Manual'}
                        {confidence > 0 ? ` ${Math.round(confidence * 100)}%` : ''}
                      </span>
                    </div>
                    <div style={{ color: '#94a3b8', lineHeight: 1.35 }}>{detail.reason || 'Review manually'}</div>
                  </div>
                </div>
              );
            })}
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
