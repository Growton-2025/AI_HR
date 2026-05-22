import React, { useMemo } from 'react';
import { Check, AlertTriangle, Wand2 } from 'lucide-react';

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

        {missingRequired.length > 0 && (
          <div style={{ margin: '14px 24px 0', padding: '10px 12px', borderRadius: 10, background: '#fffbeb', border: '1px solid #fde68a', color: '#92400e', fontSize: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
            <AlertTriangle size={14} />
            Map required fields: {missingRequired.map(t => LABELS[t] || t).join(', ')}
          </div>
        )}

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

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, padding: '16px 24px 22px', borderTop: '1px solid #e2e8f0', background: '#fff' }}>
          <div style={{ fontSize: 12, color: '#64748b' }}>
            {headers.length} column{headers.length === 1 ? '' : 's'} found
          </div>
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
            <button type="button" onClick={onCancel} disabled={busy} style={{ padding: '10px 16px', borderRadius: 10, border: '1px solid #e2e8f0', background: '#fff', fontWeight: 800, cursor: busy ? 'not-allowed' : 'pointer', color: '#334155' }}>Cancel</button>
            <button type="button" disabled={busy || missingRequired.length > 0} onClick={onImport} style={{ padding: '10px 16px', borderRadius: 10, border: 'none', background: busy || missingRequired.length > 0 ? '#cbd5e1' : '#f97316', color: '#fff', fontWeight: 800, cursor: busy || missingRequired.length > 0 ? 'not-allowed' : 'pointer' }}>
              {busy ? 'Importing...' : 'Import'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
