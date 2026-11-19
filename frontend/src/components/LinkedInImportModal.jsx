import React, { useMemo, useState } from 'react';
import { Linkedin, X } from 'lucide-react';

const LI_PROFILE = /linkedin\.com\/in\//i;

/**
 * Paste LinkedIn profile URLs (one per line); the backend scrapes them through
 * Apify, enriches them onto the candidate schema and loads them into the pool.
 * `onImport(urls, duplicatePolicy)` must return once the import has finished.
 */
export default function LinkedInImportModal({ open, busy, progress, roleName = '', maxProfiles = 50, onCancel, onImport }) {
  const [text, setText] = useState('');
  const [duplicatePolicy, setDuplicatePolicy] = useState('skip');

  const { urls, invalid } = useMemo(() => {
    const seen = new Set();
    const good = [];
    const bad = [];
    text.split(/\r?\n|,|\s+/).map(s => s.trim()).filter(Boolean).forEach(line => {
      if (!LI_PROFILE.test(line)) { bad.push(line); return; }
      const key = line.toLowerCase().replace(/^https?:\/\/(www\.)?/, '').replace(/[/?#].*$/, '').replace(/\/+$/, '');
      if (seen.has(key)) return;
      seen.add(key);
      good.push(line);
    });
    return { urls: good, invalid: bad };
  }, [text]);

  if (!open) return null;

  const tooMany = urls.length > maxProfiles;
  const canImport = !busy && urls.length > 0 && invalid.length === 0 && !tooMany;
  const processed = Number(progress?.processed_count) || 0;
  const total = Number(progress?.row_count) || urls.length;
  const stage = String(progress?.status || '');

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(15,23,42,0.65)', zIndex: 10002, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20 }}>
      <div style={{ background: '#fff', borderRadius: 20, maxWidth: 620, width: '100%', padding: 24, boxShadow: '0 24px 60px rgba(15,23,42,0.28)' }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12 }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 17, fontWeight: 800, color: '#0f172a' }}>
              <Linkedin size={18} color="#0a66c2" /> Import from LinkedIn{roleName ? ` into ${roleName}` : ''}
            </div>
            <div style={{ fontSize: 13, color: '#64748b', marginTop: 4 }}>
              Paste profile URLs, one per line. Each profile is scraped, enriched (tenure, gaps, scores, company research) and added to {roleName ? 'this role and the owner\'s pool' : 'your pool'}.
            </div>
          </div>
          <button type="button" onClick={onCancel} disabled={busy} aria-label="Close" style={{ border: 'none', background: 'transparent', cursor: busy ? 'not-allowed' : 'pointer', color: '#64748b' }}>
            <X size={18} />
          </button>
        </div>

        <textarea
          value={text}
          onChange={e => setText(e.target.value)}
          disabled={busy}
          placeholder={'https://www.linkedin.com/in/jane-doe\nhttps://www.linkedin.com/in/john-smith'}
          rows={7}
          style={{ width: '100%', marginTop: 16, padding: 12, borderRadius: 12, border: '1px solid #e2e8f0', fontSize: 13, fontFamily: 'inherit', resize: 'vertical', boxSizing: 'border-box' }}
        />

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 8, fontSize: 12, color: '#64748b', gap: 12, flexWrap: 'wrap' }}>
          <span>
            {urls.length} profile{urls.length === 1 ? '' : 's'} ready
            {tooMany ? <span style={{ color: '#dc2626' }}> · max {maxProfiles} per import</span> : null}
          </span>
          <label style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            If already in my pool:
            <select value={duplicatePolicy} onChange={e => setDuplicatePolicy(e.target.value)} disabled={busy} style={{ padding: '4px 8px', borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 12 }}>
              <option value="skip">skip</option>
              <option value="replace">replace with fresh profile</option>
            </select>
          </label>
        </div>

        {invalid.length > 0 && (
          <div style={{ marginTop: 10, padding: '8px 12px', borderRadius: 10, background: '#fef2f2', color: '#b91c1c', fontSize: 12 }}>
            Not LinkedIn profile URLs: {invalid.slice(0, 3).join(', ')}{invalid.length > 3 ? ` +${invalid.length - 3} more` : ''}
          </div>
        )}

        {busy && (
          <div style={{ marginTop: 14 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#475569', marginBottom: 6 }}>
              <span>{stage === 'scraping' ? 'Scraping LinkedIn via Apify…' : stage === 'enriching' ? 'Enriching and saving…' : 'Starting…'}</span>
              <span>{processed} / {total}</span>
            </div>
            <div style={{ height: 8, borderRadius: 999, background: '#e2e8f0', overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${total ? Math.max(4, Math.round((processed / total) * 100)) : 4}%`, background: '#f97316', transition: 'width .3s' }} />
            </div>
          </div>
        )}

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, marginTop: 18 }}>
          <button type="button" onClick={onCancel} disabled={busy} style={{ padding: '10px 16px', borderRadius: 10, border: '1px solid #e2e8f0', background: '#fff', fontWeight: 700, cursor: busy ? 'not-allowed' : 'pointer' }}>
            Cancel
          </button>
          <button
            type="button"
            disabled={!canImport}
            onClick={() => onImport(urls, duplicatePolicy)}
            style={{ padding: '10px 16px', borderRadius: 10, border: 'none', background: canImport ? '#f97316' : '#cbd5e1', color: '#fff', fontWeight: 800, cursor: canImport ? 'pointer' : 'not-allowed' }}
          >
            {busy ? 'Importing…' : `Import ${urls.length || ''}`.trim()}
          </button>
        </div>
      </div>
    </div>
  );
}
