import React from 'react';
import { FileUp, Loader2 } from 'lucide-react';

const PARSING_STATUSES = new Set(['pending', 'extracting', 'parsing']);

/**
 * Resume grid cell contents (not the <td> — both tables own their own cells).
 * States: no resume -> Upload button; parsing -> chip; otherwise -> View link.
 */
export default function ResumeCell({ resume, uploading, disabled, onView, onUpload }) {
  if (uploading) {
    return (
      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: 12, color: '#64748b' }}>
        <Loader2 size={13} style={{ animation: 'spin 1s linear infinite' }} /> Uploading…
      </span>
    );
  }

  if (resume?.id && PARSING_STATUSES.has(resume.parse_status)) {
    return (
      <span
        style={{
          display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: 12,
          color: '#92400e', background: '#fef3c7', borderRadius: 999, padding: '2px 9px',
        }}
      >
        <Loader2 size={12} style={{ animation: 'spin 1s linear infinite' }} /> Parsing…
      </span>
    );
  }

  if (resume?.id) {
    return (
      <button
        type="button"
        onClick={(e) => { e.stopPropagation(); onView?.(); }}
        title={resume.filename || 'View resume'}
        style={{
          background: 'none', border: 'none', padding: 0, cursor: 'pointer',
          color: '#2563eb', fontSize: 13, fontWeight: 600, textDecoration: 'underline',
        }}
      >
        View
      </button>
    );
  }

  return (
    <button
      type="button"
      disabled={disabled}
      onClick={(e) => { e.stopPropagation(); onUpload?.(); }}
      style={{
        display: 'inline-flex', alignItems: 'center', gap: 5,
        border: '1px solid #e2e8f0', borderRadius: 8, background: '#fff',
        padding: '3px 10px', fontSize: 12, fontWeight: 600,
        color: disabled ? '#cbd5e1' : '#475569', cursor: disabled ? 'not-allowed' : 'pointer',
      }}
    >
      <FileUp size={13} /> Upload
    </button>
  );
}
