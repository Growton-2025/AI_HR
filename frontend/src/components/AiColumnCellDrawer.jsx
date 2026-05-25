import React from 'react';
import { Clock3, ExternalLink, ListChecks, ShieldCheck, X } from 'lucide-react';

function formatDateTime(value) {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat(undefined, {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
}

export default function AiColumnCellDrawer({ open, loading, detail, title, onClose }) {
  if (!open) return null;

  const outputs = detail?.outputs || {};
  const details = detail?.details || {};
  const steps = Array.isArray(details.steps) ? details.steps : [];
  const sources = Array.isArray(details.sources) ? details.sources : [];
  const searchedAt = details.searched_at || detail?.completed_at || detail?.updated_at;
  const freshnessMeta = [
    details.web_search_tool,
    details.web_search_context_size ? `${details.web_search_context_size} context` : '',
    details.model,
  ].filter(Boolean).join(' · ');

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(15,23,42,0.35)',
        zIndex: 10025,
        display: 'flex',
        justifyContent: 'flex-end',
      }}
      onClick={(event) => event.target === event.currentTarget && onClose()}
    >
      <aside
        style={{
          width: 'min(520px, 100%)',
          height: '100%',
          background: '#fff',
          boxShadow: '-24px 0 60px rgba(15,23,42,0.18)',
          padding: 22,
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: 16,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12 }}>
          <div>
            <div style={{ fontSize: 12, fontWeight: 800, color: '#6366f1', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Cell Details</div>
            <div style={{ fontSize: 18, fontWeight: 800, color: '#0f172a', marginTop: 4 }}>{title || 'Smart Column'}</div>
          </div>
          <button type="button" onClick={onClose} style={{ width: 38, height: 38, borderRadius: 12, border: '1px solid #e2e8f0', background: '#fff', color: '#64748b', cursor: 'pointer' }}>
            <X size={16} />
          </button>
        </div>

        {loading ? (
          <div style={{ fontSize: 13, color: '#64748b' }}>Loading AI cell details…</div>
        ) : !detail ? (
          <div style={{ fontSize: 13, color: '#94a3b8' }}>No AI result found for this cell.</div>
        ) : (
          <>
            <div style={sectionStyle}>
              <div style={sectionLabelStyle}>Response</div>
              <div style={{ fontSize: 14, color: '#0f172a', whiteSpace: 'pre-wrap', lineHeight: 1.7 }}>
                {detail.primary_output || '—'}
              </div>
            </div>

            <div style={sectionStyle}>
              <div style={{ ...sectionLabelStyle, display: 'flex', alignItems: 'center', gap: 8 }}>
                <Clock3 size={14} />
                Freshness
              </div>
              <div style={{ fontSize: 13, color: '#0f172a', lineHeight: 1.65 }}>
                Last searched/updated: {formatDateTime(searchedAt)}
              </div>
              {freshnessMeta && (
                <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>
                  {freshnessMeta}
                </div>
              )}
            </div>

            <div style={sectionStyle}>
              <div style={sectionLabelStyle}>Outputs</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {Object.entries(outputs).length === 0 && <div style={{ fontSize: 12, color: '#94a3b8' }}>No structured outputs stored.</div>}
                {Object.entries(outputs).map(([key, value]) => (
                  <div key={key} style={{ border: '1px solid #eef2f7', borderRadius: 14, padding: 12, background: '#fff' }}>
                    <div style={{ fontSize: 11, fontWeight: 800, color: '#94a3b8', textTransform: 'uppercase', marginBottom: 6 }}>{key}</div>
                    <div style={{ fontSize: 13, color: '#334155', whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>{String(value || '—')}</div>
                  </div>
                ))}
              </div>
            </div>

            <div style={sectionStyle}>
              <div style={sectionLabelStyle}>Reasoning</div>
              <div style={{ fontSize: 13, color: '#334155', whiteSpace: 'pre-wrap', lineHeight: 1.7 }}>
                {details.reasoning || detail.error_message || 'No reasoning captured.'}
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <div style={sectionStyle}>
                <div style={{ ...sectionLabelStyle, display: 'flex', alignItems: 'center', gap: 8 }}>
                  <ShieldCheck size={14} />
                  Confidence
                </div>
                <div style={{ fontSize: 13, color: '#0f172a', textTransform: 'capitalize' }}>{details.confidence || '—'}</div>
              </div>
              <div style={sectionStyle}>
                <div style={sectionLabelStyle}>Status</div>
                <div style={{ fontSize: 13, color: '#0f172a', textTransform: 'capitalize' }}>{detail.status || 'idle'}</div>
              </div>
            </div>

            <div style={sectionStyle}>
              <div style={{ ...sectionLabelStyle, display: 'flex', alignItems: 'center', gap: 8 }}>
                <ListChecks size={14} />
                Steps
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {steps.length === 0 && <div style={{ fontSize: 12, color: '#94a3b8' }}>No steps captured.</div>}
                {steps.map((step, index) => (
                  <div key={`${step}-${index}`} style={{ fontSize: 12.5, color: '#334155', lineHeight: 1.6 }}>
                    {index + 1}. {step}
                  </div>
                ))}
              </div>
            </div>

            <div style={sectionStyle}>
              <div style={sectionLabelStyle}>Sources</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {sources.length === 0 && <div style={{ fontSize: 12, color: '#94a3b8' }}>No sources captured.</div>}
                {sources.map((source, index) => (
                  <div key={`${source.url || source.title || 'source'}-${index}`} style={{ border: '1px solid #eef2f7', borderRadius: 14, padding: 12 }}>
                    <div style={{ fontSize: 13, fontWeight: 700, color: '#0f172a' }}>{source.title || source.url || `Source ${index + 1}`}</div>
                    {source.note && <div style={{ fontSize: 12, color: '#475569', marginTop: 4 }}>{source.note}</div>}
                    {source.url && (
                      <a href={source.url} target="_blank" rel="noreferrer" style={{ fontSize: 12, color: '#2563eb', marginTop: 6, display: 'inline-flex', alignItems: 'center', gap: 6, textDecoration: 'none' }}>
                        Open Source
                        <ExternalLink size={12} />
                      </a>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </aside>
    </div>
  );
}

const sectionStyle = {
  border: '1px solid #eef2f7',
  borderRadius: 18,
  padding: 16,
  background: '#fcfdff',
};

const sectionLabelStyle = {
  fontSize: 11,
  fontWeight: 800,
  color: '#94a3b8',
  textTransform: 'uppercase',
  marginBottom: 8,
  letterSpacing: '0.08em',
};
