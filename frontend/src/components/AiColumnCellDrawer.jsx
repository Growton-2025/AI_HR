import React from 'react';
import { Clock3, DollarSign, ExternalLink, ListChecks, ShieldCheck, X } from 'lucide-react';

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

export function renderTextWithLinks(text) {
  if (!text || typeof text !== 'string') return text;
  const urlRegex = /(https?:\/\/[^\s,]+)/g;
  const parts = text.split(urlRegex);
  return parts.map((part, i) => {
    if (part.match(/^https?:\/\//)) {
      return (
        <a key={i} href={part} target="_blank" rel="noreferrer" style={{ color: '#2563eb', textDecoration: 'underline' }}>
          {part}
        </a>
      );
    }
    return part;
  });
}

function isMeaningfulAiValue(value) {
  const text = String(value ?? '').trim();
  const lower = text.toLowerCase();
  return text !== '' && !EMPTY_AI_RESPONSE_TEXTS.has(lower) && !lower.startsWith('needs review:');
}

function resolveDrawerResponse(detail, details, outputs) {
  if (isMeaningfulAiValue(detail?.primary_output)) return detail.primary_output;
  if (isMeaningfulAiValue(details?.response)) return details.response;
  const firstOutput = Object.values(outputs || {}).find(isMeaningfulAiValue);
  if (firstOutput != null) return firstOutput;
  if (isMeaningfulAiValue(details?.reasoning)) return details.reasoning;
  return detail ? 'No' : '';
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
            padding: '5px 8px',
            borderRadius: 8,
            background: depth ? '#f8fafc' : 'rgba(99, 102, 241, 0.06)',
            border: '1px solid #e2e8f0',
          }}>
            {renderAiValuePart(item, depth + 1)}
          </div>
        ))}
      </div>
    );
  }
  if (actual && typeof actual === 'object') {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6, width: '100%' }}>
        {Object.entries(actual).map(([k, v]) => (
          <div key={k} style={{ fontSize: depth ? 12 : 13, lineHeight: 1.5, color: '#334155' }}>
            <span style={{ fontWeight: 700, color: '#1e293b' }}>{formatKey(k)}:</span>{' '}
            {typeof v === 'object' && v !== null ? (
              <div style={{ marginTop: 4 }}>{renderAiValuePart(v, depth + 1)}</div>
            ) : (
              <span style={{ color: '#475569' }}>{typeof v === 'boolean' ? (v ? 'Yes' : 'No') : String(v ?? '—')}</span>
            )}
          </div>
        ))}
      </div>
    );
  }
  if (typeof actual === 'boolean') return actual ? 'Yes' : 'No';
  return <span style={{ whiteSpace: 'pre-wrap' }}>{typeof actual === 'string' ? renderTextWithLinks(actual) : String(actual)}</span>;
}

export function formatInlineObject(obj) {
  if (obj == null) return '—';
  if (Array.isArray(obj)) {
    return obj.map(x => typeof x === 'object' ? formatInlineObject(x) : String(x)).join(', ');
  }
  if (typeof obj === 'object') {
    return Object.entries(obj)
      .map(([k, v]) => `${formatKey(k)}: ${typeof v === 'object' ? formatInlineObject(v) : typeof v === 'boolean' ? (v ? 'Yes' : 'No') : String(v ?? '—')}`)
      .join(' · ');
  }
  return String(obj);
}

export function renderFriendlyAiValue(aiVal, options = {}) {
  const { inline } = options;
  if (aiVal == null || String(aiVal).trim() === '') {
    return '—';
  }
  const parsed = parseStructuredValue(aiVal);
  if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
    if (inline) return <span style={{ whiteSpace: 'pre-wrap' }}>{formatInlineObject(parsed)}</span>;
    return renderAiValuePart(parsed);
  }
  if (parsed && Array.isArray(parsed)) {
    if (inline) return <span style={{ whiteSpace: 'pre-wrap' }}>{formatInlineObject(parsed)}</span>;
    return renderAiValuePart(parsed);
  }
  return <span style={{ whiteSpace: 'pre-wrap' }}>{typeof aiVal === 'string' ? renderTextWithLinks(aiVal) : String(aiVal)}</span>;
}

export default function AiColumnCellDrawer({ open, loading, detail, title, onClose }) {
  if (!open) return null;

  const outputs = detail?.outputs || {};
  const details = detail?.details || {};
  const responseValue = resolveDrawerResponse(detail, details, outputs);
  const steps = Array.isArray(details.steps) ? details.steps : [];
  const sources = Array.isArray(details.sources) ? details.sources : [];
  const unknownReasons = Array.isArray(details.unknown_reasons) ? details.unknown_reasons : [];
  const verificationErrors = Array.isArray(details.verification_errors) ? details.verification_errors : [];
  const queryPlan = details.query_plan && typeof details.query_plan === 'object' ? details.query_plan : null;
  const toolResults = details.tool_results && typeof details.tool_results === 'object' ? details.tool_results : null;
  const searchedAt = details.searched_at || detail?.completed_at || detail?.updated_at;
  const aiCredits = details.ai_credits && typeof details.ai_credits === 'object' ? details.ai_credits : null;
  const aiCreditsAreZero = aiCredits
    && Number(aiCredits.usd || 0) === 0
    && Number(aiCredits.total_tokens || 0) === 0;
  const aiCreditsDisplay = aiCreditsAreZero
    ? 'No AI credits used'
    : (aiCredits?.display || details.ai_credits_display || '—');
  const aiCreditsMeta = aiCredits
    ? (aiCreditsAreZero ? [
        aiCredits.model,
        'row-only deterministic answer',
      ] : [
        aiCredits.model,
        Number.isFinite(Number(aiCredits.total_tokens)) ? `${Number(aiCredits.total_tokens).toLocaleString()} tokens` : '',
        aiCredits.usage_payload_type,
      ]).filter(Boolean).join(' · ')
    : '';
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
                {renderFriendlyAiValue(responseValue)}
              </div>
            </div>

            <div style={sectionStyle}>
              <div style={sectionLabelStyle}>Verification</div>
              <div style={{ fontSize: 13, color: '#0f172a', lineHeight: 1.65, textTransform: 'capitalize' }}>
                {details.verification_status || details.source_verification_status || 'row_context'}
              </div>
              {unknownReasons.length > 0 && (
                <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {unknownReasons.map((reason, index) => (
                    <div key={`${reason}-${index}`} style={{ fontSize: 12, color: '#92400e', lineHeight: 1.5 }}>
                      {reason}
                    </div>
                  ))}
                </div>
              )}
              {verificationErrors.length > 0 && (
                <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {verificationErrors.map((error, index) => (
                    <div key={`${error}-${index}`} style={{ fontSize: 12, color: '#b91c1c', lineHeight: 1.5 }}>
                      {error}
                    </div>
                  ))}
                </div>
              )}
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
              <div style={{ ...sectionLabelStyle, display: 'flex', alignItems: 'center', gap: 8 }}>
                <DollarSign size={14} />
                AI Credits
              </div>
              <div style={{ fontSize: 16, fontWeight: 800, color: '#0f172a', lineHeight: 1.4 }}>
                {aiCreditsDisplay}
              </div>
              {aiCreditsMeta && (
                <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>
                  {aiCreditsMeta}
                </div>
              )}
            </div>

            <div style={sectionStyle}>
              <div style={sectionLabelStyle}>Outputs</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {Object.entries(outputs).length === 0 && <div style={{ fontSize: 12, color: '#94a3b8' }}>No saved output fields.</div>}
                {Object.entries(outputs).map(([key, value]) => (
                  <div key={key} style={{ border: '1px solid #eef2f7', borderRadius: 14, padding: 12, background: '#fff' }}>
                    <div style={{ fontSize: 11, fontWeight: 800, color: '#94a3b8', textTransform: 'uppercase', marginBottom: 6 }}>{key}</div>
                    <div style={{ fontSize: 13, color: '#334155', whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>{renderFriendlyAiValue(value)}</div>
                  </div>
                ))}
              </div>
            </div>

            <div style={sectionStyle}>
              <div style={sectionLabelStyle}>Reasoning</div>
              <div style={{ fontSize: 13, color: '#334155', whiteSpace: 'pre-wrap', lineHeight: 1.7 }}>
                {renderTextWithLinks(details.reasoning || detail.error_message || 'No reasoning captured.')}
              </div>
            </div>

            {queryPlan && (
              <div style={sectionStyle}>
                <div style={sectionLabelStyle}>Query Plan</div>
                <div style={{ fontSize: 12, color: '#334155', lineHeight: 1.7 }}>
                  <div><strong>Tools:</strong> {Array.isArray(queryPlan.tool_calls) ? queryPlan.tool_calls.join(', ') : '—'}</div>
                  <div><strong>Web needed:</strong> {queryPlan.web_needed ? 'Yes' : 'No'}</div>
                  <div><strong>Strictness:</strong> {queryPlan.strictness || 'unknown_instead_of_guess'}</div>
                </div>
              </div>
            )}

            {toolResults && (
              <div style={sectionStyle}>
                <div style={sectionLabelStyle}>Tool Results</div>
                <pre style={{
                  margin: 0,
                  padding: 12,
                  borderRadius: 12,
                  background: '#0f172a',
                  color: '#e2e8f0',
                  fontSize: 11,
                  lineHeight: 1.55,
                  overflowX: 'auto',
                  whiteSpace: 'pre-wrap',
                }}>
                  {JSON.stringify(toolResults, null, 2)}
                </pre>
              </div>
            )}

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
