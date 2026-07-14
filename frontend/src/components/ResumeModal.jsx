import React, { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import { X, Download, RefreshCcw, FileText, Mail, Phone, MapPin, Linkedin, Briefcase, GraduationCap, Award, Sparkles } from 'lucide-react';
import { API_BASE } from '../store/useAppStore';

const SECTION_TITLE = {
  fontSize: 11, fontWeight: 800, letterSpacing: '0.1em', textTransform: 'uppercase',
  color: '#6366f1', borderBottom: '1px solid #e2e8f0', paddingBottom: 6, marginBottom: 12,
  display: 'flex', alignItems: 'center', gap: 6,
};

function ContactChip({ icon: Icon, value, href }) {
  if (!value) return null;
  const inner = (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: 12, color: '#475569' }}>
      <Icon size={12} color="#94a3b8" /> {value}
    </span>
  );
  return href ? (
    <a href={href} target="_blank" rel="noreferrer" style={{ textDecoration: 'none' }}>{inner}</a>
  ) : inner;
}

function formatRange(start, end, isCurrent) {
  const fmt = (v) => {
    if (!v) return '';
    const lower = String(v).toLowerCase();
    if (['present', 'current', 'now'].includes(lower)) return 'Present';
    const [y, m] = String(v).split('-');
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    return m && months[Number(m) - 1] ? `${months[Number(m) - 1]} ${y}` : y || String(v);
  };
  const from = fmt(start);
  const to = isCurrent && !end ? 'Present' : fmt(end);
  if (!from && !to) return '';
  return [from, to].filter(Boolean).join(' — ');
}

/** Document-style rendering of the structured parse. */
function ResumeDocument({ parsed, candidateName }) {
  const name = parsed.full_name || [parsed.first_name, parsed.last_name].filter(Boolean).join(' ') || candidateName || '';
  const roles = Array.isArray(parsed.roles) ? parsed.roles : [];
  const education = Array.isArray(parsed.education) ? parsed.education : [];
  const skills = Array.isArray(parsed.skills) ? parsed.skills : [];
  const certifications = Array.isArray(parsed.certifications) ? parsed.certifications : [];

  return (
    <div
      style={{
        background: '#fff', border: '1px solid #e2e8f0', borderRadius: 12,
        boxShadow: '0 1px 3px rgba(15,23,42,0.06)', padding: '32px 38px',
        maxWidth: 760, margin: '0 auto', color: '#1e293b',
      }}
    >
      <div style={{ borderBottom: '2px solid #0f172a', paddingBottom: 14, marginBottom: 18 }}>
        <div style={{ fontSize: 24, fontWeight: 800, color: '#0f172a', letterSpacing: '-0.01em' }}>{name}</div>
        {parsed.headline && (
          <div style={{ fontSize: 13.5, color: '#475569', marginTop: 4 }}>{parsed.headline}</div>
        )}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, marginTop: 10 }}>
          <ContactChip icon={Mail} value={parsed.email} href={parsed.email ? `mailto:${parsed.email}` : undefined} />
          <ContactChip icon={Phone} value={parsed.phone} />
          <ContactChip icon={MapPin} value={parsed.location || parsed.city} />
          <ContactChip icon={Linkedin} value={parsed.linkedin ? 'LinkedIn' : ''} href={parsed.linkedin} />
        </div>
      </div>

      {parsed.summary && (
        <div style={{ marginBottom: 20 }}>
          <div style={SECTION_TITLE}><Sparkles size={12} /> Summary</div>
          <p style={{ margin: 0, fontSize: 13, lineHeight: 1.65, color: '#334155' }}>{parsed.summary}</p>
        </div>
      )}

      {skills.length > 0 && (
        <div style={{ marginBottom: 20 }}>
          <div style={SECTION_TITLE}><Award size={12} /> Skills</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {skills.map((skill) => (
              <span
                key={skill}
                style={{
                  fontSize: 11.5, fontWeight: 600, color: '#334155', background: '#f1f5f9',
                  border: '1px solid #e2e8f0', borderRadius: 999, padding: '3px 10px',
                }}
              >
                {skill}
              </span>
            ))}
          </div>
        </div>
      )}

      {roles.length > 0 && (
        <div style={{ marginBottom: 20 }}>
          <div style={SECTION_TITLE}><Briefcase size={12} /> Work Experience</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {roles.map((role, index) => (
              <div key={`${role.company}-${index}`} style={{ paddingLeft: 14, borderLeft: '2px solid #e2e8f0' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 12, flexWrap: 'wrap' }}>
                  <div style={{ fontSize: 14, fontWeight: 700, color: '#0f172a' }}>
                    {role.title || 'Role'}
                    {role.company && <span style={{ fontWeight: 500, color: '#475569' }}> · {role.company}</span>}
                  </div>
                  <div style={{ fontSize: 12, color: '#94a3b8', whiteSpace: 'nowrap', fontWeight: 600 }}>
                    {formatRange(role.start_date, role.end_date, role.is_current)}
                  </div>
                </div>
                {role.location && <div style={{ fontSize: 11.5, color: '#94a3b8', marginTop: 2 }}>{role.location}</div>}
                {role.description && (
                  <p style={{ margin: '6px 0 0', fontSize: 12.5, lineHeight: 1.6, color: '#475569', whiteSpace: 'pre-wrap' }}>
                    {role.description}
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {education.length > 0 && (
        <div style={{ marginBottom: 20 }}>
          <div style={SECTION_TITLE}><GraduationCap size={12} /> Education</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {education.map((entry, index) => (
              <div key={`${entry.institution}-${index}`} style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 700, color: '#0f172a' }}>
                    {[entry.degree, entry.field].filter(Boolean).join(', ') || 'Education'}
                  </div>
                  <div style={{ fontSize: 12, color: '#475569' }}>{entry.institution}</div>
                </div>
                <div style={{ fontSize: 12, color: '#94a3b8', whiteSpace: 'nowrap', fontWeight: 600 }}>
                  {formatRange(entry.start_date, entry.end_date)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {certifications.length > 0 && (
        <div>
          <div style={SECTION_TITLE}><Award size={12} /> Certifications</div>
          <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12.5, color: '#475569', lineHeight: 1.7 }}>
            {certifications.map((cert) => <li key={cert}>{cert}</li>)}
          </ul>
        </div>
      )}
    </div>
  );
}

function UpdatesPanel({ applied, proposed }) {
  if (!applied.length && !proposed.length) return null;
  return (
    <div style={{ maxWidth: 760, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 10 }}>
      {applied.length > 0 && (
        <div style={{ background: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: 10, padding: '10px 14px' }}>
          <div style={{ fontSize: 12, fontWeight: 800, color: '#166534', marginBottom: 4 }}>
            Profile fields filled from this resume
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {applied.map((item) => (
              <span key={item.field} style={{ fontSize: 11.5, fontWeight: 600, color: '#166534', background: '#dcfce7', borderRadius: 999, padding: '2px 10px' }}>
                {item.field}
              </span>
            ))}
          </div>
        </div>
      )}
      {proposed.length > 0 && (
        <div style={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 10, padding: '10px 14px' }}>
          <div style={{ fontSize: 12, fontWeight: 800, color: '#0f172a', marginBottom: 6 }}>
            Resume differs from profile <span style={{ fontWeight: 500, color: '#94a3b8' }}>(profile kept, nothing overwritten)</span>
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <tbody>
              {proposed.slice(0, 8).map((change) => (
                <tr key={change.field} style={{ borderTop: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '5px 8px 5px 0', fontWeight: 700, color: '#475569', whiteSpace: 'nowrap', verticalAlign: 'top' }}>{change.field}</td>
                  <td style={{ padding: '5px 8px', color: '#94a3b8', verticalAlign: 'top' }}>{String(change.current_value).slice(0, 90)}</td>
                  <td style={{ padding: '5px 0 5px 8px', color: '#334155', verticalAlign: 'top' }}>{String(change.resume_value).slice(0, 90)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/**
 * Resume preview popup. PDFs render in the native viewer; other formats get a
 * formatted document view built from the structured parse, with a raw-text
 * fallback. Auth is a bearer header, so the file must be blob-fetched.
 */
export default function ResumeModal({ open, candidate, resume, onClose, onReparse }) {
  const [blobUrl, setBlobUrl] = useState('');
  const [loadError, setLoadError] = useState('');
  const [loading, setLoading] = useState(false);
  const [view, setView] = useState('formatted');

  const candidateId = candidate?.id;
  const isPdf = (resume?.content_type || '').includes('pdf');
  const parsed = resume?.parsed_json && typeof resume.parsed_json === 'object' ? resume.parsed_json : {};
  const hasParse = Boolean(parsed.full_name || parsed.summary || (parsed.roles || []).length || (parsed.skills || []).length);

  useEffect(() => {
    if (!open || !candidateId || !resume?.id) return undefined;
    let revoked = false;
    let url = '';
    setLoading(true);
    setLoadError('');
    axios
      .get(`${API_BASE}/candidates/${candidateId}/resume/file`, {
        params: { disposition: 'inline' },
        responseType: 'blob',
      })
      .then((res) => {
        url = URL.createObjectURL(res.data);
        if (!revoked) setBlobUrl(url);
      })
      .catch(() => setLoadError('Could not load the resume file.'))
      .finally(() => setLoading(false));
    return () => {
      revoked = true;
      if (url) URL.revokeObjectURL(url);
      setBlobUrl('');
    };
  }, [open, candidateId, resume?.id]);

  useEffect(() => {
    if (open) setView('formatted');
  }, [open, resume?.id]);

  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  const applied = useMemo(() => (Array.isArray(resume?.applied_fields) ? resume.applied_fields : []), [resume]);
  const proposed = useMemo(() => (Array.isArray(resume?.proposed_changes) ? resume.proposed_changes : []), [resume]);

  if (!open || !resume) return null;

  const showTabs = !isPdf && hasParse;

  return (
    <div
      style={{
        position: 'fixed', inset: 0, background: 'rgba(15,23,42,0.4)', zIndex: 10025,
        display: 'flex', justifyContent: 'center', alignItems: 'center', backdropFilter: 'blur(2px)',
      }}
      onClick={(event) => event.target === event.currentTarget && onClose()}
    >
      <div
        style={{
          width: 'min(920px, 94vw)', maxHeight: '90vh', background: '#f8fafc', borderRadius: 16,
          boxShadow: '0 24px 60px rgba(15,23,42,0.3)', display: 'flex', flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, padding: '14px 20px', borderBottom: '1px solid #e2e8f0', background: '#fff' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, minWidth: 0 }}>
            <div style={{ width: 36, height: 36, borderRadius: 10, background: '#eef2ff', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
              <FileText size={18} color="#4f46e5" />
            </div>
            <div style={{ minWidth: 0 }}>
              <div style={{ fontSize: 15.5, fontWeight: 800, color: '#0f172a', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {parsed.full_name || candidate?.name || 'Resume'}
              </div>
              <div style={{ fontSize: 11.5, color: '#94a3b8', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {resume.filename}{resume.uploaded_at ? ` · uploaded ${new Date(resume.uploaded_at).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })}` : ''}
              </div>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
            {showTabs && (
              <div style={{ display: 'flex', background: '#f1f5f9', borderRadius: 9, padding: 2 }}>
                {[['formatted', 'Formatted'], ['raw', 'Original text']].map(([key, label]) => (
                  <button
                    key={key}
                    type="button"
                    onClick={() => setView(key)}
                    style={{
                      border: 'none', borderRadius: 7, padding: '5px 12px', fontSize: 12, fontWeight: 700,
                      cursor: 'pointer',
                      background: view === key ? '#fff' : 'transparent',
                      color: view === key ? '#0f172a' : '#64748b',
                      boxShadow: view === key ? '0 1px 2px rgba(15,23,42,0.1)' : 'none',
                    }}
                  >
                    {label}
                  </button>
                ))}
              </div>
            )}
            {blobUrl && (
              <a
                href={blobUrl}
                download={resume.filename || 'resume'}
                style={{
                  display: 'inline-flex', alignItems: 'center', gap: 6, background: '#2563eb', color: '#fff',
                  borderRadius: 10, padding: '8px 14px', fontSize: 13, fontWeight: 700, textDecoration: 'none',
                }}
              >
                <Download size={15} /> Download
              </a>
            )}
            <button
              type="button"
              onClick={onClose}
              style={{ width: 36, height: 36, borderRadius: 10, border: '1px solid #e2e8f0', background: '#fff', color: '#64748b', cursor: 'pointer' }}
            >
              <X size={16} />
            </button>
          </div>
        </div>

        <div style={{ padding: 20, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 14 }}>
          {resume.parse_status === 'low_text' && (
            <div style={{ maxWidth: 760, margin: '0 auto', width: '100%', background: '#fef3c7', color: '#92400e', borderRadius: 10, padding: '10px 14px', fontSize: 13 }}>
              No text could be extracted (likely a scanned image). Download to view the original.
            </div>
          )}
          {resume.parse_status === 'failed' && (
            <div style={{ maxWidth: 760, margin: '0 auto', width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10, background: '#fee2e2', color: '#991b1b', borderRadius: 10, padding: '10px 14px', fontSize: 13 }}>
              <span>Parsing failed{resume.parse_error ? `: ${resume.parse_error}` : ''}. The file itself is intact.</span>
              {onReparse && (
                <button
                  type="button"
                  onClick={onReparse}
                  style={{ display: 'inline-flex', alignItems: 'center', gap: 5, border: '1px solid #fca5a5', background: '#fff', color: '#991b1b', borderRadius: 8, padding: '4px 10px', fontSize: 12, fontWeight: 600, cursor: 'pointer', whiteSpace: 'nowrap' }}
                >
                  <RefreshCcw size={12} /> Retry
                </button>
              )}
            </div>
          )}

          {loading && <div style={{ fontSize: 13, color: '#64748b', textAlign: 'center' }}>Loading preview…</div>}
          {loadError && <div style={{ fontSize: 13, color: '#991b1b', textAlign: 'center' }}>{loadError}</div>}

          {!loading && !loadError && isPdf && blobUrl && (
            <iframe src={blobUrl} title="Resume preview" style={{ width: '100%', height: '62vh', border: '1px solid #e2e8f0', borderRadius: 10, background: '#fff' }} />
          )}

          {!loading && !loadError && !isPdf && (
            view === 'formatted' && hasParse ? (
              <ResumeDocument parsed={parsed} candidateName={candidate?.name} />
            ) : (
              <pre
                style={{
                  maxWidth: 760, margin: '0 auto', width: '100%', boxSizing: 'border-box',
                  whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: 12.5, lineHeight: 1.6,
                  color: '#1f2937', background: '#fff', border: '1px solid #e2e8f0', borderRadius: 10,
                  padding: 20, maxHeight: '58vh', overflowY: 'auto',
                }}
              >
                {resume.extracted_text || resume.summary || 'No extracted text available.'}
              </pre>
            )
          )}

          <UpdatesPanel applied={applied} proposed={proposed} />
        </div>
      </div>
    </div>
  );
}
