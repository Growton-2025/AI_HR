import { useEffect, useState } from 'react';
import { Phone, X, Clock, Voicemail, Loader2, PlayCircle } from 'lucide-react';
import axios from 'axios';
import { API_BASE } from '../store/useAppStore';

/**
 * Past calls with one candidate.
 *
 * "View History" on an inbound callback means the CALL history — what was said
 * last time, whether it went to voicemail — not the email/LinkedIn campaign
 * threads, which are role-scoped and belong to outreach rather than dialling.
 */
export default function CallHistoryModal({ candidate, onClose }) {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!candidate?.id) return undefined;
    let cancelled = false;
    setLoading(true);
    axios
      .get(`${API_BASE}/candidates/${candidate.id}/activity`)
      .then(res => { if (!cancelled) setItems(res.data?.items || []); })
      .catch(() => { if (!cancelled) setItems([]); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [candidate?.id]);

  if (!candidate) return null;

  const fmt = (iso) => {
    if (!iso) return '';
    const d = new Date(/Z$|[+-]\d{2}:?\d{2}$/.test(iso) ? iso : `${iso}Z`);
    return Number.isNaN(d.getTime()) ? '' : d.toLocaleString('en-GB', {
      day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit',
      timeZone: 'Asia/Kolkata',
    });
  };
  const duration = (s) => {
    const n = Number(s || 0);
    if (!n) return '—';
    return n < 60 ? `${n}s` : `${Math.floor(n / 60)}m ${n % 60}s`;
  };

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, background: 'rgba(15,23,42,0.55)', backdropFilter: 'blur(3px)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 11000, padding: 24,
      }}
    >
      <div onClick={e => e.stopPropagation()} style={{
        background: '#fff', borderRadius: 18, width: 'min(720px, 100%)', maxHeight: '80vh',
        display: 'flex', flexDirection: 'column', overflow: 'hidden',
        boxShadow: '0 30px 60px rgba(15,23,42,0.3)',
      }}>
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '18px 22px', borderBottom: '1px solid #e2e8f0',
        }}>
          <div>
            <div style={{ fontSize: 17, fontWeight: 800, color: '#0f172a' }}>
              {candidate.name || 'Candidate'}
            </div>
            <div style={{ fontSize: 12, color: '#64748b', display: 'flex', alignItems: 'center', gap: 6 }}>
              <Phone size={12} /> Call history
            </div>
          </div>
          <button onClick={onClose} style={{
            background: 'transparent', border: 'none', cursor: 'pointer', color: '#64748b', padding: 6,
          }}><X size={20} /></button>
        </div>

        <div style={{ padding: '18px 22px', overflowY: 'auto' }}>
          {loading ? (
            <div style={{ padding: 40, textAlign: 'center', color: '#94a3b8' }}>
              <Loader2 size={18} /> Loading call history…
            </div>
          ) : items.length === 0 ? (
            <div style={{ padding: 40, textAlign: 'center', color: '#94a3b8', fontSize: 13 }}>
              No calls logged yet with this candidate.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {items.map(item => (
                <div key={item.id} style={{
                  border: '1px solid #e2e8f0', borderRadius: 12, padding: '13px 15px', background: '#f8fafc',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
                    <span style={{ fontSize: 13, fontWeight: 700, color: '#0f172a' }}>
                      {item.outcome || item.status || 'Call'}
                    </span>
                    {item.likely_voicemail && (
                      <span title="Detected as voicemail" style={{
                        display: 'flex', alignItems: 'center', gap: 4, fontSize: 10, fontWeight: 800,
                        padding: '3px 8px', borderRadius: 999, background: '#fef3c7', color: '#92400e',
                      }}><Voicemail size={11} /> Voicemail</span>
                    )}
                    <span style={{ marginLeft: 'auto', fontSize: 12, color: '#64748b' }}>
                      {fmt(item.occurred_at)}
                    </span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginTop: 6, fontSize: 12, color: '#64748b', flexWrap: 'wrap' }}>
                    <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                      <Clock size={12} /> {duration(item.duration_seconds)}
                    </span>
                    {item.to_number && <span>to {item.to_number}</span>}
                    {item.recording_url && (
                      <a href={item.recording_url} target="_blank" rel="noreferrer"
                        style={{ display: 'flex', alignItems: 'center', gap: 4, color: '#2563eb', fontWeight: 600 }}>
                        <PlayCircle size={12} /> Recording
                      </a>
                    )}
                  </div>
                  {item.summary && (
                    <div style={{ marginTop: 8, fontSize: 12, color: '#334155', background: '#fff', border: '1px solid #f1f5f9', borderRadius: 8, padding: '7px 10px' }}>
                      {item.summary}
                    </div>
                  )}
                  {item.transcript_preview && (
                    <div style={{ marginTop: 6, fontSize: 12, color: '#64748b', fontStyle: 'italic' }}>
                      "{item.transcript_preview}"
                    </div>
                  )}
                  {item.notes && (
                    <div style={{ marginTop: 6, fontSize: 12, color: '#475569' }}>
                      Notes: {item.notes}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
