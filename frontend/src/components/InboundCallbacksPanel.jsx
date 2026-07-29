import { useCallback, useEffect, useState } from 'react';
import { PhoneIncoming, Phone, FileClock, Plus, Loader2 } from 'lucide-react';
import axios from 'axios';
import { toast } from 'sonner';
import { API_BASE } from '../store/useAppStore';

const FILTERS = [
  { id: 'all', label: 'All' },
  { id: 'pending', label: 'Pending' },
  { id: 'resolved', label: 'Resolved' },
];

function relativeTime(iso) {
  if (!iso) return '';
  const then = new Date(iso.endsWith('Z') || /[+-]\d{2}:?\d{2}$/.test(iso) ? iso : `${iso}Z`);
  if (Number.isNaN(then.getTime())) return '';
  const mins = Math.floor((Date.now() - then.getTime()) / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins} min${mins === 1 ? '' : 's'} ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs} hour${hrs === 1 ? '' : 's'} ago`;
  return then.toLocaleString('en-GB', {
    day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit',
    timeZone: 'Asia/Kolkata',
  });
}

/**
 * Candidates who rang back after outreach.
 *
 * Resolution is intentionally outcome-independent: per the product owner,
 * "the moment you complete the callback, irrespective of whether it is
 * connected or not, it should be marked as callback completed and the number
 * here will reduce." So the callback attempt clears the row — we never inspect
 * whether it actually connected.
 */
export default function InboundCallbacksPanel({ onCallBack, onViewHistory, onChanged }) {
  const [items, setItems] = useState([]);
  const [filter, setFilter] = useState('pending');
  const [loading, setLoading] = useState(true);
  const [busyId, setBusyId] = useState(null);
  const [logging, setLogging] = useState(false);
  const [manualNumber, setManualNumber] = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const params = filter === 'all' ? {} : { status: filter };
      const res = await axios.get(`${API_BASE}/calls/inbound`, { params });
      setItems(res.data?.items || []);
    } catch (e) {
      console.error('Failed to load inbound callbacks', e);
    } finally {
      setLoading(false);
    }
  }, [filter]);

  useEffect(() => { void load(); }, [load]);

  const resolve = async (item) => {
    setBusyId(item.id);
    try {
      await axios.post(`${API_BASE}/calls/inbound/${item.id}/resolve`, {});
      toast.success(`Callback to ${item.candidate_name || item.from_number} marked complete`);
      await load();
      onChanged?.();
    } catch (e) {
      toast.error(e.response?.data?.detail || 'Could not mark the callback complete');
    } finally {
      setBusyId(null);
    }
  };

  const handleCallBack = async (item) => {
    // Dial first, then resolve regardless of how the dial went.
    try { await onCallBack?.(item); } catch (_) { /* resolve anyway */ }
    await resolve(item);
  };

  const logManual = async () => {
    const number = manualNumber.trim();
    if (!number) return;
    setLogging(true);
    try {
      await axios.post(`${API_BASE}/calls/inbound/manual`, { from_number: number });
      setManualNumber('');
      toast.success('Incoming call logged');
      await load();
      onChanged?.();
    } catch (e) {
      toast.error(e.response?.data?.detail || 'Could not log the call');
    } finally {
      setLogging(false);
    }
  };

  return (
    <div style={{ background: '#fff', borderRadius: 16, border: '1px solid #e2e8f0', padding: '20px 22px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
        <PhoneIncoming size={18} style={{ color: 'var(--accent-primary, #f97316)' }} />
        <h3 style={{ margin: 0, fontSize: 17, fontWeight: 800, color: '#0f172a' }}>Inbound Missed Callbacks</h3>
      </div>
      <p style={{ margin: '0 0 14px', fontSize: 13, color: '#64748b' }}>
        Candidates who called back after receiving outreach voicemails or emails.
        One-click callback automatically completes call tasks.
      </p>

      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
        {FILTERS.map(f => (
          <button key={f.id} onClick={() => setFilter(f.id)}
            style={{
              padding: '6px 13px', borderRadius: 999, fontSize: 12, fontWeight: 700, cursor: 'pointer',
              fontFamily: 'inherit',
              border: filter === f.id ? '1px solid #111827' : '1px solid #e2e8f0',
              background: filter === f.id ? '#111827' : '#fff',
              color: filter === f.id ? '#fff' : '#64748b',
            }}>
            {f.label}
          </button>
        ))}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginLeft: 'auto' }}>
          <input
            value={manualNumber}
            onChange={e => setManualNumber(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') void logManual(); }}
            placeholder="+91 98765 43210"
            style={{
              padding: '7px 11px', borderRadius: 8, border: '1px solid #e2e8f0',
              fontSize: 12, fontFamily: 'inherit', width: 150,
            }}
          />
          <button onClick={logManual} disabled={logging || !manualNumber.trim()}
            style={{
              padding: '7px 12px', borderRadius: 8, border: '1px solid #e2e8f0', background: '#fff',
              fontSize: 12, fontWeight: 700, color: '#0f172a', fontFamily: 'inherit',
              cursor: logging || !manualNumber.trim() ? 'not-allowed' : 'pointer',
              display: 'flex', alignItems: 'center', gap: 6,
              opacity: logging || !manualNumber.trim() ? 0.55 : 1,
            }}>
            <Plus size={13} /> Log Incoming Call
          </button>
        </div>
      </div>

      {loading ? (
        <div style={{ padding: 32, textAlign: 'center', color: '#94a3b8' }}>
          <Loader2 size={18} className="spin" /> Loading callbacks…
        </div>
      ) : items.length === 0 ? (
        <div style={{ padding: 32, textAlign: 'center', color: '#94a3b8', fontSize: 13 }}>
          No inbound callbacks {filter === 'pending' ? 'waiting' : 'to show'}.
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {items.map(item => {
            const resolved = item.status === 'resolved';
            const initials = (item.candidate_name || '?')
              .split(' ').map(p => p[0]).filter(Boolean).slice(0, 2).join('').toUpperCase();
            return (
              <div key={item.id} style={{
                display: 'flex', alignItems: 'center', gap: 14, padding: '14px 16px',
                borderRadius: 12, border: '1px solid #fde68a',
                background: resolved ? '#f8fafc' : '#fffbeb',
                opacity: resolved ? 0.72 : 1,
              }}>
                <div style={{
                  width: 38, height: 38, borderRadius: '50%', flexShrink: 0,
                  background: '#fef3c7', color: '#92400e', fontWeight: 800, fontSize: 13,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>{initials}</div>

                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                    <span style={{ fontWeight: 800, fontSize: 14, color: '#0f172a' }}>
                      {item.candidate_name || 'Unknown caller'}
                    </span>
                    {(item.candidate_title || item.candidate_company) && (
                      <span style={{ fontSize: 12, color: '#64748b' }}>
                        · {[item.candidate_title, item.candidate_company].filter(Boolean).join(' at ')}
                      </span>
                    )}
                    <span style={{
                      fontSize: 10, fontWeight: 800, padding: '3px 8px', borderRadius: 999,
                      background: resolved ? '#dcfce7' : '#fef3c7',
                      color: resolved ? '#166534' : '#92400e',
                    }}>
                      {resolved ? 'Callback Completed' : 'Missed Callback'}
                    </span>
                    {item.is_unknown && (
                      <span title="This number is not in the talent pool" style={{
                        fontSize: 10, fontWeight: 800, padding: '3px 8px', borderRadius: 999,
                        background: '#e2e8f0', color: '#475569',
                      }}>Unknown number</span>
                    )}
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 4, fontSize: 12, color: '#64748b', flexWrap: 'wrap' }}>
                    <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                      <Phone size={12} /> {item.from_number}
                    </span>
                    <span>{relativeTime(item.received_at)}</span>
                    {item.candidate_status && (
                      <span style={{ color: '#2563eb', fontWeight: 600 }}>
                        Current Status: {item.candidate_status}
                      </span>
                    )}
                  </div>
                  {item.note && (
                    <div style={{
                      marginTop: 7, fontSize: 12, color: '#475569', fontStyle: 'italic',
                      background: '#fff', border: '1px solid #f1f5f9', borderRadius: 8, padding: '6px 10px',
                    }}>"{item.note}"</div>
                  )}
                </div>

                {item.candidate_id && (
                  <button onClick={() => onViewHistory?.(item)}
                    style={{
                      padding: '8px 13px', borderRadius: 9, border: '1px solid #e2e8f0', background: '#fff',
                      fontSize: 12, fontWeight: 700, color: '#0f172a', cursor: 'pointer',
                      display: 'flex', alignItems: 'center', gap: 6, fontFamily: 'inherit', flexShrink: 0,
                    }}>
                    <FileClock size={13} /> View History
                  </button>
                )}
                {!resolved && (
                  <button onClick={() => handleCallBack(item)} disabled={busyId === item.id}
                    style={{
                      padding: '8px 14px', borderRadius: 9, border: 'none', background: '#16a34a',
                      color: '#fff', fontSize: 12, fontWeight: 700, fontFamily: 'inherit',
                      cursor: busyId === item.id ? 'wait' : 'pointer', flexShrink: 0,
                      display: 'flex', alignItems: 'center', gap: 6,
                      opacity: busyId === item.id ? 0.7 : 1,
                    }}>
                    <Phone size={13} /> {busyId === item.id ? 'Calling…' : '1-Click Call Back'}
                  </button>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
