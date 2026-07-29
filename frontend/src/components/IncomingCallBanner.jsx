import { useEffect, useState } from 'react';
import { Phone, PhoneOff } from 'lucide-react';
import axios from 'axios';
import { API_BASE } from '../store/useAppStore';
import { useVoIP } from '../context/VoIPContext';

/**
 * Quiet notice that a candidate is calling in.
 *
 * Deliberately NOT a modal and deliberately silent: a candidate can call back
 * at any moment, and the recruiter is usually mid-task (typing a filter, in a
 * drawer). This never plays a sound, never takes focus, never changes route —
 * it just offers the call. Rendered from the app shell so it follows the
 * recruiter across every page.
 */
export default function IncomingCallBanner() {
  const { incomingCall, acceptIncomingCall, dismissIncomingCall } = useVoIP();
  const [caller, setCaller] = useState(null);

  useEffect(() => {
    if (!incomingCall?.from) {
      setCaller(null);
      return undefined;
    }
    let cancelled = false;
    // Resolve the number to a candidate so the recruiter knows who it is before
    // deciding to pick up. An unmatched number still shows — a real callback
    // from an unknown number is worse to hide than to show.
    axios
      .get(`${API_BASE}/calls/inbound`, { params: { status: 'pending' } })
      .then(res => {
        if (cancelled) return;
        const digits = String(incomingCall.from).replace(/\D/g, '').slice(-10);
        const match = (res.data?.items || []).find(
          item => String(item.from_number || '').replace(/\D/g, '').slice(-10) === digits
        );
        setCaller(match || null);
      })
      .catch(() => { /* the banner is still useful without the name */ });
    return () => { cancelled = true; };
  }, [incomingCall?.from]);

  if (!incomingCall) return null;

  const name = caller?.candidate_name || 'Unknown caller';
  const subtitle = [caller?.candidate_title, caller?.candidate_company].filter(Boolean).join(' at ');

  return (
    <div
      role="status"
      aria-live="polite"
      style={{
        position: 'fixed', right: 24, bottom: 24, zIndex: 12000,
        background: '#0f172a', color: '#fff', borderRadius: 14, padding: '14px 18px',
        display: 'flex', alignItems: 'center', gap: 16, minWidth: 340,
        boxShadow: '0 20px 40px rgba(15,23,42,0.35)',
        border: '1px solid rgba(255,255,255,0.12)',
        animation: 'slideUp 0.25s ease-out',
      }}
    >
      <div style={{
        width: 38, height: 38, borderRadius: '50%', background: 'var(--accent-primary, #f97316)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
      }}>
        <Phone size={18} />
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: 0.6, color: '#fbbf24' }}>
          INCOMING CALL
        </div>
        <div style={{ fontSize: 14, fontWeight: 700, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {name}
        </div>
        <div style={{ fontSize: 12, color: '#94a3b8', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {subtitle || incomingCall.from}
        </div>
      </div>
      <button
        onClick={() => { void acceptIncomingCall(); }}
        title="Answer"
        style={{
          background: '#16a34a', color: '#fff', border: 'none', borderRadius: 10,
          padding: '9px 14px', fontSize: 13, fontWeight: 700, cursor: 'pointer',
          display: 'flex', alignItems: 'center', gap: 6, fontFamily: 'inherit',
        }}
      >
        <Phone size={14} /> Answer
      </button>
      <button
        onClick={dismissIncomingCall}
        title="Dismiss — it stays in Inbound Callbacks"
        style={{
          background: 'transparent', color: '#94a3b8', border: '1px solid rgba(255,255,255,0.18)',
          borderRadius: 10, padding: '9px 11px', cursor: 'pointer', display: 'flex', alignItems: 'center',
        }}
      >
        <PhoneOff size={14} />
      </button>
    </div>
  );
}
