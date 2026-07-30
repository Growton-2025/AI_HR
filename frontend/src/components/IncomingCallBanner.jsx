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
    if (!incomingCall) {
      setCaller(null);
      return undefined;
    }
    let cancelled = false;
    // Ask the backend who is calling rather than trusting the SDK's caller-ID:
    // the inbound webhook has already matched the number to a candidate, and
    // the browser SDK does not reliably surface a From on every call shape
    // (which is what made every caller show as "Unknown").
    axios
      .get(`${API_BASE}/calls/inbound`, { params: { status: 'pending' } })
      .then(res => {
        if (cancelled) return;
        const items = res.data?.items || [];
        const digits = String(incomingCall.from || '').replace(/\D/g, '').slice(-10);
        const byNumber = digits
          ? items.find(item => String(item.from_number || '').replace(/\D/g, '').slice(-10) === digits)
          : null;
        // Fallback when the SDK gave us no number: the call currently ringing is
        // the one the webhook just recorded, so take the newest very recent row.
        const recent = items.find(item => {
          if (!item.received_at) return false;
          const raw = item.received_at;
          const at = new Date(/Z$|[+-]\d{2}:?\d{2}$/.test(raw) ? raw : `${raw}Z`);
          return !Number.isNaN(at.getTime()) && Date.now() - at.getTime() < 120000;
        });
        setCaller(byNumber || recent || null);
      })
      .catch(() => { /* the banner is still useful without the name */ });
    return () => { cancelled = true; };
  }, [incomingCall?.callUUID, incomingCall?.from]);

  if (!incomingCall) return null;

  const number = caller?.from_number || incomingCall.from || '';
  const name = caller?.candidate_name || (number ? number : 'Unknown caller');
  const subtitle = [caller?.candidate_title, caller?.candidate_company].filter(Boolean).join(' at ')
    || (caller?.candidate_name ? number : '');

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
          {subtitle}
        </div>
      </div>
      <button
        // Hand over the resolved inbound row so the shell can open the same
        // in-call modal outbound uses (it needs the row id to build a task).
        onClick={() => { void acceptIncomingCall(caller); }}
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
