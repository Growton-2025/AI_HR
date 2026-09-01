import { useEffect, useState } from 'react';
import { Phone, PhoneOff } from 'lucide-react';
import axios from 'axios';
import { API_BASE } from '../store/useAppStore';
import { useVoIP } from '../context/VoIPContext';

/**
 * Alert that a candidate is calling in.
 *
 * Still NOT a modal and it never changes route — the recruiter is usually
 * mid-task (typing a filter, in a drawer) and must not lose their place. But it
 * is no longer silent: VoIPContext rings, raises an OS notification and flashes
 * the tab title alongside this, because a quiet bottom-right toast was missed
 * whenever the recruiter was looking at another tab. Rendered from the app shell
 * so it follows the recruiter across every page.
 */
export default function IncomingCallBanner() {
  const { incomingCall, acceptIncomingCall, dismissIncomingCall, setInboundCallerLabel } = useVoIP();
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
        const resolved = byNumber || recent || null;
        setCaller(resolved);
        // The alert already fired with the raw number; name the person now that
        // we know who it is.
        if (resolved?.candidate_name) setInboundCallerLabel?.(resolved.candidate_name);
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
      // assertive: this interrupts, unlike the polite status it used to be.
      role="alert"
      aria-live="assertive"
      style={{
        position: 'fixed', right: 24, bottom: 24, zIndex: 12000,
        background: '#0f172a', color: '#fff', borderRadius: 16, padding: '18px 22px',
        display: 'flex', alignItems: 'center', gap: 18, minWidth: 420,
        boxShadow: '0 24px 56px rgba(15,23,42,0.45), 0 0 0 3px rgba(249,115,22,0.35)',
        border: '1px solid rgba(255,255,255,0.16)',
        animation: 'slideUp 0.25s ease-out, incomingCallPulse 1.6s ease-in-out infinite',
      }}
    >
      <div style={{
        width: 46, height: 46, borderRadius: '50%', background: 'var(--accent-primary, #f97316)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
      }}>
        <Phone size={22} />
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: 0.6, color: '#fbbf24' }}>
          INCOMING CALL
        </div>
        <div style={{ fontSize: 17, fontWeight: 700, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
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
          padding: '12px 20px', fontSize: 15, fontWeight: 700, cursor: 'pointer',
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
