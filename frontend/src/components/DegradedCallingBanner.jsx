import React, { useState } from 'react';
import { AlertTriangle, PhoneOff, RefreshCcw } from 'lucide-react';
import { useVoIP } from '../context/VoIPContext';

/**
 * Standing warning about the state of this recruiter's calling line.
 *
 * Two states, both of which used to be invisible:
 *
 * 1. DEGRADED — the backend could not provision this recruiter their own SIP
 *    endpoint and fell back to the shared one. Call attribution matches the
 *    most recently updated `calls` row for that SIP username, so if a second
 *    recruiter is also on it, one person's recording and transcript get written
 *    onto the other's candidate, with no error anywhere.
 *
 * 2. NO LINE — the softphone never registered at all. Outbound says so the
 *    moment they press dial, but INBOUND just never rings: the ring-all only
 *    dials registered endpoints, so a candidate calling back reaches voicemail
 *    and the recruiter sees nothing. That silence is how a broken line reached
 *    us as a bug report about a colleague's operating system.
 *
 * Deliberately not dismissible — a recruiter needs to know before they work a
 * calling list, not after. Mounted next to IncomingCallBanner in App.jsx so it
 * follows them across every page rather than only the Calls workspace.
 */
export default function DegradedCallingBanner() {
  const { voipDegraded, voipStatus, voipError, retryVoip } = useVoIP();
  const [retrying, setRetrying] = useState(false);

  // 'error' is the terminal state of the credentials fetch / SDK login. The
  // transient states on the way there (connecting, registering) must not raise
  // an alarm on every page load.
  const noLine = voipStatus === 'error';
  if (!voipDegraded && !noLine) return null;

  const handleRetry = async () => {
    setRetrying(true);
    try {
      await retryVoip();
    } finally {
      setRetrying(false);
    }
  };

  const title = noLine ? 'YOUR CALLING LINE IS NOT SET UP' : 'CALLING IS DEGRADED';
  const body = noLine
    ? (voipError
        ? `${voipError}. Candidate callbacks will not ring on this computer.`
        : 'Candidate callbacks will not ring on this computer, and you cannot place calls.')
    : voipDegraded.reason;

  return (
    <div
      role="alert"
      aria-live="assertive"
      style={{
        position: 'fixed', left: '50%', transform: 'translateX(-50%)', top: 12,
        zIndex: 12001, maxWidth: 620,
        background: '#7c2d12', color: '#fff', borderRadius: 10,
        padding: '10px 16px', display: 'flex', alignItems: 'flex-start', gap: 12,
        boxShadow: '0 12px 28px rgba(15,23,42,0.35)',
        border: '1px solid rgba(251,191,36,0.45)',
      }}
    >
      {noLine
        ? <PhoneOff size={18} style={{ flexShrink: 0, marginTop: 2, color: '#fbbf24' }} />
        : <AlertTriangle size={18} style={{ flexShrink: 0, marginTop: 2, color: '#fbbf24' }} />}
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: 0.6, color: '#fbbf24' }}>
          {title}
        </div>
        <div style={{ fontSize: 13, lineHeight: 1.45 }}>
          {body}
        </div>
      </div>
      {noLine && (
        <button
          type="button"
          onClick={handleRetry}
          disabled={retrying}
          style={{
            background: 'rgba(255,255,255,0.14)', color: '#fff',
            border: '1px solid rgba(255,255,255,0.28)', borderRadius: 8,
            padding: '6px 11px', fontSize: 12, fontWeight: 700,
            cursor: retrying ? 'default' : 'pointer', fontFamily: 'inherit',
            display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0,
          }}
        >
          <RefreshCcw size={13} className={retrying ? 'animate-spin' : ''} />
          {retrying ? 'Retrying…' : 'Retry'}
        </button>
      )}
    </div>
  );
}
