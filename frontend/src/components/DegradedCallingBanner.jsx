import React from 'react';
import { AlertTriangle } from 'lucide-react';
import { useVoIP } from '../context/VoIPContext';

/**
 * Standing warning shown when the backend could not provision this recruiter
 * their own SIP endpoint and fell back to the shared one.
 *
 * Deliberately not dismissible. On the shared endpoint, call attribution
 * matches the most recently updated `calls` row for that SIP username, so if a
 * second recruiter is also on it, one person's recording and transcript get
 * written onto the other's candidate — with no error anywhere. A recruiter
 * needs to know that before they place calls, not after.
 *
 * Mounted next to IncomingCallBanner in App.jsx so it follows the recruiter
 * across every page rather than only the Calls workspace.
 */
export default function DegradedCallingBanner() {
  const { voipDegraded } = useVoIP();

  if (!voipDegraded) return null;

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
      <AlertTriangle size={18} style={{ flexShrink: 0, marginTop: 2, color: '#fbbf24' }} />
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: 0.6, color: '#fbbf24' }}>
          CALLING IS DEGRADED
        </div>
        <div style={{ fontSize: 13, lineHeight: 1.45 }}>
          {voipDegraded.reason}
        </div>
      </div>
    </div>
  );
}
