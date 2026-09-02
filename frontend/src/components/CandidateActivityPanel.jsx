import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Ban, CalendarClock, ClipboardList, ExternalLink, PhoneCall, PhoneIncoming,
  PhoneMissed, PhoneOff, RefreshCw, UserX, Voicemail,
} from 'lucide-react';
import { useAppStore } from '../store/useAppStore';

/**
 * The candidate's call log: every completed call with its recording, duration,
 * recruiter notes and transcript summary.
 *
 * Extracted from pages/Calls.jsx so the Conversations modal can show the same
 * timeline. It previously lived only inside the Calls workspace, so a recruiter
 * looking at a candidate's LinkedIn/Email threads had no way to see that the
 * person had already been spoken to.
 */

// Per-outcome pill color + icon for the Activity History timeline. Hayasa's
// own orange accent family for "needs another look" states — never Klenty's
// blue — plus green/gray/red for the terminal Interested / Not Interested /
// Wrong Number outcomes, matching the semantics already used by StatusDropdown.
// One neutral pill style for every outcome — the icon tone is the only signal
// (Hayasa accent = needs another look, slate = settled/terminal), so the list
// reads as one professional surface instead of a wall of colored chips.
const formatDateTime = (value) => {
  if (!value) return 'Unknown';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit'
  });
};

const ACTIVITY_OUTCOME_META = {
  'Left Voicemail': { tone: 'accent', icon: Voicemail },
  'No Answer': { tone: 'accent', icon: PhoneMissed },
  'Not Connected': { tone: 'accent', icon: PhoneOff },
  'Not Connected - Not Reachable': { tone: 'accent', icon: PhoneOff },
  'Connected - Interested': { tone: 'neutral', icon: PhoneCall },
  'Connected - Not Interested': { tone: 'neutral', icon: UserX },
  'Connected - Follow-up': { tone: 'accent', icon: CalendarClock },
  'Wrong Number': { tone: 'accent', icon: PhoneOff },
  'Unreachable': { tone: 'neutral', icon: PhoneOff },
};
const DEFAULT_ACTIVITY_OUTCOME_META = { tone: 'neutral', icon: PhoneIncoming };
const OUTCOME_PILL_BASE = { bg: '#f8fafc', border: '#e2e8f0', textColor: '#334155' };
const OUTCOME_PILL_ICON_COLOR = { accent: 'var(--accent-primary)', neutral: '#64748b' };

// A task retired by a status change — never dialled. The backend writes
// "Closed - <status>" as a synthetic outcome so the record stays in the log,
// but shown in the same pill as a real outcome it reads as another call: one
// call to a candidate in two roles appeared as two completed entries.
const RETIRED_OUTCOME_PREFIX = 'Closed - ';
const isRetiredTask = (outcome) => String(outcome || '').startsWith(RETIRED_OUTCOME_PREFIX);
const retiredReason = (outcome) => String(outcome || '').slice(RETIRED_OUTCOME_PREFIX.length);

const OutcomeBadge = ({ outcome, fallbackLabel = 'No Outcome', size = 11 }) => {
  if (isRetiredTask(outcome)) {
    return (
      <span
        title={`Not a call. This task was removed from calling when the candidate was marked "${retiredReason(outcome)}".`}
        style={{
          display: 'inline-flex', alignItems: 'center', gap: '5px',
          padding: '2px 10px', borderRadius: '999px',
          background: '#fff', color: '#64748b', border: '1px dashed #cbd5e1',
          fontSize: size, fontWeight: 600, whiteSpace: 'nowrap', fontStyle: 'italic',
        }}
      >
        <Ban size={size} color="#94a3b8" />
        Not called — {retiredReason(outcome)}
      </span>
    );
  }
  const meta = ACTIVITY_OUTCOME_META[outcome] || DEFAULT_ACTIVITY_OUTCOME_META;
  const Icon = meta.icon;
  const iconColor = OUTCOME_PILL_ICON_COLOR[meta.tone];
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: '5px',
      padding: '2px 10px', borderRadius: '999px',
      background: OUTCOME_PILL_BASE.bg, color: OUTCOME_PILL_BASE.textColor,
      border: `1px solid ${OUTCOME_PILL_BASE.border}`,
      fontSize: size, fontWeight: 700, whiteSpace: 'nowrap',
    }}>
      <Icon size={size} color={iconColor} />
      {outcome || fallbackLabel}
    </span>
  );
};

const PossibleVoicemailBadge = ({ size = 11 }) => (
  <span
    title="Auto-flagged from the call transcript — not a live detection, just a suggestion to double-check the logged outcome."
    style={{
      padding: '2px 10px', borderRadius: '999px',
      background: '#fff', color: '#64748b', border: '1px dashed #cbd5e1',
      fontSize: size, fontWeight: 600, whiteSpace: 'nowrap',
    }}
  >
    Possible Voicemail
  </span>
);

function CandidateActivityPanel({ candidateId, candidateName }) {
  const fetchCandidateActivity = useAppStore(state => state.fetchCandidateActivity);
  const syncCallRecording = useAppStore(state => state.syncCallRecording);
  const [syncingCallId, setSyncingCallId] = useState(null);
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState('');
  const itemsRef = useRef([]);

  useEffect(() => {
    itemsRef.current = items;
  }, [items]);

  const loadActivity = useCallback(async ({ force = false, silent = false } = {}) => {
    if (!silent) {
      setRefreshing(true);
    }
    if (!itemsRef.current.length) {
      setLoading(true);
    }

    const res = await fetchCandidateActivity(candidateId, { force });
    if (res.success) {
      setItems(res.data || []);
      setError('');
    } else {
      setError(res.error || 'Failed to load activity');
    }

    setLoading(false);
    setRefreshing(false);
  }, [candidateId, fetchCandidateActivity]);

  const handleSyncRecording = useCallback(async (callId) => {
    setSyncingCallId(callId);
    try {
      // syncCallRecording already clears this candidate's activity cache on
      // success, so the reload below picks up the new recording URL.
      await syncCallRecording(callId);
      await loadActivity({ force: true });
    } finally {
      setSyncingCallId(null);
    }
  }, [syncCallRecording, loadActivity]);

  useEffect(() => {
    setItems([]);
    setError('');
    setLoading(true);
    setRefreshing(false);
    loadActivity({ force: false });
  }, [candidateId, loadActivity]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '400px', background: '#f8fafc' }}>
      <div style={{ padding: '18px 24px', borderBottom: '1px solid #e2e8f0', background: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: '#0f172a' }}>
          <div style={{
            width: '38px',
            height: '38px',
            borderRadius: '12px',
            background: '#f8fafc',
            color: '#475569',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <ClipboardList size={18} />
          </div>
          <div>
            <div style={{ fontSize: '13px', fontWeight: 800 }}>Candidate Activity</div>
            <div style={{ fontSize: '12px', color: '#64748b' }}>{candidateName || 'Candidate'} timeline</div>
          </div>
        </div>
        <button
          onClick={() => loadActivity({ force: true })}
          style={{
            width: '34px',
            height: '34px',
            borderRadius: '10px',
            border: '1px solid #e2e8f0',
            background: '#fff',
            color: '#64748b',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          <RefreshCw size={15} style={{ animation: refreshing ? 'spin 1s linear infinite' : 'none' }} />
        </button>
      </div>

      <div style={{ flex: 1, padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {loading ? (
          Array.from({ length: 2 }).map((_, idx) => (
            <div
              key={`activity-skeleton-${idx}`}
              style={{ padding: '18px', borderRadius: '18px', background: '#fff', border: '1px solid #e2e8f0' }}
            >
              <div style={{ width: '40%', height: 14, borderRadius: 999, background: '#e2e8f0', marginBottom: 12 }} />
              <div style={{ width: '100%', height: 44, borderRadius: 12, background: '#f1f5f9', marginBottom: 12 }} />
              <div style={{ width: '65%', height: 12, borderRadius: 999, background: '#e2e8f0' }} />
            </div>
          ))
        ) : items.length === 0 ? (
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ maxWidth: '360px', textAlign: 'center', display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div style={{ fontSize: '15px', fontWeight: 800, color: '#0f172a' }}>No completed call activity yet</div>
              <div style={{ fontSize: '13px', color: '#64748b', lineHeight: 1.6 }}>
                {error || 'Completed Plivo calls with recordings, summaries, and outcomes will appear here as the activity timeline fills in.'}
              </div>
            </div>
          </div>
        ) : (
          items.map(item => {
            const outcomeMeta = ACTIVITY_OUTCOME_META[item.outcome] || DEFAULT_ACTIVITY_OUTCOME_META;
            const OutcomeIcon = outcomeMeta.icon;
            const outcomeIconColor = OUTCOME_PILL_ICON_COLOR[outcomeMeta.tone];
            return (
            <div key={item.id} style={{ position: 'relative', padding: '18px', borderRadius: '20px', background: '#fff', border: '1px solid #e2e8f0', boxShadow: '0 1px 3px rgba(15, 23, 42, 0.04)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: '16px', marginBottom: '14px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <div style={{ width: '36px', height: '36px', borderRadius: '12px', background: OUTCOME_PILL_BASE.bg, color: outcomeIconColor, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <OutcomeIcon size={18} />
                  </div>
                  <div>
                    <div style={{ fontSize: '15px', fontWeight: 800, color: '#0f172a' }}>Completed Call With {candidateName || 'Candidate'}</div>
                    <div style={{ marginTop: '4px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span style={{ fontSize: '12px', color: '#64748b' }}>{item.status || 'completed'}</span>
                      {item.outcome && <OutcomeBadge outcome={item.outcome} />}
                      {item.likely_voicemail && item.outcome !== 'Left Voicemail' && <PossibleVoicemailBadge />}
                    </div>
                  </div>
                </div>
                <div style={{ fontSize: '12px', color: '#64748b', fontWeight: 600, whiteSpace: 'nowrap' }}>
                  {formatDateTime(item.occurred_at)}
                </div>
              </div>

              {item.recording_url ? (
                <div style={{ marginBottom: '14px', padding: '14px', borderRadius: '14px', background: '#f8fafc', border: '1px solid #e2e8f0' }}>
                  <audio controls src={item.recording_url} style={{ width: '100%' }} />
                </div>
              ) : (
                // Plivo can take minutes to publish a recording. The Calls
                // workspace used to offer this per call in its inline row; that
                // row is gone, so the retry lives here or nowhere.
                <div style={{ marginBottom: '14px', padding: '14px', borderRadius: '14px', background: '#f8fafc', border: '1px solid #e2e8f0', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap' }}>
                  <span style={{ fontSize: '13px', color: '#64748b' }}>Recording not available for this call yet.</span>
                  <button
                    onClick={() => handleSyncRecording(item.id)}
                    disabled={syncingCallId === item.id}
                    style={{
                      padding: '8px 12px', borderRadius: '10px', border: '1px solid rgba(203,213,225,0.9)',
                      background: '#fff', color: '#334155', fontSize: '12px', fontWeight: 700,
                      cursor: syncingCallId === item.id ? 'wait' : 'pointer',
                      display: 'inline-flex', alignItems: 'center', gap: '6px', fontFamily: 'inherit',
                    }}
                  >
                    <RefreshCw size={14} className={syncingCallId === item.id ? 'animate-spin' : ''} />
                    {syncingCallId === item.id ? 'Syncing…' : 'Sync Recording'}
                  </button>
                </div>
              )}

              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px 16px', marginBottom: '14px', fontSize: '13px', color: '#475569' }}>
                <div><strong>From:</strong> {item.from_number || 'N/A'}</div>
                <div><strong>To:</strong> {item.to_number || 'N/A'}</div>
                <div><strong>Duration:</strong> {item.duration_seconds ? `${Math.floor(item.duration_seconds / 60)}m ${item.duration_seconds % 60}s` : '0s'}</div>
              </div>

              {item.notes && (
                <div style={{ marginBottom: '10px', padding: '12px 14px', borderRadius: '12px', background: '#f8fafc', border: '1px solid #e2e8f0' }}>
                  <div style={{ fontSize: '11px', fontWeight: 800, color: '#64748b', textTransform: 'uppercase', marginBottom: '4px' }}>Recruiter Notes</div>
                  <div style={{ fontSize: '13px', color: '#334155', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>{item.notes}</div>
                </div>
              )}

              {item.summary && (
                <div style={{ marginBottom: '10px', fontSize: '14px', color: '#334155', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
                  {item.summary}
                </div>
              )}

              {item.transcript_preview && (
                <div style={{ fontSize: '13px', color: '#64748b', lineHeight: 1.6 }}>
                  {item.transcript_preview}
                </div>
              )}

              {item.source_url && (
                <div style={{ marginTop: '14px' }}>
                  <a
                    href={item.source_url}
                    target="_blank"
                    rel="noreferrer"
                    style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', fontSize: '12px', fontWeight: 700, color: 'var(--accent-primary)', textDecoration: 'none' }}
                  >
                    <ExternalLink size={14} />
                    Open Source
                  </a>
                </div>
              )}
            </div>
            );
          })
        )}
      </div>
    </div>
  );
}

export default CandidateActivityPanel;
export { OutcomeBadge, PossibleVoicemailBadge, ACTIVITY_OUTCOME_META, formatDateTime };
