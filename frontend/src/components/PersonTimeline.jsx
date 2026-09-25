import { useEffect, useState } from 'react';
import { ChevronDown, ChevronUp, Link2, Linkedin, Mail, Phone, PhoneIncoming, RefreshCw, StickyNote, Tag } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';
import { OutcomeBadge, formatDateTime } from './CandidateActivityPanel';
import { TranscriptView } from './TranscriptView';

// Everything Hayasa has done with this *person* — across the master copy,
// every recruiter's copy and archived rows — in one chronological feed.
// Backed by GET /candidates/{id}/timeline (docs/candidate-history-linking-plan.md).

const TYPE_META = {
  call: { icon: Phone, label: 'Call' },
  inbound_call: { icon: PhoneIncoming, label: 'Callback' },
  linkedin_message: { icon: Linkedin, label: 'LinkedIn' },
  email_message: { icon: Mail, label: 'Email' },
  status_change: { icon: Tag, label: 'Status' },
  note: { icon: StickyNote, label: 'Note' },
  link: { icon: Link2, label: 'Linked' },
};

const who = (item) => {
  if (item.you) return 'you';
  return item.by || item.owner_email || 'team';
};

function CallEvent({ item, candidateName }) {
  const [open, setOpen] = useState(!item.collapsed);
  const hasDetail = Boolean(item.recording_url || item.transcript || item.summary);
  return (
    <>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
        {item.outcome && <OutcomeBadge outcome={item.outcome} />}
        {item.duration_seconds > 0 && (
          <span style={{ fontSize: 12, color: '#64748b' }}>{Math.floor(item.duration_seconds / 60)}m {item.duration_seconds % 60}s</span>
        )}
        {item.hangup_cause && <span style={{ fontSize: 12, color: '#94a3b8' }}>{item.hangup_cause}</span>}
        {item.archived && <span style={{ fontSize: 11, color: '#94a3b8', fontStyle: 'italic' }}>archived record</span>}
      </div>
      {item.notes && (
        <div style={{ marginTop: 8, fontSize: 13, color: '#334155', whiteSpace: 'pre-wrap' }}>{item.notes}</div>
      )}
      {hasDetail && (
        <button
          onClick={() => setOpen(v => !v)}
          style={{ marginTop: 8, background: 'none', border: 'none', padding: 0, color: '#2563eb', fontSize: 12, fontWeight: 700, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 4 }}
        >
          {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          {open ? 'Hide details' : `Show ${item.you ? '' : `${who(item)}'s `}recording & transcript`}
        </button>
      )}
      {hasDetail && open && (
        <div style={{ marginTop: 8 }}>
          {item.recording_url && <audio controls src={item.recording_url} style={{ width: '100%', marginBottom: 8 }} />}
          {item.summary && <div style={{ fontSize: 13, color: '#334155', marginBottom: 8, whiteSpace: 'pre-wrap' }}>{item.summary}</div>}
          {item.transcript && (
            <div style={{ maxHeight: 200, overflowY: 'auto', fontSize: 13, color: '#475569', background: '#fff', border: '1px solid #e2e8f0', borderRadius: 10, padding: 12 }}>
              <TranscriptView transcript={item.transcript} candidateName={candidateName} recruiterName={item.by || 'Recruiter'} />
            </div>
          )}
        </div>
      )}
    </>
  );
}

function EventBody({ item, candidateName }) {
  switch (item.type) {
    case 'call':
      return <CallEvent item={item} candidateName={candidateName} />;
    case 'inbound_call':
      return (
        <div style={{ fontSize: 13, color: '#334155' }}>
          {item.answered_at ? `Answered${item.by ? ` by ${item.by}` : ''}` : 'Not answered'}
          {item.duration_seconds > 0 ? ` · ${Math.floor(item.duration_seconds / 60)}m ${item.duration_seconds % 60}s` : ''}
          {item.hangup_cause ? ` · ${item.hangup_cause}` : ''}
          {item.notes && <div style={{ marginTop: 6, whiteSpace: 'pre-wrap' }}>{item.notes}</div>}
        </div>
      );
    case 'linkedin_message':
    case 'email_message':
      return (
        <div style={{ fontSize: 13, color: '#334155' }}>
          <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>
            {item.direction === 'inbound' ? `${candidateName || 'Candidate'} replied` : 'Sent'}
            {item.role ? ` · ${item.role}` : ''}{item.subject ? ` · ${item.subject}` : ''}
          </div>
          <div style={{ whiteSpace: 'pre-wrap', maxHeight: 120, overflow: 'hidden' }}>{item.body}</div>
        </div>
      );
    case 'status_change':
      return <div style={{ fontSize: 13, color: '#334155' }}>{item.old_status || '—'} → <strong>{item.new_status}</strong>{item.by ? ` · ${item.by}` : ''}</div>;
    case 'note':
      return <div style={{ fontSize: 13, color: '#334155', whiteSpace: 'pre-wrap' }}>{item.body}</div>;
    case 'link':
      return <div style={{ fontSize: 13, color: '#64748b' }}>Recognised as the same person (by {item.matched_on})</div>;
    default:
      return null;
  }
}

export default function PersonTimeline({ candidateId, candidateName, compact = false, limit = 0, filterTypes = null }) {
  const fetchCandidateTimeline = useAppStore(state => state.fetchCandidateTimeline);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const load = async (force = false) => {
    setLoading(true);
    const res = await fetchCandidateTimeline(candidateId, { force });
    if (res.success) { setData(res.data); setError(''); } else { setError(res.error || 'Could not load history'); }
    setLoading(false);
  };
  useEffect(() => { if (candidateId) load(); }, [candidateId]); // eslint-disable-line react-hooks/exhaustive-deps

  let items = data?.items || [];
  if (filterTypes) items = items.filter(i => filterTypes.includes(i.type));
  if (limit) items = items.slice(0, limit);
  const rows = data?.rows || [];
  const others = rows.filter(r => r.id !== candidateId);

  if (loading && !data) {
    return <div style={{ padding: compact ? 8 : 24, fontSize: 13, color: '#94a3b8' }}>Loading history…</div>;
  }
  if (error) {
    return <div style={{ padding: compact ? 8 : 24, fontSize: 13, color: '#dc2626' }}>{error}</div>;
  }
  if (!items.length) {
    return <div style={{ padding: compact ? 8 : 24, fontSize: 13, color: '#94a3b8' }}>No previous interactions with this person.</div>;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: compact ? 8 : 12 }}>
      {!compact && (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 12, color: '#64748b' }}>
          <span>
            {others.length > 0
              ? `Includes ${others.length} other record${others.length > 1 ? 's' : ''} of this person (${others.map(r => r.owner_email || 'master library').join(', ')})`
              : 'This record only'}
          </span>
          <button onClick={() => load(true)} style={{ background: 'none', border: 'none', color: '#2563eb', cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 4, fontWeight: 700 }}>
            <RefreshCw size={12} /> Refresh
          </button>
        </div>
      )}
      {items.map(item => {
        const meta = TYPE_META[item.type] || TYPE_META.note;
        const Icon = meta.icon;
        return (
          <div key={item.id} style={{ display: 'flex', gap: 12, padding: compact ? '10px 12px' : 16, borderRadius: 14, background: '#fff', border: '1px solid #e2e8f0' }}>
            <div style={{ width: 32, height: 32, borderRadius: 10, background: '#f1f5f9', color: '#475569', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
              <Icon size={16} />
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, marginBottom: 4 }}>
                <span style={{ fontSize: 12, fontWeight: 800, color: '#0f172a' }}>{meta.label}{item.type !== 'link' ? ` · ${who(item)}` : ''}</span>
                <span style={{ fontSize: 12, color: '#64748b', whiteSpace: 'nowrap' }}>{item.occurred_at ? formatDateTime(item.occurred_at) : ''}</span>
              </div>
              <EventBody item={item} candidateName={candidateName} />
            </div>
          </div>
        );
      })}
    </div>
  );
}
