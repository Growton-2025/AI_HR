/**
 * Renders a call transcript as speaker turns with real names.
 *
 * Lived inside pages/Calls.jsx, where only the View Insights modal could reach
 * it. The Conversations modal showed the same calls but printed the raw stored
 * text — "Lead: ...", truncated to a 220-character preview — so a recruiter
 * cross-checking what a candidate actually said had to reopen the Calls page or
 * replay the audio. Shared from here so both surfaces label transcripts the
 * same way.
 */
// Speaker-hint maps mirror backend/services/call_artifacts.py so the UI labels
// transcripts consistently with the normalization layer.
const RECRUITER_SPEAKER_HINTS = new Set([
  'recruiter', 'agent', 'caller', 'interviewer', 'sales', 'sales rep',
  'sales representative', 'user', 'assistant', 'speaker a', 'speaker 1',
  'channel 0', 'channel 1',
]);
const CANDIDATE_SPEAKER_HINTS = new Set([
  'candidate', 'callee', 'customer', 'client', 'prospect', 'lead',
  'speaker b', 'speaker 2',
]);

const prettifyEmailName = (email) => {
  const raw = String(email || '').trim();
  if (!raw) return '';
  const local = raw.split('@')[0] || '';
  const cleaned = local.replace(/[._-]+/g, ' ').trim();
  if (!cleaned) return '';
  return cleaned
    .split(/\s+/)
    .map(part => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
};

const recruiterDisplayName = (call) => {
  if (!call) return 'Recruiter';
  return (
    prettifyEmailName(call.plivo_recruiter_email) ||
    prettifyEmailName(call.created_by) ||
    'Recruiter'
  );
};

// Parse a raw "Speaker: text" transcript into structured turns with real names.
// Returns [{ side: 'recruiter' | 'candidate', name, text }].
export const parseTranscript = (rawText, { candidateName, recruiterName } = {}) => {
  const text = String(rawText || '').trim();
  if (!text) return [];

  const recruiterLabel = (recruiterName || 'Recruiter').trim() || 'Recruiter';
  const candidateLabel = (candidateName || 'Candidate').trim() || 'Candidate';

  const lines = text.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  // Unicode-aware on purpose: the label is the candidate's real name, and an
  // ASCII-only class silently failed to recognise "Alice Rachael Mendonça:" or
  // "Yash Chopra \u{1F340}:" as speaker labels. Those lines then fell through to the
  // "no label at all" branch and were glued onto the recruiter's previous turn,
  // so the whole call rendered as one uninterrupted recruiter monologue.
  const labeledRe = /^([\p{L}][\p{L}\p{M}\p{N}\p{S} ._'\u2019-]{0,40}?)\s*:\s*(.*)$/u;

  // The backend labels turns with the candidate's full name from the database,
  // but a caller often knows only the first name (the Conversations modal passes
  // `first_name || name`). Requiring an exact match meant "Sanjeet Simha:" did
  // not match candidateName "Sanjeet", every candidate turn fell through to the
  // unknown-speaker fallback below, and the whole call rendered as one long
  // "Recruiter" turn — the candidate's own words attributed to the recruiter.
  const firstToken = (value) => String(value || '').trim().toLowerCase().split(/\s+/)[0] || '';
  const candidateNorm = candidateLabel.toLowerCase();
  const candidateFirst = firstToken(candidateLabel);

  const matchesCandidateName = (norm) => {
    if (!candidateName || !norm) return false;
    if (norm === candidateNorm) return true;
    // "Sanjeet" vs "Sanjeet Simha", in either direction.
    if (norm.startsWith(candidateNorm + ' ') || candidateNorm.startsWith(norm + ' ')) return true;
    return Boolean(candidateFirst) && firstToken(norm) === candidateFirst;
  };

  const sideFromLabel = (label) => {
    const norm = String(label || '').trim().toLowerCase();
    if (RECRUITER_SPEAKER_HINTS.has(norm)) return 'recruiter';
    if (CANDIDATE_SPEAKER_HINTS.has(norm)) return 'candidate';
    if (matchesCandidateName(norm)) return 'candidate';
    if (recruiterName && norm === recruiterLabel.toLowerCase()) return 'recruiter';
    return null;
  };

  const turns = [];
  let anyLabeled = false;
  let firstUnknownSide = null;
  let seenRecruiterLabel = false;
  const unknownSideMap = {};

  for (const line of lines) {
    const match = line.match(labeledRe);
    if (match) {
      const label = match[1];
      const body = match[2].trim();
      let side = sideFromLabel(label);
      if (side) {
        anyLabeled = true;
        if (side === 'recruiter') seenRecruiterLabel = true;
      } else {
        // Unlabeled/unknown speaker: assign first distinct -> recruiter, second -> candidate.
        const key = label.toLowerCase();
        if (!(key in unknownSideMap)) {
          if (seenRecruiterLabel) {
            // A recognised recruiter label already appeared, so whoever this
            // is, they are the other side of the call.
            unknownSideMap[key] = 'candidate';
          } else if (!firstUnknownSide) {
            firstUnknownSide = 'recruiter';
            unknownSideMap[key] = 'recruiter';
          } else {
            unknownSideMap[key] = 'candidate';
          }
        }
        side = unknownSideMap[key];
        anyLabeled = true;
      }
      if (!body) continue;
      const name = side === 'recruiter' ? recruiterLabel : candidateLabel;
      const last = turns[turns.length - 1];
      if (last && last.side === side) {
        last.text += ' ' + body;
      } else {
        turns.push({ side, name, text: body });
      }
    } else {
      // No label at all: append to previous turn if present.
      const last = turns[turns.length - 1];
      if (last) {
        last.text += ' ' + line;
      } else {
        turns.push({ side: 'recruiter', name: recruiterLabel, text: line });
      }
    }
  }

  if (!anyLabeled) {
    // An unlabelled transcript carries no evidence of who spoke, so this used
    // to split it into sentences and alternate speakers. On a 22-second
    // voicemail greeting — one automated voice — that rendered as a four-turn
    // conversation between the recruiter and the candidate, neither of whom
    // said any of it. Returning nothing makes TranscriptView fall back to the
    // plain text, which is the honest reading: these are the words, and we do
    // not know who said them.
    return [];
  }

  return turns;
};

export const TranscriptView = ({ transcript, candidateName, recruiterName, fallback }) => {
  const turns = parseTranscript(transcript, { candidateName, recruiterName });
  if (!turns.length) {
    return (
      <div style={{ whiteSpace: 'pre-wrap' }}>
        {transcript || fallback}
      </div>
    );
  }
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {turns.map((turn, idx) => {
        const isRecruiter = turn.side === 'recruiter';
        return (
          <div
            key={idx}
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: isRecruiter ? 'flex-start' : 'flex-end',
            }}
          >
            <span style={{
              fontSize: 11, fontWeight: 800, letterSpacing: '0.02em',
              color: isRecruiter ? '#6366f1' : '#0f766e',
              marginBottom: 4, textTransform: 'none',
            }}>
              {turn.name}
            </span>
            <div style={{
              maxWidth: '82%',
              padding: '9px 13px',
              borderRadius: isRecruiter ? '4px 14px 14px 14px' : '14px 4px 14px 14px',
              background: isRecruiter ? 'rgba(99,102,241,0.08)' : 'rgba(15,118,110,0.08)',
              border: `1px solid ${isRecruiter ? 'rgba(99,102,241,0.18)' : 'rgba(15,118,110,0.18)'}`,
              color: '#334155',
              fontSize: 13,
              lineHeight: 1.55,
              whiteSpace: 'pre-wrap',
            }}>
              {turn.text}
            </div>
          </div>
        );
      })}
    </div>
  );
};
export default TranscriptView;
