import { VoIPProvider, useVoIP, reportTiming } from '../context/VoIPContext';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import StatusDropdown from '../components/StatusDropdown';
import { useSearchParams } from 'react-router-dom';
import { 
  Phone, Calendar, CheckCircle2, List, PhoneCall, 
  Search, RefreshCw, MoreHorizontal, User, 
  Trash2, X, ChevronLeft, Send, MessageSquare, 
  CheckSquare, ExternalLink, Clock, PhoneForwarded, Mail,
  ClipboardList, Layers, PhoneIncoming, Loader2, Linkedin,
  Smile, Meh, Frown
} from 'lucide-react';
import axios from 'axios';
import { API_BASE, BACKEND_BASE, canonicalCallsQuery, useAppStore } from '../store/useAppStore';
import { toast } from 'sonner';
import { useShallow } from 'zustand/react/shallow';

const formatLocalDate = (dateString) => {
  if (!dateString) return 'N/A';
  // dateString is typically YYYY-MM-DD
  const [year, month, day] = dateString.split('-').map(Number);
  if (!year || !month || !day) return dateString;
  const date = new Date(year, month - 1, day);
  return date.toLocaleDateString(undefined, { 
    month: 'short', 
    day: 'numeric', 
    year: 'numeric' 
  });
};

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
const parseTranscript = (rawText, { candidateName, recruiterName } = {}) => {
  const text = String(rawText || '').trim();
  if (!text) return [];

  const recruiterLabel = (recruiterName || 'Recruiter').trim() || 'Recruiter';
  const candidateLabel = (candidateName || 'Candidate').trim() || 'Candidate';

  const lines = text.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  const labeledRe = /^([A-Za-z][A-Za-z0-9 ._-]{0,40}?)\s*:\s*(.*)$/;

  const sideFromLabel = (label) => {
    const norm = String(label || '').trim().toLowerCase();
    if (RECRUITER_SPEAKER_HINTS.has(norm)) return 'recruiter';
    if (CANDIDATE_SPEAKER_HINTS.has(norm)) return 'candidate';
    if (candidateName && norm === candidateLabel.toLowerCase()) return 'candidate';
    if (recruiterName && norm === recruiterLabel.toLowerCase()) return 'recruiter';
    return null;
  };

  const turns = [];
  let anyLabeled = false;
  let firstUnknownSide = null;
  const unknownSideMap = {};

  for (const line of lines) {
    const match = line.match(labeledRe);
    if (match) {
      const label = match[1];
      const body = match[2].trim();
      let side = sideFromLabel(label);
      if (side) {
        anyLabeled = true;
      } else {
        // Unlabeled/unknown speaker: assign first distinct -> recruiter, second -> candidate.
        const key = label.toLowerCase();
        if (!(key in unknownSideMap)) {
          if (!firstUnknownSide) {
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
    // Fully unlabeled transcript: split into sentences and alternate speakers.
    const sentences = text
      .split(/(?<=[.!?])\s+/)
      .map(s => s.trim())
      .filter(Boolean);
    if (sentences.length >= 2 && sentences.length <= 40) {
      return sentences.map((sentence, idx) => {
        const side = idx % 2 === 0 ? 'recruiter' : 'candidate';
        return {
          side,
          name: side === 'recruiter' ? recruiterLabel : candidateLabel,
          text: sentence,
        };
      });
    }
    return [];
  }

  return turns;
};

const TranscriptView = ({ transcript, candidateName, recruiterName, fallback }) => {
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

const TABS = [
  { id: 'today', label: 'Due Today', icon: Clock },
  { id: 'upcoming', label: 'Upcoming', icon: Calendar },
  { id: 'completed', label: 'Completed', icon: CheckCircle2 },
  { id: 'lists', label: 'Call Lists', icon: Layers },
];

// Call status options — each drives a different next step (see cadence below).
const OUTCOMES = [
  'Not Connected',
  'Not Connected - Not Reachable',
  'Connected - Interested',
  'Connected - Not Interested',
  'Connected - Follow-up',
  'Wrong Number'
];

// Outcomes that mean "did not connect": the backend schedules the next attempt
// in the Day 1 → 2 → 4 → 7 → 10 cadence (5 attempts, then auto-Unreachable).
const FAILED_OUTCOMES = new Set(['Not Connected', 'Not Connected - Not Reachable']);
const FOLLOWUP_OUTCOME = 'Connected - Follow-up';

const formatCallTimer = (totalSeconds) => {
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  const mmss = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
  return hours > 0 ? `${hours}:${mmss}` : mmss;
};
const WRONG_NUMBER_OUTCOME = 'Wrong Number';
const FINAL_ATTEMPT_PREFIX = 'Call 5';

const isFinalAttempt = (call) => (call?.task_title || '').startsWith(FINAL_ATTEMPT_PREFIX);

// "15:30:00" → "15:30" for display.
const formatDueTime = (value) => {
  const text = String(value || '');
  return /^\d{2}:\d{2}/.test(text) ? text.slice(0, 5) : text;
};

const CALL_PANEL_STYLE = {
  background: 'rgba(255,255,255,0.86)',
  backdropFilter: 'blur(16px)',
  border: '1px solid rgba(226,232,240,0.92)',
  boxShadow: '0 18px 36px rgba(15,23,42,0.05)',
};

// Klenty-style dense calls table: Prospect | Purpose | Status | Date | Outcome | Actions.
// Single source of truth for colSpan so header/skeleton/empty-state/expanded-row spans
// never drift — the exact "magic number colSpan" landmine that bit Roles.jsx.
const CALLS_TABLE_COL_COUNT = 6;
const CALLS_TABLE_TH_STYLE = {
  textAlign: 'left', padding: '12px 16px', fontSize: '11px', fontWeight: 800,
  color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em',
  borderBottom: '1px solid #e2e8f0', whiteSpace: 'nowrap',
};

const CALL_PRIMARY_BUTTON = {
  background: '#111827',
  color: '#fff',
  border: '1px solid #111827',
  borderRadius: '12px',
  fontWeight: 700,
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: '8px',
  boxShadow: '0 12px 24px rgba(15,23,42,0.12)',
};

const CALL_SECONDARY_BUTTON = {
  background: '#fff',
  color: '#334155',
  border: '1px solid rgba(203,213,225,0.9)',
  borderRadius: '12px',
  fontWeight: 700,
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: '8px',
};

const SUMMARY_PLACEHOLDER_PATTERNS = [
  "transcript isn't fully provided",
  'transcript is not fully provided',
  'please share the full',
  'please share the full or additional content',
  'please share additional content',
  'i can help summarize a typical recruitment call',
  'when provided with full details',
  'for me to assist you appropriately',
  'not enough information',
  'insufficient information',
];

const hasPlaceholderSummary = (value) => {
  const text = (value || '').trim().toLowerCase();
  if (!text) return false;
  return SUMMARY_PLACEHOLDER_PATTERNS.some(pattern => text.includes(pattern));
};

const needsPostCallArtifacts = (callData) => {
  if (!callData) return true;
  if (!callData.recording_url) return true;
  if (!(callData.transcript || '').trim()) return true;
  if (!(callData.summary || '').trim()) return true;
  if (hasPlaceholderSummary(callData.summary)) return true;
  return false;
};

// AI-derived sentiment of the Lead's side of the conversation (Klenty calls this
// "Sentiments" in Call IQ) — distinct from `outcome`, which the recruiter sets by hand.
const SENTIMENT_META = {
  Positive: { icon: Smile, color: '#059669', bg: '#ecfdf5', border: '#a7f3d0' },
  Neutral: { icon: Meh, color: '#64748b', bg: '#f8fafc', border: '#e2e8f0' },
  Negative: { icon: Frown, color: '#dc2626', bg: '#fef2f2', border: '#fecaca' },
};

const SentimentBadge = ({ sentiment, reason, size = 12 }) => {
  const meta = SENTIMENT_META[sentiment];
  if (!meta) return null;
  const Icon = meta.icon;
  return (
    <span
      title={reason || sentiment}
      style={{
        display: 'inline-flex', alignItems: 'center', gap: '5px', padding: '2px 8px',
        borderRadius: '999px', background: meta.bg, color: meta.color, fontWeight: 700,
        fontSize: size, border: `1px solid ${meta.border}`, width: 'fit-content',
        // Dotted underline on the label signals "hover for why" in tight spaces
        // (table cells) where the full reason can't be shown inline.
        textDecoration: reason ? 'underline dotted' : 'none', textUnderlineOffset: '2px',
      }}
    >
      <Icon size={size} /> {sentiment}
    </span>
  );
};

const PENDING_ANALYSIS_WINDOW_MS = 30 * 60 * 1000;
const BACKGROUND_ANALYSIS_POLL_MS = 9000;

const isPendingAnalysis = (callData) => (
  callData?.status === 'completed'
  && needsPostCallArtifacts(callData)
  && Boolean(callData.completed_at)
  && (Date.now() - new Date(callData.completed_at).getTime()) < PENDING_ANALYSIS_WINDOW_MS
);

const SOFTPHONE_PREPARING_TIMEOUT_MS = 12000;
const SOFTPHONE_FIRST_CLICK_RECOVERY_MS = 18000;
const isDocumentVisible = () => typeof document === 'undefined' || document.visibilityState === 'visible';

const cleanVoipReasonText = (value) => (
  String(value || '')
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
);

const buildCallWrapUpMeta = (event, candidateName) => {
  const firstName = candidateName?.trim()?.split(/\s+/)?.[0] || 'The candidate';
  const reasonText = cleanVoipReasonText(event?.reasonText || event?.message || '');
  const lowerReason = reasonText.toLowerCase();

  const baseMeta = {
    label: 'Wrap-up',
    title: 'Call ended',
    message: 'Capture the outcome and next step below.',
    tone: '#8b6b44',
    bg: '#f8f5ef',
    border: 'rgba(148, 115, 77, 0.22)',
    suggestedOutcome: '',
  };

  if (event?.type === 'failed') {
    if (lowerReason.includes('busy')) {
      return {
        ...baseMeta,
        title: 'Call did not connect',
        message: reasonText
          ? `Plivo reported the browser call as busy: ${reasonText}.`
          : 'Plivo reported the browser call as busy, so it did not connect.',
      };
    }

    if (lowerReason.includes('no answer') || lowerReason.includes('timeout') || lowerReason.includes('unanswered')) {
      return {
        ...baseMeta,
        title: 'No answer',
        message: `${firstName} did not answer the browser call.`,
        suggestedOutcome: 'Not Connected',
      };
    }

    if (lowerReason.includes('rejected') || lowerReason.includes('declined') || lowerReason.includes('cancel')) {
      return {
        ...baseMeta,
        title: 'Call declined',
        message: `${firstName} declined or ended the browser call before it connected.`,
      };
    }

    return {
      ...baseMeta,
      title: 'Call could not connect',
      message: reasonText
        ? `The browser call did not connect: ${reasonText}.`
        : 'The browser call did not connect. Review the outcome and next step below.',
    };
  }

  if (event?.origin === 'local') {
    return {
      ...baseMeta,
      title: 'Call ended',
      message: 'You ended the browser call. Capture the outcome and next step below.',
    };
  }

  if (lowerReason.includes('busy')) {
    return {
      ...baseMeta,
      title: 'Call ended',
      message: reasonText
        ? `The browser call ended with a busy signal from Plivo: ${reasonText}.`
        : 'The browser call ended with a busy signal from Plivo.',
    };
  }

  if (lowerReason.includes('no answer') || lowerReason.includes('timeout') || lowerReason.includes('unanswered')) {
    return {
      ...baseMeta,
      title: 'No answer',
      message: `${firstName} did not answer the call.`,
      suggestedOutcome: 'Not Connected',
    };
  }

  if (
    lowerReason.includes('remote')
    || lowerReason.includes('hangup')
    || lowerReason.includes('hang up')
    || lowerReason.includes('terminated')
    || lowerReason.includes('rejected')
    || lowerReason.includes('declined')
  ) {
    return {
      ...baseMeta,
      title: 'Candidate disconnected',
      message: `${firstName} ended the call from their side.`,
      tone: '#334155',
      bg: '#f8fafc',
      border: 'rgba(148, 163, 184, 0.24)',
    };
  }

  if (reasonText) {
    return {
      ...baseMeta,
      message: `The browser call ended: ${reasonText}.`,
      tone: '#334155',
      bg: '#f8fafc',
      border: 'rgba(148, 163, 184, 0.24)',
    };
  }

  return {
    ...baseMeta,
    message: `${firstName} ended or disconnected the browser call.`,
    tone: '#334155',
    bg: '#f8fafc',
    border: 'rgba(148, 163, 184, 0.24)',
  };
};

function ConversationHistoryPanel({ candidateId, candidateName, platform }) {
  const fetchChatHistory = useAppStore(state => state.fetchChatHistory);
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState('');
  const messagesEndRef = useRef(null);
  const stateRef = useRef({ messages: [], loaded: false });
  const requestSeqRef = useRef(0);

  useEffect(() => {
    stateRef.current = { messages, loaded };
  }, [messages, loaded]);

  const loadMessages = useCallback(async ({ showLoader = false, silent = false, force = false } = {}) => {
    const requestSeq = ++requestSeqRef.current;
    const hasCachedMessages = stateRef.current.messages.length > 0 || stateRef.current.loaded;
    const shouldBlock = showLoader && !hasCachedMessages;

    if (shouldBlock) {
      setLoading(true);
    } else if (!silent) {
      setRefreshing(true);
    }

    const res = await fetchChatHistory(0, candidateId, platform, force);
    if (requestSeqRef.current !== requestSeq) return;

    if (res.success) {
      setMessages(res.messages || []);
      setError('');
    } else {
      setError(res.error || `Failed to fetch ${platform} history`);
    }

    setLoaded(true);
    setLoading(false);
    if (!silent) setRefreshing(false);
  }, [candidateId, fetchChatHistory, platform]);

  useEffect(() => {
    requestSeqRef.current = 0;
    stateRef.current = { messages: [], loaded: false };
    setMessages([]);
    setError('');
    setLoaded(false);
    setLoading(true);
    setRefreshing(false);
    loadMessages({ showLoader: true });
  }, [candidateId, platform, loadMessages]);

  useEffect(() => {
    const interval = setInterval(() => {
      if (isDocumentVisible()) loadMessages({ silent: true });
    }, 5000);
    return () => clearInterval(interval);
  }, [loadMessages]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const accent = platform === 'linkedin' ? '#0a66c2' : '#f97316';
  const icon = platform === 'linkedin' ? <ExternalLink size={18} /> : <Mail size={18} />;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '400px', background: '#f8fafc' }}>
      <div style={{ padding: '18px 24px', borderBottom: '1px solid #e2e8f0', background: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: '#0f172a' }}>
          <div style={{
            width: '38px',
            height: '38px',
            borderRadius: '12px',
            background: platform === 'linkedin' ? '#eff6ff' : '#fff7ed',
            color: accent,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            {icon}
          </div>
          <div>
            <div style={{ fontSize: '13px', fontWeight: 800 }}>{platform === 'linkedin' ? 'LinkedIn Responses' : 'Email Responses'}</div>
            <div style={{ fontSize: '12px', color: '#64748b' }}>{candidateName || 'Candidate'} conversation history</div>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            padding: '7px 12px',
            borderRadius: '999px',
            background: refreshing ? '#eff6ff' : '#fff',
            border: `1px solid ${refreshing ? '#bfdbfe' : '#e2e8f0'}`,
            color: refreshing ? '#2563eb' : '#64748b',
            fontSize: '11px',
            fontWeight: 700,
            transition: 'all 0.2s ease'
          }}>
            {loading && !loaded
              ? 'Loading thread...'
              : refreshing
                ? 'Checking for replies...'
                : error && messages.length === 0
                  ? 'Refresh failed'
                  : `${messages.length} message${messages.length === 1 ? '' : 's'}`}
          </div>
          <button
            onClick={() => loadMessages({ showLoader: messages.length === 0 })}
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
      </div>

      <div style={{ position: 'relative', flex: 1, padding: '20px 24px', display: 'flex', flexDirection: 'column', gap: '16px', overflowY: 'auto' }}>
        {loading && !loaded ? (
          <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: '16px', flex: 1 }}>
            {Array.from({ length: 4 }).map((_, idx) => {
              const isIncoming = idx % 2 === 0;
              return (
                <div key={`call-thread-skeleton-${idx}`} style={{ display: 'flex', flexDirection: 'column', alignItems: isIncoming ? 'flex-start' : 'flex-end', gap: '6px' }}>
                  <div style={{ width: 92, height: 10, borderRadius: 999, background: '#e2e8f0' }} />
                  <div style={{
                    width: `${isIncoming ? 56 : 44}%`,
                    minWidth: '190px',
                    maxWidth: '78%',
                    height: idx === 2 ? 66 : 50,
                    borderRadius: isIncoming ? '8px 18px 18px 18px' : '18px 8px 18px 18px',
                    background: 'linear-gradient(90deg,#f1f5f9 20%,#e2e8f0 50%,#f1f5f9 80%)',
                    backgroundSize: '200% 100%',
                    animation: 'shimmer 1.2s linear infinite'
                  }} />
                </div>
              );
            })}
          </div>
        ) : messages.length === 0 ? (
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ maxWidth: '340px', textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px' }}>
              <div style={{
                width: '64px',
                height: '64px',
                borderRadius: '20px',
                background: platform === 'linkedin' ? '#eff6ff' : '#fff7ed',
                color: accent,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                {icon}
              </div>
              <div style={{ fontSize: '15px', fontWeight: 700, color: '#0f172a' }}>
                {error ? `Could not load ${platform} conversation` : `No ${platform} responses yet`}
              </div>
              <div style={{ fontSize: '13px', color: '#64748b', lineHeight: 1.6 }}>
                {error
                  ? 'The latest refresh did not complete. You can retry without leaving the call workspace.'
                  : 'New replies will appear here automatically as the candidate responds.'}
              </div>
              {error && (
                <button
                  onClick={() => loadMessages({ showLoader: true })}
                  style={{
                    padding: '10px 14px',
                    borderRadius: '10px',
                    border: '1px solid #e2e8f0',
                    background: '#fff',
                    color: '#0f172a',
                    fontWeight: 700,
                    cursor: 'pointer'
                  }}
                >
                  Retry
                </button>
              )}
            </div>
          </div>
        ) : (
          messages.map((msg, idx) => {
            const isCandidate = msg.type === 'REPLY' || msg.is_reply || msg.type === 'INBOX' || msg.direction === 'inbound';
            const senderName = isCandidate ? candidateName?.split(' ')?.[0] || 'Candidate' : 'You';
            const time = msg.time || msg.created_at || msg.timestamp;
            const body = msg.email_body || msg.message || msg.text || '';
            const formattedTime = time ? new Date(time) : null;
            const readableTime = formattedTime && !isNaN(formattedTime.getTime())
              ? `${formattedTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} ${formattedTime.toLocaleDateString()}`
              : time;

            return (
              <div key={`${platform}-message-${idx}`} style={{ display: 'flex', flexDirection: 'column', alignItems: isCandidate ? 'flex-start' : 'flex-end', animation: 'msgFadeIn 0.24s ease' }}>
                <div style={{ fontSize: '11px', color: '#94a3b8', marginBottom: '4px', padding: '0 4px', fontWeight: 600 }}>
                  {senderName}{readableTime ? ` • ${readableTime}` : ''}
                </div>
                <div style={{
                  maxWidth: '80%',
                  padding: '10px 15px',
                  borderRadius: isCandidate ? '6px 18px 18px 18px' : '18px 6px 18px 18px',
                  background: isCandidate ? '#fff' : accent,
                  color: isCandidate ? '#0f172a' : '#fff',
                  fontSize: '14px',
                  lineHeight: '1.5',
                  boxShadow: isCandidate ? '0 1px 4px rgba(0,0,0,0.07)' : '0 10px 18px -14px rgba(15,23,42,0.5)',
                  border: isCandidate ? '1px solid #e2e8f0' : 'none',
                  overflowWrap: 'break-word'
                }}>
                  <div dangerouslySetInnerHTML={{ __html: body }} />
                </div>
              </div>
            );
          })
        )}
        {refreshing && messages.length > 0 && (
          <div style={{
            position: 'sticky',
            top: 0,
            alignSelf: 'center',
            padding: '6px 12px',
            borderRadius: '999px',
            background: '#fff',
            border: '1px solid #e2e8f0',
            color: '#64748b',
            fontSize: '11px',
            fontWeight: 700,
            boxShadow: '0 10px 20px -14px rgba(15,23,42,0.35)'
          }}>
            Syncing latest replies...
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
    </div>
  );
}

function CandidateActivityPanel({ candidateId, candidateName }) {
  const fetchCandidateActivity = useAppStore(state => state.fetchCandidateActivity);
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
          items.map(item => (
            <div key={item.id} style={{ position: 'relative', padding: '18px', borderRadius: '20px', background: '#fff', border: '1px solid #e2e8f0', boxShadow: '0 1px 3px rgba(15, 23, 42, 0.04)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: '16px', marginBottom: '14px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <div style={{ width: '36px', height: '36px', borderRadius: '12px', background: '#f8fafc', color: '#475569', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <PhoneIncoming size={18} />
                  </div>
                  <div>
                    <div style={{ fontSize: '15px', fontWeight: 800, color: '#0f172a' }}>Completed Call With {candidateName || 'Candidate'}</div>
                    <div style={{ fontSize: '12px', color: '#64748b' }}>
                      {item.status || 'completed'}{item.outcome ? ` • ${item.outcome}` : ''}
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
              ) : null}

              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px 16px', marginBottom: '14px', fontSize: '13px', color: '#475569' }}>
                <div><strong>From:</strong> {item.from_number || 'N/A'}</div>
                <div><strong>To:</strong> {item.to_number || 'N/A'}</div>
                <div><strong>Duration:</strong> {item.duration_seconds ? `${Math.floor(item.duration_seconds / 60)}m ${item.duration_seconds % 60}s` : '0s'}</div>
              </div>

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
                    style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', fontSize: '12px', fontWeight: 700, color: '#2563eb', textDecoration: 'none' }}
                  >
                    <ExternalLink size={14} />
                    Open Source
                  </a>
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default function Calls() {
  const {
    calls,
    fetchCalls,
    callLists,
    fetchCallLists,
    callListsLastFetchedAt,
    callsLastQueryKey,
    callStats,
    fetchCallStats,
    callStatsLastFetchedAt,
    callsLastFetchedAt,
    updateCall,
    deleteCall,
    deleteCallList,
    createCallList,
    clearCallsState,
    syncCallRecording,
    sidebarWidth,
  } = useAppStore(useShallow((state) => ({
    calls: state.calls,
    fetchCalls: state.fetchCalls,
    callLists: state.callLists,
    fetchCallLists: state.fetchCallLists,
    callListsLastFetchedAt: state.callListsLastFetchedAt,
    callsLastQueryKey: state.callsLastQueryKey,
    callStats: state.callStats,
    fetchCallStats: state.fetchCallStats,
    callStatsLastFetchedAt: state.callStatsLastFetchedAt,
    callsLastFetchedAt: state.callsLastFetchedAt,
    updateCall: state.updateCall,
    deleteCall: state.deleteCall,
    deleteCallList: state.deleteCallList,
    createCallList: state.createCallList,
    clearCallsState: state.clearCallsState,
    syncCallRecording: state.syncCallRecording,
    sidebarWidth: state.sidebarWidth,
  })));

  const [statusOverrides, setStatusOverrides] = useState({});

  const [searchParams, setSearchParams] = useSearchParams();
  // Deep-link restore: remember the list_id from the URL until callLists arrive.
  const pendingListIdRef = useRef(Number(searchParams.get('list_id')) || null);

  const [activeTab, setActiveTab] = useState(() => {
    const tab = searchParams.get('tab');
    return TABS.some(t => t.id === tab) ? tab : 'today';
  });
  const [loading, setLoading] = useState(false);
  const [isRevalidating, setIsRevalidating] = useState(false);
  const [selectedList, setSelectedList] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [callingCandidate, setCallingCandidate] = useState(null); // The call object
  const [expandedCallId, setExpandedCallId] = useState(null);
  const [syncingCallId, setSyncingCallId] = useState(null);
  const [deletingCallIds, setDeletingCallIds] = useState(() => new Set());
  const [deletingListIds, setDeletingListIds] = useState(() => new Set());
  
  // List Creation State
  const [isCreatingList, setIsCreatingList] = useState(false);
  const [newListName, setNewListName] = useState('');
  const [isSubmittingList, setIsSubmittingList] = useState(false);
  const [fetchNotice, setFetchNotice] = useState('');
  const retryTimerRef = useRef(null);
  const callsRef = useRef(calls);
  const callListsRef = useRef(callLists);

  useEffect(() => {
    callsRef.current = calls;
  }, [calls]);

  useEffect(() => {
    callListsRef.current = callLists;
  }, [callLists]);

  useEffect(() => () => {
    if (retryTimerRef.current) {
      clearTimeout(retryTimerRef.current);
    }
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined' || window.scrollX === 0) return;
    window.scrollTo({ left: 0, top: window.scrollY });
  }, [sidebarWidth, activeTab, selectedList]);

  // Restore the URL's list_id once the call lists have loaded.
  useEffect(() => {
    const pendingListId = pendingListIdRef.current;
    if (!pendingListId) return;
    if (!(callLists || []).length && !callListsLastFetchedAt) return; // wait for the first fetch
    const match = (callLists || []).find(list => Number(list.id) === pendingListId);
    pendingListIdRef.current = null;
    if (match) {
      clearCallsState();
      setActiveTab('lists');
      setSelectedList(match);
    }
  }, [callLists, callListsLastFetchedAt, clearCallsState]);

  // Keep tab + selected list in the URL so a refresh restores this exact view.
  useEffect(() => {
    if (pendingListIdRef.current) return; // still restoring from the URL
    const next = {};
    if (selectedList?.id) {
      next.tab = 'lists';
      next.list_id = String(selectedList.id);
    } else if (activeTab !== 'today') {
      next.tab = activeTab;
    }
    setSearchParams(next, { replace: true });
  }, [activeTab, selectedList, setSearchParams]);

  const fetchData = useCallback(async () => {
    if (retryTimerRef.current) {
      clearTimeout(retryTimerRef.current);
      retryTimerRef.current = null;
    }

    const hasCache = activeTab === 'lists' && !selectedList
      ? callListsRef.current.length > 0
      : callsRef.current && callsRef.current.length > 0;
    if (!hasCache) setLoading(true);
    else setIsRevalidating(true);

    const statsPromise = fetchCallStats({ force: true }).then(res => {
      if (!res?.success) {
        console.error('Failed to refresh call stats:', res?.error);
      }
      return res;
    });

    try {
      if (activeTab === 'lists' && !selectedList) {
        const [listsRes] = await Promise.all([
          fetchCallLists({ force: true }),
          statsPromise,
        ]);
        if (!listsRes?.success) {
          setFetchNotice(hasCache ? 'Server is slow. Showing the last available call lists.' : 'Server is slow. Trying again shortly.');
          retryTimerRef.current = setTimeout(() => {
            fetchData();
          }, 2000);
          return;
        }
      } else {
        const params = {};
        if (activeTab === 'today') params.due_filter = 'today';
        else if (activeTab === 'upcoming') params.due_filter = 'upcoming';
        else if (activeTab === 'completed') params.status = 'completed';
        
        if (selectedList) {
          params.list_id = selectedList.id;
          params.status = 'pending'; // Default to pending in list view
        } else if (activeTab === 'today' || activeTab === 'upcoming') {
          params.status = 'pending';
        }

        const [callsRes] = await Promise.all([
          fetchCalls(params),
          statsPromise,
        ]);
        if (!callsRes?.success) {
          setFetchNotice(hasCache ? 'Server is slow. Showing the last available call tasks.' : 'Server is slow. Trying again shortly.');
          retryTimerRef.current = setTimeout(() => {
            fetchData();
          }, 2000);
          return;
        }
      }
      setFetchNotice('');
    } catch (e) {
      console.error('fetchData detailed error:', e);
      setFetchNotice(hasCache ? 'Server is slow. Showing the last available call data.' : 'Server is slow. Trying again shortly.');
      retryTimerRef.current = setTimeout(() => {
        fetchData();
      }, 2000);
    } finally {
      setLoading(false);
      setIsRevalidating(false);
    }
  }, [activeTab, selectedList, fetchCalls, fetchCallLists, fetchCallStats]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Background post-call analysis poller: recently completed calls that are
  // still missing recording/transcript/summary keep syncing even after the
  // review modal closes, so their rows update in place when analysis lands.
  // Requests run sequentially to keep backend load at one sync at a time.
  useEffect(() => {
    const pendingIds = (calls || []).filter(isPendingAnalysis).map(c => c.id);
    if (!pendingIds.length) return undefined;

    let cancelled = false;
    let syncing = false;
    const interval = setInterval(async () => {
      if (cancelled || syncing || !isDocumentVisible()) return;
      syncing = true;
      try {
        for (const id of pendingIds) {
          if (cancelled) break;
          await syncCallRecording(id);
        }
      } finally {
        syncing = false;
      }
    }, BACKGROUND_ANALYSIS_POLL_MS);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [calls, syncCallRecording]);

  useEffect(() => {
    if (!selectedList || !callListsLastFetchedAt) return;
    const selectedListStillExists = (callLists || []).some(list => Number(list.id) === Number(selectedList.id));
    if (!selectedListStillExists) {
      clearCallsState();
      setSelectedList(null);
    }
  }, [selectedList, callLists, callListsLastFetchedAt, clearCallsState]);

  const stats = [
    { label: 'DUE TODAY', value: callStats.due_today, icon: Phone, color: '#334155', bg: '#f8fafc' },
    { label: 'UPCOMING', value: callStats.upcoming, icon: Clock, color: '#8b6b44', bg: '#fcf8f2' },
    { label: 'COMPLETED', value: callStats.completed, icon: CheckCircle2, color: '#166534', bg: '#f3faf5' },
    { label: 'CALL LISTS', value: callStats.active_lists, icon: List, color: '#475569', bg: '#f8fafc' },
  ];

  const handleDial = (call) => {
    setCallingCandidate(call);
  };

  const handleDeleteCall = async (callId) => {
    if (deletingCallIds.has(callId)) return;
    if (!window.confirm('Remove this candidate from the call list?')) return;
    setDeletingCallIds(prev => new Set(prev).add(callId));
    try {
      const res = await deleteCall(callId);
      if (res.success) toast.success('Removed from list');
      else toast.error(res.error || 'Failed to remove');
    } finally {
      setDeletingCallIds(prev => {
        const next = new Set(prev);
        next.delete(callId);
        return next;
      });
    }
  };

  const handleDeleteList = async (listId, name) => {
    if (deletingListIds.has(listId)) return;
    if (!window.confirm(`Delete the list "${name}"? This will remove all associated tasks.`)) return;
    setDeletingListIds(prev => new Set(prev).add(listId));
    try {
      const res = await deleteCallList(listId);
      if (res.success) {
        if (Number(selectedList?.id) === Number(listId)) {
          clearCallsState();
          setSelectedList(null);
        }
        toast.success('List deleted');
      }
      else toast.error(res.error || 'Failed to delete list');
    } finally {
      setDeletingListIds(prev => {
        const next = new Set(prev);
        next.delete(listId);
        return next;
      });
    }
  };

  const handleCreateList = async () => {
    if (isSubmittingList) return;
    if (!newListName.trim()) {
      setIsCreatingList(false);
      return;
    }
    setIsSubmittingList(true);
    const res = await createCallList(newListName.trim());
    setIsSubmittingList(false);
    
    if (res.success) {
      toast.success('List created');
      setNewListName('');
      setIsCreatingList(false);
    } else {
      toast.error(res.error || 'Failed to create list');
    }
  };

  const handleSyncRecording = async (callId) => {
    setSyncingCallId(callId);
    const res = await syncCallRecording(callId);
    setSyncingCallId(null);

    if (!res.success) {
      toast.error(res.error || 'Failed to sync recording');
      return;
    }

    if (res.data?.recording_url) {
      toast.success('Recording synced');
      return;
    }

    toast('Recording is not ready in Plivo yet');
  };

  // Build the key through the SAME canonicalizer the store uses so the view
  // never mismatches its cache entry (param order / number-vs-string list_id).
  const currentCallsQueryKey = selectedList
    ? canonicalCallsQuery({ list_id: selectedList.id, status: 'pending' })
    : activeTab === 'today'
      ? canonicalCallsQuery({ due_filter: 'today', status: 'pending' })
      : activeTab === 'upcoming'
        ? canonicalCallsQuery({ due_filter: 'upcoming', status: 'pending' })
        : activeTab === 'completed'
          ? canonicalCallsQuery({ status: 'completed' })
          : '';
  const callsForCurrentQuery = callsLastQueryKey === currentCallsQueryKey ? (calls || []) : [];
  const filteredCalls = callsForCurrentQuery.filter(c =>
    (c.candidate_name || '').toLowerCase().includes(searchQuery.toLowerCase()) ||
    (c.candidate_title || '').toLowerCase().includes(searchQuery.toLowerCase())
  );
  // Show skeleton only when actively loading and no data exists yet for this view
  const isWaitingForCurrentQuery = callsLastQueryKey !== currentCallsQueryKey;
  const hasCurrentCallsData = callsLastQueryKey === currentCallsQueryKey && Boolean(callsLastFetchedAt);
  const showCallsLoading = (
    activeTab !== 'lists' &&
    (loading || isWaitingForCurrentQuery) &&
    !hasCurrentCallsData
  );
  const showListsLoading = (activeTab === 'lists' && !selectedList && loading && !callLists.length);

  return (
    <VoIPProvider>
    <div className="calls-page" style={{ padding: '24px 0 12px', background: 'transparent', minHeight: '100vh', fontFamily: '"Inter", sans-serif', width: '100%', overflowX: 'hidden' }}>
      <header className="calls-hero" style={{ ...CALL_PANEL_STYLE, marginBottom: '28px', padding: '24px 28px', borderRadius: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px 24px' }}>
        <div>
          <div style={{ fontSize: '11px', fontWeight: 700, color: '#8b6b44', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '6px' }}>
            Call operations
          </div>
          <h1 style={{ fontSize: '28px', fontWeight: 800, color: '#0f172a', marginBottom: '8px' }}>Calls Workspace</h1>
          <p style={{ color: '#64748b', fontSize: '15px' }}>Track your calling progress and call lists</p>
          {fetchNotice && (
            <div style={{ marginTop: '12px', padding: '10px 14px', borderRadius: '12px', background: '#fcf8f2', color: '#8b6b44', fontSize: '13px', fontWeight: 600, border: '1px solid rgba(194,124,63,0.2)' }}>
              {fetchNotice}
            </div>
          )}
        </div>
        {isRevalidating && (
          <div style={{ padding: '8px 16px', borderRadius: '12px', background: '#f8fafc', color: '#475569', fontSize: '12px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px', border: '1px solid rgba(203,213,225,0.9)' }}>
            <RefreshCw size={14} className="revalidating" /> Updating...
          </div>
        )}
      </header>

      {/* Stats Cards */}
      <div className="calls-stats-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: '20px', marginBottom: '40px' }}>
        {stats.map((stat, i) => (
          <div className="calls-stat-card" key={i} style={{ ...CALL_PANEL_STYLE, padding: '24px', borderRadius: '20px', minWidth: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
              <div style={{ width: '40px', height: '40px', borderRadius: '12px', background: stat.bg, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <stat.icon size={20} color={stat.color} />
              </div>
              <span style={{ fontSize: '12px', fontWeight: 700, color: '#94a3b8', letterSpacing: '0.05em' }}>{stat.label}</span>
            </div>
            <div style={{ fontSize: '32px', fontWeight: 800, color: '#0f172a' }}>
              {!callStatsLastFetchedAt ? <Loader2 size={24} className="animate-spin" color="#cbd5e1" /> : stat.value}
            </div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <div className="calls-tabs" style={{ display: 'flex', flexWrap: 'wrap', borderBottom: '1px solid #e2e8f0', marginBottom: '32px', columnGap: '24px', rowGap: '10px' }}>
      {TABS.map(tab => (
          <button
            key={tab.id}
            onClick={() => { clearCallsState(); setActiveTab(tab.id); setSelectedList(null); }}
            style={{
              padding: '12px 4px', background: 'none', border: 'none', borderBottom: activeTab === tab.id ? '2px solid #111827' : '2px solid transparent',
              color: activeTab === tab.id ? '#111827' : '#64748b', fontSize: '14px', fontWeight: 600,
              cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '-1px',
              transition: 'all 0.2s'
            }}
          >
            <tab.icon size={18} />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Content Area */}
      <div className="calls-content-panel" style={{ ...CALL_PANEL_STYLE, borderRadius: '24px', overflow: 'hidden' }}>
        {activeTab === 'lists' && !selectedList ? (
          <div className="calls-lists-section" style={{ padding: '24px' }}>
            <div className="calls-lists-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '20px' }}>
              {showListsLoading && Array.from({ length: 4 }).map((_, idx) => (
                <div
                  key={`list-skeleton-${idx}`}
                  style={{ padding: '24px', borderRadius: '20px', border: '1px solid #e2e8f0', background: '#fff' }}
                >
                  <div style={{ width: 40, height: 40, borderRadius: 12, background: '#e2e8f0', marginBottom: 16 }} />
                  <div style={{ width: '55%', height: 16, borderRadius: 8, background: '#e2e8f0', marginBottom: 10 }} />
                  <div style={{ width: '35%', height: 12, borderRadius: 8, background: '#f1f5f9' }} />
                </div>
              ))}
              
              {/* Create List Card */}
              <div
                className="calls-list-card calls-list-create-card"
                style={{ 
                  padding: '24px', borderRadius: '20px', 
                  border: isCreatingList ? '1px solid #111827' : '1px dashed #cbd5e1',
                  background: isCreatingList ? '#fff' : '#f8fafc',
                  cursor: isCreatingList ? 'default' : 'pointer',
                  transition: 'all 0.2s ease',
                  display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center',
                  minHeight: '160px',
                  boxShadow: isCreatingList ? '0 10px 15px -3px rgba(0, 0, 0, 0.05)' : 'none'
                }}
                onClick={() => !isCreatingList && setIsCreatingList(true)}
                onMouseEnter={e => !isCreatingList && (e.currentTarget.style.borderColor = '#94a3b8')}
                onMouseLeave={e => !isCreatingList && (e.currentTarget.style.borderColor = '#cbd5e1')}
              >
                {!isCreatingList ? (
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px' }}>
                    <div style={{ width: '40px', height: '40px', borderRadius: '12px', background: '#fff', border: '1px solid #e2e8f0', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <List size={20} color="#64748b" />
                    </div>
                    <span style={{ fontSize: '15px', fontWeight: 600, color: '#64748b' }}>Create New List</span>
                  </div>
                ) : (
                  <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                    <input
                      autoFocus
                      type="text"
                      placeholder="e.g. Frontend Q1 Hires"
                      value={newListName}
                      onChange={e => setNewListName(e.target.value)}
                      onKeyDown={e => e.key === 'Enter' && !isSubmittingList && handleCreateList()}
                      disabled={isSubmittingList}
                      style={{
                        width: '100%', padding: '10px 14px', borderRadius: '10px',
                        border: '1px solid #e2e8f0', fontSize: '14px', outline: 'none',
                        boxSizing: 'border-box'
                      }}
                    />
                    <div style={{ display: 'flex', gap: '8px' }}>
                      <button 
                        onClick={(e) => { e.stopPropagation(); handleCreateList(); }}
                        disabled={isSubmittingList}
                        style={{ ...CALL_PRIMARY_BUTTON, flex: 1, padding: '8px', fontSize: '13px', cursor: isSubmittingList ? 'wait' : 'pointer' }}
                      >
                        {isSubmittingList ? 'Saving...' : 'Save List'}
                      </button>
                      <button 
                        onClick={(e) => { e.stopPropagation(); setIsCreatingList(false); setNewListName(''); }}
                        disabled={isSubmittingList}
                        style={{ ...CALL_SECONDARY_BUTTON, padding: '8px 12px', fontSize: '13px', cursor: isSubmittingList ? 'wait' : 'pointer' }}
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                )}
              </div>

              {callLists.map(list => {
                const isDeletingList = deletingListIds.has(list.id);
                const isPendingList = Boolean(list.is_pending);
                const isListDisabled = isDeletingList || isPendingList;

                return (
                <div
                  className="calls-list-card"
                  key={list.id} 
                  onClick={() => {
                    if (isListDisabled) return;
                    clearCallsState();
                    setSelectedList(list);
                  }}
                  style={{ 
                    padding: '24px', borderRadius: '20px', border: '1px solid #e2e8f0', cursor: isListDisabled ? 'wait' : 'pointer',
                    transition: 'all 0.2s',
                    opacity: isDeletingList ? 0.55 : 1
                  }}
                  onMouseEnter={e => {
                    if (isListDisabled) return;
                    e.currentTarget.style.borderColor = '#111827';
                    e.currentTarget.style.transform = 'translateY(-2px)';
                  }}
                  onMouseLeave={e => {
                    e.currentTarget.style.borderColor = '#e2e8f0';
                    e.currentTarget.style.transform = 'none';
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
                    <div style={{ width: '40px', height: '40px', borderRadius: '12px', background: '#f8fafc', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      {isDeletingList || isPendingList ? (
                        <Loader2 size={20} color="#475569" className="animate-spin" />
                      ) : (
                        <List size={20} color="#475569" />
                      )}
                    </div>
                    <button 
                      onClick={(e) => {
                        e.stopPropagation();
                        if (isListDisabled) return;
                        handleDeleteList(list.id, list.name);
                      }}
                      disabled={isListDisabled}
                      style={{ padding: '6px', background: 'none', border: 'none', color: '#94a3b8', cursor: isListDisabled ? 'wait' : 'pointer', borderRadius: '8px', opacity: isListDisabled ? 0.6 : 1 }}
                      onMouseEnter={e => { if (!isListDisabled) e.currentTarget.style.color = '#ef4444'; }}
                      onMouseLeave={e => { if (!isListDisabled) e.currentTarget.style.color = '#94a3b8'; }}
                    >
                      {isDeletingList ? <Loader2 size={16} className="animate-spin" /> : <Trash2 size={16} />}
                    </button>
                  </div>
                  <h3 style={{ fontSize: '16px', fontWeight: 700, color: '#0f172a', marginBottom: '4px' }}>{list.name}</h3>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <User size={12} color="#94a3b8" />
                    <span style={{ fontSize: '13px', color: '#64748b' }}>
                      {isDeletingList ? 'Deleting...' : isPendingList ? 'Saving...' : `${list.candidate_count} Pending`}
                    </span>
                  </div>
                </div>
                );
              })}
              {(callLists || []).length === 0 && (
                <div style={{ gridColumn: '1/-1', padding: '60px', textAlign: 'center', color: '#94a3b8' }}>No call lists found yet.</div>
              )}
            </div>
          </div>
        ) : (
          <div className="calls-workspace" style={{ minHeight: '400px' }}>
            <div className="calls-workspace-toolbar" style={{ padding: '20px 24px', borderBottom: '1px solid #f1f5f9', display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
              {selectedList && (
                <button 
                  onClick={() => { clearCallsState(); setSelectedList(null); }}
                  style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#64748b', display: 'flex', alignItems: 'center', gap: '4px', padding: '4px' }}
                >
                  <ChevronLeft size={20} />
                </button>
              )}
              <h2 style={{ fontSize: '16px', fontWeight: 700, color: '#0f172a' }}>
                {selectedList ? selectedList.name : activeTab === 'today' ? 'Due Today' : activeTab === 'upcoming' ? 'Upcoming' : 'Completed'}
              </h2>
              {selectedList && (
                <span style={{ fontSize: '12px', background: '#f8fafc', color: '#475569', padding: '2px 8px', borderRadius: '20px', fontWeight: 600, border: '1px solid rgba(203,213,225,0.9)' }}>
                  {(filteredCalls || []).length} Contacts
                </span>
              )}
              
              <div style={{ position: 'relative', marginLeft: 'auto', width: 'min(240px, 100%)', minWidth: '180px', flex: '1 1 220px' }}>
                <Search size={14} color="#94a3b8" style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)' }} />
                <input 
                  type="text" 
                  placeholder="Search candidates..."
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                  style={{ 
                    width: '100%', padding: '8px 12px 8px 32px', borderRadius: '10px', 
                    border: '1px solid #e2e8f0', fontSize: '13px', outline: 'none'
                  }}
                />
              </div>
            </div>

            <div className="calls-table-wrap" style={{ padding: '0 24px 24px', overflowX: 'auto' }}>
              <table className="calls-table" style={{ width: '100%', minWidth: 880, borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={CALLS_TABLE_TH_STYLE}>Prospect</th>
                    <th style={CALLS_TABLE_TH_STYLE}>Purpose</th>
                    <th style={CALLS_TABLE_TH_STYLE}>Status</th>
                    <th style={CALLS_TABLE_TH_STYLE}>{activeTab === 'completed' ? 'Completed' : 'Scheduled'}</th>
                    <th style={CALLS_TABLE_TH_STYLE}>Outcome</th>
                    <th style={{ ...CALLS_TABLE_TH_STYLE, textAlign: 'right' }}>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {showCallsLoading && Array.from({ length: 5 }).map((_, idx) => (
                    <tr key={`call-skeleton-${idx}`}>
                      <td colSpan={CALLS_TABLE_COL_COUNT} style={{ padding: '16px', borderBottom: '1px solid #f1f5f9' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                          <div style={{ width: 36, height: 36, borderRadius: '50%', background: '#e2e8f0', flexShrink: 0 }} />
                          <div style={{ flex: 1 }}>
                            <div style={{ width: '28%', height: 14, borderRadius: 8, background: '#e2e8f0', marginBottom: 8 }} />
                            <div style={{ width: '44%', height: 12, borderRadius: 8, background: '#f1f5f9' }} />
                          </div>
                        </div>
                      </td>
                    </tr>
                  ))}
                  {!showCallsLoading && (filteredCalls || []).map(call => {
                    const isDeletingCall = deletingCallIds.has(call.id);
                    const isExpanded = expandedCallId === call.id;
                    const dialDisabled = call.status === 'completed' || isDeletingCall || Boolean(call.candidate_phone_wrong);
                    return (
                      <React.Fragment key={call.id}>
                        <tr
                          className="calls-table-row"
                          style={{ opacity: isDeletingCall ? 0.55 : 1, transition: 'background 0.15s' }}
                          onMouseEnter={e => { e.currentTarget.style.background = '#fff7ed'; }}
                          onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; }}
                        >
                          <td style={{ padding: '14px 16px', borderBottom: '1px solid #f1f5f9', verticalAlign: 'top' }}>
                            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
                              <div style={{ width: '34px', height: '34px', borderRadius: '50%', background: '#f1f5f9', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '12px', fontWeight: 700, color: '#64748b', flexShrink: 0 }}>
                                {call.candidate_name?.split(' ').map(n => n[0]).join('') || '?'}
                              </div>
                              <div style={{ minWidth: 0 }}>
                                <div style={{ fontSize: '14px', fontWeight: 700, color: '#0f172a', whiteSpace: 'nowrap' }}>{call.candidate_name || 'Anonymous'}</div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap', fontSize: '12px', color: '#94a3b8', marginTop: '3px' }}>
                                  <span style={{ display: 'flex', alignItems: 'center', gap: '4px', whiteSpace: 'nowrap' }}>
                                    <Phone size={11} /> {call.candidate_phone || 'N/A'}
                                  </span>
                                  {call.candidate_linkedin && (
                                    <a
                                      href={call.candidate_linkedin}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      title="Open LinkedIn profile"
                                      onClick={e => e.stopPropagation()}
                                      style={{ display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--accent-primary)', fontWeight: 600, textDecoration: 'none', whiteSpace: 'nowrap' }}
                                    >
                                      <Linkedin size={11} /> LinkedIn
                                    </a>
                                  )}
                                </div>
                                {call.candidate_phone_wrong && (
                                  <span style={{ display: 'inline-block', marginTop: '4px', padding: '2px 8px', borderRadius: '999px', background: '#fef2f2', color: '#b91c1c', fontWeight: 700, fontSize: '10px', border: '1px solid #fecaca' }}>
                                    Wrong number — calling paused
                                  </span>
                                )}
                              </div>
                            </div>
                          </td>
                          <td style={{ padding: '14px 16px', borderBottom: '1px solid #f1f5f9', verticalAlign: 'top', fontSize: '13px', color: '#475569', maxWidth: 200 }}>
                            {call.task_title || call.candidate_title || '—'}
                          </td>
                          <td style={{ padding: '14px 16px', borderBottom: '1px solid #f1f5f9', verticalAlign: 'top' }}>
                            <StatusDropdown
                              status={statusOverrides[call.candidate_id] ?? call.candidate_status}
                              candidateId={call.candidate_id}
                              optimistic
                              onUpdate={(id, newStatus) => setStatusOverrides(prev => ({ ...prev, [id]: newStatus }))}
                            />
                          </td>
                          <td style={{ padding: '14px 16px', borderBottom: '1px solid #f1f5f9', verticalAlign: 'top', fontSize: '12px', color: '#94a3b8', whiteSpace: 'nowrap' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                              <Calendar size={11} />
                              {call.status === 'completed'
                                ? (call.completed_at ? new Date(call.completed_at).toLocaleDateString() : 'Unknown')
                                : `${formatLocalDate(call.due_date)}${call.due_time ? ` ${formatDueTime(call.due_time)}` : ''}`
                              }
                            </div>
                          </td>
                          <td style={{ padding: '14px 16px', borderBottom: '1px solid #f1f5f9', verticalAlign: 'top', fontSize: '12px', whiteSpace: 'nowrap' }}>
                            {call.status === 'completed' ? (
                              <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
                                <span style={{ color: call.outcome === 'Unreachable' ? '#b45309' : '#059669', fontWeight: 600, textTransform: 'capitalize' }}>
                                  {call.outcome || 'No Outcome'}
                                </span>
                                <span style={{ color: '#2563eb', fontWeight: 600 }}>
                                  {call.duration ? `${Math.floor(call.duration / 60)}m ${call.duration % 60}s` : '0s'}
                                </span>
                                <SentimentBadge sentiment={call.sentiment} reason={call.sentiment_reason} size={10} />
                                {isPendingAnalysis(call) && (
                                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: '5px', padding: '2px 8px', borderRadius: '999px', background: '#f5f3ff', color: '#8b5cf6', fontWeight: 700, fontSize: '10px', border: '1px solid #ddd6fe', width: 'fit-content' }}>
                                    <RefreshCw size={9} className="animate-spin" /> Analyzing…
                                  </span>
                                )}
                              </div>
                            ) : (
                              <span style={{ color: '#cbd5e1' }}>—</span>
                            )}
                          </td>
                          <td style={{ padding: '14px 16px', borderBottom: '1px solid #f1f5f9', verticalAlign: 'top' }}>
                            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                              {call.status === 'completed' && (
                                <button
                                  onClick={() => setExpandedCallId(isExpanded ? null : call.id)}
                                  style={{
                                    padding: '7px 12px', borderRadius: '10px', fontSize: '12px', fontWeight: 700,
                                    background: '#fff', color: '#334155', border: '1px solid rgba(203,213,225,0.9)', cursor: 'pointer',
                                    display: 'flex', alignItems: 'center', gap: '6px', whiteSpace: 'nowrap'
                                  }}
                                >
                                  {isExpanded ? 'Hide Details' : 'View Insights'}
                                </button>
                              )}
                              <button
                                onClick={() => handleDeleteCall(call.id)}
                                disabled={isDeletingCall}
                                style={{ padding: '7px', background: 'none', border: 'none', color: '#94a3b8', cursor: isDeletingCall ? 'wait' : 'pointer', borderRadius: '8px', opacity: isDeletingCall ? 0.6 : 1 }}
                                onMouseEnter={e => { if (!isDeletingCall) e.currentTarget.style.color = '#ef4444'; }}
                                onMouseLeave={e => { if (!isDeletingCall) e.currentTarget.style.color = '#94a3b8'; }}
                                title="Remove from list"
                              >
                                {isDeletingCall ? <Loader2 size={17} className="animate-spin" /> : <Trash2 size={17} />}
                              </button>
                              <button
                                onClick={() => handleDial(call)}
                                disabled={dialDisabled}
                                title={call.candidate_phone_wrong ? 'Number tagged as wrong — update the candidate’s phone to resume calling' : undefined}
                                style={{
                                  padding: '7px 14px', borderRadius: '10px', fontSize: '13px', fontWeight: 700,
                                  background: dialDisabled ? '#f1f5f9' : 'var(--accent-primary)',
                                  color: dialDisabled ? '#94a3b8' : '#fff',
                                  border: dialDisabled ? '1px solid rgba(203,213,225,0.9)' : '1px solid var(--accent-primary)',
                                  cursor: dialDisabled ? 'not-allowed' : 'pointer',
                                  display: 'flex', alignItems: 'center', gap: '8px'
                                }}
                              >
                                <PhoneCall size={15} />
                              </button>
                            </div>
                          </td>
                        </tr>
                        {isExpanded && (
                          <tr>
                            <td colSpan={CALLS_TABLE_COL_COUNT} style={{ padding: 0, borderBottom: '1px solid #f1f5f9' }}>
                              <div className="calls-expanded-details" style={{
                                padding: '20px', background: '#f8fafc', margin: '0 16px 16px',
                                borderRadius: '16px', border: '1.5px solid #e2e8f0', display: 'flex', flexDirection: 'column', gap: '20px'
                              }}>
                                <div style={{ background: '#fff', padding: '16px', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
                                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', alignItems: 'center', marginBottom: '12px' }}>
                                    <h4 style={{ fontSize: '12px', fontWeight: 800, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Call Recording</h4>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                                      {call.recording_url && (
                                        <a
                                          href={call.recording_url}
                                          target="_blank"
                                          rel="noreferrer"
                                          style={{ fontSize: '12px', fontWeight: 700, color: 'var(--accent-primary)', textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '6px' }}
                                        >
                                          <ExternalLink size={14} />
                                          Open Recording
                                        </a>
                                      )}
                                      {!call.recording_url && (
                                        <button
                                          onClick={() => handleSyncRecording(call.id)}
                                          disabled={syncingCallId === call.id}
                                          style={{
                                            padding: '8px 12px',
                                            borderRadius: '10px',
                                            border: '1px solid rgba(203,213,225,0.9)',
                                            background: '#fff',
                                            color: '#334155',
                                            fontSize: '12px',
                                            fontWeight: 700,
                                            cursor: syncingCallId === call.id ? 'wait' : 'pointer',
                                            display: 'inline-flex',
                                            alignItems: 'center',
                                            gap: '6px'
                                          }}
                                        >
                                          <RefreshCw size={14} style={{ animation: syncingCallId === call.id ? 'spin 1s linear infinite' : 'none' }} />
                                          {syncingCallId === call.id ? 'Syncing...' : 'Sync Recording'}
                                        </button>
                                      )}
                                    </div>
                                  </div>
                                  {call.recording_url ? (
                                    <audio controls src={call.recording_url} style={{ width: '100%' }} />
                                  ) : (
                                    <div style={{ color: '#64748b', fontSize: '13px' }}>
                                      Recording not available for this call yet.
                                    </div>
                                  )}
                                  {(call.recording_source || call.recording_synced_at) && (
                                    <div style={{ marginTop: '10px', fontSize: '12px', color: '#94a3b8' }}>
                                      {call.recording_source ? `Source: ${call.recording_source}` : ''}
                                      {call.recording_source && call.recording_synced_at ? ' • ' : ''}
                                      {call.recording_synced_at ? `Synced ${formatDateTime(call.recording_synced_at)}` : ''}
                                    </div>
                                  )}
                                </div>

                                <div className="calls-insights-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                                  <div style={{ background: '#fff', padding: '20px', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px', marginBottom: '12px' }}>
                                      <h4 style={{ fontSize: '12px', fontWeight: 800, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em', margin: 0 }}>AI Summary</h4>
                                      <SentimentBadge sentiment={call.sentiment} reason={call.sentiment_reason} />
                                    </div>
                                    <div style={{ fontSize: '14px', color: '#334155', lineHeight: '1.6', whiteSpace: 'pre-wrap' }}>
                                      {call.summary || (call.recording_url ? 'Processing summary...' : 'No summary available.')}
                                    </div>
                                    {call.sentiment_reason && (
                                      <div style={{ marginTop: '10px', paddingTop: '10px', borderTop: '1px dashed #e2e8f0', fontSize: '12px', color: '#94a3b8', fontStyle: 'italic' }}>
                                        “{call.sentiment_reason}”
                                      </div>
                                    )}
                                  </div>
                                  <div style={{ background: '#fff', padding: '20px', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
                                    <h4 style={{ fontSize: '12px', fontWeight: 800, color: '#64748b', textTransform: 'uppercase', marginBottom: '12px', letterSpacing: '0.05em' }}>Full Transcript</h4>
                                    <div style={{
                                      fontSize: '13px', color: '#64748b', lineHeight: '1.6', height: '200px', overflowY: 'auto',
                                      paddingRight: '12px'
                                    }}>
                                      {call.transcript ? (
                                        <TranscriptView
                                          transcript={call.transcript}
                                          candidateName={call.candidate_name}
                                          recruiterName={recruiterDisplayName(call)}
                                        />
                                      ) : (
                                        <div style={{ whiteSpace: 'pre-wrap' }}>
                                          {call.recording_url ? 'Transcribing call...' : 'No transcript available.'}
                                        </div>
                                      )}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            </td>
                          </tr>
                        )}
                      </React.Fragment>
                    );
                  })}
                  {!showCallsLoading && (filteredCalls || []).length === 0 && !loading && (
                    <tr>
                      <td colSpan={CALLS_TABLE_COL_COUNT} style={{ padding: '60px', textAlign: 'center', color: '#94a3b8' }}>
                        No candidates matching your query.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {callingCandidate && (
        <CallingModal
          call={callingCandidate}
          onClose={() => setCallingCandidate(null)}
          onRefresh={fetchData}
        />
      )}
    </div>
    </VoIPProvider>
  );
}

function CallingModal({ call, onClose, onRefresh }) {
  const [activeTab, setActiveTab] = useState('calls');
  const [callState, setCallState] = useState('preparing_softphone');
  const [callWrapUpMeta, setCallWrapUpMeta] = useState(null);
  const [outcome, setOutcome] = useState('');
  // Follow-up slot — required when the outcome is "Connected - Follow-up".
  const [followupDueDate, setFollowupDueDate] = useState('');
  const [followupDueTime, setFollowupDueTime] = useState('');
  const [saving, setSaving] = useState(false);
  const [candidateStatus, setCandidateStatus] = useState(call.candidate_status || '');
  const [liveNotes, setLiveNotes] = useState(call.candidate_notes || '');
  const [notesLoading, setNotesLoading] = useState(Boolean(call.candidate_id));
  const originalNotesRef = useRef(call.candidate_notes || '');
  const [isAnswering, setIsAnswering] = useState(false);
  const [initiationError, setInitiationError] = useState('');
  const [initiationErrorCode, setInitiationErrorCode] = useState('');
  const [initiationActionLabel, setInitiationActionLabel] = useState('');
  const [initiationActionUrl, setInitiationActionUrl] = useState('');
  const [softphoneRecoveryAttempt, setSoftphoneRecoveryAttempt] = useState(0);
  const { updateCall, initiateCall, fetchCalls, syncCallRecording, updateCandidateNotes } = useAppStore(useShallow((state) => ({
    updateCall: state.updateCall,
    initiateCall: state.initiateCall,
    fetchCalls: state.fetchCalls,
    syncCallRecording: state.syncCallRecording,
    updateCandidateNotes: state.updateCandidateNotes,
  })));
  const {
    activeCall,
    answerCall,
    rejectCall,
    placeCall,
    ensureMicrophonePermission,
    waitForPlivoDial,
    voipStatus,
    voipError,
    voipErrorCode,
    voipActionLabel,
    voipActionUrl,
    voipMeta,
    voipConnectionEvent,
    voipCallEvent,
    agentEmail,
    endpointUsername,
    retryVoip,
    startDialTone,
    stopDialTone,
  } = useVoIP();
  const isInitiated = useRef(false);
  const lastHandledCallEventRef = useRef(0);
  const autoRetriedSoftphoneRef = useRef(false);
  // Timing diagnostics: when the modal opened and what the softphone status
  // was at that moment, so the wait-for-registration leg is measurable.
  const modalOpenedAtRef = useRef(performance.now());
  const voipStatusAtOpenRef = useRef(voipStatus);
  // Real call-duration tracking: stamp the wall-clock time when the call
  // actually connects and when it ends, so we can report true talk time
  // instead of a placeholder value.
  const connectedAtRef = useRef(null);
  const endedAtRef = useRef(null);
  // Live call timer shown next to the "Connected" label while the call is active.
  const [callElapsedSeconds, setCallElapsedSeconds] = useState(0);
  const [reviewCallData, setReviewCallData] = useState(call);
  const reviewSummary = (reviewCallData?.summary || '').trim();
  const reviewTranscript = (reviewCallData?.transcript || '').trim();
  const showReviewSummary = reviewSummary && !hasPlaceholderSummary(reviewSummary);

  // Fetch the freshest candidate notes when the modal opens so the recruiter
  // always sees the latest version (call.candidate_notes can be stale).
  useEffect(() => {
    if (!call.candidate_id) return;
    let cancelled = false;
    setNotesLoading(true);
    axios.get(`${API_BASE}/candidates/${call.candidate_id}`)
      .then(res => {
        if (!cancelled) {
          const fetched = res.data?.notes || '';
          setLiveNotes(fetched);
          originalNotesRef.current = fetched;
        }
      })
      .catch(() => {
        if (!cancelled) setLiveNotes(call.candidate_notes || '');
      })
      .finally(() => {
        if (!cancelled) setNotesLoading(false);
      });
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [call.candidate_id]);

  const effectiveError = initiationError || voipError || 'Browser VoIP could not be established.';
  const effectiveErrorCode = initiationErrorCode || voipErrorCode || '';
  const effectiveActionLabel = initiationActionLabel || voipActionLabel || '';
  const effectiveActionUrl = initiationActionUrl || voipActionUrl || '';
  const hasBlockingVoipError = voipStatus === 'error' || callState === 'error';
  const displayedAgentEmail = agentEmail || voipMeta?.agent_email || call?.recruiter_email || 'Loading...';

  const triggerCall = useCallback(async () => {
    setInitiationError('');
    setInitiationErrorCode('');
    setInitiationActionLabel('');
    setInitiationActionUrl('');
    setCallWrapUpMeta(null);
    
    // Check local VoIP status before initiating
    if (voipStatus === 'error') {
      setInitiationError(voipError || 'Browser VoIP is unavailable.');
      setInitiationErrorCode(voipErrorCode || '');
      setInitiationActionLabel(voipActionLabel || '');
      setInitiationActionUrl(voipActionUrl || '');
      setCallState('error');
      return;
    }

    setCallState('connecting');
    isInitiated.current = true;
    // Immediate audible feedback: the real Plivo ringback only starts once the
    // remote leg rings (~5s in), so play a local ring for the setup gap.
    startDialTone?.();
    if (modalOpenedAtRef.current) {
      reportTiming('modal_open_to_dial_start', performance.now() - modalOpenedAtRef.current, `voipStatusAtOpen=${voipStatusAtOpenRef.current}`);
    }
    const clickStart = performance.now();

    try {
      const micStart = performance.now();
      const micResult = await ensureMicrophonePermission();
      reportTiming('mic_permission', performance.now() - micStart);
      if (!micResult?.success) {
        const message = micResult?.error || 'Microphone permission is required to place a Plivo browser call';
        setInitiationError(message);
        setInitiationErrorCode('microphone_permission_denied');
        setCallState('error');
        toast.error(message);
        return;
      }

      const initiateStart = performance.now();
      const res = await initiateCall(call.id, { plivoUsername: endpointUsername });
      reportTiming('initiate_api', performance.now() - initiateStart);
      if (!res.success) {
        const message = res.error || 'Failed to start browser VoIP call';
        setInitiationError(message);
        setInitiationErrorCode(res.errorCode || '');
        setInitiationActionLabel(res.actionLabel || '');
        setInitiationActionUrl(res.actionUrl || '');
        setCallState('error');
        toast.error(message);
        if (res.errorCode === 'call_task_not_found') {
          await onRefresh?.();
          onClose();
        }
        return;
      }

      if (placeCall && call.candidate_phone) {
        const placeStart = performance.now();
        const placeResult = await placeCall(call.candidate_phone);
        reportTiming('place_call', performance.now() - placeStart);
        if (!placeResult?.success) {
          const message = placeResult?.error || 'Browser VoIP could not start the call';
          setInitiationError(message);
          setInitiationErrorCode('browser_call_start_failed');
          setCallState('error');
          toast.error(message);
          return;
        }
      }

      const handshakeStart = performance.now();
      const dialState = await waitForPlivoDial(endpointUsername);
      reportTiming('dial_handshake', performance.now() - handshakeStart);
      reportTiming('click_to_webhook_total', performance.now() - clickStart);
      if (!dialState?.success) {
        const message = dialState?.error || 'Plivo browser call did not reach the backend';
        setInitiationError(message);
        setInitiationErrorCode(dialState?.code || 'plivo_dial_webhook_timeout');
        setCallState('error');
        toast.error(message);
        return;
      }

      setCallState('waiting_for_invite');
    } catch (e) {
      const message = 'Connection error while starting browser VoIP call';
      setInitiationError(message);
      setInitiationErrorCode('voip_call_start_failed');
      setCallState('error');
      toast.error(message);
    }
  }, [call.id, endpointUsername, ensureMicrophonePermission, initiateCall, onClose, onRefresh, placeCall, startDialTone, waitForPlivoDial, voipActionLabel, voipActionUrl, voipError, voipErrorCode, voipStatus, call.candidate_phone]);

  // Kill the local dialing tone the moment the call reaches any state where
  // it should not ring: connected, ended, errored, or an incoming-answer ask.
  useEffect(() => {
    if (['active', 'ended', 'review', 'error', 'answer_required'].includes(callState)) {
      stopDialTone?.();
    }
  }, [callState, stopDialTone]);

  useEffect(() => {
    if (isInitiated.current) return;

    if (voipStatus === 'error') {
      setInitiationError(voipError || 'Browser VoIP is unavailable.');
      setInitiationErrorCode(voipErrorCode || '');
      setInitiationActionLabel(voipActionLabel || '');
      setInitiationActionUrl(voipActionUrl || '');
      setCallState('error');
      return;
    }

    if (voipStatus !== 'registered') {
      setCallState('preparing_softphone');
      return;
    }

    triggerCall();
  }, [triggerCall, voipActionLabel, voipActionUrl, voipError, voipErrorCode, voipStatus]);

  useEffect(() => {
    if (callState === 'review') return;

    if (voipStatus === 'error') {
      setInitiationError(voipError || 'Browser VoIP failed.');
      setInitiationErrorCode(voipErrorCode || '');
      setInitiationActionLabel(voipActionLabel || '');
      setInitiationActionUrl(voipActionUrl || '');
      setCallState('error');
      return;
    }

    if (activeCall?.state === 'connected' || voipStatus === 'connected') {
      setCallState('active');
      return;
    }

    if (activeCall?.state === 'answer_required' || voipStatus === 'answer_required') {
      setCallState('answer_required');
      return;
    }

    if (activeCall?.state === 'invite_received' || voipStatus === 'invite_received') {
      setCallState('invite_received');
      return;
    }

    if (isInitiated.current && !activeCall && voipStatus === 'registered' && callState === 'connecting') {
      setCallState('waiting_for_invite');
    }
  }, [activeCall, callState, voipActionLabel, voipActionUrl, voipError, voipErrorCode, voipStatus]);

  useEffect(() => {
    const recovered = ['registered', 'answer_required', 'invite_received', 'connected'].includes(voipStatus);
    if (!recovered) return;

    autoRetriedSoftphoneRef.current = false;
    setSoftphoneRecoveryAttempt(0);
    setInitiationError('');
    setInitiationErrorCode('');
    setInitiationActionLabel('');
    setInitiationActionUrl('');

    if (callState !== 'error') return;

    if (activeCall?.state === 'connected' || voipStatus === 'connected') {
      setCallState('active');
      return;
    }

    if (activeCall?.state === 'answer_required' || voipStatus === 'answer_required') {
      setCallState('answer_required');
      return;
    }

    if (activeCall?.state === 'invite_received' || voipStatus === 'invite_received') {
      setCallState('invite_received');
      return;
    }

    if (isInitiated.current && voipStatus === 'registered') {
      setCallState('waiting_for_invite');
      return;
    }

    setCallState('preparing_softphone');
  }, [activeCall, callState, voipStatus]);

  useEffect(() => {
    if (callState !== 'preparing_softphone' || isInitiated.current) return undefined;
    if (['registered', 'answer_required', 'invite_received', 'connected', 'error'].includes(voipStatus)) {
      return undefined;
    }

    const timeoutId = window.setTimeout(() => {
      if (isInitiated.current) return;
      if (['registered', 'answer_required', 'invite_received', 'connected', 'error'].includes(voipStatus)) {
        return;
      }

      if (!autoRetriedSoftphoneRef.current && retryVoip) {
        autoRetriedSoftphoneRef.current = true;
        setSoftphoneRecoveryAttempt(1);
        retryVoip();
        return;
      }

      const exhausted = Boolean(voipConnectionEvent?.maxRetriesReached);
      const message = voipConnectionEvent?.error
        || (exhausted ? 'Plivo softphone registration failed' : 'Plivo softphone registration timed out');
      setInitiationError(message);
      setInitiationErrorCode(exhausted ? 'softphone_registration_failed' : 'softphone_registration_timeout');
      setCallState('error');
    }, softphoneRecoveryAttempt > 0 ? SOFTPHONE_FIRST_CLICK_RECOVERY_MS : SOFTPHONE_PREPARING_TIMEOUT_MS);

    return () => window.clearTimeout(timeoutId);
  }, [callState, retryVoip, softphoneRecoveryAttempt, voipConnectionEvent, voipStatus]);

  useEffect(() => {
    if ((callState === 'answer_required' || callState === 'invite_received' || callState === 'active') && !activeCall) {
      setCallState('ended');
    }
  }, [activeCall, callState]);

  // Stamp connect/end times to measure the real call duration.
  useEffect(() => {
    if (callState === 'active' && connectedAtRef.current === null) {
      connectedAtRef.current = Date.now();
      endedAtRef.current = null;
    }
    if ((callState === 'ended' || callState === 'review') && connectedAtRef.current !== null && endedAtRef.current === null) {
      endedAtRef.current = Date.now();
    }
  }, [callState]);

  // Tick the visible timer once per second while the call is active.
  useEffect(() => {
    if (callState !== 'active') return undefined;
    const tick = () => {
      if (connectedAtRef.current !== null) {
        setCallElapsedSeconds(Math.max(0, Math.floor((Date.now() - connectedAtRef.current) / 1000)));
      }
    };
    tick();
    const intervalId = window.setInterval(tick, 1000);
    return () => window.clearInterval(intervalId);
  }, [callState]);

  useEffect(() => {
    if (!voipCallEvent?.at || lastHandledCallEventRef.current === voipCallEvent.at || callState === 'review') {
      return;
    }

    lastHandledCallEventRef.current = voipCallEvent.at;

    if (voipCallEvent.type === 'dialing') {
      return;
    }

    if (voipCallEvent.type === 'connected') {
      setCallWrapUpMeta(null);
      return;
    }

    if (!['terminated', 'failed'].includes(voipCallEvent.type)) {
      return;
    }

    const nextMeta = buildCallWrapUpMeta(voipCallEvent, call.candidate_name);
    setCallWrapUpMeta(nextMeta);
    if (!outcome && nextMeta.suggestedOutcome) {
      setOutcome(nextMeta.suggestedOutcome);
    }
    setCallState('ended');

    if (voipCallEvent.origin !== 'local') {
      if (voipCallEvent.type === 'failed') {
        toast.error(nextMeta.title);
      } else {
        toast.info(nextMeta.title);
      }
    }
  }, [call.candidate_name, callState, outcome, voipCallEvent]);

  useEffect(() => {
    let t;
    if (callState === 'review' && needsPostCallArtifacts(reviewCallData)) {
      const fetchReviewData = async () => {
         if (!isDocumentVisible()) return;
         try {
           // Keep syncing until recording, transcript, and summary are all healthy.
           if (needsPostCallArtifacts(reviewCallData)) {
             console.log('Proactively syncing call artifacts for Call', call.id);
             await syncCallRecording(call.id);
           }

           // background: refresh only this modal's data — never disturb the
           // Calls page's own fetch/request state (doing so left the page
           // stale until a manual refresh).
           const res = await fetchCalls({ list_id: call.list_id }, { background: true });
           if (res.success && res.data) {
             const updated = res.data.find(c => c.id === call.id);
             if (updated) setReviewCallData(updated);
           }
         } catch(e) {}
      };
      t = setInterval(fetchReviewData, 5000);
      fetchReviewData(); // Run immediately on enter
    }
    return () => clearInterval(t);
  }, [callState, call.id, call.list_id, fetchCalls, syncCallRecording, reviewCallData]);

  const handleAnswer = async () => {
    setIsAnswering(true);
    const result = await answerCall();
    setIsAnswering(false);

    if (!result.success) {
      const message = result.error || 'Failed to answer browser VoIP call';
      setInitiationError(message);
      toast.error(message);
      return;
    }

    setCallState('invite_received');
  };

  const handleEndCall = async () => {
    setCallWrapUpMeta(buildCallWrapUpMeta({ type: 'terminated', origin: 'local' }, call.candidate_name));
    await rejectCall();
    setCallState('ended');
  };

  const handleRetryVoip = () => {
    setInitiationError('');
    setInitiationErrorCode('');
    setInitiationActionLabel('');
    setInitiationActionUrl('');
    setCallWrapUpMeta(null);
    setCallState('preparing_softphone');
    isInitiated.current = false;
    autoRetriedSoftphoneRef.current = false;
    connectedAtRef.current = null;
    endedAtRef.current = null;
    setSoftphoneRecoveryAttempt(0);
    retryVoip();
  };

  const handleBlockingAction = async () => {
    if (!effectiveActionUrl) return;

    if (/^https?:\/\//.test(effectiveActionUrl)) {
      window.location.href = effectiveActionUrl;
      return;
    }

    window.location.href = effectiveActionUrl.startsWith('/') ? effectiveActionUrl : `${BACKEND_BASE}/${effectiveActionUrl}`;
  };

  const handleSaveLog = async () => {
    if (!outcome) {
      toast.error('Please select a call status');
      return;
    }
    const isFollowUpOutcome = outcome === FOLLOWUP_OUTCOME;
    if (isFollowUpOutcome && (!followupDueDate || !followupDueTime)) {
      toast.error('Please pick the follow-up date and time');
      return;
    }
    setSaving(true);
    try {
      // Real talk time = end timestamp − connect timestamp (in whole seconds).
      // Falls back to 0 when the call never actually connected.
      const measuredDuration = (connectedAtRef.current && endedAtRef.current)
        ? Math.max(0, Math.round((endedAtRef.current - connectedAtRef.current) / 1000))
        : 0;
      const payload = {
        status: isFollowUpOutcome ? 'pending' : 'completed',
        outcome,
        notes: liveNotes,
        duration: measuredDuration
      };
      if (isFollowUpOutcome) {
        payload.due_date = followupDueDate;
        payload.due_time = followupDueTime;
      }

      const res = await updateCall(call.id, payload);
      if (!res?.success) {
        toast.error(res?.error || 'Failed to save log');
        return;
      }

      // Save the edited notes back to the candidate profile so they're visible
      // in Manage Roles and Talent Pool.
      const currentNotes = liveNotes.trim();
      if (call.candidate_id && currentNotes !== originalNotesRef.current.trim()) {
        await updateCandidateNotes(call.candidate_id, currentNotes);
      }

      const result = res.data || {};
      if (result.auto_unreachable) {
        toast.warning('5th failed attempt — candidate marked Unreachable and moved to cooldown');
      } else if (result.wrong_number_tagged) {
        toast.warning('Number tagged as wrong. Calling is paused — source an alternate number to resume.');
      } else if (result.scheduled_next_title === 'Follow-up Call') {
        toast.success(`Follow-up scheduled for ${formatLocalDate(followupDueDate)} at ${formatDueTime(followupDueTime)}`);
      } else if (result.scheduled_next_title) {
        toast.success(`Call logged — next attempt scheduled (${result.scheduled_next_title})`);
      } else {
        toast.success('Call log saved');
      }

      onRefresh();
      setCallState('review');
    } catch (e) {
      toast.error('Failed to save log');
    } finally {
      setSaving(false);
    }
  };

  const tabs = [
    { id: 'linkedin', label: 'LinkedIn', icon: ExternalLink },
    { id: 'email', label: 'Email', icon: MessageSquare },
    { id: 'calls', label: 'Calls', icon: Phone },
    { id: 'tasks', label: 'Tasks', icon: CheckSquare },
  ];
  
  const callStatusMeta = callState === 'preparing_softphone'
    ? { label: 'Connecting', tone: '#2563eb', bg: '#eff6ff', message: '' }
    : callState === 'connecting'
      ? { label: 'Connecting call…', tone: '#2563eb', bg: '#eff6ff', message: '' }
      : callState === 'waiting_for_invite'
        ? { label: 'Ringing', tone: '#2563eb', bg: '#eff6ff', message: '' }
        : callState === 'answer_required'
          ? { label: 'Incoming Call', tone: '#d97706', bg: '#fef3c7', message: '' }
          : callState === 'invite_received'
            ? { label: 'Connecting', tone: '#2563eb', bg: '#eff6ff', message: '' }
            : callState === 'active'
              ? { label: 'Connected', tone: '#10b981', bg: '#ecfdf5', message: '' }
              : callState === 'ended'
                ? {
                    label: callWrapUpMeta?.title || 'Call Ended',
                    tone: callWrapUpMeta?.tone || '#475569',
                    bg: callWrapUpMeta?.bg || '#f8fafc',
                    message: '',
                  }
                : callState === 'error'
                  ? { label: 'Not Reachable', tone: '#dc2626', bg: '#fef2f2', message: '' }
                  : { label: 'Processing', tone: '#8b5cf6', bg: '#f5f3ff', message: '' };

  const isLiveCallState = ['preparing_softphone', 'connecting', 'waiting_for_invite', 'answer_required', 'invite_received', 'active', 'error'].includes(callState);
  const handleCloseModal = useCallback(async () => {
    if (isLiveCallState || activeCall || isInitiated.current) {
      await rejectCall();
    }
    onClose();
  }, [activeCall, isLiveCallState, onClose, rejectCall]);

  return (
    <div className="call-modal-overlay" style={{ position: 'fixed', inset: 0, background: 'rgba(15, 23, 42, 0.7)', backdropFilter: 'blur(4px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 10000 }}>
      <div className="call-modal-shell" style={{ background: '#fff', borderRadius: '24px', width: '100%', maxWidth: '720px', overflow: 'hidden', boxShadow: '0 25px 50px -12px rgba(0,0,0,0.25)', border: '1px solid #e2e8f0' }}>
        <div className="call-modal-header" style={{ padding: '24px', borderBottom: '1px solid #f1f5f9', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h2 style={{ fontSize: '18px', fontWeight: 800, color: '#0f172a', marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              {call.candidate_name}
              {call.candidate_linkedin && (
                <a
                  href={call.candidate_linkedin}
                  target="_blank"
                  rel="noopener noreferrer"
                  title="Open LinkedIn profile"
                  style={{ display: 'inline-flex', alignItems: 'center', color: 'var(--accent-primary)' }}
                >
                  <Linkedin size={16} />
                </a>
              )}
            </h2>
            <p style={{ fontSize: '13px', color: '#64748b' }}>Candidate Conversations</p>
          </div>
          <div className="call-modal-tabs" style={{ display: 'flex', background: '#f1f5f9', padding: '4px', borderRadius: '12px', gap: '4px' }}>
            {tabs.map(tab => (
              <button 
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                style={{ 
                  padding: '6px 12px', borderRadius: '8px', border: activeTab === tab.id ? '1px solid #111827' : '1px solid transparent', background: activeTab === tab.id ? '#111827' : 'transparent',
                  color: activeTab === tab.id ? '#fff' : '#64748b', fontSize: '12px', fontWeight: 700, 
                  cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px',
                  boxShadow: activeTab === tab.id ? '0 6px 12px rgba(15,23,42,0.12)' : 'none'
                }}
              >
                <tab.icon size={14} />
                {tab.label}
              </button>
            ))}
          </div>
          <button onClick={handleCloseModal} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#94a3b8' }}>
            <X size={20} />
          </button>
        </div>

        <div className="call-modal-content" style={{ minHeight: '400px', background: '#fff' }}>
          {activeTab === 'calls' ? (
            <div className="call-modal-body" style={{ padding: '32px 40px 40px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
              <div className="call-status-banner" style={{
                alignSelf: 'stretch',
                marginBottom: '28px',
                padding: '10px 16px',
                borderRadius: '16px',
                background: callStatusMeta.bg,
                color: callStatusMeta.tone,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: '12px',
                transition: 'all 0.3s ease'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{
                    width: 8, height: 8, borderRadius: '50%', background: callStatusMeta.tone, flexShrink: 0,
                    animation: ['Ringing', 'Connecting'].includes(callStatusMeta.label) ? 'ping 1.5s cubic-bezier(0,0,0.2,1) infinite' : 'none',
                    display: 'inline-block',
                  }} />
                  <span style={{ fontSize: '13px', fontWeight: 700 }}>{callStatusMeta.label}</span>
                </div>
                <div style={{
                  padding: '4px 10px',
                  borderRadius: '999px',
                  background: '#fff',
                  border: `1px solid ${callStatusMeta.tone}22`,
                  fontSize: '11px',
                  fontWeight: 700,
                  color: '#475569',
                }}>
                  {call.list_name || 'Call Task'}
                </div>
              </div>

              {/* Call Mode / Agent Cards Removed */}

              {hasBlockingVoipError && (
                <div style={{
                  alignSelf: 'stretch', marginBottom: '20px', padding: '20px',
                  borderRadius: '16px', background: '#fef2f2', border: '1px solid #fecaca',
                  display: 'flex', flexDirection: 'column', gap: '12px', alignItems: 'center', textAlign: 'center'
                }}>
                  <div style={{ color: '#b91c1c', fontSize: '14px', fontWeight: 700 }}>
                    Could not connect the call
                  </div>
                  <div style={{ color: '#dc2626', fontSize: '13px' }}>
                    Please check your microphone and internet connection, then try again.
                  </div>
                  {effectiveActionLabel && effectiveActionUrl && (
                    <button onClick={handleBlockingAction} style={{ ...CALL_PRIMARY_BUTTON, display: 'block', width: '100%', padding: '12px', fontSize: '14px' }}>
                      {effectiveActionLabel}
                    </button>
                  )}
                </div>
              )}

              {isLiveCallState ? (
                <>
                  <h3 style={{ fontSize: '20px', fontWeight: 700, color: '#0f172a', marginBottom: '4px', textAlign: 'center' }}>
                    {call.candidate_name || 'Candidate'}
                  </h3>
                  <p style={{ fontSize: '13px', color: '#64748b', marginBottom: '20px' }}>
                    {call.candidate_phone}
                  </p>
                  <div style={{ fontSize: '14px', fontWeight: 600, color: callStatusMeta.tone, marginBottom: callState === 'active' ? '8px' : '36px' }}>
                    {callState === 'active' ? 'Connected'
                      : callState === 'error' ? 'Not Reachable'
                      : callState === 'answer_required' ? 'Incoming Call'
                      : callState === 'preparing_softphone' || callState === 'invite_received' ? 'Connecting...'
                      : 'Ringing...'}
                  </div>
                  {callState === 'active' && (
                    <div style={{ fontSize: '22px', fontWeight: 700, color: '#0f172a', fontVariantNumeric: 'tabular-nums', marginBottom: '28px' }}>
                      {formatCallTimer(callElapsedSeconds)}
                    </div>
                  )}

                  {callState === 'answer_required' ? (
                    <div className="call-modal-actions" style={{ display: 'flex', gap: '12px', width: '100%' }}>
                      <button
                        onClick={handleAnswer}
                        disabled={isAnswering}
                        style={{
                          flex: 1,
                          padding: '16px',
                          background: '#dcfce7',
                          color: '#15803d',
                          border: 'none',
                          borderRadius: '16px',
                          fontSize: '15px',
                          fontWeight: 700,
                          cursor: isAnswering ? 'not-allowed' : 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          gap: '8px'
                        }}
                      >
                        <PhoneIncoming size={20} /> {isAnswering ? 'Answering...' : 'Answer Call'}
                      </button>
                      <button
                        onClick={handleEndCall}
                        style={{
                          flex: 1,
                          padding: '16px',
                          background: '#fee2e2',
                          color: '#ef4444',
                          border: 'none',
                          borderRadius: '16px',
                          fontSize: '15px',
                          fontWeight: 700,
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          gap: '8px'
                        }}
                      >
                        <X size={20} /> Reject
                      </button>
                    </div>
                  ) : callState === 'error' ? (
                    <div className="call-modal-actions" style={{ display: 'flex', gap: '12px', width: '100%' }}>
                      {effectiveActionLabel && effectiveActionUrl && (
                        <button
                          onClick={handleBlockingAction}
                          style={{
                            ...CALL_PRIMARY_BUTTON,
                            flex: 1,
                            padding: '16px',
                            borderRadius: '16px',
                            fontSize: '15px',
                          }}
                        >
                          {effectiveActionLabel}
                        </button>
                      )}
                      <button
                        onClick={handleRetryVoip}
                        style={{
                          ...CALL_SECONDARY_BUTTON,
                          flex: 1,
                          padding: '16px',
                          borderRadius: '16px',
                          fontSize: '15px',
                        }}
                      >
                        <RefreshCw size={20} /> Try Again
                      </button>
                      <button
                        onClick={handleCloseModal}
                        style={{
                          ...CALL_SECONDARY_BUTTON,
                          flex: 1,
                          padding: '16px',
                          borderRadius: '16px',
                          fontSize: '15px',
                        }}
                      >
                        <X size={20} /> Close
                      </button>
                    </div>
                  ) : (
                    <button 
                      onClick={handleEndCall}
                      style={{ 
                        width: '100%', padding: '16px', background: '#fee2e2', color: '#ef4444', 
                        border: 'none', borderRadius: '16px', fontSize: '15px', fontWeight: 700, 
                        cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px'
                      }}
                    >
                      <X size={20} /> {callState === 'active' ? 'End Call' : 'Cancel Call'}
                    </button>
                  )}

                  {/* Wrap-up fields available during the live call — same state as the
                      post-call log screen, so anything set here carries over on save. */}
                  {callState === 'active' && (
                    <div style={{ alignSelf: 'stretch', marginTop: '28px', textAlign: 'left' }}>
                      <div style={{ marginBottom: '24px' }}>
                        <label style={{ display: 'block', fontSize: '12px', fontWeight: 700, color: '#94a3b8', marginBottom: '8px', textTransform: 'uppercase' }}>Call Status</label>
                        <select
                          value={outcome}
                          onChange={e => setOutcome(e.target.value)}
                          style={{ width: '100%', padding: '12px 16px', borderRadius: '12px', border: '1px solid rgba(203,213,225,0.9)', fontSize: '14px', outline: 'none', background: '#fff' }}
                        >
                          <option value="">Select a status...</option>
                          {OUTCOMES.map(o => <option key={o} value={o}>{o}</option>)}
                        </select>
                      </div>

                      <div style={{ marginBottom: '24px' }}>
                        <label style={{ display: 'block', fontSize: '12px', fontWeight: 700, color: '#94a3b8', marginBottom: '8px', textTransform: 'uppercase' }}>Candidate Status</label>
                        <div>
                          <StatusDropdown
                            status={candidateStatus}
                            candidateId={call.candidate_id}
                            optimistic
                            onUpdate={(id, newStatus) => setCandidateStatus(newStatus)}
                          />
                        </div>
                        <div style={{ marginTop: '6px', fontSize: '11px', color: '#94a3b8' }}>
                          Updates across Manage Roles and Talent Pool.
                        </div>
                      </div>

                      <div>
                        <label style={{ display: 'block', fontSize: '12px', fontWeight: 700, color: '#94a3b8', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                          Notes
                        </label>
                        <textarea
                          placeholder={notesLoading ? 'Loading notes…' : 'Add notes about this candidate...'}
                          value={liveNotes}
                          onChange={e => setLiveNotes(e.target.value)}
                          disabled={notesLoading}
                          style={{
                            width: '100%', padding: '12px 16px',
                            borderRadius: '12px',
                            border: '1px solid rgba(203,213,225,0.9)', fontSize: '14px', minHeight: '120px',
                            resize: 'none', background: notesLoading ? '#f8fafc' : '#fff',
                            boxSizing: 'border-box', outline: 'none', lineHeight: 1.6,
                            color: notesLoading ? '#94a3b8' : '#334155',
                          }}
                        />
                      </div>
                    </div>
                  )}
                </>
              ) : callState === 'ended' ? (
                <div style={{ width: '100%' }}>
                  {callWrapUpMeta && (
                    <div style={{
                      marginBottom: '20px',
                      padding: '12px 16px',
                      borderRadius: '12px',
                      background: callWrapUpMeta.bg || '#f8fafc',
                      border: `1px solid ${callWrapUpMeta.border || '#e2e8f0'}`,
                      display: 'flex', alignItems: 'center', gap: '10px',
                    }}>
                      <span style={{ width: 8, height: 8, borderRadius: '50%', background: callWrapUpMeta.tone || '#475569', flexShrink: 0 }} />
                      <span style={{ fontSize: '13px', fontWeight: 700, color: callWrapUpMeta.tone || '#475569' }}>
                        {callWrapUpMeta.title || 'Call Ended'}
                      </span>
                    </div>
                  )}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px', padding: '12px 16px', background: '#f8fafc', borderRadius: '12px', flexWrap: 'wrap' }}>
                    <PhoneCall size={18} color="#2563eb" />
                    <span style={{ fontSize: '15px', fontWeight: 700, color: '#0f172a' }}>Log Call Details</span>
                    {call.candidate_status && (
                      <span style={{
                        marginLeft: 'auto', padding: '4px 10px', borderRadius: '999px',
                        background: '#eef2ff', color: '#4338ca', border: '1px solid #c7d2fe',
                        fontSize: '11px', fontWeight: 800, letterSpacing: '0.02em', whiteSpace: 'nowrap',
                      }}>
                        Candidate status: {call.candidate_status}
                      </span>
                    )}
                  </div>

                  <div style={{ marginBottom: '24px' }}>
                    <label style={{ display: 'block', fontSize: '12px', fontWeight: 700, color: '#94a3b8', marginBottom: '8px', textTransform: 'uppercase' }}>Call Status</label>
                    <select
                      value={outcome}
                      onChange={e => setOutcome(e.target.value)}
                      style={{ width: '100%', padding: '12px 16px', borderRadius: '12px', border: '1px solid rgba(203,213,225,0.9)', fontSize: '14px', outline: 'none', background: '#fff' }}
                    >
                      <option value="">Select a status...</option>
                      {OUTCOMES.map(o => <option key={o} value={o}>{o}</option>)}
                    </select>
                  </div>

                  <div style={{ marginBottom: '24px' }}>
                    <label style={{ display: 'block', fontSize: '12px', fontWeight: 700, color: '#94a3b8', marginBottom: '8px', textTransform: 'uppercase' }}>Candidate Status</label>
                    <div>
                      <StatusDropdown
                        status={candidateStatus}
                        candidateId={call.candidate_id}
                        optimistic
                        onUpdate={(id, newStatus) => setCandidateStatus(newStatus)}
                      />
                    </div>
                    <div style={{ marginTop: '6px', fontSize: '11px', color: '#94a3b8' }}>
                      Updates across Manage Roles and Talent Pool.
                    </div>
                  </div>

                  {FAILED_OUTCOMES.has(outcome) && !isFinalAttempt(call) && (
                    <div style={{ padding: '12px', background: '#eff6ff', color: '#1e40af', borderRadius: '12px', fontSize: '13px', marginBottom: '24px', border: '1px solid #bfdbfe', display: 'flex', gap: '8px', alignItems: 'flex-start' }}>
                      <Calendar size={16} style={{ marginTop: '2px', flexShrink: 0 }} />
                      <div>
                        <strong style={{ display: 'block', marginBottom: '4px' }}>Automated Call Cadence</strong>
                        The next attempt in the Day 1 → 2 → 4 → 7 → 10 cadence will be scheduled automatically when you save.
                      </div>
                    </div>
                  )}

                  {FAILED_OUTCOMES.has(outcome) && isFinalAttempt(call) && (
                    <div style={{ padding: '12px', background: '#fef3c7', color: '#92400e', borderRadius: '12px', fontSize: '13px', marginBottom: '24px', border: '1px solid #fde68a', display: 'flex', gap: '8px', alignItems: 'flex-start' }}>
                      <Clock size={16} style={{ marginTop: '2px', flexShrink: 0 }} />
                      <div>
                        <strong style={{ display: 'block', marginBottom: '4px' }}>Final Attempt</strong>
                        This is the 5th attempt. On save, the candidate will be marked <strong>Unreachable</strong> and moved to the cooldown pool — no further calls will be scheduled.
                      </div>
                    </div>
                  )}

                  {outcome === WRONG_NUMBER_OUTCOME && (
                    <div style={{ padding: '12px', background: '#fef2f2', color: '#991b1b', borderRadius: '12px', fontSize: '13px', marginBottom: '24px', border: '1px solid #fecaca', display: 'flex', gap: '8px', alignItems: 'flex-start' }}>
                      <Phone size={16} style={{ marginTop: '2px', flexShrink: 0 }} />
                      <div>
                        <strong style={{ display: 'block', marginBottom: '4px' }}>Wrong Number</strong>
                        This number will be tagged as wrong and the calling cadence paused. Source an alternate number (edit the candidate&apos;s contact info) to resume calling.
                      </div>
                    </div>
                  )}

                  {outcome === FOLLOWUP_OUTCOME && (
                    <div className="call-followup-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '24px', padding: '16px', background: '#f0fdf4', borderRadius: '12px', border: '1px solid #bbf7d0' }}>
                      <div style={{ gridColumn: '1 / -1', fontSize: '13px', color: '#166534', display: 'flex', gap: '8px', alignItems: 'flex-start' }}>
                        <Calendar size={16} style={{ marginTop: '2px', flexShrink: 0 }} />
                        <span>Pick the slot the candidate asked for — they will reappear in the calling list at that date and time.</span>
                      </div>
                      <div>
                        <label style={{ display: 'block', fontSize: '12px', fontWeight: 700, color: '#94a3b8', marginBottom: '8px' }}>Follow-up Date *</label>
                        <input type="date" value={followupDueDate} onChange={e => setFollowupDueDate(e.target.value)} min={new Date().toISOString().split('T')[0]} style={{ width: '100%', padding: '12px', borderRadius: '12px', border: '1px solid rgba(203,213,225,0.9)', background: '#fff', boxSizing: 'border-box' }} />
                      </div>
                      <div>
                        <label style={{ display: 'block', fontSize: '12px', fontWeight: 700, color: '#94a3b8', marginBottom: '8px' }}>Follow-up Time *</label>
                        <input type="time" value={followupDueTime} onChange={e => setFollowupDueTime(e.target.value)} style={{ width: '100%', padding: '12px', borderRadius: '12px', border: '1px solid rgba(203,213,225,0.9)', background: '#fff', boxSizing: 'border-box' }} />
                      </div>
                    </div>
                  )}

                  {/* Notes — pre-populated with existing candidate notes, fully editable */}
                  <div style={{ marginBottom: '24px' }}>
                    <label style={{ display: 'block', fontSize: '12px', fontWeight: 700, color: '#94a3b8', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                      Notes
                    </label>
                    <textarea
                      placeholder={notesLoading ? 'Loading notes…' : 'Add notes about this candidate...'}
                      value={liveNotes}
                      onChange={e => setLiveNotes(e.target.value)}
                      disabled={notesLoading}
                      style={{
                        width: '100%', padding: '12px 16px',
                        borderRadius: '12px',
                        border: '1px solid rgba(203,213,225,0.9)', fontSize: '14px', minHeight: '120px',
                        resize: 'none', background: notesLoading ? '#f8fafc' : '#fff',
                        boxSizing: 'border-box', outline: 'none', lineHeight: 1.6,
                        color: notesLoading ? '#94a3b8' : '#334155',
                      }}
                    />
                  </div>

                  <div className="call-modal-actions" style={{ display: 'flex', gap: '12px' }}>
                    <button style={{ ...CALL_SECONDARY_BUTTON, flex: 1, padding: '14px' }} onClick={handleEndCall}>Cancel</button>
                    <button onClick={handleSaveLog} disabled={saving} style={{ ...CALL_PRIMARY_BUTTON, flex: 1, padding: '14px', cursor: saving ? 'not-allowed' : 'pointer', opacity: saving ? 0.7 : 1 }}>
                      {saving ? 'Saving...' : 'Save'}
                    </button>
                  </div>
                </div>
              ) : (
                <div style={{ width: '100%' }}>
                  <div style={{ padding: '20px', background: '#f8fafc', borderRadius: '16px', border: '1px solid #e2e8f0', marginBottom: '24px' }}>
                    <h4 style={{ fontSize: '14px', fontWeight: 800, color: '#0f172a', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <PhoneCall size={16} color="#2563eb" /> Post-Call Analysis
                    </h4>
                    
                    {reviewCallData?.recording_url ? (
                      <div style={{ marginBottom: '24px' }}>
                        <div style={{ fontSize: '13px', fontWeight: 700, color: '#64748b', marginBottom: '8px' }}>Recording</div>
                        <audio src={reviewCallData.recording_url} controls style={{ width: '100%', height: '40px' }} />
                      </div>
                    ) : (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '12px', background: '#eff6ff', borderRadius: '8px', color: '#1e40af', fontSize: '13px', marginBottom: '24px' }}>
                        <RefreshCw size={14} style={{ animation: 'spin 2s linear infinite' }} /> Polling for recording stream from Plivo...
                      </div>
                    )}
                    
                    {showReviewSummary && (
                      <div style={{ marginBottom: '24px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px', marginBottom: '8px' }}>
                          <div style={{ fontSize: '13px', fontWeight: 700, color: '#64748b' }}>AI Summary</div>
                          <SentimentBadge sentiment={reviewCallData?.sentiment} reason={reviewCallData?.sentiment_reason} />
                        </div>
                        <p style={{ fontSize: '14px', lineHeight: 1.6, color: '#334155' }}>{reviewSummary}</p>
                        {reviewCallData?.sentiment_reason && (
                          <p style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px dashed #e2e8f0', fontSize: '12px', color: '#94a3b8', fontStyle: 'italic' }}>
                            “{reviewCallData.sentiment_reason}”
                          </p>
                        )}
                      </div>
                    )}

                    <div style={{ marginBottom: '16px' }}>
                      <div style={{ fontSize: '13px', fontWeight: 700, color: '#64748b', marginBottom: '8px' }}>
                        Transcript
                      </div>
                      {reviewTranscript ? (
                        <div style={{ background: '#fff', padding: '16px', borderRadius: '12px', border: '1px solid #e2e8f0', maxHeight: '200px', overflowY: 'auto', fontSize: '13px', lineHeight: 1.6, color: '#475569' }}>
                          <TranscriptView
                            transcript={reviewTranscript}
                            candidateName={reviewCallData?.candidate_name}
                            recruiterName={recruiterDisplayName(reviewCallData)}
                          />
                        </div>
                      ) : reviewCallData?.completed_at && (new Date() - new Date(reviewCallData.completed_at)) > PENDING_ANALYSIS_WINDOW_MS ? (
                        <div style={{ padding: '24px', textAlign: 'center', background: '#fff', borderRadius: '12px', border: '1px dashed #cbd5e1', color: '#94a3b8', fontSize: '13px' }}>
                          <div style={{ marginBottom: '12px' }}>Analysis didn't complete for this call.</div>
                          <button
                            onClick={() => syncCallRecording(call.id)}
                            style={{ padding: '8px 16px', borderRadius: '10px', border: '1px solid #cbd5e1', background: '#fff', color: '#334155', fontSize: '12px', fontWeight: 700, cursor: 'pointer' }}
                          >
                            Retry sync
                          </button>
                        </div>
                      ) : (
                        <div style={{ padding: '24px', textAlign: 'center', background: '#fff', borderRadius: '12px', border: '1px dashed #cbd5e1', color: '#94a3b8', fontSize: '13px' }}>
                          Transcribing recording...
                        </div>
                      )}
                    </div>
                  </div>
                  <button onClick={onClose} style={{ ...CALL_SECONDARY_BUTTON, width: '100%', padding: '14px', color: '#0f172a' }}>Close Window</button>
                  {needsPostCallArtifacts(reviewCallData) && (
                    <p style={{ marginTop: '10px', textAlign: 'center', fontSize: '12px', color: '#94a3b8' }}>
                      Analysis continues in the background — this row will update when it's ready.
                    </p>
                  )}
                </div>
              )}
            </div>
          ) : activeTab === 'linkedin' || activeTab === 'email' ? (
            <ConversationHistoryPanel
              candidateId={call.candidate_id}
              candidateName={call.candidate_name}
              platform={activeTab}
            />
          ) : activeTab === 'tasks' ? (
            <CandidateActivityPanel
              candidateId={call.candidate_id}
              candidateName={call.candidate_name}
            />
          ) : null}
        </div>
      </div>
      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        @keyframes ping {
          0% { transform: scale(1); opacity: 1; }
          70%, 100% { transform: scale(1.5); opacity: 0; }
        }
      `}</style>
    </div>
  );
}
