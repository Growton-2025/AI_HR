import { useVoIP } from '../context/VoIPContext';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { 
  Phone, Calendar, CheckCircle2, List, PhoneCall, 
  Search, RefreshCw, MoreHorizontal, User, 
  Trash2, X, ChevronLeft, Send, MessageSquare, 
  CheckSquare, ExternalLink, Clock, PhoneForwarded, Mail,
  ClipboardList, Layers, PhoneIncoming
} from 'lucide-react';
import { useAppStore } from '../store/useAppStore';
import { toast } from 'sonner';

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

const TABS = [
  { id: 'today', label: 'Due Today', icon: Clock },
  { id: 'upcoming', label: 'Upcoming', icon: Calendar },
  { id: 'completed', label: 'Completed', icon: CheckCircle2 },
  { id: 'lists', label: 'Call Lists', icon: Layers },
];

const OUTCOMES = [
  'Left Voicemail',
  'Connected - Interested',
  'Connected - Not Interested',
  'Connected - Follow up later',
  'No Answer',
  'Wrong Number'
];

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
    const interval = setInterval(() => loadMessages({ silent: true }), 5000);
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
            background: '#eff6ff',
            color: '#2563eb',
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
                {error || 'Completed FreJun calls with recordings, summaries, and outcomes will appear here as the activity timeline fills in.'}
              </div>
            </div>
          </div>
        ) : (
          items.map(item => (
            <div key={item.id} style={{ position: 'relative', padding: '18px', borderRadius: '20px', background: '#fff', border: '1px solid #e2e8f0', boxShadow: '0 1px 3px rgba(15, 23, 42, 0.04)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: '16px', marginBottom: '14px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <div style={{ width: '36px', height: '36px', borderRadius: '12px', background: '#eff6ff', color: '#2563eb', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
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
    calls, fetchCalls, 
    callLists, fetchCallLists,
    callsLastQueryKey,
    callStats, fetchCallStats,
    updateCall, deleteCall, deleteCallList, createCallList, clearCallsState, syncCallRecording
  } = useAppStore();

  const [activeTab, setActiveTab] = useState('today');
  const [loading, setLoading] = useState(false);
  const [isRevalidating, setIsRevalidating] = useState(false);
  const [selectedList, setSelectedList] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [callingCandidate, setCallingCandidate] = useState(null); // The call object
  const [expandedCallId, setExpandedCallId] = useState(null);
  const [syncingCallId, setSyncingCallId] = useState(null);
  
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

    void fetchCallStats().then(res => {
      if (!res?.success) {
        console.error('Failed to refresh call stats:', res?.error);
      }
    });

    try {
      if (activeTab === 'lists' && !selectedList) {
        const listsRes = await fetchCallLists();
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

        const callsRes = await fetchCalls(params);
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

  const stats = [
    { label: 'DUE TODAY', value: callStats.due_today, icon: Phone, color: '#2563eb', bg: '#eff6ff' },
    { label: 'UPCOMING', value: callStats.upcoming, icon: Clock, color: '#f59e0b', bg: '#fffbeb' },
    { label: 'COMPLETED', value: callStats.completed, icon: CheckCircle2, color: '#10b981', bg: '#ecfdf5' },
    { label: 'CALL LISTS', value: callStats.active_lists, icon: List, color: '#8b5cf6', bg: '#f5f3ff' },
  ];

  const handleDial = (call) => {
    setCallingCandidate(call);
  };

  const handleDeleteCall = async (callId) => {
    if (!window.confirm('Remove this candidate from the call list?')) return;
    const res = await deleteCall(callId);
    if (res.success) toast.success('Removed from list');
    else toast.error(res.error || 'Failed to remove');
  };

  const handleDeleteList = async (listId, name) => {
    if (!window.confirm(`Delete the list "${name}"? This will remove all associated tasks.`)) return;
    const res = await deleteCallList(listId);
    if (res.success) {
      if (selectedList?.id === listId) setSelectedList(null);
      toast.success('List deleted');
    }
    else toast.error(res.error || 'Failed to delete list');
  };

  const handleCreateList = async () => {
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

    toast('Recording is not ready in FreJun yet');
  };

  const filteredCalls = (calls || []).filter(c => 
    (c.candidate_name || '').toLowerCase().includes(searchQuery.toLowerCase()) ||
    (c.candidate_title || '').toLowerCase().includes(searchQuery.toLowerCase())
  );
  const currentCallsQueryKey = selectedList
    ? `list_id=${selectedList.id}&status=pending`
    : activeTab === 'today'
      ? 'due_filter=today&status=pending'
      : activeTab === 'upcoming'
        ? 'due_filter=upcoming&status=pending'
        : activeTab === 'completed'
          ? 'status=completed'
          : '';
  const showCallsLoading = activeTab !== 'lists' && loading && callsLastQueryKey !== currentCallsQueryKey;
  const showListsLoading = activeTab === 'lists' && !selectedList && loading && !callLists.length;

  return (
    <div style={{ padding: '32px', background: '#f8fafc', minHeight: '100vh', fontFamily: '"Inter", sans-serif' }}>
      <header style={{ marginBottom: '32px', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1 style={{ fontSize: '28px', fontWeight: 800, color: '#0f172a', marginBottom: '8px' }}>Tasks Dashboard</h1>
          <p style={{ color: '#64748b', fontSize: '15px' }}>Track your calling progress and lists</p>
          {fetchNotice && (
            <div style={{ marginTop: '12px', padding: '10px 14px', borderRadius: '12px', background: '#fff7ed', color: '#c2410c', fontSize: '13px', fontWeight: 600, border: '1px solid #fdba74' }}>
              {fetchNotice}
            </div>
          )}
        </div>
        {isRevalidating && (
          <div style={{ padding: '8px 16px', borderRadius: '12px', background: '#eff6ff', color: '#2563eb', fontSize: '12px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
            <RefreshCw size={14} className="revalidating" /> Updating...
          </div>
        )}
      </header>

      {/* Stats Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '20px', marginBottom: '40px' }}>
        {stats.map((stat, i) => (
          <div key={i} style={{ padding: '24px', background: '#fff', borderRadius: '20px', border: '1px solid #e2e8f0', boxShadow: '0 1px 3px rgba(0,0,0,0.01)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
              <div style={{ width: '40px', height: '40px', borderRadius: '12px', background: stat.bg, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <stat.icon size={20} color={stat.color} />
              </div>
              <span style={{ fontSize: '12px', fontWeight: 700, color: '#94a3b8', letterSpacing: '0.05em' }}>{stat.label}</span>
            </div>
            <div style={{ fontSize: '32px', fontWeight: 800, color: '#0f172a' }}>{stat.value}</div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', borderBottom: '1px solid #e2e8f0', marginBottom: '32px', gap: '32px' }}>
        {TABS.map(tab => (
          <button
            key={tab.id}
            onClick={() => { clearCallsState(); setActiveTab(tab.id); setSelectedList(null); }}
            style={{
              padding: '12px 4px', background: 'none', border: 'none', borderBottom: activeTab === tab.id ? '2px solid #2563eb' : '2px solid transparent',
              color: activeTab === tab.id ? '#2563eb' : '#64748b', fontSize: '14px', fontWeight: 600,
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
      <div style={{ background: '#fff', borderRadius: '24px', border: '1px solid #e2e8f0', overflow: 'hidden' }}>
        {activeTab === 'lists' && !selectedList ? (
          <div style={{ padding: '24px' }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '20px' }}>
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
                style={{ 
                  padding: '24px', borderRadius: '20px', 
                  border: isCreatingList ? '2px solid #3b82f6' : '1px dashed #cbd5e1', 
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
                      onKeyDown={e => e.key === 'Enter' && handleCreateList()}
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
                        style={{ flex: 1, padding: '8px', background: '#2563eb', color: '#fff', border: 'none', borderRadius: '8px', fontSize: '13px', fontWeight: 600, cursor: isSubmittingList ? 'wait' : 'pointer' }}
                      >
                        {isSubmittingList ? 'Saving...' : 'Save List'}
                      </button>
                      <button 
                        onClick={(e) => { e.stopPropagation(); setIsCreatingList(false); setNewListName(''); }}
                        disabled={isSubmittingList}
                        style={{ padding: '8px 12px', background: '#f1f5f9', color: '#64748b', border: 'none', borderRadius: '8px', fontSize: '13px', fontWeight: 600, cursor: 'pointer' }}
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                )}
              </div>

              {callLists.map(list => (
                <div 
                  key={list.id} 
                  onClick={() => { clearCallsState(); setSelectedList(list); }}
                  style={{ 
                    padding: '24px', borderRadius: '20px', border: '1px solid #e2e8f0', cursor: 'pointer',
                    transition: 'all 0.2s'
                  }}
                  onMouseEnter={e => { e.currentTarget.style.borderColor = '#2563eb'; e.currentTarget.style.transform = 'translateY(-2px)'; }}
                  onMouseLeave={e => { e.currentTarget.style.borderColor = '#e2e8f0'; e.currentTarget.style.transform = 'none'; }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
                    <div style={{ width: '40px', height: '40px', borderRadius: '12px', background: '#eff6ff', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <List size={20} color="#2563eb" />
                    </div>
                    <button 
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteList(list.id, list.name);
                      }}
                      style={{ padding: '6px', background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer', borderRadius: '8px' }}
                      onMouseEnter={e => e.currentTarget.style.color = '#ef4444'}
                      onMouseLeave={e => e.currentTarget.style.color = '#94a3b8'}
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                  <h3 style={{ fontSize: '16px', fontWeight: 700, color: '#0f172a', marginBottom: '4px' }}>{list.name}</h3>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <User size={12} color="#94a3b8" />
                    <span style={{ fontSize: '13px', color: '#64748b' }}>{list.candidate_count} Pending</span>
                  </div>
                </div>
              ))}
              {(callLists || []).length === 0 && (
                <div style={{ gridColumn: '1/-1', py: 60, textAlign: 'center', color: '#94a3b8' }}>No call lists found yet.</div>
              )}
            </div>
          </div>
        ) : (
          <div style={{ minHeight: '400px' }}>
            <div style={{ padding: '20px 24px', borderBottom: '1px solid #f1f5f9', display: 'flex', alignItems: 'center', gap: '16px' }}>
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
                <span style={{ fontSize: '12px', background: '#eff6ff', color: '#2563eb', padding: '2px 8px', borderRadius: '20px', fontWeight: 600 }}>
                  {(filteredCalls || []).length} Contacts
                </span>
              )}
              
              <div style={{ position: 'relative', marginLeft: 'auto', width: '240px' }}>
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

            <div style={{ padding: '0 24px' }}>
              {showCallsLoading && Array.from({ length: 5 }).map((_, idx) => (
                <div
                  key={`call-skeleton-${idx}`}
                  style={{ padding: '20px 0', borderBottom: '1px solid #f1f5f9', display: 'flex', alignItems: 'center', gap: '16px' }}
                >
                  <div style={{ width: 40, height: 40, borderRadius: '50%', background: '#e2e8f0' }} />
                  <div style={{ flex: 1 }}>
                    <div style={{ width: '28%', height: 14, borderRadius: 8, background: '#e2e8f0', marginBottom: 8 }} />
                    <div style={{ width: '44%', height: 12, borderRadius: 8, background: '#f1f5f9', marginBottom: 8 }} />
                    <div style={{ width: '36%', height: 10, borderRadius: 8, background: '#f8fafc' }} />
                  </div>
                </div>
              ))}
              {!showCallsLoading && (
                <>
              {(filteredCalls || []).map(call => (
                <React.Fragment key={call.id}>
                <div 
                  style={{ 
                    padding: '20px 0', borderBottom: '1px solid #f1f5f9', display: 'flex', 
                    alignItems: 'center', gap: '16px'
                  }}
                >
                  <div style={{ width: '40px', height: '40px', borderRadius: '50%', background: '#f1f5f9', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '14px', fontWeight: 700, color: '#64748b' }}>
                    {call.candidate_name?.split(' ').map(n => n[0]).join('') || '?' }
                  </div>
                  <div style={{ flex: 1 }}>
                    <h3 style={{ fontSize: '15px', fontWeight: 700, color: '#0f172a', marginBottom: '2px' }}>{call.candidate_name || 'Anonymous'}</h3>
                    <p style={{ fontSize: '13px', color: '#64748b', marginBottom: '4px' }}>
                      {call.task_title || call.candidate_title || 'No Title'}
                    </p>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '12px', color: '#94a3b8' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <Phone size={12} /> {call.candidate_phone || 'N/A'}
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <Calendar size={12} /> 
                        {call.status === 'completed' 
                          ? `Completed: ${call.completed_at ? new Date(call.completed_at).toLocaleDateString() : 'Unknown'}`
                          : `Due: ${formatLocalDate(call.due_date)}`
                        }
                      </div>
                      {call.status === 'completed' && (
                        <>
                          <div style={{ color: '#2563eb', fontWeight: 600 }}>
                            {call.duration ? `${Math.floor(call.duration / 60)}m ${call.duration % 60}s` : '0s'}
                          </div>
                          <div style={{ color: '#059669', fontWeight: 600, textTransform: 'capitalize' }}>
                            {call.outcome || 'No Outcome'}
                          </div>
                        </>
                      )}
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: '8px' }}>
                    {call.status === 'completed' && (
                      <button
                        onClick={() => setExpandedCallId(expandedCallId === call.id ? null : call.id)}
                        style={{
                          padding: '8px 12px', borderRadius: '10px', fontSize: '12px', fontWeight: 700,
                          background: '#fff', color: '#2563eb', border: '1.5px solid #dbeafe', cursor: 'pointer',
                          display: 'flex', alignItems: 'center', gap: '6px'
                        }}
                      >
                        {expandedCallId === call.id ? 'Hide Details' : 'View Insights'}
                      </button>
                    )}
                    <button 
                      onClick={() => handleDeleteCall(call.id)}
                      style={{ padding: '8px', background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer', borderRadius: '8px' }}
                      onMouseEnter={e => e.currentTarget.style.color = '#ef4444'}
                      onMouseLeave={e => e.currentTarget.style.color = '#94a3b8'}
                      title="Remove from list"
                    >
                      <Trash2 size={18} />
                    </button>
                    <button 
                      onClick={() => handleDial(call)}
                      disabled={call.status === 'completed'}
                      style={{ 
                        padding: '8px 16px', borderRadius: '10px', fontSize: '13px', fontWeight: 700,
                        background: call.status === 'completed' ? '#f1f5f9' : '#2563eb',
                        color: call.status === 'completed' ? '#94a3b8' : '#fff',
                        border: 'none', cursor: call.status === 'completed' ? 'not-allowed' : 'pointer',
                        display: 'flex', alignItems: 'center', gap: '8px'
                      }}
                    >
                      <PhoneCall size={16} /> 
                    </button>
                  </div>
                </div>

                {/* Expanded Details Section */}
                {expandedCallId === call.id && (
                  <div style={{ 
                    padding: '24px', background: '#f8fafc', borderRadius: '16px', margin: '0 0 20px 56px',
                    border: '1.5px solid #e2e8f0', display: 'flex', flexDirection: 'column', gap: '20px'
                  }}>
                    <div style={{ background: '#fff', padding: '16px', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', alignItems: 'center', marginBottom: '12px' }}>
                        <h4 style={{ fontSize: '12px', fontWeight: 800, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Call Recording</h4>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                          {(call.frejun_link || call.frejun_summary_url) && (
                            <a
                              href={call.frejun_link || call.frejun_summary_url}
                              target="_blank"
                              rel="noreferrer"
                              style={{ fontSize: '12px', fontWeight: 700, color: '#2563eb', textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '6px' }}
                            >
                              <ExternalLink size={14} />
                              Open FreJun
                            </a>
                          )}
                          {!call.recording_url && (
                            <button
                              onClick={() => handleSyncRecording(call.id)}
                              disabled={syncingCallId === call.id}
                              style={{
                                padding: '8px 12px',
                                borderRadius: '10px',
                                border: '1px solid #dbeafe',
                                background: '#eff6ff',
                                color: '#1d4ed8',
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

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                      <div style={{ background: '#fff', padding: '20px', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
                        <h4 style={{ fontSize: '12px', fontWeight: 800, color: '#64748b', textTransform: 'uppercase', marginBottom: '12px', letterSpacing: '0.05em' }}>AI Summary</h4>
                        <div style={{ fontSize: '14px', color: '#334155', lineHeight: '1.6', whiteSpace: 'pre-wrap' }}>
                          {call.summary || (call.recording_url ? 'Processing summary...' : 'No summary available.')}
                        </div>
                      </div>
                      <div style={{ background: '#fff', padding: '20px', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
                        <h4 style={{ fontSize: '12px', fontWeight: 800, color: '#64748b', textTransform: 'uppercase', marginBottom: '12px', letterSpacing: '0.05em' }}>Full Transcript</h4>
                        <div style={{ 
                          fontSize: '13px', color: '#64748b', lineHeight: '1.6', height: '200px', overflowY: 'auto', 
                          paddingRight: '12px', whiteSpace: 'pre-wrap' 
                        }}>
                          {call.transcript || (call.recording_url ? 'Transcribing call...' : 'No transcript available.')}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </React.Fragment>
            ))}
              {(filteredCalls || []).length === 0 && !loading && (
                <div style={{ padding: '60px', textAlign: 'center', color: '#94a3b8' }}>No candidates matching your query.</div>
              )}
                </>
              )}
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
  );
}

function CallingModal({ call, onClose, onRefresh }) {
  const [activeTab, setActiveTab] = useState('calls');
  const [callState, setCallState] = useState('preparing_softphone');
  const [outcome, setOutcome] = useState('');
  const [notes, setNotes] = useState('');
  const [createFollowup, setCreateFollowup] = useState(false);
  const [followupTitle, setFollowupTitle] = useState(call?.task_title || '');
  const [followupDueDate, setFollowupDueDate] = useState('');
  const [saving, setSaving] = useState(false);
  const [isAnswering, setIsAnswering] = useState(false);
  const [initiationError, setInitiationError] = useState('');
  const [initiationErrorCode, setInitiationErrorCode] = useState('');
  const [initiationActionLabel, setInitiationActionLabel] = useState('');
  const [initiationActionUrl, setInitiationActionUrl] = useState('');
  const { updateCall, initiateCall, fetchCalls, syncCallRecording } = useAppStore();
  const { activeCall, answerCall, rejectCall, voipStatus, voipError, voipErrorCode, voipActionLabel, voipActionUrl, voipMeta, agentEmail, retryVoip } = useVoIP();
  const isInitiated = useRef(false);
  const [reviewCallData, setReviewCallData] = useState(call);

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

    try {
      const res = await initiateCall(call.id);
      if (!res.success) {
        const message = res.error || 'Failed to start browser VoIP call';
        setInitiationError(message);
        setInitiationErrorCode(res.errorCode || '');
        setInitiationActionLabel(res.actionLabel || '');
        setInitiationActionUrl(res.actionUrl || '');
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
  }, [call.id, initiateCall, voipActionLabel, voipActionUrl, voipError, voipErrorCode, voipStatus]);

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
    if ((callState === 'answer_required' || callState === 'invite_received' || callState === 'active') && !activeCall) {
      setCallState('ended');
    }
  }, [activeCall, callState]);

  useEffect(() => {
    let t;
    if (callState === 'review') {
      const fetchReviewData = async () => {
         try {
           // Proactively trigger a sync from FreJun API (Legacy Pattern)
           if (!reviewCallData?.recording_url) {
             console.log('Proactively syncing recording for Call', call.id);
             await syncCallRecording(call.id);
           }

           const res = await fetchCalls({ list_id: call.list_id }, { force: true, updateState: false });
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
  }, [callState, call.id, call.list_id, fetchCalls, syncCallRecording, reviewCallData?.recording_url]);

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
    await rejectCall();
    setCallState('ended');
  };

  const handleRetryVoip = () => {
    setInitiationError('');
    setInitiationErrorCode('');
    setInitiationActionLabel('');
    setInitiationActionUrl('');
    setCallState('preparing_softphone');
    isInitiated.current = false;
    retryVoip();
  };

  const handleBlockingAction = () => {
    if (!effectiveActionUrl) return;
    window.location.href = effectiveActionUrl;
  };

  const handleSaveLog = async () => {
    if (!outcome) {
      toast.error('Please select an outcome');
      return;
    }
    if (createFollowup && !followupDueDate) {
      toast.error('Please select a due date for the follow-up task');
      return;
    }
    setSaving(true);
    try {
      const payload = {
        status: createFollowup ? 'pending' : 'completed',
        outcome,
        notes,
        duration: Math.floor(Math.random() * 300) + 30
      };
      if (createFollowup) {
        payload.due_date = followupDueDate;
        payload.task_title = followupTitle.trim();
      }

      await updateCall(call.id, payload);
      toast.success(createFollowup ? 'Follow-up task scheduled' : 'Call log saved');
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
    ? { label: 'Preparing', tone: '#2563eb', bg: '#eff6ff', message: 'Registering the browser softphone...' }
    : callState === 'connecting'
      ? { label: 'Connecting', tone: '#2563eb', bg: '#eff6ff', message: 'Starting the browser VoIP call...' }
      : callState === 'waiting_for_invite'
        ? { label: 'Waiting', tone: '#d97706', bg: '#fef3c7', message: 'Waiting for FreJun to deliver the browser invite...' }
        : callState === 'answer_required'
          ? { label: 'Answer Required', tone: '#d97706', bg: '#fef3c7', message: 'The browser invite is ready. Answer to join the call.' }
          : callState === 'invite_received'
            ? { label: 'Joining', tone: '#2563eb', bg: '#eff6ff', message: 'Connecting browser audio...' }
            : callState === 'active'
              ? { label: 'Connected', tone: '#10b981', bg: '#ecfdf5', message: 'Two-way browser VoIP call is active.' }
              : callState === 'ended'
                ? { label: 'Wrap-up', tone: '#f97316', bg: '#fff7ed', message: 'Capture the outcome...' }
                : callState === 'error'
                  ? { label: 'VoIP Error', tone: '#dc2626', bg: '#fef2f2', message: effectiveError }
                  : { label: 'Review', tone: '#8b5cf6', bg: '#f5f3ff', message: 'AI processing recording...' };

  const isLiveCallState = ['preparing_softphone', 'connecting', 'waiting_for_invite', 'answer_required', 'invite_received', 'active', 'error'].includes(callState);

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(15, 23, 42, 0.7)', backdropFilter: 'blur(4px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 10000 }}>
      <div style={{ background: '#fff', borderRadius: '24px', width: '100%', maxWidth: '720px', overflow: 'hidden', boxShadow: '0 25px 50px -12px rgba(0,0,0,0.25)', border: '1px solid #e2e8f0' }}>
        <div style={{ padding: '24px', borderBottom: '1px solid #f1f5f9', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h2 style={{ fontSize: '18px', fontWeight: 800, color: '#0f172a', marginBottom: '4px' }}>{call.candidate_name}</h2>
            <p style={{ fontSize: '13px', color: '#64748b' }}>Candidate Conversations</p>
          </div>
          <div style={{ display: 'flex', background: '#f1f5f9', padding: '4px', borderRadius: '12px', gap: '4px' }}>
            {tabs.map(tab => (
              <button 
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                style={{ 
                  padding: '6px 12px', borderRadius: '8px', border: 'none', background: activeTab === tab.id ? '#fff' : 'transparent',
                  color: activeTab === tab.id ? '#0f172a' : '#64748b', fontSize: '12px', fontWeight: 700, 
                  cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px',
                  boxShadow: activeTab === tab.id ? '0 1px 3px rgba(0,0,0,0.1)' : 'none'
                }}
              >
                <tab.icon size={14} />
                {tab.label}
              </button>
            ))}
          </div>
          <button onClick={onClose} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#94a3b8' }}>
            <X size={20} />
          </button>
        </div>

        <div style={{ minHeight: '400px', background: '#fff' }}>
          {activeTab === 'calls' ? (
            <div style={{ padding: '32px 40px 40px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
              <div style={{
                alignSelf: 'stretch',
                marginBottom: '28px',
                padding: '14px 16px',
                borderRadius: '16px',
                background: callStatusMeta.bg,
                color: callStatusMeta.tone,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: '12px',
                transition: 'all 0.25s ease'
              }}>
                <div>
                  <div style={{ fontSize: '12px', fontWeight: 800, letterSpacing: '0.06em', textTransform: 'uppercase' }}>{callStatusMeta.label}</div>
                  <div style={{ fontSize: '13px', marginTop: '4px' }}>{callStatusMeta.message}</div>
                </div>
                <div style={{
                  padding: '6px 10px',
                  borderRadius: '999px',
                  background: '#fff',
                  border: `1px solid ${callStatusMeta.tone}22`,
                  fontSize: '11px',
                  fontWeight: 800
                }}>
                  {call.list_name || 'Call Task'}
                </div>
              </div>

              <div style={{ alignSelf: 'stretch', display: 'flex', justifyContent: 'space-between', gap: '12px', marginBottom: '24px' }}>
                <div style={{ flex: 1, padding: '12px 14px', borderRadius: '14px', background: '#f8fafc', border: '1px solid #e2e8f0' }}>
                  <div style={{ fontSize: '11px', fontWeight: 800, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '4px' }}>Call Mode</div>
                  <div style={{ fontSize: '14px', fontWeight: 700, color: '#0f172a' }}>Browser VoIP</div>
                </div>
                <div style={{ flex: 1, padding: '12px 14px', borderRadius: '14px', background: '#f8fafc', border: '1px solid #e2e8f0' }}>
                  <div style={{ fontSize: '11px', fontWeight: 800, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '4px' }}>Softphone Agent</div>
                  <div style={{ fontSize: '14px', fontWeight: 700, color: '#0f172a' }}>{displayedAgentEmail}</div>
                </div>
              </div>

              {hasBlockingVoipError && (
                <div style={{ 
                  alignSelf: 'stretch', marginBottom: '20px', padding: '20px', 
                  borderRadius: '16px', background: '#fff7ed', border: '1px solid #fed7aa', 
                  display: 'flex', flexDirection: 'column', gap: '12px', alignItems: 'center', textAlign: 'center'
                }}>
                  <div style={{ color: '#9a3412', fontSize: '14px', fontWeight: 700 }}>
                    Browser VoIP Unavailable
                  </div>
                  <div style={{ color: '#c2410c', fontSize: '13px' }}>
                    {effectiveError}
                  </div>
                  {effectiveErrorCode === 'browser_calling_disabled' && voipMeta?.agent_id && (
                    <div style={{ color: '#9a3412', fontSize: '12px', fontWeight: 600 }}>
                      Seat: {displayedAgentEmail} • Agent ID: {voipMeta.agent_id}
                    </div>
                  )}
                  {effectiveActionLabel && effectiveActionUrl && (
                    <button
                      onClick={handleBlockingAction}
                      style={{
                        display: 'block', width: '100%', padding: '12px', background: '#2563eb',
                        color: '#fff', borderRadius: '12px', fontSize: '14px', fontWeight: 700,
                        border: 'none', cursor: 'pointer'
                      }}
                    >
                      {effectiveActionLabel}
                    </button>
                  )}
                  {effectiveErrorCode && (
                    <div style={{ color: '#9a3412', fontSize: '12px', fontWeight: 700 }}>
                      Error code: {effectiveErrorCode}
                    </div>
                  )}
                </div>
              )}

              {isLiveCallState ? (
                <>
                  <div style={{ 
                    width: '120px', height: '120px', borderRadius: '50%', background: '#eff6ff', 
                    display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '32px',
                    position: 'relative',
                    transition: 'transform 0.25s ease, box-shadow 0.25s ease',
                    transform: callState === 'active' ? 'scale(1.03)' : 'scale(1)',
                    boxShadow: callState === 'active' ? '0 18px 30px -24px rgba(37,99,235,0.5)' : '0 0 0 0 rgba(37,99,235,0.18)'
                  }}>
                    {callState === 'answer_required' ? <PhoneIncoming size={48} color="#d97706" /> : <Phone size={48} color="#2563eb" />}
                    {(callState === 'preparing_softphone' || callState === 'connecting' || callState === 'waiting_for_invite' || callState === 'invite_received') && (
                      <div style={{ 
                        position: 'absolute', inset: -10, borderRadius: '50%', 
                        border: '2px solid #2563eb', animation: 'ping 2s cubic-bezier(0, 0, 0.2, 1) infinite' 
                      }} />
                    )}
                  </div>
                  <h3 style={{ fontSize: '24px', fontWeight: 800, color: '#0f172a', marginBottom: '8px' }}>
                    {callState === 'preparing_softphone' ? 'Preparing Browser Softphone...' :
                     callState === 'connecting' ? `Calling ${call.candidate_name?.split(' ')[0] || 'Candidate'}...` :
                     callState === 'waiting_for_invite' ? 'Waiting For Browser Invite...' :
                     callState === 'answer_required' ? 'Answer Browser Call' :
                     callState === 'invite_received' ? 'Joining Browser Call...' :
                     callState === 'active' ? `Active call with ${call.candidate_name || 'Candidate'}` :
                     'Browser VoIP unavailable'}
                  </h3>
                  <p style={{ fontSize: '18px', color: '#64748b', fontWeight: 500, marginBottom: '16px' }}>{call.candidate_phone}</p>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: callStatusMeta.tone, fontSize: '14px', fontWeight: 700, marginBottom: '40px' }}>
                    {callState === 'answer_required' ? <PhoneIncoming size={14} /> : <RefreshCw size={14} style={{ animation: (callState === 'active' || callState === 'error') ? 'none' : 'spin 2s linear infinite' }} />}
                    {callState === 'preparing_softphone' ? 'Registering softphone' :
                     callState === 'connecting' ? 'Initiating call' :
                     callState === 'waiting_for_invite' ? 'Waiting for browser invite' :
                     callState === 'answer_required' ? 'Invite received' :
                     callState === 'invite_received' ? 'Connecting browser audio' :
                     callState === 'active' ? 'Connected' :
                     'Needs attention'}
                  </div>

                  {callState === 'answer_required' ? (
                    <div style={{ display: 'flex', gap: '12px', width: '100%' }}>
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
                    <div style={{ display: 'flex', gap: '12px', width: '100%' }}>
                      {effectiveActionLabel && effectiveActionUrl && (
                        <button
                          onClick={handleBlockingAction}
                          style={{
                            flex: 1,
                            padding: '16px',
                            background: '#2563eb',
                            color: '#fff',
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
                          {effectiveActionLabel}
                        </button>
                      )}
                      <button
                        onClick={handleRetryVoip}
                        style={{
                          flex: 1,
                          padding: '16px',
                          background: '#eff6ff',
                          color: '#2563eb',
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
                        <RefreshCw size={20} /> Retry VoIP
                      </button>
                      <button
                        onClick={onClose}
                        style={{
                          flex: 1,
                          padding: '16px',
                          background: '#fff',
                          color: '#334155',
                          border: '1px solid #cbd5e1',
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
                </>
              ) : callState === 'ended' ? (
                <div style={{ width: '100%' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px', padding: '12px 16px', background: '#f8fafc', borderRadius: '12px' }}>
                    <PhoneCall size={18} color="#2563eb" />
                    <span style={{ fontSize: '15px', fontWeight: 700, color: '#0f172a' }}>Log Call Details</span>
                  </div>

                  <div style={{ marginBottom: '24px' }}>
                    <label style={{ display: 'block', fontSize: '12px', fontWeight: 700, color: '#94a3b8', marginBottom: '8px', textTransform: 'uppercase' }}>Outcome / Stage</label>
                    <select 
                      value={outcome}
                      onChange={e => setOutcome(e.target.value)}
                      style={{ width: '100%', padding: '12px 16px', borderRadius: '12px', border: '1.5px solid #e2e8f0', fontSize: '14px', outline: 'none' }}
                    >
                      <option value="">Select an outcome...</option>
                      {OUTCOMES.map(o => <option key={o} value={o}>{o}</option>)}
                    </select>
                  </div>

                  {['Left Voicemail', 'No Answer'].includes(outcome) && !createFollowup && (
                    <div style={{ padding: '12px', background: '#eff6ff', color: '#1e40af', borderRadius: '12px', fontSize: '13px', marginBottom: '24px', border: '1px solid #bfdbfe', display: 'flex', gap: '8px', alignItems: 'flex-start' }}>
                      <Calendar size={16} style={{ marginTop: '2px', flexShrink: 0 }} />
                      <div>
                        <strong style={{ display: 'block', marginBottom: '4px' }}>Automated Call Sequence</strong>
                        Because the call went unanswered, the next step in the Call 1 → Day 2 → Day 4 → Day 7 sequence will be scheduled automatically when you save.
                      </div>
                    </div>
                  )}

                  <div style={{ marginBottom: '24px' }}>
                    <label style={{ display: 'block', fontSize: '12px', fontWeight: 700, color: '#94a3b8', marginBottom: '8px', textTransform: 'uppercase' }}>Notes & Next Steps</label>
                    <textarea 
                      placeholder="Summarize the call and note any next steps..."
                      value={notes}
                      onChange={e => setNotes(e.target.value)}
                      style={{ width: '100%', padding: '12px 16px', borderRadius: '12px', border: '1.5px solid #e2e8f0', fontSize: '14px', minHeight: '100px', resize: 'none' }}
                    />
                  </div>

                  <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer', marginBottom: '32px' }}>
                    <input 
                      type="checkbox" 
                      checked={createFollowup}
                      onChange={e => setCreateFollowup(e.target.checked)}
                    />
                    <span style={{ fontSize: '14px' }}>Create a follow-up task</span>
                  </label>

                  {createFollowup && (
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 180px', gap: '12px', marginBottom: '32px' }}>
                      <div>
                        <label style={{ display: 'block', fontSize: '12px', fontWeight: 700, color: '#94a3b8', marginBottom: '8px' }}>Task Title</label>
                        <input type="text" value={followupTitle} onChange={e => setFollowupTitle(e.target.value)} style={{ width: '100%', padding: '12px', borderRadius: '12px', border: '1.5px solid #e2e8f0' }} />
                      </div>
                      <div>
                        <label style={{ display: 'block', fontSize: '12px', fontWeight: 700, color: '#94a3b8', marginBottom: '8px' }}>Due Date</label>
                        <input type="date" value={followupDueDate} onChange={e => setFollowupDueDate(e.target.value)} min={new Date().toISOString().split('T')[0]} style={{ width: '100%', padding: '12px', borderRadius: '12px', border: '1.5px solid #e2e8f0' }} />
                      </div>
                    </div>
                  )}

                  <div style={{ display: 'flex', gap: '12px' }}>
                    <button style={{ flex: 1, padding: '14px', background: '#fff', border: '1.5px solid #e2e8f0', borderRadius: '12px', fontWeight: 700, cursor: 'pointer' }} onClick={handleEndCall}>Cancel</button>
                    <button onClick={handleSaveLog} disabled={saving} style={{ flex: 1, padding: '14px', background: '#2563eb', color: '#fff', border: 'none', borderRadius: '12px', fontWeight: 700, cursor: saving ? 'not-allowed' : 'pointer' }}>
                      {saving ? 'Saving...' : 'Save Call Log'}
                    </button>
                  </div>
                </div>
              ) : (
                <div style={{ width: '100%' }}>
                  <div style={{ padding: '20px', background: '#f8fafc', borderRadius: '16px', border: '1px solid #e2e8f0', marginBottom: '24px' }}>
                    <h4 style={{ fontSize: '14px', fontWeight: 800, color: '#0f172a', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <PhoneCall size={16} color="#2563eb" /> Post-Call Analysis
                    </h4>
                    
                    {reviewCallData?.recording_url || reviewCallData?.frejun_link ? (
                      <div style={{ marginBottom: '24px' }}>
                        <div style={{ fontSize: '13px', fontWeight: 700, color: '#64748b', marginBottom: '8px' }}>Recording</div>
                        <audio src={reviewCallData.recording_url || reviewCallData.frejun_link} controls style={{ width: '100%', height: '40px' }} />
                      </div>
                    ) : (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '12px', background: '#eff6ff', borderRadius: '8px', color: '#1e40af', fontSize: '13px', marginBottom: '24px' }}>
                        <RefreshCw size={14} style={{ animation: 'spin 2s linear infinite' }} /> Processing audio recording from FreJun...
                      </div>
                    )}
                    
                    {reviewCallData?.summary && (
                      <div style={{ marginBottom: '24px' }}>
                        <div style={{ fontSize: '13px', fontWeight: 700, color: '#64748b', marginBottom: '8px' }}>AI Summary</div>
                        <p style={{ fontSize: '14px', lineHeight: 1.6, color: '#334155' }}>{reviewCallData.summary}</p>
                      </div>
                    )}

                    <div style={{ marginBottom: '16px' }}>
                      <div style={{ fontSize: '13px', fontWeight: 700, color: '#64748b', marginBottom: '8px' }}>
                        Transcript {(!reviewCallData?.transcript && reviewCallData?.completed_at && (new Date() - new Date(reviewCallData.completed_at)) > 600000) ? '(Fallback AI Triggered)' : ''}
                      </div>
                      {reviewCallData?.transcript ? (
                        <div style={{ background: '#fff', padding: '16px', borderRadius: '12px', border: '1px solid #e2e8f0', maxHeight: '200px', overflowY: 'auto', fontSize: '13px', lineHeight: 1.6, color: '#475569', whiteSpace: 'pre-wrap' }}>
                          {reviewCallData.transcript}
                        </div>
                      ) : (
                        <div style={{ padding: '24px', textAlign: 'center', background: '#fff', borderRadius: '12px', border: '1px dashed #cbd5e1', color: '#94a3b8', fontSize: '13px' }}>
                          {reviewCallData?.completed_at && (new Date() - new Date(reviewCallData.completed_at)) > 600000 
                            ? "FreJun native AI didn't reply in 10 mins. Running OpenAI fallback analysis..."
                            : "Waiting for FreJun AI analysis..."}
                        </div>
                      )}
                    </div>
                  </div>
                  <button onClick={onClose} style={{ width: '100%', padding: '14px', background: '#fff', border: '1px solid #e2e8f0', borderRadius: '12px', fontWeight: 700, color: '#0f172a', cursor: 'pointer' }}>Close Window</button>
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
