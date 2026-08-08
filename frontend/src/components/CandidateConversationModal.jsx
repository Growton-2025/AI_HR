import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Linkedin, Loader2, Mail, MessageSquare, Phone, RefreshCcw, Send, X } from 'lucide-react'
import { toast } from 'sonner'
import { useAppStore } from '../store/useAppStore'
import StatusDropdown from './StatusDropdown'
import CandidateActivityPanel from './CandidateActivityPanel'

const EMPTY_THREAD = { messages: [], loaded: false, error: '' }

// Detects whether a stored body is HTML (email clients send HTML with <br>,
// <p>, entities) versus plain text. Deliberately narrow so plain-text bodies
// that merely contain a stray "<" (e.g. "salary < 10L") aren't misclassified.
function looksLikeHtml(str) {
  return /<\s*(br|p|div|span|a|strong|b|i|em|u|ul|ol|li|table|tr|td|blockquote|h[1-6])\b|<\/\s*[a-z]+\s*>|&(nbsp|amp|lt|gt|quot|#39|apos);/i.test(str)
}

function decodeEntities(str) {
  return str
    .replace(/&nbsp;/gi, ' ')
    .replace(/&lt;/gi, '<')
    .replace(/&gt;/gi, '>')
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'")
    .replace(/&apos;/gi, "'")
    .replace(/&amp;/gi, '&')
}

// Convert an HTML email body to plain text with real newlines so the same
// pre-wrap rendering path works for both HTML and plain-text messages. This
// avoids literal "<br><br>" leaking into the UI and is XSS-safe (no HTML is
// injected into the DOM).
function htmlToText(str) {
  return str
    .replace(/<\s*br\s*\/?\s*>/gi, '\n')
    .replace(/<\s*\/\s*(p|div|li|tr|h[1-6]|blockquote)\s*>/gi, '\n')
    .replace(/<[^>]+>/g, '')
    .replace(/\n{3,}/g, '\n\n')
}

function messageBody(message) {
  const raw = message?.email_body || message?.text || message?.message || message?.content || message?.html_body || message?.body || ''
  if (!raw || typeof raw !== 'string') return raw || ''
  if (!looksLikeHtml(raw)) return raw
  return decodeEntities(htmlToText(raw)).trim()
}

function messageTime(message) {
  const value = message?.time || message?.created_at || message?.timestamp
  if (!value) return ''
  const parsed = new Date(value)
  return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toLocaleString()
}

function isIncomingMessage(message) {
  if (message?.direction === 'inbound') return true
  if (message?.direction === 'outbound') return false
  return ['INBOX', 'REPLY', 'REPLIED', 'LEAD', 'INCOMING'].includes(String(message?.type || '').toUpperCase())
}

export default function CandidateConversationModal({
  candidate,
  roleId,
  onClose,
  updateStatus,
  onStatusChanged,
}) {
  const defaultPlatform = candidate?.li_response_text || candidate?.li_status === 'replied' ? 'linkedin' : 'email'
  const [platform, setPlatform] = useState(defaultPlatform)
  const [threads, setThreads] = useState({ email: EMPTY_THREAD, linkedin: EMPTY_THREAD })
  const [replyText, setReplyText] = useState('')
  const [sending, setSending] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const endRef = useRef(null)
  const fetchChatHistory = useAppStore(state => state.fetchChatHistory)
  const sendChatReply = useAppStore(state => state.sendChatReply)

  const loadThread = useCallback(async (targetPlatform, force = false) => {
    // Email/LinkedIn threads are role-scoped. Opened without a role — e.g. from
    // an inbound callback, which belongs to a person rather than a role — the
    // request would 422 rather than return anything useful, so say so instead.
    if (!roleId) {
      setThreads(previous => ({
        ...previous,
        [targetPlatform]: {
          messages: [], loaded: true, syncing: false,
          error: 'Outreach history is shown per role — open this candidate from a role to see it.',
        },
      }))
      return { success: false }
    }
    const result = await fetchChatHistory(roleId, candidate.id, targetPlatform, force)
    setThreads(previous => {
      // A just-sent message takes the provider a while to ingest — a refetch
      // that replaces the array wholesale made the sent bubble VANISH for
      // minutes. Carry optimistic messages forward until the fetched thread
      // actually contains them (matched by body).
      const fetched = result.messages || []
      const prevMessages = previous[targetPlatform]?.messages || []
      const stillPending = prevMessages.filter(message =>
        message._pending && !fetched.some(f =>
          String(f.email_body || '').trim() === String(message.email_body || '').trim()
        )
      )
      return {
        ...previous,
        [targetPlatform]: result.success
          ? { messages: [...fetched, ...stillPending], loaded: true, error: '', syncing: Boolean(result.syncing) }
          : {
              messages: prevMessages, loaded: true, syncing: false,
              // Coerce: some failures return a structured payload, and this value
              // is rendered directly.
              error: typeof result.error === 'string' && result.error
                ? result.error
                : 'Failed to load conversation',
            },
      }
    })
    return result
  }, [candidate.id, fetchChatHistory, roleId])

  useEffect(() => {
    setPlatform(defaultPlatform)
    setThreads({ email: EMPTY_THREAD, linkedin: EMPTY_THREAD })
    setReplyText('')
    void loadThread(defaultPlatform)
    void loadThread(defaultPlatform === 'email' ? 'linkedin' : 'email')
  }, [candidate.id, defaultPlatform, loadThread])

  const activeThread = threads[platform] || EMPTY_THREAD
  const messages = activeThread.messages || []

  // Only scroll when a message is actually added — the steady poll below
  // replaces the array every cycle, and yanking the recruiter back to the
  // bottom while they are reading older messages would be hostile.
  const lastMessageKey = messages.length
    ? `${messages.length}-${messageTime(messages[messages.length - 1])}`
    : ''
  const lastMessageKeyRef = useRef('')
  useEffect(() => {
    if (lastMessageKey && lastMessageKey !== lastMessageKeyRef.current) {
      lastMessageKeyRef.current = lastMessageKey
      endRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [lastMessageKey])

  // Fast poll whichever thread is currently syncing
  useEffect(() => {
    const intervalId = setInterval(() => {
      if (document.visibilityState !== 'visible') return;
      if (threads.email?.syncing) {
        void loadThread('email');
      }
      if (threads.linkedin?.syncing) {
        void loadThread('linkedin');
      }
    }, 2000);

    return () => clearInterval(intervalId);
  }, [threads.email?.syncing, threads.linkedin?.syncing, loadThread]);

  // Steady poll of the open thread so replies that land while the modal is
  // open appear on their own — recruiters shouldn't need "Manual Sync" to see
  // a candidate answer. The backend serves this from its in-memory cache and
  // refreshes itself from HeyReach/Smartlead when stale, so a 15s cadence is
  // cheap on both sides.
  const threadsRef = useRef(threads)
  useEffect(() => { threadsRef.current = threads }, [threads])
  useEffect(() => {
    if (platform === 'calls') return undefined
    const intervalId = setInterval(() => {
      if (document.visibilityState !== 'visible') return
      const thread = threadsRef.current[platform]
      // Errored threads keep polling too — a transient timeout must
      // self-heal, not freeze the modal until Manual Sync.
      if (thread?.loaded && !thread.syncing) void loadThread(platform)
    }, 15000)
    return () => clearInterval(intervalId)
  }, [platform, loadThread])

  // Calls sits alongside the message threads: a recruiter opening a candidate's
  // conversation needs to see they have already been spoken to, not just what
  // was written. Same timeline the Calls workspace shows.
  const tabs = useMemo(() => ([
    { id: 'linkedin', label: 'LinkedIn', icon: Linkedin, hasResponse: Boolean(candidate.li_response_text) },
    { id: 'email', label: 'Email', icon: Mail, hasResponse: Boolean(candidate.response) },
    { id: 'calls', label: 'Calls', icon: Phone },
  ]), [candidate.li_response_text, candidate.response])

  const handleRefresh = async () => {
    setRefreshing(true)
    await Promise.all([loadThread('email', true), loadThread('linkedin', true)])
    setRefreshing(false)
  }

  const handleSend = async () => {
    const text = replyText.trim()
    if (!text || sending) return

    const optimistic = {
      type: 'SENT',
      email_body: text,
      time: new Date().toISOString(),
      sender_name: 'You',
      _pending: true,
    }
    setThreads(previous => ({
      ...previous,
      [platform]: {
        ...previous[platform],
        loaded: true,
        messages: [...(previous[platform]?.messages || []), optimistic],
      },
    }))
    setReplyText('')
    setSending(true)
    const result = await sendChatReply(roleId, candidate.id, text, platform)
    if (result.success) {
      // Delivered to the provider; the bubble stays optimistic (merged into
      // refetches) until the provider's thread includes it, but stop saying
      // "Sending…" — it's sent.
      setThreads(previous => ({
        ...previous,
        [platform]: {
          ...previous[platform],
          messages: (previous[platform]?.messages || []).map(message =>
            message === optimistic ? { ...message, _delivered: true } : message
          ),
        },
      }))
      window.setTimeout(() => void loadThread(platform, true), 1500)
    } else {
      setThreads(previous => ({
        ...previous,
        [platform]: {
          ...previous[platform],
          messages: (previous[platform]?.messages || []).filter(message => message !== optimistic),
        },
      }))
      setReplyText(text)
      toast.error(result.error || 'Failed to send reply')
    }
    setSending(false)
  }

  return (
    <div className="candidate-conversation-overlay" onClick={onClose}>
      <div className="candidate-conversation-modal" onClick={event => event.stopPropagation()}>
        <header className="candidate-conversation-header">
          <div>
            <div className="candidate-conversation-title">
              {candidate.first_name || candidate.name} {candidate.last_name || ''}
              <span>ID: {candidate.id}</span>
            </div>
            <div className="candidate-conversation-sync">
              <MessageSquare size={14} />
              Real-time Sync Active
            </div>
          </div>
          <div className="candidate-conversation-actions">
            <StatusDropdown
              status={candidate.status}
              candidateId={candidate.id}
              updateStatus={updateStatus}
              optimistic
              onUpdate={onStatusChanged}
            />
            <button type="button" className="btn btn-secondary btn-sm" onClick={handleRefresh} disabled={refreshing}>
              <RefreshCcw size={14} className={refreshing ? 'animate-spin' : ''} />
              {refreshing ? 'Syncing…' : 'Manual Sync'}
            </button>
            <button type="button" className="candidate-conversation-close" onClick={onClose}>
              <X size={20} />
            </button>
          </div>
        </header>

        <nav className="candidate-conversation-tabs">
          {tabs.map(tab => (
            <button
              type="button"
              key={tab.id}
              className={platform === tab.id ? 'active' : ''}
              onClick={() => setPlatform(tab.id)}
            >
              <tab.icon size={18} />
              {tab.label}
              {tab.hasResponse && platform !== tab.id && <span className="candidate-conversation-dot" />}
            </button>
          ))}
        </nav>

        <main className="candidate-conversation-messages">
          {platform === 'calls' ? (
            <CandidateActivityPanel
              candidateId={candidate.id}
              candidateName={candidate.first_name || candidate.name}
            />
          ) : !activeThread.loaded ? (
            <div className="candidate-conversation-empty">
              <Loader2 size={32} className="animate-spin" />
              <strong>Loading latest messages…</strong>
              <span>This usually takes a few seconds</span>
            </div>
          ) : activeThread.error && messages.length === 0 ? (
            <div className="candidate-conversation-empty">
              <MessageSquare size={32} />
              <strong>Conversation unavailable</strong>
              <span>{activeThread.error}</span>
            </div>
          ) : messages.length === 0 ? (
            <div className="candidate-conversation-empty">
              <MessageSquare size={32} />
              <strong>No messages yet</strong>
              <span>Start the conversation below</span>
            </div>
          ) : (
            messages.map((message, index) => {
              const incoming = isIncomingMessage(message)
              return (
                <div key={`${messageTime(message)}-${index}`} className={`candidate-message ${incoming ? 'incoming' : 'outgoing'}`}>
                  <div className="candidate-message-meta">
                    {incoming ? (message.sender_name || candidate.first_name || candidate.name) : 'Me'}
                    {messageTime(message) ? ` · ${messageTime(message)}` : ''}
                  </div>
                  <div className="candidate-message-body">{messageBody(message)}</div>
                  {message._pending && !message._delivered && <small>Sending…</small>}
                </div>
              )
            })
          )}
          <div ref={endRef} />
        </main>

        {platform !== 'calls' && (
        <footer className="candidate-conversation-composer">
          <div>
            <textarea
              value={replyText}
              onChange={event => setReplyText(event.target.value)}
              onKeyDown={event => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault()
                  void handleSend()
                }
              }}
              placeholder={`Reply via ${platform === 'linkedin' ? 'LinkedIn' : 'Email'}…`}
              rows={1}
            />
            <button type="button" onClick={handleSend} disabled={!replyText.trim() || sending}>
              {sending ? <Loader2 size={17} className="animate-spin" /> : <Send size={17} />}
            </button>
          </div>
          <span>Press Enter to send · Shift + Enter for new line</span>
        </footer>
        )}
      </div>
    </div>
  )
}
