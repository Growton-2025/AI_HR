import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Linkedin, Loader2, Mail, MessageSquare, RefreshCcw, Send, X } from 'lucide-react'
import { toast } from 'sonner'
import { useAppStore } from '../store/useAppStore'
import StatusDropdown from './StatusDropdown'

const EMPTY_THREAD = { messages: [], loaded: false, error: '' }

function messageBody(message) {
  return message?.email_body || message?.text || message?.message || message?.content || message?.html_body || message?.body || ''
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
    const result = await fetchChatHistory(roleId, candidate.id, targetPlatform, force)
    setThreads(previous => ({
      ...previous,
      [targetPlatform]: result.success
        ? { messages: result.messages || [], loaded: true, error: '', syncing: Boolean(result.syncing) }
        : { messages: [], loaded: true, error: result.error || 'Failed to load conversation', syncing: false },
    }))
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

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

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

  const tabs = useMemo(() => ([
    { id: 'linkedin', label: 'LinkedIn', icon: Linkedin, hasResponse: Boolean(candidate.li_response_text) },
    { id: 'email', label: 'Email', icon: Mail, hasResponse: Boolean(candidate.response) },
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
          {!activeThread.loaded ? (
            <div className="candidate-conversation-empty">
              <Loader2 size={32} className="animate-spin" />
              <strong>Loading latest messages…</strong>
              <span>This usually takes a few seconds</span>
            </div>
          ) : activeThread.error ? (
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
                  {message._pending && <small>Sending…</small>}
                </div>
              )
            })
          )}
          <div ref={endRef} />
        </main>

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
      </div>
    </div>
  )
}
