import React, { useEffect, useState } from 'react'
import axios from 'axios'
import { Loader2, Send } from 'lucide-react'
import { toast } from 'sonner'
import { API_BASE } from '../store/useAppStore'

export default function RoleEmailSendModal({ roleId, candidate, onClose, onSent }) {
  const [setup, setSetup] = useState({ sender_account_id: '', sender_email: '', subject: '', initial_body: '', configured: false })
  const [accounts, setAccounts] = useState([])
  const [busy, setBusy] = useState(true)

  useEffect(() => {
    let cancelled = false
    Promise.all([
      axios.get(`${API_BASE}/outreach/roles/${roleId}/email-setup`),
      axios.get(`${API_BASE}/outreach/smartlead/email-accounts`),
    ]).then(([setupRes, accountRes]) => {
      if (cancelled) return
      const value = setupRes.data || {}
      setSetup({ sender_account_id: value.sender_account_id || '', sender_email: value.sender_email || '', subject: value.subject || '', initial_body: value.initial_body || '', configured: Boolean(value.configured) })
      setAccounts(accountRes.data?.accounts || [])
    }).catch(error => {
      toast.error(error.response?.data?.detail || 'Could not load campaign setup')
      onClose()
    }).finally(() => !cancelled && setBusy(false))
    return () => { cancelled = true }
  }, [roleId, onClose])

  const confirm = async () => {
    setBusy(true)
    try {
      if (!setup.configured) await axios.put(`${API_BASE}/outreach/roles/${roleId}/email-setup`, setup)
      const res = await axios.post(`${API_BASE}/outreach/roles/${roleId}/candidates/${candidate.id}/send-email`)
      toast.success(res.data?.already_enrolled ? 'Candidate was already enrolled' : 'Candidate enrolled in the role campaign')
      onSent?.()
      onClose()
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Could not send email')
    } finally { setBusy(false) }
  }

  return <div className="modal-overlay" onClick={() => !busy && onClose()}>
    <div className="modal-content" style={{ maxWidth: 620 }} onClick={event => event.stopPropagation()}>
      <h3 className="modal-title">Send to {candidate.name}</h3>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 18 }}>{setup.configured ? 'Explicitly confirm this candidate using the saved role setup.' : 'Configure this role once; later candidates reuse these settings.'}</p>
      {busy ? <Loader2 className="animate-spin" /> : !setup.configured && <div style={{ display: 'grid', gap: 14 }}>
        <label style={{ fontSize: 12, fontWeight: 700 }}>From email
          <select className="input-field" value={setup.sender_account_id} onChange={event => {
            const account = accounts.find(item => String(item.id) === event.target.value)
            setSetup(current => ({ ...current, sender_account_id: event.target.value, sender_email: account?.email || '' }))
          }} style={{ width: '100%', marginTop: 6 }}>
            <option value="">Select a connected sender…</option>
            {accounts.map(account => <option key={account.id} value={account.id}>{account.email}{account.name ? ` · ${account.name}` : ''}</option>)}
          </select>
        </label>
        <label style={{ fontSize: 12, fontWeight: 700 }}>Subject<input className="input-field" value={setup.subject} onChange={event => setSetup(current => ({ ...current, subject: event.target.value }))} style={{ width: '100%', marginTop: 6 }} /></label>
        <label style={{ fontSize: 12, fontWeight: 700 }}>Initial email body<textarea className="input-field" rows={9} value={setup.initial_body} onChange={event => setSetup(current => ({ ...current, initial_body: event.target.value }))} style={{ width: '100%', marginTop: 6, resize: 'vertical' }} placeholder="Hi {{first_name}},…" /></label>
      </div>}
      <div className="modal-footer">
        <button className="btn btn-secondary" disabled={busy} onClick={onClose}>Cancel</button>
        <button className="btn btn-primary" disabled={busy || !setup.sender_account_id || !setup.subject.trim() || !setup.initial_body.trim()} onClick={confirm}>{busy ? <Loader2 size={15} className="animate-spin" /> : <Send size={15} />} Confirm send</button>
      </div>
    </div>
  </div>
}
