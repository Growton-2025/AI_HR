import { useEffect, useState } from 'react'
import axios from 'axios'
import { Loader2, Plus, X } from 'lucide-react'
import { toast } from 'sonner'
import { API_BASE } from '../store/useAppStore'

const DEFAULT_SUBJECT = 'A {{role_name}} opportunity for you'
const DEFAULT_BODY = `Hi {{first_name}},

I came across your profile and thought your experience could be a strong fit for our {{role_name}} opportunity.

Would you be open to a brief conversation?

Best regards`

export default function RoleCreateModal({ role = null, onClose, onSubmit }) {
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [accounts, setAccounts] = useState([])
  const [form, setForm] = useState({
    name: role?.name || '', heyreach_campaign_id: '', smartlead_sender_account_id: '',
    email_subject: DEFAULT_SUBJECT, email_body: DEFAULT_BODY,
  })

  useEffect(() => {
    let cancelled = false
    Promise.all([
      axios.get(`${API_BASE}/outreach/smartlead/email-accounts`),
      role?.id ? axios.get(`${API_BASE}/roles/id/${role.id}/activation`) : Promise.resolve({ data: {} }),
    ])
      .then(([accountRes, activationRes]) => {
        if (cancelled) return
        setAccounts(accountRes.data?.accounts || [])
        const saved = activationRes.data || {}
        setForm(current => ({
          ...current,
          name: role?.name || current.name,
          heyreach_campaign_id: saved.heyreach_campaign_id || '',
          smartlead_sender_account_id: saved.smartlead_sender_account_id || '',
          email_subject: saved.email_subject || current.email_subject,
          email_body: saved.email_body || current.email_body,
        }))
      })
      .catch(error => {
        if (cancelled) return
        toast.error(error.response?.data?.detail || 'Could not load Smartlead senders')
      })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [])

  const submit = async () => {
    setSubmitting(true)
    try {
      const result = await onSubmit({
        ...form,
        name: form.name.trim(),
        heyreach_campaign_id: Number(form.heyreach_campaign_id),
        smartlead_sender_account_id: Number(form.smartlead_sender_account_id),
        email_subject: form.email_subject.trim(),
        email_body: form.email_body.trim(),
      })
      if (!result?.success) throw new Error(result?.error || (role ? 'Activation failed' : 'Failed to create role'))
      onClose()
    } catch (error) {
      toast.error(error.message || (role ? 'Activation failed' : 'Failed to create role'))
    } finally {
      setSubmitting(false)
    }
  }

  const invalid = !form.name.trim() || !Number(form.heyreach_campaign_id)
    || !form.smartlead_sender_account_id || !form.email_subject.trim() || !form.email_body.trim()

  return <div className="modal-overlay" onClick={onClose}>
    <div className="modal-content" style={{ maxWidth: 720, maxHeight: '90vh', padding: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column' }} onClick={event => event.stopPropagation()}>
      <div style={{ padding: '20px 22px', borderBottom: '1px solid #e2e8f0', display: 'flex', justifyContent: 'space-between', flexShrink: 0 }}>
        <div><h3 className="modal-title" style={{ margin: 0 }}>{role ? `Activate ${role.name}` : 'Create and activate role'}</h3><p style={{ margin: '5px 0 0', color: '#64748b', fontSize: 12 }}>Smartlead creates an empty campaign named after this role. HeyReach links the existing campaign ID.</p></div>
        <button className="icon-btn" onClick={onClose}><X size={18} /></button>
      </div>
      <div style={{ padding: 22, display: 'grid', gap: 15, overflowY: 'auto', minHeight: 0 }}>
        <label>Role name<input autoFocus={!role} disabled={Boolean(role)} className="input-field" value={form.name} onChange={e => setForm(f => ({ ...f, name: e.target.value }))} placeholder="Senior Sales Director" style={{ width: '100%', marginTop: 6 }} /></label>
        <label>HeyReach campaign ID<input className="input-field" type="text" inputMode="numeric" pattern="[0-9]*" value={form.heyreach_campaign_id} onChange={e => setForm(f => ({ ...f, heyreach_campaign_id: e.target.value.replace(/\D/g, '') }))} placeholder="Paste campaign ID" style={{ width: '100%', marginTop: 6 }} /></label>
        <label>Smartlead sender<select className="input-field" disabled={loading} value={form.smartlead_sender_account_id} onChange={e => setForm(f => ({ ...f, smartlead_sender_account_id: e.target.value }))} style={{ width: '100%', marginTop: 6 }}>
          <option value="">{loading ? 'Loading connected senders…' : 'Select a connected sender…'}</option>
          {accounts.map(account => <option key={account.id} value={account.id} disabled={!account.connected}>{account.email}{account.name ? ` · ${account.name}` : ''}{account.connected ? '' : ' · disconnected'}</option>)}
        </select></label>
        <label>Email subject<input className="input-field" maxLength={500} value={form.email_subject} onChange={e => setForm(f => ({ ...f, email_subject: e.target.value }))} style={{ width: '100%', marginTop: 6 }} /></label>
        <label>Email body<textarea className="input-field" rows={8} maxLength={20000} value={form.email_body} onChange={e => setForm(f => ({ ...f, email_body: e.target.value }))} style={{ width: '100%', marginTop: 6, resize: 'vertical' }} /></label>
      </div>
      <div className="modal-footer" style={{ margin: 0, padding: '16px 22px', borderTop: '1px solid #e2e8f0', flexShrink: 0, background: 'var(--surface, #fff)' }}>
        <button className="btn btn-secondary" onClick={onClose}>Cancel</button>
        <button className="btn btn-primary" disabled={loading || submitting || invalid} onClick={submit}>{submitting ? <Loader2 size={15} className="animate-spin" /> : <Plus size={15} />} {role ? 'Save and activate' : 'Create role'}</button>
      </div>
    </div>
  </div>
}
