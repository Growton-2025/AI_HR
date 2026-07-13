import { useEffect, useState } from 'react'
import axios from 'axios'
import { Loader2, Save, X } from 'lucide-react'
import { toast } from 'sonner'
import { API_BASE } from '../store/useAppStore'

export default function RoleEmailSendModal({ roleId, roleName, onClose, onSaved }) {
  const [setup, setSetup] = useState({ sender_account_id: '', campaign_id: '' })
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
      setSetup({
        sender_account_id: String(value.sender_account_id || ''),
        campaign_id: String(value.campaign_id || ''),
      })
      setAccounts(accountRes.data?.accounts || [])
    }).catch(error => {
      toast.error(error.response?.data?.detail || 'Could not load Smartlead setup')
      onClose()
    }).finally(() => !cancelled && setBusy(false))
    return () => { cancelled = true }
  }, [roleId])

  const save = async () => {
    setBusy(true)
    try {
      const res = await axios.put(`${API_BASE}/outreach/roles/${roleId}/email-setup`, {
        sender_account_id: Number(setup.sender_account_id),
        campaign_id: Number(setup.campaign_id),
      })
      toast.success(`Email setup saved for ${roleName}`)
      onSaved?.(res.data)
      onClose()
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Could not save Smartlead setup')
    } finally {
      setBusy(false)
    }
  }

  const selectedAccount = accounts.find(account => String(account.id) === String(setup.sender_account_id))
  const invalid = !setup.sender_account_id

  return <div className="modal-overlay" onClick={() => !busy && onClose()}>
    <div className="modal-content" style={{ maxWidth: 680, padding: 0, overflow: 'hidden' }} onClick={event => event.stopPropagation()}>
      <div style={{ padding: '20px 22px', borderBottom: '1px solid #e2e8f0', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h3 className="modal-title" style={{ margin: 0 }}>Email outreach setup</h3>
          <p style={{ color: '#64748b', fontSize: 12, marginTop: 5 }}>Saved for {roleName}. Links an existing Smartlead campaign — shortlisted candidates are pushed to it automatically.</p>
        </div>
        <button className="icon-btn" onClick={onClose} disabled={busy} aria-label="Close"><X size={18} /></button>
      </div>

      <div style={{ padding: 22 }}>
        {busy && accounts.length === 0 ? <div style={{ minHeight: 180, display: 'grid', placeItems: 'center' }}><Loader2 className="animate-spin" /></div> : <div style={{ display: 'grid', gap: 16 }}>
          <label style={{ fontSize: 12, fontWeight: 700, color: '#334155' }}>From email
            <select className="input-field" value={setup.sender_account_id} onChange={event => setSetup(current => ({ ...current, sender_account_id: event.target.value }))} style={{ width: '100%', marginTop: 7 }}>
              <option value="">Select a connected sender…</option>
              {accounts.map(account => <option key={account.id} value={account.id} disabled={!account.connected}>
                {account.email}{account.name ? ` · ${account.name}` : ''}{account.connected ? '' : ' · disconnected'}
              </option>)}
            </select>
            {selectedAccount?.warmup_status && <span style={{ display: 'block', marginTop: 5, color: '#64748b', fontSize: 11 }}>Warmup: {selectedAccount.warmup_status}</span>}
          </label>

          <label style={{ fontSize: 12, fontWeight: 700, color: '#334155' }}>Smartlead campaign ID
            <input className="input-field" type="text" inputMode="numeric" pattern="[0-9]*" value={setup.campaign_id} onChange={event => setSetup(current => ({ ...current, campaign_id: event.target.value.replace(/\D/g, '') }))} placeholder="Optional — leave blank to auto-create" style={{ width: '100%', marginTop: 7 }} />
            <span style={{ display: 'block', marginTop: 5, color: '#64748b', fontSize: 11 }}>Paste an ID to link an existing campaign, or leave blank to create one named after this role. The sequence and content are configured inside Smartlead.</span>
          </label>
        </div>}
      </div>

      <div className="modal-footer" style={{ margin: 0, padding: '16px 22px', borderTop: '1px solid #e2e8f0', background: '#f8fafc' }}>
        <button className="btn btn-secondary" disabled={busy} onClick={onClose}>Cancel</button>
        <button className="btn btn-primary" disabled={busy || invalid} onClick={save}>{busy ? <Loader2 size={15} className="animate-spin" /> : <Save size={15} />} Save setup</button>
      </div>
    </div>
  </div>
}
