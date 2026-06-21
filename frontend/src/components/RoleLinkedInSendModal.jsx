import { useEffect, useState } from 'react'
import axios from 'axios'
import { Linkedin, Loader2 } from 'lucide-react'
import { toast } from 'sonner'
import { API_BASE } from '../store/useAppStore'

export default function RoleLinkedInSendModal({ roleId, candidate, onClose, onScheduled }) {
  const [busy, setBusy] = useState(true)
  const [setup, setSetup] = useState({ message: '', campaign_id: '', configured: false })

  useEffect(() => {
    let active = true
    ;(async () => {
      try {
        const setupRes = await axios.get(`${API_BASE}/outreach/roles/${roleId}/linkedin-setup`)
        const saved = setupRes.data || {}
        if (!active) return
        setSetup({ message: saved.message || saved.follow_up_message || '', campaign_id: saved.campaign_id || '', configured: Boolean(saved.configured) })
      } catch (error) {
        toast.error(error.response?.data?.detail || 'Could not load LinkedIn setup'); onClose()
      } finally { if (active) setBusy(false) }
    })()
    return () => { active = false }
  }, [roleId])

  const confirm = async () => {
    setBusy(true)
    try {
      if (!setup.configured) await axios.put(`${API_BASE}/outreach/roles/${roleId}/linkedin-setup`, setup)
      const res = await axios.post(`${API_BASE}/outreach/roles/${roleId}/candidates/${candidate.id}/send-linkedin`)
      toast.success(res.data?.already_scheduled ? 'LinkedIn outreach was already scheduled' : 'LinkedIn outreach scheduled for 3 minutes')
      onScheduled?.(); onClose()
    } catch (error) { toast.error(error.response?.data?.detail || 'Could not schedule LinkedIn outreach') }
    finally { setBusy(false) }
  }

  return <div className="modal-overlay" onClick={() => !busy && onClose()}>
    <div className="modal-content" style={{ maxWidth: 620 }} onClick={e => e.stopPropagation()}>
      <h3 className="modal-title">LinkedIn outreach to {candidate.name}</h3>
      <p style={{ color: '#64748b', fontSize: 13 }}>{setup.configured ? 'Uses the saved role setup and dispatches in 3 minutes.' : 'Configure this role once. Only this candidate is scheduled.'}</p>
      {!setup.configured && <div style={{ display: 'grid', gap: 14 }}>
        <label>Message<textarea className="input-field" rows={7} maxLength={8000} value={setup.message} onChange={e => setSetup(s => ({ ...s, message: e.target.value }))} placeholder="Hi {{firstName}},…" /></label>
      </div>}
      {setup.configured && setup.campaign_id && <div style={{ fontSize: 12, color: '#64748b' }}>Campaign ID: <strong>{setup.campaign_id}</strong></div>}
      <div className="modal-footer"><button className="btn btn-secondary" disabled={busy} onClick={onClose}>Cancel</button><button className="btn btn-primary" disabled={busy || (!setup.configured && !setup.message.trim())} onClick={confirm}>{busy ? <Loader2 size={15} className="animate-spin" /> : <Linkedin size={15} />} Confirm · 3 min</button></div>
    </div>
  </div>
}
