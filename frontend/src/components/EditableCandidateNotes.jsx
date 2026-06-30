import { useEffect, useRef, useState } from 'react'
import { X } from 'lucide-react'
import { useAppStore } from '../store/useAppStore'

export default function EditableCandidateNotes({ candidateId, initialNotes = '' }) {
  const [notes, setNotes] = useState(initialNotes || '')
  const [isEditing, setIsEditing] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const savingRef = useRef(false)
  const updateCandidateNotes = useAppStore(state => state.updateCandidateNotes)

  useEffect(() => {
    setNotes(initialNotes || '')
  }, [initialNotes])

  const handleSave = async () => {
    if (savingRef.current) return
    if (notes === (initialNotes || '')) {
      setIsEditing(false)
      return
    }
    savingRef.current = true
    setIsSaving(true)
    setIsEditing(false)
    const result = await updateCandidateNotes(candidateId, notes)
    savingRef.current = false
    setIsSaving(false)
    if (!result.success) setIsEditing(true)
  }

  return (
    <>
      <button
        type="button"
        onClick={() => setIsEditing(true)}
        title={initialNotes || 'Add candidate notes'}
        className="candidate-notes-trigger"
      >
        {initialNotes || 'Add notes'}
      </button>

      {isEditing && (
        <div className="modal-overlay" onClick={() => !isSaving && setIsEditing(false)}>
          <div className="candidate-notes-modal" onClick={event => event.stopPropagation()}>
            <div className="candidate-notes-modal-header">
              <strong>Candidate Notes</strong>
              <button type="button" className="icon-btn" disabled={isSaving} onClick={() => setIsEditing(false)}>
                <X size={18} />
              </button>
            </div>
            <div className="candidate-notes-modal-body">
              <textarea
                autoFocus
                value={notes}
                maxLength={5000}
                onChange={event => setNotes(event.target.value)}
                placeholder="Add notes about this candidate..."
                rows={6}
              />
              <span>{notes.length} / 5000</span>
            </div>
            <div className="candidate-notes-modal-footer">
              <button type="button" className="btn btn-secondary" disabled={isSaving} onClick={() => setIsEditing(false)}>
                Cancel
              </button>
              <button type="button" className="btn btn-primary" disabled={isSaving} onClick={handleSave}>
                {isSaving ? 'Saving…' : 'Save Notes'}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
