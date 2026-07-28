import { useState, useEffect } from 'react';
import { useAppStore } from '../store/useAppStore';
import { useShallow } from 'zustand/react/shallow';
import { toast } from 'sonner';

export default function AddToListModal({ selectedCount, onClose, onSuccess, candidateIds }) {
  const { callLists, fetchCallLists, createCallList, addCandidatesToCallList } = useAppStore(useShallow((state) => ({
    callLists: state.callLists,
    fetchCallLists: state.fetchCallLists,
    createCallList: state.createCallList,
    addCandidatesToCallList: state.addCandidatesToCallList,
  })));
  const [loading, setLoading] = useState(false);
  const [newListName, setNewListName] = useState('');
  const [selectedListId, setSelectedListId] = useState('');
  const [mode, setMode] = useState('select'); // 'select' or 'create'

  useEffect(() => {
    fetchCallLists({ force: true });
  }, [fetchCallLists]);

  const handleAction = async () => {
    if (loading) return;
    setLoading(true);
    try {
      let listId = selectedListId;
      if (mode === 'create') {
        if (!newListName.trim()) return;
        const res = await createCallList(newListName.trim());
        if (res.success) listId = res.data.id;
        else throw new Error(res.error);
      }

      if (!listId) return;
      const res = await addCandidatesToCallList(candidateIds, listId);
      if (res.success) {
        // Report the count the backend actually inserted — a concurrent add can
        // make it differ from the local selection size.
        const addedCount = Number(res.data?.added_count ?? selectedCount);
        // Candidates with no phone number are refused server-side; say so
        // explicitly rather than letting the list quietly come up short.
        const skippedNoPhone = Number(res.data?.skipped_no_phone ?? 0);
        if (addedCount > 0) {
          const added = `${addedCount} candidate${addedCount === 1 ? '' : 's'} added to call list`;
          if (skippedNoPhone > 0) {
            toast.success(added, {
              description: `${skippedNoPhone} not added — no phone number available.`,
            });
          } else {
            toast.success(added);
          }
        } else if (skippedNoPhone > 0) {
          toast.warning(
            `No candidates added — ${skippedNoPhone} had no phone number available.`
          );
        } else {
          toast.info('Selected candidates are already in this call list');
        }
        onSuccess();
      } else {
        toast.error(res.error || 'Failed to add candidates to list');
      }
    } catch (e) {
      toast.error(e.message || 'Failed to add candidates to list');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="add-to-list-overlay" style={{ position: 'fixed', inset: 0, background: 'rgba(15, 23, 42, 0.7)', backdropFilter: 'blur(4px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 10000 }}>
      <div className="add-to-list-card" style={{ background: '#fff', borderRadius: '24px', width: '100%', maxWidth: '440px', padding: '32px', boxShadow: '0 25px 50px -12px rgba(0,0,0,0.25)', border: '1px solid #e2e8f0' }}>
        <h3 style={{ fontSize: '20px', fontWeight: 800, color: '#0f172a', marginBottom: '8px' }}>Add to Call List</h3>
        <p style={{ color: '#64748b', fontSize: '14px', marginBottom: '24px' }}>Choose a list to add {selectedCount} candidates to.</p>

        <div style={{ display: 'flex', gap: 8, marginBottom: 20, padding: 4, background: '#f8fafc', borderRadius: 12 }}>
          <button
            onClick={() => !loading && setMode('select')}
            style={{
              flex: 1, padding: '8px', border: 'none', borderRadius: 8, fontSize: 13, fontWeight: 700,
              background: mode === 'select' ? '#fff' : 'transparent',
              color: mode === 'select' ? '#0f172a' : '#64748b',
              boxShadow: mode === 'select' ? '0 1px 3px rgba(0,0,0,0.1)' : 'none',
              cursor: loading ? 'wait' : 'pointer'
            }}
          >Existing List</button>
          <button
            onClick={() => !loading && setMode('create')}
            style={{
              flex: 1, padding: '8px', border: 'none', borderRadius: 8, fontSize: 13, fontWeight: 700,
              background: mode === 'create' ? '#fff' : 'transparent',
              color: mode === 'create' ? '#0f172a' : '#64748b',
              boxShadow: mode === 'create' ? '0 1px 3px rgba(0,0,0,0.1)' : 'none',
              cursor: loading ? 'wait' : 'pointer'
            }}
          >+ New List</button>
        </div>

        {mode === 'select' ? (
          <select
            value={selectedListId}
            onChange={e => setSelectedListId(e.target.value)}
            disabled={loading}
            style={{
              width: '100%', padding: '12px 16px', borderRadius: 12, border: '1.5px solid #e2e8f0',
              fontSize: 14, outline: 'none', marginBottom: 24, background: '#fff'
            }}
          >
            <option value="">Select a list...</option>
            {callLists.map(l => (
              <option key={l.id} value={l.id} disabled={Boolean(l.is_pending)}>
                {l.name} ({l.is_pending ? 'saving...' : `${l.candidate_count} candidates`})
              </option>
            ))}
          </select>
        ) : (
          <input
            type="text"
            placeholder="List name (e.g. Frontend Devs Today)"
            value={newListName}
            onChange={e => setNewListName(e.target.value)}
            disabled={loading}
            style={{
              width: '100%', padding: '12px 16px', borderRadius: 12, border: '1.5px solid #e2e8f0',
              fontSize: 14, outline: 'none', marginBottom: 24, boxSizing: 'border-box'
            }}
          />
        )}

        <div style={{ display: 'flex', gap: 12 }}>
          <button
            onClick={onClose}
            disabled={loading}
            style={{ flex: 1, padding: '14px', background: '#f1f5f9', color: '#475569', border: 'none', borderRadius: 12, fontWeight: 700, cursor: loading ? 'wait' : 'pointer', opacity: loading ? 0.7 : 1 }}
          >Cancel</button>
          <button
            onClick={handleAction}
            disabled={loading || (mode === 'select' && !selectedListId) || (mode === 'create' && !newListName.trim())}
            style={{
              flex: 1, padding: '14px', background: '#f97316', color: '#fff', border: 'none', borderRadius: 12,
              fontWeight: 700, cursor: (loading || (mode === 'select' && !selectedListId) || (mode === 'create' && !newListName.trim())) ? 'not-allowed' : 'pointer',
              opacity: (loading || (mode === 'select' && !selectedListId) || (mode === 'create' && !newListName.trim())) ? 0.6 : 1
            }}
          >
            {loading ? 'Adding...' : 'Add Candidates'}
          </button>
        </div>
      </div>
    </div>
  );
}
