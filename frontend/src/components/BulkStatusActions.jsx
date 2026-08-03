import { useState } from 'react';
import { Check, ChevronDown } from 'lucide-react';
import { toast } from 'sonner';
import { useAppStore } from '../store/useAppStore';
import { RECRUITMENT_STAGES } from './StatusDropdown';

// Mirrors TERMINAL_CANDIDATE_STATUSES in backend/api/routes/calls.py. Setting
// one of these also completes the candidate's pending call tasks — a surprising
// thing to discover after applying it to hundreds of rows, so the confirm
// dialog says so. The backend remains the authority and applies the rule
// regardless of what this list says; it only drives the warning text.
const TERMINAL_STATUSES_UI = new Set([
  'not interested', 'high ctc', 'shared with customer', 'for future',
  'shortlist - rejected', 'duplicate', 'rejected',
]);

/**
 * "Add to Shortlist" + "Set Status" for the selection bar.
 *
 * Shortlisting is not separate membership — 'Shortlisted' is a value in
 * RECRUITMENT_STAGES — so both buttons drive the same bulk endpoint.
 *
 * Shared by Roles and Talent Pool so the two selection bars cannot drift.
 * Deliberately takes explicit `candidateIds` rather than reading selection
 * state: the two pages model "select all filtered" differently (Roles
 * materialises ids into selectedIds, Talent Pool keeps a separate flag), and
 * that selection logic has been the source of real bugs. Each page passes the
 * ids it means.
 */
export default function BulkStatusActions({ candidateIds, onApplied }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const bulkUpdateCandidateStatus = useAppStore(state => state.bulkUpdateCandidateStatus);

  const ids = candidateIds || [];
  if (!ids.length) return null;

  const apply = async (status) => {
    const terminal = TERMINAL_STATUSES_UI.has(status.toLowerCase());
    const warning = terminal
      ? '\n\nThis status also closes any pending call tasks for these candidates, removing them from the calling loop.'
      : '';
    // No undo. Always state the exact count before rewriting records.
    if (!window.confirm(
      `Set status to "${status}" for ${ids.length} candidate${ids.length === 1 ? '' : 's'}?${warning}`
    )) return;

    setMenuOpen(false);
    setBusy(true);
    try {
      const res = await bulkUpdateCandidateStatus(ids, status);
      if (!res.success) {
        toast.error(res.error || 'Could not update statuses');
        return;
      }
      const skipped = res.skipped ? ` · ${res.skipped} skipped (not yours)` : '';
      const closed = res.closedCallTasks
        ? ` · ${res.closedCallTasks} call task${res.closedCallTasks === 1 ? '' : 's'} closed`
        : '';
      toast.success(
        `${res.updated} candidate${res.updated === 1 ? '' : 's'} set to "${status}"${skipped}${closed}`
      );
      await onApplied?.(res);
    } finally {
      setBusy(false);
    }
  };

  const baseButton = {
    padding: '8px 16px', border: '1px solid rgba(255,255,255,0.18)',
    borderRadius: '10px', fontSize: 13, fontWeight: 700,
    cursor: busy ? 'wait' : 'pointer', fontFamily: 'inherit',
    display: 'flex', alignItems: 'center', gap: 8, transition: 'all 0.2s',
    opacity: busy ? 0.7 : 1,
  };

  return (
    <>
      <button
        onClick={() => apply('Shortlisted')}
        disabled={busy}
        style={{ ...baseButton, background: '#16a34a', color: '#fff' }}
      >
        <Check size={14} /> Add to Shortlist
      </button>

      <div style={{ position: 'relative' }}>
        <button
          onClick={() => setMenuOpen(v => !v)}
          disabled={busy}
          style={{ ...baseButton, background: '#fff', color: '#0f172a' }}
        >
          {busy ? 'Updating…' : 'Set Status'} <ChevronDown size={14} />
        </button>
        {menuOpen && !busy && (
          <div style={{
            position: 'absolute', bottom: 'calc(100% + 8px)', left: 0, zIndex: 1100,
            background: '#fff', borderRadius: 12, border: '1px solid #e2e8f0',
            boxShadow: '0 20px 40px rgba(15,23,42,0.28)', padding: 6,
            minWidth: 230, maxHeight: 320, overflowY: 'auto',
          }}>
            {RECRUITMENT_STAGES.map(stage => (
              <button
                key={stage}
                onClick={() => apply(stage)}
                style={{
                  display: 'block', width: '100%', textAlign: 'left',
                  padding: '9px 12px', borderRadius: 8, border: 'none',
                  background: 'transparent', color: '#0f172a',
                  fontSize: 13, fontWeight: 600, cursor: 'pointer', fontFamily: 'inherit',
                }}
                onMouseEnter={e => { e.currentTarget.style.background = '#f1f5f9'; }}
                onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; }}
              >
                {stage}
              </button>
            ))}
          </div>
        )}
      </div>
    </>
  );
}
