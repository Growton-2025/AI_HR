import React, { useEffect } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { AlertTriangle, Check, Loader2, X } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';

const stageText = (status) => {
  if (status === 'scraping') return 'Reading LinkedIn profiles';
  if (status === 'enriching') return 'Enriching and saving profiles';
  if (status === 'refreshing') return 'Finalizing';
  return 'Reading the sheet';
};

/**
 * Imports that keep running after their window is closed. One small card per
 * import, bottom right, on every page. Finished imports stay a few seconds, failed
 * ones until dismissed.
 */
export default function ImportProgressDock() {
  const { activeImports, dismissImport, resumeRunningImports, isAuthenticated } = useAppStore(useShallow(state => ({
    activeImports: state.activeImports,
    dismissImport: state.dismissImport,
    resumeRunningImports: state.resumeRunningImports,
    isAuthenticated: state.isAuthenticated,
  })));

  useEffect(() => { if (isAuthenticated) resumeRunningImports(); }, [isAuthenticated, resumeRunningImports]);

  const items = Object.values(activeImports || {}).filter(item => !item.hidden).sort((a, b) => a.uploadId - b.uploadId);

  // Successful imports clear themselves; the toast has already said what happened.
  useEffect(() => {
    const timers = items
      .filter(item => item.finished && String(item.progress?.status || '').toLowerCase() !== 'failed')
      .map(item => window.setTimeout(() => dismissImport(item.uploadId), 12000));
    return () => timers.forEach(window.clearTimeout);
  }, [items.map(i => `${i.uploadId}:${i.finished}`).join(','), dismissImport]); // eslint-disable-line react-hooks/exhaustive-deps

  if (items.length === 0) return null;

  return (
    <div style={{ position: 'fixed', right: 20, bottom: 20, zIndex: 9000, display: 'flex', flexDirection: 'column', gap: 10, width: 320, maxWidth: 'calc(100vw - 40px)' }}>
      {items.map(item => {
        const status = String(item.progress?.status || '').toLowerCase();
        const failed = status === 'failed';
        const done = item.finished && !failed;
        const total = Number(item.progress?.row_count) || 0;
        const processed = Math.min(Number(item.progress?.processed_count) || 0, total || Infinity);
        const percent = done ? 100 : total > 0 ? Math.max(4, Math.round((processed / total) * 100)) : 6;
        const summary = [
          Number(item.progress?.inserted) > 0 ? `${Number(item.progress.inserted)} new` : null,
          Number(item.progress?.updated) > 0 ? `${Number(item.progress.updated)} updated` : null,
          Number(item.progress?.skipped) > 0 ? `${Number(item.progress.skipped)} skipped` : null,
        ].filter(Boolean).join(', ');
        return (
          <div key={item.uploadId} role="status" style={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 14, padding: '12px 14px', boxShadow: '0 12px 32px rgba(15,23,42,0.18)' }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
              <div style={{ marginTop: 1, flexShrink: 0 }}>
                {failed
                  ? <AlertTriangle size={16} color="#dc2626" />
                  : done
                    ? <Check size={16} color="#15803d" />
                    : <Loader2 size={16} color="var(--accent-primary)" style={{ animation: 'importDockSpin 1s linear infinite' }} />}
              </div>
              <div style={{ minWidth: 0, flex: 1 }}>
                <div style={{ fontSize: 13, fontWeight: 800, color: '#0f172a', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {item.label}{item.roleName ? ` → ${item.roleName}` : ''}
                </div>
                <div style={{ marginTop: 2, fontSize: 12, color: failed ? '#b91c1c' : '#64748b' }}>
                  {failed
                    ? (item.progress?.error_message || 'Import failed')
                    : done
                      ? (summary || 'Nothing new')
                      : `${stageText(status)}${total > 0 ? ` · ${processed} / ${total}` : ''}`}
                </div>
              </div>
              <button type="button" onClick={() => dismissImport(item.uploadId)} aria-label="Hide" title={item.finished ? 'Dismiss' : 'Hide (the import keeps running)'} style={{ border: 'none', background: 'transparent', color: '#94a3b8', cursor: 'pointer', padding: 0, flexShrink: 0 }}>
                <X size={15} />
              </button>
            </div>
            {!failed && (
              <div style={{ height: 5, borderRadius: 999, overflow: 'hidden', background: '#e2e8f0', marginTop: 10 }}>
                <div style={{ height: '100%', width: `${percent}%`, borderRadius: 999, background: done ? '#16a34a' : 'var(--accent-primary)', transition: 'width 320ms ease' }} />
              </div>
            )}
          </div>
        );
      })}
      <style>{`@keyframes importDockSpin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
