import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import { toast } from 'sonner';
import { FileSpreadsheet, Linkedin, Loader2, RefreshCw, Trash2, X } from 'lucide-react';
import { API_BASE } from '../store/useAppStore';
import { loadGoogleIdentity } from '../utils/googleIdentity';
import CsvMappingModal from './CsvMappingModal';

const DONE = ['completed', 'completed_with_errors', 'failed'];

const errText = (err, fallback) => {
  const detail = err?.response?.data?.detail;
  if (typeof detail === 'string') return detail;
  if (detail?.message) return detail.message;
  return err?.message || fallback;
};
const errCode = (err) => err?.response?.data?.detail?.code || '';

const overlay = { position: 'fixed', inset: 0, background: 'rgba(15,23,42,0.65)', zIndex: 10002, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20 };
const card = { background: '#fff', borderRadius: 20, maxWidth: 640, width: '100%', maxHeight: '90vh', overflow: 'auto', padding: 24, boxShadow: '0 24px 60px rgba(15,23,42,0.28)' };
const ghostBtn = (disabled) => ({ padding: '10px 16px', borderRadius: 10, border: '1px solid #e2e8f0', background: '#fff', fontWeight: 700, fontSize: 13, cursor: disabled ? 'not-allowed' : 'pointer' });
const primaryBtn = (enabled) => ({ padding: '10px 16px', borderRadius: 10, border: 'none', background: enabled ? '#f97316' : '#cbd5e1', color: '#fff', fontWeight: 800, fontSize: 13, cursor: enabled ? 'pointer' : 'not-allowed' });

/**
 * Import candidates from a Google Sheet: connect Google, paste the link, pick the
 * tab, confirm the column mapping (same screen as a spreadsheet upload), then
 * import. Rows with a LinkedIn profile URL are scraped and enriched through Apify.
 * Saved sheets can be re-run with "Sync now". `onDone` fires after any finished run.
 */
export default function GoogleSheetImportModal({ open, roleId = null, roleName = '', onClose, onDone }) {
  const [google, setGoogle] = useState(null);          // { configured, connected, google_email, client_id, scope }
  const [connecting, setConnecting] = useState(false);
  const [sheetUrl, setSheetUrl] = useState('');
  const [tabs, setTabs] = useState(null);              // { sheet_title, tabs, selected_tab, max_rows }
  const [tab, setTab] = useState('');
  const [loading, setLoading] = useState(false);
  const [preview, setPreview] = useState(null);
  const [mapping, setMapping] = useState({});
  const [enrichLinkedin, setEnrichLinkedin] = useState(true);
  const [enrichmentMode, setEnrichmentMode] = useState('none');
  const [progress, setProgress] = useState(null);
  const [busy, setBusy] = useState(false);
  const [connections, setConnections] = useState([]);
  const [syncingId, setSyncingId] = useState(null);
  const codeClientRef = useRef(null);
  const aliveRef = useRef(false);

  const loadConnections = useCallback(async () => {
    try {
      const res = await axios.get(`${API_BASE}/import/gsheet/connections`, { params: roleId ? { role_id: roleId } : {}, timeout: 60000 });
      if (aliveRef.current) setConnections(res.data?.connections || []);
    } catch { /* the list is a convenience; the import itself still works */ }
  }, [roleId]);

  useEffect(() => {
    if (!open) return undefined;
    aliveRef.current = true;
    setSheetUrl(''); setTabs(null); setTab(''); setPreview(null); setMapping({}); setProgress(null);
    setBusy(false); setSyncingId(null); setEnrichLinkedin(true); setEnrichmentMode('none');
    axios.get(`${API_BASE}/integrations/google/status`, { timeout: 60000 })
      .then(res => { if (aliveRef.current) setGoogle(res.data || {}); })
      .catch(err => { if (aliveRef.current) { setGoogle({ configured: false, connected: false }); toast.error(errText(err, 'Could not check the Google connection')); } });
    loadConnections();
    return () => { aliveRef.current = false; };
  }, [open, loadConnections]);

  // Build the consent client ahead of the click: browsers only allow the popup
  // when it opens directly from the user's click, not after an awaited load.
  useEffect(() => {
    if (!open || !google?.configured || !google?.client_id || google?.connected) return;
    let cancelled = false;
    loadGoogleIdentity().then((g) => {
      if (cancelled) return;
      codeClientRef.current = g.accounts.oauth2.initCodeClient({
        client_id: google.client_id,
        scope: google.scope,
        ux_mode: 'popup',
        callback: async (resp) => {
          if (!resp?.code) { setConnecting(false); toast.error('Google sign-in was cancelled.'); return; }
          try {
            const res = await axios.post(`${API_BASE}/integrations/google/connect`, { code: resp.code }, { timeout: 60000 });
            setGoogle(prev => ({ ...prev, connected: true, google_email: res.data?.google_email || null }));
            toast.success('Google connected.');
          } catch (err) {
            toast.error(errText(err, 'Could not connect Google'));
          } finally {
            setConnecting(false);
          }
        },
        error_callback: () => { setConnecting(false); },
      });
    }).catch(() => { if (!cancelled) toast.error('Google sign-in did not load. Check your connection and reopen this window.'); });
    return () => { cancelled = true; };
  }, [open, google?.configured, google?.client_id, google?.connected, google?.scope]);

  const connectGoogle = () => {
    if (!codeClientRef.current) { toast.error('Google sign-in is still loading. Try again in a moment.'); return; }
    setConnecting(true);
    codeClientRef.current.requestCode();
  };

  const disconnectGoogle = async () => {
    try {
      await axios.delete(`${API_BASE}/integrations/google`, { timeout: 60000 });
      setGoogle(prev => ({ ...prev, connected: false, google_email: null }));
    } catch (err) { toast.error(errText(err, 'Could not disconnect Google')); }
  };

  // A dead Google connection sends the user back to the connect step.
  const handleError = (err, fallback) => {
    if (['reconnect', 'not_connected'].includes(errCode(err))) {
      setGoogle(prev => ({ ...(prev || {}), connected: false }));
    }
    toast.error(errText(err, fallback));
  };

  const loadPreview = async (tabTitle) => {
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/import/gsheet/preview`, { sheet_url: sheetUrl.trim(), tab: tabTitle || null, role_id: roleId, use_llm: Boolean(roleId) }, { timeout: 180000 });
      const data = res.data || {};
      const counts = data.linkedin_url_counts || {};
      setPreview(data);
      setMapping(data.suggested_mapping || {});
      setEnrichLinkedin(Object.keys(counts).length > 0);
    } catch (err) {
      handleError(err, 'Could not read the sheet');
    } finally {
      setLoading(false);
    }
  };

  const loadSheet = async () => {
    if (!sheetUrl.trim()) return;
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/import/gsheet/tabs`, { sheet_url: sheetUrl.trim(), role_id: roleId }, { timeout: 60000 });
      const data = res.data || {};
      setTabs(data);
      setTab(data.selected_tab || '');
      if ((data.tabs || []).length <= 1) {
        await loadPreview(data.selected_tab);
        return;
      }
    } catch (err) {
      handleError(err, 'Could not open the sheet');
    } finally {
      setLoading(false);
    }
  };

  const pollUpload = async (uploadId, first) => {
    let d = first || {};
    for (let attempt = 0; attempt < 3600; attempt += 1) {
      if (!aliveRef.current) return d;
      const poll = await axios.get(`${API_BASE}/candidates/uploads/${uploadId}`, { timeout: 60000 });
      d = poll.data || {};
      setProgress(d);
      if (DONE.includes(String(d.status || '').toLowerCase())) break;
      await new Promise(resolve => window.setTimeout(resolve, 1500));
    }
    return d;
  };

  const report = (d) => {
    if (String(d.status || '').toLowerCase() === 'failed') {
      toast.error(d.error_message || 'Google Sheet import failed');
      return;
    }
    const parts = [];
    if (Number(d.inserted) > 0) parts.push(`${Number(d.inserted)} new`);
    if (Number(d.updated) > 0) parts.push(`${Number(d.updated)} updated`);
    if (Number(d.skipped) > 0) parts.push(`${Number(d.skipped)} skipped`);
    if (Number(d.role_assigned_count) > 0) parts.push(`${Number(d.role_assigned_count)} added to ${roleName || 'the role'}`);
    toast.success(parts.length ? `Google Sheet import complete: ${parts.join(', ')}.` : 'Google Sheet import complete: nothing new.');
    const errs = Array.isArray(d.errors) ? d.errors : [];
    if (errs.length > 0) {
      const text = errs.slice(0, 3).join(' ');
      toast.warning(text.length > 300 ? `${text.slice(0, 297)}…` : text);
    }
    onDone?.(d);
  };

  const linkedinHeader = useMemo(() => Object.keys(mapping).find(h => mapping[h] === 'linkedin') || '', [mapping]);
  const linkedinRows = Number(preview?.linkedin_url_counts?.[linkedinHeader]) || 0;
  const totalRows = Number(preview?.row_count) || 0;

  const commit = async () => {
    setBusy(true);
    setProgress({ status: 'processing', processed_count: 0, row_count: totalRows });
    try {
      const res = await axios.post(`${API_BASE}/import/gsheet/commit`, {
        sheet_url: sheetUrl.trim(), tab: preview?.tab || tab || null, role_id: roleId, mapping,
        enrich_linkedin: enrichLinkedin && linkedinRows > 0, enrichment_mode: enrichmentMode,
      }, { timeout: 180000 });
      setProgress(res.data || {});
      const d = await pollUpload(res.data?.upload_id, res.data);
      report(d);
      loadConnections();
    } catch (err) {
      setProgress(null);
      handleError(err, 'Google Sheet import failed');
    } finally {
      setBusy(false);
    }
  };

  const syncNow = async (connection) => {
    setSyncingId(connection.id);
    setProgress({ status: 'processing', processed_count: 0, row_count: 0 });
    try {
      const res = await axios.post(`${API_BASE}/import/gsheet/connections/${connection.id}/sync`, {}, { timeout: 180000 });
      setProgress(res.data || {});
      const d = await pollUpload(res.data?.upload_id, res.data);
      report(d);
      loadConnections();
    } catch (err) {
      handleError(err, 'Sync failed');
    } finally {
      setSyncingId(null);
      setProgress(null);
    }
  };

  const removeConnection = async (connection) => {
    try {
      await axios.delete(`${API_BASE}/import/gsheet/connections/${connection.id}`, { timeout: 60000 });
      setConnections(prev => prev.filter(c => c.id !== connection.id));
    } catch (err) { toast.error(errText(err, 'Could not remove the saved sheet')); }
  };

  if (!open) return null;

  // ---- step 2: the shared column-mapping screen
  if (preview) {
    const finished = DONE.includes(String(progress?.status || '').toLowerCase());
    const extraPanel = (
      <div style={{ margin: '14px 24px 0', padding: 12, borderRadius: 12, background: linkedinRows > 0 ? '#eff6ff' : '#f8fafc', border: `1px solid ${linkedinRows > 0 ? '#bfdbfe' : '#e2e8f0'}`, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 7, fontSize: 13, fontWeight: 800, color: '#0f172a' }}>
            <Linkedin size={15} color="#0a66c2" />
            {linkedinRows > 0
              ? `${linkedinRows} of ${totalRows} rows have a LinkedIn profile URL`
              : 'No LinkedIn profile URLs found in the mapped LinkedIn column'}
          </div>
          <div style={{ marginTop: 3, fontSize: 12, color: '#64748b' }}>
            {linkedinRows > 0
              ? `Those ${linkedinRows} are scraped through Apify (one paid lookup each, unless already in the pool) and loaded with full work history. The sheet's email, phone, notes and extra columns are kept. The other ${Math.max(0, totalRows - linkedinRows)} rows load from the sheet values.`
              : 'All rows will load from the sheet values, like a spreadsheet upload.'}
          </div>
        </div>
        <label style={{ display: 'inline-flex', alignItems: 'center', gap: 7, fontSize: 12, fontWeight: 800, color: '#0f172a', whiteSpace: 'nowrap' }}>
          <input type="checkbox" checked={enrichLinkedin && linkedinRows > 0} disabled={busy || linkedinRows === 0} onChange={e => setEnrichLinkedin(e.target.checked)} />
          Enrich from LinkedIn
        </label>
      </div>
    );
    return (
      <CsvMappingModal
        title={`Map columns: ${preview.sheet_title || 'Google Sheet'} / ${preview.tab}`}
        subtitle={roleName ? `Review the suggestions, then import into ${roleName} and the owner's pool.` : 'Review the suggestions, then import into your pool.'}
        headers={preview.headers || []}
        mapping={mapping}
        details={preview.mapping_details || {}}
        requiredTargets={enrichLinkedin && linkedinRows > 0 ? ['linkedin'] : (preview.required_targets || ['first_name', 'last_name', 'linkedin', 'city', 'title'])}
        targetOptions={preview.target_options || []}
        rowCount={totalRows}
        progress={progress}
        enrichmentMode={enrichmentMode}
        busy={busy}
        extraPanel={extraPanel}
        importLabel={`Import ${totalRows} row${totalRows === 1 ? '' : 's'}`}
        onEnrichmentModeChange={setEnrichmentMode}
        onChange={(header, value) => setMapping(prev => ({ ...prev, [header]: value }))}
        onCancel={() => { if (finished || !busy) { if (finished) onClose?.(); else { setPreview(null); setProgress(null); } } }}
        onImport={commit}
      />
    );
  }

  // ---- step 1: connect, link, tab, saved sheets
  const working = loading || Boolean(syncingId);
  const syncStage = String(progress?.status || '');
  return (
    <div style={overlay}>
      <div style={card}>
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12 }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 17, fontWeight: 800, color: '#0f172a' }}>
              <FileSpreadsheet size={18} color="#16a34a" /> Import from Google Sheet{roleName ? ` into ${roleName}` : ''}
            </div>
            <div style={{ fontSize: 13, color: '#64748b', marginTop: 4 }}>
              Connect Google, paste the sheet link and confirm the column mapping. Rows with a LinkedIn profile URL are enriched through Apify.
            </div>
          </div>
          <button type="button" onClick={onClose} disabled={working} aria-label="Close" style={{ border: 'none', background: 'transparent', cursor: working ? 'not-allowed' : 'pointer', color: '#64748b' }}>
            <X size={18} />
          </button>
        </div>

        {google === null && (
          <div style={{ marginTop: 20, display: 'flex', alignItems: 'center', gap: 8, color: '#64748b', fontSize: 13 }}>
            <Loader2 size={15} style={{ animation: 'gsheetSpin 1s linear infinite' }} /> Checking Google connection…
          </div>
        )}

        {google && !google.configured && (
          <div style={{ marginTop: 18, padding: '12px 14px', borderRadius: 12, background: '#fffbeb', border: '1px solid #fde68a', color: '#92400e', fontSize: 13 }}>
            Google Sheets import is not set up on the server yet. An admin needs to add the Google client secret and token key.
          </div>
        )}

        {google?.configured && !google.connected && (
          <div style={{ marginTop: 18, padding: 14, borderRadius: 12, background: '#f8fafc', border: '1px solid #e2e8f0', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
            <div style={{ fontSize: 13, color: '#334155' }}>
              <div style={{ fontWeight: 800, color: '#0f172a' }}>Step 1: connect your Google account</div>
              <div style={{ marginTop: 3, fontSize: 12, color: '#64748b' }}>Read-only access to your spreadsheets. You can disconnect at any time.</div>
            </div>
            <button type="button" onClick={connectGoogle} disabled={connecting} style={primaryBtn(!connecting)}>
              {connecting ? 'Waiting for Google…' : 'Connect Google'}
            </button>
          </div>
        )}

        {google?.configured && google.connected && (
          <>
            <div style={{ marginTop: 16, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10, fontSize: 12, color: '#64748b' }}>
              <span>Connected{google.google_email ? ` as ${google.google_email}` : ''}</span>
              <button type="button" onClick={disconnectGoogle} disabled={working} style={{ border: 'none', background: 'transparent', color: '#b91c1c', fontWeight: 700, fontSize: 12, cursor: working ? 'not-allowed' : 'pointer' }}>Disconnect</button>
            </div>

            <label style={{ display: 'block', marginTop: 12, fontSize: 12, fontWeight: 800, color: '#334155' }}>
              Google Sheet link
              <input
                value={sheetUrl}
                disabled={working}
                onChange={e => { setSheetUrl(e.target.value); setTabs(null); setTab(''); }}
                onKeyDown={e => { if (e.key === 'Enter' && sheetUrl.trim() && !working) (tabs ? loadPreview(tab) : loadSheet()); }}
                placeholder="https://docs.google.com/spreadsheets/d/…"
                style={{ width: '100%', marginTop: 6, padding: '10px 12px', borderRadius: 10, border: '1px solid #cbd5e1', fontSize: 13, fontWeight: 500, boxSizing: 'border-box' }}
              />
            </label>

            {tabs && (tabs.tabs || []).length > 1 && (
              <label style={{ display: 'block', marginTop: 12, fontSize: 12, fontWeight: 800, color: '#334155' }}>
                Tab in “{tabs.sheet_title}”
                <select value={tab} disabled={working} onChange={e => setTab(e.target.value)} style={{ width: '100%', marginTop: 6, padding: '10px 12px', borderRadius: 10, border: '1px solid #cbd5e1', fontSize: 13, background: '#fff' }}>
                  {tabs.tabs.map(t => <option key={t.gid} value={t.title}>{t.title}</option>)}
                </select>
              </label>
            )}

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 10, marginTop: 14 }}>
              <span style={{ fontSize: 12, color: '#94a3b8' }}>The first row must be the column headers.{tabs?.max_rows ? ` Up to ${tabs.max_rows} rows per import.` : ''}</span>
              <button
                type="button"
                disabled={working || !sheetUrl.trim()}
                onClick={() => (tabs ? loadPreview(tab) : loadSheet())}
                style={primaryBtn(!working && Boolean(sheetUrl.trim()))}
              >
                {loading ? 'Reading sheet…' : 'Continue'}
              </button>
            </div>

            {connections.length > 0 && (
              <div style={{ marginTop: 20, borderTop: '1px solid #e2e8f0', paddingTop: 14 }}>
                <div style={{ fontSize: 12, fontWeight: 800, color: '#334155', marginBottom: 8 }}>Connected sheets</div>
                {connections.map(c => (
                  <div key={c.id} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10, padding: '9px 0', borderBottom: '1px solid #f1f5f9' }}>
                    <div style={{ minWidth: 0 }}>
                      <a href={c.sheet_url} target="_blank" rel="noreferrer" style={{ fontSize: 13, fontWeight: 700, color: '#0f172a', textDecoration: 'none' }}>
                        {c.sheet_title || 'Google Sheet'} / {c.tab_title}
                      </a>
                      <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>
                        {c.last_synced_at ? `Last synced ${new Date(`${c.last_synced_at}Z`).toLocaleString()}` : 'Not synced yet'}
                        {c.enrich_linkedin ? ' · LinkedIn enrichment on' : ''}
                      </div>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0 }}>
                      <button type="button" disabled={working} onClick={() => syncNow(c)} title="Read the sheet again and import only people not already in the pool" style={{ ...ghostBtn(working), padding: '7px 11px', fontSize: 12, display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                        <RefreshCw size={13} style={syncingId === c.id ? { animation: 'gsheetSpin 1s linear infinite' } : undefined} />
                        {syncingId === c.id ? 'Syncing…' : 'Sync now'}
                      </button>
                      <button type="button" disabled={working} onClick={() => removeConnection(c)} aria-label="Remove saved sheet" title="Remove this saved sheet (imported candidates stay)" style={{ border: 'none', background: 'transparent', color: '#94a3b8', cursor: working ? 'not-allowed' : 'pointer' }}>
                        <Trash2 size={15} />
                      </button>
                    </div>
                  </div>
                ))}
                {syncingId && (
                  <div style={{ marginTop: 10, fontSize: 12, color: '#475569' }}>
                    {syncStage === 'scraping' ? 'Scraping new LinkedIn profiles via Apify…' : syncStage === 'enriching' ? 'Enriching and saving…' : 'Reading the sheet…'}
                    {Number(progress?.row_count) > 0 ? ` ${Number(progress?.processed_count) || 0} / ${Number(progress.row_count)}` : ''}
                  </div>
                )}
              </div>
            )}
          </>
        )}

        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 18 }}>
          <button type="button" onClick={onClose} disabled={working} style={ghostBtn(working)}>Close</button>
        </div>
      </div>
      <style>{`@keyframes gsheetSpin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
