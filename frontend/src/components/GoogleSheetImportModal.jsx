import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import { toast } from 'sonner';
import { FileSpreadsheet, Linkedin, Loader2, RefreshCw, SlidersHorizontal, Trash2, X } from 'lucide-react';
import { useShallow } from 'zustand/react/shallow';
import { API_BASE, useAppStore } from '../store/useAppStore';
import { loadGoogleIdentity } from '../utils/googleIdentity';
import CsvMappingModal from './CsvMappingModal';

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
 * Saved sheets can be re-run with "Sync now", or reopened with their saved mapping.
 *
 * The import itself is tracked by the app store (`trackImport`), so this window can be
 * closed while it runs; pages refresh their lists on the store's `importFinishedTick`.
 */
export default function GoogleSheetImportModal({ open, roleId = null, roleName = '', onClose }) {
  const { activeImports, trackImport, importFinishedTick } = useAppStore(useShallow(state => ({
    activeImports: state.activeImports,
    trackImport: state.trackImport,
    importFinishedTick: state.importFinishedTick,
  })));
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
  const [uploadId, setUploadId] = useState(null);       // the import started from this window, if any
  const [starting, setStarting] = useState(false);      // true only while the start request is in flight
  const [busy, setBusy] = useState(false);
  const [smartMatching, setSmartMatching] = useState(false);
  const [connections, setConnections] = useState([]);
  const [syncingId, setSyncingId] = useState(null);
  const urlTouchedRef = useRef(false);
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
    setSheetUrl(''); setTabs(null); setTab(''); setPreview(null); setMapping({}); setUploadId(null);
    setStarting(false); setBusy(false); setSmartMatching(false); setSyncingId(null);
    setEnrichLinkedin(true); setEnrichmentMode('none');
    urlTouchedRef.current = false;
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

  // `saved` is a remembered connection: its confirmed mapping is laid over the fresh
  // suggestions, for the columns that still exist in the sheet.
  const loadPreview = async (tabTitle, { url = sheetUrl, saved = null } = {}) => {
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/import/gsheet/preview`, { sheet_url: url.trim(), tab: tabTitle || null, role_id: roleId, use_llm: false }, { timeout: 180000 });
      const data = res.data || {};
      const counts = data.linkedin_url_counts || {};
      const suggested = data.suggested_mapping || {};
      const remembered = {};
      Object.entries(saved?.mapping || {}).forEach(([header, target]) => {
        if (typeof target === 'string' && header in suggested) remembered[header] = target;
      });
      setPreview(data);
      setMapping({ ...suggested, ...remembered });
      setEnrichLinkedin(saved ? Boolean(saved.enrich_linkedin) && Object.keys(counts).length > 0 : Object.keys(counts).length > 0);
      if (saved?.enrichment_mode) setEnrichmentMode(saved.enrichment_mode);
    } catch (err) {
      handleError(err, 'Could not read the sheet');
    } finally {
      setLoading(false);
    }
  };

  // The mapping screen opens on the instant rule-based suggestions; the AI pass runs only on request.
  const smartMatch = async () => {
    setSmartMatching(true);
    try {
      const res = await axios.post(`${API_BASE}/import/gsheet/preview`, { sheet_url: sheetUrl.trim(), tab: preview?.tab || tab || null, role_id: roleId, use_llm: true }, { timeout: 180000 });
      const data = res.data || {};
      setPreview(data);
      setMapping(data.suggested_mapping || {});
      toast.success('Smart match updated the suggestions. Review them before importing.');
    } catch (err) {
      handleError(err, 'Smart match failed');
    } finally {
      setSmartMatching(false);
    }
  };

  const reviewConnection = async (connection) => {
    urlTouchedRef.current = true;
    setSheetUrl(connection.sheet_url);
    await loadPreview(connection.tab_title, { url: connection.sheet_url, saved: connection });
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

  const linkedinHeader = useMemo(() => Object.keys(mapping).find(h => mapping[h] === 'linkedin') || '', [mapping]);
  const linkedinRows = Number(preview?.linkedin_url_counts?.[linkedinHeader]) || 0;
  const totalRows = Number(preview?.row_count) || 0;

  const importLabel = roleName ? 'Google Sheet import' : 'Google Sheet import to your pool';

  const commit = async () => {
    setBusy(true);
    setStarting(true);
    try {
      const res = await axios.post(`${API_BASE}/import/gsheet/commit`, {
        sheet_url: sheetUrl.trim(), tab: preview?.tab || tab || null, role_id: roleId, mapping,
        enrich_linkedin: enrichLinkedin && linkedinRows > 0, enrichment_mode: enrichmentMode,
      }, { timeout: 180000 });
      trackImport(res.data?.upload_id, { label: importLabel, roleId, roleName }, res.data || null);
      setUploadId(res.data?.upload_id || null);
    } catch (err) {
      handleError(err, 'Google Sheet import failed');
    } finally {
      setStarting(false);
      setBusy(false);
    }
  };

  const syncNow = async (connection) => {
    setSyncingId(connection.id);
    try {
      const res = await axios.post(`${API_BASE}/import/gsheet/connections/${connection.id}/sync`, {}, { timeout: 180000 });
      trackImport(res.data?.upload_id, { label: 'Google Sheet sync', roleId, roleName }, res.data || null);
      toast.success('Sync started. You can close this window; you will be notified when it finishes.');
      loadConnections();
    } catch (err) {
      handleError(err, 'Sync failed');
    } finally {
      setSyncingId(null);
    }
  };

  // Saved sheets show their new "last synced" time once any import finishes.
  useEffect(() => { if (open && importFinishedTick > 0) loadConnections(); }, [open, importFinishedTick, loadConnections]);

  // The most recently used sheet is offered as the starting link.
  useEffect(() => {
    if (!open || urlTouchedRef.current || sheetUrl || connections.length === 0) return;
    setSheetUrl(connections[0].sheet_url || '');
  }, [open, connections, sheetUrl]);

  // Imports still running for this role (or the pool), whichever window started them.
  const running = Object.values(activeImports || {}).filter(item => !item.finished && (item.roleId ?? null) === (roleId ?? null));
  const tracked = uploadId ? activeImports?.[uploadId] : null;
  const progress = starting
    ? { status: 'processing', processed_count: 0, row_count: Number(preview?.row_count) || 0 }
    : (tracked?.progress || null);

  const removeConnection = async (connection) => {
    try {
      await axios.delete(`${API_BASE}/import/gsheet/connections/${connection.id}`, { timeout: 60000 });
      setConnections(prev => prev.filter(c => c.id !== connection.id));
    } catch (err) { toast.error(errText(err, 'Could not remove the saved sheet')); }
  };

  if (!open) return null;

  // ---- step 2: the shared column-mapping screen
  if (preview) {
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
        runsInBackground
        onSmartMatch={smartMatch}
        smartMatching={smartMatching}
        extraPanel={extraPanel}
        importLabel={`Import ${totalRows} row${totalRows === 1 ? '' : 's'}`}
        onEnrichmentModeChange={setEnrichmentMode}
        onChange={(header, value) => setMapping(prev => ({ ...prev, [header]: value }))}
        onCancel={() => { if (progress) onClose?.(); else if (!busy) setPreview(null); }}
        onImport={commit}
      />
    );
  }

  // ---- step 1: connect, link, tab, saved sheets
  const working = loading || Boolean(syncingId);
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

        {running.length > 0 && (
          <div style={{ marginTop: 16, padding: '10px 12px', borderRadius: 12, background: '#fff7ed', border: '1px solid #fed7aa', color: '#9a3412', fontSize: 12, fontWeight: 700 }}>
            {running.map(item => {
              const stage = String(item.progress?.status || '');
              const total = Number(item.progress?.row_count) || 0;
              return (
                <div key={item.uploadId} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Loader2 size={13} style={{ animation: 'gsheetSpin 1s linear infinite', flexShrink: 0 }} />
                  {item.label} is running: {stage === 'scraping' ? 'reading LinkedIn profiles' : stage === 'enriching' ? 'enriching and saving' : 'reading the sheet'}
                  {total > 0 ? ` (${Number(item.progress?.processed_count) || 0} / ${total})` : ''}. You can close this window.
                </div>
              );
            })}
          </div>
        )}

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
                onChange={e => { urlTouchedRef.current = true; setSheetUrl(e.target.value); setTabs(null); setTab(''); }}
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
                      <button type="button" disabled={working} onClick={() => reviewConnection(c)} title="Open this sheet with its saved column mapping, to check or change it before importing" style={{ ...ghostBtn(working), padding: '7px 11px', fontSize: 12, display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                        <SlidersHorizontal size={13} /> Review mapping
                      </button>
                      <button type="button" disabled={working} onClick={() => syncNow(c)} title="Read the sheet again and import only people not already in the pool" style={{ ...ghostBtn(working), padding: '7px 11px', fontSize: 12, display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                        <RefreshCw size={13} style={syncingId === c.id ? { animation: 'gsheetSpin 1s linear infinite' } : undefined} />
                        {syncingId === c.id ? 'Starting…' : 'Sync now'}
                      </button>
                      <button type="button" disabled={working} onClick={() => removeConnection(c)} aria-label="Remove saved sheet" title="Remove this saved sheet (imported candidates stay)" style={{ border: 'none', background: 'transparent', color: '#94a3b8', cursor: working ? 'not-allowed' : 'pointer' }}>
                        <Trash2 size={15} />
                      </button>
                    </div>
                  </div>
                ))}
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
