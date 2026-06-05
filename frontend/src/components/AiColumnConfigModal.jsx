import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { toast } from 'sonner';
import { ChevronRight, Globe, Play, X } from 'lucide-react';
import { longOperationAxios as aiColumnAxios } from '../api/longTimeoutAxios';
import { API_BASE, getRequestErrorMessage } from '../store/useAppStore';
import HayasaBrand from './HayasaBrand';

const AI_RUN_TIMEOUT_MS = 120000;
const FIELD_CATALOG_TIMEOUT_MS = 120000;
const DEFAULT_OUTPUT_SCHEMA = [
  { key: 'result', label: 'Result', type: 'text', primary: true },
];

function suggestColumnName(prompt = '') {
  const compact = String(prompt || '')
    .replace(/\{[^}]+\}/g, '')
    .replace(/\s+/g, ' ')
    .trim();
  if (!compact) return 'AI Result';
  const firstWords = compact.split(' ').slice(0, 6).join(' ');
  return firstWords.charAt(0).toUpperCase() + firstWords.slice(1);
}

const inputStyle = {
  width: '100%',
  boxSizing: 'border-box',
  padding: '11px 12px',
  borderRadius: 10,
  border: '1px solid #dbe3ee',
  background: '#fff',
  color: '#0f172a',
  fontSize: 13,
  outline: 'none',
  fontFamily: 'inherit',
};

export default function AiColumnConfigModal({
  selectedIds = new Set(),
  viewScope,
  recruiterFilterId,
  roleId,
  onClose,
  onColumnsCreated,
  onColumnDefinitionCreated,
  onColumnRunFailed,
}) {
  const promptRef = useRef(null);
  const fieldPickerRef = useRef(null);
  const fieldOptionRefs = useRef([]);
  const lastFieldMousePositionRef = useRef({ x: null, y: null });
  const [prompt, setPrompt] = useState('');
  const [columnName, setColumnName] = useState('');
  const [useWebSearch, setUseWebSearch] = useState(false);
  const [presets, setPresets] = useState([]);
  const [presetLoading, setPresetLoading] = useState(false);
  const [presetError, setPresetError] = useState('');
  const [selectedPresetId, setSelectedPresetId] = useState('');
  const [outputSchema, setOutputSchema] = useState(DEFAULT_OUTPUT_SCHEMA);
  const [requiredFields, setRequiredFields] = useState([]);
  const [contextInputs, setContextInputs] = useState({ our_product: '', pitch_context: '' });
  const [roleContext, setRoleContext] = useState({});
  const [running, setRunning] = useState(false);
  const [fieldGroups, setFieldGroups] = useState([]);
  const [fieldCatalogLoading, setFieldCatalogLoading] = useState(false);
  const [fieldCatalogError, setFieldCatalogError] = useState('');
  const [slashPicker, setSlashPicker] = useState(null);
  const [highlightedFieldIndex, setHighlightedFieldIndex] = useState(0);

  const selectedIdArray = useMemo(
    () => Array.from(selectedIds || []).map(Number).filter(Number.isFinite),
    [selectedIds],
  );
  const selectedCount = selectedIdArray.length;
  const resolvedName = columnName.trim() || suggestColumnName(prompt);
  const selectedPreset = useMemo(
    () => (presets || []).find((preset) => preset.id === selectedPresetId) || null,
    [presets, selectedPresetId],
  );
  const groupedPresets = useMemo(() => {
    const groups = [];
    for (const preset of presets || []) {
      let group = groups.find((item) => item.category === preset.category);
      if (!group) {
        group = { category: preset.category || 'Presets', presets: [] };
        groups.push(group);
      }
      group.presets.push(preset);
    }
    return groups;
  }, [presets]);

  const flattenedFields = useMemo(() => {
    return (fieldGroups || []).flatMap((group) => {
      const groupName = group?.group || 'Fields';
      return (group?.items || []).map((item) => ({
        key: item.key,
        label: item.label || item.key,
        group: groupName,
        token: item.token || `{${item.key}}`,
        sample: item.sample || '',
      }));
    });
  }, [fieldGroups]);

  const visibleFieldSuggestions = useMemo(() => {
    const query = String(slashPicker?.query || '').trim().toLowerCase();
    const filtered = flattenedFields.filter((field) => {
      if (!query) return true;
      return [
        field.label,
        field.key,
        field.group,
        field.sample,
      ].some((value) => String(value || '').toLowerCase().includes(query));
    });
    return filtered.slice(0, 40);
  }, [flattenedFields, slashPicker]);

  const groupedVisibleSuggestions = useMemo(() => {
    const grouped = [];
    for (const field of visibleFieldSuggestions) {
      let group = grouped.find((item) => item.group === field.group);
      if (!group) {
        group = { group: field.group, items: [] };
        grouped.push(group);
      }
      group.items.push(field);
    }
    return grouped;
  }, [visibleFieldSuggestions]);

  useEffect(() => {
    let cancelled = false;
    const params = new URLSearchParams();
    if (viewScope) params.set('view_scope', viewScope);
    if (recruiterFilterId) params.set('recruiter_filter_id', recruiterFilterId);
    if (roleId) params.set('role_id', roleId);

    setFieldCatalogLoading(true);
    setFieldCatalogError('');
    aiColumnAxios
      .get(`${API_BASE}/ai-columns/field-catalog?${params.toString()}`, { timeout: FIELD_CATALOG_TIMEOUT_MS })
      .then((res) => {
        if (cancelled) return;
        setFieldGroups(res.data?.groups || []);
        setRoleContext(res.data?.role_context || {});
      })
      .catch((error) => {
        if (cancelled) return;
        setFieldGroups([]);
        setRoleContext({});
        setFieldCatalogError(getRequestErrorMessage(error, 'Could not load fields'));
      })
      .finally(() => {
        if (!cancelled) setFieldCatalogLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [viewScope, recruiterFilterId, roleId]);

  useEffect(() => {
    let cancelled = false;
    setPresetLoading(true);
    setPresetError('');
    aiColumnAxios
      .get(`${API_BASE}/ai-columns/presets`, { timeout: FIELD_CATALOG_TIMEOUT_MS })
      .then((res) => {
        if (!cancelled) setPresets(res.data?.presets || []);
      })
      .catch((error) => {
        if (cancelled) return;
        setPresetError(getRequestErrorMessage(error, 'Could not load smart column presets'));
        setPresets([]);
      })
      .finally(() => {
        if (!cancelled) setPresetLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (highlightedFieldIndex >= visibleFieldSuggestions.length) {
      setHighlightedFieldIndex(0);
    }
  }, [highlightedFieldIndex, visibleFieldSuggestions.length]);

  useEffect(() => {
    if (!slashPicker || visibleFieldSuggestions.length === 0) return;
    const activeOption = fieldOptionRefs.current[highlightedFieldIndex];
    const picker = fieldPickerRef.current;
    if (!activeOption || !picker) return;

    const optionTop = activeOption.offsetTop;
    const optionBottom = optionTop + activeOption.offsetHeight;
    const visibleTop = picker.scrollTop;
    const visibleBottom = visibleTop + picker.clientHeight;

    if (optionTop < visibleTop) {
      picker.scrollTo({ top: Math.max(0, optionTop - 8), behavior: 'smooth' });
    } else if (optionBottom > visibleBottom) {
      picker.scrollTo({
        top: optionBottom - picker.clientHeight + 8,
        behavior: 'smooth',
      });
    }
  }, [highlightedFieldIndex, slashPicker, visibleFieldSuggestions.length]);

  const closeSlashPicker = useCallback(() => {
    setSlashPicker(null);
    setHighlightedFieldIndex(0);
  }, []);

  const openSlashPickerFromCaret = useCallback((nextPrompt, caretIndex) => {
    const caret = Number.isFinite(caretIndex) ? caretIndex : nextPrompt.length;
    const slashIndex = nextPrompt.lastIndexOf('/', Math.max(0, caret - 1));
    if (slashIndex < 0) {
      closeSlashPicker();
      return;
    }

    const query = nextPrompt.slice(slashIndex + 1, caret);
    const beforeSlash = slashIndex === 0 ? '' : nextPrompt[slashIndex - 1];
    const slashStartsToken = !beforeSlash || /\s|[\(\[,.:;]/.test(beforeSlash);
    const queryStillActive = !/[\s{}]/.test(query);
    if (!slashStartsToken || !queryStillActive) {
      closeSlashPicker();
      return;
    }

    const beforeCaret = nextPrompt.slice(0, caret);
    const lines = beforeCaret.split('\n');
    const lineIndex = lines.length - 1;
    const colIndex = lines[lines.length - 1]?.length || 0;
    setSlashPicker({
      start: slashIndex,
      end: caret,
      query,
      top: Math.min(188, 42 + lineIndex * 21),
      left: Math.min(440, 12 + colIndex * 7),
    });
    setHighlightedFieldIndex(0);
  }, [closeSlashPicker]);

  const handlePromptChange = (event) => {
    const nextPrompt = event.target.value;
    const caret = event.target.selectionStart ?? nextPrompt.length;
    setPrompt(nextPrompt);
    openSlashPickerFromCaret(nextPrompt, caret);
  };

  const insertFieldToken = useCallback((field) => {
    if (!field || !slashPicker) return;
    const token = field.token || `{${field.key}}`;
    const before = prompt.slice(0, slashPicker.start);
    const after = prompt.slice(slashPicker.end);
    const nextPrompt = `${before}${token}${after}`;
    const nextCaret = before.length + token.length;
    setPrompt(nextPrompt);
    closeSlashPicker();
    window.requestAnimationFrame(() => {
      if (!promptRef.current) return;
      promptRef.current.focus();
      promptRef.current.setSelectionRange(nextCaret, nextCaret);
    });
  }, [closeSlashPicker, prompt, slashPicker]);

  const handlePromptKeyDown = (event) => {
    if (!slashPicker) return;
    if (event.key === 'Escape') {
      event.preventDefault();
      closeSlashPicker();
      return;
    }
    if (event.key === 'ArrowDown') {
      event.preventDefault();
      setHighlightedFieldIndex((idx) => Math.min(idx + 1, Math.max(0, visibleFieldSuggestions.length - 1)));
      return;
    }
    if (event.key === 'ArrowUp') {
      event.preventDefault();
      setHighlightedFieldIndex((idx) => Math.max(0, idx - 1));
      return;
    }
    if ((event.key === 'Enter' || event.key === 'Tab') && visibleFieldSuggestions.length > 0) {
      event.preventDefault();
      insertFieldToken(visibleFieldSuggestions[highlightedFieldIndex] || visibleFieldSuggestions[0]);
    }
  };

  const handlePromptClick = (event) => {
    const caret = event.target.selectionStart ?? prompt.length;
    openSlashPickerFromCaret(prompt, caret);
  };

  const handlePresetChange = (event) => {
    const nextId = event.target.value;
    setSelectedPresetId(nextId);
    const nextPreset = (presets || []).find((preset) => preset.id === nextId);
    if (!nextPreset) {
      setOutputSchema(DEFAULT_OUTPUT_SCHEMA);
      setRequiredFields([]);
      return;
    }
    setColumnName(nextPreset.label || '');
    setPrompt(nextPreset.prompt_template || '');
    setUseWebSearch(nextPreset.mode === 'web_research');
    setOutputSchema(
      Array.isArray(nextPreset.output_schema) && nextPreset.output_schema.length
        ? nextPreset.output_schema
        : DEFAULT_OUTPUT_SCHEMA,
    );
    setRequiredFields((nextPreset.required_inputs || []).filter(Boolean));
  };

  const updateContextInput = (key, value) => {
    setContextInputs((prev) => ({ ...prev, [key]: value }));
  };

  const presetRunError = useMemo(() => {
    if (!selectedPreset) return '';
    const required = new Set(selectedPreset.required_inputs || []);
    if (required.has('role.job_description') && !String(roleContext?.job_description || '').trim()) {
      return roleId
        ? 'This preset needs a saved job description on the selected role.'
        : 'Filter Talent Pool to a role with a saved job description before running this preset.';
    }
    if (
      required.has('context.pitch_context')
      && !String(contextInputs.pitch_context || '').trim()
    ) {
      return 'Add pitch context before running this preset.';
    }
    if (
      required.has('context.our_product_or_pitch_context')
      && !String(contextInputs.our_product || '').trim()
      && !String(contextInputs.pitch_context || '').trim()
    ) {
      return 'Add our product or pitch context before running this preset.';
    }
    return '';
  }, [contextInputs.our_product, contextInputs.pitch_context, roleContext?.job_description, roleId, selectedPreset]);

  const handleRun = async () => {
    if (!selectedCount) {
      toast.error('Select one or more rows first');
      return;
    }
    if (!prompt.trim()) {
      toast.error('Enter a prompt for the smart column');
      return;
    }
    if (presetRunError) {
      toast.error(presetRunError);
      return;
    }

    setRunning(true);
    let definition = null;
    try {
      const saveRes = await aiColumnAxios.post(`${API_BASE}/ai-columns`, {
        name: resolvedName,
        prompt_template: prompt.trim(),
        mode: useWebSearch ? 'web_research' : 'auto',
        output_schema: outputSchema,
        required_fields: requiredFields,
        only_run_if: {
          required_fields: requiredFields,
          summary: requiredFields.length ? `Only run rows with ${requiredFields.join(', ')}` : '',
        },
        context_inputs: contextInputs,
        view_scope: viewScope,
        recruiter_filter_id: recruiterFilterId,
      }, { timeout: AI_RUN_TIMEOUT_MS });

      definition = saveRes.data || {};
      onColumnDefinitionCreated?.({
        definition,
        columnName: definition.name || resolvedName,
        columnDefinitionId: definition.id,
        candidateIds: selectedIdArray,
        selectionMode: 'selected_ids',
      });

      const runRes = await aiColumnAxios.post(`${API_BASE}/ai-columns/run`, {
        column_definition_id: definition.id,
        selection_mode: 'selected_ids',
        selected_ids: selectedIdArray,
        view_scope: viewScope,
        recruiter_filter_id: recruiterFilterId,
        role_id: roleId || null,
      }, { timeout: AI_RUN_TIMEOUT_MS });

      toast.success(`Running "${definition.name || resolvedName}" on ${selectedCount} row${selectedCount === 1 ? '' : 's'}`);
      onColumnsCreated?.({
        columnName: definition.name || resolvedName,
        columnDefinitionId: definition.id,
        runId: runRes.data?.run_id,
        candidateIds: selectedIdArray,
        selectionMode: 'selected_ids',
      });
      onClose?.();
    } catch (error) {
      if (definition?.id) {
        onColumnRunFailed?.({
          columnDefinitionId: definition.id,
          candidateIds: selectedIdArray,
        });
      }
      toast.error(getRequestErrorMessage(error, 'Failed to start smart column run'));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 10020,
        background: 'rgba(15,23,42,0.55)',
        backdropFilter: 'blur(8px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 18,
      }}
      onClick={(event) => event.target === event.currentTarget && !running && onClose?.()}
    >
      <div
        style={{
          width: 'min(720px, 100%)',
          maxHeight: '92vh',
          overflow: 'auto',
          background: '#fff',
          borderRadius: 18,
          boxShadow: '0 32px 80px rgba(15,23,42,0.24)',
          border: '1px solid rgba(226,232,240,0.9)',
        }}
        onClick={(event) => event.stopPropagation()}
      >
        <div style={{ padding: '20px 22px', borderBottom: '1px solid #eef2f7', background: '#f8fafc' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <div style={{
                width: 38,
                height: 38,
                borderRadius: 12,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: '#0f172a',
                color: '#fff',
              }}>
                <HayasaBrand size="compact" tone="dark" iconOnly />
              </div>
              <div>
                <div style={{ fontSize: 18, fontWeight: 800, color: '#0f172a' }}>Smart Column</div>
                <div style={{ fontSize: 12, color: '#64748b', fontWeight: 700 }}>
                  {selectedCount} selected row{selectedCount === 1 ? '' : 's'}
                </div>
              </div>
            </div>
            <button
              type="button"
              disabled={running}
              onClick={onClose}
              style={{ background: '#fff', border: '1px solid #e2e8f0', width: 36, height: 36, borderRadius: 10, cursor: running ? 'not-allowed' : 'pointer', color: '#64748b' }}
            >
              <X size={16} />
            </button>
          </div>
        </div>

        <div style={{ padding: 22, display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div>
            <label style={{ display: 'block', fontSize: 12, fontWeight: 800, color: '#334155', marginBottom: 7 }}>
              Preset
            </label>
            <select
              value={selectedPresetId}
              onChange={handlePresetChange}
              disabled={running || presetLoading}
              style={inputStyle}
            >
              <option value="">Custom prompt</option>
              {groupedPresets.map((group) => (
                <optgroup key={group.category} label={group.category}>
                  {group.presets.map((preset) => (
                    <option key={preset.id} value={preset.id}>{preset.label}</option>
                  ))}
                </optgroup>
              ))}
            </select>
            {presetLoading && <div style={{ fontSize: 11, color: '#64748b', marginTop: 6 }}>Loading presets...</div>}
            {presetError && <div style={{ fontSize: 11, color: '#b45309', marginTop: 6 }}>{presetError}</div>}
            {selectedPreset?.description && (
              <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5, marginTop: 7 }}>
                {selectedPreset.description}
              </div>
            )}
          </div>

          <div>
            <label style={{ display: 'block', fontSize: 12, fontWeight: 800, color: '#334155', marginBottom: 7 }}>
              Column name
            </label>
            <input
              value={columnName}
              onChange={(event) => setColumnName(event.target.value)}
              placeholder={suggestColumnName(prompt)}
              disabled={running}
              style={inputStyle}
            />
          </div>

          {selectedPreset && (
            <div style={{ border: '1px solid #dbe3ee', borderRadius: 12, padding: 14, background: '#f8fafc' }}>
              <div style={{ fontSize: 12, fontWeight: 800, color: '#334155', marginBottom: 8 }}>
                Preset inputs
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 7, marginBottom: 10 }}>
                {(selectedPreset.required_inputs || []).map((input) => (
                  <span key={input} style={{ border: '1px solid #dbe3ee', borderRadius: 8, background: '#fff', padding: '4px 7px', fontSize: 11, fontWeight: 700, color: '#475569' }}>
                    {input}
                  </span>
                ))}
              </div>
              {(selectedPreset.required_inputs || []).includes('candidate.city') && (
                <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5, marginBottom: 10 }}>
                  Rows missing city will be skipped.
                </div>
              )}
              {((selectedPreset.context_fields || []).includes('our_product')) && (
                <label style={{ display: 'block', marginBottom: 10 }}>
                  <span style={{ display: 'block', fontSize: 11, fontWeight: 800, color: '#475569', marginBottom: 5 }}>Our product</span>
                  <textarea
                    value={contextInputs.our_product}
                    onChange={(event) => updateContextInput('our_product', event.target.value)}
                    rows={3}
                    disabled={running}
                    placeholder="What you sell, who it helps, and the problem it solves."
                    style={{ ...inputStyle, resize: 'vertical', lineHeight: 1.5 }}
                  />
                </label>
              )}
              {((selectedPreset.context_fields || []).includes('pitch_context')) && (
                <label style={{ display: 'block' }}>
                  <span style={{ display: 'block', fontSize: 11, fontWeight: 800, color: '#475569', marginBottom: 5 }}>Pitch context</span>
                  <textarea
                    value={contextInputs.pitch_context}
                    onChange={(event) => updateContextInput('pitch_context', event.target.value)}
                    rows={3}
                    disabled={running}
                    placeholder="Offer, audience, angle, and tone constraints for outreach."
                    style={{ ...inputStyle, resize: 'vertical', lineHeight: 1.5 }}
                  />
                </label>
              )}
              {presetRunError && (
                <div style={{ fontSize: 12, color: '#b45309', lineHeight: 1.5, marginTop: 10 }}>
                  {presetRunError}
                </div>
              )}
            </div>
          )}

          <div style={{ position: 'relative' }}>
            <label style={{ display: 'block', fontSize: 12, fontWeight: 800, color: '#334155', marginBottom: 7 }}>
              Prompt
            </label>
            <textarea
              ref={promptRef}
              value={prompt}
              onChange={handlePromptChange}
              onKeyDown={handlePromptKeyDown}
              onClick={handlePromptClick}
              rows={9}
              disabled={running}
              placeholder="Example: Find this candidate's current location. Use the full row data first, and give a concise answer."
              style={{ ...inputStyle, minHeight: 210, resize: 'vertical', lineHeight: 1.65 }}
            />
            {slashPicker && (
              <div
                ref={fieldPickerRef}
                style={{
                  position: 'absolute',
                  zIndex: 3,
                  top: slashPicker.top,
                  left: slashPicker.left,
                  width: 390,
                  maxWidth: 'calc(100% - 24px)',
                  maxHeight: 330,
                  overflow: 'auto',
                  background: '#fff',
                  border: '1px solid #dbe3ee',
                  borderRadius: 12,
                  boxShadow: '0 18px 45px rgba(15, 23, 42, 0.18)',
                  padding: 8,
                }}
              >
                <div style={{ padding: '6px 8px 9px', borderBottom: '1px solid #eef2f7', marginBottom: 4 }}>
                  <div style={{ fontSize: 12, fontWeight: 800, color: '#0f172a' }}>
                    Insert row field
                  </div>
                  <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>
                    Type after / to search, Enter to insert.
                  </div>
                </div>
                {fieldCatalogLoading ? (
                  <div style={{ padding: 12, fontSize: 12, color: '#64748b', fontWeight: 700 }}>
                    Loading fields...
                  </div>
                ) : fieldCatalogError ? (
                  <div style={{ padding: 12, fontSize: 12, color: '#b45309', lineHeight: 1.45 }}>
                    {fieldCatalogError}. You can still type prompt text normally.
                  </div>
                ) : visibleFieldSuggestions.length === 0 ? (
                  <div style={{ padding: 12, fontSize: 12, color: '#64748b', fontWeight: 700 }}>
                    No matching fields
                  </div>
                ) : (
                  groupedVisibleSuggestions.map((group) => (
                    <div key={group.group} style={{ paddingTop: 4 }}>
                      <div style={{ padding: '6px 8px 4px', fontSize: 10, color: '#94a3b8', fontWeight: 900, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                        {group.group}
                      </div>
                      {group.items.map((field) => {
                        const flatIndex = visibleFieldSuggestions.findIndex((item) => item.key === field.key);
                        const isActive = flatIndex === highlightedFieldIndex;
                        return (
                          <button
                            key={field.key}
                            ref={(node) => {
                              if (flatIndex >= 0) fieldOptionRefs.current[flatIndex] = node;
                            }}
                            type="button"
                            onMouseMove={(event) => {
                              const last = lastFieldMousePositionRef.current;
                              if (last.x === event.clientX && last.y === event.clientY) return;
                              lastFieldMousePositionRef.current = { x: event.clientX, y: event.clientY };
                              setHighlightedFieldIndex(flatIndex);
                            }}
                            onMouseDown={(event) => {
                              event.preventDefault();
                              insertFieldToken(field);
                            }}
                            style={{
                              width: '100%',
                              border: 'none',
                              borderRadius: 9,
                              background: isActive ? '#eff6ff' : '#fff',
                              color: '#0f172a',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'space-between',
                              gap: 10,
                              padding: '8px 9px',
                              cursor: 'pointer',
                              textAlign: 'left',
                              fontFamily: 'inherit',
                            }}
                          >
                            <span style={{ minWidth: 0 }}>
                              <span style={{ display: 'block', fontSize: 13, fontWeight: 750, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                {field.label}
                              </span>
                              <span style={{ display: 'block', fontSize: 11, color: '#64748b', marginTop: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                {field.token}
                              </span>
                              {field.sample && (
                                <span style={{ display: 'block', fontSize: 10, color: '#94a3b8', marginTop: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                  {field.sample}
                                </span>
                              )}
                            </span>
                            <ChevronRight size={14} color={isActive ? '#2563eb' : '#cbd5e1'} style={{ flexShrink: 0 }} />
                          </button>
                        );
                      })}
                    </div>
                  ))
                )}
              </div>
            )}
          </div>

          <label
            style={{
              display: 'flex',
              alignItems: 'flex-start',
              gap: 12,
              padding: '12px 14px',
              borderRadius: 12,
              border: '1px solid #dbe3ee',
              background: useWebSearch ? '#f5f3ff' : '#fff',
              cursor: running ? 'not-allowed' : 'pointer',
            }}
          >
            <input
              type="checkbox"
              checked={useWebSearch}
              disabled={running}
              onChange={(event) => setUseWebSearch(event.target.checked)}
              style={{ marginTop: 3 }}
            />
            <Globe size={18} color="#4338ca" style={{ flexShrink: 0, marginTop: 1 }} />
            <span style={{ fontSize: 13, color: '#334155', lineHeight: 1.55 }}>
              <span style={{ fontWeight: 800, color: '#0f172a' }}>Use web search</span>
              <span style={{ display: 'block', marginTop: 4, fontSize: 12, color: '#64748b' }}>
                Auto uses row data when possible and adds web only for public or time-sensitive prompts. On forces live web research.
              </span>
            </span>
          </label>

          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'center', paddingTop: 4 }}>
            <div style={{ fontSize: 12, color: selectedCount ? '#64748b' : '#b45309', fontWeight: 700 }}>
              {selectedCount ? `${selectedCount} row${selectedCount === 1 ? '' : 's'} will run independently.` : 'Select rows in the table before running.'}
            </div>
            <div style={{ display: 'flex', gap: 10 }}>
              <button
                type="button"
                disabled={running}
                onClick={onClose}
                style={{
                  border: '1px solid #dbe3ee',
                  borderRadius: 10,
                  background: '#fff',
                  color: '#334155',
                  padding: '10px 14px',
                  cursor: running ? 'not-allowed' : 'pointer',
                  fontWeight: 700,
                  fontSize: 13,
                }}
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={running || !selectedCount || !prompt.trim()}
                onClick={handleRun}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 8,
                  border: 'none',
                  borderRadius: 10,
                  background: '#0f172a',
                  color: '#fff',
                  padding: '10px 14px',
                  cursor: running || !selectedCount || !prompt.trim() ? 'not-allowed' : 'pointer',
                  fontWeight: 800,
                  fontSize: 13,
                  opacity: running || !selectedCount || !prompt.trim() ? 0.55 : 1,
                }}
              >
                <Play size={14} />
                {running ? 'Starting...' : `Run ${useWebSearch ? 'with web' : 'without web'}`}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
